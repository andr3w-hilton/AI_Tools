"""
Authenticated Hand Gesture Mouse Control
Combines face recognition with hand gesture control.
Only allows mouse control when authorized users are detected.
Shows 'UNAUTHORIZED' interstitial when an unknown person is detected.

Commands:
  --enroll NAME    Enroll a new user (up to 3 users max)
  --list           List all enrolled users
  --delete NAME    Delete a specific user
  --delete-all     Delete all enrolled users
  (no args)        Run authenticated gesture control
"""

# Suppress warnings before imports
import os
import warnings

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

# Suppress absl logging
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

# Suppress pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, module='face_recognition_models')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from datetime import datetime
import time
import sys
import pickle
import argparse

# Suppress MediaPipe protobuf warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# face_recognition library for actual face identification
try:
    import face_recognition
except ImportError:
    print("ERROR: face_recognition library not installed")
    print("Install it with: pip install face-recognition")
    print("Note: On Windows, you may need to install dlib first")
    print("  pip install cmake")
    print("  pip install dlib")
    print("  pip install face-recognition")
    sys.exit(1)

MAX_USERS = 3


class AuthenticatedGestureController:
    def __init__(self, encoding_file="authorized_faces.pkl", init_camera=True):
        """Initialize the authenticated gesture controller"""
        self.encoding_file = os.path.join(os.path.dirname(__file__), encoding_file)
        self.authorized_users = {}  # Dict of {name: encoding}
        self.current_user = None  # Track who is currently recognized
        self.load_authorized_faces()

        # Authentication state
        self.is_authorized = False
        self.auth_check_interval = 0.5  # Check face every 0.5 seconds
        self.last_auth_check = 0
        self.face_not_found_count = 0
        self.unauthorized_detected = False

        # Lazy initialization - only set up camera/mediapipe when needed
        self._camera_initialized = False
        self.video_capture = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None

        if init_camera:
            self._init_camera_and_mediapipe()

    def _init_camera_and_mediapipe(self):
        """Initialize camera and MediaPipe (called lazily)"""
        if self._camera_initialized:
            return

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise Exception("Could not open camera")

        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Get camera frame dimensions
        self.cam_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Control zone mapping
        self.control_zone_x_min = 0.2
        self.control_zone_x_max = 0.8
        self.control_zone_y_min = 0.1
        self.control_zone_y_max = 0.7

        # Smoothing parameters
        self.smooth_factor = 5
        self.prev_x, self.prev_y = 0, 0

        # Click detection
        self.is_hand_closed = False
        self.click_cooldown = 0.5
        self.last_click_time = 0

        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01

        # Initialize MediaPipe with C++ warnings suppressed
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Redirect stderr at OS level to suppress TensorFlow/MediaPipe C++ warnings
        # Flush any pending output first
        sys.stderr.flush()
        stderr_fd = sys.stderr.fileno()
        stderr_copy = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)

        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            # Force model loading by doing a dummy process
            import numpy as np
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.hands.process(dummy_frame)
        finally:
            # Restore stderr
            os.dup2(stderr_copy, stderr_fd)
            os.close(stderr_copy)
            os.close(devnull_fd)

        self._camera_initialized = True
        print("Authenticated Gesture Controller Initialized")
        print(f"Screen size: {self.screen_width}x{self.screen_height}")

    def load_authorized_faces(self):
        """Load authorized face encodings from file"""
        if os.path.exists(self.encoding_file):
            with open(self.encoding_file, 'rb') as f:
                self.authorized_users = pickle.load(f)
            if self.authorized_users:
                names = ", ".join(self.authorized_users.keys())
                print(f"Loaded {len(self.authorized_users)} authorized user(s): {names}")
            return True
        else:
            self.authorized_users = {}
            return False

    def save_authorized_faces(self):
        """Save authorized face encodings to file"""
        with open(self.encoding_file, 'wb') as f:
            pickle.dump(self.authorized_users, f)

    def add_user(self, name, encoding):
        """Add a new authorized user"""
        if len(self.authorized_users) >= MAX_USERS:
            print(f"Cannot add user. Maximum of {MAX_USERS} users reached.")
            print("Delete a user first with --delete NAME")
            return False
        self.authorized_users[name] = encoding
        self.save_authorized_faces()
        print(f"User '{name}' enrolled successfully!")
        return True

    def delete_user(self, name):
        """Delete an authorized user"""
        if name in self.authorized_users:
            del self.authorized_users[name]
            self.save_authorized_faces()
            print(f"User '{name}' deleted successfully!")
            return True
        else:
            print(f"User '{name}' not found.")
            return False

    def delete_all_users(self):
        """Delete all authorized users"""
        self.authorized_users = {}
        if os.path.exists(self.encoding_file):
            os.remove(self.encoding_file)
        print("All users deleted successfully!")

    def list_users(self):
        """List all enrolled users"""
        if not self.authorized_users:
            print("No users enrolled.")
            print(f"Use --enroll NAME to add a user (max {MAX_USERS})")
        else:
            print(f"\nEnrolled users ({len(self.authorized_users)}/{MAX_USERS}):")
            print("-" * 30)
            for i, name in enumerate(self.authorized_users.keys(), 1):
                print(f"  {i}. {name}")
            print("-" * 30)

    def enroll_face(self, name):
        """Capture and enroll a user's face"""
        # Check if we can add more users
        if len(self.authorized_users) >= MAX_USERS:
            print(f"\nCannot enroll. Maximum of {MAX_USERS} users already enrolled.")
            print("Delete a user first with --delete NAME")
            return False

        # Check if name already exists
        if name in self.authorized_users:
            print(f"\nUser '{name}' already enrolled.")
            print("Delete them first with --delete NAME if you want to re-enroll.")
            return False

        print("\n" + "="*50)
        print(f"FACE ENROLLMENT: {name}")
        print("="*50)
        print(f"Users enrolled: {len(self.authorized_users)}/{MAX_USERS}")
        print("Position your face in the frame")
        print("Press SPACE to capture, 'q' to cancel")
        print("="*50 + "\n")

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame)

            # Draw enrollment UI
            cv2.rectangle(display_frame, (10, 10), (450, 100), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Enrolling: {name}",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if len(face_locations) == 1:
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.putText(display_frame, "Face detected - Press SPACE to enroll",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif len(face_locations) > 1:
                cv2.putText(display_frame, "Multiple faces - Only show your face",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No face detected - Position yourself",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.putText(display_frame, "Press 'q' to cancel",
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Face Enrollment', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Enrollment cancelled")
                break
            elif key == ord(' ') and len(face_locations) == 1:
                # Capture face encoding
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    self.add_user(name, face_encodings[0])

                    # Show confirmation
                    cv2.rectangle(display_frame, (0, 0),
                                (self.cam_width, self.cam_height), (0, 255, 0), 20)
                    cv2.putText(display_frame, f"{name} ENROLLED!",
                               (self.cam_width//2 - 150, self.cam_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                    cv2.imshow('Face Enrollment', display_frame)
                    cv2.waitKey(2000)
                    break

        cv2.destroyAllWindows()
        return True

    def check_authorization(self, frame):
        """Check if the current face is authorized. Returns (is_authorized, face_found, user_name)"""
        if not self.authorized_users:
            return False, False, None

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reduce frame size for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        # Find faces
        face_locations = face_recognition.face_locations(small_frame)

        if not face_locations:
            return False, False, None  # No face found

        # Get encodings
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # Get list of known encodings and names
        known_encodings = list(self.authorized_users.values())
        known_names = list(self.authorized_users.keys())

        for encoding in face_encodings:
            # Compare with all authorized faces
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)

            # Check if any match
            if True in matches:
                match_index = matches.index(True)
                return True, True, known_names[match_index]  # Authorized face found

        return False, True, None  # Face found but not authorized

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_fist_closed(self, hand_landmarks):
        """Detect if hand is closed (fist)"""
        palm_center = hand_landmarks.landmark[9]
        fingertip_indices = [4, 8, 12, 16, 20]

        distances = []
        for tip_idx in fingertip_indices:
            tip = hand_landmarks.landmark[tip_idx]
            distance = self.calculate_distance(tip, palm_center)
            distances.append(distance)

        avg_distance = np.mean(distances)
        threshold = 0.15
        return avg_distance < threshold

    def smooth_coordinates(self, x, y):
        """Apply smoothing to mouse coordinates"""
        smooth_x = int(self.prev_x + (x - self.prev_x) / self.smooth_factor)
        smooth_y = int(self.prev_y + (y - self.prev_y) / self.smooth_factor)
        self.prev_x = smooth_x
        self.prev_y = smooth_y
        return smooth_x, smooth_y

    def hand_to_screen_coords(self, hand_x, hand_y):
        """Convert hand coordinates to screen coordinates"""
        normalized_x = (hand_x - self.control_zone_x_min) / (self.control_zone_x_max - self.control_zone_x_min)
        normalized_y = (hand_y - self.control_zone_y_min) / (self.control_zone_y_max - self.control_zone_y_min)

        normalized_x = max(0.0, min(1.0, normalized_x))
        normalized_y = max(0.0, min(1.0, normalized_y))

        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)

        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return screen_x, screen_y

    def draw_unauthorized_overlay(self, frame):
        """Draw the UNAUTHORIZED interstitial overlay"""
        height, width = frame.shape[:2]

        # Create dark red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (20, 20), (width-20, height-20), (0, 0, 255), 4)

        # Draw warning icon (triangle)
        triangle_pts = np.array([
            [width//2, height//2 - 80],
            [width//2 - 60, height//2 + 20],
            [width//2 + 60, height//2 + 20]
        ], np.int32)
        cv2.polylines(frame, [triangle_pts], True, (0, 0, 255), 4)
        cv2.putText(frame, "!", (width//2 - 12, height//2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # UNAUTHORIZED text
        text = "UNAUTHORIZED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, height//2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Sub text
        sub_text = "Access Denied - Gesture Control Disabled"
        sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        sub_x = (width - sub_size[0]) // 2
        cv2.putText(frame, sub_text, (sub_x, height//2 + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_waiting_overlay(self, frame):
        """Draw overlay when no face is detected"""
        height, width = frame.shape[:2]

        # Create dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw message
        text = "WAITING FOR FACE"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        sub_text = "Position your face in the camera"
        sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        sub_x = (width - sub_size[0]) // 2
        cv2.putText(frame, sub_text, (sub_x, height//2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def draw_authorized_overlay(self, frame, hand_detected, fist_closed):
        """Draw status overlay when authorized"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw control zone rectangle
        zone_x1 = int(self.control_zone_x_min * width)
        zone_x2 = int(self.control_zone_x_max * width)
        zone_y1 = int(self.control_zone_y_min * height)
        zone_y2 = int(self.control_zone_y_max * height)
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (100, 100, 255), 2)
        cv2.putText(frame, "Control Zone", (zone_x1 + 5, zone_y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # Background for status
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Authorization status with user name
        user_display = self.current_user if self.current_user else "USER"
        cv2.putText(frame, f"AUTHORIZED: {user_display}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(frame, (380, 30), 8, (0, 255, 0), -1)

        # Hand status
        status = "Hand Detected" if hand_detected else "No Hand Detected"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Fist status
        if hand_detected:
            fist_status = "CLICKING" if fist_closed else "Open Hand"
            fist_color = (0, 255, 255) if fist_closed else (255, 255, 255)
            cv2.putText(frame, fist_status, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fist_color, 2)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit | Move to corner for FAILSAFE",
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self):
        """Main loop for authenticated hand gesture control"""
        if not self.authorized_users:
            print("\nNo authorized users enrolled!")
            print("Please run with --enroll NAME first")
            return

        print("\n" + "="*50)
        print("Authenticated Hand Gesture Control Started")
        print("="*50)
        names = ", ".join(self.authorized_users.keys())
        print(f"- Authorized users: {names}")
        print("- Close your fist to click")
        print("- Press 'q' to quit")
        print("="*50 + "\n")

        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                frame = cv2.flip(frame, 1)
                current_time = time.time()

                # Periodic authorization check
                if current_time - self.last_auth_check > self.auth_check_interval:
                    self.last_auth_check = current_time
                    is_auth, face_found, user_name = self.check_authorization(frame)

                    if face_found:
                        self.face_not_found_count = 0
                        if is_auth:
                            if not self.is_authorized or self.current_user != user_name:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] {user_name} detected - Control enabled")
                            self.is_authorized = True
                            self.current_user = user_name
                            self.unauthorized_detected = False
                        else:
                            if not self.unauthorized_detected:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] UNAUTHORIZED face detected - Control disabled")
                            self.is_authorized = False
                            self.current_user = None
                            self.unauthorized_detected = True
                    else:
                        self.face_not_found_count += 1
                        # Grace period of ~2 seconds before disabling
                        if self.face_not_found_count > 4:
                            if self.is_authorized:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Face lost - Control disabled")
                            self.is_authorized = False
                            self.current_user = None
                            self.unauthorized_detected = False

                hand_detected = False
                fist_closed = False

                # Only process hand gestures if authorized
                if self.is_authorized:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        hand_detected = True

                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_draw.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS
                            )

                            index_finger_tip = hand_landmarks.landmark[8]
                            screen_x, screen_y = self.hand_to_screen_coords(
                                index_finger_tip.x,
                                index_finger_tip.y
                            )
                            smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)

                            try:
                                pyautogui.moveTo(smooth_x, smooth_y)
                            except pyautogui.FailSafeException:
                                print("\nFAILSAFE triggered! Exiting...")
                                break

                            fist_closed = self.is_fist_closed(hand_landmarks)

                            if fist_closed and not self.is_hand_closed:
                                if current_time - self.last_click_time > self.click_cooldown:
                                    try:
                                        pyautogui.click()
                                        self.last_click_time = current_time
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Click!")
                                    except pyautogui.FailSafeException:
                                        print("\nFAILSAFE triggered! Exiting...")
                                        break

                            self.is_hand_closed = fist_closed

                    self.draw_authorized_overlay(frame, hand_detected, fist_closed)
                elif self.unauthorized_detected:
                    self.draw_unauthorized_overlay(frame)
                else:
                    self.draw_waiting_overlay(frame)

                cv2.imshow('Authenticated Gesture Control - Press Q to Quit', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except pyautogui.FailSafeException:
            print("\nFAILSAFE triggered! Mouse moved to corner.")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Resources released")


def main():
    parser = argparse.ArgumentParser(
        description='Authenticated Hand Gesture Mouse Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python %(prog)s --enroll Andrew    Enroll Andrew's face
  python %(prog)s --list             List all enrolled users
  python %(prog)s --delete Andrew    Delete Andrew
  python %(prog)s --delete-all       Delete all users
  python %(prog)s                    Run gesture control

Maximum users: {MAX_USERS}
        """
    )
    parser.add_argument('--enroll', metavar='NAME',
                       help='Enroll a new user with the given name')
    parser.add_argument('--list', action='store_true',
                       help='List all enrolled users')
    parser.add_argument('--delete', metavar='NAME',
                       help='Delete a specific user')
    parser.add_argument('--delete-all', action='store_true',
                       help='Delete all enrolled users')
    args = parser.parse_args()

    banner = r"""
    +=====================================================================+
    |   _   _    _    _   _ ____     ____ _____ ____ _____ _   _ ____  _____|
    |  | | | |  / \  | \ | |  _ \   / ___| ____/ ___|_   _| | | |  _ \| ____|
    |  | |_| | / _ \ |  \| | | | | | |  _|  _| \___ \ | | | | | | |_) |  _|  |
    |  |  _  |/ ___ \| |\  | |_| | | |_| | |___ ___) || | | |_| |  _ <| |___ |
    |  |_| |_/_/   \_\_| \_|____/   \____|_____|____/ |_|  \___/|_| \_\_____|
    |                                                                       |
    |    ____ ___  _   _ _____ ____   ___  _                                |
    |   / ___/ _ \| \ | |_   _|  _ \ / _ \| |                               |
    |  | |  | | | |  \| | | | | |_) | | | | |                               |
    |  | |__| |_| | |\  | | | |  _ <| |_| | |___                            |
    |   \____\___/|_| \_| |_| |_| \_\\___/|_____|                           |
    |                                                                       |
    |               [*] Authenticated Hand Gesture Control [*]              |
    |                      Face-Locked Mouse Control                        |
    +=====================================================================+
    """
    print(banner)

    try:
        # Only initialize camera for commands that need it
        needs_camera = args.enroll or (not args.list and not args.delete and not args.delete_all)
        controller = AuthenticatedGestureController(init_camera=needs_camera)

        if args.list:
            controller.list_users()
        elif args.delete_all:
            confirm = input("Are you sure you want to delete ALL users? (yes/no): ")
            if confirm.lower() == 'yes':
                controller.delete_all_users()
            else:
                print("Cancelled.")
        elif args.delete:
            controller.delete_user(args.delete)
        elif args.enroll:
            controller.enroll_face(args.enroll)
        else:
            controller.run()

    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("\nPlease ensure:")
        print("1. Camera is connected and not in use")
        print("2. Required packages are installed:")
        print("   pip install opencv-python mediapipe pyautogui face-recognition")
        print("3. You have necessary permissions")


if __name__ == "__main__":
    main()
