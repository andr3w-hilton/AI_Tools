"""
Hand Gesture Mouse Control
Uses MediaPipe to track hand movements and control the mouse cursor
Close your hand to perform a left click
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from datetime import datetime
import time


class HandGestureController:
    def __init__(self):
        """Initialize the hand gesture controller"""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise Exception("Could not open camera")

        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Get camera frame dimensions
        self.cam_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Control zone mapping - smaller hand movement covers full screen
        # These values define the active zone in the camera frame (0.0 to 1.0)
        # Adjust these to change sensitivity - smaller range = less hand movement needed
        self.control_zone_x_min = 0.2  # Left boundary
        self.control_zone_x_max = 0.8  # Right boundary
        self.control_zone_y_min = 0.1  # Top boundary
        self.control_zone_y_max = 0.7  # Bottom boundary

        # Smoothing parameters
        self.smooth_factor = 5
        self.prev_x, self.prev_y = 0, 0

        # Click detection
        self.is_hand_closed = False
        self.click_cooldown = 0.5  # seconds between clicks
        self.last_click_time = 0

        # PyAutoGUI settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.01

        print("Hand Gesture Controller Initialized")
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        print(f"Camera size: {self.cam_width}x{self.cam_height}")
        print(f"Control zone: X({self.control_zone_x_min}-{self.control_zone_x_max}), Y({self.control_zone_y_min}-{self.control_zone_y_max})")

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_fist_closed(self, hand_landmarks):
        """
        Detect if hand is closed (fist) by checking if fingertips
        are close to palm
        """
        # Get palm center (landmark 9 is middle finger base)
        palm_center = hand_landmarks.landmark[9]

        # Check distances from fingertips to palm
        # Thumb tip (4), Index tip (8), Middle tip (12), Ring tip (16), Pinky tip (20)
        fingertip_indices = [4, 8, 12, 16, 20]

        distances = []
        for tip_idx in fingertip_indices:
            tip = hand_landmarks.landmark[tip_idx]
            distance = self.calculate_distance(tip, palm_center)
            distances.append(distance)

        # If average distance is small, hand is closed
        avg_distance = np.mean(distances)
        threshold = 0.15  # Adjust this value for sensitivity

        return avg_distance < threshold

    def smooth_coordinates(self, x, y):
        """Apply smoothing to mouse coordinates for smoother movement"""
        smooth_x = int(self.prev_x + (x - self.prev_x) / self.smooth_factor)
        smooth_y = int(self.prev_y + (y - self.prev_y) / self.smooth_factor)

        self.prev_x = smooth_x
        self.prev_y = smooth_y

        return smooth_x, smooth_y

    def hand_to_screen_coords(self, hand_x, hand_y):
        """
        Convert hand coordinates (0-1 range) to screen coordinates
        with inverted x-axis to match natural movement.
        Maps the control zone to full screen for less hand movement.
        """
        # Map hand position within control zone to 0-1 range
        # If hand is outside control zone, clamp it to the edges
        normalized_x = (hand_x - self.control_zone_x_min) / (self.control_zone_x_max - self.control_zone_x_min)
        normalized_y = (hand_y - self.control_zone_y_min) / (self.control_zone_y_max - self.control_zone_y_min)

        # Clamp to 0-1 range
        normalized_x = max(0.0, min(1.0, normalized_x))
        normalized_y = max(0.0, min(1.0, normalized_y))

        # Map directly to screen - hand left = cursor left, hand right = cursor right
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)

        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return screen_x, screen_y

    def draw_overlay(self, frame, hand_detected, fist_closed):
        """Draw status overlay on the frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw control zone rectangle on frame
        zone_x1 = int(self.control_zone_x_min * width)
        zone_x2 = int(self.control_zone_x_max * width)
        zone_y1 = int(self.control_zone_y_min * height)
        zone_y2 = int(self.control_zone_y_max * height)
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (100, 100, 255), 2)
        cv2.putText(frame, "Control Zone", (zone_x1 + 5, zone_y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # Background for status
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status text
        status = "Hand Detected" if hand_detected else "No Hand Detected"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Fist status
        if hand_detected:
            fist_status = "CLICKING" if fist_closed else "Open Hand"
            fist_color = (0, 255, 255) if fist_closed else (255, 255, 255)
            cv2.putText(frame, fist_status, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fist_color, 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit | Move to corner to FAILSAFE",
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 255), 1)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main loop for hand gesture control"""
        print("\n" + "="*50)
        print("Hand Gesture Mouse Control Started")
        print("="*50)
        print("Instructions:")
        print("- Move your hand to control the mouse")
        print("- Close your fist to click")
        print("- Press 'q' to quit")
        print("- Move mouse to top-left corner for FAILSAFE stop")
        print("="*50 + "\n")

        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Flip frame horizontally for mirror view
                frame = cv2.flip(frame, 1)

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame
                results = self.hands.process(rgb_frame)

                hand_detected = False
                fist_closed = False

                # If hand is detected
                if results.multi_hand_landmarks:
                    hand_detected = True

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        # Get index finger tip position (landmark 8)
                        index_finger_tip = hand_landmarks.landmark[8]

                        # Convert to screen coordinates
                        screen_x, screen_y = self.hand_to_screen_coords(
                            index_finger_tip.x,
                            index_finger_tip.y
                        )

                        # Apply smoothing
                        smooth_x, smooth_y = self.smooth_coordinates(
                            screen_x,
                            screen_y
                        )

                        # Move mouse
                        try:
                            pyautogui.moveTo(smooth_x, smooth_y)
                        except pyautogui.FailSafeException:
                            print("\nFAILSAFE triggered! Exiting...")
                            break

                        # Check if fist is closed
                        fist_closed = self.is_fist_closed(hand_landmarks)

                        # Handle clicking with cooldown
                        current_time = time.time()
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

                # Draw overlay
                self.draw_overlay(frame, hand_detected, fist_closed)

                # Display the frame
                cv2.imshow('Hand Gesture Control - Press Q to Quit', frame)

                # Check for quit
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
    """Main entry point"""
    print("="*50)
    print("Hand Gesture Mouse Control System")
    print("="*50)
    print()

    try:
        controller = HandGestureController()
        controller.run()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("\nPlease ensure:")
        print("1. Camera is connected and not in use")
        print("2. Required packages are installed")
        print("3. You have necessary permissions")


if __name__ == "__main__":
    main()

