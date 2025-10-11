"""
Basic Facial Recognition Application
Uses OpenCV to detect human faces in real-time from webcam feed
"""

import cv2
import sys
from datetime import datetime


class FaceDetector:
    def __init__(self):
        """Initialize the face detector with Haar Cascade classifier"""
        # Load the pre-trained Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Initialize video capture from default camera (0)
        self.video_capture = cv2.VideoCapture(0)

        if not self.video_capture.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        self.face_detected = False
        self.face_count = 0

    def detect_faces(self, frame):
        """
        Detect faces in the given frame

        Args:
            frame: The image frame to process

        Returns:
            faces: Array of detected face coordinates
        """
        # Convert frame to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def draw_face_boxes(self, frame, faces):
        """
        Draw rectangles around detected faces

        Args:
            frame: The image frame to draw on
            faces: Array of face coordinates
        """
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label
            cv2.putText(
                frame,
                'Face Detected',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    def add_status_overlay(self, frame, face_count):
        """
        Add status information overlay to the frame

        Args:
            frame: The image frame to draw on
            face_count: Number of faces detected
        """
        # Add background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Add status text
        status_text = f"Faces Detected: {face_count}"
        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Add instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def run(self):
        """Main loop to capture and process video frames"""
        print("Starting face detection...")
        print("Press 'q' to quit")

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.video_capture.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # Detect faces
                faces = self.detect_faces(frame)
                self.face_count = len(faces)

                # Track face detection state
                if self.face_count > 0 and not self.face_detected:
                    self.face_detected = True
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Human face(s) detected!")
                elif self.face_count == 0 and self.face_detected:
                    self.face_detected = False
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No faces detected")

                # Draw rectangles around faces
                self.draw_face_boxes(frame, faces)

                # Add status overlay
                self.add_status_overlay(frame, self.face_count)

                # Display the resulting frame
                cv2.imshow('Face Recognition - Press Q to Quit', frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


def main():
    """Main entry point"""
    print("=" * 50)
    print("Basic Facial Recognition Application")
    print("=" * 50)
    print()

    detector = FaceDetector()
    detector.run()


if __name__ == "__main__":
    main()
