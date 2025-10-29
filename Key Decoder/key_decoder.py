"""
Key Decoder Application
Analyzes physical keys using computer vision to determine pin count and bitting codes
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class KeyDecoder:
    def __init__(self):
        """Initialize the key decoder with default parameters"""
        self.cap = None
        self.calibrated = False

        # Edge detection parameters (adjustable)
        self.canny_low = 50
        self.canny_high = 150
        self.blur_kernel = 5

        # Key detection parameters
        self.min_contour_area = 2000  # Reduced for smaller keys in frame
        self.min_aspect_ratio = 2.0  # More lenient - keys are typically long and narrow

        # Display settings
        self.window_name = "Key Decoder"
        self.show_debug = False

        # Capture mode
        self.captured = False
        self.captured_image = None
        self.captured_pin_count = 0
        self.captured_bitting = []
        self.captured_cuts = []  # Store cut positions for visualization

    def start_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize and start the webcam

        Args:
            camera_index: Camera device index (default 0)

        Returns:
            True if camera started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False

        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        return True

    def stop_camera(self):
        """Release camera and close windows"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess frame for key detection

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (gray image, edge-detected image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (self.blur_kernel, self.blur_kernel), 0)

        # Edge detection using Canny
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return gray, edges

    def find_key_contour(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the key contour in the edge-detected image

        Args:
            edges: Edge-detected binary image

        Returns:
            Key contour array or None if not found
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            # Get bounding rectangle to check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(min(w, h), 1)

            if aspect_ratio >= self.min_aspect_ratio:
                valid_contours.append((contour, area))

        if not valid_contours:
            return None

        # Return the largest valid contour (most likely the key)
        key_contour = max(valid_contours, key=lambda x: x[1])[0]
        return key_contour

    def extract_key_profile(self, contour: np.ndarray, gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the blade profile of the key from the contour

        Args:
            contour: Key contour
            gray_img: Grayscale image

        Returns:
            Tuple of (rotated key region, angle of rotation)
        """
        # Get minimum area rectangle (to handle rotated keys)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Get center, size and angle
        center, size, angle = rect

        # Rotate the image to align the key horizontally
        if size[0] < size[1]:
            angle = angle + 90
            size = (size[1], size[0])

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the entire image
        height, width = gray_img.shape
        rotated = cv2.warpAffine(gray_img, M, (width, height))

        # Extract the key region
        key_width = int(size[0])
        key_height = int(size[1])

        center_x, center_y = int(center[0]), int(center[1])

        # Calculate extraction bounds
        x1 = max(0, center_x - key_width // 2)
        x2 = min(width, center_x + key_width // 2)
        y1 = max(0, center_y - key_height // 2)
        y2 = min(height, center_y + key_height // 2)

        key_region = rotated[y1:y2, x1:x2]

        return key_region, angle

    def analyze_bitting(self, key_region: np.ndarray) -> Tuple[int, List[int], List[Tuple[int, float]]]:
        """
        Analyze the key bitting to determine pin count and depths

        Args:
            key_region: Extracted and aligned key region

        Returns:
            Tuple of (pin count, bitting depths list, cut positions)
        """
        if key_region.size == 0:
            return 0, [], []

        height, width = key_region.shape

        # Focus on the BOTTOM edge of the key (where cuts are)
        # Take a slice from the bottom half of the key
        bottom_start = height // 2
        edge_region = key_region[bottom_start:height, :]

        # Apply edge detection to find the profile
        edges = cv2.Canny(edge_region, 30, 100)

        # Find the bottom edge profile by scanning each column
        profile = []
        for col in range(width):
            column_slice = edges[:, col]
            edge_points = np.where(column_slice > 0)[0]

            if len(edge_points) > 0:
                # Use the LAST (bottommost) edge point in the bottom half
                profile.append(edge_points[-1])
            else:
                profile.append(0)  # No edge found, use top of region

        if len(profile) < 10:
            return 0, [], []

        # Smooth the profile
        profile = np.array(profile, dtype=np.float32)
        profile = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (15, 15), 0).flatten()

        # Find valleys in the bottom edge profile (cuts = lower points on bottom edge)
        # Since we're looking at the bottom edge, cuts appear as local minima (valleys)
        cuts = []
        window_size = max(10, width // 20)

        for i in range(window_size, len(profile) - window_size):
            window = profile[i-window_size:i+window_size]
            if profile[i] == np.min(window):
                # This is a local minimum (cut in the bottom edge)
                # Adjust y-coordinate to be relative to full key height
                adjusted_y = bottom_start + profile[i]
                cuts.append((i, adjusted_y))

        # Filter cuts that are too close together
        filtered_cuts = []
        min_distance = width // 12  # Minimum distance between cuts

        if cuts:
            filtered_cuts.append(cuts[0])
            for cut in cuts[1:]:
                if cut[0] - filtered_cuts[-1][0] >= min_distance:
                    filtered_cuts.append(cut)

        pin_count = len(filtered_cuts)

        # Calculate bitting depths (0-9 scale, 0 = shallowest, 9 = deepest)
        # For bottom edge: higher y value = deeper cut (further from top)
        # Lower y value = shallower cut (closer to top, less material removed)
        bitting_codes = []
        if filtered_cuts:
            depths = [cut[1] for cut in filtered_cuts]
            min_depth = min(depths)  # Shallowest cut (highest point on bottom edge)
            max_depth = max(depths)  # Deepest cut (lowest point on bottom edge)
            depth_range = max_depth - min_depth

            if depth_range > 0:
                for depth in depths:
                    # Normalize to 0-9 scale
                    # Lower y (closer to top) = shallow = low number
                    # Higher y (closer to bottom) = deep = high number
                    normalized = (depth - min_depth) / depth_range
                    code = int(normalized * 9)
                    bitting_codes.append(code)
            else:
                bitting_codes = [0] * pin_count

        return pin_count, bitting_codes, filtered_cuts

    def draw_overlay(self, frame: np.ndarray, pin_count: int, bitting_codes: List[int],
                     key_detected: bool) -> np.ndarray:
        """
        Draw information overlay on the frame

        Args:
            frame: Input frame
            pin_count: Number of pins detected
            bitting_codes: List of bitting depth codes
            key_detected: Whether a key was detected

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "KEY DECODER", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        # Detection status
        if key_detected:
            status_text = "KEY DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "NO KEY DETECTED"
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Pin count
        cv2.putText(frame, f"Pin Count: {pin_count}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bitting code
        if bitting_codes:
            bitting_str = " - ".join(map(str, bitting_codes))
            cv2.putText(frame, f"Bitting: {bitting_str}", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Bitting: N/A", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Instructions
        if self.captured:
            cv2.putText(frame, "CAPTURED - Press 'R' to resume | 'Q' to quit",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "'S' to capture | 'Q' quit | 'D' debug | 'C' calibrate",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def capture_analysis(self, key_region: np.ndarray, pin_count: int,
                        bitting_codes: List[int], cuts: List[Tuple[int, float]]):
        """
        Capture and visualize the key analysis with annotations

        Args:
            key_region: Extracted key region
            pin_count: Number of pins detected
            bitting_codes: List of bitting codes
            cuts: List of (x_position, depth) tuples for each cut
        """
        # Create a color version of the key region for visualization
        key_color = cv2.cvtColor(key_region, cv2.COLOR_GRAY2BGR)

        height, width = key_region.shape

        # Draw the cuts and labels
        for i, (cut_x, depth) in enumerate(cuts):
            x = int(cut_x)
            # Draw vertical line at cut position
            cv2.line(key_color, (x, 0), (x, height), (0, 255, 0), 2)

            # Draw circle at the cut point
            y = int(depth)
            cv2.circle(key_color, (x, y), 5, (0, 0, 255), -1)

            # Add bitting code label
            if i < len(bitting_codes):
                label = str(bitting_codes[i])
                # Position label below the key
                label_y = height - 20 if i % 2 == 0 else height - 40
                cv2.putText(key_color, label, (x - 10, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Create a larger display image with info panel
        display_height = max(600, height * 3)
        display_width = max(800, width * 2)

        display_img = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Resize and center the key image
        scale = min(display_width * 0.8 / width, display_height * 0.5 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        key_resized = cv2.resize(key_color, (new_width, new_height))

        # Center the key image
        y_offset = 50
        x_offset = (display_width - new_width) // 2
        display_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = key_resized

        # Add info panel below
        info_y = y_offset + new_height + 50

        cv2.putText(display_img, "KEY ANALYSIS CAPTURED", (50, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        cv2.putText(display_img, f"Pin Count: {pin_count}", (50, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if bitting_codes:
            bitting_str = " - ".join(map(str, bitting_codes))
            cv2.putText(display_img, f"Bitting Code: {bitting_str}", (50, info_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(display_img, "(Front to Back)", (50, info_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Add legend
        legend_y = info_y + 130
        cv2.putText(display_img, "Legend:", (50, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.line(display_img, (50, legend_y + 15), (80, legend_y + 15), (0, 255, 0), 2)
        cv2.putText(display_img, "Cut position", (90, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.circle(display_img, (65, legend_y + 45), 5, (0, 0, 255), -1)
        cv2.putText(display_img, "Cut depth point", (90, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_img, "0 = Shallow, 9 = Deep", (90, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        self.captured = True
        self.captured_image = display_img
        self.captured_pin_count = pin_count
        self.captured_bitting = bitting_codes
        self.captured_cuts = cuts

    def run(self):
        """Main application loop"""
        if not self.start_camera():
            return

        print("Key Decoder started!")
        print("Hold a key up to the camera with the blade profile visible.")
        print("Press 'S' to capture analysis, 'R' to resume, 'Q' to quit")
        print("Press 'D' for debug view, 'C' to adjust calibration")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Preprocess
            gray, edges = self.preprocess_frame(frame)

            # Find key contour
            key_contour = self.find_key_contour(edges)

            pin_count = 0
            bitting_codes = []
            cuts = []
            key_detected = False

            if key_contour is not None:
                key_detected = True

                # Draw key contour
                cv2.drawContours(frame, [key_contour], -1, (0, 255, 0), 2)

                # Extract and analyze key profile
                key_region, angle = self.extract_key_profile(key_contour, gray)

                if key_region.size > 0:
                    pin_count, bitting_codes, cuts = self.analyze_bitting(key_region)

                    # Show extracted key region if debug mode
                    if self.show_debug and key_region.shape[0] > 0 and key_region.shape[1] > 0:
                        # Resize for display
                        display_region = cv2.resize(key_region, None, fx=2, fy=2)
                        cv2.imshow("Key Region", display_region)

            # Draw overlay
            frame = self.draw_overlay(frame, pin_count, bitting_codes, key_detected)

            # Show debug view if enabled
            if self.show_debug:
                debug_display = cv2.resize(edges, (edges.shape[1]//2, edges.shape[0]//2))
                cv2.imshow("Edge Detection", debug_display)

                # Draw ALL contours in debug mode to see what's being detected
                debug_frame = frame.copy()
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 1)

                # Show contour info
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 500:  # Only show significant contours
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = max(w, h) / max(min(w, h), 1)
                        cv2.putText(debug_frame, f"A:{int(area)} R:{aspect_ratio:.1f}",
                                  (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                debug_frame_small = cv2.resize(debug_frame, (debug_frame.shape[1]//2, debug_frame.shape[0]//2))
                cv2.imshow("All Contours", debug_frame_small)

            # Display main frame or captured image
            if self.captured:
                cv2.imshow(self.window_name, self.captured_image)
            else:
                cv2.imshow(self.window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                self.show_debug = not self.show_debug
                if not self.show_debug:
                    cv2.destroyWindow("Edge Detection")
                    cv2.destroyWindow("Key Region")
                    cv2.destroyWindow("All Contours")
                print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('c') or key == ord('C'):
                self.adjust_calibration()
            elif key == ord('s') or key == ord('S'):
                # Capture/Snapshot the current detection
                if key_detected and pin_count > 0:
                    self.capture_analysis(key_region, pin_count, bitting_codes, cuts)
                    print(f"CAPTURED! Pin Count: {pin_count}, Bitting: {'-'.join(map(str, bitting_codes))}")
                else:
                    print("No key detected - cannot capture")
            elif key == ord('r') or key == ord('R'):
                # Resume live view
                if self.captured:
                    self.captured = False
                    print("Resumed live view")

        self.stop_camera()
        print("Key Decoder stopped.")

    def adjust_calibration(self):
        """Interactive calibration adjustment"""
        print("\n=== CALIBRATION MODE ===")
        print("Adjust edge detection parameters:")
        print(f"Current Canny Low: {self.canny_low}")
        print(f"Current Canny High: {self.canny_high}")
        print(f"Current Min Area: {self.min_contour_area}")
        print("\nUse trackbars in the new window to adjust")

        # Create calibration window with trackbars
        cv2.namedWindow("Calibration")
        cv2.createTrackbar("Canny Low", "Calibration", self.canny_low, 255, self.on_canny_low_change)
        cv2.createTrackbar("Canny High", "Calibration", self.canny_high, 255, self.on_canny_high_change)
        cv2.createTrackbar("Min Area", "Calibration", self.min_contour_area // 100, 500, self.on_min_area_change)

    def on_canny_low_change(self, value):
        self.canny_low = value

    def on_canny_high_change(self, value):
        self.canny_high = value

    def on_min_area_change(self, value):
        self.min_contour_area = value * 100


def main():
    """Entry point for the key decoder application"""
    decoder = KeyDecoder()
    try:
        decoder.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        decoder.stop_camera()
    except Exception as e:
        print(f"Error: {e}")
        decoder.stop_camera()


if __name__ == "__main__":
    main()
