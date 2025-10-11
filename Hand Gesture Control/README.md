# Hand Gesture Mouse Control

Control your computer mouse using hand gestures detected through your webcam. Move your hand to control the cursor, and close your fist to click!

## Features

- **Hand Tracking**: Uses MediaPipe for accurate real-time hand detection
- **Mouse Control**: Move your hand to control the cursor position
- **Gesture Recognition**: Close your fist to perform a left click
- **Smooth Movement**: Built-in smoothing for natural cursor movement
- **Visual Feedback**: On-screen display showing hand detection and gesture status
- **Safety Features**: FAILSAFE mode - move mouse to corner to stop

## How It Works

The application uses:
- **MediaPipe Hands**: Google's ML solution for hand tracking
- **OpenCV**: For camera access and video processing
- **PyAutoGUI**: For controlling mouse movements and clicks

### Gesture Detection

- **Open Hand**: Index finger controls cursor position
- **Closed Fist**: Triggers a mouse click (with cooldown to prevent multiple clicks)

The system detects a closed fist by measuring the distance between fingertips and the palm center.

## Requirements

- Python 3.7+ (Python 3.11 or 3.12 recommended due to package availability)
- Webcam
- opencv-python
- mediapipe
- pyautogui
- numpy

## Installation

### Step 1: Install Dependencies

For Python 3.11 or 3.12 (recommended):
```bash
py -3.12 -m pip install -r requirements.txt
# OR
py -3.11 -m pip install -r requirements.txt
```

For Python 3.14 (if you encounter issues, see troubleshooting):
```bash
pip install -r requirements.txt
```

### Step 2: Test Your Camera

Make sure your camera works and isn't being used by another application.

## Usage

Run the application:

```bash
python hand_gesture_mouse.py
```

Or with a specific Python version:
```bash
py -3.12 hand_gesture_mouse.py
```

### Controls

- **Move your hand** in front of the camera to move the cursor
- **Close your fist** to perform a left click
- **Press 'q'** to quit the application
- **Move mouse to top-left corner** for emergency stop (FAILSAFE)

### Tips for Best Performance

1. **Lighting**: Use good lighting so your hand is clearly visible
2. **Background**: Plain backgrounds work better than busy ones
3. **Distance**: Keep your hand 1-2 feet from the camera
4. **Movement**: Move slowly at first to get used to the control
5. **Calibration**: The system inverts the x-axis for natural mirror-like control

## Configuration

You can adjust these parameters in `hand_gesture_mouse.py`:

### Sensitivity Settings (lines 25-27)
```python
min_detection_confidence=0.7  # Lower = more sensitive detection
min_tracking_confidence=0.7   # Lower = more sensitive tracking
```

### Smoothing (line 39)
```python
self.smooth_factor = 5  # Higher = smoother but slower response
```

### Click Settings (lines 45-46)
```python
self.click_cooldown = 0.5  # Seconds between clicks
```

### Fist Detection Threshold (line 81)
```python
threshold = 0.15  # Lower = easier to trigger click
```

## Troubleshooting

### Camera Issues
- **Error opening camera**: Close other apps using the camera (Teams, Zoom, etc.)
- **Permission denied**: Check camera permissions in Windows Settings > Privacy > Camera

### Installation Issues
- **Python 3.14 problems**: Use Python 3.11 or 3.12 instead
- **MediaPipe errors**: Try `pip install --upgrade mediapipe`
- **PyAutoGUI not working**: May need to run as administrator on some systems

### Performance Issues
- **Laggy cursor**: Increase `smooth_factor` for smoother movement
- **Hand not detected**: Improve lighting or adjust `min_detection_confidence`
- **Accidental clicks**: Increase `click_cooldown` or adjust `threshold`
- **Cursor moves opposite way**: Adjust the x-axis inversion in `hand_to_screen_coords()`

### Click Detection Issues
- **Clicks not registering**: Lower the `threshold` value (line 81)
- **Too many clicks**: Increase `click_cooldown` value
- **Need tighter fist**: Increase `threshold` value

## Safety Features

### FAILSAFE Mode
PyAutoGUI's FAILSAFE is enabled by default. If something goes wrong:
1. Quickly move your physical mouse to the top-left corner of the screen
2. The program will immediately stop

This is a safety feature to prevent the program from taking over your computer.

### Exit Options
- Press 'q' in the video window
- Use FAILSAFE (move to corner)
- Ctrl+C in the terminal

## Code Structure

### Main Classes
- `HandGestureController`: Main controller class
  - `calculate_distance()`: Calculate distance between hand landmarks
  - `is_fist_closed()`: Detect if hand is in fist position
  - `smooth_coordinates()`: Apply smoothing to mouse movement
  - `hand_to_screen_coords()`: Convert hand position to screen coordinates
  - `draw_overlay()`: Display status information
  - `run()`: Main control loop

### Hand Landmarks
MediaPipe detects 21 landmarks on each hand:
- Landmark 8: Index finger tip (used for cursor position)
- Landmark 9: Middle finger base (used as palm center)
- Landmarks 4, 8, 12, 16, 20: All fingertips (used for fist detection)

## Future Enhancements

Possible additions:
- Right-click gesture (e.g., two fingers closed)
- Scroll gesture (e.g., pinch to zoom)
- Double-click gesture
- Drag and drop functionality
- Multi-hand support for different functions
- Custom gesture mapping
- Gesture recording and playback

## Known Limitations

- Requires good lighting conditions
- May not work well with very fast movements
- Single hand tracking only
- Left click only (no right-click yet)
- Requires visible camera view of hand

## License

Educational project for learning computer vision and gesture control.
