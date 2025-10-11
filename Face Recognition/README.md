# Basic Facial Recognition Application

A Python application that uses your built-in camera to detect human faces in real-time.

## Features

- Real-time face detection using OpenCV
- Visual indicators when faces are detected
- On-screen display showing:
  - Number of faces detected
  - Timestamp
  - Bounding boxes around detected faces
- Console logging of face detection events

## Requirements

- Python 3.7 or higher (Note: Python 3.14 has limited package support)
- opencv-python
- numpy

## Installation

### Option 1: Using Python 3.11 or 3.12 (Recommended)

Python 3.14 is very new and many packages don't have pre-built wheels yet. For best results, use Python 3.11 or 3.12:

```bash
# If you have multiple Python versions installed
py -3.12 -m pip install -r requirements.txt
# OR
py -3.11 -m pip install -r requirements.txt
```

### Option 2: Direct Installation (Python 3.14)

If you must use Python 3.14, you'll need to install a C++ compiler first:

1. Install Microsoft Visual Studio Build Tools:
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++" workload
   - This allows numpy to build from source

2. Then install the packages:
```bash
pip install -r requirements.txt
```

### Option 3: Try Pre-release Wheels

```bash
pip install --pre numpy opencv-python
```

## Usage

Run the application:

```bash
python face_recognition_app.py
```

Or with a specific Python version:

```bash
py -3.12 face_recognition_app.py
```

### Controls

- Press 'q' to quit the application
- The camera feed will display in a window titled "Face Recognition - Press Q to Quit"

## How It Works

The application uses:
- **OpenCV (cv2)**: For camera access and image processing
- **Haar Cascade Classifier**: Pre-trained model for face detection
- **Real-time Processing**: Captures video frames and processes them continuously

When a face is detected:
- Green rectangle drawn around the face
- "Face Detected" label displayed
- Console message logged with timestamp
- Face count displayed on screen

## Troubleshooting

### Camera Access Issues
- Ensure no other application is using your camera
- Check camera permissions in Windows Settings > Privacy > Camera
- Try restarting the application

### Installation Issues
- If packages fail to install, try using Python 3.11 or 3.12
- Make sure pip is up to date: `python -m pip install --upgrade pip`
- For numpy build errors, install Visual Studio Build Tools

### Performance Issues
- Close other applications using the camera
- Reduce the video resolution if needed
- Ensure adequate lighting for better detection

## Code Structure

- `FaceDetector`: Main class handling detection logic
  - `detect_faces()`: Detects faces in a frame
  - `draw_face_boxes()`: Draws rectangles around detected faces
  - `add_status_overlay()`: Adds status information overlay
  - `run()`: Main loop for video capture and processing

## License

This is a basic educational example for learning face detection with OpenCV.
