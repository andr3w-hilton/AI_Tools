# Key Decoder

A computer vision application that analyzes physical keys using your laptop camera to determine the number of pins and bitting codes.

## Features

- **Real-time key detection**: Automatically detects and identifies keys in the camera frame
- **Pin counting**: Determines the number of pins from the key cuts
- **Bitting analysis**: Calculates bitting depth codes (0-9 scale, where 0 is shallowest and 9 is deepest)
- **Interactive calibration**: Adjust detection parameters for different lighting conditions
- **Debug mode**: View edge detection and extracted key profile for troubleshooting

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

## Installation

The required packages should already be installed, but if needed:

```bash
pip install opencv-python numpy
```

## Usage

Run the application:

```bash
python key_decoder.py
```

### Controls

- **Q**: Quit the application
- **D**: Toggle debug mode (shows edge detection and extracted key region)
- **C**: Open calibration window to adjust detection parameters

### How to Use

1. Run the application
2. Hold a key up to your camera with the **blade profile clearly visible**
3. Position the key horizontally with the cuts facing up or to the side
4. The application will display:
   - Detection status (KEY DETECTED / NO KEY DETECTED)
   - Pin count
   - Bitting codes from front to back

### Tips for Best Results

- **Good lighting**: Ensure the key is well-lit with even lighting
- **Contrast**: Hold the key against a contrasting background (e.g., white key on dark background)
- **Steady hand**: Keep the key as steady as possible
- **Horizontal orientation**: Position the key horizontally for best detection
- **Clear profile**: Make sure the blade edge with cuts is clearly visible
- **Distance**: Hold the key 6-12 inches from the camera

### Calibration

If the key is not being detected properly:

1. Press **C** to open the calibration window
2. Adjust the trackbars:
   - **Canny Low**: Lower threshold for edge detection (default: 50)
   - **Canny High**: Upper threshold for edge detection (default: 150)
   - **Min Area**: Minimum contour area to consider as a key (default: 5000)
3. Observe the main window to see the effect of changes
4. Enable debug mode (press **D**) to see edge detection results

## How It Works

1. **Image Preprocessing**: Converts camera frames to grayscale and applies filters
2. **Edge Detection**: Uses Canny edge detection to find key boundaries
3. **Contour Detection**: Identifies the key shape by finding large, elongated contours
4. **Profile Extraction**: Rotates and extracts the key blade profile
5. **Bitting Analysis**: Scans the top edge to find cuts (valleys) and measures their depths
6. **Depth Coding**: Normalizes cut depths to a 0-9 scale for standard bitting codes

## Limitations

- Currently optimized for standard pin tumbler keys (house keys like Kwikset, Schlage)
- Accuracy depends on lighting conditions and camera quality
- Bitting codes are relative measurements (not absolute depth measurements)
- May require calibration for different key types or lighting conditions

## Future Enhancements

- Support for different key types (dimple keys, tubular keys)
- Machine learning model for more robust detection
- Database of common key systems for automatic identification
- Export functionality to save key profiles
- Reference object support for absolute measurements

## Notes

This tool is designed for educational purposes and legitimate locksmithing applications. Always ensure you have authorization before duplicating keys.
