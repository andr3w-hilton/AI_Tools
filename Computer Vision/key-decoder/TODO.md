# Key Decoder - Future Improvements

## Current Status
The key decoder is functional with the following features working:
- Real-time key detection using webcam
- Pin count detection
- Bitting code analysis (0-9 scale)
- Capture mode to freeze and annotate analysis
- Focus on bottom edge cuts (ignoring top shoulder)

## Potential Improvements

### 1. Fine-tune Cut Detection Sensitivity
**Issue:** May be missing some cuts or detecting false positives
**Solution:** Adjust the window size and threshold parameters in the `analyze_bitting` method to better identify actual cuts vs noise

### 2. Improve Depth Measurement Accuracy
**Issue:** Bitting codes might not accurately reflect actual cut depths
**Solution:**
- Calibrate depth scaling based on key type
- Consider using reference markers
- Improve edge detection parameters for clearer cut profiles

### 3. Add Smoothing/Averaging Over Multiple Frames
**Issue:** Values are "twitchy" and jump around in real-time
**Solution:**
- Capture and average detection results over 5-10 frames before displaying
- Only update display when readings are consistent
- Add confidence indicator showing stability of reading

### 4. Adjust Detection Window Size
**Issue:** The area analyzed might be too large or too small
**Solution:**
- Make the bottom edge slice configurable (currently bottom half)
- Add parameter to control what percentage of key height to analyze
- Could expose this in calibration mode

### 5. Add Visual Guides in Live View
**Issue:** Hard to know optimal key positioning
**Solution:**
- Add alignment guides/overlay showing ideal key position
- Show target zone for key placement
- Add real-time feedback on key orientation and distance

### 6. Export Captured Images
**Issue:** Can't save analysis results for later reference
**Solution:**
- Add 'E' or 'X' key to export current captured image
- Save with timestamp and bitting code in filename
- Option to save raw key image + annotated analysis
- Could also generate text file with pin count and codes

## Notes
- Algorithm now focuses on bottom edge (cuts) rather than top edge (shoulder)
- Capture feature (press 'S') helps mitigate the "twitchy" real-time values
- Debug mode ('D') useful for troubleshooting detection issues
