# Pose Estimation with Push-Up Counter

## Overview
This project implements pose estimation using the YOLO pose detection model to track a person's push-ups in real-time. The system detects key points on the body, calculates joint angles, and counts push-ups based on movement patterns.

## Features
- Real-time pose estimation using YOLO
- Angle calculation between key joints
- Push-up counting based on elbow movement
- Supports webcam and video file input
- Highlights key points and movement on video output

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy ultralytics torch
```

## Installation and Setup
1. Clone the repository or download the script.
2. Install the required dependencies using the command above.
3. Ensure you have the YOLO model file (`yolov8n-pose.pt` or `yolov11n-pose.pt`) in your working directory.
4. Run the script with the desired video input:


## How It Works
1. The script captures frames from a webcam or video file.
2. YOLO is used to detect key points of the body (shoulder, elbow, and wrist).
3. The system calculates the angle at the elbow joint using:
   
   ```python
   radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
   angle = np.abs(radians * 180.0 / np.pi)
   ```

4. Push-up count is based on the elbow angle:
   - If the angle drops below a threshold (e.g., 90°), the system marks the "down" position.
   - If the angle rises back above a threshold (e.g., 160°), a push-up is counted.
5. The video output displays the key points and push-up count in real time.

## Example Output
The system overlays:
- Keypoints on the detected joints
- Lines connecting the joints
- The real-time push-up count
- The angle of the elbow

## Customization
- Modify the threshold angles (`threshold_down` and `threshold_up`) to fine-tune the detection.
- Adjust the model file path to use a different YOLO model.
- Change the video source to process different inputs.

## License
This project is open-source and available for educational and research purposes.

---
If you encounter any issues, feel free to ask for help!

