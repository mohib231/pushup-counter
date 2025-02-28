import cv2
import math
import numpy as np
from ultralytics import YOLO

def calculate_angle(a, b, c):
    """Calculate the angle between three points (in degrees)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Specify video input source (webcam or file)
video_source = 'pushup.mp4'  # Replace with video file path if needed
cap = cv2.VideoCapture(video_source)

# Get frame properties (if writing output video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Optional: VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Load YOLO pose model
try:
    model = YOLO('yolo11l-pose.pt')
    print("Loaded YOLOv11-pose model.")
except Exception as e:
    print(f"Failed to load YOLOv11-pose: {str(e)}. Loading YOLOv8n-pose.pt...")
    model = YOLO('yolov8n-pose.pt')

# Pushup counter variables
counter = 0
is_down = False
threshold_down = 90   # Angle when going down
threshold_up = 160    # Angle when going up

# Variable to store the initial "up" shoulder y-coordinate
initial_shoulder_y = None
shoulder_tolerance = 10  # Tolerance in pixels for considering the pose "the same"

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run pose detection on the frame
        results = model(frame, verbose=False)

        # Check for detected keypoints
        if results[0].keypoints is not None and results[0].keypoints.xy.shape[1] > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            # Extract keypoints for right shoulder, elbow, wrist (indices may vary)
            shoulder = keypoints[6]
            elbow = keypoints[8]
            wrist = keypoints[10]

            # Calculate the elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # If we haven't recorded the starting pose yet and we're in an "up" position, do so.
            if initial_shoulder_y is None and angle > threshold_up:
                initial_shoulder_y = shoulder[1]

            # Pushup counting logic using elbow angle
            if angle < threshold_down and not is_down:
                is_down = True
            elif angle > threshold_up and is_down:
                # Before counting, check if the shoulder returned to the "up" position
                if initial_shoulder_y is not None and abs(shoulder[1] - initial_shoulder_y) < shoulder_tolerance:
                    counter += 1
                    is_down = False
                    # Optional: Update the reference shoulder position in case the user shifts slightly
                    initial_shoulder_y = shoulder[1]
                # If the shoulder isn’t at the expected position, wait for a more complete return

            # Visual feedback: show angle and pushup count
            color = (0, 255, 0) if is_down else (0, 0, 255)
            cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Pushups: {counter}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw keypoints and lines for shoulder, elbow, and wrist
            for point in [shoulder, elbow, wrist]:
                cv2.circle(frame, tuple(point.astype(int)), 5, (255, 0, 0), -1)
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (0, 255, 0), 2)
            cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Pushup Counter', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
