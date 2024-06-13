from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def webcam_play_run(cap, model, track_history):
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()

        if success:
            result = model.track(frame, persist=True, conf=0.5, iou=0.5)

            # Check if any objects were detected
            if result[0].boxes.cls.numel() > 0:
                # Get the boxes and track IDs
                cls = result[0].boxes.cls.int().cpu().item()
                boxes = result[0].boxes.xywh.cpu()
                track_ids = result[0].boxes.id.int().cpu().tolist()
                name = result[0].names[cls]

                # Visualize the results on the frame
                annotated_frame = result[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    print("cls number ", cls, "class name", name)
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking .... ", annotated_frame)
            else:
                # If no objects are detected, just display the frame
                cv2.imshow("YOLOv8 Tracking .... ", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

if __name__ == "__main__":
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)  # webcam device : 0

    # Store the track history
    track_history = defaultdict(lambda: [])

    webcam_play_run(cap=cap, model=model, track_history=track_history)
