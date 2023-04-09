import torch
import cv2
import numpy as np
import time
import audio


def yolo():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    start_time = time.time()

    audio.say('I CAN SEE A...')

    # Open the video capture device (webcam)
    lst = []
    cap = cv2.VideoCapture(0)
    while time.time() - start_time < 10:
        ret, frame = cap.read()

        # Make detections
        results = model(frame)
        cv2.imshow('YOLO', np.squeeze(results.render()))

        df = results.pandas().xyxy[0]

        for i in df['name']:
            if i not in lst:
                lst.append(i)
                new_label = i
                time.sleep(2)
                confidence = df[df['name'] == new_label]['confidence'].max().item()

                print(new_label)

                if confidence >= 0.35:
                    # Determine direction based on object's bounding box center
                    bbox_center_x = (df[df['name'] == new_label]['xmax'].iloc[0] + df[df['name'] == new_label]['xmin'].iloc[0]) / 2
                    bbox_center_y = (df[df['name'] == new_label]['ymax'].iloc[0] + df[df['name'] == new_label]['ymin'].iloc[0]) / 2
                    frame_center_x = frame.shape[1] / 2
                    frame_center_y = frame.shape[0] / 2

                    if bbox_center_x > frame_center_x:
                        direction_x = "to your right"
                    else:
                        direction_x = "to your left"

                    if bbox_center_y < frame_center_y: # Flip the comparison for vertical direction
                        direction_y = "above you" # Flip the direction description
                    else:
                        direction_y = "below you" # Flip the direction description

                    audio.say(f"{new_label} {direction_x} {direction_y}")

    cap.release()

    cv2.destroyAllWindows()
