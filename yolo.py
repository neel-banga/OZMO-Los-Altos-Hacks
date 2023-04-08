import torch
import cv2
import numpy as np
import time
import audio

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

start_time = time.time()

# Open the video capture device (webcam)
lst = []
cap = cv2.VideoCapture(0)
while time.time() - start_time < 5:
    ret, frame = cap.read()

    # Make detections
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render())) 

    df = results.pandas().xyxy[0]

    for i in df['name']:
        if i not in lst:
            #lst.append(i)
            new_label = i
            time.sleep(2)
            confidence = df['confidence'].max().item()

            #if confidence >= 0.5:
            lst.append(i)
                
            #confidence = df['confidence'].max().item()
            # might use min(), might use max() - still deciding
            print(confidence)

cap.release()

cv2.destroyAllWindows()

for i in lst:
    print(i)
    audio.say(i)