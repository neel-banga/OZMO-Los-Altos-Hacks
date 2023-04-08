import torch
import cv2
import numpy as np
import time
import audio

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

start_time = time.time()

audio.say('DETECTING OBJECTS...')

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
            #lst.append(i)
            new_label = i
            time.sleep(2)
            confidence = df['confidence'].max().item()
            
            print(new_label)

            if confidence >= 0.3:
                lst.append(new_label)

print(lst)

audio.say('I CAN SEE A')

for i in lst:
    print(i)
    audio.say(i)

audio.say('NEAR YOU')

cap.release()

cv2.destroyAllWindows()