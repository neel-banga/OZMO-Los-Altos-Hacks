import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import audio
import time

def read_text(image):

    string = pytesseract.image_to_string(image)

    return string

def read_label():

    start_time = time.time()

    cap = cv2.VideoCapture(0)

    while time.time() - start_time < 20:
        ret, frame = cap.read()
        
        if not ret:
            break

        cv2.imshow('Real-time', cap.read()[1])

        try:    
            text = read_text(frame)
            
            if text != None:
                audio.say(text)
                break
        
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
