import numpy as np
import cv2
from playsound import playsound

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def paint_face(image):
    image1 = image.copy()
    rectangles1 = face_cascade.detectMultiScale(image)
    #rectangles2 = eyes_cascade.detectMultiScale(image)

    for (x,y,w,h) in rectangles1:
    #for (x,y,w,h) in rectangles2:
        cv2.rectangle(image1, (x,y), (x+w,y+h), (0,0,255), 2)
        #if w>0 
            #playsound('alarma.mp3')
    return image1

captura = cv2.VideoCapture(0)

while True:
    res, video = captura.read(0)
    if res:
        video = paint_face(video)
        cv2.imshow('Manu test to detect faces', video)
        tecla = cv2.waitKey(1)
        if tecla==27:
            break

captura.release()
cv2.destroyAllWindows()