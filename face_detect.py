import numpy as np
import cv2
from playsound import playsound

cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cascada_ojos = cv2.CascadeClassifier('haarcascade_eye.xml')

def pintar_cara(imagen):
    imagen1 = imagen.copy()
    rectangulos1 = cascada_cara.detectMultiScale(imagen)
    #rectangulos2 = cascada_ojos.detectMultiScale(imagen)

    for (x,y,w,h) in rectangulos1:
    #for (x,y,w,h) in rectangulos2:
        cv2.rectangle(imagen1, (x,y), (x+w,y+h), (0,0,255), 2)
        #if w>0 
            #playsound('alarma.mp3')
    return imagen1

captura = cv2.VideoCapture(0)

while True:
    res, video = captura.read(0)
    if res:
        video = pintar_cara(video)
        cv2.imshow('Prueba de Manu para detectar caras', video)
        tecla = cv2.waitKey(1)
        if tecla==27:
            break

captura.release()
cv2.destroyAllWindows()