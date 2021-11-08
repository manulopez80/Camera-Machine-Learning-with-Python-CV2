import numpy as np
import cv2
from sklearn.metrics import pairwise

# variable global para el fondo
bg = None

#-------------------------------------------------------------
# Calcula la media ponderada, la acumula y actualiza el fondo
#-------------------------------------------------------------
def run_avg(image, media_ponderada):
    global bg

    # inicializa el fondo la primera vez
    if bg is None:
        bg = image.copy().astype("float")
        return

    # si ya existe algún fondo, actualiza la media ponderada del fondo
    cv2.accumulateWeighted(image, bg, media_ponderada)


# -----------------------------------------------
# Separación de la region de la mano en la imagen
# -----------------------------------------------
def segment(image, threshold=25):
    global bg

    # Encontramos la diferencia absoluta entre el fondo y el frame actual
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # Obtenemos el threshold a partir de la diferencia anterior
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # obtenemos los contornos en la imagen de umbral
    # findContours recibe una imagen binaria
    # RETR_EXTERNAL es el modo de obtención del contorno
    # CHAIN_APPROX_SIMPLE es el método de aproximación de los contornos
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si no se detectan contornos no hacemos nada
    if len(contours) == 0:
        return
    else:
        # Obtenemos el maximo contorno de la mano y devolvemos tanto el contorno
        # como el threshold
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)



#--------------------------------------------------------------
# Cuenta el numero de dedos en la region de la mano segmentada
#--------------------------------------------------------------
def count(thresholded, segmented):
    
    # encontramos la envolvente convexa de la region segmentada
    env_convex = cv2.convexHull(segmented)

    # buscamos los puntos extremos de la envolvente
    extreme_top    = tuple(env_convex[env_convex[:, :, 1].argmin()][0])
    extreme_bottom = tuple(env_convex[env_convex[:, :, 1].argmax()][0])
    extreme_left   = tuple(env_convex[env_convex[:, :, 0].argmin()][0])
    extreme_right  = tuple(env_convex[env_convex[:, :, 0].argmax()][0])

    # obtenemos el centro de la palma
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # calculamos la maxima distancia Euclídea entre el centro de la palma
    # y los puntos más extremos de la envolvente
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # generamos el radio del circulo al 80% de la máxima distancia Euclídea obtenida
    radius = int(0.8 * maximum_distance)

    # generamos la circunferencia del circulo
    circumference = (2 * np.pi * radius)

    # dubujamos la imagen circular que tiene la palma y los dedos
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # Hacemos el bit-wise AND con el threshold de la mano utilizando la imagen circular
    # como máscara, la cual genera los cortes de los dedos
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Encontramos los contornos en la imagen circular resultante
    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contador de dedos
    contador_dedos = 0

    # loop through the contours found
    for c in contours:
        # Obtenemos los valores del contorno 
        (x, y, w, h) = cv2.boundingRect(c)

        # Incrementamos el contador de dedos solo si:
        # 1. La region de contorno no es la muñeca
        # 2. El numero de puntos a lo largo del contorno no excede
        #    del 25% de la imagen circular 
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            contador_dedos += 1

    return contador_dedos


#-------------------
# FUNCION PRINCIPAL
#-------------------
if __name__ == "__main__":

    # Inicializar peso para media ponderada
    media_ponderada = 0.5

    # Se obtiene la referencia a la webcam
    camera = cv2.VideoCapture(0)

    # Coordenadas del recuadro de captura
    top, right, bottom, left = 10, 350, 225, 590

    # inicializamos el numero de frames
    num_frames = 0

    while True:
        # obtenemos el frame actual
        res, frame = camera.read()

        # rotamos el frame para que no se vea como un espejo
        frame = cv2.flip(frame, 1) #0 vertical, 1 horizontal

        # obtenemos la altura y el ancho del frame
        # height, width = frame.shape[:2]

        # obtenemos una copia del frame del tamaño del recuadro, la convertimos
        # a escala de grises y la difuminamos (desenfoque gaussiano) para
        # que sea más sencillo después separar el fondo
        sub_frame = frame[top:bottom, right:left]
        sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
        sub_frame = cv2.GaussianBlur(sub_frame, (7, 7), 0)

        # Para separar el fondo, acumulamos frames hasta un umbral (30)
        # para obtener su media ponderada y ser calibrado cuando supere el umbral
        # (es importante que no haya ningun movimiento en la camara durante este primer paso)
        if num_frames < 30:
            run_avg(sub_frame, media_ponderada)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # separamos la region de la mano del frame completo en una tupla
            hand = segment(sub_frame)
            if hand is not None:
                # separamos la tupla en dos variables
                thresholded, segmented = hand

                # Dibujamos los contornos en el frame completo y 
                # mostramos en otra ventana el threshold
                cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))

                # Contamos el numero de dedos y lo escribimos en pantalla
                num_dedos = count(thresholded, segmented)

                cv2.putText(frame, str(num_dedos), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
                cv2.imshow("Thresholded", thresholded)

        # dibujamos el rectangulo en verde en el frame actual
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        # incrementamos el numero de frames
        num_frames += 1

        # Mostramos el frame completo con el rectangulo
        cv2.imshow("Video", frame)

        # Si la tecla pulsada es la 'q' salimos del bucle
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# liberamos memoria de recursos
camera.release()
cv2.destroyAllWindows()

