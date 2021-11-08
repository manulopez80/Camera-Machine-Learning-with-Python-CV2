import cv2
import numpy as np
from pygame import mixer

# Color a detectar para que suene
lower = [17, 15, 100] # color de piel rosado
upper = [80, 76, 220] # color de piel moreno

# Inicializamos el mixer
mixer.init()

# region coordinates
platillo_corto_top, platillo_corto_bottom, platillo_corto_left, platillo_corto_right  = 50, 150, 120, 220
platillo_largo_top, platillo_largo_bottom, platillo_largo_left, platillo_largo_right = 50, 150, 320, 420
tambor1_top, tambor1_bottom, tambor1_left, tambor1_right = 50, 150, 580, 680
tambor2_top, tambor2_bottom, tambor2_left, tambor2_right = 50, 150, 780, 880
redoble_top, redoble_bottom, redoble_left, redoble_right = 50, 150, 1030, 1130

#----------------------
# Reproducir sonidos
#----------------------
def play_platillo_corto():
	mixer.music.load('platillo_corto.mp3')
	mixer.music.play()
	print("Detectado P1")

def play_platillo_largo():
	mixer.music.load('platillo_largo.mp3')
	mixer.music.play()
	print("Detectado P2")

def play_tambor1():
	mixer.music.load('tambor1.mp3')
	mixer.music.play()
	print("Detectado T1")

def play_tambor2():
	mixer.music.load('tambor2.mp3')
	mixer.music.play()
	print("Detectado T2")

def play_redoble():
	mixer.music.load('redoble.mp3')
	mixer.music.play()
	print("Detectado R")

#------------------------------------------------------------
# Encontramos los contornos y devolvemos la cantidad de ellos
#------------------------------------------------------------
def findContours(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresholded = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
	contours, _  = cv2.findContours(thresholded.copy(),
									cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)
	return len(contours)

# Booleanos para reproducir sonido o no
hay_tambor2 = False
hay_tambor1  = False
hay_platillo_corto = False
hay_platillo_largo = False
hay_redoble = False

#----------------------
# FUNCION PRINCIPAL
#----------------------
if __name__ == "__main__":

	# obtenemos referencia de la camara
	cam = cv2.VideoCapture(0)

	# modificamos sus dimensiones y fps
	cam.set(3, 1280)
	cam.set(4, 720)
	cam.set(cv2.CAP_PROP_FPS, 60)

	while True:

		# leemos un frame de la camara
		status, frame = cam.read()

		# generamos una copia y evitamos el efecto espejo 
		clone = frame.copy()
		clone = cv2.flip(clone, 1)
		clone = cv2.resize(clone, (1280,720))

		# obtenemos las regiones de los sonidos
		reg_platillo_corto = clone[platillo_corto_top:platillo_corto_bottom, platillo_corto_left:platillo_corto_right]
		reg_platillo_largo = clone[platillo_largo_top:platillo_largo_bottom, platillo_largo_left:platillo_largo_right]
		reg_tambor1  = clone[tambor1_top:tambor1_bottom, tambor1_left:tambor1_right]
		reg_tambor2 = clone[tambor2_top:tambor2_bottom, tambor2_left:tambor2_right]
		reg_redoble = clone[redoble_top:redoble_bottom, redoble_left:redoble_right]

		# suavizamos esas regiones
		reg_platillo_corto = cv2.GaussianBlur(reg_platillo_corto, (7, 7), 0)
		reg_platillo_largo = cv2.GaussianBlur(reg_platillo_largo, (7, 7), 0)
		reg_tambor1  = cv2.GaussianBlur(reg_tambor1,  (7, 7), 0)
		reg_tambor2 = cv2.GaussianBlur(reg_tambor2, (7, 7), 0)
		reg_redoble = cv2.GaussianBlur(reg_redoble, (7, 7), 0)

		l = np.array(lower, dtype="uint8")
		u = np.array(upper, dtype="uint8")

		mask_platillo_corto = cv2.inRange(reg_platillo_corto, l, u)
		mask_platillo_largo = cv2.inRange(reg_platillo_largo, l, u)
		mask_tambor1  = cv2.inRange(reg_tambor1,  l, u)
		mask_tambor2 = cv2.inRange(reg_tambor2, l, u)
		mask_redoble = cv2.inRange(reg_redoble, l, u)
		
		out_platillo_corto  = cv2.bitwise_and(reg_platillo_corto, reg_platillo_corto, mask=mask_platillo_corto)
		out_platillo_largo  = cv2.bitwise_and(reg_platillo_largo, reg_platillo_largo, mask=mask_platillo_largo)
		out_tambor1   = cv2.bitwise_and(reg_tambor1,  reg_tambor1,  mask=mask_tambor1)
		out_tambor2  = cv2.bitwise_and(reg_tambor2, reg_tambor2, mask=mask_tambor2)
		out_redoble  = cv2.bitwise_and(reg_redoble, reg_redoble, mask=mask_redoble)

		contours_platillo_corto = findContours(out_platillo_corto)
		contours_platillo_largo = findContours(out_platillo_largo)
		contours_tambor1  = findContours(out_tambor1)
		contours_tambor2 = findContours(out_tambor2)
		contours_redoble  = findContours(out_redoble)
		
		# Lógica para que suenen

		if (contours_platillo_corto > 0) and (hay_platillo_corto == False):
			play_platillo_corto()
			hay_platillo_corto = True
		elif (contours_platillo_corto == 0):
			hay_platillo_corto = False	

		if (contours_platillo_largo > 0) and (hay_platillo_largo == False):
			play_platillo_largo()
			hay_platillo_largo = True
		elif (contours_platillo_largo == 0):
			hay_platillo_largo = False	

		if (contours_tambor1 > 0) and (hay_tambor1 == False):
			play_tambor1()
			hay_tambor1 = True
		elif (contours_tambor1 == 0):
			hay_tambor1 = False

		if (contours_tambor2 > 0) and (hay_tambor2 == False):
			play_tambor2()
			hay_tambor2 = True
		elif (contours_tambor2 == 0):
			hay_tambor2 = False

		if (contours_redoble > 0) and (hay_redoble == False):
			play_redoble()
			hay_redoble = True
		elif (contours_redoble == 0):
			hay_redoble = False


		# Pintamos los rectángulos
		cv2.rectangle(clone, (platillo_corto_left,platillo_corto_top), (platillo_corto_right,platillo_corto_bottom), (21,142,202,0.5), 4)
		cv2.rectangle(clone, (platillo_largo_left,platillo_largo_top), (platillo_largo_right,platillo_largo_bottom), (21,142,202,0.5), 4)
		cv2.rectangle(clone, (tambor1_left,tambor1_top), (tambor1_right,tambor1_bottom), (34,25,151,0.5), 4)
		cv2.rectangle(clone, (tambor2_left,tambor2_top), (tambor2_right,tambor2_bottom), (34,25,151,0.5), 4)
		cv2.rectangle(clone, (redoble_left,redoble_top), (redoble_right,redoble_bottom), (0,0,0,0.5), 4)

		# Escribimos los textos dentro de los rectángulos
		fuente = cv2.FONT_HERSHEY_SIMPLEX

		cv2.putText(clone, text="P1", org=(platillo_corto_left+20,platillo_corto_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(21,142,202,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="P2", org=(platillo_largo_left+20,platillo_largo_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(21,142,202,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="T1", org=(tambor1_left+20,tambor1_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(34,25,151,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="T2", org=(tambor2_left+20,tambor2_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(34,25,151,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="R", org=(redoble_left+30,redoble_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(0,0,0,0.5), thickness=2, lineType=cv2.LINE_8 )


		# Mostramos el video
		cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Video", clone)

		# Salimos si pulsamos escape
		if cv2.waitKey(1) & 0XFF == 27:
			break

	# Liberamos la camara y destruimos las ventanas
	cam.release()
	cv2.destroyAllWindows()
