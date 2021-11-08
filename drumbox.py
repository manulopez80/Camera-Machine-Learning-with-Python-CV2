import cv2
import numpy as np
from pygame import mixer

# detect skin colour
lower = [17, 15, 100] # pink
upper = [80, 76, 220] # dark

# Mixer initialization
mixer.init()

# region coordinates
short_plate_top, short_plate_bottom, short_plate_left, short_plate_right  = 50, 150, 120, 220
long_plate_top, long_plate_bottom, long_plate_left, long_plate_right = 50, 150, 320, 420
drum1_top, drum1_bottom, drum1_left, drum1_right = 50, 150, 580, 680
drum2_top, drum2_bottom, drum2_left, drum2_right = 50, 150, 780, 880
drumroll_top, drumroll_bottom, drumroll_left, drumroll_right = 50, 150, 1030, 1130

#----------------------
# Reproduce sounds
#----------------------
def play_short_plate():
	mixer.music.load('short_plate.mp3')
	mixer.music.play()
	print("Detectado P1")

def play_long_plate():
	mixer.music.load('long_plate.mp3')
	mixer.music.play()
	print("Detectado P2")

def play_drum1():
	mixer.music.load('drum1.mp3')
	mixer.music.play()
	print("Detectado T1")

def play_drum2():
	mixer.music.load('drum2.mp3')
	mixer.music.play()
	print("Detectado T2")

def play_drumroll():
	mixer.music.load('drumroll.mp3')
	mixer.music.play()
	print("Detectado R")

#--------------------
# Find the contours 
#--------------------
def findContours(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresholded = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
	contours, _  = cv2.findContours(thresholded.copy(),
									cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)
	return len(contours)

# Reproduce sound or not
there_is_drum2 = False
there_is_drum1  = False
there_is_short_plate = False
there_is_long_plate = False
there_is_drumroll = False

#----------------------
# MAIN
#----------------------
if __name__ == "__main__":

	# camera reference
	cam = cv2.VideoCapture(0)

	# update size and fps
	cam.set(3, 1280)
	cam.set(4, 720)
	cam.set(cv2.CAP_PROP_FPS, 60)

	while True:

		# read one camera frame
		status, frame = cam.read()

		# create a copy and avoid mirror effect 
		clone = frame.copy()
		clone = cv2.flip(clone, 1)
		clone = cv2.resize(clone, (1280,720))

		# get sound zones
		reg_short_plate = clone[short_plate_top:short_plate_bottom, short_plate_left:short_plate_right]
		reg_long_plate = clone[long_plate_top:long_plate_bottom, long_plate_left:long_plate_right]
		reg_drum1  = clone[drum1_top:drum1_bottom, drum1_left:drum1_right]
		reg_drum2 = clone[drum2_top:drum2_bottom, drum2_left:drum2_right]
		reg_drumroll = clone[drumroll_top:drumroll_bottom, drumroll_left:drumroll_right]

		# soften zones
		reg_short_plate = cv2.GaussianBlur(reg_short_plate, (7, 7), 0)
		reg_long_plate = cv2.GaussianBlur(reg_long_plate, (7, 7), 0)
		reg_drum1  = cv2.GaussianBlur(reg_drum1,  (7, 7), 0)
		reg_drum2 = cv2.GaussianBlur(reg_drum2, (7, 7), 0)
		reg_drumroll = cv2.GaussianBlur(reg_drumroll, (7, 7), 0)

		l = np.array(lower, dtype="uint8")
		u = np.array(upper, dtype="uint8")

		mask_short_plate = cv2.inRange(reg_short_plate, l, u)
		mask_long_plate = cv2.inRange(reg_long_plate, l, u)
		mask_drum1  = cv2.inRange(reg_drum1,  l, u)
		mask_drum2 = cv2.inRange(reg_drum2, l, u)
		mask_drumroll = cv2.inRange(reg_drumroll, l, u)
		
		out_short_plate  = cv2.bitwise_and(reg_short_plate, reg_short_plate, mask=mask_short_plate)
		out_long_plate  = cv2.bitwise_and(reg_long_plate, reg_long_plate, mask=mask_long_plate)
		out_drum1   = cv2.bitwise_and(reg_drum1,  reg_drum1,  mask=mask_drum1)
		out_drum2  = cv2.bitwise_and(reg_drum2, reg_drum2, mask=mask_drum2)
		out_drumroll  = cv2.bitwise_and(reg_drumroll, reg_drumroll, mask=mask_drumroll)

		contours_short_plate = findContours(out_short_plate)
		contours_long_plate = findContours(out_long_plate)
		contours_drum1  = findContours(out_drum1)
		contours_drum2 = findContours(out_drum2)
		contours_drumroll  = findContours(out_drumroll)
		
		# Logic to play

		if (contours_short_plate > 0) and (there_is_short_plate == False):
			play_short_plate()
			there_is_short_plate = True
		elif (contours_short_plate == 0):
			there_is_short_plate = False	

		if (contours_long_plate > 0) and (there_is_long_plate == False):
			play_long_plate()
			there_is_long_plate = True
		elif (contours_long_plate == 0):
			there_is_long_plate = False	

		if (contours_drum1 > 0) and (there_is_drum1 == False):
			play_drum1()
			there_is_drum1 = True
		elif (contours_drum1 == 0):
			there_is_drum1 = False

		if (contours_drum2 > 0) and (there_is_drum2 == False):
			play_drum2()
			there_is_drum2 = True
		elif (contours_drum2 == 0):
			there_is_drum2 = False

		if (contours_drumroll > 0) and (there_is_drumroll == False):
			play_drumroll()
			there_is_drumroll = True
		elif (contours_drumroll == 0):
			there_is_drumroll = False


		# Paint rectangles
		cv2.rectangle(clone, (short_plate_left,short_plate_top), (short_plate_right,short_plate_bottom), (21,142,202,0.5), 4)
		cv2.rectangle(clone, (long_plate_left,long_plate_top), (long_plate_right,long_plate_bottom), (21,142,202,0.5), 4)
		cv2.rectangle(clone, (drum1_left,drum1_top), (drum1_right,drum1_bottom), (34,25,151,0.5), 4)
		cv2.rectangle(clone, (drum2_left,drum2_top), (drum2_right,drum2_bottom), (34,25,151,0.5), 4)
		cv2.rectangle(clone, (drumroll_left,drumroll_top), (drumroll_right,drumroll_bottom), (0,0,0,0.5), 4)

		# Text inside rectangles
		fuente = cv2.FONT_HERSHEY_SIMPLEX

		cv2.putText(clone, text="P1", org=(short_plate_left+20,short_plate_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(21,142,202,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="P2", org=(long_plate_left+20,long_plate_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(21,142,202,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="T1", org=(drum1_left+20,drum1_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(34,25,151,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="T2", org=(drum2_left+20,drum2_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(34,25,151,0.5), thickness=2, lineType=cv2.LINE_8 )

		cv2.putText(clone, text="R", org=(drumroll_left+30,drumroll_bottom-35), 
			  fontFace = fuente, fontScale = 1.5, color=(0,0,0,0.5), thickness=2, lineType=cv2.LINE_8 )


		# Show video
		cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Video", clone)

		# Exit pressing Escape
		if cv2.waitKey(1) & 0XFF == 27:
			break

	# Free camera and destroy windows
	cam.release()
	cv2.destroyAllWindows()
