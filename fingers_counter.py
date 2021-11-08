import numpy as np
import cv2
from sklearn.metrics import pairwise

# backgroung global variable
bg = None

#-------------------------------------------------------------
# Media weight calculation & accumulation. Background update
#-------------------------------------------------------------
def run_avg(image, media_weight):
    global bg

    # Initialize background first time
    if bg is None:
        bg = image.copy().astype("float")
        return

    # if background exists, update media weight background
    cv2.accumulateWeighted(image, bg, media_weight)


# ----------------------------
# Hand region image distance 
# ----------------------------
def segment(image, threshold=25):
    global bg

    # Absolute differente between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # Get threshold through difference before
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Get contours in the image dawn
    # findContours gets a binary image
    # RETR_EXTERNAL = contours getting mode
    # CHAIN_APPROX_SIMPLE = contours closeness mode
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours do nothing
    if len(contours) == 0:
        return
    else:
        # Get maximum hand contour and return contour and threshold
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)



#--------------------------------------------------------------
# Count fingers number in segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    
    # find convex covering of segmented region
    env_convex = cv2.convexHull(segmented)

    # find end covering points
    extreme_top    = tuple(env_convex[env_convex[:, :, 1].argmin()][0])
    extreme_bottom = tuple(env_convex[env_convex[:, :, 1].argmax()][0])
    extreme_left   = tuple(env_convex[env_convex[:, :, 0].argmin()][0])
    extreme_right  = tuple(env_convex[env_convex[:, :, 0].argmax()][0])

    # get hand palm center
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # calculate maximum euclidean distance between hand palm center 
    # and the end covering points 
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # generate circle ratio at 80% maximum euclidean distance
    radius = int(0.8 * maximum_distance)

    # circle circumference
    circumference = (2 * np.pi * radius)

    # draw circle image for hand palm and fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # bit-wise AND for hand threshold using circle image like mask, that 
    # generate hands cutting
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # find resultant circle image contours
    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # finger counter
    finger_counter = 0

    # loop through the contours found
    for c in contours:
        # get contour values
        (x, y, w, h) = cv2.boundingRect(c)

        # Increment finger contour only if:
        # 1. contour region is not the wrist
        # 2. Points contour number does is not bigger than 
        #    25% of the circle image 
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            finger_counter += 1

    return finger_counter


#-------------------
# MAIN
#-------------------
if __name__ == "__main__":

    # Initialize media weight
    media_weight = 0.5

    # Reference to cam
    camera = cv2.VideoCapture(0)

    # Capture coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # frames number
    num_frames = 0

    while True:
        # get current frame
        res, frame = camera.read()

        # rotate the frame to avoid mirron effect
        frame = cv2.flip(frame, 1) #0 vertical, 1 horizontal

        # get height and width frame
        # height, width = frame.shape[:2]

        # get frame rectangle copy, convert it to grayscale 
        # and blur it (gauss soft focus) to split the background
        # more easy
        sub_frame = frame[top:bottom, right:left]
        sub_frame = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
        sub_frame = cv2.GaussianBlur(sub_frame, (7, 7), 0)

        # To split background, accumulate frames until dawn (30)
        # to get media weight and to be calibrated when dawn is exceeded
        # (important: not camera movement during this step)
        if num_frames < 30:
            run_avg(sub_frame, media_weight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # separate hand region from complete frame in a tuple
            hand = segment(sub_frame)
            if hand is not None:
                # separate tuple in two variables
                thresholded, segmented = hand

                # draw contours in complete frame and 
                # show the threshold in other window
                cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))

                # count fingers number and send it to screen
                num_finger = count(thresholded, segmented)

                cv2.putText(frame, str(num_finger), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
                cv2.imshow("Thresholded", thresholded)

        # draw rectangle in green in curren frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        # increase frames number
        num_frames += 1

        # Show complete frame in rectangle
        cv2.imshow("Video", frame)

        # Press 'q' to and loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# free memory and resources
camera.release()
cv2.destroyAllWindows()

