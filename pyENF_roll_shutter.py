# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2




# Input the video stream
video_filepath = "Test_recording/test2-shrink_factor_2/resized_MVI_0288.avi"

video = cv2.VideoCapture(video_filepath)

if video.isOpened() == False:
    print("Error Opening the video stream or file")
# Collect the row signal from the buffered frames

while video.isOpened():
    ret, frame = video.read()
    if ret == True:

        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

video.release()

cv2.destroyAllWindows()
# estimate the idle period for the given row signal

# Estimate the ENF signal using the idle period and the row signal collected

# Cross-correlation coefficient to compare the estimated signals from the power recordings and
# the estimated video frames.