# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2



def extract_row_pixel_average(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3: #its an RGB frame
        average_frame_across_rgb = np.mean(frame,axis=2)
        average_frame_across_column = np.mean(average_frame_across_rgb,axis=1)
    else:
        average_frame_across_column = np.mean(frame, axis=1)
    
    average_frame_across_column = np.reshape(average_frame_across_column,(frame_shape[0],1))

    return average_frame_across_column




# Input the video stream
video_filepath = "Test_recording/test2-shrink_factor_2/resized_MVI_0288.avi"
video = cv2.VideoCapture(video_filepath)

#Validating the read of input video
if video.isOpened() == False:
    print("Error Opening the video stream or file")

# Video specifics extraction
total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
height_of_frame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # total number of rows
width_of_frame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # total number of colums
size_of_row_signal = int(np.multiply(total_number_of_frames,height_of_frame))
print(size_of_row_signal)
row_signal = np.zeros((size_of_row_signal,1),dtype=float)

# Collect the row signal from the buffered frames
frame_counter = 0
while video.isOpened():
    ret, frame = video.read()
    if ret is True:
        frame_shape = frame.shape
        start_index = frame_counter*height_of_frame
        row_signal[start_index:start_index+height_of_frame] = extract_row_pixel_average(frame)
        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    frame_counter += 1
    print(frame_counter)

video.release()

cv2.destroyAllWindows()
# estimate the idle period for the given row signal

# Estimate the ENF signal using the idle period and the row signal collected

# Cross-correlation coefficient to compare the estimated signals from the power recordings and
# the estimated video frames.