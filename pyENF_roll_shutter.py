# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2
import pickle
import pyenf
from scipy import signal,io
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt

# Constants
open_video_to_extract_Row_signal = False  # set it to 1 to extract, else 0 to use the dump file
video_filepath = "Recordings/resized_MVI_0292.avi"


def extract_row_pixel(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3:  # its an RGB frame
        average_frame_across_rgb = np.mean(frame, axis=2)
        average_frame_across_column = np.mean(average_frame_across_rgb, axis=1)
    else:
        average_frame_across_column = np.mean(frame, axis=1)

    average_frame_across_column = np.reshape(average_frame_across_column, (frame_shape[0],))

    return average_frame_across_column


# Input the video stream
video = cv2.VideoCapture(video_filepath)

# Validating the read of input video
if not video.isOpened():
    print("Error Opening the video stream or file")

# Video specifics extraction
total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
height_of_frame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # total number of rows
width_of_frame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # total number of columns
frame_rate = float(video.get(cv2.CAP_PROP_FPS))
size_of_row_signal = int(np.multiply(total_number_of_frames, height_of_frame))
# print(size_of_row_signal)


# row_signal = np.zeros((size_of_row_signal, 1), dtype=float)
row_signal = np.zeros((total_number_of_frames, height_of_frame, 1), dtype=float)
# Collect the row signal from the buffered frames
if open_video_to_extract_Row_signal is True:
    frame_counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            frame_shape = frame.shape
            start_index = frame_counter * height_of_frame
            # row_signal[start_index:start_index + height_of_frame] = extract_row_pixel(frame)
            row_signal[frame_counter, :, 0] = extract_row_pixel(frame)
            # cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        frame_counter += 1
        # print(frame_counter)
    video.release()
    cv2.destroyAllWindows()
    # store the variables for faster future use
    store_variable_file = open('row_signal.pkl', 'wb')
    pickle.dump(row_signal, store_variable_file)
    store_variable_file.close()
    print("Extracted Row Signal and stored in dump.\n")
else:
    load_variable_file = open('row_signal.pkl', 'rb')
    row_signal = pickle.load(load_variable_file)
    load_variable_file.close()
    print("Loaded the Row Signal. \n")

time = np.arange(0.0,size_of_row_signal)

# For a static video, clean the row signal with its video signal
# that should leave only the ENF signal
# Refer to { Exploiting Rolling Shutter For ENF Signal Extraction From Video }
# row_signal = video_signal + enf_signal
# average_of_each_row_element(row_signal) = average_of_each_row_element(video_signal) [since average of enf is 0]
# enf_signal = row_signal - average_of_each_row_element(row_signal)

average_of_each_row_element = np.mean(row_signal, axis=0)
enf_video_signal = row_signal - average_of_each_row_element
flattened_enf_signal = row_signal.flatten()
# plotting the spectrogram of the signal

f,t,Sxx = signal.spectrogram(flattened_enf_signal,fs=frame_rate*height_of_frame, nperseg=None, mode='psd')

plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))

#plt.specgram(flattened_enf_signal,Fs=frame_rate)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

fig, (ax1,ax2) = plt.subplots(nrows=2)
ax2.use_sticky_edges = False
ax1.plot(time,flattened_enf_signal)
Pxx, freq, bins, im = ax2.specgram(flattened_enf_signal,NFFT=8192,Fs=int(frame_rate*height_of_frame), noverlap=0)

plt.show()
io.wavfile.write("video_plain_original.wav",rate=int(frame_rate*height_of_frame),data=flattened_enf_signal)
# estimate the idle period for the given row signal

# Estimate the ENF signal using the idle period and the row signal collected

# Cross-correlation coefficient to compare the estimated signals from the power recordings and
# the estimated video frames.
