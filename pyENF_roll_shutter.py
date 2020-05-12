# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2
import pickle
import pyenf
from scipy import signal, io
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa

# Constants
open_video_to_extract_Row_signal = False  # set it to True to extract, else False to use the dump file
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
    store_variable_file = open('Recordings/row_signal.pkl', 'wb')
    pickle.dump(row_signal, store_variable_file)
    store_variable_file.close()
    print("Extracted Row Signal and stored in dump.\n")
else:
    load_variable_file = open('Recordings/row_signal.pkl', 'rb')
    row_signal = pickle.load(load_variable_file)
    load_variable_file.close()
    print("Loaded the Row Signal. \n")

time = np.arange(0.0, size_of_row_signal)

# For a static video, clean the row signal with its video signal
# that should leave only the ENF signal
# Refer to { Exploiting Rolling Shutter For ENF Signal Extraction From Video }
# row_signal = video_signal + enf_signal
# average_of_each_row_element(row_signal) = average_of_each_row_element(video_signal) [since average of enf is 0]
# enf_signal = row_signal - average_of_each_row_element(row_signal)

# Estimate the ENF signal using the row signal collected
average_of_each_row_element = np.mean(row_signal, axis=0)
enf_video_signal = row_signal - average_of_each_row_element
flattened_enf_signal = enf_video_signal.flatten() # the matrix shape ENF data is flattened to one dim data

fs = 1000 # downsampling frequency
filename = "mediator.wav"
# Writing the ENF data to the wav file for data type conversion
io.wavfile.write(filename, rate=int(frame_rate * height_of_frame), data=flattened_enf_signal)

signal0, fs = librosa.load(filename, sr=fs)

video_signal_object = pyenf.pyENF(signal0=signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0)
spectro_strip, frequency_support = video_signal_object.compute_spectrogam_strips()
weights = video_signal_object.compute_combining_weights_from_harmonics()
OurStripCell, initial_frequency = video_signal_object.compute_combined_spectrum(spectro_strip, weights, frequency_support)
ENF = video_signal_object.compute_ENF_from_combined_strip(OurStripCell, initial_frequency)
fig, (video,power,corr) = plt.subplots(3,1,sharex=True)
video.plot(ENF[:-7])
video.set_title("ENF Signal")
#video.ylabel("Frequency (Hz)")
#video.xlabel("Time (sec)")
#plt.show()

power_signal_filename = 'Recordings/80D_power_recording_7.wav'
power_signal0, fs = librosa.load(power_signal_filename, sr=fs)
power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0)
power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
power_weights = power_signal_object.compute_combining_weights_from_harmonics()
power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip, power_weights, power_frequency_support)
power_ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)

power.plot(power_ENF[:-7])
power.set_title("Power ENF Signal")
#power.ylabel("Frequency (Hz)")
#power.xlabel("Time (sec)")
#plt.show()

print("Correlating the signal")
enf_corr = signal.correlate(ENF,power_ENF,mode='same')

corr.plot(enf_corr)
corr.axhline(0.5,ls=':')
fig.tight_layout()
fig.show()
# estimate the idle period for the given row signal



# Cross-correlation coefficient to compare the estimated signals from the power recordings and
# the estimated video frames.
