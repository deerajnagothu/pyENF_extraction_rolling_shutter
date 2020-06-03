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
from skimage.util import img_as_float
from skimage.segmentation import slic

# Constants
open_video_to_extract_Row_signal = True  # set it to True to extract, else False to use the dump file
# video_folder = "Recordings/Sample1/"
# video_rec_name = "resized_MVI_0292.avi"
video_folder = "Recordings/Corridor/"
video_rec_name = "resized_corridor_surveillance.mp4"

power_rec_name = "80D_power_recording_7.wav"
numSegments = 500  # number of superpixel segments per frame
# video_rec_name = "resized_MVI_0288.avi"
# power_rec_name = "80D_power_recording_3_20min.wav"
video_filepath = video_folder + video_rec_name
power_filepath = video_folder + power_rec_name
look_for_motion_after = 1  # periodic number of frames after which the mask is calculated and used for superpixel
motion_detection_threshold = 40  # threshold decides after how many pixel changes, it should apply Superpixel mask


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

# Generating superpixel segment from first frame of the video
ret, frame = video.read()
frame = img_as_float(frame)
segments = slic(frame, n_segments=numSegments, sigma=5, start_label=1)  # Initializing superpixels segments using SLIC algorithm
motion_mask = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=150, detectShadows=True)
# ones superpixel is created for final AND operation with superpixel mask. The superpixels effected is set to zero,
# so same pixels are also set to zero in ones superpixel
master_ones_Superpixel_mask = np.ones([height_of_frame,width_of_frame],dtype=int)
if open_video_to_extract_Row_signal is True:
    frame_counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            frame_shape = frame.shape
            start_index = frame_counter * height_of_frame
            # row_signal[start_index:start_index + height_of_frame] = extract_row_pixel(frame)
            if frame_counter % look_for_motion_after == 0:  # add delays between the computation of necessary
                frame_segments = segments.copy()  # creating a copy of SLIC segments for each frame
                new_frame_with_mask = motion_mask.apply(frame)  # applying the background subtractor to frame
                motion_threshold = np.count_nonzero(new_frame_with_mask)  # count how many pixels were effected
                #print(motion_threshold)
                if motion_threshold >= motion_detection_threshold:  # no. of pixels effected more than threshold then apply mask
                    new_frame_with_mask[new_frame_with_mask == 255] = 1  # 255 represents white pixels which are motion detections
                    new_frame_with_mask[new_frame_with_mask == 127] = 1  # 127 represents gray pixels which are shadow of object
                    superpixel_motion_mask = np.multiply(frame_segments, new_frame_with_mask)  # multiplying to see which superpixels were effected
                    effected_superpixels = np.unique(superpixel_motion_mask)
                    for each_superpixel in effected_superpixels: # all the effected superpixels are set to zero
                        frame_segments[frame_segments == each_superpixel] = 0
                    ones_Superpixel_mask = master_ones_Superpixel_mask.copy()
                    ones_Superpixel_mask[frame_segments == 0] = 0
                    if frame.shape[2] == 3: # its RGB frame, so the mask is applied to each layer individually
                        frame[:, :, 0] = np.multiply(ones_Superpixel_mask, frame[:, :, 0])
                        frame[:, :, 1] = np.multiply(ones_Superpixel_mask, frame[:, :, 1])
                        frame[:, :, 2] = np.multiply(ones_Superpixel_mask, frame[:, :, 2])
                    else: # for grayscale
                        frame = np.multiply(ones_Superpixel_mask, frame)
            row_signal[frame_counter, :, 0] = extract_row_pixel(frame)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        frame_counter += 1
        # print(frame_counter)
    video.release()
    cv2.destroyAllWindows()
    # store the variables for faster future use
    variable_location = video_folder + "row_signal.pkl"
    store_variable_file = open(variable_location, 'wb')
    pickle.dump(row_signal, store_variable_file)
    store_variable_file.close()
    print("Extracted Row Signal and stored in dump.\n")
else:
    variable_location = video_folder + "row_signal.pkl"
    load_variable_file = open(variable_location, 'rb')
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
flattened_enf_signal = enf_video_signal.flatten()  # the matrix shape ENF data is flattened to one dim data

fs = 500  # downsampling frequency
nfft = 8192
frame_size = 1  # change it to 6 for videos with large length recording
overlap = 0
filename = "mediator.wav"
# Writing the ENF data to the wav file for data type conversion
io.wavfile.write(filename, rate=int(frame_rate * height_of_frame), data=flattened_enf_signal)

signal0, fs = librosa.load(filename, sr=fs)

video_signal_object = pyenf.pyENF(signal0=signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
spectro_strip, frequency_support = video_signal_object.compute_spectrogam_strips()
weights = video_signal_object.compute_combining_weights_from_harmonics()
OurStripCell, initial_frequency = video_signal_object.compute_combined_spectrum(spectro_strip, weights,
                                                                                frequency_support)
ENF = video_signal_object.compute_ENF_from_combined_strip(OurStripCell, initial_frequency)
fig, (video, power, corr) = plt.subplots(3, 1, sharex=True)
video.plot(ENF[:-7])
video.set_title("ENF Signal 1")

power_signal_filename = power_filepath
power_signal0, fs = librosa.load(power_signal_filename, sr=fs)
power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
power_weights = power_signal_object.compute_combining_weights_from_harmonics()
power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                            power_weights,
                                                                                            power_frequency_support)
power_ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)

power.plot(power_ENF[:-7])
power.set_title("Power ENF Signal")
# power.ylabel("Frequency (Hz)")
# power.xlabel("Time (sec)")
# plt.show()

print("Correlating the signal")
enf_corr = signal.correlate(ENF, power_ENF, mode='same')

corr.plot(enf_corr)
corr.axhline(0.5, ls=':')
fig.tight_layout()
fig.show()

# A. Superpixel segments generated once using the slic algorithm from sklearn
# B. Motion detection algorithm to see any changes in the pixels
#   1. probably see which pixels changed and set that to 1, and rest all to zero.
#   2. Now layer this with superpixel segment, and detect which superpixels were affected, expect an array of SP numbers
#   3. Set those to 0 in the segment, AND with the image. Now extract the row average by avoiding the neglected pixel
# C. since the camera is stationary, the superpixels need not be generated for every frame.
# D. A better motion detection algo would be good, something other than gaussian. Refer to rasp pi pyimagesearch
