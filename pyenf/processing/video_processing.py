# Reading the input video stream
# extracting the row signal
import numpy as np
import cv2

class Videoprocessing:
    def __init__(self,video_file_name):
        self.video_file_name = video_file_name # video object which could be read from a file or live stream
        self.video = cv2.VideoCapture(self.video_file_name)
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = np.divide(self.total_frames,self.framerate)

    #def rgb_to_row_signal(self)