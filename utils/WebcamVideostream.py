
from threading import Thread

import cv2

# Code to thread reading camera input.
# Adapted from: Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideostream:
    def __init__(self, src=0, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        if height is not None and width is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self, flip=False, to_RGB=False):

        if flip:
            frame = cv2.flip(self.frame, 1)
        if to_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # return the frame most recently read
        return (self.grabbed, frame)

    def size(self):
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        self.stopped = True