
from threading import Thread

import cv2

# Code to thread reading camera input.
# Adapted from: Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideostream:
    def __init__(self, src=0, width=None, height=None):
        super(WebcamVideostream, self).__init__()

        # Init videocamera stream
        self.stream = cv2.VideoCapture(src)

        if width is not None and height is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.grabbed = None
        self.frame = None

        # initialize the variable used to indicate if the thread should
        # be stopped
        self._stopped = False
        self._webcam_t = None

    def start(self):
        if self._webcam_t is None:
            self._stopped = False
            self._webcam_t = Thread(target=self.update, args=())
            self._webcam_t.start()

    def stop(self):
        if self._webcam_t is not None:
            self._stopped = True
            self._webcam_t = None

    def update(self):
        while not self._stopped:
            # Read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

        self.stream.release()

    def read(self, flip=False, to_RGB=False):

        frame = self.frame
        if flip:
            frame = cv2.flip(frame, 1)
        if to_RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # return the frame most recently read
        return (self.grabbed, frame)

    def size(self):
        return self.stream.get(3), self.stream.get(4)
