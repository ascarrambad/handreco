
import time
from threading import Thread

import cv2

import utils.image as imgutils

_stopped = False

def _display_func(img_retrival_func, name, size, show_fps, flip, isRGB):

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if size is not None:
        cv2.resizeWindow(name, *size)

    if show_fps:
        t_start = time.time()
        frame_counter = 0

    global _stopped
    while not _stopped:
        frame = img_retrival_func()

        if frame is not None:
            if isRGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if flip:
                frame = cv2.flip(frame, 1)

            if show_fps:
                frame_counter += 1

                t_now = time.time()
                seconds = t_now - t_start
                fps = int(frame_counter / seconds)
                imgutils.draw_fps_on_image(f'FPS: {fps}', frame)

                # Reset fps sample after 400 frames
                if frame_counter >= 400:
                    t_start = time.time()
                    frame_counter = 0

            cv2.imshow(name, frame)
            cv2.waitKey(1)

    cv2.destroyWindow(name)

class GUIVideoWindow(object):
    def __init__(self, name, img_retrival_func, size=None):
        super(GUIVideoWindow, self).__init__()


        self.name = name
        self.size = size
        self._img_retrival_func = img_retrival_func
        global _stopped
        _stopped = False
        self._display_t = None

    def start(self, show_fps=False, flip=False, isRGB=True):
        if self._display_t is None:
            global _stopped
            _stopped = False
            self._display_t = Thread(target=_display_func,
                                     args=(self._img_retrival_func, self.name, self.size, show_fps, flip, isRGB))
            self._display_t.start()

    def terminate(self):
        if self._display_t is not None:
            global _stopped
            _stopped = True
            self._display_t = None


