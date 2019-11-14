
import time
from multiprocessing import Process

import cv2

import utils.image as imgutils

def _display_func(name, size, show_fps, img_retrival_func):

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if size is not None:
        cv2.resizeWindow(name, *size)

    if show_fps:
        t_start = time.time()
        frame_counter = 0

    while True:
        frame = img_retrival_func()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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


class GUIVideoWindow(object):
    def __init__(self, name, img_retrival_func, size=None):
        self.name = name
        self.size = size
        self._img_retrival_func = img_retrival_func

    def start(self, show_fps=False):
        self._display_p = Process(target=_display_func,
                                  args=(self.name, self.size, show_fps, self._img_retrival_func))
        self._display_p.start()

    def terminate(self):
        self._display_p.terminate()
        cv2.destroyWindow(self.name)