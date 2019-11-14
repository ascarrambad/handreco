
import datetime
import argparse

import cv2
import numpy as np
from pynput.keyboard import Key, Listener

from utils.WebcamVideostream import WebcamVideostream
from utils.SpeechRecognizer import SpeechRecognizer
from utils.GUIVideoWindow import GUIVideoWindow
from utils.DecisionMaker import DecisionMaker

from InferenceManager import InferenceManager
from WorkersManager import WorkersManager
from ServicesManager import ServicesManager

import utils.gui as guiutils

# Main loop stop var
_terminate_proc = False

# Key Listener
def on_keypress(key):
    global _terminate_proc
    if key == Key.esc:
        _terminate_proc = True

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--show-fps',
        dest='show_fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument(
        '-srv',
        '--services',
        dest='services',
        type=int,
        default=0,
        help='Enable IoT services.')
    args = parser.parse_args()
    enable_services = bool(args.services)
    enable_gui = bool(args.display)
    show_fps = bool(args.show_fps)

    # Initialize videostream
    video_capture = WebcamVideostream(src=args.video_source, width=args.width, height=args.height)
    video_capture.start()

    # Init keyboard listener
    listener = Listener(on_press=on_keypress)
    listener.start()

    # Init speech recognizer
    # speech_rec = SpeechRecognizer()
    # speech_rec.start()

    # Init detection parameters
    cap_params = {}
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['num_hands_detect'] = args.num_hands

    # Init Workers
    workers = WorkersManager(args.num_workers,
                             args.queue_size,
                             True,
                             InferenceManager)
    workers.start(cap_params)

    # Init services parameters
    if enable_services:
        decision_mk = DecisionMaker()
        services = ServicesManager()

    # Init GUI windows
    if enable_gui:
        main_window = GUIVideoWindow('Webcam feed', workers.output_q.get, size=(450,300))
        main_window.start(show_fps=show_fps)
        cropped_out_window = GUIVideoWindow('HandFrame', workers.cropped_output_q.get)
        cropped_out_window.start()

    # Start main loop
    global _terminate_proc
    while not _terminate_proc:
        # Read frame and add it to input queue
        _, frame = video_capture.read(flip=True, to_RGB=True)
        workers.input_q.put(frame)

        # Gesture recognition works only after magic word
        if True:
            try:
                inferences = workers.inferences_q.get_nowait()
            except Exception as e:
                inferences = None
                pass

            if (inferences is not None):
                # Display inferences
                guiutils.draw_inferences(inferences)

                # Service call
                if enable_services:
                    decision = decision_mk.inferences_to_action(inferences)
                    if decision is not None: # Check if decision is available
                        action, payload = decision
                        services.call(action, 'window_left', payload)

    listener.stop()
    workers.terminate()
    # speech_rec.terminate()
    video_capture.stop()

    main_window.terminate()
    cropped_out_window.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
