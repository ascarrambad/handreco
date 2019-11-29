
import os
import argparse

import cv2
from pynput.keyboard import Key, Listener

# Hack to import top-level modules
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

from InferenceManager import InferenceManager
from utils.GUIVideoWindow import GUIVideoWindow
import utils.image as imgutils

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
    args = parser.parse_args()

    # Begin capturing
    cap = cv2.VideoCapture(args.video_source)

    # Init keyboard listener
    listener = Listener(on_press=on_keypress)
    listener.start()

    # Init inference manager (onyl for detection)
    inference_manager = InferenceManager(detection_only=True)

    currentPath = ''
    currentExample = ''
    data_path = './hand_classification/Gestures/'

    print('Do you want to : \n 1 - Add new gesture \
                            \n 2 - Add examples to existing gesture \
                            \n 3 - Add garbage examples')

    menu_choice = input()
    while(menu_choice not in ['1','2','3']):
        print('Please enter 1 or 2 or 3')
        menu_choice = input()

    if menu_choice == '1':
        print('Enter a name for the gesture you want to add :')
        name_gesture = input()

        # Create folder for gesture
        if not os.path.exists(data_path + name_gesture):
            os.makedirs(data_path + name_gesture + '/' + name_gesture + '_1/')
            currentPath = data_path + name_gesture + '/' + name_gesture + '_1/'
            currentExample = name_gesture + '_1_'

    elif menu_choice == '2':
        # Display current Gestures
        dirs = os.listdir(data_path)
        dirs_choice = ''
        possible_choices = []
        i = 1
        for _dir in dirs:
            dirs_choice += str(i) + ' - ' + str(_dir) + ' / '
            possible_choices.append(str(i))
            i+=1

        # Ask user to choose to which gesture to add examples
        print('Choose one of the following gesture:')
        print(dirs_choice)
        choice = input()
        while(choice not in possible_choices and dirs[int(choice)-1]=='garbage'):
            print('Please enter one of the following (not garbage): ' + str(possible_choices))
            choice = input()

        # Count number of files to increment new example directory
        subdirs = os.listdir(data_path + dirs[int(choice)-1] + '/')
        index = len(subdirs) + 1

        # Create new example directory
        if not os.path.exists(data_path + dirs[int(choice)-1] + '/' + dirs[int(choice)-1] + '_' + str(index) + '/'):
            os.makedirs(data_path + dirs[int(choice)-1] + '/' + dirs[int(choice)-1] + '_' + str(index) + '/')

            #Update current path
            currentPath = data_path + dirs[int(choice)-1] + '/' + dirs[int(choice)-1] + '_' + str(index) + '/'
            currentExample = dirs[int(choice)-1] + '_' + str(index) + '_'
            name_gesture = dirs[int(choice) - 1]

    elif menu_choice == '3':
        # Create folder for gesture
        if not os.path.exists(data_path + 'Garbage/'):
            os.makedirs(data_path + 'Garbage/Garbage_1/')
            currentPath = data_path + 'Garbage/Garbage_1/'
            currentExample = 'Garbage_1_'
            name_gesture = 'Garbage'
        else:
            # Count number of files to increment new example directory
            subdirs = os.listdir(data_path + 'Garbage/')
            index = len(subdirs) + 1
            # Create new example directory
            if not os.path.exists(data_path + 'Garbage/Garbage_' + str(index) + '/'):
                os.makedirs(data_path + 'Garbage/Garbage_' + str(index) + '/')

                #Update current path
                currentPath = data_path + 'Garbage/Garbage_' + str(index) + '/'
                currentExample ='Garbage_' + str(index) + '_'
                name_gesture = 'Garbage'

    gui = GUIVideoWindow('Videosource', lambda: cap.read()[1])
    gui.start(isRGB=False)

    print('You\'ll now be prompted to record the gesture you want to add. \n \
                Please place your hand beforehand facing the camera, and press the escape key when ready. \n \
                When finished press the escape key again.')

    # Wait for user keypress
    global _terminate_proc
    while not _terminate_proc:
        pass

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(currentPath + name_gesture + '.avi', fourcc, 25.0, (640, 480))

    # Main loop
    _terminate_proc = False
    while not _terminate_proc:
        _, frame = cap.read()
        if frame is not None:
            out.write(frame)

    # Release everything if job is finished
    gui.terminate()
    cap.release()
    out.release()

    vid = cv2.VideoCapture(currentPath + name_gesture + '.avi')

    # Check if the video
    if (not vid.isOpened()):
        print('Error opening video stream or file')
        return

    _iter = 1
    # Read until video is completed
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if ret:
            print('   Processing frame: ' + str(_iter))
            # Resize and convert to RGB for NN to work with
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect object
            boxes, scores = inference_manager.detect(frame)

            # get region of interest
            res = imgutils.get_box_image(1, 0.2, scores, boxes, 320, 180, frame)

            # Save cropped image
            if(res is not None):
                cv2.imwrite(currentPath + currentExample + str(_iter) + '.png', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

            _iter += 1
        # Break the loop
        else:
            break

    print('   Processed ' + str(_iter) + ' frames!')

    vid.release()


if __name__ == '__main__':
    main()
