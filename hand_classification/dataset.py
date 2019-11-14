
import os

import cv2
import numpy as np
from sklearn.utils import shuffle

def _normalize(data_path, imsize):

    gestures = os.listdir(data_path)

    for gesture in gestures:
        subdirs = os.listdir(data_path + gesture + '/')
        for subdir in subdirs:
            files = os.listdir(data_path + gesture + '/' + subdir + '/')
            for file in files:
                if(file.endswith(".png")):
                    path = data_path + gesture + '/' + subdir + '/' + file
                    # Read image
                    im = cv2.imread(path)

                    height, width, channels = im.shape
                    if not height == width == imsize:
                        # Resize image
                        im = cv2.resize(im, (imsize, imsize), interpolation=cv2.INTER_AREA)
                        # Write image
                        cv2.imwrite(path, im)

def _read_data(data_path, req_gestures):

    x = []
    y = []

    count_classes = 0
    gestures = os.listdir(data_path)

    if 'all' in req_gestures:
        req_gestures = gestures.copy()

    for gesture in gestures:
        if gesture in req_gestures:
            print('>> Working on gesture : ' + gesture)
            subdirs = os.listdir(data_path + gesture + '/')

            for subdir in subdirs:
                files = os.listdir(data_path + gesture + '/' + subdir + '/')
                print('>> Working on examples : ' + subdir)

                for file in files:
                    if file.endswith('.png'):
                        path = data_path + gesture + '/' + subdir + '/' + file
                        # Read image
                        im = cv2.imread(path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        im = im.astype(dtype='float64')
                        im = im[...,None] # Add dimension at last position

                        x.append(im)
                        y.append(count_classes)

            count_classes += 1

    x = np.array(x)
    y = np.array(y)

    x = x / 255.

    return x, y

def _split_data(x, y, split=0.85):
    maxIndex = int(split * x.shape[0])
    x_train = x[:maxIndex]
    x_test = x[maxIndex:]
    y_train = y[:maxIndex]
    y_test = y[maxIndex:]
    return x_train, y_train, x_test, y_test

def load_data(imsize=28, gestures=['all']):
    data_path = './hand_classification/Gestures/'

    _normalize(data_path, imsize)

    x, y = _read_datadata_path, (gestures)
    x, y = shuffle(x, y, random_state=0)

    return _split_data(x, y)