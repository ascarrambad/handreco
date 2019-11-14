
import os
import argparse

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

import dataset

def plot_metrics(hist):
    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()

def train(img_dims, model_name, epochs, batch_size, learning_rate):

    model_name = f'./hand_classification/models/{model_name}_{epochs}.h5'

    # Load Data
    x_train, y_train, x_test, y_test = dataset.load_data(poses=["all"])
    num_classes = np.unique(y_test).size

    # Reshapes
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_dims, img_dims)
        x_test = x_test.reshape(x_test.shape[0], 1, img_dims, img_dims)
        input_shape = (1, img_dims, img_dims)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_dims, img_dims, 1)
        x_test = x_test.reshape(x_test.shape[0], img_dims, img_dims, 1)
        input_shape = (img_dims, img_dims, 1)

    # Class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Model structure
    bnorm_axis = 1 if K.image_data_format() == 'channels_first' else -1
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=bnorm_axis, momentum=0.99, epsilon=0.001, center=True, scale=False))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=bnorm_axis, momentum=0.99, epsilon=0.001, center=True, scale=False))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    # Model training
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=(x_test, y_test))
    # Model Evaluation
    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(model_name)

    # Plot metrics
    plot_metrics(hist)

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    epochs = 5
    parser.add_argument(
        '-lrate',
        '--learning_rate',
        dest='lrate',
        type=float,
        default=0.001,
        help='Learning rate used for model training.')
    parser.add_argument(
        '-bsize',
        '--batch_size',
        dest='bsize',
        type=int,
        default=128,
        help='Batch size used for model training.')
    parser.add_argument(
        '-epochs',
        '--num-epochs',
        dest='epochs',
        type=int,
        default=10,
        help='Number of epochs used for model training.')
    parser.add_argument(
        '-mname',
        '--model-name',
        dest='model_name',
        type=str,
        default='model',
        help='Name of the saved model file.')
    parser.add_argument(
        '-imdim',
        '--image-dimensions',
        dest='imdims',
        type=int,
        default=28,
        help='Dimensions of the input image.')
    args = parser.parse_args()

    train(args.imdims, args.model_name, args.epochs, args.bsize, args.lrate)
