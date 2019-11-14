import os
import click
import random
import numpy as np
import soundfile as sf
import tensorflow as tf
from keras.layers import Input, Reshape, Conv2D, MaxPool2D, LSTM, Dropout, UpSampling2D, Activation


# ----------------------------------------------------------------------------------------------------------------------

def _create_rnn():
    model = tf.keras.Sequential()
    model.add(Input(shape=(210, 210,)))
    model.add(Conv2D(512, 2, activation="relu"))
    model.add(Conv2D(512, 2, activation="relu"))
    model.add(MaxPool2D(2))

    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(UpSampling2D(2))
    model.add(Conv2D(256, 2, activation="relu"))
    model.add(Conv2D(256, 2, activation="relu"))
    model.add(Conv2D(210 * 210, 2, activation="relu"))
    model.add(Reshape((210, 210,)))
    model.add(Activation("softmax"))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# ----------------------------------------------------------------------------------------------------------------------


def _preprocess(input_folder, list):
    _list = []
    _noise_list = []
    for _file in list:
        _data, _rate = sf.read(os.path.join(input_folder, _file))
        if not len(_data) == 44100:
            continue
        _list.append(
            np.reshape(_data, newshape=(210, 210)))

        _noise = np.random.uniform(low=-0.09, high=0.09, size=_data.shape[0])
        _noise = np.add(_data, _noise)
        _noise_list.append(
            np.reshape(_noise, newshape=(210, 210)))

    return np.asarray(_list), np.asarray(_noise_list)


# ----------------------------------------------------------------------------------------------------------------------


@click.command()
@click.option("--input_folder", "-i",
              default=r"D:\data_sets\sounds\_pro",
              help="")
def start(input_folder):
    file_list = [f for f in os.listdir(input_folder) if f.__contains__(".ogg")]
    random.shuffle(file_list)

    _piece_size = int(len(file_list) / 8)
    _piece_list = []
    for x in range(0, len(file_list), _piece_size):
        _piece_list.append(file_list[x:(x + _piece_size)])

    for _piece in _piece_list:
        _original_list, _noised_list = _preprocess(input_folder, _piece)
        print(_original_list.shape, _noised_list.shape)

    model = _create_rnn()
    # model.fit(_original_list, _noised_list,
    #           batch_size=64, epochs=10)


if __name__ == '__main__':
    start()
