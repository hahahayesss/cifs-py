import os
import click
import random
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def preprocess(data):
    _noise = np.random.uniform(low=-0.09, high=0.09, size=data.shape[0])
    _with_noise = np.add(data, _noise)

    # ------------------------------------------------------------------------------------------------------------------

    # 12000 * 44100 = 529200000
    # 44100 / 25 = 1764
    # 42 * 42 = 1764
    # 529200000 / 1764 = 300000

    # _data = data.reshape((300000, 42, 42, 1))
    # _with_noise = _with_noise.reshape((300000, 42, 42, 1))

    _data = data.reshape((330750, 1600, 1))
    _with_noise = _with_noise.reshape((330750, 1600, 1))

    # ------------------------------------------------------------------------------------------------------------------

    val_data = _with_noise[-30750:]
    val_answers = _data[-30750:]

    train_data = _with_noise[:-30750]
    train_answers = _data[:-30750]

    return train_data, train_answers, val_data, val_answers


# def create_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(42, (3, 3), activation="relu", input_shape=(42, 42, 1)))
#     model.add(layers.Conv2D(42, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(84, (3, 3), activation="relu"))
#     model.add(layers.Conv2D(84, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((3, 3)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(2000, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(42 * 42, activation='relu'))
#     model.add(layers.Reshape((42, 42, 1)))
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                   loss=tf.keras.losses.Hinge(),
#                   metrics=[tf.keras.metrics.Hinge()])
#     return model

# def create_model():
#     model = models.Sequential()
#     model.add(layers.Conv1D(42, 3, activation="relu", input_shape=(1764, 1)))
#     model.add(layers.Conv1D(42, 3, activation="relu"))
#     model.add(layers.MaxPooling1D(2))
#     model.add(layers.Conv1D(84, 3, activation="relu"))
#     model.add(layers.Conv1D(84, 3, activation="relu"))
#     model.add(layers.MaxPooling1D(3))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(2000, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(1764, activation='relu'))
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                   loss=tf.keras.losses.Hinge(),
#                   metrics=[tf.keras.metrics.Hinge()])
#     return model

def get_crop_shape(target, refer):
    ch = target.get_shape()[1] - refer.get_shape()[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2)


def create_model():
    inputs = Input(shape=(1600, 1))

    conv1 = Conv1D(40, 1, activation="relu")(inputs)
    conv1 = Conv1D(40, 1, activation="relu")(conv1)
    pool1 = MaxPooling1D(2)(conv1)

    conv2 = Conv1D(40, 1, activation="relu")(pool1)
    pool2 = MaxPooling1D(2)(conv2)

    conv3 = Conv1D(40, 1, activation="relu")(pool2)
    pool3 = MaxPooling1D(2)(conv3)

    conv4 = Conv1D(40, 1, activation="relu")(pool3)
    pool4 = MaxPooling1D(2)(conv4)

    conv5 = Conv1D(40, 1, activation="relu")(pool4)
    pool5 = MaxPooling1D(2)(conv5)

    conv6 = Conv1D(40, 1, activation="relu")(pool5)
    up6 = UpSampling1D(2)(conv6)
    concat6 = concatenate([up6, pool4], axis=1)

    conv7 = Conv1D(80, 1, activation="relu")(concat6)
    conv7 = Conv1D(80, 1, activation="relu")(conv7)

    up7 = UpSampling1D(2)(conv7)
    ch = get_crop_shape(up7, pool3)
    crop7 = Cropping1D(cropping=ch)(up7)
    concat7 = concatenate([pool3, crop7], axis=2)

    conv8 = Conv1D(80, 1, activation="relu")(concat7)
    conv8 = Conv1D(80, 1, activation="relu")(conv8)

    up8 = UpSampling1D(2)(conv8)
    ch = get_crop_shape(pool2, up8)
    crop8 = Cropping1D(cropping=ch)(pool2)
    concat8 = concatenate([up8, crop8], axis=2)

    conv9 = Conv1D(80, 1, activation="relu")(concat8)
    conv9 = Conv1D(80, 1, activation="relu")(conv9)

    up9 = UpSampling1D(2)(conv9)
    ch = get_crop_shape(pool1, up9)
    crop9 = Cropping1D(cropping=ch)(pool1)
    concat9 = concatenate([up9, crop9], axis=2)

    conv10 = Conv1D(80, 1, activation="relu")(concat9)
    conv10 = Conv1D(80, 1, activation="relu")(conv10)

    up10 = UpSampling1D(2)(conv10)
    ch = get_crop_shape(inputs, up10)
    crop10 = Cropping1D(cropping=ch)(inputs)
    concat10 = concatenate([up10, crop10], axis=2)

    conv11 = Conv1D(64, 1, activation="relu")(concat10)
    conv11 = Conv1D(32, 1, activation="relu")(conv11)
    conv11 = Conv1D(1, 1, activation="relu")(conv11)

    model = Model(inputs, conv11)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


@click.command()
@click.option("--input_file", "-i",
              default=r"D:\data_sets\sounds\__pro\_last.flac",
              help="")
def start(input_file):
    _data, _rate = sf.read(input_file)
    _data = np.asarray(_data)[:12000 * 44100]

    train_data, train_answers, val_data, val_answers = preprocess(_data)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = create_model()
    print(model.summary())

    history = model.fit(train_data, train_answers, batch_size=64, epochs=10,
                        validation_data=(val_data, val_answers))

    col_list = []
    for _col in history.history.keys():
        plt.plot(history.history[_col])
        col_list.append(_col)
    plt.legend(col_list, loc='upper left')
    plt.show()

    tf.saved_model.save(model, r"D:\data_sets\sounds")


if __name__ == '__main__':
    start()
