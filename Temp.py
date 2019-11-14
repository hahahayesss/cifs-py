import os
import time
import click
import shutil
import numpy as np
import soundfile as sf

from multiprocessing import Process


@click.command()
@click.option("--input_folder", "-i",
              default=r"D:\data_sets\sounds\_pro",
              help="")
@click.option("--output_folder", "-o",
              default=None,
              help="")
def start(input_folder, output_folder):
    _file_list = [f for f in os.listdir(input_folder) if f.__contains__(".ogg")]
    _piece_size = int(len(_file_list) / 100)

    _piece_counter = 0
    max_min = 0.0
    smallest_max = ""
    for index, _file in enumerate(_file_list):
        _data, _rate = sf.read(os.path.join(input_folder, _file))

        _max = np.max(_data)
        if max_min > _max:
            max_min = _max
            smallest_max = _file

        if index % _piece_size == 0:
            print("%", _piece_counter)
            _piece_counter += 1

    print(smallest_max, max_min)


if __name__ == '__main__':
    start()
