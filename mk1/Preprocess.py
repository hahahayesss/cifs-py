import os
import time
import click
import shutil
import numpy as np
import soundfile as sf

from multiprocessing import Process


@click.command()
@click.option("--input_folder", "-i",
              default=r"D:\data_sets\sounds\raw_0",
              help="")
@click.option("--output_folder", "-o",
              default=r"D:\data_sets\sounds\_pro",
              help="")
def start(input_folder, output_folder):
    _sound_files = [f for f in os.listdir(input_folder) if f.__contains__(".ogg")]
    _pool_size = int(len(_sound_files) / 20)

    for x in range(0, len(_sound_files), _pool_size):
        Process(target=_process_manager, args=(input_folder,
                                               _sound_files[x:(x + _pool_size)],
                                               output_folder)).start()
        print(x, ": Process manager's started")


def _process_manager(input_folder, sound_file_list, output_folder):
    for _file in sound_file_list:
        __process = Process(target=__piece, args=(input_folder,
                                                  _file,
                                                  output_folder))
        __process.start()
        while __process.is_alive():
            time.sleep(0.2)


def __piece(input_folder, file, output_folder):
    _data, _rate = sf.read(os.path.join(input_folder, file))
    if len(_data) < _rate * 5:
        return
    elif len(_data) > _rate * 5:
        _data = _data[:_rate * 5]

    for index, __data in enumerate(np.reshape(_data, newshape=(5, _rate))):
        sf.write(os.path.join(output_folder, str(index) + "-" + file), __data, _rate)


if __name__ == '__main__':
    start()
