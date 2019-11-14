import os
import click
import numpy as np
import soundfile as sf

from multiprocessing import Process


@click.command()
@click.option("--input_folder", "-b",
              default=r"D:\data_sets\sounds\raw_0",
              help="Input folder")
@click.option("--output_folder", "-b",
              default=r"D:\data_sets\sounds\raw_1",
              help="Output folder")
def start(input_folder, output_folder):
    sounds = [f for f in os.listdir(input_folder) if f.__contains__(".ogg")]

    size = int(len(sounds) / 12)
    for x in range(0, len(sounds), size):
        Process(target=_trim, args=(input_folder, sounds[x:(x + size)], output_folder)).start()


def _trim(input_folder, sound_list, output_folder):
    for sound in sound_list:
        data, sample_rate = sf.read(os.path.join(input_folder, sound))
        sf.write(os.path.join(output_folder, sound), data[:(sample_rate * 5)], sample_rate)


if __name__ == '__main__':
    start()
