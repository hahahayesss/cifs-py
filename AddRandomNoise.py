import os
import click
import numpy as np
import soundfile as sf

from multiprocessing import Process

noise_level = [[-0.009, 0.009],
               [-0.09, 0.09],
               [-0.9, 0.9]]


def _add_noise(input_folder, sound_list, output_folder):
    for index, level in enumerate(noise_level):
        for sound in sound_list:
            try:
                data, sample_rate = sf.read(os.path.join(input_folder, sound))
                print(sample_rate)
                noise = np.random.uniform(low=level[0], high=level[1], size=(data.shape[0]))
                data = np.add(data, noise)
                sf.write(os.path.join(output_folder, sound.split(".")[0] + "_" + str(index) + ".ogg"),
                         data[:(sample_rate * 5)], sample_rate)
            except:
                print("Error : " + sound)


@click.command()
@click.option("--input_folder", "-b",
              default=r"D:\data_sets\sounds\raw_1",
              help="Input folder")
@click.option("--output_folder", "-b",
              default=r"D:\data_sets\sounds\raw_2",
              help="Output folder")
def start(input_folder, output_folder):
    sounds = [f for f in os.listdir(input_folder) if f.__contains__(".ogg")]

    size = int(len(sounds) / 24)
    for x in range(0, len(sounds), size):
        Process(target=_add_noise, args=(input_folder, sounds[x:(x + size)], output_folder)).start()


if __name__ == '__main__':
    start()
