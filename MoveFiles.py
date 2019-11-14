import os
import click
import shutil
import numpy as np
import soundfile as sf


@click.command()
@click.option("--input_folder", "-b",
              default=r"D:\data_sets\sounds\raw_0",
              help="Input folder")
@click.option("--output_folder", "-b",
              default=r"D:\data_sets\sounds\raw_0",
              help="Output folder")
def start(input_folder, output_folder):
    folders = [f for f in os.listdir(input_folder)]

    counter = 0
    for folder in folders:
        base_folder = os.path.join(input_folder, folder)
        files = [f for f in os.listdir(base_folder) if f.__contains__(".ogg")]

        for file in files:
            file_name = str(counter)
            while len(file_name) <= 10:
                file_name = "0" + file_name
            file_name = file_name + ".ogg"

            shutil.move(os.path.join(base_folder, file), os.path.join(output_folder, file_name))
            counter += 1


if __name__ == '__main__':
    start()
