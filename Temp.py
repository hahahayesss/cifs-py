import os
import time
import click
import shutil
import numpy as np
import soundfile as sf

folder = r"D:\data_sets\sounds\_pro"
files = [f for f in os.listdir(folder) if f.__contains__(".ogg")]
np.random.shuffle(files)
for x in range(0, 13000, 1000):
    temp_files = files[x:x + 1000]

    data_list = []
    for y, file in enumerate(temp_files):
        _data, _rate = sf.read(
            os.path.join(folder, file))
        data_list.extend(_data)

        if y % 100 == 0:
            print(y)

    if not len(data_list) == 1000 * 44100:
        print("NOT")

    if len(data_list) > 1000 * 44100:
        print("bigger")
        data_list = data_list[:1000 * 44100]

    if len(data_list) < 1000 * 44100:
        print("smaller")

    sf.write(os.path.join(folder, "__data" + str(x) + ".flac"), data_list, 44100)
    print("\nNew")

# for x in:
#     _data, _rate = sf.read(os.path.join(folder, x))
#     if not _data.shape[0] == 44100:
#         print("Not ", x)
