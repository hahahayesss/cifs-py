import os
import time
import click
import shutil
import numpy as np
import soundfile as sf

folder = r"D:\data_sets\sounds\__pro"
file = os.path.join(folder, "_last.flac")

_arr, _rate = sf.read(file)
_arr = np.asarray(_arr)[:12000 * 44100]
print(_arr.shape)

_arr = _arr.reshape((108000, 70, 70, 1))
print(_arr.shape)

# sf.write(os.path.join(folder, "_last.flac"), _arr, 44100)
