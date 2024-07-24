import pathlib
import sys

import numpy as np
from PIL import Image

from .check import open_group_ro

z = open_group_ro(sys.argv[1])
lengths = z["/labels/lengths/0"][0, 0, :, :]

rand_mask = np.random.random(lengths.shape) < 0.5

roi = lengths.copy()
roi[roi != 0] = 255
roi = roi.astype(np.uint8)

pos = roi.copy()
pos[rand_mask] = 0

base = pathlib.Path(sys.argv[1]).stem

Image.fromarray(roi).save(base + "_rand_roi.jpg")

Image.fromarray(pos).save(base + "_rand_pos.jpg")
