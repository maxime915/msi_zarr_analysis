import pathlib
import sys

import numpy as np
from PIL import Image

from .iter_chunks import iter_loaded_chunks
from .check import open_group_ro

path = sys.argv[1]
mz_val = float(sys.argv[2])
mz_tol = float(sys.argv[3])
clip_f = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

print(f"{mz_val=} {mz_tol=} {clip_f=}")

z = open_group_ro(path)
z_int = z["/0"]
z_mzs = z["/labels/mzs/0"]
z_len = z["/labels/lengths/0"]

roi = z_len[0, 0, :, :]
roi[roi != 0] = 255
roi = roi.astype(np.uint8)

img = np.zeros(shape=z_int.shape[2:], dtype=z_int.dtype)

for (cy, cx) in iter_loaded_chunks(z_int, skip=2):
    n_int = z_int[:, 0, cy, cx]
    n_len = z_len[0, 0, cy, cx]
    n_mzs = z_mzs[:, 0, cy, cx]

    for i in range(n_int.shape[1]):
        for j in range(n_int.shape[2]):
            len = n_len[i, j]
            mz_band = n_mzs[:len, i, j]
            low = np.searchsorted(mz_band, mz_val - mz_tol, side="left")
            hi = np.searchsorted(mz_band, mz_val + mz_tol, side="right")

            if low == hi:
                continue

            img[i + cy.start, j + cx.start,] = n_int[
                low:hi, i, j
            ].sum(axis=0)

# take the highest values
pos = np.zeros_like(roi)
img_max = img.max()
target = clip_f * img_max
print(f"{img_max=} {target=}")
pos[img >= target] = 255

base = pathlib.Path(path).stem

# compute the TPR
valid_values = pos[roi > 0]
tpr = (valid_values > 0.5 * valid_values.max()).sum() / valid_values.size

if min(tpr, 1-tpr) < 0.4:
    print(f"{tpr=} => exiting")
    sys.exit(0)

print(f"{tpr=}")

Image.fromarray(roi).save(base + f"_{mz_val}_{mz_tol}_roi.jpg")

Image.fromarray(pos).save(base + f"_{mz_val}_{mz_tol}_pos.jpg")
