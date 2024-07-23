# %%

import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
from omezarrmsi import OMEZarrMSI

from omezarrmsi.plots.mz_slice import mz_slice

import context  # noqa
from msi_zarr_analysis.preprocessing.binning import fticr_binning

# %%

de_tol = 2e-3  # tolerance of the deisotoping function
slim_dir = pathlib.Path.home() / "datasets" / "COMULIS-slim-msi"
dest_dir = slim_dir.parent / f"slim-deisotoping-{de_tol:.1e}"

files = sorted(dest_dir.iterdir())
assert len(files) == 6

files = [f for f in files if "317norm" in f.name]
assert len(files) == 3, files

ozm_datasets = {f.stem[6:8]: OMEZarrMSI(f, mode="r") for f in files}
assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

# %%

# %%

# bin_le, bin_re, _ = fticr_binning(150.0, 1150.0, 2869580)
bin_le, bin_re, _ = fticr_binning(150.0, 1150.0, 2869580)
bin_c = 0.5 * (bin_le + bin_re)
bin_w = bin_re - bin_le

# %%

lipids = [
    205.19508,
    243.26824,
    285.27881,
    305.24751,
    317.21112,
    321.24242,
    335.22168,
    337.23734,
    353.23225,
    355.2479,
    361.23734,
    371.24282,
    377.23226,
    380.25603,
    496.33976,
    524.37107,
    546.35541,
    550.35033,
    552.40237,
    594.37654,
    596.33469,
    610.37146,
    650.43915,
    664.41841,
    666.43406,
    734.56943,
    758.56943,
    782.56943,
    786.60073,
    798.56435,
    806.56943,
    810.52796,
    810.60073,
    812.54361,
    814.55926,
    828.53852,
    832.56983,
]

# %%

n_rows = 20
r_idx = random.sample(list(range(len(bin_c))), n_rows)
a = 0.5  # <- the correct value

max_val = -np.inf
ion_images: list[np.ndarray] = []
for idx in r_idx:
    mz_c = bin_c[idx]
    mz_lo = mz_c - a * bin_w[idx]
    mz_hi = mz_c + a * bin_w[idx]
    mask, data = mz_slice(ozm_datasets["13"], mz_lo, mz_hi)
    max_val = max(max_val, data[mask].max())
    data[~mask] = np.nan
    ion_images.append(data)

lipid_ion_images: list[np.ndarray] = []
for mz in lipids:
    idx = int(np.argmin(np.abs(bin_c - mz)))
    mz_c = bin_c[idx]
    mz_lo = mz_c - a * bin_w[idx]
    mz_hi = mz_c + a * bin_w[idx]
    mask, data = mz_slice(ozm_datasets["13"], mz_lo, mz_hi)
    max_val = max(max_val, data[mask].max())
    data[~mask] = np.nan
    lipid_ion_images.append(data)

for data in ion_images:
    data /= max_val

for data in lipid_ion_images:
    data /= max_val

# %%

fig, axes = plt.subplots(n_rows, 1, squeeze=False, figsize=(10, 1 * n_rows))

for r, img in enumerate(ion_images):
    axes[r, 0].set_axis_off()
    axes[r, 0].imshow(img, interpolation="nearest", vmin=0.0, vmax=1.0)

# %%

fig, axes = plt.subplots(len(lipids), 1, squeeze=False, figsize=(10, 1 * len(lipids)))

for r, img in enumerate(lipid_ion_images):
    axes[r, 0].set_axis_off()
    axes[r, 0].imshow(img, interpolation="nearest", vmin=0.0, vmax=1.0)

# %%
