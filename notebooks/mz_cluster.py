# %% imports & routines

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.errors


# %% data loading

data_path = (
    pathlib.Path.home()
    / "datasets/Slim imzML/Region 15/no normalization/region15_nonorm_sample.zarr"
)
data_zarr = zarr.open_group(data_path, mode="r")

# does it crash ?
all_mzs = np.reshape(data_zarr["/labels/mzs/0"][...], (-1,))
value = np.unique(all_mzs)

# %% show

diffs_mDa = 1e3 * (value[1:] - value[:-1])
# diffs[diffs > 1.36353e-2] = np.nan
# diffs[diffs > 1.e-1] = np.nan
# diffs[diffs > 4.6e-2] = np.nan
diffs_mDa[diffs_mDa > 2.662] = np.nan
# almost nothing below

min_diff = diffs_mDa[~np.isnan(diffs_mDa) & (diffs_mDa > 0)].min()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(value[:-1], diffs_mDa, ".")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_ylabel("y diff (mDa)")
# ax.set_yticks(sorted([min_diff, 0.046, 0.1, 0.2, 0.5, 1.0, 2.0, 2.662]))
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.tight_layout()

# %%

min_diff

# %%

files = [
    "Region 14/317 peak area Norm/region14_317norm_sample.zarr",
    "Region 14/no normalization/region14_nonorm_sample.zarr",
    "Region 15/317 peak area Norm/region15_317norm_sample.zarr",
    "Region 15/no normalization/region15_nonorm_sample.zarr",
    "Region 13/No Normalization/region13_nonorm_sample.zarr",
    "Region 13/317 peak area Norm/region13_317norm_sample.zarr",
]

def _open_r(path: pathlib.Path):
    try:
        return zarr.open_group(path, mode="r")
    except zarr.errors.GroupNotFoundError as err:
        raise FileNotFoundError(path) from err

paths = [pathlib.Path.home() / "datasets/Slim imzML" / f for f in files]
# paths = [pathlib.Path("../tmp_r13.zarr"), pathlib.Path("../tmp_r14.zarr"), pathlib.Path("../tmp_r15.zarr")]
groups = [_open_r(p.absolute()) for p in paths]

# %%

values_mzs = [np.unique(g["/labels/mzs/0"][...]) for g in groups]
print(f"{[len(v) for v in values_mzs]=}")

# %%

higher_bound_len_mzs = [np.sum(g["/labels/lengths/0"][...]) for g in groups[::2]]
print(f"{higher_bound_len_mzs=}")

print(f"{groups[0]['/labels/mzs/0'].shape[0]=}")
print(f"{groups[2]['/labels/mzs/0'].shape[0]=}")
print(f"{groups[4]['/labels/mzs/0'].shape[0]=}")

# %%

ratios = [len(v) / b for (v, b) in zip(values_mzs[::2], higher_bound_len_mzs)]
print(f"{ratios=}")

# %%

print(f"{(values_mzs[0] != values_mzs[1]).any()=}")
print(f"{(values_mzs[2] != values_mzs[3]).any()=}")
print(f"{(values_mzs[4] != values_mzs[5]).any()=}")

# %%

mzs_sets = [set(v) for v in values_mzs[::2]]

print(f"{len(mzs_sets[0] - mzs_sets[1])=}")
print(f"{len(mzs_sets[0] - mzs_sets[2])=}")
print(f"{len(mzs_sets[1] - mzs_sets[0])=}")
print(f"{len(mzs_sets[1] - mzs_sets[2])=}")
print(f"{len(mzs_sets[2] - mzs_sets[0])=}")
print(f"{len(mzs_sets[2] - mzs_sets[1])=}")

# %%

print(f"{(groups[0]['/labels/mzs/0'][...] != groups[1]['/labels/mzs/0'][...]).any()=}")
print(f"{(groups[2]['/labels/mzs/0'][...] != groups[3]['/labels/mzs/0'][...]).any()=}")
print(f"{(groups[4]['/labels/mzs/0'][...] != groups[5]['/labels/mzs/0'][...]).any()=}")

# %%

# %%
