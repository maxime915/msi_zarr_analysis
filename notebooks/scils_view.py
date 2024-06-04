# %% imports & routines

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from omezarrmsi import OMEZarrMSI
from omezarrmsi.proc.select import where_spectra


def view(
    dataset_: OMEZarrMSI,
    yx_lst: list[tuple[int, int]],
    mz_lo: float,
    mz_hi: float,
) -> list[tuple[np.ndarray, np.ndarray]]:

    mask = np.zeros_like(dataset_.z_len[0, 0])
    for y, x in yx_lst:
        mask[y, x] = 1

    ret_lst = []

    dataset_ = where_spectra(np.expand_dims(mask, 0), dataset_)
    for s_mzs, s_int in dataset_.iter_spectra():
        # filter based on m/Z
        lo = np.searchsorted(s_mzs, mz_lo, side="left")
        hi = np.searchsorted(s_mzs, mz_hi, side="right")

        ret_lst.append((s_mzs[lo:hi], s_int[lo:hi]))

    return ret_lst


# %% data loading

dataset_path = {
    "_____r13____": pathlib.Path("../tmp_r13.zarr"),
    "_____r13_317": pathlib.Path("../tmp_r13_n317.zarr"),
    "slim_r13____": pathlib.Path.home() / "datasets/Slim imzML/Region 13/No Normalization/region13_nonorm_sample.zarr",
    "slim_r13_317": pathlib.Path.home() / "datasets/Slim imzML/Region 13/317 peak area Norm/region13_317norm_sample.zarr",
}
yx_coords: list[tuple[int, int]] = [
    (10, 110),
    (15, 110),
    (15, 115),
    (15, 120),
]

# dataset_path = {
#     "_____r14____": pathlib.Path("../tmp_r14.zarr"),
#     "_____r14_317": pathlib.Path("../tmp_r14_n317.zarr"),
#     "slim_r14____": pathlib.Path.home() / "datasets/Slim imzML/Region 14/no normalization/region14_nonorm_sample.zarr",
#     "slim_r14_317": pathlib.Path.home() / "datasets/Slim imzML/Region 14/317 peak area Norm/region14_317norm_sample.zarr",
# }
# yx_coords: list[tuple[int, int]] = [
#     (15, 200),
#     (20, 200),
#     (20, 205),
#     (20, 210),
# ]

# dataset_path = {
#     "_____r15____": pathlib.Path("../tmp_r15.zarr"),
#     "_____r15_317": pathlib.Path("../tmp_r15_n317.zarr"),
#     "slim_r15____": pathlib.Path.home() / "datasets/Slim imzML/Region 15/no normalization/region15_nonorm_sample.zarr",
#     "slim_r15_317": pathlib.Path.home() / "datasets/Slim imzML/Region 15/317 peak area Norm/region15_317norm_sample.zarr",
# }
# yx_coords: list[tuple[int, int]] = [
#     (10, 20),
#     (15, 20),
#     (15, 25),
#     (15, 30),
# ]

dataset_dict = {key: OMEZarrMSI(path, mode="r") for key, path in dataset_path.items()}

mask: np.ndarray | None = None
for key, ds in dataset_dict.items():
    ds_mask = ds.z_len[0, 0] > 0  # type: ignore
    if not ds_mask.any():
        raise ValueError(f"empty dataset: {key}")
    if mask is None:
        mask = ds_mask
    else:
        mask = np.logical_and(mask, ds_mask)
        if not mask.any():
            raise ValueError(f"inconsistent dataset: {key}")
if mask is None:
    raise ValueError("no dataset")

# z_lengths: np.ndarray = dataset.z_len[0, 0]
# val, counts = np.unique(z_lengths, return_counts=True)

# for v, c in zip(val, counts):
#     print(f"seen {v} : {c} times")

# %% show mask

plt.imshow(mask)

# %% display

# yx_coords: list[tuple[int, int]] = [
#     (21, 150),
#     (21, 200),
# ]

# # yx_coords: list[tuple[int, int]] = [
# #     (10, 110),
# #     (15, 110),
# #     (15, 115),
# #     (15, 120),
# #     (20, 200),
# # ]
# # mz_range: tuple[float, float] = (350.145, 350.170)
# mz_range: tuple[float, float] = (350.153, 350.154)

mz_range: tuple[float, float]
mz_range = (350.145, 350.170)
# mz_range = (316.0, 318.0)


height, width = mask.shape

if any(x < 0 or x >= width or y < 0 or y >= height for (y, x) in yx_coords):
    plt.imshow(np.where(mask, mask, np.nan))
    plt.title("Valid spectra")
    plt.tight_layout
    raise ValueError(f"coordinates are invalid for {width=} {height=}")

for y, x in yx_coords:
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"coordinates {y=}, {x=} are invalid: {height=}, {width=}")
    if mask[y, x] == 0:
        raise ValueError(f"no spectrum is present at {y=}, {x=}")

spectra_dict = {
    key: view(ds, yx_coords, *mz_range)
    for key, ds in dataset_dict.items()
}

# %%

fig, ax = plt.subplots(
    len(spectra_dict),
    1,
    squeeze=False,
    figsize=(8, 3 * len(spectra_dict)),
)
# fig.suptitle("Multiple spectra")

for idx, (key, spectra) in enumerate(spectra_dict.items()):
    for s_mzs, s_int in spectra:
        p = ax[idx, 0].plot(s_mzs, s_int, "-o")
        ax[idx, 0].stem(s_mzs, s_int, p[0].get_color())
    ax[idx, 0].set_title(key)
    ax[idx, 0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

fig.tight_layout()

# %%

# assert False

mz_range: tuple[float, float]
mz_range = (350.145, 350.170)
# mz_range = (316.0, 318.0)


height, width = mask.shape

if any(x < 0 or x >= width or y < 0 or y >= height for (y, x) in yx_coords):
    plt.imshow(np.where(mask, mask, np.nan))
    plt.title("Valid spectra")
    plt.tight_layout
    raise ValueError(f"coordinates are invalid for {width=} {height=}")

for y, x in yx_coords:
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"coordinates {y=}, {x=} are invalid: {height=}, {width=}")
    if mask[y, x] == 0:
        raise ValueError(f"no spectrum is present at {y=}, {x=}")

selected_keys = ["slim_r13_317"]

spectra_dict = {
    key: view(ds, yx_coords, *mz_range)
    for key, ds in dataset_dict.items() if key in selected_keys
}

# %%

fig, ax = plt.subplots(
    len(spectra_dict),
    1,
    squeeze=False,
    figsize=(8, 3 * len(spectra_dict)),
)
fig.suptitle("Multiple spectra")

for idx, (key, spectra) in enumerate(spectra_dict.items()):
    print(f"{idx=}, {key=}")
    for s_mzs, s_int in spectra:
        p = ax[idx, 0].plot(s_mzs, s_int, "-o")
        # ax[idx, 0].stem(s_mzs, s_int, p[0].get_color())
        print(f"\t{list(s_mzs)} {list(s_int)}")
    ax[idx, 0].set_title(key)

fig.tight_layout()

# %%

plt.figure(figsize=(2, 2))

# p = plt.plot([0, 1, 2], [0, 1, 0])
# plt.stem([0, 1, 2], [0, 1, 0], p[0].get_color())

p = plt.plot([1], [1])
plt.stem([1], [1], p[0].get_color())

plt.tight_layout()
plt.xlim((0, 2))

# %%
