"""
- use the binning with info provided by Sam
- find a good threshold
- mask-out values below said threshold
- groups neighboring values together
- compute new bins based on the grouping
- bin the datasetS with the new binning
- implemented a supervised method
"""

# %%

import pathlib
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from omezarrmsi import OMEZarrMSI
from omezarrmsi.proc.bin import spectra_sum
from skimage.measure import label as label_components

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from msi_zarr_analysis.preprocessing.binning import fticr_binning, bin_and_flatten_v2  # noqa:E402

# %%

tag = "nonorm"  # "317norm"

de_tol = 2e-3  # tolerance of the deisotoping function
slim_dir = pathlib.Path.home() / "datasets" / "COMULIS-slim-msi"
dest_dir = slim_dir.parent / f"slim-deisotoping-{de_tol:.1e}"

files = sorted(dest_dir.iterdir())
assert len(files) == 6

files = [f for f in files if tag in f.name]
assert len(files) == 3, files

ozm_datasets = {f.stem[6:8]: OMEZarrMSI(f, mode="r") for f in files}
assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

# %%

bin_l, bin_r, _ = fticr_binning(115.0, 1150.0, 2869580)
bin_c = 0.5 * (bin_l + bin_r)
bin_w = bin_r - bin_l

assert (bin_w > 0).all()
# make sure than bin_l = bin_lr[:-1] and bin_r = bin_lr[1:]
assert bin_l.base is bin_r.base
assert bin_r.base is not None and len(bin_r.base) == len(bin_r) + 1
assert (bin_l[1:] == bin_r[:-1]).all()
bin_lr = bin_r.base

# %%

spec_sum_dct = {key: spectra_sum(ozm, bin_lr) for key, ozm in ozm_datasets.items()}

# %%

fig, ax = plt.subplots(figsize=(6, 6))
ax.violinplot(
    [np.log10(1 + s) for s in spec_sum_dct.values()],
    showmeans=False,
    showmedians=True,
)

labels = list(spec_sum_dct.keys())
ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax.set_xlim(0.25, len(labels) + 0.75)

# %% experimentally found: we could try to find another approach but not today
thresh = 1e4  # works for both normalization methods

# %%  remove below threshold
above_thresh = (
    (spec_sum_dct["13"] > thresh)
    * (spec_sum_dct["14"] > thresh)
    * (spec_sum_dct["15"] > thresh)
)
above_thresh = above_thresh.astype(np.int64)

# leading: if this bin is 1 and the previous is 0
leading = np.empty_like(above_thresh)
leading[0] = above_thresh[0]
leading[1:] = above_thresh[1:] * (1 - above_thresh[:-1])

# trailing: if this bin is 1 and the next one is 0
trailing = np.empty_like(above_thresh)
trailing[-1] = above_thresh[-1]
trailing[:-1] = above_thresh[:-1] * (1 - above_thresh[1:])

# as many leading as trailing
assert sum(leading) == sum(trailing)

merged_l = bin_l[leading.nonzero()]
merged_r = bin_r[trailing.nonzero()]
merged_c = 0.5 * (merged_r + merged_l)
merged_w = merged_r - merged_l

# %%

plt.plot(merged_c, merged_w)
plt.yscale("log")

# %% copied from notebooks/make_dataset.py


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    dataset_y: np.ndarray
    groups: np.ndarray
    bin_l: np.ndarray
    bin_r: np.ndarray
    regions: np.ndarray | None  # only used after merging


def make_tabular(
    ozm: OMEZarrMSI, label: np.ndarray, bin_lo: np.ndarray, bin_hi: np.ndarray
):
    "label: float[y, x, L]"
    mask_yx = np.any(label, axis=-1)
    n_rows = int(mask_yx.sum())
    n_feat = len(bin_lo)

    # build dataset_x
    dataset_x = np.empty((n_rows, n_feat), dtype=ozm.z_int.dtype)

    # populate dataset
    ys, xs = bin_and_flatten_v2(dataset_x, ozm.group, mask_yx, bin_lo, bin_hi)
    dataset_y = label[ys, xs, :]

    # find blobs in the label + make groups
    group_lst: list[np.ndarray] = []
    for cls_ in range(label.shape[-1]):
        label_cls_ = label[..., cls_]
        components, num = label_components(
            label_cls_ > 0, background=0, return_num=True
        )
        for group_idx in range(1, num + 1):
            mask = (components == group_idx).astype(int)
            assert mask.any()
            group_lst.append(mask)

    groups_stack = np.stack(group_lst, axis=-1)
    groups_yx = groups_stack.argmax(axis=-1)
    invalid_flag = np.iinfo(groups_yx.dtype).min
    groups_yx[~groups_stack.any(axis=-1)] = invalid_flag

    # find groups
    groups = groups_yx[ys, xs]
    assert (groups != invalid_flag).all(), "all rows in groups should be valid"

    return Tabular(dataset_x, dataset_y, groups, bin_lo, bin_hi, None)


def merge_tabular(*datasets: Tabular):
    if any(ds_.regions is not None for ds_ in datasets):
        raise ValueError("only non-merged datasets are allowed")

    if not datasets:
        raise ValueError("at least one dataset must be provided")
    bin_l = datasets[0].bin_l
    bin_r = datasets[0].bin_r
    if any((ds_.bin_l != bin_l).any() for ds_ in datasets):
        raise ValueError("inconsistent bin_l")
    if any((ds_.bin_r != bin_r).any() for ds_ in datasets):
        raise ValueError("inconsistent bin_r")

    ds_x: list[np.ndarray] = []
    ds_y: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    regions: list[np.ndarray] = []

    g_offset = 0
    for idx, ds_ in enumerate(datasets):
        ds_x.append(ds_.dataset_x)
        ds_y.append(ds_.dataset_y)

        if ds_.groups.min() != 0:
            raise ValueError(f"{ds_.groups.min()=} but expected 0")
        groups.append(ds_.groups + g_offset)
        g_offset += ds_.groups.max() + 1

        regions.append(np.zeros_like(ds_.groups) + idx)

    return Tabular(
        np.concatenate(ds_x),
        np.concatenate(ds_y),
        np.concatenate(groups),
        bin_l,
        bin_r,
        np.concatenate(regions),
    )


def save_tabular(path: pathlib.Path, ds: Tabular):
    assert ds.regions is not None
    np.savez(path, *ds)  # type:ignore


# %%

lbl_dct = {
    key: np.stack([ds_.get_label(lbl)[0, 0] for lbl in ds_.labels()], axis=-1)
    for key, ds_ in ozm_datasets.items()
}

# %%

merged_ds = merge_tabular(
    *[
        make_tabular(ozm, lbl, merged_l, merged_r)
        for ozm, lbl in zip(ozm_datasets.values(), lbl_dct.values(), strict=True)
    ]
)

# %%

save_tabular(
    slim_dir.parent / f"slim-deisotoping-{de_tol:.1e}-binned" / f"binned_{tag}.npz",
    merged_ds,
)
