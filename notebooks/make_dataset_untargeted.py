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
from typing import NamedTuple, Literal

import matplotlib.pyplot as plt
import numpy as np
from omezarrmsi import OMEZarrMSI
from omezarrmsi.proc.bin import spectra_sum
from skimage.measure import label as label_components

from msi_zarr_analysis.preprocessing.binning import fticr_binning, bin_and_flatten_v2

# %%

tag: Literal["nonorm", "317norm"] = "nonorm"

de_tol = 2e-3  # tolerance of the deisotoping function
datasets_dir = pathlib.Path.home() / "datasets"
non_binned_dir = datasets_dir / f"slim-deisotoping-{de_tol:.1e}"

files = sorted(non_binned_dir.iterdir())
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
    annotation_overlap: np.ndarray
    annotation_idx: np.ndarray
    bin_l: np.ndarray
    bin_r: np.ndarray
    regions: np.ndarray | None  # only used after merging
    coord_y: np.ndarray
    coord_x: np.ndarray


def make_tabular(
    ozm: OMEZarrMSI, label: np.ndarray, bin_lo: np.ndarray, bin_hi: np.ndarray, bin_method: Literal["sum", "integration"],
):
    "label: float[y, x, L]"

    # bin without duplicate first
    mask_yx = np.any(label, axis=-1)
    n_spec = int(mask_yx.sum())
    n_feat = len(bin_lo)

    spectra = np.zeros((n_spec, n_feat), dtype=ozm.z_int.dtype)
    ys, xs = bin_and_flatten_v2(spectra, ozm.group, mask_yx, bin_lo, bin_hi, bin_method)

    annotation_cover = label[ys, xs]
    annotation_idx = - np.ones((ys.size, label.shape[-1]), dtype=int)
    curr_idx = 1
    for cls_ in range(label.shape[-1]):
        label_cls_ = label[..., cls_]
        match label_components(label_cls_ > 0, background=0, return_num=True):
            case components, num:
                pass
            case _:
                raise RuntimeError(f"unexpected return value from label_component: {_}")
        for group_idx in range(1, num + 1):
            mask = np.asarray(components == group_idx, dtype=int)
            assert mask.any()
            flat_mask = mask[ys, xs] != 0
            annotation_idx[flat_mask, cls_] =  curr_idx
            curr_idx += 1

    return Tabular(
        dataset_x=spectra,
        annotation_overlap=annotation_cover,
        annotation_idx=annotation_idx,
        bin_l=bin_l,
        bin_r=bin_r,
        regions=None,
        coord_y=ys,
        coord_x=xs,
    )


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
    ann_cover: list[np.ndarray] = []
    ann_idx: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    coords_y: list[np.ndarray] = []
    coords_x: list[np.ndarray] = []

    g_offset = 0
    for idx, ds_ in enumerate(datasets):
        ds_x.append(ds_.dataset_x)
        ann_cover.append(ds_.annotation_overlap)

        if ds_.annotation_idx[ds_.annotation_idx != -1].min() != 1:
            raise ValueError(f"invalid min found in {ds_.annotation_idx}")
        copy = ds_.annotation_idx.copy()
        copy[copy != -1] += g_offset
        ann_idx.append(copy)
        g_offset = copy.max()

        regions.append(np.zeros_like(copy) + idx)
        coords_y.append(ds_.coord_y)
        coords_x.append(ds_.coord_x)

    return Tabular(
        np.concatenate(ds_x),
        np.concatenate(ann_cover),
        np.concatenate(ann_idx),
        bin_l,
        bin_r,
        np.concatenate(regions),
        np.concatenate(coords_y),
        np.concatenate(coords_x),
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

bin_method: Literal["sum", "integration"] = "sum"

merged_ds = merge_tabular(
    *[
        make_tabular(ozm, lbl, merged_l, merged_r, bin_method)
        for ozm, lbl in zip(ozm_datasets.values(), lbl_dct.values(), strict=True)
    ]
)

# %%

try:
    # detect if the cell is ran in a notebook to avoid saving by mistake with run all
    get_ipython()  # type:ignore
    val = input("do you want to save? [y/n]")
    if val.lower() in ["y", "yes"]:
        raise NameError("go to the except clause")
except NameError:
    save_tabular(
        datasets_dir / f"slim-deisotoping-{de_tol:.1e}-binned" / f"binned_{bin_method}_{tag}.npz",
        merged_ds,
    )

# %%
