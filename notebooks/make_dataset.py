# %%

import pathlib
from typing import NamedTuple

import numpy as np
from omezarrmsi import OMEZarrMSI

from skimage.measure import label as label_components

import sys

sys.path.insert(0, str((pathlib.Path(__file__).parent.parent).resolve()))

from msi_zarr_analysis.preprocessing.binning import bin_and_flatten_v2  # noqa: E402
from msi_zarr_analysis.cli.utils import bins_from_csv  # noqa: E402


# %%

# pkg_dir = pathlib.Path(__file__).parent.parent
home_ds_dir = pathlib.Path("/home/maxime/datasets/COMULIS-slim-msi")

ds_name_lst = [
    # "nonslim_nonorm",
    # "nonslim_317norm",
    "slim_nonorm",
    "slim_317norm",
]
all_version_ms_dict = {
    "r13": (
        # OMEZarrMSI(pkg_dir / "tmp_r13.zarr", mode="r"),
        # OMEZarrMSI(pkg_dir / "tmp_r13_n317.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region13_nonorm_sample.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region13_317norm_sample.zarr", mode="r"),
    ),
    "r14": (
        # OMEZarrMSI(pkg_dir / "tmp_r14.zarr", mode="r"),
        # OMEZarrMSI(pkg_dir / "tmp_r14_n317.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region14_nonorm_sample.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region14_317norm_sample.zarr", mode="r"),
    ),
    "r15": (
        # OMEZarrMSI(pkg_dir / "tmp_r15.zarr", mode="r"),
        # OMEZarrMSI(pkg_dir / "tmp_r15_n317.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region15_nonorm_sample.zarr", mode="r"),
        OMEZarrMSI(home_ds_dir / "region15_317norm_sample.zarr", mode="r"),
    ),
}

# %%

ds_idx = 1
ds_name = ds_name_lst[ds_idx] + "_v2"

label_keys = ["ls+", "ls-", "sc+", "sc-"]
ds_dct = {key: tpl[ds_idx] for key, tpl in all_version_ms_dict.items()}
lbl_dct = {
    region_: np.stack([ds_.get_label(label_)[0, 0] for label_ in label_keys], axis=-1)
    for region_, ds_ in ds_dct.items()
}

# %%

# how many YX pixels have more than one label ?
{k: np.count_nonzero(v.sum(axis=-1) != v.max(axis=-1)) for k, v in lbl_dct.items()}

# %%

# how many YX pixels have at least one label ?
{k: np.count_nonzero(v.max(axis=-1)) for k, v in lbl_dct.items()}

# %%


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    dataset_y: np.ndarray
    groups: np.ndarray
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

    return Tabular(dataset_x, dataset_y, groups, None)


def merge_tabular(*datasets: Tabular):
    if any(ds_.regions is not None for ds_ in datasets):
        raise ValueError("only non-merged datasets are allowed")

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
        np.concatenate(regions),
    )


# %%

# bin_ticks = np.linspace(300.0, 305.0, 10)
# bin_lo = bin_ticks[:-1]
# bin_hi = bin_ticks[1:]
bin_lo, bin_hi = bins_from_csv("../mz value + lipid name.csv")

merged_ds = merge_tabular(
    *[
        make_tabular(omz, lbl, bin_lo, bin_hi)
        for omz, lbl in zip(ds_dct.values(), lbl_dct.values())
    ]
)

# %%


def save_tabular(path: pathlib.Path, ds: Tabular):
    assert ds.regions is not None
    # bug in PyLance: *ds does not mean a tuple of ndarray, but laid_out does...
    laid_out = (ds.dataset_x, ds.dataset_y, ds.groups, ds.regions)
    assert all((l == r).all() for (l, r) in zip(ds, laid_out, strict=True))
    np.savez(path, *laid_out)


# %%

destination = pathlib.Path("/home/maxime/repos/msi_zarr_analysis/datasets")
assert destination.is_dir()
save_tabular(destination / f"saved_msi_merged_{ds_name}.npz", merged_ds)

# %%

# loaded_ds = Tabular(*np.load(destination / f"saved_msi_merged_{ds_name}.npz").values())

# for arr_l, arr_r in zip(loaded_ds, merged_ds):
#     print(f"{(arr_l != arr_r).any()=}")

# %%
