import time
import warnings
from typing import Tuple, Optional, Dict

import numba as nb
import numpy as np
import numpy.typing as npt
import zarr

from msi_zarr_analysis.preprocessing.binning import flatten, bin_and_flatten


def masked_size(
    group: zarr.Group,
    onehot_cls: npt.NDArray[np.dtype("bool")],
    y_slice: slice,
    x_slice: slice,
) -> Tuple[int, int]:

    intensities_itemsize = group["/0"].dtype.itemsize
    lengths = group["/labels/lengths/0"][0, 0, y_slice, x_slice]

    onehot_cls_ = onehot_cls[y_slice, x_slice]

    band_count = onehot_cls_.sum(axis=-1)
    element_count = (lengths * band_count).sum()

    return band_count.sum(), intensities_itemsize * element_count


def check_roi_mask(
    roi_mask: Optional[npt.NDArray[np.dtype("bool")]],
    valid_coordinates: npt.NDArray[np.dtype("bool")],
) -> npt.NDArray[np.dtype("bool")]:

    if not roi_mask:
        return valid_coordinates

    # is this useful ? only if manually selected... -> roi_mask need to be optional
    for (y, x) in zip(*np.nonzero(np.logical_and(roi_mask, ~valid_coordinates))):
        warnings.warn(f"label at ({x=}, {y=}) does not match any spectrum")

    return np.logical_and(roi_mask, valid_coordinates)


def build_class_masks(
    z: zarr.Group,
    cls_dict: Dict[str, npt.NDArray[np.dtype("bool")]],
    roi_mask: Optional[npt.NDArray[np.dtype("bool")]] = None,
    append_background_cls: bool = False,
) -> Tuple[npt.NDArray, npt.NDArray, Tuple[str]]:
    valid_coordinates = z["/labels/lengths/0"][0, 0, ...] > 0

    if not roi_mask:
        roi_mask = valid_coordinates
    else:
        roi_mask = check_roi_mask(roi_mask, valid_coordinates)

    names, masks = zip(*cls_dict.items())

    for mask in masks:
        # check invalid selection
        dead_pixels = sum(mask[~valid_coordinates])
        if dead_pixels:
            warnings.warn(f"found {dead_pixels} labelled pixels without any data")

        # remove non ROI
        mask &= roi_mask

    if append_background_cls:
        if "background" in cls_dict:
            raise ValueError(f"{cls_dict.keys()} already contains 'background'")

        neg_bg = np.zeros_like(roi_mask)
        for mask in masks:
            neg_bg |= mask

        masks += (roi_mask & ~neg_bg,)
        names += ("background",)

    return np.stack(masks, axis=-1), roi_mask, names


def bin_array_dataset(
    z: zarr.Group,
    onehot_cls: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:

    # estimate the size of the dataset to know if it will fit in memory
    rows, data_size = masked_size(z, onehot_cls, y_slice, x_slice)
    if data_size > 8 * 2**30:
        raise RuntimeError(
            (
                f"estimated {data_size = } (approx. "
                f"{data_size/2**30} GiB) is too large to fit into"
                " the main memory, aborting"
            )
        )
    if data_size > 4 * 2**30:
        warnings.warn(
            f"estimated {data_size = } (approx. "
            f"{data_size/2**30} GiB) is quite large"
        )

    if len(bin_lo) != len(bin_hi):
        raise ValueError(f"inconsistent bin sizes: {bin_lo} VS {bin_hi}")
    if len(bin_lo) < 2:
        raise ValueError(f"not enough bins, expected > 1, found {len(bin_lo)}")
    if any(lo >= hi for lo, hi in zip(bin_lo, bin_hi)):
        raise ValueError(f"inconsistent bin low/high: {bin_lo=} {bin_hi=}")

    n_bins = len(bin_lo)

    # build the dataset
    dataset_x = np.zeros(shape=(rows, n_bins))
    dataset_y = np.zeros(shape=(rows,))

    if z.attrs["pims-msi"]["binary_mode"] != "processed":
        raise ValueError("only 'processed' type is supported in binned mode")
        # warnings.warn("optimizations for 'continuous' files are not implemented yet")

    print("starting to read data from disk...")

    start_time = time.time()
    bin_and_flatten(
        dataset_x=dataset_x,
        dataset_y=dataset_y,
        z=z,
        onehot_cls=onehot_cls,
        y_slice=y_slice,
        x_slice=x_slice,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
    )
    elapsed_time = time.time() - start_time

    print(f"done reading data from disk ! {elapsed_time:.3f}")

    return dataset_x, dataset_y


def nonbinned_array_dataset(
    z: zarr.Group,
    onehot_cls: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
) -> Tuple[npt.NDArray, npt.NDArray]:

    # estimate the size of the dataset to know if it will fit in memory
    rows, data_size = masked_size(z, onehot_cls, y_slice, x_slice)
    if data_size > 8 * 2**30:
        raise RuntimeError(
            (
                f"estimated {data_size = } (approx. "
                f"{data_size/2**30} GiB) is too large to fit into"
                " the main memory, aborting"
            )
        )
    if data_size > 4 * 2**30:
        warnings.warn(
            f"estimated {data_size = } (approx. "
            f"{data_size/2**30} GiB) is quite large"
        )

    n_attributes = z["/labels/mzs/0"].shape[0]

    # build the dataset
    dataset_x = np.zeros(shape=(rows, n_attributes))
    dataset_y = np.zeros(shape=(rows,))

    if z.attrs["pims-msi"]["binary_mode"] != "continuous":
        raise ValueError("only 'continuous' type is supported in non-binned mode")

    print("starting to read data from disk...")

    start_time = time.time()
    flatten(
        dataset_x=dataset_x,
        dataset_y=dataset_y,
        z=z,
        onehot_cls=onehot_cls,
        y_slice=y_slice,
        x_slice=x_slice,
    )
    elapsed_time = time.time() - start_time

    print(f"done reading data from disk ! {elapsed_time:.3f}")

    return dataset_x, dataset_y
