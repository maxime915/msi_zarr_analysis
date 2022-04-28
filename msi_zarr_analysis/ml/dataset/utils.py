import time
import warnings
from typing import Tuple, Optional, Dict

import numba as nb
import numpy as np
import numpy.typing as npt
import zarr

from msi_zarr_analysis.utils.iter_chunks import iter_nd_chunks


def masked_size(
    group: zarr.Group,
    mask: npt.NDArray[np.dtype("bool")],
    y_slice: slice,
    x_slice: slice,
) -> Tuple[int, int]:

    intensities_itemsize = group["/0"].dtype.itemsize
    lengths = group["/labels/lengths/0"][0, 0, y_slice, x_slice]
    row_lengths = lengths[mask[y_slice, x_slice] > 0]
    element_count = row_lengths.sum()

    return row_lengths.size, intensities_itemsize * element_count


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
):
    valid_coordinates = z["/labels/lengths/0"][0, 0, ...] > 0

    if not roi_mask:
        roi_mask = valid_coordinates
    else:
        roi_mask = check_roi_mask(roi_mask, valid_coordinates)

    cls_mask = len(cls_dict) * np.ones_like(roi_mask, dtype=int)
    classes_name = []

    # build all classes, assuming background class=0
    for idx, (cls_name_, cls_mask_) in enumerate(cls_dict.items()):
        # warn about overlap ?
        cls_mask[cls_mask_ > 0] = idx
        classes_name.append(cls_name_)

    if not append_background_cls:
        # removed background pixels from the ROI
        bg_mask = cls_mask == len(cls_dict)
        roi_mask[bg_mask] = 0
        cls_mask[bg_mask] = -len(cls_dict)
    else:
        classes_name.append("background")

    return cls_mask, roi_mask, classes_name


@nb.jit(nopython=True)
def bin_band(bins, mzs, ints, bin_lo, bin_hi, reduction):

    lows = np.searchsorted(mzs, bin_lo, side="left")
    highs = np.searchsorted(mzs, bin_hi, side="right")

    for idx in range(lows.size):
        lo = lows[idx]
        hi = highs[idx]

        bins[idx] = reduction(ints[lo:hi])


@nb.jit(nopython=True, parallel=True)
def fill_binned_dataset_processed_chunk(
    dataset_x: npt.NDArray,
    dataset_y: npt.NDArray,
    ints: npt.NDArray,
    mzs: npt.NDArray,
    lengths: npt.NDArray,
    roi: npt.NDArray,
    cls: npt.NDArray,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
    start_idx: int,
) -> int:

    idx_y, idx_x = roi.nonzero()
    count = idx_y.size

    for idx in nb.prange(count):
        row_offset = start_idx + idx

        y, x = idx_y[idx], idx_x[idx]

        len_band = lengths[y, x]
        mz_band = mzs[:len_band, y, x]
        int_band = ints[:len_band, y, x]

        bin_band(
            dataset_x[row_offset],
            mz_band,
            int_band,
            bin_lo,
            bin_hi,
            reduction=np.sum,
        )

        dataset_y[row_offset] = cls[y, x]

    return start_idx + count


def fill_nonbinned_dataset_continuous_chunk(
    dataset_x: npt.NDArray,
    dataset_y: npt.NDArray,
    ints: npt.NDArray,
    roi: npt.NDArray,
    cls: npt.NDArray,
    start_idx: int,
) -> int:

    low = start_idx
    high = start_idx + np.count_nonzero(roi)

    dataset_x[low:high, :] = ints[:, roi].T
    dataset_y[low:high] = cls[roi].T

    return high


def fill_binned_dataset_processed(
    dataset_x: npt.NDArray,
    dataset_y: npt.NDArray,
    z: zarr.Group,
    roi_mask: npt.NDArray[np.dtype("bool")],
    cls_mask: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
) -> None:

    z_ints = z["/0"]
    z_mzs = z["/labels/mzs/0"]
    z_lengths = z["/labels/lengths/0"]

    row_idx = 0

    # load chunks
    for cy, cx in iter_nd_chunks(z_ints, y_slice, x_slice, skip=2):
        # load from disk
        c_ints = z_ints[:, 0, cy, cx]
        c_mzs = z_mzs[:, 0, cy, cx]
        c_lengths = z_lengths[0, 0, cy, cx]

        c_roi = roi_mask[cy, cx]
        c_pos = cls_mask[cy, cx]

        row_idx = fill_binned_dataset_processed_chunk(
            dataset_x,
            dataset_y,
            c_ints,
            c_mzs,
            c_lengths,
            c_roi,
            c_pos,
            bin_lo,
            bin_hi,
            row_idx,
        )

    if row_idx != dataset_x.shape[0]:
        print(f"{row_idx=} mismatch for {dataset_x.shape=}")


def fill_nonbinned_dataset_continuous(
    dataset_x: npt.NDArray,
    dataset_y: npt.NDArray,
    z: zarr.Group,
    roi_mask: npt.NDArray[np.dtype("bool")],
    cls_mask: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
) -> None:

    z_ints = z["/0"]

    row_idx = 0

    # load chunks
    for cy, cx in iter_nd_chunks(z_ints, y_slice, x_slice, skip=2):
        # load from disk
        c_ints = z_ints[:, 0, cy, cx]

        c_roi = roi_mask[cy, cx]
        c_pos = cls_mask[cy, cx]

        row_idx = fill_nonbinned_dataset_continuous_chunk(
            dataset_x,
            dataset_y,
            c_ints,
            c_roi,
            c_pos,
            row_idx,
        )

    if row_idx != dataset_x.shape[0]:
        print(f"{row_idx=} mismatch for {dataset_x.shape=}")


def bin_array_dataset(
    z: zarr.Group,
    roi_mask: npt.NDArray[np.dtype("bool")],
    cls_mask: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:

    # estimate the size of the dataset to know if it will fit in memory
    rows, data_size = masked_size(z, roi_mask, y_slice, x_slice)
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
    fill_binned_dataset_processed(
        dataset_x,
        dataset_y,
        z,
        roi_mask,
        cls_mask,
        y_slice,
        x_slice,
        bin_lo,
        bin_hi,
    )
    elapsed_time = time.time() - start_time

    print(f"done reading data from disk ! {elapsed_time:.3f}")

    return dataset_x, dataset_y


def nonbinned_array_dataset(
    z: zarr.Group,
    roi_mask: npt.NDArray[np.dtype("bool")],
    cls_mask: npt.NDArray[np.dtype("int")],
    y_slice: slice,
    x_slice: slice,
) -> Tuple[npt.NDArray, npt.NDArray]:

    # estimate the size of the dataset to know if it will fit in memory
    rows, data_size = masked_size(z, roi_mask, y_slice, x_slice)
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
    fill_nonbinned_dataset_continuous(
        dataset_x,
        dataset_y,
        z,
        roi_mask,
        cls_mask,
        y_slice,
        x_slice,
    )
    elapsed_time = time.time() - start_time

    print(f"done reading data from disk ! {elapsed_time:.3f}")

    return dataset_x, dataset_y
