"mz_slice: extract single lipids"

import numpy as np
import numpy.typing as npt
import numba as nb
import zarr
from msi_zarr_analysis.utils.check import open_group_ro

from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks, clean_slice_tuple


@nb.jit(nopython=True)
def bin_band(bins, mzs, ints, bin_lo, bin_hi, reduction):

    lows = np.searchsorted(mzs, bin_lo, side="left")
    highs = np.searchsorted(mzs, bin_hi, side="right")

    for idx in range(lows.size):
        lo = lows[idx]
        hi = highs[idx]

        bins[idx] = reduction(ints[lo:hi])


@nb.jit(nopython=True, parallel=True)
def bin_processed_chunk(
    output: npt.NDArray,  # (bins, z, y, x)
    ints: npt.NDArray,  # (max-chan, z, y, x)
    mzs: npt.NDArray,  # (max-chan, z, y, x)
    lengths: npt.NDArray,  # (max-chan, z, y, x)
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
) -> None:
    "bin a chunk of [c, z, y, x] to [c', z, y, x]"

    idx_z, idx_y, idx_x = lengths.nonzero()
    count = idx_z.size

    for idx in nb.prange(count):

        z, y, x = idx_z[idx], idx_y[idx], idx_x[idx]

        len_band = lengths[z, y, x]
        mz_band = mzs[:len_band, z, y, x]
        int_band = ints[:len_band, z, y, x]

        bin_band(
            output[:, z, y, x],
            mz_band,
            int_band,
            bin_lo,
            bin_hi,
            reduction=np.sum,
        )


def _bin_processed(
    z_ints: zarr.Array,  # (max-chan, z, y, x)
    z_mzs: zarr.Array,  # (max-chan, z, y, x)
    z_lengths: zarr.Array,  # (max-chan, z, y, x)
    z_dest_ints: zarr.Array,  # (bins, z, y, x)
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
):
    "bin an processed array of [c, z, y, x] to [c', z, y, x]"

    # load chunks
    for cz, cy, cx in iter_loaded_chunks(z_ints, slice(None), y_slice, x_slice, skip=1):
        # allocate an array of the right size for c_dest
        c_dest = np.zeros(
            shape=(
                len(bin_lo),
                cz.stop - cz.start,
                cy.stop - cy.start,
                cx.stop - cx.start,
            )
        )

        # load from disk
        c_ints = z_ints[:, cz, cy, cx]
        c_mzs = z_mzs[:, cz, cy, cx]
        c_lengths = z_lengths[0, cz, cy, cx]

        bin_processed_chunk(
            c_dest,
            c_ints,
            c_mzs,
            c_lengths,
            bin_lo,
            bin_hi,
        )

        # write to disk
        z_dest_ints[:, cz, cy, cx] = c_dest


@nb.jit(nopython=True, parallel=True)
def bin_and_flatten_chunk(
    dataset_x: npt.NDArray,  # (rows, bins)
    dataset_y: npt.NDArray,  # (rows,)
    ints: npt.NDArray,  # (max-chan, y, x)
    mzs: npt.NDArray,  # (max-chan, y, x)
    lengths: npt.NDArray,  # (max-chan, y, x)
    cls: npt.NDArray,  # (y, x, classes)
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
    start_idx: int,
) -> int:
    "bin and flatten a processed array"

    idx_y, idx_x, idx_c = cls.nonzero()

    count = idx_y.size

    for idx in nb.prange(count):
        row_offset = start_idx + idx

        y, x, c = idx_y[idx], idx_x[idx], idx_c[idx]

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

        dataset_y[row_offset] = c

    return start_idx + count


def bin_and_flatten(
    dataset_x: npt.NDArray,  # (rows, bins)
    dataset_y: npt.NDArray,  # (rows,)
    z: zarr.Group,
    onehot_cls: npt.NDArray,  # (y, x, classes)
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
) -> None:

    z_ints = z["/0"]
    z_mzs = z["/labels/mzs/0"]
    z_lengths = z["/labels/lengths/0"]

    row_idx = 0

    # load chunks
    for cy, cx in iter_loaded_chunks(z_ints, y_slice, x_slice, skip=2):
        # load from disk
        c_ints = z_ints[:, 0, cy, cx]
        c_mzs = z_mzs[:, 0, cy, cx]
        c_lengths = z_lengths[0, 0, cy, cx]

        c_cls = onehot_cls[cy, cx]

        row_idx = bin_and_flatten_chunk(
            dataset_x,
            dataset_y,
            c_ints,
            c_mzs,
            c_lengths,
            c_cls,
            bin_lo,
            bin_hi,
            row_idx,
        )

    assert row_idx == dataset_x.shape[0], f"{row_idx=} mismatch for {dataset_x.shape=}"


def flatten_chunk(
    dataset_x: npt.NDArray,  # (rows, attrs)
    dataset_y: npt.NDArray,  # (rows,)
    ints: npt.NDArray,  # (attrs, y, x)
    cls: npt.NDArray,  # (y, x, classes)
    start_idx: int,
) -> int:
    "flatten a chunk of a continuous array"

    idx_y, idx_x, idx_c = cls.nonzero()

    for k, (y, x, c) in enumerate(zip(idx_y, idx_x, idx_c), start=start_idx):
        dataset_x[k, :] = ints[:, y, x]
        dataset_y[k] = c

    return start_idx + idx_c.size


def flatten(
    dataset_x: npt.NDArray,  # (rows, attrs)
    dataset_y: npt.NDArray,  # (rows,)
    z: zarr.Group,
    onehot_cls: npt.NDArray,  # (y, x, classes)
    y_slice: slice,
    x_slice: slice,
) -> None:
    "flatten a continuous array"

    z_ints = z["/0"]

    row_idx = 0

    # load chunks
    for cy, cx in iter_loaded_chunks(z_ints, y_slice, x_slice, skip=2):
        # load from disk
        c_ints = z_ints[:, 0, cy, cx]

        c_cls = onehot_cls[cy, cx]

        row_idx = flatten_chunk(
            dataset_x,
            dataset_y,
            c_ints,
            c_cls,
            row_idx,
        )

    assert row_idx == dataset_x.shape[0], f"{row_idx=} mismatch for {dataset_x.shape=}"


def bin_processed_lo_hi(
    z_path: str,
    destination_path: str,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
):
    z = open_group_ro(z_path)

    z_ints = z["/0"]
    z_mzs = z["/labels/mzs/0"]
    z_lengths = z["/labels/lengths/0"]

    destination = zarr.open_group(destination_path, mode="w-")

    destination["/0"] = zarr.zeros(
        shape=(len(bin_lo),) + z_ints.shape[1:],
        chunks=z_ints.chunks,
        compressor=z_ints.compressor,
        order=z_ints.order,
    )
    destination["/labels/mzs/0"] = np.reshape((bin_lo + bin_hi) / 2, (-1, 1, 1, 1))
    destination["/labels/lengths/0"] = len(bin_lo) * (z_lengths[:] > 0)

    z_dest_ints = destination["/0"]

    # TODO add description of transformation
    for key, value in z.attrs.items():
        destination.attrs[key] = value
    if "pims-msi" not in destination.attrs:
        destination.attrs["pims-msi"] = {}
    bkp = destination.attrs["pims-msi"]
    bkp["binary_mode"] = "continuous"
    destination.attrs["pims-msi"] = bkp

    return _bin_processed(
        z_ints,
        z_mzs,
        z_lengths,
        z_dest_ints,
        *clean_slice_tuple(z["/0"].shape[2:], y_slice, x_slice),
        bin_lo,
        bin_hi,
    )


def bin_processed_val_tol(
    z_path: str,
    destination_path: str,
    values: npt.NDArray,
    tolerances: npt.NDArray,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
):

    bin_lo = values - tolerances
    bin_hi = values + tolerances

    return bin_processed_lo_hi(
        z_path, destination_path, bin_lo, bin_hi, y_slice, x_slice
    )
