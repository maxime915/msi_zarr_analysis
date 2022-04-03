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
    output: npt.NDArray,
    ints: npt.NDArray,
    mzs: npt.NDArray,
    lengths: npt.NDArray,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
) -> None:

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
    z: zarr.Group,
    destination: zarr.Group,
    y_slice: slice,
    x_slice: slice,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
) -> npt.NDArray:

    z_ints = z["/0"]
    z_mzs = z["/labels/mzs/0"]
    z_lengths = z["/labels/lengths/0"]

    destination["/0"] = zarr.zeros(
        shape=(len(bin_lo),) + z_ints.shape[1:],
        chunks=z_ints.chunks,
        compressor=z_ints.compressor,
        order=z_ints.order,
    )
    destination["/labels/mzs/0"] = np.reshape((bin_lo + bin_hi) / 2, (-1, 1, 1, 1))
    destination["/labels/lengths/0"] = len(bin_lo) * (z_lengths[:] > 0)

    z_dest_int = destination["/0"]

    # load chunks
    for cz, cy, cx in iter_loaded_chunks(z_ints, slice(None), y_slice, x_slice, skip=1):
        # load from disk

        # allocate an array of the right size for c_dest
        c_dest = np.zeros(
            shape=(
                len(bin_lo),
                cz.stop - cz.start,
                cy.stop - cy.start,
                cx.stop - cx.start,
            )
        )

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

        z_dest_int[:, cz, cy, cx] = c_dest


def bin_processed_lo_hi(
    z_path: str,
    destination_path: str,
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
):
    z = open_group_ro(z_path)

    destination = zarr.open_group(destination_path, mode="w-")

    return _bin_processed(
        z,
        destination,
        *clean_slice_tuple(z["/0"].shape[2:], y_slice, x_slice),
        bin_lo,
        bin_hi
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
