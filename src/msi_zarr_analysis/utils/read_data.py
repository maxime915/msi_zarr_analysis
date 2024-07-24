"utility to read part of the data from OME-Zarr MS group to a Numpy array"

import numpy as np
import numba as nb
import numpy.typing as npt
import zarr

from .iter_chunks import clean_slice_tuple, iter_loaded_chunks


@nb.jit(nopython=True, parallel=True)
def bin_processed_chunk(
    output: npt.NDArray,
    ints: npt.NDArray,
    mzs: npt.NDArray,
    lengths: npt.NDArray,
    mz_lo: float,
    mz_hi: float,
) -> None:

    idx_y, idx_x = lengths.nonzero()
    count = idx_y.size

    for idx in nb.prange(count):

        y, x = idx_y[idx], idx_x[idx]

        len_band = lengths[y, x]
        mz_band = mzs[:len_band, y, x]

        idx_lo = np.searchsorted(mz_band, mz_lo, side="left")
        idx_hi = np.searchsorted(mz_band, mz_hi, side="right")
        
        output[y, x] = np.sum(ints[idx_lo:idx_hi, y, x])


def read_yx_slice(
    z_group: zarr.Group,
    mz_low: float,
    mz_high: float,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> npt.NDArray:
    """read a slice of YX data with mz from mz_low to mz_high, assuming z=0

    continuous-file optimization TODO

    Parameters
    ----------
    z_group : zarr.Group
        _description_
    mz_low : float
        _description_
    mz_high : float
        _description_
    y_slice : slice
        _description_
    x_slice : slice
        _description_
    """

    z_ints = z_group["/0"]
    z_mzs = z_group["/labels/mzs/0"]
    z_lengths = z_group["/labels/lengths/0"]
    
    y_slice, x_slice = clean_slice_tuple(z_ints.shape[2:], y_slice, x_slice)

    reduced = np.zeros(
        shape=z_ints.shape[2:],
        dtype=z_ints.dtype,
    )

    # load chunks
    for cy, cx in iter_loaded_chunks(z_ints, y_slice, x_slice, skip=2):
        # write buffer
        c_dest = reduced[cy, cx]
        
        c_lengths = z_lengths[0, 0, cy, cx]
    
        # small optimization for uneven spectra (?)
        len_cap = c_lengths.max()

        # load from disk
        c_ints = z_ints[:len_cap, 0, cy, cx]
        c_mzs = z_mzs[:len_cap, 0, cy, cx]

        bin_processed_chunk(
            c_dest,
            c_ints,
            c_mzs,
            c_lengths,
            mz_low,
            mz_high,
        )
    
    return reduced
