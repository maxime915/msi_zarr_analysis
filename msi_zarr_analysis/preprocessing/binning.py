"mz_slice: extract single lipids"

from functools import partial
from typing import Dict, Tuple, TypeVar, cast
import numpy as np
import numpy.typing as npt
import numba as nb
import zarr
from msi_zarr_analysis.utils.check import open_group_ro

from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks, clean_slice_tuple


@nb.jit(nopython=True)
def bin_band(bins, mzs, ints, bin_lo, bin_hi):

    lows = np.searchsorted(mzs, bin_lo, side="left")
    highs = np.searchsorted(mzs, bin_hi, side="right")

    for idx in range(lows.size):
        lo = lows[idx]
        hi = highs[idx]

        bins[idx] = np.sum(ints[lo:hi])


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
    lengths: npt.NDArray,  # (y, x)
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
        )

        dataset_y[row_offset] = c

    return start_idx + count


def bin_and_flatten_chunk_v2(
    dataset_x: npt.NDArray,  # (rows, bins)
    ints: npt.NDArray,  # (max-chan, y, x)
    mzs: npt.NDArray,  # (max-chan, y, x)
    lengths: npt.NDArray,  # (y, x)
    mask: npt.NDArray[np.bool_],  # (y, x)
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
    start_idx: int,
):
    "bin and flatten a processed array"

    # only consider (y, x) coordinates valid wrt the mask
    idx_y, idx_x = mask.nonzero()

    for row_offset, (y, x) in enumerate(
        zip(idx_y, idx_x, strict=True),
        start=start_idx,
    ):
        _len = lengths[y, x]

        bin_band(
            dataset_x[row_offset],
            mzs[:_len, y, x],
            ints[:_len, y, x],
            bin_lo,
            bin_hi,
        )

    return start_idx + len(idx_y), idx_y, idx_x


KeyType = TypeVar("KeyType")


def bin_spectrum_dict(
    spectrum_dict: Dict[KeyType, Tuple[np.ndarray, np.ndarray]],
    bin_lo: npt.NDArray,  # (bins,)
    bin_hi: npt.NDArray,  # (bins,)
) -> Dict[KeyType, np.ndarray]:
    "NOTE: to save memory, spectrum_dict will be consumed"

    binned_spectrum_dict = {}
    while spectrum_dict:
        key, (s_mzs, s_int) = spectrum_dict.popitem()

        binned_spectra = np.zeros(shape=bin_lo.shape, dtype=s_int.dtype)
        bin_band(binned_spectra, s_mzs, s_int, bin_lo, bin_hi)

        binned_spectrum_dict[key] = binned_spectra

    return binned_spectrum_dict


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


def bin_and_flatten_v2(
    dataset_x: npt.NDArray[np.float64],  # (rows, bins)
    z: zarr.Group,
    mask: npt.NDArray[np.bool_],  # (y, x)
    bin_lo: npt.NDArray[np.float64],  # (bins,)
    bin_hi: npt.NDArray[np.float64],  # (bins,)
):

    z_ints = cast(zarr.Array, z["/0"])
    z_mzs = cast(zarr.Array, z["/labels/mzs/0"])
    z_lengths = cast(zarr.Array, z["/labels/lengths/0"])
    row_idx = 0

    # shape checks
    n_rows, n_bins = dataset_x.shape
    if bin_lo.shape != (n_bins,):
        raise ValueError(f"feat mismatch: {dataset_x.shape=} but {bin_lo.shape=}")
    if bin_hi.shape != (n_bins,):
        raise ValueError(f"feat mismatch: {dataset_x.shape=} but {bin_hi.shape=}")
    if np.count_nonzero(mask) != n_rows:
        raise ValueError(f"{np.count_nonzero(mask)=} but {n_rows=}")

    ys = -1 * np.ones((n_rows,), np.int64)
    xs = -1 * np.ones((n_rows,), np.int64)

    # load chunks
    for cy, cx in iter_loaded_chunks(z_ints, skip=2):
        new_row_idx, c_idx_y, c_idx_x = bin_and_flatten_chunk_v2(
            dataset_x,
            z_ints[:, 0, cy, cx],
            z_mzs[:, 0, cy, cx],
            z_lengths[0, 0, cy, cx],
            mask[cy, cx],
            bin_lo,
            bin_hi,
            row_idx,
        )
        ys[row_idx:new_row_idx] = c_idx_y + cy.start
        xs[row_idx:new_row_idx] = c_idx_x + cx.start
        row_idx = new_row_idx

    assert row_idx == dataset_x.shape[0], f"{row_idx=} mismatch for {dataset_x.shape=}"
    return ys, xs


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

    z_ints = cast(zarr.Array, z["/0"])

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

    z_ints = cast(zarr.Array, z["/0"])
    z_mzs = cast(zarr.Array, z["/labels/mzs/0"])
    z_lengths = cast(zarr.Array, z["/labels/lengths/0"])

    destination = zarr.open_group(destination_path, mode="w-")

    # create empty array
    destination.create(
        "/0",
        shape=(len(bin_lo),) + cast(tuple[int, int, int], z_ints.shape[1:]),
        chunks=z_ints.chunks,
        compressor=z_ints.compressor,
        order=z_ints.order,
    )
    destination["/labels/mzs/0"] = np.reshape((bin_lo + bin_hi) / 2, (-1, 1, 1, 1))
    destination["/labels/lengths/0"] = len(bin_lo) * (cast(np.ndarray, z_lengths[:]) > 0)

    z_dest_ints = cast(zarr.Array, destination["/0"])

    # TODO add description of transformation
    for key, value in z.attrs.items():
        destination.attrs[key] = value
    if "pims-msi" not in destination.attrs:
        destination.attrs["pims-msi"] = {}
    bkp = destination.attrs["pims-msi"]
    bkp["binary_mode"] = "continuous"
    destination.attrs["pims-msi"] = bkp

    y_slice, x_slice = clean_slice_tuple(
        cast(tuple[int, int], z["/0"].shape[2:]), y_slice, x_slice
    )
    return _bin_processed(
        z_ints,
        z_mzs,
        z_lengths,
        z_dest_ints,
        y_slice,
        x_slice,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
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


@nb.jit(signature_or_function="f8(f8, f8, i8, f8)", nopython=True)
def _fn_target(mz_lo: float, mz_hi: float, num: int, f: float):
    "> 0 => f is too high, < 0 => f is too low, may return non finite values"

    mz = mz_lo
    for _ in range(num):
        dm = f * mz ** 2
        mz += dm

    return mz - mz_hi


@nb.jit("void(f8[:], f8, f8)", nopython=True)
def _fill_binning(mz_: np.ndarray, mz_lo: float, f: float):
    mz_[0] = mz_lo
    for i in range(1, mz_.shape[0]):
        dm = f * (mz_[i - 1]) ** 2
        mz_[i] = mz_[i - 1] + dm


def fticr_binning(mz_lo: float, mz_hi: float, num: int, tol: float | None = None):
    """Assume a mass resolution inversely proportional to the square of the mass
    to generate bins for the given interval.

    Args:
        mz_lo (float): lowest value of the interval
        mz_hi (float): highest value of the interval
        num (int): number of bins
        tol (float, opt.): tolerance such that `0 <= bin_right[-1] - mz_hi <= tol`
        the default (None) uses (mz_hi - mz_lo) / num

    Returns:
        np.ndarray: left edges of each bin (the first item is mz_lo)
        np.ndarray: right edges of each bin (the last item is g.e.t. mz_hi)
        float: the estimated proportionality constant
    """

    if tol is None:
        tol = (mz_hi - mz_lo) / num
    target = partial(_fn_target, mz_lo=mz_lo, mz_hi=mz_hi, num=num)

    # find a first non-inf value
    f0 = 1.0 / mz_lo / num
    while True:
        v0 = target(f=f0)
        if not np.isinf(v0):
            break
        f0 *= 0.5

    # find a high bound
    f_hi = f0
    v_hi = v0
    mul = 1.5
    while v_hi < 0:
        tmp = (f_hi * mul, target(f=f_hi * mul))
        if np.isinf(tmp[1]):
            mul = mul ** (1 / 3)
        else:
            f_hi, v_hi = tmp

    # find a low bound
    f_lo = f0
    v_lo = v0
    mul = 0.7
    while v_lo > 0:
        f_lo *= mul  # no need to protect against inf here
        v_lo = target(f=f_lo)

    # binary search
    while True:
        f_next = 0.5 * (f_lo + f_hi)
        v_next = target(f=f_next)

        if v_next > tol:
            f_hi = f_next
        elif v_next < 0:  # don't accept low values, even if very close
            f_lo = f_next
        else:
            break

    # num + 1 to include the rightmost edge as well
    mz_ = np.empty((num + 1,), float)
    _fill_binning(mz_, mz_lo, f_next)

    return mz_[:-1], mz_[1:], f_next
