import contextlib
import pathlib
import typing

import numpy as np
import zarr
from scipy.ndimage import label

from omezarrmsi import OMEZarrMSI


def get_mask(dataset: OMEZarrMSI):
    "return an [Z, Y, X] mask of all valid spectra"

    # [Z, Y, X]
    n_len: np.ndarray = dataset.z_len[0]  # type:ignore
    match label(n_len > 0):
        case labeled, _:
            pass
        case default:
            raise ValueError(f"unexpected return from label: {default!r}")

    values, counts = np.unique(labeled, return_counts=True)

    # artificially set the counts of len==0 to 0
    for idx, val in enumerate(values):
        if ((labeled == val) == (n_len == 0)).all():
            counts[idx] = 0
            break
    else:
        if (n_len == 0).any():
            raise ValueError("expected to find a component for len=0")

    # only keep the biggest component
    mask = labeled == values[np.argmax(counts)]
    n_len[~mask] = 0

    # find outliers in number of values per spectra
    len_med = np.median(n_len[n_len > 0])
    diff_to_med = np.where(n_len > 0, np.abs(n_len - len_med), 0)
    med_dev = np.median(diff_to_med[n_len > 0])
    outliers = diff_to_med > 10.0 * med_dev

    n_len[outliers] = 0

    return n_len


def _new_shape(new_len: np.ndarray[typing.Any, np.dtype[np.int_]]):
    def _len(idx: np.ndarray):
        if idx.size == 0:
            return 0
        return idx.max() + 1 - idx.min()
    return tuple(_len(idx) for idx in new_len.nonzero())


def copy_dataset(
    src: OMEZarrMSI,
    dst: OMEZarrMSI,
    mask: np.ndarray[typing.Any, np.dtype[np.bool_]],
):
    """
    Args:
        src (OMEZarrMSI): source dataset
        dst (OMEZarrMSI): destination dataset
        mask array[(z, y, x), bool]: mask selecting which spectra should be copied
    """

    def _copy(z_d: zarr.Array, z_s: zarr.Array, idx_d: tuple[slice, ...], idx_s: tuple[slice, ...]):
        # load all values
        values = z_s[(slice(None),) + idx_s]

        # remove extra channels (useless padding)
        values = values[:z_d.shape[0]]
        # remove empty spectra
        values[:, ~mask[idx_s]] = 0

        z_d[(slice(None),) + idx_d] = values

    # translation from destination to source
    lo_s: list[int] = [(idx.min() if idx.size > 0 else 0) for idx in mask.nonzero()]

    # copy chunk by chunk (of destination: reads are cheaper than writes)
    for idx_d in dst.idx_by_chunk(c=None):
        # map _d -> _s
        idx_s = tuple(
            slice(i.start + lo, i.stop + lo)
            for (i, lo) in zip(idx_d, lo_s, strict=True)
        )

        _copy(dst.z_int, src.z_int, idx_d, idx_s)
        _copy(dst.z_mzs, src.z_mzs, idx_d, idx_s)
        _copy(dst.z_len, src.z_len, idx_d, idx_s)


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")

    args = parser.parse_args()
    args_source = pathlib.Path(args.source)
    if not args_source.is_dir():
        raise ValueError(f"{args_source=} is not a directory")
    args_destination = pathlib.Path(args.destination)
    if args_destination.exists():
        raise ValueError(f"{args_destination} already exists")

    source = OMEZarrMSI(args_source, mode="r")
    new_len = get_mask(source)

    with contextlib.ExitStack() as stack:
        destination_store = zarr.DirectoryStore(args_destination)
        destination = OMEZarrMSI.create(
            destination_store,
            (new_len.max(),) + _new_shape(new_len),
            source.z_int.dtype,
            source.z_mzs.dtype,
            source.imzML_binary_mode,
            chunks=True,
            extra_metadata={"pims-msi": source.metadata["pims-msi"]},
        )
        stack.callback(destination.rmdir, True)

        # copy to destination
        copy_dataset(source, destination, new_len > 0)

        # hooray
        stack.pop_all()


if __name__ == "__main__":
    _main()
