"normalization"

from typing import List

import numba as nb
import numpy as np
import numpy.typing as npt
import zarr

from ..utils.check import open_group_ro
from ..utils.iter_chunks import clean_slice_tuple, iter_loaded_chunks

# this implementation has a bad CPU utilization : it appear to be worse in deep
# spectra:
#   Deeper spectra mean narrow chunks and therefore more frequent disk access
#   Is there a way to avoid this bottleneck ?

# TODO cleanup
#   - base implementation that apply a numpy function to each chunk to get the scaling factor for each spectra at once
#   - per spectra implementation that can use numba (for mean and median)

def _jit_scale(scale_fn: "function"):
    # @nb.jit(nopython=True)
    # def _scale(arr: npt.NDArray):
    #     return scale_fn(arr)

    # return _scale
    return scale_fn


# see https://www.scirp.org/journal/paperinformation.aspx?paperid=86606
__NME_TO_FN = {
    "tic": _jit_scale(np.sum),
    "sum": _jit_scale(np.sum),
    # "mean": _jit_scale(np.mean),  # unsupported : appending zeros changes the results
    "max": _jit_scale(np.max),
    "vect": _jit_scale(np.linalg.norm),
    # "median": _jit_scale(np.median),  # unsupported : appending zeros changes the results
}


def valid_norms() -> List[str]:
    return list(__NME_TO_FN.keys())


# @nb.jit(nopython=True, parallel=True)
def normalize_chunk(
    output: npt.NDArray,
    ints: npt.NDArray,
    lengths: npt.NDArray,
    scale_fn,
):
    "this operation is in-place: the data is a buffer, output may be the same as ints"

    # this should be faster ! doesn't require numba either....
    scale = scale_fn(ints, axis=0)
    scale[scale == 0] = np.nan
    output[...] = ints / scale

    # idx_z, idx_y, idx_x = lengths.nonzero()
    # count = idx_z.size


    # for idx in nb.prange(count):
    #     z, y, x = idx_z[idx], idx_y[idx], idx_x[idx]

    #     len_band = lengths[z, y, x]
    #     int_band = ints[:len_band, z, y, x]

    #     output[:len_band, z, y, x] = int_band / scale_fn(int_band)


def _normalize(
    z: zarr.Group,
    destination: zarr.Group,
    y_slice: slice,
    x_slice: slice,
    scale_fn: "function",
):

    z_ints = z["/0"]
    z_lengths = z["/labels/lengths/0"][0]

    # create empty array
    destination.create(
        "/0",
        shape=z_ints.shape,
        chunks=z_ints.chunks,
        compressor=z_ints.compressor,
        order=z_ints.order,
        fill_value=0.0,
    )
    # copy labels
    zarr.copy_store(z.store, destination.store, excludes=["^0"], if_exists="replace")

    z_dest_int = destination["/0"]

    # TODO add description of transformation
    for key, value in z.attrs.items():
        destination.attrs[key] = value
    if "pims-msi" not in destination.attrs:
        destination.attrs["pims-msi"] = {}
    bkp = destination.attrs["pims-msi"]
    bkp["binary_mode"] = "continuous"
    destination.attrs["pims-msi"] = bkp

    # load chunks
    for cz, cy, cx in iter_loaded_chunks(z_ints, slice(None), y_slice, x_slice, skip=1):
        c_lengths = z_lengths[cz, cy, cx]
        max_len = c_lengths.max()
        
        # allocate an array of the right size for c_dest
        c_dest = np.zeros(
            shape=(
                max_len,
                cz.stop - cz.start,
                cy.stop - cy.start,
                cx.stop - cx.start,
            )
        )

        # load from disk
        c_ints = z_ints[:max_len, cz, cy, cx]

        normalize_chunk(
            c_dest,
            c_ints,
            c_lengths,
            scale_fn,
        )

        # write to disk
        z_dest_int[:max_len, cz, cy, cx] = c_dest


def normalize_array(
    z_path: str,
    destination_path: str,
    norm_name: str,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
):

    norm_name = str(norm_name).lower()

    if norm_name not in __NME_TO_FN:
        raise ValueError(f"norm '{norm_name}' not in {valid_norms()=}")

    scale_fn = __NME_TO_FN[norm_name]

    z = open_group_ro(z_path)

    destination = zarr.open_group(destination_path, mode="w-")

    return _normalize(
        z,
        destination,
        *clean_slice_tuple(z["/0"].shape[2:], y_slice, x_slice),
        scale_fn,
    )
