"utility to iterate the chunks of an array"

import re

from itertools import product
from math import ceil
from typing import Iterable, Iterator, Optional, Tuple, Protocol

import zarr
from zarr.storage import listdir


class Array(Protocol):
    """Array Protocol for multidimensional chunked arrays"""

    @property
    def shape(self) -> Tuple[int, ...]:
        """shape: a n-dimensional tuple of int representing the length of each \
           dimension of the array"""

    @property
    def chunks(self) -> Tuple[int, ...]:
        """chunks: a n-dimensional tuple of int representing the length of each \
           dimension of the chunks composing the array"""

    @property
    def ndim(self) -> int:
        """the number of axes in the array, must be equal to the length of both\
            shape and chunks"""


def clean_slice(
    dim: int,
    sl: slice,
) -> slice:
    """return a cleaned slice for a given length, converting start, stop and step
    to non negative integer values. Only unit step are supported.

    Args:
        dim (int): the length of the axis
        sl (slice): the slice regarding the axis

    Raises:
        ValueError: if any parameter of the slice is not an int or NoneType
        IndexError: if any parameter of the slice is out of bounds

    Returns:
        slice: a slice with only non negative integer values
    """

    # make sure steps are all 1 or None
    if sl.step not in [1, None]:
        raise ValueError(f"slice should have step of 1: {sl} ({sl.step=})")

    start = sl.start
    if start is None:
        start = 0
    try:
        _start = int(start)
        assert _start == start
        start = _start
    except (AssertionError, ValueError):
        raise ValueError(f"invalid start for slice {sl} ({start=})")

    stop = sl.stop
    if stop is None:
        stop = dim
    try:
        _stop = int(stop)
        assert _stop == stop
        stop = _stop
    except (AssertionError, ValueError):
        raise ValueError(f"invalid stop for slice {sl} ({stop=})")

    # update slices to have >= 0 start
    if start < 0:
        start += dim
        if start < 0:
            raise IndexError(f"bad start for slice {sl} ({start=} < 0)")
    if start >= dim:
        raise IndexError(f"bad start for slice {sl} ({start=} >= {dim=}")

    # update slices to have <= len max
    if stop < 0:
        stop += dim
        if stop < 0:
            raise IndexError(f"bad stop for slice {sl} ({stop=} < 0)")
    if stop > dim:
        raise IndexError(f"bad stop for slice {sl} ({stop=} > {dim=})")

    # must not be empty
    if stop <= start:
        raise ValueError(f"empty slice {sl} ({stop=} <= {start=})")

    return slice(start, stop, 1)


def clean_slice_tuple(
    shape: Iterable[int],
    *slices: slice,
) -> Tuple[slice, ...]:
    """return a tuple of cleaned slices for a given shape, converting start,
    stop and step to non negative integer values for each axis.

    Args:
        shape (Iterable[int]): the length of each axis
        slices (Iterable[int]): the slice for each axis

    Raises:
        ValueError: if any parameter of the slice is not an int or NoneType
        IndexError: if any parameter of the slice is out of bounds

    Returns:
        slice: a slice with only non negative integer values
    """

    return tuple(clean_slice(dim, sl) for dim, sl in zip(shape, slices))


def _iter_chunks(
    slices: Iterable[slice],
    chunks: Iterable[int],
) -> Iterator[Tuple[slice, ...]]:
    """raw chunk iterator routine

    Args:
        slices (Iterable[slice]): MUST BE CLEANED USING clean_slice_tuple
        chunks (Iterable[int]): array chunk shape

    Yields:
        Iterator[Tuple[slice, ...]]: the slices in the raw array
    """

    # find low chunk idx
    low_chunk_idx = [s.start // c for s, c in zip(slices, chunks)]

    # find high chunk idx
    high_chunk_idx = [ceil(s.stop / c) for s, c in zip(slices, chunks)]

    # build range for each chunk
    chunk_ranges = [range(lo, hi) for lo, hi in zip(low_chunk_idx, high_chunk_idx)]

    # iterate the product of all ranges
    for chunk_idx in product(*chunk_ranges):
        # compute coordinates based on chunk_idx and slices
        yield tuple(
            slice(max(i * c, s.start), min((i + 1) * c, s.stop))
            for i, c, s in zip(chunk_idx, chunks, slices)
        )


def iter_spatial_chunks(
    array: Array,
    z: slice = slice(None),
    y: slice = slice(None),
    x: slice = slice(None),
) -> Iterator[Tuple[slice, slice, slice]]:
    """iterate all spatial chunks for a OME-Zarr MSI array. No slice is returned
    for the channel axis.

    Args:
        array (Array): an array corresponsing to the image (or one of its labels)
        z (slice, optional): the range for the Z axis. Defaults to slice(None).
        y (slice, optional): the range for the Y axis. Defaults to slice(None).
        x (slice, optional): the range for the X axis. Defaults to slice(None).

    Yields:
        Iterator[Tuple[slice, slice, slice]]: slices of non overlapping chunks
    """

    assert array.ndim == 4

    yield from iter_nd_chunks(array, z, y, x, skip=1)


def iter_nd_chunks(
    array: Array,
    *slices: Optional[slice],
    skip: int = 0,
) -> Iterator[Tuple[slice, ...]]:
    """iterate the chunks over the n first axis of the array.

    Args:
        array (Array): an array with at least n dimensions
        *slices: (slice): a slice per axis to be iterated

    Yields:
        Iterator[Tuple[slice, ...]]: slice of non overlapping chunks over the n\
            first axis.
    """

    if not isinstance(skip, int) or skip < 0:
        raise ValueError(f"{skip=} must be a non negative integer")
    if skip >= array.ndim:
        raise ValueError(f"{skip=} must be lower than {array.ndim=}")

    if not slices:
        slices = (array.ndim - skip) * [slice(None)]
    elif len(slices) != array.ndim - skip:
        raise ValueError(f"{slices=} should have {array.ndim-skip=} elements")

    slices = clean_slice_tuple(array.shape[skip:], *slices)

    chunk_width = array.chunks[skip:]

    yield from _iter_chunks(slices, chunk_width)


def iter_loaded_chunks(
    array: zarr.Array,
    *slices: Optional[slice],
    skip: int = 0,
) -> Iterator[Tuple[slice, ...]]:
    """Iterate the array chunks by chunks, only giving slices that store
    elements.

    The performance are much better on sparse arrays since some part of the
    domain are never loaded. On dense array there is a performance cost but most
    use will not benefit from it.

    If the array has more dimension than `skip` and `slices` specify together,
    the rightmost dimensions are ignored. E.g. only
    `array.shape[skip: skip+len(slices)]` is considered if slices is not None.

    If the chunk shape does not divide the array shape, some yielded idx will
    have a shape smaller than the chunk shape to avoid going out of bounds.

    Parameters
    ----------
    array : zarr.Array
        the array to iterate chunks from
    slices : slice
        a domain to restrict chunks, by default all the domain
    skip : int, optional
        the number of dimension to skip (starting from the left), by default 0

    Yields
    -------
    Iterator[Tuple[slice, ...]]
        idx corresponding the the chunk, always included in the slices domain

    Raises
    ------
    ValueError
        on bad parameter

    Usage
    -----
    ```py
    for chunk in iter_loaded_chunks(arr):
        data = arr[chunk]  # a single chunk of the array

    # this will skip idx outside of slice(low, high), ...
    for chunk in iter_loaded_chunks(arr, slice(low, high), ...):
        window_chunk = arr[chunk]
    ```

    Note
    ----
    The following function is inspired by zarr.core.Array.nchunks_initialized :
    this is the only reference that enumerate all chunks, I couldn't find any
    public API yet.
    """

    if not isinstance(skip, int) or skip < 0:
        raise ValueError(f"{skip=} must be a non negative integer")
    if skip >= array.ndim:
        raise ValueError(f"{skip=} must be lower than {array.ndim=}")

    if not slices:
        slices = (array.ndim - skip) * [slice(None)]
    elif len(slices) != array.ndim - skip:
        raise ValueError(f"{slices=} should have {array.ndim-skip=} elements")

    slices = clean_slice_tuple(array.shape[skip:], *slices)

    chunk_width = array.chunks[skip:]

    # key patter for chunk keys
    prog = re.compile(r"\.".join([r"\d+"] * min(1, array.ndim)))

    # get chunk keys
    chunk_keys = (k for k in listdir(array.chunk_store, array.path) if prog.match(k))

    # get integer indices
    chunks_as_tpl = (tuple(int(i) for i in c.split(".")[skip:]) for c in chunk_keys)

    if skip > 0:
        # remove duplicates due to some axes being ignored
        chunks_as_tpl = dict.fromkeys(chunks_as_tpl).keys()
    else:
        # already unique, make sure it is a list to persist
        chunks_as_tpl = list(chunks_as_tpl)

    for chunk_idx in chunks_as_tpl:
        # compute coordinates based on chunk_idx and slices
        slice_tpl = tuple(
            slice(max(i * c, s.start), min((i + 1) * c, s.stop))
            for i, c, s in zip(chunk_idx, chunk_width, slices)
        )

        # prevent yielding empty slices
        for s in slice_tpl:
            if s.stop <= s.start:
                break
        else:
            yield slice_tpl
