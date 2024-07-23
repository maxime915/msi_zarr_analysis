"autocrop: crop array automatically to highlight nonzero data"

from typing import Tuple, Sequence, Optional
import numpy as np


def bounds(array: np.ndarray, *, per: Optional[int] = None):

    if per is not None:
        array = array.copy()
        threshold = np.percentile(array, per)
        array[array < threshold] = 0

    indices = array.nonzero()

    def _bound(idx: np.ndarray) -> Tuple[int, int]:
        if idx.size == 0:
            return 0, 0
        return idx.min(), idx.max() + 1

    return [_bound(idx) for idx in indices]


def autocrop(array: np.ndarray):
    """crop array automatically to highlight nonzero data

    Args:
        array (np.ndarray): numpy array

    Returns:
        Tuple[slice, slice]: _description_
    """
    "autocrop: crop 2d array automatically to exclude missing data"

    return tuple(slice(lo, hi) for (lo, hi) in bounds(array))


def autocrop_multi(arrays: Sequence[np.ndarray], *, per: int | None = None):

    bound_lst = [bounds(a, per=per) for a in arrays]
    try:
        bound_arr = np.array(bound_lst, dtype=int)  # array, axis, lo-hi
    except ValueError as e:
        n_dims = set(len(dims_bounds) for dims_bounds in bound_lst)
        if len(n_dims) > 1:
            raise ValueError("inconsistent dimensions")
        if not all(n_dims):
            raise ValueError("at least one array has no dimensions")
        raise ValueError("invalid sizes") from e

    # TODO remove the (0, 0) if not all arrays have it that way

    lowest = bound_arr[:, :, 0].min(axis=0)
    highest = bound_arr[:, :, 1].max(axis=0)

    return tuple(slice(lo, hi) for (lo, hi) in zip(lowest, highest))
