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
