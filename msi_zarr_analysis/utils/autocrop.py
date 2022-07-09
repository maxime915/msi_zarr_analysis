"autocrop: crop array automatically to highlight nonzero data"

from typing import Tuple
import numpy as np


def autocrop(array: np.ndarray) -> Tuple[slice, slice]:
    """crop array automatically to highlight nonzero data

    Args:
        array (np.ndarray): numpy array

    Returns:
        Tuple[slice, slice]: _description_
    """
    "autocrop: crop 2d array automatically to exclude missing data"

    indices = array.nonzero()

    if not indices:
        return ()

    # if there are no nonzero, idx.min() will raise an exception
    if indices[0].size == 0:
        return (slice(0, 0),) * len(indices)

    return tuple(slice(idx.min(), idx.max() + 1) for idx in indices)
