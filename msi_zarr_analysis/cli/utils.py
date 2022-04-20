"utils for the CLI tool"

from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from numpy import typing as npt
import pandas as pd


def split_csl(csl: str) -> List[int]:
    if not csl:
        return []
    try:
        return [int(part.strip()) for part in csl.split(",")]
    except ValueError as e:
        raise ValueError(f"invalid comma separated list: {csl!r}")


def load_img_mask(path: str) -> npt.NDArray[np.dtype("bool")]:
    # load data from the image
    mask = Image.open(path)
    mask_np = np.array(mask.getdata()).reshape(mask.size[::-1])
    # convert to binary boolean mask
    mask_np = mask_np > mask_np.max() / 2
    return mask_np


# def combine_masks(*masks: npt.NDArray[np.dtype("int")]) -> npt.NDArray[np.dtype("int")]:

#     combined = masks[0]
#     for mask in masks[1:]:
#         combined = np.maximum(combined, mask)

#     return combined


# def combine_and_load_masks(*path_lst: str) -> npt.NDArray[np.dtype("int")]:
#     masks = [load_img_mask(path, i) for i, path in enumerate(path_lst, start=1)]
#     return combine_masks(*masks)


def uniform_bins(
    mz_low: float, mz_high: float, n_bins: int
) -> Tuple[npt.NDArray[np.dtype("f8")], npt.NDArray[np.dtype("f8")]]:
    edges = np.histogram_bin_edges(0, bins=n_bins, range=(mz_low, mz_high))
    bin_lo = edges[:-1]
    bin_hi = edges[1:]
    return bin_lo, bin_hi


def bins_from_csv(
    csv_path: str,
    mz_column: Union[int, str] = 0,
    tol_column: Union[int, str] = 1,
) -> Tuple[npt.NDArray[np.dtype("f8")], npt.NDArray[np.dtype("f8")]]:

    dataframe: pd.DataFrame = pd.read_csv(csv_path, sep=None, engine='python')

    if isinstance(mz_column, int):
        mz_column = dataframe.columns[mz_column]
    if isinstance(tol_column, int):
        tol_column = dataframe.columns[tol_column]

    mz_series = dataframe[mz_column]
    tol_series = dataframe[tol_column]

    return (mz_series - tol_series).to_numpy(), (mz_series + tol_series).to_numpy()
