"utils for the CLI tool"

from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from numpy import typing as npt
import pandas as pd
from functools import wraps
import click


def parser_callback(ctx, param, value: str):
    """click callback to parse a value to either None, int or float

    Args:
        ctx (_type_): ignored
        param (_type_): ignored
        value (str): value to parse
    """
    # None value
    if value == "None":
        return None

    # maybe int
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    # maybe float
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # return as str
    return value


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



class RegionParam:
    "helper class to avoid duplication in similar Click command"

    x_low: int
    x_high: int
    y_low: int
    y_high: int

    def __init__(self, kwargs):
        for name in RegionParam.__annotations__:
            setattr(self, name, kwargs[name])
    
    def validated(self) -> "RegionParam":
        return self

    @staticmethod
    def add_click_options(fn):
        "add options to create a Region Param from the arguments passed"
        
        @click.option("--x-low", type=int, default=0)
        @click.option("--x-high", type=int, default=-1)
        @click.option("--y-low", type=int, default=0)
        @click.option("--y-high", type=int, default=-1)
        @wraps(fn)
        def _wrapped(*a, **kw):
            return fn(*a, **kw)

        return _wrapped

    @property
    def y_slice(self) -> slice:
        return slice(self.y_low, self.y_high, 1)
    
    @property
    def x_slice(self) -> slice:
        return slice(self.x_low, self.x_high, 1)
    


class BinningParam:
    "helper class to avoid duplication in similar Click command"

    mz_low: float
    mz_high: float
    n_bins: int
    bin_csv_path: str

    def __init__(self, kwargs):
        for name in BinningParam.__annotations__:
            setattr(self, name, kwargs[name])

    def validated(self) -> "BinningParam":
        if not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError(f"{self.n_bins=} should be an int > 1")
        return self

    @staticmethod
    def add_click_options(fn):
        "add options to create a Binning Param from the arguments passed"

        @click.option("--mz-low", type=float, default=200.0)
        @click.option("--mz-high", type=float, default=850.0)
        @click.option("--n-bins", type=int, default=100)
        @click.option(
            "--bin-csv-path",
            type=click.Path(exists=True),
            help=(
                "CSV file containing the m/Z values in the first column and the "
                "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
                "and 'b-bins'"
            ),
        )
        @wraps(fn)
        def _wrapped(*a, **kw):
            return fn(*a, **kw)

        return _wrapped

    def get_bins(
        self,
    ) -> Tuple[npt.NDArray[np.dtype("f8")], npt.NDArray[np.dtype("f8")]]:

        if self.bin_csv_path:
            return bins_from_csv(self.bin_csv_path)

        return uniform_bins(self.mz_low, self.mz_high, self.n_bins)


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

    dataframe: pd.DataFrame = pd.read_csv(csv_path, sep=None, engine="python")

    if isinstance(mz_column, int):
        mz_column = dataframe.columns[mz_column]
    if isinstance(tol_column, int):
        tol_column = dataframe.columns[tol_column]

    mz_series = dataframe[mz_column]
    tol_series = dataframe[tol_column]

    return (mz_series - tol_series).to_numpy(), (mz_series + tol_series).to_numpy()
