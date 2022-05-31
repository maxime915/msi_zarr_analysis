
import numpy as np
import numba as nb

from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks



def get_mass_range_chunk(
    c_mzs: np.ndarray,
    c_len: np.ndarray,
):
    valid_idx = c_len > 0
    valid_mzs = c_mzs[:, valid_idx]
    
    lows = valid_mzs[0, ...]
    highs = valid_mzs[c_len[valid_idx]-1, ...]

    return lows.min(), highs.max()



def get_mass_range(zarr_path: str):
    ms_group = open_group_ro(zarr_path)
    
    mzs = ms_group["/labels/mzs/0"]
    lengths = ms_group["/labels/lengths/0"]

    min_ = +np.inf
    max_ = -np.inf

    for cz, cy, cx in iter_loaded_chunks(mzs, skip=1):
        c_len = lengths[0, cz, cy, cx]
        c_mzs = mzs[:c_len.max(), cz, cy, cx]
        
        min__, max__ = get_mass_range_chunk(c_mzs, c_len)
        
        if min__ < min_:
            min_ = min__
        if max__ > max_:
            max_ = max__

        print(f"{min__=} {max__=}")
    
    return min_, max_



        