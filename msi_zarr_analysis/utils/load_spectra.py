"load_spectra: load spectra given an ROI"

import logging
from typing import Dict, Tuple

import numpy as np
import zarr

from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

def load_intensities(
    ms_group: zarr.Group,
    selection: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:

    # set bounds to avoid loading useless chunks
    y_slice, x_slice = slice(0, 0), slice(0, 0)
    y_nz, x_nz = selection.nonzero()
    try:
        y_slice = slice(int(y_nz.min()), int(y_nz.max()+1))
    except ValueError:
        pass
    try:
        x_slice = slice(int(x_nz.min()), int(x_nz.max()+1))
    except ValueError:
        pass
    
    logging.info("y_slice: %s, x_slice: %s", y_slice, x_slice)
    
    spectrum_dict = {}

    # map the results to the zarr arrays
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]

    # yield all rows
    for cy, cx in iter_loaded_chunks(intensities, y_slice, x_slice, skip=2):
        c_len = lengths[0, 0, cy, cx]
        len_cap = c_len.max()  # small optimization for uneven spectra
        c_int = intensities[:len_cap, 0, cy, cx]

        c_mask = selection[cy, cx]
            
        for y, x in zip(*c_mask.nonzero()):
            length = c_len[y, x]
            if length == 0:
                continue
    
            spectrum_dict[(y+cy.start, x+cx.start)] = c_int[:length, y, x]

    return spectrum_dict