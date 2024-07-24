"load_spectra: load spectra given an ROI"

import logging
from typing import Dict, Tuple

import numpy as np
import zarr
from msi_zarr_analysis.utils.autocrop import autocrop
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks


def load_intensities(
    ms_group: zarr.Group,
    selection: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:

    # set bounds to avoid loading useless chunks
    y_slice, x_slice = autocrop(selection)

    logging.info("y_slice: %s, x_slice: %s", y_slice, x_slice)

    spectrum_dict = {}

    # map the results to the zarr arrays
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]

    # iterate all spectra
    for cy, cx in iter_loaded_chunks(intensities, y_slice, x_slice, skip=2):
        c_len = lengths[0, 0, cy, cx]
        len_cap = c_len.max()  # small optimization for uneven spectra
        c_int = intensities[:len_cap, 0, cy, cx]

        c_mask = selection[cy, cx]

        for y, x in zip(*c_mask.nonzero()):
            length = c_len[y, x]
            if length == 0:
                continue

            spectrum_dict[(y + cy.start, x + cx.start)] = c_int[:length, y, x]

    return spectrum_dict


def load_spectra(
    ms_group: zarr.Group,
    selection: np.ndarray,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:

    # set bounds to avoid loading useless chunks
    y_slice, x_slice = autocrop(selection)

    logging.info("y_slice: %s, x_slice: %s", y_slice, x_slice)

    spectrum_dict = {}

    # map the results to the zarr arrays
    z_int = ms_group["/0"]
    z_mzs = ms_group["/labels/mzs/0"]
    z_len = ms_group["/labels/lengths/0"]

    # continuous mode optimization
    continuous_mode = ms_group.attrs["pims-msi"]["binary_mode"] == "continuous"
    if continuous_mode:
        s_mzs = z_mzs[: z_len[0, 0, 0, 0], 0, 0, 0]

    # iterate all spectra
    for cy, cx in iter_loaded_chunks(z_int, y_slice, x_slice, skip=2):
        c_len = z_len[0, 0, cy, cx]
        len_cap = c_len.max()

        c_int = z_int[:len_cap, 0, cy, cx]
        if not continuous_mode:
            c_mzs = z_mzs[:len_cap, 0, cy, cx]

        c_mask = selection[cy, cx]

        for y, x in zip(*c_mask.nonzero()):
            s_len = c_len[y, x]
            if s_len == 0:
                continue

            s_int = c_int[:s_len, y, x]
            if not continuous_mode:
                s_mzs = c_mzs[:s_len, y, x]

            spectrum_dict[(y + cy.start, x + cx.start)] = (s_mzs, s_int)

    return spectrum_dict
