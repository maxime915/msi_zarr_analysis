"""
"""

from enum import Enum
from typing import Dict, Tuple
import numpy as np
from pyopenms import MSExperiment, MSSpectrum, MzMLFile, PeakPickerHiRes

from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks
from msi_zarr_analysis.utils.load_spectra import load_spectra


def main():
    exp = MSExperiment()
    group = open_group_ro("datasets/comulis13.zarr")

    z_int = group["/0"]
    z_mzs = group["/labels/mzs/0"]
    z_len = group["/labels/lengths/0"]

    n_len = z_len[0, 0]

    mask = np.where(np.random.random(size=n_len.shape) < 0.1, n_len, 0)

    peak_picker = PeakPickerHiRes()
    for coord, (s_mzs, s_int) in load_spectra(group, mask).items():
        spectrum = MSSpectrum()
        spectrum.set_peaks((s_mzs, s_int))
        picked = MSSpectrum()
        peak_picker.pick(spectrum, picked)
        ps_mzs, ps_int = picked.get_peaks()

        pass

    for cy, cx in iter_loaded_chunks(z_int, skip=2):
        print(f"{cy=} {cx=}")

        c_mask = mask[cy, cx]
        if np.count_nonzero(c_mask) == 0:
            continue
        c_len = n_len[cy, cx]
        len_cap = c_len.max()

        c_int = z_int[:len_cap, 0, cy, cx]
        c_mzs = z_mzs[:len_cap, 0, cy, cx]

        for y, x in zip(*c_mask.nonzero()):
            s_len = c_len[y, x]
            if s_len == 0:
                continue

            spectrum = MSSpectrum()
            spectrum.set_peaks((c_mzs[:s_len, y, x], c_int[:s_len, y, x]))

            # TODO is this required ?
            spectrum.sortByPosition()
            spectrum.setMSLevel(1)

            exp.addSpectrum(spectrum)

    MzMLFile().store("test_file_r13.mzML", exp)


if __name__ == "__main__":
    main()
else:
    raise ValueError()
