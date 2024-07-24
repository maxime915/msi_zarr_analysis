"""
centroiding: centroid spectra of a mass spectrometry dataset
"""

from enum import Enum
from typing import Dict, Tuple

import numpy as np
from pyopenms import (
    MSSpectrum,
    PeakPickerCWT,
    PeakPickerHiRes,
    PeakPickerIterative,
    PeakPickerMaxima,
    PeakPickerMRM,
    PeakPickerSH,
)


class PickerKind(Enum):
    PeakPickerCWT = 0
    PeakPickerHiRes = 1
    PeakPickerIterative = 2
    PeakPickerMRM = 3
    PeakPickerMaxima = 4
    PeakPickerSH = 5


__PICKER_MAPPING = {
    PickerKind.PeakPickerCWT: PeakPickerCWT,
    PickerKind.PeakPickerHiRes: PeakPickerHiRes,
    PickerKind.PeakPickerIterative: PeakPickerIterative,
    PickerKind.PeakPickerMRM: PeakPickerMRM,
    PickerKind.PeakPickerMaxima: PeakPickerMaxima,
    PickerKind.PeakPickerSH: PeakPickerSH,
}


def centroid_dict(
    spectrum_dict: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    picker_kind: PickerKind = PickerKind.PeakPickerHiRes,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    "Apply centroiding on a dict that maps keys to spectra"

    picked_dict: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    peak_picker = __PICKER_MAPPING[picker_kind]()
    for coord, spectrum in spectrum_dict.items():

        ms_spectrum = MSSpectrum()
        ms_spectrum.set_peaks(spectrum)

        picked = MSSpectrum()
        peak_picker.pick(ms_spectrum, picked)

        picked_dict[coord] = picked.get_peaks()

    return picked_dict
