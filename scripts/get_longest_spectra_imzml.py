"get_longest_spectra_imzml: make a .CSV file for the longest spectrum in an imzML file"


import sys
import pathlib

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser

def make_csv(path_imzml: str) -> None:
    "create and save a CSV file for the longest spectrum"
    
    parser = ImzMLParser(path_imzml)
    
    parser.coordinates
    parser.intensityLengths
    
    spectrum_idx = np.argmax(parser.intensityLengths)
    coordinate = parser.coordinates[spectrum_idx]
    
    s_mzs, s_int = parser.getspectrum(spectrum_idx)
    
    df = pd.DataFrame(np.stack([s_mzs, s_int], axis=1), columns=("m/z", "intensity"), dtype=float)
    
    stem = pathlib.Path(path_imzml).stem

    df.to_csv(stem + f"-x={coordinate[0]}-y={coordinate[1]}.csv")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        make_csv(arg)