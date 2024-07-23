import sys

from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np

def write_random_file(path: str):

    bin_dict = np.load("notebooks/bins32768.npz")
    bin_lo = bin_dict["bin_lo"]
    bin_hi = bin_dict["bin_hi"]
    bin_width = bin_hi - bin_lo

    def r_spectrum() -> "tuple[np.ndarray, np.ndarray]":
        # rejection sampling : follow the mzs distribution

        s_mzs = bin_lo + np.random.rand(bin_width.size) * bin_width

        r_mask = np.random.uniform(size=s_mzs.shape) < 0.25
        s_mzs = s_mzs[r_mask]

        s_int = np.abs(np.random.normal(1e3, 1e3, size=s_mzs.shape))

        return s_mzs, s_int

    with ImzMLWriter(path, mode="processed") as writer:

        for y in range(128):
            for x in range(128):
                s = r_spectrum()
                writer.addSpectrum(*s, (x + 1, y + 1))


if __name__ == "__main__":
    write_random_file(sys.argv[1])
