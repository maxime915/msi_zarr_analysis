import timeit
import context

import numpy as np

from msi_zarr_analysis.preprocessing import binning


def make_target(bin_fn):
    
    n_bin = 32
    s_len_mean = 8192
    s_len_std = 512
    n_spectra = 512
    
    edges = 100.0 + 1050.0 * np.random.rand(n_bin + 1)
    edges.sort()
    
    bin_lo = edges[:-1]
    bin_hi = edges[+1:]
    
    def r_spectrum():
        
        s_len = int(np.random.normal(s_len_mean, s_len_std))
        s_mzs = 100.0 + 1050.0 * np.random.rand(s_len)
        s_mzs.sort()
        
        s_int = np.random.rand(s_len)
        
        return s_mzs, s_int
    
    dataset = tuple(r_spectrum() for _ in range(n_spectra))
    
    def target():
        for (s_mzs, s_int) in dataset:
            s_bin = np.zeros((n_bin,))
            bin_fn(s_bin, s_mzs, s_int, bin_lo, bin_hi)

    return target


def time_fns():
    
    target_no = make_target(binning._bin_band_no)
    target_ = make_target(binning.bin_band)
    
    res_no = np.array(timeit.repeat(target_no, number=5, repeat=25))
    res_ = np.array(timeit.repeat(target_, number=5, repeat=25))
    
    print(f"{res_no.mean() = :.3f} {res_no.std() = :.3f}")
    print(f"{res_.mean() = :.3f} {res_.std() = :.3f}")

if __name__ == "__main__":
    time_fns()
