import context

import contextlib
import shutil

import zarr
import numpy as np

import pytest

from msi_zarr_analysis.preprocessing import binning


def test_run():
    pytest.skip("too long to run")
    
    bin_dict = np.load("notebooks/bins1024.npz")
    bin_lo = bin_dict["bin_lo"]
    bin_hi = bin_dict["bin_hi"]

    src_path = "datasets/test_imzml_1.zarr/"

    with contextlib.ExitStack() as stack:
        tmp_store_1 = zarr.TempStore()
        stack.push(lambda *_: shutil.rmtree(tmp_store_1.path))

        tmp_store_2 = zarr.TempStore()
        stack.push(lambda *_: shutil.rmtree(tmp_store_2.path))

        binning.bin_processed_lo_hi(src_path, tmp_store_1, bin_lo, bin_hi)


def test_bin_no():

    n_bins = 64
    s_len = 1024

    for _ in range(10):
        edges = np.random.rand(n_bins + 1)
        edges.sort()

        bin_lo = edges[:-1]
        bin_hi = edges[+1:]

        s_mzs = np.random.rand(s_len)
        s_mzs.sort()
        s_int = np.random.rand(s_len)

        binned_left = np.zeros((n_bins,))
        binning.bin_band(binned_left, s_mzs, s_int, bin_lo, bin_hi)

        binned_right = np.zeros((n_bins,))
        binning._bin_band_no(binned_right, s_mzs, s_int, bin_lo, bin_hi)

        assert np.allclose(binned_left, binned_right)


def test_fail_overlap():

    n_bins = 64
    s_len = 1024

    edges = np.random.rand(n_bins + 1)
    edges.sort()

    bin_lo = edges[:-1]
    bin_hi = edges[+1:]

    # create overlap
    bin_hi[31] = bin_lo[33]

    bins = np.zeros((n_bins,))
    s_mzs = np.random.rand(s_len)
    s_mzs.sort()
    s_int = np.random.rand(s_len)

    with pytest.raises(AssertionError):
        binning._bin_band_no(bins, s_mzs, s_int, bin_lo, bin_hi)
