"iter all the masses in an array"

import sys
from collections import Counter

import zarr


import numpy as np

from ..utils.iter_chunks import iter_loaded_chunks
from ..utils.check import open_group_ro


def get_ctr(path: str) -> Counter:
    "count all masses in one image"
    z =open_group_ro(path)

    mzs = z["/labels/mzs/0"]
    lengths = z["/labels/lengths/0"]

    ctr = Counter()
    for y, x in zip(*lengths[0, 0].nonzero()):
        band = mzs[:, 0, y, x]

        for mass in band:
            ctr[mass] += 1

    return ctr


def cache_merged_masses(path: str):
    "cache_merged_masses"
    z = open_group_ro(path)
    z_mzs = z["/labels/mzs/0"]
    z_lengths = z["/labels/lengths/0"]

    common_set = set()

    idx = 0
    for (cy, cx) in iter_loaded_chunks(z_lengths, skip=2):
        print(f"{idx=} {cy=} {cx=}")

        lengths = z_lengths[0, 0, cy, cx]
        non_zeros = lengths.nonzero()
        
        # skip if no non zero values before loading mzs from disk
        if non_zeros[0].size == 0:
            continue

        mzs = z_mzs[:, 0, cy, cx]

        for y, x in zip(*non_zeros):
            print(f"\ti={idx: 4d}, {y=: 4d}, {x=: 4d}", end=' ')

            band_set = set(mzs[:, y, x])
            common_set.update(band_set)

            print(f"#band={len(band_set): 10d}, #total={len(common_set)}")

            idx += 1

    z = zarr.open_group(path, mode="a")
    z["/cached/merged_masses/0"] = sorted(common_set)


for _file in sys.argv[1:]:
    print(f"iteration {_file=}")
    cache_merged_masses(_file)
