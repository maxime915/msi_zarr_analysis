import sys
import numpy as np
from collections import Counter

from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

# iter all chunks of a 
if __name__ != "__main__":
    raise ValueError("mz_info may not be imported")

if len(sys.argv) != 2:
    raise ValueError("expected exactly one argument")

zarr_path = sys.argv[1]

z = open_group_ro(zarr_path)
mzs = z["/labels/mzs/0"]
z_lengths = z["/labels/lengths/0"]

# load full 2d array
n_lengths = z_lengths[0, 0]


len_counter = Counter()
commons = set()

loaded_chunks = list(iter_loaded_chunks(mzs, skip=2))

for idx, (cy, cx) in enumerate(loaded_chunks):
    print(f"{idx=} / {len(loaded_chunks)} {cy=} {cx=}")
    
    # avoid loading useless chunks    
    c_lens = n_lengths[cy, cx]
    len_cap = c_lens.max()

    c_mzs = mzs[:len_cap, 0, cy, cx]
    
    for y, x in np.argwhere(c_lens):
        s_len = c_lens[y, x]
        len_counter[s_len] += 1
        commons.update(c_mzs[:s_len, y, x])

    print(f"{len(commons)    =:8d}")
    print(f"{len(len_counter)=:8d}")

lengths = n_lengths[n_lengths > 0]
print(f"{np.mean(lengths)=:8.2f}")
print(f"{np.std(lengths) =:8.2f}")
print(f"{np.min(lengths) =:8d}")
print(f"{np.max(lengths) =:8d}")

print(f"{len(commons)    =:8d}")
print(f"{len(len_counter)=:8d}")
print(f"{len_counter=!r}")

# comulis13 (26 chunks*)
# np.mean(lengths)=256547.43
# np.std(lengths) =12419.30
# np.min(lengths) =  203349
# np.max(lengths) =  406594
# len(commons)    = 7607503

# comulis14 (23 chunks*)
# np.mean(lengths)=257484.32
# np.std(lengths) =12359.97
# np.min(lengths) =  210897
# np.max(lengths) =  466309
# len(commons)    = 7608585

# comulis15 (9 chunks*)
# np.mean(lengths)=256564.77
# np.std(lengths) =12667.68
# np.min(lengths) =  205358
# np.max(lengths) =  401546
# len(commons)    = 7583151
