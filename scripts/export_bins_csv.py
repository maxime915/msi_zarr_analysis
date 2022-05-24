import sys

import numpy as np
import pandas as pd

assert len(sys.argv) > 1, "expected at least one argument"


def save(path: str):
    assert path[-4:] == ".npz", "expected a numpy z archive"
    prefix = "saved_prog_binning_"
    *base, fname = path.split("/")
    assert fname[: len(prefix)] == prefix, f"filename should start with {prefix=}"

    suffix = fname[len(prefix) : -4]

    archive = np.load(path)

    collected_bin_hi = archive["collected_bin_hi"]
    collected_bin_lo = archive["collected_bin_lo"]

    # sort bin order
    for iteration in range(collected_bin_hi.shape[0]):
        lows = collected_bin_lo[iteration, :]
        highs = collected_bin_hi[iteration, :]

        sorted_indices = np.argsort(lows)

        collected_bin_lo[iteration, :] = lows[sorted_indices]
        collected_bin_hi[iteration, :] = highs[sorted_indices]

    df = pd.DataFrame()

    # export to pandas
    for bin_idx in range(collected_bin_hi.shape[1]):
        lows = collected_bin_lo[:, bin_idx]
        highs = collected_bin_hi[:, bin_idx]

        df[f"bin-{bin_idx+1}-left"] = lows
        df[f"bin-{bin_idx+1}-right"] = highs

    prefix = "/".join(base)
    if prefix:
        prefix += "/"

    df.to_csv(
        prefix + f"saved_bins_coordinate_{suffix}.csv",
        header=True,
        index=False,
    )


# list() to consume the iterator
list(map(save, sys.argv[1:]))
