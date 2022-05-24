import sys

import numpy as np
import zarr
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

def inspect_spectra(path: str):
    z: zarr.Group = zarr.open_group(path, mode='r')

    z_int = z["/0"]
    z_len = z["/labels/lengths/0"]

    lengths = z_len[0, 0, ...]

    print(f"*** inspecting {path}: ***")
    for cy, cx in iter_loaded_chunks(z_int, skip=2):
        c_len = lengths[cy, cx]

        max_len = c_len.max()
        c_int = z_int[:max_len, 0, cy, cx]

        for y, x in zip(*c_len.nonzero()):
            spec = c_int[: c_len[y, x], y, x]
            print((
                f"y={y+cy.start:3d} x={x+cx.start:3d}"
                f" .len={c_len[y, x]:3d}"
                f" .min={np.min(spec):.1e}"
                f" .max={np.max(spec):.1e}"
                f" .med={np.median(spec):.1e}"
                f" .mean={np.mean(spec):.1e}"
                f" .sum={np.sum(spec):.1e}"
            ))

list(map(inspect_spectra, sys.argv[1:]))
