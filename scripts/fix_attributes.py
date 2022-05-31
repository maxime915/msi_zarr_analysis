"fix_attributes: temporary fix to add the binary mode to the missing attributes"

import sys

import zarr

PIMS_MSI = "pims-msi"
BINARY_MODE = "binary_mode"

def fix_attributes(path: str):

    z = zarr.open_group(path, mode="rw")

    mzs = z["/labels/mzs/0"]
    is_processed = any(s > 1 for s in mzs.shape[1:])
    mode = "processed" if is_processed else "continuous"

    if PIMS_MSI not in z.attrs:
        print(f"{PIMS_MSI=} not found in the attrs")
        z.attrs[PIMS_MSI] = {BINARY_MODE: mode}
    elif BINARY_MODE not in z.attrs[PIMS_MSI]:
        print(f"{BINARY_MODE=} not found in the {PIMS_MSI=} attrs")
        cpy = z.attrs[PIMS_MSI]
        cpy[BINARY_MODE] = mode
        z.attrs[PIMS_MSI] = cpy
    elif mode != z.attrs[PIMS_MSI][BINARY_MODE]:
        print(f"{mode=} != {z.attrs[PIMS_MSI][BINARY_MODE]=}")
    else:
        print(f"zarr at {path=} is fine")


list(map(fix_attributes, sys.argv[1:]))

