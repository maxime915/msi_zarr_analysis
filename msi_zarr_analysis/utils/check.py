import zarr


def check_group(z: zarr.Group, strict: bool = False) -> None:

    arrays = [arr.path for _, arr in z.arrays(recurse=True)]

    # must-have's
    for label in ["0", "labels/mzs/0", "labels/lengths/0"]:
        if label not in arrays:
            raise ValueError(f"Group at {z.store.path} should have {label=}")

    try:
        mode = z.attrs["pims-msi"]["binary_mode"]
    except:
        raise ValueError("unable to find {'pims-msi':{'binary_mode':''} } attribute")

    if not strict:
        return

    z_int = z["/0"]
    z_mzs = z["/labels/mzs/0"]
    z_len = z["/labels/lengths/0"]

    try:
        c, z, y, x = z_int.shape
    except:
        raise ValueError(f"array should be 4 dimensional ({z_int.shape=})")

    if z_len.shape != (1, z, y, x):
        raise ValueError("invalid shape for lengths label")

    if mode == "continuous":
        if z_mzs.shape != (c, 1, 1, 1):
            raise ValueError("invalid shape for m/Z label")
    elif mode == "processed":
        if z_mzs.shape != (c, z, y, x):
            raise ValueError("invalid shape for m/Z label")
    else:
        raise ValueError(f"invalid binary_mode: {mode}")


def open_group_ro(path: str, strict_check: bool = False) -> zarr.Group:
    z = zarr.open(path, mode='r')
    check_group(z, strict=strict_check)
    return z
