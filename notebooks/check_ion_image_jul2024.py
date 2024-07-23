# %%

import pathlib
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import omezarrmsi as ozm
from omezarrmsi.plots.mz_slice import mz_slice

# %%


def read_csv(path: pathlib.Path):

    # open the text file in UTF8
    with open(path, "r", encoding="utf8") as f_obj:
        content = f_obj.readlines()

    # remove lines with comments at the beginning
    # -> these cannot be removed automatically because the comment prefix (#) is
    #   also used to denote other values in the dataset (good job ScilsLab)
    content = [line for line in content if not line.startswith("#")]

    return pd.read_csv(
        StringIO("\n".join(content)),
        sep=";",
    )


# %%

base_dir = pathlib.Path(__file__).parent.parent / "Cytomine MSI 12072024"

csv_dct = {
    key: read_csv(base_dir / f"Lipid selection Region {key}.csv")
    for key in ["13", "14", "15"]
}

# %%

# load the dataset (slim, normalized, before deisotopping)

slim_dir = pathlib.Path.home() / "datasets" / "COMULIS-slim-msi"
files = {
    "13": ["region13_317norm_sample.zarr", "region13_nonorm_sample.zarr"],
    "14": ["region14_317norm_sample.zarr", "region14_nonorm_sample.zarr"],
    "15": ["region15_317norm_sample.zarr", "region15_nonorm_sample.zarr"],
}

for n1, n2 in files.values():
    assert (slim_dir / n1).is_dir()
    assert (slim_dir / n2).is_dir()

# %%

key = "15"
fname = files[key][0]

csv_file = csv_dct[key]
zf = ozm.OMEZarrMSI(slim_dir / fname, mode="r")

data: list[tuple[np.ndarray, np.ndarray]] = []
for _, row in csv_file.iterrows():
    lipid_name: str = row["Name"]
    lipid_mu: float = row["m/z"]
    lipid_w: float = row["Interval Width (+/- Da)"]

    mask, img = mz_slice(zf, lipid_mu - 1.0 * lipid_w, lipid_mu + 1.0 * lipid_w)
    data.append((mask, img))

fig, axes = plt.subplots(ncols=1, nrows=len(data), figsize=(8, 1.6 * len(data)))

for idx, (mask, img) in enumerate(data):
    axes[idx].set_title(csv_file.iloc[idx]["Name"] + " : " + str(csv_file.iloc[idx]["m/z"]))
    img = img.copy()
    img[~mask] = np.nan
    axes[idx].imshow(img)
    axes[idx].set_axis_off()

fig.tight_layout()

# %%
