from typing import Tuple

from cytomine.models import SliceInstanceCollection
import pandas as pd


def get_page_bin_indices(
    image_id: int,
    lipid: str,
    csv_lipid_mz_path: str,
) -> Tuple[int, int, float, float]:
    """fetch the TIFF page and the bin's index corresponding to a lipied

    Parameters
    ----------
    image_id : int
        Cytomine id
    lipid : str
        name of the lipid in the CSV and Cytomine namespaces
    csv_path: str
        path to the 'mz value + lipid name.csv' file

    Returns
    -------
    Tuple[int, int]
        TIFF page, bin index

    Raises
    ------
    ValueError
        if the lipid is not found in either Cytomine or the CSV file
    """

    slice_collection = SliceInstanceCollection().fetch_with_filter(
        "imageinstance", image_id
    )
    name_to_slice = dict((slice_.zName, slice_.zStack) for slice_ in slice_collection)

    try:
        tiff_page = name_to_slice[lipid]
    except KeyError as e:
        raise ValueError(f"unable to find {lipid=} in the Cytomine image") from e

    ds = pd.read_csv(csv_lipid_mz_path, sep=None, engine="python")

    row = ds[ds.Name == lipid]

    if len(row) == 0:
        raise ValueError(f"unable to find {lipid=} in the CSV file")

    if len(row) > 1:
        raise ValueError(f"duplicate entries for {lipid=}")
    
    idx = row.index[0]
    center = row.loc[idx, "m/z"]
    width = row.loc[idx, "Interval Width (+/- Da)"]

    return tiff_page, idx, center - .5 * width, center + .5 * width
