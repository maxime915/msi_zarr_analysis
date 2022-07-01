from typing import Tuple, Generator
import warnings

from cytomine.models import (
    SliceInstanceCollection,
    AnnotationCollection,
    Annotation,
    TermCollection,
    Term,
)
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
    Tuple[int, int, float, float]
        TIFF page, bin index, m/Z low, m/Z high

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


def get_lipid_dataframe(
    csv_lipid_mz_path: str
) -> pd.DataFrame:
    return pd.read_csv(csv_lipid_mz_path, sep=None, engine="python")


def get_lipid_names(
    csv_lipid_mz_path: str,
) -> pd.Series:
    "get the names of the lipid in the order that matches the automated binning"
    
    ds = get_lipid_dataframe(csv_lipid_mz_path)
    return ds.Name
    

def iter_annotation_single_term(
    annotation_collection: AnnotationCollection,
    term_collection: TermCollection,
) -> Generator[Tuple[Annotation, Term], None, None]:
    """iterate annotation and their terms in a collection, assuming each annotation
    has only one term.

    Args:
        annotation_collection (AnnotationCollection): source of annotation
        term_collection (TermCollection): collection to map from ID to instances

    Yields:
        Tuple[Annotation, Term]: annotation and their term
    """

    for annotation in annotation_collection:

        if not annotation.term:
            # no term found
            warnings.warn(f"no term for {annotation.id=}")
            continue

        if len(annotation.term) > 1:
            warnings.warn(
                f"too many terms for {annotation.id=} {len(annotation.term)=}"
            )
            continue

        term = term_collection.find_by_attribute("id", annotation.term[0])

        assert term is not None

        yield annotation, term
