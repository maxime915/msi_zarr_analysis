"translated annotated data via template matching"

import warnings
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import rasterio.features
import zarr
from cytomine.models import AnnotationCollection, TermCollection, ImageInstance
from shapely import wkt
from shapely.affinity import affine_transform
from msi_zarr_analysis.ml.dataset.translate_annotation import TemplateTransform, get_destination_mask
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import iter_annoation_single_term

from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

from . import Dataset


def build_onehot_annotation(
    annotation_collection: AnnotationCollection,
    image_height: int,
    image_width: int,
) -> Tuple[List[str], npt.NDArray]:
    # [classes], np[dims..., classes]

    term_collection = TermCollection().fetch_with_filter(
        "project", annotation_collection.project
    )

    mask_dict = {}

    for annotation, term in iter_annoation_single_term(annotation_collection, term_collection):

        # load geometry
        geometry = wkt.loads(annotation.location)
        # change the coordinate system
        geometry = affine_transform(geometry, [1, 0, 0, -1, 0, image_height])
        # rasterize annotation
        mask = rasterio.features.rasterize(
            [geometry], out_shape=(image_height, image_width)
        )

        if not mask.any():
            warnings.warn(f"empty mask found {annotation.id=}")

        try:
            mask_dict[term.name] |= mask
        except KeyError:
            mask_dict[term.name] = mask

    if not mask_dict:
        raise ValueError("no annotation found")

    term_list, mask_list = zip(*mask_dict.items())

    return list(term_list), np.stack(mask_list, axis=-1)



def generate_spectra(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
):
    # map the results to the zarr arrays
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]
    
    z_mask, roi = get_destination_mask(
        ms_group,
        bin_idx,
        tiff_path,
        tiff_page_idx,
        onehot_annotations,
        transform
    )
    
    # yield all rows
    for cy, cx in iter_loaded_chunks(intensities, *roi, skip=2):

        c_len = lengths[0, 0, cy, cx]
        len_cap = c_len.max()  # small optimization for uneven spectra
        c_int = intensities[:len_cap, 0, cy, cx]

        c_mask = z_mask[cy, cx]

        for y, x, class_idx in zip(*c_mask.nonzero()):
            length = c_len[y, x]
            if length == 0:
                continue

            yield c_int[:length, y, x], class_idx


class CytomineTranslated(Dataset):
    """
    NOTE: the connection to a cytomine server via the cytomine python client
    must be established before any method is called.
    """

    def __init__(
        self,
        annotation_project_id: int,
        annotation_image_id: int,
        zarr_path: str,
        bin_idx: int,
        tiff_path: str,
        tiff_page_idx: int,
        transform_template_rot90: int = 0,
        transform_template_flip_ud: bool = False,
        transform_template_flip_lr: bool = False,
        select_users: Iterable[int] = (),
        select_terms: Iterable[int] = (),
        cache_data: bool = True,
        attribute_name_list: List[str] = (),
    ) -> None:
        super().__init__()

        annotations = AnnotationCollection()
        annotations.project = annotation_project_id
        annotations.image = annotation_image_id
        annotations.users = list(select_users)
        annotations.terms = list(select_terms)
        annotations.showTerm = True
        annotations.showWKT = True
        annotations.fetch()

        self.ms_group = open_group_ro(zarr_path)
        self.bin_idx = bin_idx
        self.tiff_path = tiff_path
        self.tiff_page_idx = tiff_page_idx
        self.transform_template = TemplateTransform(
            transform_template_rot90,
            transform_template_flip_ud,
            transform_template_flip_lr,
        )

        image_instance = ImageInstance().fetch(id=annotation_image_id)

        term_names, onehot_annotations = build_onehot_annotation(
            annotation_collection=annotations,
            image_height=image_instance.height,
            image_width=image_instance.width,
        )

        self.term_names = term_names
        self.onehot_annotations = onehot_annotations

        self.cache_data = bool(cache_data)
        self._cached_table = None

        self.attribute_name_list = list(attribute_name_list)

    def __raw_iter(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        yield from generate_spectra(
            self.ms_group,
            self.bin_idx,
            self.tiff_path,
            self.tiff_page_idx,
            self.onehot_annotations,
            self.transform_template,
        )

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:

        if self.cache_data:
            for row in zip(*self.as_table()):
                yield row
            return

        for profile, class_idx in self.__raw_iter():
            yield np.array(profile), class_idx

    def is_table_like(self) -> bool:
        try:
            _ = self.as_table()
            return True
        except (ValueError, IndexError):
            return False

    def __load_ds(self) -> Tuple[npt.NDArray, npt.NDArray]:
        attributes, classes = zip(*self.__raw_iter())
        dtype = attributes[0].dtype
        return np.array(attributes, dtype=dtype), np.array(classes)

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_table:
            self._cached_table = self.__load_ds()

        if not self.cache_data:
            # remove cache if it shouldn't be there
            tmp, self._cached_table = self._cached_table, None
            return tmp

        return self._cached_table

    def attribute_names(self) -> List[str]:
        if self.attribute_name_list:
            return self.attribute_name_list
        return [str(v) for v in self.ms_group["/labels/mzs/0"][:, 0, 0, 0]]
