"translated annotated data via template matching"

from typing import Iterable, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import zarr
from cytomine.models import AnnotationCollection, ImageInstance
from msi_zarr_analysis.ml.dataset.translate_annotation import (
    TemplateTransform,
    build_onehot_annotation,
    get_destination_mask,
    save_bin_class_image,
)
from msi_zarr_analysis.preprocessing.binning import bin_and_flatten
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

from . import Dataset, Tabular


def generate_spectra(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
    save_to: str = "",
):
    "this yields all spectra instead of returning a table"

    # map the results to the zarr arrays
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]

    z_mask, roi = get_destination_mask(
        ms_group,
        bin_idx,
        tiff_path,
        tiff_page_idx,
        onehot_annotations,
        transform,
    )

    save_bin_class_image(ms_group, z_mask, save_to=save_to)

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
        save_image: bool = False,
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

        self.term_names, self.onehot_annotations = build_onehot_annotation(
            annotation_collection=annotations,
            image_height=image_instance.height,
            image_width=image_instance.width,
        )

        self.cache_data = bool(cache_data)
        self._cached_table = None

        self.attribute_name_list = list(attribute_name_list)

        self.save_to = ""
        if save_image:
            self.save_to = (
                f"saved_overlay_{image_instance.name or image_instance.filename}"
                + f"_{self.term_names[0]}_{self.term_names[1]}.png"
            )
            if isinstance(save_image, str):  # act as a prefix
                self.save_to = save_image + "_" + self.save_to

    def __raw_iter(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        yield from generate_spectra(
            self.ms_group,
            self.bin_idx,
            self.tiff_path,
            self.tiff_page_idx,
            self.onehot_annotations,
            self.transform_template,
            save_to=self.save_to,
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

    def class_names(self) -> List[str]:
        return self.term_names


class CytomineTranslatedProgressiveBinningFactory:
    def __init__(
        self,
        annotation_project_id: int,
        annotation_image_id: int,
        zarr_binned_path: str,
        bin_idx: int,
        tiff_path: str,
        tiff_page_idx: int,
        transform_template_rot90: int = 0,
        transform_template_flip_ud: bool = False,
        transform_template_flip_lr: bool = False,
        select_users: Iterable[int] = (),
        select_terms: Iterable[int] = (),
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

        self.binned_group = open_group_ro(zarr_binned_path)
        self.bin_idx = bin_idx
        self.tiff_path = tiff_path
        self.tiff_page_idx = tiff_page_idx
        self.transform_template = TemplateTransform(
            transform_template_rot90,
            transform_template_flip_ud,
            transform_template_flip_lr,
        )

        image_instance = ImageInstance().fetch(id=annotation_image_id)

        self.term_names, self.onehot_annotations = build_onehot_annotation(
            annotation_collection=annotations,
            image_height=image_instance.height,
            image_width=image_instance.width,
        )

    def _build_mask(self):

        attr_name = "__cached_build_mask"

        if hasattr(self, attr_name):
            return getattr(self, attr_name)

        mask, roi = get_destination_mask(
            self.binned_group,
            self.bin_idx,
            self.tiff_path,
            self.tiff_page_idx,
            self.onehot_annotations,
            self.transform_template,
        )

        setattr(self, attr_name, (mask, roi))
        return (mask, roi)
    
    @property
    def dataset_rows(self) -> int:
        (
            mask,  # array[y, x, n_class] one-hot encoding of the annotations
            _,  # (y_slice, x_slice) for the region of interest
        ) = self._build_mask()
        
        n_rows = mask.sum()  
        
        return n_rows      

    def bin(
        self, processed_group: zarr.Group, bin_lo: np.ndarray, bin_hi: np.ndarray
    ) -> Dataset:

        assert bin_lo.shape == bin_hi.shape, "inconsistent bounds"
        assert all(lo < hi for lo, hi in zip(bin_lo, bin_hi)), "inconsistent bounds"

        (
            mask,  # array[y, x, n_class] one-hot encoding of the annotations
            roi,  # (y_slice, x_slice) for the region of interest
        ) = self._build_mask()

        n_rows = mask.sum()
        n_bins = bin_lo.size

        z_ints = processed_group["/0"]

        dataset_x = np.empty((n_rows, n_bins), dtype=z_ints.dtype)
        dataset_y = np.empty((n_rows,), dtype=int)

        bin_and_flatten(
            dataset_x=dataset_x,
            dataset_y=dataset_y,
            z=processed_group,
            onehot_cls=mask,
            y_slice=roi[0],
            x_slice=roi[1],
            bin_lo=bin_lo,
            bin_hi=bin_hi,
        )

        bin_names = [f"{lo}-{hi}" for lo, hi in zip(bin_lo, bin_hi)]

        return Tabular(dataset_x, dataset_y, bin_names, self.term_names)
