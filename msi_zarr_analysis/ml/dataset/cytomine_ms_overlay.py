"translated annotated data via template matching"

import logging
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import zarr
from cytomine.models import AnnotationCollection, ImageInstance
from msi_zarr_analysis.ml.dataset.translate_annotation import (
    TemplateTransform,
    build_onehot_annotation,
    get_destination_mask,
    load_annotation,
    match_template_ms_overlay,
    rasterize_annotation_dict,
    save_bin_class_image,
    translate_annotation_dict,
)
from msi_zarr_analysis.preprocessing.binning import bin_and_flatten
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks
from msi_zarr_analysis.utils.load_spectra import load_spectra

from . import Dataset, GroupCollection, Tabular


def generate_spectra(
    ms_group: zarr.Group,
    ms_template_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
    save_to: str = "",
):
    "this yields all spectra instead of returning a table"

    z_mask, roi = get_destination_mask(
        ms_template_group,
        bin_idx,
        tiff_path,
        tiff_page_idx,
        onehot_annotations,
        transform,
    )

    if save_to:
        tic = ms_template_group["/0"][:, 0, ...].sum(axis=0)

        save_bin_class_image(
            transform.inverse_transform_mask(tic),
            transform.inverse_transform_mask(z_mask),
            save_to=save_to,
        )

    # map the results to the zarr arrays
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]

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
        *,
        term_list: List[str] = None,
        zarr_template_path: str = None,
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

        self.ms_template_group = self.ms_group
        if zarr_template_path:
            self.ms_template_group = open_group_ro(zarr_template_path)
            if self.ms_group["/0"].shape != self.ms_template_group["/0"].shape:
                raise ValueError("inconsistent shape between template and value groups")

        image_instance = ImageInstance().fetch(id=annotation_image_id)

        self.term_names, self.onehot_annotations = build_onehot_annotation(
            annotation_collection=annotations,
            image_height=image_instance.height,
            image_width=image_instance.width,
            term_list=term_list,
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
            self.ms_template_group,
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


def cytomine_translated_with_groups(
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
    attribute_names: List[str] = (),
    *,
    term_list: List[str] = None,
    zarr_template_path: str = None,
    allow_duplicates: bool = False,
) -> GroupCollection:
    """build a dataset after doing template matching on an overlay

    Args:
        annotation_project_id (int): ID of the Cytomine project containing the images
        annotation_image_id (int): ID of the annotated Cytomine ID
        zarr_path (str): path to the Zarr dataset to use for the dataset instances
        bin_idx (int): channel index of the zarr image /0 in zarr_path
        tiff_path (str): path to the tiff file that contains the overlay
        tiff_page_idx (int): index of the page of the tiff_path to perform the overlay on
        transform_template_rot90 (int, optional): How many time the template should be rotated counter clockwise before doing the template matching. Defaults to 0.
        transform_template_flip_ud (bool, optional): Whether the template should be flipped in the vertical dimension before doing the template matching. Defaults to False.
        transform_template_flip_lr (bool, optional): Whether the template should be flipped in the horizontal dimension before doing the template matching. Defaults to False.
        select_users (Iterable[int], optional): List of users to select the annotation, leave empty to avoid any filtering. Defaults to ().
        select_terms (Iterable[int], optional): List of terms to select the annotations, leave empty to avoid any filtering. Defaults to ().
        attribute_names (List[str], optional): Names of each of the features, leave empty to guess from the zarr path. Defaults to ().
        term_list (List[str], optional): Names of the term to use. Defaults to None.
        zarr_template_path (str, optional): path to the Zarr dataset to use for the template matching, use zarr_path if this value is falsy. Defaults to None.
        allow_duplicates (bool, optional): Whether duplicate spectra should be allowed for the same class. Defaults to False.

    Returns:
        GroupCollection: a dataset with groups corresponding to every annotations
    """

    if allow_duplicates:
        logging.warning(
            "allowing duplicates is not recommended and can introduce "
            "biases in the dataset"
        )

    transform_template = TemplateTransform(
        transform_template_rot90,
        transform_template_flip_ud,
        transform_template_flip_lr,
    )

    ms_group = open_group_ro(zarr_path)
    ms_template_group = ms_group

    if zarr_template_path:
        ms_template_group = open_group_ro(zarr_template_path)
        if ms_group["/0"].shape != ms_template_group["/0"].shape:
            raise ValueError("inconsistent shape between template and value groups")

    # template matching between the template and overlay
    matching_result, crop_idx = match_template_ms_overlay(
        ms_group=ms_template_group,
        bin_idx=bin_idx,
        tiff_path=tiff_path,
        tiff_page_idx=tiff_page_idx,
        transform=transform_template,
    )

    # fetch all annotations
    annotations = AnnotationCollection()
    annotations.project = annotation_project_id
    annotations.image = annotation_image_id
    annotations.users = list(select_users)
    annotations.terms = list(select_terms)
    annotations.showTerm = True
    annotations.showWKT = True
    annotations.fetch()
    image_instance = ImageInstance().fetch(id=annotation_image_id)

    # filter matching annotation & correct geometry
    annotation_dict = load_annotation(
        annotations,
        image_height=image_instance.height,
        term_list=term_list,
    )

    # class names
    class_names = term_list
    if not class_names:
        class_names = list(annotation_dict.keys())

    # attribute names
    attribute_names = list(attribute_names)
    if not attribute_names:
        attribute_names = [str(v) for v in ms_group["/labels/mzs/0"][:, 0, 0, 0]]

    # update values from the annotation dict
    translate_annotation_dict(
        annotation_dict,
        template_transform=transform_template,
        matching_result=matching_result,
        crop_idx=crop_idx,
    )

    # have a numpy array like the template for each annotation
    rasterized_dict = rasterize_annotation_dict(
        annotation_dict, ms_template_group["/0"].shape[-2:]
    )

    # ROI for the annotation : select all interesting pixels
    selection = 0
    for annotation_lst in rasterized_dict.values():
        for mask in annotation_lst:
            selection = np.logical_or(mask, selection)
    spectrum_dict = load_spectra(ms_group, selection)

    # collect all annotations as independent Tabular datasets
    dataset_lst: List[Tabular] = []
    for class_idx, annotation_lst in enumerate(rasterized_dict.values()):

        # gather spectrum info into a dict (y, x) -> {group_idx...}
        spectra_choices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for annotation_idx, mask in enumerate(annotation_lst):
            # [(y, x)]
            coordinates = list(zip(*mask.nonzero()))

            for coord in coordinates:
                spectra_choices[coord].append(annotation_idx)

        # regroup per annotation index
        groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for coord, group_choices in spectra_choices.items():

            if allow_duplicates:
                for idx in group_choices:
                    groups[idx].append(coord)
            else:
                if len(group_choices) > 1:
                    logging.info(
                        "spectrum at %s was randomly assigned"
                        " to one of its annotations",
                        coord,
                    )
                idx = random.choice(group_choices)
                groups[idx].append(coord)

        # build dataset for each annotation
        for _, coord_set in groups.items():

            if not coord_set:
                logging.info("empty mask found")
                continue

            spectra = (spectrum_dict.get(c, None) for c in coord_set)
            spectra = [spectrum for spectrum in spectra if spectrum is not None]

            if not spectra:
                logging.info("annotation mapped to no spectra")
                continue

            ds_x = np.stack(spectra)
            ds_y = np.full(shape=ds_x.shape[:1], fill_value=class_idx, dtype=int)

            ds = Tabular(ds_x, ds_y, attribute_names, class_names)
            dataset_lst.append(ds)

    merged = GroupCollection.merge_datasets(*dataset_lst)
    logging.info("merged.data.shape=%s", merged.data.shape)
    logging.info("merged.target.shape=%s", merged.target.shape)

    return merged
