"translated annotated data via template matching"

import functools
import logging
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Mapping, Tuple, Sequence, Union

import numpy as np
import numpy.typing as npt
import zarr
from cytomine.models import AnnotationCollection, ImageInstance
from msi_zarr_analysis.ml.dataset.translate_annotation import (
    ParsedAnnotation,
    TemplateTransform,
    build_onehot_annotation,
    get_annotation_mapping,
    get_destination_mask,
    match_template_ms_overlay,
    match_template_ms_overlay_multi,
    parse_annotation_mapping,
    rasterize_annotation_mapping,
    save_bin_class_image,
    translate_parsed_annotation_mapping,
)
from msi_zarr_analysis.preprocessing.binning import bin_and_flatten
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks
from msi_zarr_analysis.utils.load_spectra import load_intensities, load_spectra

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
        tic: np.ndarray = ms_template_group["/0"][:, 0, ...].sum(axis=0)  # type: ignore

        save_bin_class_image(
            transform.inverse_transform_mask(tic),
            transform.inverse_transform_mask(z_mask),
            save_to=save_to,
        )

    # map the results to the zarr arrays
    intensities: zarr.Array = ms_group["/0"]  # type: ignore
    lengths: zarr.Array = ms_group["/labels/lengths/0"]  # type:ignore

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
        select_users: Sequence[int] = (),
        select_terms: Sequence[int] = (),
        cache_data: bool = True,
        attribute_name_list: Sequence[str] = (),
        save_image: Union[bool, str] = False,
        *,
        term_list: Sequence[str] = (),
        zarr_template_path: str = "",
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

    def class_names(self) -> Sequence[str]:
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
        select_users: Sequence[int] = (),
        select_terms: Sequence[int] = (),
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

        dataset_x = np.empty((n_rows, n_bins), dtype=z_ints.dtype)  # type: ignore
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


def get_overlay_annotations(
    project_id: int,
    image_id: int,
    classes: Mapping[str, Sequence[int]],
    *,
    select_users: Sequence[int] = (),
) -> Dict[str, List[ParsedAnnotation]]:

    # get unique list of terms to fetch
    select_terms = []
    for ids in classes.values():
        select_terms.extend(ids)
    select_terms = list(set(select_terms))

    # fetch all annotations
    annotations = AnnotationCollection()
    annotations.project = project_id
    annotations.image = image_id
    annotations.users = list(select_users)
    annotations.terms = select_terms
    annotations.showTerm = True
    annotations.showWKT = True
    annotations.fetch()
    image = ImageInstance().fetch(id=image_id)

    annotation_mapping = get_annotation_mapping(annotations, classes)

    return parse_annotation_mapping(annotation_mapping, image.height, image.width)


@functools.lru_cache(maxsize=8)
def __overlay_template_matching_multi_lipid_cached(
    zarr_template_path: str,
    zarr_channel: Tuple[int, ...],
    overlay_tiff_path: str,
    overlay_pages: Tuple[int, ...],
    transform_template_rot90: int = 0,
    transform_template_flip_ud: bool = False,
    transform_template_flip_lr: bool = False,
):
    template_transform = TemplateTransform(
        transform_template_rot90,
        transform_template_flip_ud,
        transform_template_flip_lr,
    )

    ms_template_group = open_group_ro(zarr_template_path)

    # template matching between the template and overlay
    matching_result, crop_idx = match_template_ms_overlay_multi(
        ms_group=ms_template_group,
        bin_idx=zarr_channel,
        tiff_path=overlay_tiff_path,
        tiff_page_idx=overlay_pages,
        transform=template_transform,
    )

    logging.info("crop_idx: %s", crop_idx)
    logging.info("ms_template.shape: %s", ms_template_group["/0"].shape)
    logging.info("matching_results: %s", matching_result)

    return template_transform, matching_result, crop_idx


def translate_annotation_mapping_overlay_to_template_multi_lipid(
    annotation_mapping: Mapping[str, Sequence[ParsedAnnotation]],
    zarr_template_path: str,
    lipid_to_channel: Mapping[str, int],
    overlay_tiff_path: str,
    lipid_to_page: Mapping[str, int],
    transform_template_rot90: int = 0,
    transform_template_flip_ud: bool = False,
    transform_template_flip_lr: bool = False,
    *,
    clear_cache: bool = False,
) -> Dict[str, List[ParsedAnnotation]]:

    if clear_cache:
        __overlay_template_matching_multi_lipid_cached.cache_clear()

    common = set(lipid_to_channel.keys()).intersection(lipid_to_page.keys())
    channel_indices = tuple(lipid_to_channel[lipid] for lipid in common)
    page_indices = tuple(lipid_to_page[lipid] for lipid in common)

    template_transform, matching_result, crop_idx = (
        __overlay_template_matching_multi_lipid_cached(
            zarr_template_path,
            channel_indices,
            overlay_tiff_path,
            page_indices,
            transform_template_rot90,
            transform_template_flip_ud,
            transform_template_flip_lr,
        )
    )

    return translate_parsed_annotation_mapping(
        annotation_mapping, template_transform, matching_result, crop_idx
    )


@functools.lru_cache(maxsize=8)
def __overlay_template_matching_cached(
    zarr_template_path: str,
    zarr_channel_index: int,
    overlay_tiff_path: str,
    overlay_tiff_page: int,
    transform_template_rot90: int = 0,
    transform_template_flip_ud: bool = False,
    transform_template_flip_lr: bool = False,
):
    template_transform = TemplateTransform(
        transform_template_rot90,
        transform_template_flip_ud,
        transform_template_flip_lr,
    )

    ms_template_group = open_group_ro(zarr_template_path)

    # template matching between the template and overlay
    matching_result, crop_idx = match_template_ms_overlay(
        ms_group=ms_template_group,
        bin_idx=zarr_channel_index,
        tiff_path=overlay_tiff_path,
        tiff_page_idx=overlay_tiff_page,
        transform=template_transform,
    )

    logging.info("crop_idx: %s", crop_idx)
    logging.info("ms_template.shape: %s", ms_template_group["/0"].shape)
    logging.info("matching_results: %s", matching_result)

    return template_transform, matching_result, crop_idx


def translate_annotation_mapping_overlay_to_template(
    annotation_mapping: Mapping[str, Sequence[ParsedAnnotation]],
    zarr_template_path: str,
    zarr_channel_index: int,
    overlay_tiff_path: str,
    overlay_tiff_page: int,
    transform_template_rot90: int = 0,
    transform_template_flip_ud: bool = False,
    transform_template_flip_lr: bool = False,
    *,
    clear_cache: bool = False,
) -> Dict[str, List[ParsedAnnotation]]:

    if clear_cache:
        __overlay_template_matching_cached.cache_clear()

    template_transform, matching_result, crop_idx = __overlay_template_matching_cached(
        zarr_template_path,
        zarr_channel_index,
        overlay_tiff_path,
        overlay_tiff_page,
        transform_template_rot90,
        transform_template_flip_ud,
        transform_template_flip_lr,
    )

    return translate_parsed_annotation_mapping(
        annotation_mapping, template_transform, matching_result, crop_idx
    )


def build_spectrum_dict(
    annotation_mapping: Mapping[str, Sequence[ParsedAnnotation]],
    zarr_path: str,
    *,
    allow_duplicates: bool = False,
) -> Tuple[
    Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    Dict[str, Dict[int, List[Tuple[int, int]]]],
]:

    ms_group = open_group_ro(zarr_path)
    image_dimensions: Tuple[int, int] = ms_group["/0"].shape[-2:]  # type: ignore

    # have a numpy array like the template for each annotation
    rasterized_annotations = rasterize_annotation_mapping(
        annotation_mapping,
        image_dimensions,
    )

    # ROI for the annotation : select all interesting pixels
    selection = np.zeros(image_dimensions, dtype=int)
    for annotation_lst in rasterized_annotations.values():
        for annotation in annotation_lst:
            selection = np.logical_or(annotation.raster, selection)
    spectrum_dict = load_spectra(ms_group, selection)

    # collect all annotations as independent Tabular datasets
    annotation_coordinates: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
    for class_name, annotation_lst in rasterized_annotations.items():

        # gather spectrum info into a dict (y, x) -> {group_idx...}
        spectra_choices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for annotation_idx, annotation in enumerate(annotation_lst):
            # [(y, x)]
            coordinates: List[Tuple[int, int]] = list(zip(*annotation.raster.nonzero()))  # type: ignore

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

        # convert to a normal dict
        annotation_coordinates[class_name] = dict(groups)

    return spectrum_dict, annotation_coordinates


def make_collection(
    spectrum_dict: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    annotation_coordinates: Dict[str, Dict[int, List[Tuple[int, int]]]],
    attribute_names: Sequence[str],
) -> GroupCollection:

    # class names
    class_names = list(annotation_coordinates.keys())

    # attribute names
    attribute_names = list(attribute_names)

    # collect all annotations as independent Tabular datasets
    dataset_lst: List[Tabular] = []
    for class_idx, groups in enumerate(annotation_coordinates.values()):

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


def make_rasterized_mask(
    annotation_mapping: Mapping[str, Sequence[ParsedAnnotation]],
    attribute_names: Sequence[str],
    zarr_path: str,
):
    ms_group = open_group_ro(zarr_path)
    dataset: zarr.Array = ms_group["/0"]  # type: ignore
    image_dimensions = dataset.shape[-2:]

    # class names
    class_names = list(annotation_mapping.keys())

    # attribute names
    attribute_names = list(attribute_names)
    if not attribute_names:
        attribute_names = [str(v) for v in ms_group["/labels/mzs/0"][:, 0, 0, 0]]

    # have a numpy array like the template for each annotation
    rasterized_annotations = rasterize_annotation_mapping(
        annotation_mapping,
        image_dimensions,
    )

    yx_tic: np.ndarray = dataset[:, 0, :, :].sum(axis=0)
    foreground: np.ndarray = ms_group["/labels/lengths/0"][0, 0, :, :] > 0  # type: ignore

    return foreground, yx_tic, rasterized_annotations


def collect_spectra_zarr(
    annotation_mapping: Mapping[str, Sequence[ParsedAnnotation]],
    attribute_names: Sequence[str],
    zarr_path: str,
    *,
    allow_duplicates: bool = False,
) -> GroupCollection:

    if allow_duplicates:
        logging.warning(
            "allowing duplicates is not recommended and can introduce "
            "biases in the dataset"
        )

    ms_group = open_group_ro(zarr_path)
    image_dimensions: Tuple[int, int] = ms_group["/0"].shape[-2:]  # type: ignore

    # class names
    class_names = list(annotation_mapping.keys())

    # attribute names
    attribute_names = list(attribute_names)
    if not attribute_names:
        attribute_names = [str(v) for v in ms_group["/labels/mzs/0"][:, 0, 0, 0]]

    # have a numpy array like the template for each annotation
    rasterized_annotations = rasterize_annotation_mapping(
        annotation_mapping,
        image_dimensions,
    )

    # ROI for the annotation : select all interesting pixels
    selection = np.zeros(image_dimensions, int)
    for annotation_lst in rasterized_annotations.values():
        for annotation in annotation_lst:
            selection = np.logical_or(annotation.raster, selection)
    spectrum_dict = load_intensities(ms_group, selection)

    # collect all annotations as independent Tabular datasets
    dataset_lst: List[Tabular] = []
    for class_idx, annotation_lst in enumerate(rasterized_annotations.values()):

        # gather spectrum info into a dict (y, x) -> {group_idx...}
        spectra_choices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for annotation_idx, annotation in enumerate(annotation_lst):
            # [(y, x)]
            coordinates: List[Tuple[int, int]] = list(zip(*annotation.raster.nonzero()))  # type: ignore

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
    select_users: Sequence[int] = (),
    select_terms: Sequence[int] = (),
    attribute_names: Sequence[str] = (),
    *,
    term_list: Sequence[str] = (),
    zarr_template_path: str = "",
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

    annotation_dict = get_overlay_annotations(
        annotation_project_id,
        annotation_image_id,
        classes={term: [id_] for term, id_ in zip(term_list, select_terms)},
        select_users=select_users,
    )

    annotation_dict = translate_annotation_mapping_overlay_to_template(
        annotation_mapping=annotation_dict,
        zarr_template_path=zarr_template_path or zarr_path,
        zarr_channel_index=bin_idx,
        overlay_tiff_path=tiff_path,
        overlay_tiff_page=tiff_page_idx,
        transform_template_rot90=transform_template_rot90,
        transform_template_flip_ud=transform_template_flip_ud,
        transform_template_flip_lr=transform_template_flip_lr,
    )

    return collect_spectra_zarr(
        annotation_dict,
        attribute_names,
        zarr_path,
        allow_duplicates=allow_duplicates,
    )
