"translated annotated data via template matching"

from collections import defaultdict
import logging
import warnings
from typing import Dict, List, NamedTuple, Tuple, overload

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio.features
import tifffile
import zarr
from cytomine.models import Annotation, AnnotationCollection, TermCollection
from msi_zarr_analysis.utils.cytomine_utils import iter_annotation_single_term
from PIL import Image
from scipy.optimize import minimize_scalar
from shapely import affinity, wkt
from shapely.geometry import Polygon
from skimage.feature import match_template

# Datatype definitions


class TemplateTransform(NamedTuple):
    # hom many time should the template be rotated (np.rot90(..., k=))
    rotate_90: int = 0
    # should the template be flipped ?
    flip_ud: bool = False
    flip_lr: bool = False

    @staticmethod
    def none():
        return TemplateTransform()

    def transform_template(self, template):
        "suppose YX array"
        if self.flip_ud:
            template = np.flip(template, axis=0)
        if self.flip_lr:
            template = np.flip(template, axis=1)

        if self.rotate_90 != 0:
            template = np.rot90(template, k=self.rotate_90)

        return np.ascontiguousarray(template)

    def inverse_transform_mask(self, mask):
        "suppose YX array"
        if self.rotate_90 != 0:
            mask = np.rot90(mask, k=-self.rotate_90)

        if self.flip_lr:
            mask = np.flip(mask, axis=1)
        if self.flip_ud:
            mask = np.flip(mask, axis=0)

        return np.ascontiguousarray(mask)

    def inverse_transform_coordinate(self, y, x, shape):
        "y, x: numpy arrays, shape: (height, width [, ...])"

        # keep in the [-2, 2) range
        rot_90 = (self.rotate_90 + 2) % 4 - 2

        # do rotation (counter clockwise)
        for _ in range(0, -rot_90):
            # rotate index
            x, y = y, (shape[1] - 1 - x)
            # rotate shape
            shape = (shape[1], shape[0]) + shape[2:]

        # do rotation (clockwise)
        for _ in range(-rot_90, 0):
            # rotate index
            x, y = (shape[0] - 1 - y), x
            # rotate shape
            shape = (shape[1], shape[0]) + shape[2:]

        # flip LR
        if self.flip_lr:
            x = shape[1] - 1 - x

        # flip UD
        if self.flip_ud:
            y = shape[0] - 1 - y

        return y, x, shape


class MatchingResult(NamedTuple):
    # coordinates in the source coordinates for the top left of the scaled matching template
    x_top_left: int
    y_top_left: int
    # how much should the template be scaled to match the source
    scale: float
    # area
    y_slice: slice
    x_slice: slice

    def map_yx(self, y_source, x_source):
        "map some (y, x) coordinates from the (source) image to the pre-processed template"

        # translate
        y = y_source - self.y_top_left
        x = x_source - self.x_top_left

        # scale down
        y = y / self.scale
        x = x / self.scale

        # integer pixel (floored down)
        y = np.int32(np.floor(y))
        x = np.int32(np.floor(x))

        return y, x

    def map_yx_reverse(self, y_template, x_template):
        """"""

        y = y_template * self.scale
        x = x_template * self.scale

        y = y + self.y_top_left
        x = x + self.x_top_left

        y = np.int32(np.floor(y))
        x = np.int32(np.floor(x))

        return y, x

    def map_xy(self, x_source, y_source):
        "map some (x, y) coordinates from the (source) image to the pre-processed template"

        y, x = self.map_yx(y_source, x_source)
        return x, y

    map_ij = map_yx


# annotation masks


def translate_geometry(
    image_geometry,
    template_transform: TemplateTransform,
    matching_result: MatchingResult,
    crop_idx: Tuple[slice, slice],
):
    """translate annotation from the overlay to the template

    Args:
        image_geometry (Polygon): geometry of the annotation, with image coordinates (XY at the top left, Y axis descending)
        template_transform (TemplateTransform): transformation applied to the template, pre template-matching
        matching_result (MatchingResult): results of the template matching algorithm
        crop_idx (Tuple[slice, slice]): cropping of the template, pre template-matching

    Returns:
        Annotation: translated annotation
    """

    # translation is based on the template rectangle, not the annotation
    rect = Polygon(
        (
            (matching_result.x_slice.start, matching_result.y_slice.start),
            (matching_result.x_slice.stop, matching_result.y_slice.start),
            (matching_result.x_slice.stop, matching_result.y_slice.stop),
            (matching_result.x_slice.start, matching_result.y_slice.stop),
        )
    )
    geometry = image_geometry

    # 1. template matching

    # 1.1 translate
    geometry = affinity.translate(
        geometry, xoff=-matching_result.x_top_left, yoff=-matching_result.y_top_left
    )
    rect = affinity.translate(
        rect, xoff=-matching_result.x_top_left, yoff=-matching_result.y_top_left
    )

    # 1.2 scale
    f = 1 / matching_result.scale
    geometry = affinity.scale(geometry, f, f, origin=(0, 0))
    rect = affinity.scale(rect, f, f, origin=(0, 0))

    # 2. (template) transform

    # 2.0 translate such that (0, 0) is the center of the rect
    center = rect.centroid
    geometry = affinity.translate(geometry, -center.x, -center.y)
    rect = affinity.translate(rect, -center.x, -center.y)

    # 2.1. apply rotation
    geometry = affinity.rotate(
        geometry,
        template_transform.rotate_90 * 90,
        origin=(0, 0),
    )
    rect = affinity.rotate(
        rect,
        template_transform.rotate_90 * 90,
        origin=(0, 0),
    )

    # 2.2. apply flipping
    if template_transform.flip_lr:
        geometry = affinity.affine_transform(geometry, [-1, 0, 0, 1, 0, 0])
        # rectangle is unaffected
    if template_transform.flip_ud:
        geometry = affinity.affine_transform(geometry, [1, 0, 0, -1, 0, 0])
        # rectangle is unaffected

    # 3. cropping
    (min_x, min_y, _, _) = rect.bounds
    geometry = affinity.translate(
        geometry,
        crop_idx[1].start - min_x,
        crop_idx[0].start - min_y,
    )
    # no need to translate rect, it is unused

    return geometry


def rasterize_annotation_dict(
    annotation_dict: Dict[str, List[Annotation]],
    dest_shape: Tuple[int, int],
    *,
    key="template_geometry",
) -> Dict[str, List[np.ndarray]]:
    """make a dict of rasterized annotations for each term

    Args:
        annotation_dict (Dict[str, List[Annotation]]): dict of annotations for each term
        dest_shape (Tuple[int, int]): height and width of the template

    Returns:
        Dict[str, List[np.ndarray]]: dict of rasterized annotations for each term
    """

    def rasterize(annotation):
        if isinstance(key, str):
            geometry = getattr(annotation, key)
        else:
            try:
                geometry = key(annotation)
            except TypeError as e:
                raise ValueError("key must be a str or a callable") from e

        mask = rasterio.features.rasterize(
            [geometry],
            out_shape=dest_shape,
            all_touched=False,
        )

        return mask

    return {
        term: [rasterize(annotation) for annotation in annotation_list]
        for term, annotation_list in annotation_dict.items()
    }


def translate_annotation_dict(
    annotation_dict: Dict[str, List[Annotation]],
    template_transform: TemplateTransform,
    matching_result: MatchingResult,
    crop_idx: Tuple[slice, slice],
):
    for annotation_lst in annotation_dict.values():
        for annotation_item in annotation_lst:
            annotation_item.template_geometry = translate_geometry(
                annotation_item.image_geometry,
                template_transform,
                matching_result,
                crop_idx,
            )


def add_image_geometry(
    annotation: Annotation,
    image_height: int,
):
    # load geometry
    geometry = wkt.loads(annotation.location)
    # change the coordinate system
    geometry = affinity.affine_transform(geometry, [1, 0, 0, -1, 0, image_height])
    # store it in the annotation
    annotation.image_geometry = geometry


def load_annotation(
    annotation_collection: AnnotationCollection,
    image_height: int,
    *,
    term_list: List[str],
) -> Dict[str, List[Annotation]]:

    term_collection = TermCollection().fetch_with_filter(
        "project", annotation_collection.project
    )

    polygon_sets_per_term: Dict[str, set] = defaultdict(set)
    annotation_dict = {term: [] for term in term_list}

    for annotation, term in iter_annotation_single_term(
        annotation_collection, term_collection
    ):
        # avoid duplicate annotations
        if annotation.location in polygon_sets_per_term[term]:
            continue
        polygon_sets_per_term[term].add(annotation.location)

        add_image_geometry(annotation, image_height)

        try:
            annotation_dict[term.name].append(annotation)
        except KeyError as e:
            raise ValueError(f"invalid term {term} found") from e
    return annotation_dict


def build_onehot_annotation(
    annotation_collection: AnnotationCollection,
    image_height: int,
    image_width: int,
    *,
    term_list: List[str] = None,
) -> Tuple[List[str], npt.NDArray]:
    # [classes], np[dims..., classes]

    term_collection = TermCollection().fetch_with_filter(
        "project", annotation_collection.project
    )

    mask_dict = {}

    for annotation, term in iter_annotation_single_term(
        annotation_collection, term_collection
    ):

        # load geometry
        geometry = wkt.loads(annotation.location)
        # change the coordinate system
        geometry = affinity.affine_transform(geometry, [1, 0, 0, -1, 0, image_height])
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

    if term_list:
        if set(term_list) != set(mask_dict.keys()):
            raise ValueError(
                f"{term_list=!r} inconsistent with {list(mask_dict.keys())=}"
            )

        stacks = [mask_dict[term] for term in term_list]
        return term_list, np.stack(stacks, axis=-1)

    term_list, mask_list = zip(*mask_dict.items())

    return list(term_list), np.stack(mask_list, axis=-1)


# Data loading (from disk)


def load_ms_template(
    z_group: zarr.Group,
    bin_idx: int,
) -> Tuple[Tuple[slice, slice], npt.NDArray]:
    "use m/Z bounds ? use the CSV ? use the binned array directly ?"

    ms_data = z_group["/0"][bin_idx, 0, ...]
    nz_y, nz_x = ms_data.nonzero()

    # .min() will raise an Exception on empty array
    if nz_y.size == 0:
        return ms_data, (slice(None), slice(None))

    crop_idx = (
        slice(nz_y.min(), 1 + nz_y.max()),
        slice(nz_x.min(), 1 + nz_x.max()),
    )

    ms_data = ms_data[crop_idx]

    return crop_idx, ms_data


def load_tif_file(page_idx: int, disk_path: str):
    "load the overlay data from a local path"

    store = tifffile.imread(disk_path, aszarr=True)
    z = zarr.open(store)  # pages, chan, height, width

    # first page
    image = z[page_idx, ...]  # C=[RGB], H, W
    image = np.transpose(image, (1, 2, 0))  # H, W, C

    if len(image.shape) != 3:
        raise ValueError(f"invalid {image.shape=} (expected 3 elements)")
    if image.shape[2] != 3:
        raise ValueError(f"expected third axis to be 3 channels ({image.shape=})")

    return image


# Image transformation


def rgb_to_grayscale(rgb):
    "H, W, C=[RGB]"
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def colorize_data(intensity, cmap=plt.cm.viridis):
    # H, W, C=[RGBA]
    colored = plt.cm.ScalarMappable(cmap=plt.cm.viridis).to_rgba(intensity)
    colored = np.uint8(np.round(255 * colored))  # as 8bit color
    colored = colored[:, :, :3]  # strip alpha
    return colored


def scale_image(
    image: npt.NDArray, scale: float, mode: int = cv2.INTER_NEAREST
) -> npt.NDArray:
    """resize an image with a given scale

    Args:
        image (npt.NDArray): source image
        scale (float): scaling factor
        mode (int, optional): OpenCV interpolation mode. Defaults to cv2.INTER_NEAREST.

    Returns:
        npt.NDArray: scaled image
    """
    return cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=mode)


# pre-processing


def filter_hsv(
    image_hsv: np.ndarray,
    *,
    threshold_hue_low: np.uint8 = 0,
    threshold_hue_high: np.uint8 = 180,
    threshold_saturation_low: np.uint8 = 0,
    threshold_value_low: np.uint8 = 0,
) -> np.ndarray:
    image_hsv[image_hsv[..., 0] < threshold_hue_low] = 0
    image_hsv[image_hsv[..., 0] > threshold_hue_high] = 0
    image_hsv[image_hsv[..., 1] < threshold_saturation_low] = 0
    image_hsv[image_hsv[..., 2] < threshold_value_low] = 0
    return image_hsv


def threshold_overlay(overlay_img):
    "select MS peaks from the overlay based on the colormap (assuming default)"

    overlay_hsv = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2HSV)
    filtered = filter_hsv(
        overlay_hsv,
        threshold_saturation_low=60,
        threshold_hue_low=20,
        threshold_hue_high=110,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_HSV2RGB)


def threshold_ms(color_ms_data):

    template_hsv = cv2.cvtColor(color_ms_data, cv2.COLOR_RGB2HSV)
    filtered = filter_hsv(
        template_hsv,
        threshold_hue_high=110,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_HSV2RGB)


# Template matching implementation


def get_template_matching_score(grayscale_overlay, grayscale_ms_data):
    "template matching via best translation"

    # translate the image and compute the correlation score
    result = match_template(grayscale_overlay, grayscale_ms_data)

    # find best translation's coordinates
    max_score_idx = np.argmax(result)
    ij = np.unravel_index(max_score_idx, result.shape)
    y, x = ij

    max_score = result[ij]
    return max_score, (x, y)


def match_template_multiscale_scipy(
    grayscale_overlay,
    grayscale_ms_data,
    max_scale: float,
    tol: float = 1e-3,
):
    # we can refine the interval using bound_scale_in_sample
    low = 1.0
    high = max_scale

    def cost_fun(scale: float) -> float:
        if scale > high or scale < low:
            # return a positive value to grow when going out of the interval
            return 2 * np.abs(scale - 0.5 * (low + high))

        scaled = scale_image(grayscale_ms_data, scale)

        score, _ = get_template_matching_score(grayscale_overlay, scaled)

        return -score

    def get_match_info(scale: float):
        # build scaled image
        if scale > high or scale < low:
            raise ValueError("invalid scale")

        scaled = scale_image(grayscale_ms_data, scale)

        score, coords = get_template_matching_score(grayscale_overlay, scaled)

        return score, scaled.shape, coords

    res = minimize_scalar(
        cost_fun,
        bracket=(low, high),
        tol=tol,
        method="Brent",
    )

    best_score, (height, width), (x, y) = get_match_info(res.x)

    return best_score, res.x, (height, width), (x, y)


def match_template_multiscale(
    image_rgb,
    template_rgb,
) -> MatchingResult:
    "the template should already be transformed to be aligned with the image"

    image_th = threshold_overlay(image_rgb)
    image_th_gs = rgb_to_grayscale(image_th)

    template_th = threshold_ms(template_rgb)
    template_th_gs = rgb_to_grayscale(template_th)

    max_scale = min(
        o_s / t_s for o_s, t_s in zip(image_th_gs.shape, template_th_gs.shape)
    )

    (score, scale, (height, width), (tl_x, tl_y)) = match_template_multiscale_scipy(
        image_th_gs, template_th_gs, max_scale
    )

    return MatchingResult(
        tl_x,
        tl_y,
        scale,
        y_slice=slice(tl_y, tl_y + height),
        x_slice=slice(tl_x, tl_x + width),
    )


def get_destination_mask_from_result(
    onehot_annotation: npt.NDArray,
    yx_dest_shape: Tuple[int, int],
    transform: TemplateTransform,
    match_result: MatchingResult,
    crop_idx: Tuple[slice, slice],
    ms_template_shape: Tuple[int, int],
):
    """yield all spectra-class tuples from the dataset

    Parameters
    ----------
    onehot_annotation : npt.NDArray
        array of shape [height, width, nClasses] indicating the classes mapped to a (y, x) pixel
    ms_group : zarr.Group
        the group containing the MS data
    transform : TemplateTransform
        the transformation to build the template from the ms data
    match_result : MatchingResult
        the results from the template matching algorithm
    crop_idx : Tuple[slice, slice]
        where the (transformed) template can be found in the (transformed) ms data
    ms_template_shape : Tuple[int, int]
        the shape of the template (after transformation)

    Yields
    ------
    Tuple[npt.NDArray, int]
        spectra and class
    """

    y_overlay, x_overlay = np.nonzero(onehot_annotation.any(axis=-1))

    y_template, x_template = match_result.map_yx(y_overlay, x_overlay)

    y_cropped, x_cropped, _ = transform.inverse_transform_coordinate(
        y_template, x_template, ms_template_shape
    )

    y_ms = y_cropped + crop_idx[0].start
    x_ms = x_cropped + crop_idx[1].start

    # map the results to the zarr arrays
    shape = yx_dest_shape + onehot_annotation.shape[-1:]
    z_mask = np.zeros(shape, dtype=bool)

    try:
        z_mask[y_ms, x_ms, :] = onehot_annotation[y_overlay, x_overlay, :]
    except IndexError as e:
        logging.error("unable to translate annotation: %s", e)

        logging.error("z_mask.shape: %s", z_mask.shape)
        logging.error("y_ms: [%d, %d]", y_ms.min(), y_ms.max())
        logging.error("x_ms: [%d, %d]", x_ms.min(), x_ms.max())

        logging.error("onehot_annotation.shape: %s", onehot_annotation.shape)
        logging.error("y_overlay: [%d, %d]", y_overlay.min(), y_overlay.max())
        logging.error("x_overlay: [%d, %d]", x_overlay.min(), x_overlay.max())

        raise e

    # build ROI

    roi = (
        slice(int(y_ms.min()), int(1 + y_ms.max())),
        slice(int(x_ms.min()), int(1 + x_ms.max())),
    )

    return z_mask, roi


def match_template_ms_overlay(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    transform: TemplateTransform = TemplateTransform.none(),
) -> Tuple[MatchingResult, Tuple[slice, slice]]:
    """perform the template matching algorithm, using the MS data as a template
    on the overlay image.

    Args:
        ms_group (zarr.Group): MS data, stored in OME-NGFF compliant format
        bin_idx (int): channel of the MS data that should be used as template
        tiff_path (str): path to the overlay tiff file
        tiff_page_idx (int): page of the overlay to be used as image
        transform (TemplateTransform, optional): transformation to apply to the \
            template before performing the algorithm. Defaults to TemplateTransform.none().

    Returns:
        Tuple[
            MatchingResult, : the mapping from the (transformed) template to the \
                image
            Tuple[slice, slice], : an index for the region used as template in \
                the MS data
        ]
    """

    overlay = load_tif_file(page_idx=tiff_page_idx, disk_path=tiff_path)

    crop_idx, ms_template = load_ms_template(ms_group, bin_idx=bin_idx)

    ms_template = transform.transform_template(ms_template)

    matching_result = match_template_multiscale(
        overlay,
        colorize_data(ms_template),
    )

    return matching_result, crop_idx


def get_template_matching_data(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
) -> dict:

    overlay = load_tif_file(page_idx=tiff_page_idx, disk_path=tiff_path)

    crop_idx, ms_template = load_ms_template(ms_group, bin_idx=bin_idx)

    ms_template = transform.transform_template(ms_template)

    matching_result = match_template_multiscale(
        overlay,
        colorize_data(ms_template),
    )

    logging.info("crop_idx: %s", crop_idx)
    logging.info("ms_template.shape: %s", ms_template.shape)
    logging.info("matching_results: %s", matching_result)

    return dict(
        onehot_annotation=onehot_annotations,
        yx_dest_shape=ms_group["/0"].shape[2:],
        transform=transform,
        match_result=matching_result,
        crop_idx=crop_idx,
        ms_template_shape=ms_template.shape,
    )


def save_bin_class_image(
    tic: npt.NDArray,
    ms_mask: npt.NDArray,
    save_to: str = "",
) -> None:
    "save a visual representation of the data presence with the class repartition"

    if ms_mask.shape[2] != 2:
        raise ValueError("only binary classification is supported")

    if not save_to:
        return

    # take the nonzeros of both tic and mask (axis class)
    nzy, nzx = np.nonzero(tic + ms_mask.max(axis=2))

    crop = (
        slice(nzy.min(), nzy.max() + 1),
        slice(nzx.min(), nzx.max() + 1),
        slice(None),
    )

    tic = tic[crop[:2]]
    ms_mask = ms_mask[crop]

    tic = np.abs(tic)
    # only account for pixels with larger value
    data_presence = 1.0 * (tic > np.mean(tic) / 1e2)

    ms_mask = 255 * ms_mask.astype(np.uint8)
    data_presence = 255 * data_presence.astype(np.uint8)

    Image.fromarray(
        np.stack([ms_mask[..., 0], ms_mask[..., 1], data_presence], axis=-1), mode="RGB"
    ).save(save_to)


def get_destination_mask(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
):

    return get_destination_mask_from_result(
        **get_template_matching_data(
            ms_group=ms_group,
            bin_idx=bin_idx,
            tiff_path=tiff_path,
            tiff_page_idx=tiff_page_idx,
            onehot_annotations=onehot_annotations,
            transform=transform,
        )
    )
