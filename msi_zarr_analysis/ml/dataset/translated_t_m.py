"translated annotated data via template matching"

import warnings
from typing import Iterable, Iterator, List, Tuple, NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio.features
import tifffile
import zarr
from cytomine.models import AnnotationCollection, TermCollection, ImageInstance
from scipy.optimize import minimize_scalar
from shapely import wkt
from shapely.affinity import affine_transform
from skimage.feature import match_template
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import iter_annoation_single_term

from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks

from . import Dataset

# Datatype definitions


class TemplateTransform(NamedTuple):
    # hom many time should the template be rotated (np.rot90(..., k=))
    rotate_90: int = 0
    # should the template be flipped ?
    flip_ud: bool = False
    flip_lr: bool = False

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
    x_top_left: float = 0.0
    y_top_left: float = 0.0
    # how much should the template be scaled to match the source
    scale: float = 1.1

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

    def map_xy(self, x_source, y_source):
        "map some (x, y) coordinates from the (source) image to the pre-processed template"

        y, x = self.map_yx(y_source, x_source)
        return x, y

    map_ij = map_yx


# annotation masks


def build_onehot_annotation(
    annotation_collection: AnnotationCollection,
    image_height: int,
    image_width: int,
) -> Tuple[List[int], npt.NDArray]:
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


def threshold_overlay(overlay_img):
    "select MS peaks from the overlay based on the colormap (assuming default)"
    overlay_img = np.copy(overlay_img)

    # yellow part
    #   - rg close to 200
    #   - b close to 50-100
    mask_yellow = np.ones(shape=overlay_img.shape[:2], dtype=bool)
    mask_yellow[overlay_img[..., 0] < 150] = 0
    mask_yellow[overlay_img[..., 1] < 150] = 0
    mask_yellow[overlay_img[..., 2] > 120] = 0

    # bluish part
    #   - r close to 50-100
    #   - gb close to 100-150
    mask_bluish = np.ones(shape=overlay_img.shape[:2], dtype=bool)
    mask_bluish[overlay_img[..., 0] > 110] = 0
    mask_bluish[overlay_img[..., 1] < 100] = 0

    overlay_img[~mask_yellow & ~mask_bluish] = 0

    return overlay_img


def threshold_ms(color_ms_data, margin: float = 5.0):
    color_ms_data = np.copy(color_ms_data)

    # remove the purple background at [68,  1, 84]
    mask = np.abs(color_ms_data[..., 0] - 68) < margin
    mask &= np.abs(color_ms_data[..., 1] - 1) < margin
    mask &= np.abs(color_ms_data[..., 2] - 84) < margin

    color_ms_data[mask] = 0

    return color_ms_data


# Templata matching implementation


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

    return MatchingResult(tl_x, tl_y, scale)


def generate_spectra_from_result(
    onehot_annotation: npt.NDArray,
    ms_group: zarr.Group,
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
    intensities = ms_group["/0"]
    lengths = ms_group["/labels/lengths/0"]

    z_mask = np.zeros(intensities.shape[2:] + onehot_annotation.shape[-1:], dtype=bool)

    z_mask[y_ms, x_ms, :] = onehot_annotation[y_overlay, x_overlay, :]

    # build ROI

    roi = (
        slice(int(y_ms.min()), int(1 + y_ms.max())),
        slice(int(x_ms.min()), int(1 + x_ms.max())),
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


def generate_spectra(
    ms_group: zarr.Group,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    onehot_annotations: npt.NDArray,
    transform: TemplateTransform,
):

    overlay = load_tif_file(page_idx=tiff_page_idx, disk_path=tiff_path)

    crop_idx, ms_template = load_ms_template(ms_group, bin_idx=bin_idx)

    ms_template = transform.transform_template(ms_template)

    matching_result = match_template_multiscale(
        overlay,
        colorize_data(ms_template),
    )

    yield from generate_spectra_from_result(
        onehot_annotation=onehot_annotations,
        ms_group=ms_group,
        transform=transform,
        match_result=matching_result,
        crop_idx=crop_idx,
        ms_template_shape=ms_template.shape,
    )


class CytomineTranslated(Dataset):
    """
    sliced_overlay_id: int, the Cytomine ID of an image with

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

        terms = TermCollection().fetch_with_filter("project", annotation_project_id)

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

        _, onehot_annotations = build_onehot_annotation(
            annotation_collection=annotations,
            image_height=image_instance.height,
            image_width=image_instance.width,
        )

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
