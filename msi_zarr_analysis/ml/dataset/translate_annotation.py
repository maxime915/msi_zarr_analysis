"translated annotated data via template matching"

import warnings
from typing import List, NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio.features
import tifffile
import zarr
from cytomine.models import AnnotationCollection, TermCollection
from matplotlib import colors
from msi_zarr_analysis.utils.cytomine_utils import iter_annoation_single_term
from PIL import Image
from scipy.optimize import minimize_scalar
from shapely import wkt
from shapely.affinity import affine_transform
from skimage.feature import match_template

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
) -> Tuple[List[str], npt.NDArray]:
    # [classes], np[dims..., classes]

    term_collection = TermCollection().fetch_with_filter(
        "project", annotation_collection.project
    )

    mask_dict = {}

    for annotation, term in iter_annoation_single_term(
        annotation_collection, term_collection
    ):

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

    z_mask[y_ms, x_ms, :] = onehot_annotation[y_overlay, x_overlay, :]

    # build ROI

    roi = (
        slice(int(y_ms.min()), int(1 + y_ms.max())),
        slice(int(x_ms.min()), int(1 + x_ms.max())),
    )

    return z_mask, roi


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

    print(f"{crop_idx=!r}")
    print(f"{ms_template.shape=!r}")
    print(f"{matching_result=!r}")

    return dict(
        onehot_annotation=onehot_annotations,
        yx_dest_shape=ms_group["/0"].shape[2:],
        transform=transform,
        match_result=matching_result,
        crop_idx=crop_idx,
        ms_template_shape=ms_template.shape,
    )


def save_bin_class_image(
    ms_group: zarr.Group,
    ms_mask: npt.NDArray,
    save_to: str = "",
) -> None:
    "save a visual representation of the data presence with the class repartition"

    if ms_mask.shape[2] != 2:
        raise ValueError("only binary classification is supported")

    if not save_to:
        return

    # compute tic
    tic = ms_group["/0"][:, 0, ...].sum(axis=0)

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
