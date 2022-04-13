import pathlib
import time
from typing import Tuple, NamedTuple
import warnings

from numpy import typing as npt
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import tifffile
import zarr
from cytomine.models import AnnotationCollection, ImageInstance, Project, TermCollection
from cytomine.models import SliceInstanceCollection
from shapely import wkt
from skimage.feature import match_template
import PIL.Image
from scipy.optimize import minimize_scalar
from shapely.affinity import affine_transform

# Datatype definitions


class TemplateTransform(NamedTuple):
    # hom many time should the template be rotated (np.rot90(..., k=))
    rotate_90: int = 0
    # should the template be flipped ?
    flip_ud: bool = False
    flip_lr: bool = False

    def transform_template(self, template):
        if self.flip_ud:
            template = np.flip(template, axis=0)
        if self.flip_lr:
            template = np.flip(template, axis=1)

        if self.rotate_90 != 0:
            template = np.rot90(template, k=self.rotate_90)

        return np.ascontiguousarray(template)

    def inverse_transform_mask(self, mask):
        if self.rotate_90 != 0:
            mask = np.rot90(mask, k=-self.rotate_90)
        
        if self.flip_lr:
            mask = np.flip(mask, axis=1)
        if self.flip_up:
            mask = np.flip(mask, axis=0)
        
        return np.ascontiguousarray(mask)


class MatchingResult(NamedTuple):
    # coordinates in the source coordinates for the top left of the scaled matching template
    x_top_left: float = 0.0
    y_top_left: float = 0.0
    # how much should the template be scaled to match the source
    scale: float = 1.1

    def map_xy(self, x_source, y_source):
        "map some (x, y) coordinates from the (source) image to the pre-processed template"
        # translate
        y = y_source - self.y_top_left
        x = x_source - self.x_top_left

        # scale down
        y = y / self.scale
        x = x / self.scale

        # integer pixel (floored down)
        y = np.int32(np.floor(y))
        x = np.int32(np.floor(x))

        return x, y

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

    map_ij = map_yx


# Cytomine data & routine

__region13_flipped = {
    "image_id": 544231581,
    "disk_path": "Region013-Flipped.ome.tif",
}

__test_annotations = {
    "project_id": 542576374,
    "image_id": 544231581,
    "user_selection": (534530561,),
}

__adjusted_viridis = {
    "image_id": 545025763,
    "disk_path": "Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
}

__adjusted_grayscale_annotations = {
    "project_id": 542576374,
    "image_id": 545025783,
}


def download_image(
    image_id,
    disk_path,
    overwrite: bool = False,
):
    """
    download the image to the given path if it doesn't exist"""
    # assumes connected client

    image = ImageInstance().fetch(id=image_id)

    disk_path = pathlib.Path(disk_path)

    if disk_path.exists() and not overwrite:
        return

    image.download(str(disk_path))


def get_annotations(
    project_id,
    image_id,
    user_selection=(),
    save_all: bool = False,
):
    """
    download the annotation and returns a dict mapping terms to mask for the
    image"""

    # assumes connected client

    project = Project().fetch(id=project_id)
    image = ImageInstance().fetch(id=image_id)

    terms = TermCollection().fetch_with_filter("project", project.id)

    # fetch all annotation matching some filters
    annotations = AnnotationCollection()
    annotations.project = project.id
    annotations.image = image.id
    annotations.users = list(user_selection)
    annotations.showTerm = True
    annotations.showWKT = True
    annotations.fetch()

    mask_dict = {}

    if not annotations:
        raise ValueError("no annotation found")

    for annotation in annotations:
        if not annotation.term:
            print(f"term-less annotation: {annotation}")
            continue

        term_name = terms.find_by_attribute("id", annotation.term[0]).name

        # get the mask corresponding to the annotation
        geometry = wkt.loads(annotation.location)
        # flip the coordinates
        geometry = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])
        # rasterize annotation
        mask = rasterio.features.rasterize(
            [geometry], out_shape=(image.height, image.width)
        )

        if not mask.any():
            warnings.warn(f"empty mask found, skipping {annotation.id=}")
            continue

        try:
            mask_dict[term_name] |= mask
        except KeyError:
            mask_dict[term_name] = mask

    if save_all:
        for term_name, mask in mask_dict.items():
            print(f"{term_name=}: {np.count_nonzero(mask)} pixels")

            save_raw(colorize_data(mask), "RGB", f"annotation_{term_name}.png")

            # # save a figure with every annotation to know where they are
            # fig, ax = plt.subplots()
            # ax.set_axis_off()
            # ax.set_title(f"annotation mask: {term_name}")
            # ax.matshow(mask)
            # fig.savefig(f"annotation_{term_name}")
            # plt.close(fig)

    return mask_dict


# Data loading (from disk)


def get_ms_data(bin_idx):
    "load the MS data from a local path"

    z_group = zarr.open_group("comulis13_binned.zarr", mode="r")

    ms_data = z_group["/0"][bin_idx, 0]
    nz_y, nz_x = ms_data.nonzero()

    try:
        # crop it (useless to include border)
        ms_data = ms_data[
            nz_y.min() : 1 + nz_y.max(),
            nz_x.min() : 1 + nz_x.max(),
        ]
    except ValueError as e:
        # handle case where there is no data
        if e.args != (
            "zero-size array to reduction operation minimum which has no identity",
        ):
            raise e

    return ms_data


def load_tif_file(page_idx: int, disk_path: str, image_id: int = None):
    "load the overlay data from a local path"

    store = tifffile.imread(disk_path, aszarr=True)
    z = zarr.open(store)  # pages, chan, height, width

    # first page
    image = z[page_idx, ...]  # C=[RGB], H, W
    image = np.transpose(image, (1, 2, 0))  # H, W, C

    assert len(image.shape) == 3
    assert image.shape[2] == 3

    return image


# Image transformation


def rgb_to_grayscale(rgb):
    "H, W, C=[RGB]"
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def colorize_data(intensity):
    # H, W, C=[RGBA]
    colored = plt.cm.ScalarMappable(cmap=plt.cm.viridis).to_rgba(intensity)
    colored = np.uint8(np.round(255 * colored))  # as 8bit color
    colored = colored[:, :, :3]  # strip alpha
    return colored


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


# storing & plotting


def save_raw(
    data,
    mode: str,
    destination: str,
):
    PIL.Image.fromarray(data, mode).save(destination)


def save_plot(
    data,
    *patches,
    grayscale: bool,
    destination: str,
):
    "save an array as an image (RGB or grayscale)"

    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()

    ax.imshow(data, cmap=plt.cm.gray if grayscale else None)

    for patch in patches:
        ax.add_patch(patch)

    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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

        scaled = cv2.resize(grayscale_ms_data, dsize=None, fx=scale, fy=scale)

        score, _ = get_template_matching_score(grayscale_overlay, scaled)

        return -score

    def get_match_info(scale: float):
        # build scaled image
        if scale > high or scale < low:
            raise ValueError("invalid scale")

        scaled = cv2.resize(grayscale_ms_data, dsize=None, fx=scale, fy=scale)

        score, coords = get_template_matching_score(grayscale_overlay, scaled)

        return score, scaled.shape, coords

    res = minimize_scalar(
        cost_fun,
        bracket=(low, high),
        tol=tol,
        method="Brent",
    )
    # print(f"{res=!r}")

    best_score, (height, width), (x, y) = get_match_info(res.x)

    return best_score, res.x, (height, width), (x, y)


def match_template_multiscale_iterative_refinement(
    grayscale_overlay,
    grayscale_ms_data,
    max_scale: float,
    tol: float = 1e-2,
):
    low = 1.0
    high = max_scale

    def evaluate_score(scale: float):
        # build scaled image
        if scale > high or scale < low:
            raise ValueError(f"invalid {scale=}")

        scaled = cv2.resize(grayscale_ms_data, dsize=None, fx=scale, fy=scale)

        score, coords = get_template_matching_score(grayscale_overlay, scaled)

        return score, scaled.shape, coords

    while high - low > 2 * tol:
        # wastefull : low and high were already computed before...
        low, high = bound_scale_in_sample(
            grayscale_overlay,
            grayscale_ms_data,
            low,
            high,
            n_points=6,
        )

    best_scale = 0.5 * (low + high)

    best_score, (height, width), (x, y) = evaluate_score(best_scale)

    return best_score, best_scale, (height, width), (x, y)


def bound_scale_in_sample(
    grayscale_overlay,
    grayscale_ms_data,
    low: float,
    high: float,
    n_points: int = 12,
):
    "find bounds for the best scale in np.linspace(low, high, n_points)"

    def evaluate_score(scale: float):
        # build scaled image
        scaled = cv2.resize(grayscale_ms_data, dsize=None, fx=scale, fy=scale)

        score, _ = get_template_matching_score(grayscale_overlay, scaled)

        return score

    delayed_fun = joblib.delayed(evaluate_score)

    scale_range = np.linspace(low, high, n_points)
    scores = joblib.Parallel(n_jobs=-1)(delayed_fun(scale) for scale in scale_range)

    max_idx = np.argmax(scores)

    left = scale_range[max_idx - 1] if max_idx > 0 else low
    right = scale_range[max_idx + 1] if max_idx + 1 < len(scale_range) else high

    return left, right


def match_template_multiscale(
    image_rgb,
    template_rgb,
    save_to: str = None,
    save_tmp: bool = False,
    verbose: bool = False,
) -> MatchingResult:

    image_th = threshold_overlay(image_rgb)
    image_th_gs = rgb_to_grayscale(image_th)

    if save_tmp:
        save_raw(image_rgb, mode="RGB", destination="image.png")
        save_raw(image_th, mode="RGB", destination="image-th.png")
        save_raw(np.uint8(image_th_gs), mode="L", destination="image-th-gs.png")

    template_th = threshold_ms(template_rgb)
    template_th_gs = rgb_to_grayscale(template_th)

    if save_tmp:
        save_raw(template_rgb, mode="RGB", destination="template.png")
        save_raw(template_th, mode="RGB", destination="template-th.png")
        save_raw(np.uint8(template_th_gs), mode="L", destination="template-gs.png")

    max_scale = min(
        o_s / t_s for o_s, t_s in zip(image_th_gs.shape, template_th_gs.shape)
    )

    start = time.time()

    (score, scale, (height, width), (tl_x, tl_y)) = match_template_multiscale_scipy(
        image_th_gs, template_th_gs, max_scale
    )

    end = time.time()

    if verbose:
        print(f"template matching: time={end-start:.2f}s {score=:.3f} {scale=:.3f}")
        print(f"{scale=:.3f} {tl_y=:.3f} {tl_x=:.3f}")

    if save_to:
        save_plot(
            image_rgb,
            plt.Rectangle((tl_x, tl_y), width, height, edgecolor="g", facecolor="none"),
            grayscale=False,
            destination=save_to,
        )

    return MatchingResult(tl_x, tl_y, scale)


## Evaluation


def make_overlay(
    background,
    image,
    scale,
    y,
    x,
    save_to: str = None,
):

    # scaled up image
    resized = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    # scaled up image in the top left corner
    fused = np.zeros_like(background)
    fused[
        : resized.shape[0],
        : resized.shape[1],
    ] = resized

    # scanled up & translated image
    translated = cv2.warpAffine(
        fused,
        np.float32([[1, 0, x], [0, 1, y]]),
        dsize=fused.shape[:2][::-1],
        borderMode=cv2.BORDER_WRAP,
    )

    # make a dimmed version of the mask
    mask = 255 - np.uint8(0.5 * 255.0 * (translated > 10).any(axis=2))

    # compute the overlay
    overlay = PIL.Image.composite(
        PIL.Image.fromarray(background, mode="RGB"),
        PIL.Image.fromarray(translated, mode="RGB"),
        PIL.Image.fromarray(mask, mode="L"),
    )

    if save_to:
        overlay.save(save_to)

    return overlay


## Main


def check_match_annotations(
    image_dict: dict,
    annotation_dict: dict,
    transform: TemplateTransform,
    suffix: str,
    lipid: str,
):

    download_image(**image_dict)
    print("image downloaded")

    mask_dict = get_annotations(
        **annotation_dict,
        save_all=True,
    )
    print("annotation downloaded")

    page_idx, bin_idx = get_page_bin_indices(lipid=lipid, **image_dict)

    ms_data = get_ms_data(bin_idx=bin_idx)
    ms_data = transform.transform_template(ms_data)
    color_ms_data = colorize_data(ms_data)

    overlay_img = load_tif_file(page_idx=page_idx, **image_dict)
    print("data loaded")

    matching_result = match_template_multiscale(
        overlay_img, color_ms_data, save_to=f"matched_template_{suffix}.png"
    )
    print("mapping loaded")

    map_yx = matching_result.map_yx

    for term, mask in mask_dict.items():
        # this only fetches each annotated pixel once
        translated_mask = np.zeros(shape=ms_data.shape, dtype=bool)
        translated_mask[map_yx(*np.nonzero(mask))] = 1.0
        sample = ms_data[translated_mask]
        print(f"{term=} {sample.shape=}")

        # this duplicates entries to weight them proportionally
        sample = ms_data[map_yx(*np.nonzero(mask))]
        print(f"{term=} {sample.shape=}")


def match_region13_flipped(lipid: str):
    check_match_annotations(
        __region13_flipped,
        __test_annotations,
        TemplateTransform(flip_lr=True),
        "overlay",
        lipid,
    )


def match_adjusted(lipid: str):
    check_match_annotations(
        __adjusted_viridis,
        __adjusted_grayscale_annotations,
        TemplateTransform(rotate_90=1, flip_ud=True),
        "adjusted",
        lipid,
    )


def check_error(
    iteration: int,
    scale_std: float = 0.1,
    tl_y_std: float = 2,
    tl_x_std: float = 2,
    background_noise: float = 0,
    mask_noise: float = 0,
):
    background = cv2.cvtColor(
        cv2.imread("Microscope-Images_Regions-013_scaled.png"), cv2.COLOR_BGR2RGB
    )

    transform = TemplateTransform(rotate_90=1, flip_ud=True)
    ms_data = get_ms_data(bin_idx=14)  # in this case, the bin doesn't matter that much
    ms_data = transform.transform_template(ms_data)
    ms_data = colorize_data(ms_data)

    # only used for the overlay, otherwise this is cheating
    mask = threshold_ms(ms_data)

    # results obtained from running the algorithm
    base_scale = 6.974
    base_tl_y = 127.0
    base_tl_x = 246.0

    def add_noise(arr: npt.NDArray, level: float):
        noise = np.random.normal(0, level, size=arr.shape)
        noise = noise.reshape(arr.shape)
        return np.uint8(arr + noise).clip(0, 255)

    def do_once():
        scale = base_scale + np.random.normal(0, scale_std)
        tl_y = np.floor(base_tl_y + np.random.normal(0, tl_y_std))
        tl_x = np.floor(base_tl_x + np.random.normal(0, tl_x_std))

        overlay = make_overlay(
            add_noise(background, background_noise),
            add_noise(mask, mask_noise),
            scale,
            tl_y,
            tl_x,
        )

        result = match_template_multiscale(overlay, ms_data)

        return (
            scale,
            result.scale,
            tl_y,
            result.y_top_left,
            tl_x,
            result.x_top_left,
        )

    # print(do_once())

    worker = joblib.delayed(do_once)

    errors = joblib.Parallel(n_jobs=4)(worker() for _ in range(iteration))

    errors = pd.DataFrame(
        errors,
        columns=["true_scale", "pred_scale", "true_y", "pred_y", "true_x", "pred_x"],
    )

    print(errors.to_csv(index=None))
    exit(0)


def save_all_bins():
    for channel_idx in range(37):
        ms_data = get_ms_data(channel_idx)
        ms_data = ms_data
        color_ms_data = colorize_data(ms_data)

        save_raw(color_ms_data, "RGB", f"ms_{channel_idx}.png")


def main():

    # check_error(100)

    print("starting")

    from connect_from_json import connect

    connect()
    print("connected")

    match_region13_flipped("LysoPPC")
    match_adjusted("LysoPPC")


def get_page_bin_indices(
    image_id: int, lipid: str, disk_path: str = None
) -> Tuple[int, int]:
    """fetch the TIFF page and the bin's index corresponding to a lipied

    Parameters
    ----------
    image_id : int
        Cytomine id
    lipid : str
        name of the lipid in the CSV and Cytomine namespaces
    disk_path : str, optional
        ignored parameter to accept **__image_dict

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

    ds = pd.read_csv("mz value + lipid name.csv", sep=None, engine="python")

    ms_bin = ds[ds.Name == lipid].index

    if ms_bin.size == 0:
        raise ValueError(f"unable to find {lipid=} in the CSV file")

    if ms_bin.size > 1:
        raise ValueError(f"duplicate entries {ms_bin=!r} for {lipid=}")

    return tiff_page, ms_bin[0]


if __name__ == "__main__":
    main()
