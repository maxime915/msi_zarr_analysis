import json
from typing import NamedTuple, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
from cytomine import Cytomine
from msi_zarr_analysis.ml.dataset.translate_annotation import (
    TemplateTransform,
    colorize_data,
    load_ms_template,
    load_tif_file,
    match_template_multiscale,
)
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import get_page_bin_indices
from numpy import typing as npt


class DSConfig(NamedTuple):
    image_id_overlay: int  # Cytomine ID for the overlay image
    local_overlay_path: str  # local path of the (downloaded) overlay
    lipid_tm: str  # name of the lipid to base the template matching on

    project_id: int  # project id
    annotated_image_id: int  # image with the annotations

    zarr_path: str  # path to the non-binned zarr image

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only

    zarr_template_path: str = None  # use another group for the template matching


def make_overlay(
    background,
    image,
    scale,
    y,
    x,
):

    # scaled up image
    resized = cv2.resize(
        image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    # scaled up image in the top left corner
    translated = np.zeros_like(background)
    translated[
        y : y + resized.shape[0],
        x : x + resized.shape[1],
    ] = resized

    # make a dimmed version of the mask
    mask = 255 - np.uint8(0.5 * 255.0 * (translated > 10).any(axis=2))

    # compute the overlay
    overlay = PIL.Image.composite(
        PIL.Image.fromarray(background, mode="RGB"),
        PIL.Image.fromarray(translated, mode="RGB"),
        PIL.Image.fromarray(mask, mode="L"),
    )

    return np.asarray(overlay)


def check_error(
    config: DSConfig,
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

    ms_group = open_group_ro(config.zarr_template_path)

    transform = TemplateTransform(
        rotate_90=config.transform_rot90,
        flip_lr=config.transform_flip_lr,
        flip_ud=config.transform_flip_ud,
    )

    with open("config_cytomine.json") as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    with Cytomine(host_url, pub_key, priv_key):
        bin_csv_path = "mz value + lipid name.csv"
        page_idx, bin_idx, *_ = get_page_bin_indices(
            config.image_id_overlay, config.lipid_tm, bin_csv_path
        )

    _, ms_template = load_ms_template(ms_group, bin_idx=bin_idx)
    ms_template = transform.transform_template(ms_template)
    colored_template = colorize_data(ms_template)

    overlay = load_tif_file(page_idx=page_idx, disk_path=config.local_overlay_path)

    matching_result = match_template_multiscale(
        overlay,
        colored_template,
    )

    # results obtained from running the algorithm
    base_scale = matching_result.scale
    base_tl_y = matching_result.y_top_left
    base_tl_x = matching_result.x_top_left

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
            add_noise(colored_template, mask_noise),
            scale,
            int(tl_y),
            int(tl_x),
        )

        result = match_template_multiscale(overlay, colored_template)

        return (
            scale,
            result.scale,
            tl_y,
            result.y_top_left,
            tl_x,
            result.x_top_left,
        )

    worker = joblib.delayed(do_once)

    errors = joblib.Parallel(n_jobs=4)(worker() for _ in range(iteration))

    errors = pd.DataFrame(
        errors,
        columns=["true_scale", "pred_scale", "true_y", "pred_y", "true_x", "pred_x"],
    )

    print(errors.to_csv(index=None))

    return errors


def main():

    source = {
        "name": "region_13",
        "args": {
            "image_id_overlay": 545025763,
            "local_overlay_path": "datasets/Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
            "lipid_tm": "LysoPPC",
            "project_id": 542576374,
            "annotated_image_id": 545025783,
            "transform_rot90": 1,
            "transform_flip_ud": True,
            "transform_flip_lr": False,
            "annotation_users_id": (),
            "zarr_template_path": "datasets/comulis13_binned.zarr",
        },
        "base": "datasets/comulis13",
    }

    zarr_path = source["base"] + "_binned.zarr"
    config = DSConfig(
        **source["args"],
        zarr_path=zarr_path,
    )

    errors = check_error(config, iteration=100)

    # plot now
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # problem: tol was 0.001 but mean error is close to 0.015

    ax1.boxplot((errors.true_scale - errors.pred_scale) / errors.true_scale)
    ax1.set_title("Scale Relative Error\n(true - prediction) / true")
    ax1.get_xaxis().set_visible(False)

    ax2.boxplot(errors.true_y - errors.pred_y)
    ax2.set_title("Coordinate (y) Error\n(true - prediction)")
    ax2.get_xaxis().set_visible(False)

    ax3.boxplot(errors.true_x - errors.pred_x)
    ax3.set_title("Coordinate (x) Error\n(true - prediction)")
    ax3.get_xaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig("template_matching_error.png")


if __name__ == "__main__":
    main()
