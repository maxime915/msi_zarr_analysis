import datetime
import functools
import json
import logging
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from msi_zarr_analysis import VERSION
from msi_zarr_analysis.ml.dataset import GroupCollection
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (
    collect_spectra_zarr,
    get_overlay_annotations,
    translate_annotation_mapping_overlay_to_template,
)
from msi_zarr_analysis.utils.cytomine_utils import (
    get_lipid_dataframe,
    get_page_bin_indices,
)
from scipy.stats import mannwhitneyu


@functools.lru_cache(maxsize=1)
def datetime_str() -> str:
    "return a datetime representation, constant through the program"
    return datetime.datetime.now().strftime("%W-%w_%H-%M-%S")


class DSConfig(NamedTuple):
    image_id_overlay: int  # Cytomine ID for the overlay image
    local_overlay_path: str  # local path of the (downloaded) overlay
    lipid_tm: str  # name of the lipid to base the template matching on

    project_id: int  # project id
    annotated_image_id: int  # image with the annotations

    zarr_path: str  # path to the non-binned zarr image

    classes: Dict[str, List[int]]

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only

    zarr_template_path: str = None  # use another group for the template matching


def run(
    config_path: str,
    bin_csv_path: str,
    *ds_config: DSConfig,
):
    if not ds_config:
        raise ValueError("a list of dataset configuration is required")

    from cytomine import Cytomine

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    lipid_df = get_lipid_dataframe(bin_csv_path)

    # build ds
    with Cytomine(host_url, pub_key, priv_key):

        dataset_to_be_merged: List[GroupCollection] = []

        for ds_config_itm in ds_config:

            # build all datasets and merge them if there are more than one
            page_idx, bin_idx, *_ = get_page_bin_indices(
                ds_config_itm.image_id_overlay, ds_config_itm.lipid_tm, bin_csv_path
            )

            annotation_dict = get_overlay_annotations(
                ds_config_itm.project_id,
                ds_config_itm.image_id_overlay,
                ds_config_itm.classes,
                select_users=ds_config_itm.annotation_users_id,
            )

            annotation_dict = translate_annotation_mapping_overlay_to_template(
                annotation_dict,
                ds_config_itm.zarr_template_path,
                bin_idx,
                ds_config_itm.local_overlay_path,
                page_idx,
                ds_config_itm.transform_rot90,
                ds_config_itm.transform_flip_ud,
                ds_config_itm.transform_flip_lr,
            )

            dataset_to_be_merged.append(
                collect_spectra_zarr(
                    annotation_dict,
                    lipid_df.Name,
                    ds_config_itm.zarr_path,
                )
            )

        if len(dataset_to_be_merged) == 1:
            ds = dataset_to_be_merged[0]
        else:
            ds = GroupCollection.merge_collections(*dataset_to_be_merged)

    logger = logging.getLogger()

    # Mann-Whitney U test

    n_feature = ds.data.shape[1]
    classes = np.unique(ds.target)
    assert classes.size == 2
    mask_pos = ds.target == classes[0]
    mask_neg = ds.target == classes[1]
    
    labels = ds.dataset.attribute_names()

    # apply bonferroni correction with p-value of 0.05
    target = 5e-2 / n_feature
        
    tab_string = """
\t\\begin{subtable}[ht]{0.3\\textwidth}
\t\t\\begin{tabular}{clc}
\t\t\t\\toprule
\t\t\t{}  Index & Label & $p$ value \\\\
\t\t\t\\midrule
"""

    important_features = 0
    for feature_idx in range(n_feature):
        # two vector
        ds_pos = ds.data[mask_pos, feature_idx]
        ds_neg = ds.data[mask_neg, feature_idx]

        # The feature is significant if the samples are independent.
        # The samples are independent if the p-value is less than the target.
        _, p_value = mannwhitneyu(ds_pos, ds_neg)
        if p_value < target:
            # the feature is important, log it
            
            # pretty format for the p_value
            if np.round(p_value, 3) == 0:
                p_value = "%.2E" % p_value
            else:
                p_value = "%.3f" % p_value
            
            feature = labels[feature_idx]
            if len(feature) > 20:
                feature = feature[:17] + "..."
            
            tab_string += f"\t\t{feature_idx} & {feature} & {p_value} \\\\\n"

            important_features += 1
            
    tab_string += """\t\t\t\\bottomrule
\t\t\\end{tabular}
\t\t\\caption{}
\t\t\\label{}
\t\\end{subtable}
"""

    logger.info("results for feature selection: %s", tab_string)


def main():
    def filter(record: logging.LogRecord) -> bool:
        if record.name == "root":
            return True

        if record.name == "cytomine.client":
            record.name = "cyt-client"
            return record.levelno != logging.DEBUG

        return False

    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addFilter(filter)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)s] [%(levelname)s] : %(message)s",
        datefmt="%j %H:%M:%S",
    )

    # main handler is stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.NOTSET)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(filter)
    logger.addHandler(stream_handler)

    logger.info("starting %s (VERSION: %s)", __file__, str(VERSION))

    data_sources = [
        {
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
        },
        {
            "name": "region_14",
            "args": {
                "image_id_overlay": 548365416,
                "local_overlay_path": "datasets/Region014-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365416,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "datasets/comulis14_binned.zarr",
            },
            "base": "datasets/comulis14",
        },
        {
            "name": "region_15",
            "args": {
                "image_id_overlay": 548365463,
                "local_overlay_path": "datasets/Region015-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365463,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "datasets/comulis15_binned.zarr",
            },
            "base": "datasets/comulis15",
        },
    ]

    normalizations = [
        "",
        "_norm_2305",
        "_norm_max",
        "_norm_tic",
        "_norm_vect",
    ]

    problem_classes = {
        "LS SC": {  # both merged
            "LS": [544926097, 544926081],
            "SC": [544926052, 544924846],
        },
        "LS_n LS_p": {  # irradiation on LS
            "LS_n": [544926097],
            "LS_p": [544926081],
        },
        "SC_n SC_p": {  # irradiation on SC
            "SC_n": [544926052],
            "SC_p": [544924846],
        },
        "LS_n SC_n": {  # LS vs SC when irradiated
            "LS_n": [544926097],
            "SC_n": [544926052],
        },
        "LS_p SC_p": {  # LS vs SC before irradiation
            "LS_p": [544926081],
            "SC_p": [544924846],
        },
    }

    for normalization in normalizations:

        for name, classes in problem_classes.items():

            ds_lst = []
            for source in data_sources:
                zarr_path = source["base"] + normalization + "_binned.zarr"
                ds_lst.append(
                    DSConfig(
                        **source["args"],
                        classes=classes,
                        zarr_path=zarr_path,
                    )
                )

            base = (
                name
                + (normalization or "_no_norm")
            )
            file_handler = logging.FileHandler(
                f"logs/statistical_tests/{base}.log", mode="a"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(filter)
            logger.addHandler(file_handler)

            logger.info("norm: '%s', problem: '%s'", normalization, name)

            run(
                "config_cytomine.json",
                "mz value + lipid name.csv",
                *ds_lst,
            )

            logger.info("done")

            logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
