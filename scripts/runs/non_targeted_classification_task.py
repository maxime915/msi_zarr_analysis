import json
import logging
import pathlib
import time
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
import requests
from msi_zarr_analysis import VERSION
from msi_zarr_analysis.ml.dataset import GroupCollection
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (
    build_spectrum_dict,
    get_overlay_annotations,
    make_collection,
    translate_annotation_mapping_overlay_to_template,
)
from msi_zarr_analysis.ml.saga import saga
from msi_zarr_analysis.preprocessing.binning import bin_spectrum_dict
from msi_zarr_analysis.preprocessing.centroiding import centroid_dict
from msi_zarr_analysis.utils.cytomine_utils import (
    get_page_bin_indices,
)
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold


class MLConfig(NamedTuple):
    model: BaseEstimator
    cv_saga: BaseCrossValidator
    cv_assessment: BaseCrossValidator
    saga_budget_per_run: float


class DSConfig(NamedTuple):
    image_id_overlay: int  # Cytomine ID for the overlay image
    local_overlay_path: str  # local path of the (downloaded) overlay
    lipid_tm: str  # name of the lipid to base the template matching on

    project_id: int  # project id
    annotated_image_id: int  # image with the annotations

    zarr_path: str  # path to the non-binned zarr image

    classes: Dict[str, List[int]]

    bin_lo: np.ndarray
    bin_hi: np.ndarray

    apply_centroid: bool

    save_image: Union[bool, str] = False

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only

    zarr_template_path: str = None  # use another group for the template matching


def get_connection(config_path: str, max_retry: int = 10):        

    from cytomine import Cytomine

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    # build ds
    try:
        connection = Cytomine(host_url, pub_key, priv_key)
        return connection
    except requests.ConnectionError as e:
        if max_retry <= 0:
            raise e
        return get_connection(config_path)


def run(
    base: str,
    config_path: str,
    ml_config: MLConfig,
    bin_csv_path: str,
    *ds_config: DSConfig,
):
    if not ds_config:
        raise ValueError("a list of dataset configuration is required")

    # build ds
    with get_connection(config_path):

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

            spectrum_dict, annotation_coordinates = build_spectrum_dict(
                annotation_dict, ds_config_itm.zarr_path
            )

            if ds_config_itm.apply_centroid:
                spectrum_dict = centroid_dict(spectrum_dict)

            spectrum_dict = bin_spectrum_dict(
                spectrum_dict,
                ds_config_itm.bin_lo,
                ds_config_itm.bin_hi,
            )

            attribute_names = [
                str(v)
                for v in np.round(
                    (ds_config_itm.bin_lo + ds_config_itm.bin_hi) / 2, decimals=3
                )
            ]

            ds = make_collection(spectrum_dict, annotation_coordinates, attribute_names)
            dataset_to_be_merged.append(ds)

        if len(dataset_to_be_merged) == 1:
            ds = dataset_to_be_merged[0]
        else:
            ds = GroupCollection.merge_collections(*dataset_to_be_merged)

    non_targeted_assessment(
        ds,
        ml_config,
    )


def non_targeted_assessment(
    collection: GroupCollection,
    ml_config: MLConfig,
):
    logger = logging.getLogger()

    collection.dataset.check_dataset(print_=True)
    class_names = collection.dataset.class_names()
    logger.info(f"terms: {class_names}")

    rng = np.random.default_rng(785609876)

    def _do_saga(ds_x: np.ndarray, ds_y: np.ndarray, groups: np.ndarray):
        "perform the SAGA algorithm on given dataset"
        
        selection, score = saga(
            ml_config.model,
            time_budget=ml_config.saga_budget_per_run,
            data=ds_x,
            target=ds_y,
            groups=groups,
            cv=ml_config.cv_saga,
            rng=rng,
        )

        logging.debug("SAGA returned with score %f", score)

        return selection, score

    def _assess_saga(ds_x: np.ndarray, ds_y: np.ndarray, groups: np.ndarray):
        "assess the quality of the SAGA algorithm"

        scores = []

        for train_idx, test_idx in ml_config.cv_assessment.split(ds_x, ds_y, groups):
            train_set = (ds_x[train_idx], ds_y[train_idx])

            selection, _ = _do_saga(*train_set, groups[train_idx])

            model_ = clone(ml_config.model)
                        
            model_.fit(ds_x[train_idx][:, selection], ds_y[train_idx])
            score = model_.score(ds_x[test_idx][:, selection], ds_y[test_idx])
            
            logging.info("assess saga (in loop): %s features (%s total), %f", np.sum(selection), selection.shape, score)
            
            scores.append(score)
            

        return np.array(scores)

    logger.info("starting assessment procedure")
    start = time.time()
    scores = _assess_saga(collection.data, collection.target, collection.groups)
    logger.info("duration: %s", time.time() - start)
    logger.info("scores: %s (%s pm %s)", scores, scores.mean(), scores.std(ddof=1))

    logger.info("starting selection procedure")
    start = time.time()
    selection, _ = _do_saga(collection.data, collection.target, collection.groups)
    logger.info("duration: %s", time.time() - start)
    logger.info(
        "%d feature selected by SAGA (out of %d)", selection.sum(), selection.size
    )

    logging.info("training model")
    model_ = clone(ml_config.model)
    model_.fit(collection.data[:, selection], collection.target)
    feature_names = pd.Series(collection.dataset.attribute_names())
    feature_names = feature_names[selection]

    fi_mean = model_.feature_importances_
    if hasattr(model_, "estimators_"):
        fi_std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
    else:
        fi_std = np.full_like(fi_mean, np.nan)

    # 1: feature importance
    logging.info("feature importances:")
    for idx, (feature, importance, importance_std) in enumerate(
        zip(feature_names, fi_mean, fi_std)
    ):
        if len(feature) > 20:
            feature = feature[:17] + "..."
        logging.info("  %02d (%8s) : %7.4f (pm %7.4f)", idx, feature, importance, importance_std)


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

    normalizations = [ # time saving: remove other normalization
        # "",
        "_norm_2305",
        # "_norm_max",
        # "_norm_tic",
        # "_norm_vect",
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

    ml_config = MLConfig(
        model=ExtraTreesClassifier(
            n_estimators=200, max_depth=20, max_features="sqrt", n_jobs=-1
        ),
        cv_saga=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=31),
        cv_assessment=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=32),
        saga_budget_per_run=21600,  # 6h per SAGA call, approx 18 days total
    )

    log_dir = pathlib.Path("logs/non_targeted_classification_task")
    log_dir.mkdir(parents=True, exist_ok=True)

    # build bins
    n_bins = 10000
    mz_lo = 100
    mz_hi = 1150
    bin_mz = np.linspace(mz_lo, mz_hi, n_bins + 1)
    bin_lo = bin_mz[:-1]
    bin_hi = bin_mz[1:]

    for normalization in normalizations:

        for centroid in [True, False]:

            for name, classes in problem_classes.items():

                ds_lst = []
                for source in data_sources:
                    # un-binned file
                    zarr_path = source["base"] + normalization + ".zarr"

                    ds_lst.append(
                        DSConfig(
                            **source["args"],
                            classes=classes,
                            save_image=False,
                            zarr_path=zarr_path,
                            bin_lo=bin_lo,
                            bin_hi=bin_hi,
                            apply_centroid=centroid,
                        )
                    )

                # only grouped CV: more fair
                for grouped_cv in [True]:  #, False]:

                    base = (
                        name
                        + (normalization or "_no_norm")
                        + ("_gcv" if grouped_cv else "_cv")
                        + ("_cen" if centroid else "")
                    )
                    file_handler = logging.FileHandler(
                        (log_dir / base).with_suffix(".log"), mode="a"
                    )
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(formatter)
                    file_handler.addFilter(filter)
                    logger.addHandler(file_handler)

                    logger.info("norm: '%s', problem: '%s'", normalization, name)

                    run(
                        base,
                        "config_cytomine.json",
                        ml_config,
                        "mz value + lipid name.csv",
                        *ds_lst,
                        # grouped_cv=grouped_cv,
                    )

                    logger.info("done")

                logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
