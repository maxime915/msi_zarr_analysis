import json
import logging
import time
from typing import Dict, NamedTuple, Optional, List, Union, Tuple, Any, Type

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier

from msi_zarr_analysis import VERSION
from msi_zarr_analysis.ml.dataset import GroupCollection
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (
    cytomine_translated_with_groups,
)
from msi_zarr_analysis.utils.cytomine_utils import (
    get_lipid_dataframe,
    get_page_bin_indices,
)


class MLConfig(NamedTuple):
    choices: Dict[Type[BaseEstimator], Dict[str, Any]]
    cv_fold_outer: Optional[int] = None
    cv_fold_inner: Optional[int] = None


class DSConfig(NamedTuple):
    image_id_overlay: int  # Cytomine ID for the overlay image
    local_overlay_path: str  # local path of the (downloaded) overlay
    lipid_tm: str  # name of the lipid to base the template matching on

    project_id: int  # project id
    annotated_image_id: int  # image with the annotations

    zarr_path: str  # path to the non-binned zarr image

    term_list: List[str]  # force order on the classes

    save_image: Union[bool, str] = False

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only
    annotation_terms_id: Tuple[int] = ()  # select these terms only

    zarr_template_path: str = None  # use another group for the template matching


def run(
    config_path: str,
    ml_config: MLConfig,
    bin_csv_path: str,
    *ds_config: DSConfig,
    grouped_cv: bool,
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

            ds = cytomine_translated_with_groups(
                annotation_project_id=ds_config_itm.project_id,
                annotation_image_id=ds_config_itm.annotated_image_id,
                zarr_path=ds_config_itm.zarr_path,
                bin_idx=bin_idx,
                tiff_path=ds_config_itm.local_overlay_path,
                tiff_page_idx=page_idx,
                transform_template_rot90=ds_config_itm.transform_rot90,
                transform_template_flip_ud=ds_config_itm.transform_flip_ud,
                transform_template_flip_lr=ds_config_itm.transform_flip_lr,
                select_users=ds_config_itm.annotation_users_id,
                select_terms=ds_config_itm.annotation_terms_id,
                attribute_names=lipid_df.Name,
                term_list=ds_config_itm.term_list,
                zarr_template_path=ds_config_itm.zarr_template_path,
            )

            dataset_to_be_merged.append(ds)

        if len(dataset_to_be_merged) == 1:
            ds = dataset_to_be_merged[0]
        else:
            ds = GroupCollection.merge_collections(*dataset_to_be_merged)

    model_selection_assessment(
        ds,
        ml_config,
        grouped_cv=grouped_cv,
    )


def model_selection_assessment(
    collection: GroupCollection,
    ml_config: MLConfig,
    *,
    grouped_cv: bool,
):
    logger = logging.getLogger()

    collection.dataset.check_dataset(print_=True)
    class_names = collection.dataset.class_names()
    logger.info(f"terms: {class_names}")

    # StratifiedKFold ignores groups
    CVType = StratifiedGroupKFold if grouped_cv else StratifiedKFold
    # one for each loop
    outer = CVType(n_splits=ml_config.cv_fold_outer, shuffle=True, random_state=31)
    inner = CVType(n_splits=ml_config.cv_fold_inner, shuffle=True, random_state=32)

    def select_model_(ds_x: np.ndarray, ds_y: np.ndarray, groups: np.ndarray):
        "select one model that best fits a dataset"

        best_res = None
        best_model = None
        best_score = -1

        for model, arg_dict in ml_config.choices.items():

            folds = list(inner.split(ds_x, ds_y, groups))
            search = GridSearchCV(model(), arg_dict, cv=folds, n_jobs=-1)
            search.fit(ds_x, ds_y)

            logger.debug(
                "in-loop: %s, %s, %s", model, search.best_params_, search.best_score_
            )

            if search.best_score_ > best_score:
                best_res = search
                best_model = model
                best_score = search.best_score_

        if best_model is None or best_res is None:
            raise ValueError("empty choice dict!")

        return best_model, best_res

    def assessment_(ds_x: np.ndarray, ds_y: np.ndarray, groups: np.ndarray):
        "assess the quality of the selection procedure"

        scores = []

        for train_idx, test_idx in outer.split(ds_x, ds_y, groups):
            train_set = (ds_x[train_idx], ds_y[train_idx])
            test_set = (ds_x[test_idx], ds_y[test_idx])

            model, res = select_model_(*train_set, groups[train_idx])

            estimator = model(**res.best_params_)
            estimator.fit(*train_set)

            scores.append(estimator.score(*test_set))

        return np.array(scores)

    logger.info("starting assessment procedure")
    start = time.time()
    scores = assessment_(collection.data, collection.target, collection.groups)
    logger.info("duration: %s", time.time() - start)
    logger.info("scores: %s (%s pm %s)", scores, scores.mean(), scores.std(ddof=1))

    logger.info("starting selection procedure")
    start = time.time()
    model, res = select_model_(collection.data, collection.target, collection.groups)
    logger.info("duration: %s", time.time() - start)
    logger.info("model: %s, parameters: %s", model, res.best_params_)

    logging.info("training model")
    estimator = model(**res.best_params_).fit(collection.data, collection.target)

    # 1: feature importance
    logging.info("feature importances:")
    for idx, (feature, importance) in enumerate(
        zip(collection.dataset.attribute_names(), estimator.feature_importances_)
    ):
        if len(feature) > 20:
            feature = feature[:17] + "..."
        logging.info("  %02d (%20s) : %.4f", idx, feature, importance)

    # TODO 2: segmentation mask for the whole image

    pass


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

    classification_problems = {
        "SC_n_SC_p": {
            "term_list": ["SC negative AREA", "SC positive AREA"],
            "annotation_terms_id": (544926052, 544924846),
        },
        "LS_n_LS_p": {
            "term_list": ["LivingStrata negative AREA", "LivingStrata positive AREA"],
            "annotation_terms_id": (544926097, 544926081),
        },
        "LS_n_SC_n": {
            "term_list": ["LivingStrata negative AREA", "SC negative AREA"],
            "annotation_terms_id": (544926097, 544926052),
        },
        "LS_p_SC_p": {
            "term_list": ["LivingStrata positive AREA", "SC positive AREA"],
            "annotation_terms_id": (544926081, 544924846),
        },
    }

    for normalization in normalizations:

        for name, class_problem in classification_problems.items():

            ds_lst = []
            for source in data_sources:
                zarr_path = source["base"] + normalization + "_binned.zarr"
                ds_lst.append(
                    DSConfig(
                        **source["args"],
                        **class_problem,
                        save_image=False,
                        zarr_path=zarr_path,
                    )
                )

            for grouped_cv in [True, False]:

                base = (
                    name
                    + (normalization or "_no_norm")
                    + ("_gcv" if grouped_cv else "_cv")
                )
                file_handler = logging.FileHandler(
                    f"logs/classification_task/{base}.log", mode="a"
                )
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                file_handler.addFilter(filter)
                logger.addHandler(file_handler)

                logger.info("norm: '%s', problem: '%s'", normalization, name)

                run(
                    "config_cytomine.json",
                    MLConfig(
                        choices={
                            ExtraTreesClassifier: {
                                "n_estimators": [200, 500, 1000],
                                "max_features": ["sqrt", None],
                                "max_depth": [20, None],
                            },
                            RandomForestClassifier: {
                                "n_estimators": [200, 500, 1000],
                                "max_features": ["sqrt", None],
                                "max_depth": [20, None],
                            },
                            DecisionTreeClassifier: {
                                "max_depth": [1, 20, None],
                            },
                        },
                        cv_fold_inner=5,
                        cv_fold_outer=5,
                    ),
                    "mz value + lipid name.csv",
                    *ds_lst,
                    grouped_cv=grouped_cv,
                )

                logger.info("done")

                logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
