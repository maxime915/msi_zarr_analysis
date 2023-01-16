import contextlib
import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from msi_zarr_analysis.ml.dataset.translate_annotation import ParsedAnnotation

try:
    import msi_zarr_analysis
except (ModuleNotFoundError, ImportError):
    # add msi_zarr_analysis to the path if it was not imported
    package_path = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    if package_path not in sys.path:
        sys.path.append(package_path)


from msi_zarr_analysis import VERSION
from msi_zarr_analysis.ml.dataset import GroupCollection
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (
    collect_spectra_zarr,
    get_overlay_annotations,
    translate_annotation_mapping_overlay_to_template,
    make_rasterized_mask,
)
from msi_zarr_analysis.utils.autocrop import autocrop
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import (
    get_lipid_dataframe,
    get_page_bin_indices,
)


def build_segmentation_mask(
    zarr_path: str,
    estimator: BaseEstimator,
) -> np.ndarray:

    z_group = open_group_ro(zarr_path)
    z_ints = z_group["/0"]
    z_lens = z_group["/labels/lengths/0"]

    # assume small dataset
    n_ints = z_ints[:, 0, ...]
    n_lens = z_lens[0, 0, ...]
    selection_mask = n_lens > 0

    # select valid spectra only
    dataset = n_ints[:, selection_mask].T
    prediction = estimator.predict(dataset)

    # build prediction 2D mask
    prediction_mask = np.zeros(shape=n_lens.shape, dtype=np.uint8)
    prediction_mask[selection_mask] = prediction

    # build colored mask
    #   no prediction: transparent
    #   red: class 1
    #   green: class 2
    mask = np.stack(
        [
            np.where(prediction_mask, 200, 50),  # R
            np.where(prediction_mask, 50, 200),  # G
            np.where(prediction_mask, 50, 30),  # B
            np.where(n_lens, 255, 0),  # A
        ],
        axis=-1,
    )

    return mask


def save_ground_truths(
    annotation_dict: Dict[str, List[ParsedAnnotation]],
    attribute_names: List[str],
    zarr_path: str,
    gt_path: str,
    colors_per_class: Dict[str, np.ndarray],
):
    pass
    stem = pathlib.Path(zarr_path).stem

    foreground, yx_tic, rasterized_annotations = make_rasterized_mask(
        annotation_dict, attribute_names, zarr_path
    )
    yx_tic[~foreground] = np.nan

    fig, ax = plt.subplots(dpi=250)
    cmap = mpl.cm.get_cmap().copy()
    cmap.set_bad(color="#DDDDDD")
    ax.matshow(yx_tic, interpolation="nearest", cmap=cmap)

    for cls_, an_lst in rasterized_annotations.items():
        col_ = colors_per_class[cls_]
        col_[-1] = 0.4
        for an in an_lst:
            rgba_mask = np.stack([an.raster] * 4, axis=-1)
            rgba_mask = np.where(rgba_mask, col_, 0)
            ax.matshow(rgba_mask, interpolation="nearest")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig(gt_path + "_" + stem + ".png")
    plt.close(fig)


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

    classes: Dict[str, List[int]]

    save_image: Union[bool, str] = False

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only

    zarr_template_path: str = None  # use another group for the template matching


def run(
    base: str,
    config_path: str,
    ml_config: MLConfig,
    bin_csv_path: str,
    *ds_config: DSConfig,
    grouped_cv: bool,
    res_dir: pathlib.Path,
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

    all_classes = set()
    for ds_config_itm in ds_config:
        all_classes.update(ds_config_itm.classes.keys())
    colors_per_class = dict(
        zip(all_classes, mpl.cm.rainbow(np.linspace(0, 1, len(all_classes))))
    )

    gt_mask_dir = res_dir / "ground_truth"
    gt_mask_dir.mkdir(exist_ok=True)
    gt_path = str(gt_mask_dir / base)

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

            save_ground_truths(
                annotation_dict,
                lipid_df.Name,
                ds_config_itm.zarr_path,
                gt_path,
                colors_per_class,
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

    estimator = model_selection_assessment(
        ds,
        ml_config,
        grouped_cv=grouped_cv,
    )

    class_masks_dir = res_dir / "class_masks"
    class_masks_dir.mkdir(exist_ok=True)
    path = str(class_masks_dir / base)

    with Cytomine(host_url, pub_key, priv_key):

        for ds_config_itm in ds_config:
            mask = build_segmentation_mask(ds_config_itm.zarr_path, estimator)
            stem = pathlib.Path(ds_config_itm.zarr_path).stem

            # build all datasets and merge them if there are more than one
            page_idx, bin_idx, *_ = get_page_bin_indices(
                ds_config_itm.image_id_overlay, ds_config_itm.lipid_tm, bin_csv_path
            )

            template_group = open_group_ro(ds_config_itm.zarr_template_path)
            template_lipid = template_group["/0"][bin_idx, 0, ...]

            crop_idx = autocrop(template_lipid)
            mask = mask[crop_idx]
            template_lipid = template_lipid[crop_idx]

            fig, ax = plt.subplots(dpi=250)

            ax.imshow(template_lipid, interpolation="nearest")
            ax.imshow(mask, interpolation="nearest", alpha=0.4)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            fig.tight_layout()
            fig.savefig(path + "_" + stem + "_annotated.png")
            plt.close(fig)

            fig, ax = plt.subplots(dpi=250)

            ax.imshow(template_lipid, interpolation="nearest")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            fig.tight_layout()
            fig.savefig(path + "_" + stem + ".png")
            plt.close(fig)


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

            try:
                estimator = model(**res.best_params_, n_jobs=-1)
            except TypeError:
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
    try:
        estimator = model(**res.best_params_, n_jobs=-1)
    except TypeError:
        estimator = model(**res.best_params_)
    estimator.fit(collection.data, collection.target)

    avg_gini = estimator.feature_importances_
    if hasattr(estimator, "estimators_"):
        std_gini = np.std(
            [tree.feature_importances_ for tree in estimator.estimators_], axis=0
        )
    else:
        std_gini = np.full_like(avg_gini, np.nan)

    # 1: feature importance
    logging.info("feature importances:")
    for idx, (feature, importance, stddev) in enumerate(
        zip(collection.dataset.attribute_names(), avg_gini, std_gini)
    ):
        if len(feature) > 20:
            feature = feature[:17] + "..."
        logging.info("  %02d (%20s) : %.4f +/- %.4f", idx, feature, importance, stddev)

    return estimator


def _get_log_formatter():
    return logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)s] [%(levelname)s] : %(message)s",
        datefmt="%j %H:%M:%S",
    )


def _logger_filter(record: logging.LogRecord) -> bool:
    if record.name == "root":
        return True

    if record.name == "cytomine.client":
        record.name = "cyt-client"
        return record.levelno != logging.DEBUG

    return False


def do_runs(res_dir: pathlib.Path, logs_dir: pathlib.Path, logger: logging.Logger):

    USE_SLIM_DATASETS = bool(os.environ.get("SLIM", ""))
    pf = "slim_" if USE_SLIM_DATASETS else ""

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
                "zarr_template_path": f"datasets/{pf}comulis13_binned.zarr",
            },
            "base": f"datasets/{pf}comulis13",
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
                "zarr_template_path": f"datasets/{pf}comulis14_binned.zarr",
            },
            "base": f"datasets/{pf}comulis14",
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
                "zarr_template_path": f"datasets/{pf}comulis15_binned.zarr",
            },
            "base": f"datasets/{pf}comulis15",
        },
    ]

    normalizations = [
        "",
        "_norm_2305",
        # "_norm_max",
        # "_norm_tic",
        # "_norm_vect",
    ]
    if USE_SLIM_DATASETS:
        normalizations = ["", "_norm_317"]

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
    )

    for normalization in normalizations:

        for name, classes in problem_classes.items():

            ds_lst = []
            for source in data_sources:
                zarr_path = source["base"] + normalization + "_binned.zarr"
                ds_lst.append(
                    DSConfig(
                        **source["args"],
                        classes=classes,
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
                file_handler = logging.FileHandler(logs_dir / (base + ".log"), mode="a")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(_get_log_formatter())
                file_handler.addFilter(_logger_filter)
                logger.addHandler(file_handler)

                logger.info("norm: '%s', problem: '%s'", normalization, name)

                run(
                    base,
                    "config_cytomine.json",
                    ml_config,
                    "mz value + lipid name.csv",
                    *ds_lst,
                    grouped_cv=grouped_cv,
                    res_dir=res_dir,
                )

                logger.info("done")

                logger.removeHandler(file_handler)


def main():
    exp_dir = pathlib.Path(__file__).parent / "exp-res" / "classification-task"
    res_dir = exp_dir / datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    res_dir.mkdir(parents=True)
    logs_dir = res_dir / "logs"
    logs_dir.mkdir()

    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_logger_filter)

    # main handler is stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.NOTSET)
    stream_handler.setFormatter(_get_log_formatter())
    stream_handler.addFilter(_logger_filter)
    logger.addHandler(stream_handler)

    logger.info("starting %s (VERSION: %s)", __file__, str(VERSION))

    with open(logs_dir / "combined.log", mode="w") as log_file:
        with contextlib.redirect_stdout(log_file):
            with contextlib.redirect_stderr(sys.stdout):
                try:
                    do_runs(res_dir, logs_dir)
                except Exception as e:
                    logger.exception(e)


if __name__ == "__main__":
    main()
