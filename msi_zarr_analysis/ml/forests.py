from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from msi_zarr_analysis.ml.dataset import (
    Dataset,
    ZarrContinuousNonBinned,
    ZarrProcessedBinned,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from .utils import (
    check_class_imbalance,
    compare_score_imbalance,
    evaluate_cv,
    feature_importance_forest_mdi,
    feature_importance_model_mds,
)


def build_model(description: str):
    kwargs = {
        "n_jobs": 4,
        # "max_top_depth": 2,
        # "n_estimators": 2,
    }
    if description == "extra_trees":
        return ExtraTreesClassifier(**kwargs)
    if description == "random_forests":
        return RandomForestClassifier(**kwargs)
    raise ValueError(f"invalid choice: {description}")


def check_correlation(dataset_x):
    return

    for i in range(dataset_x.shape[1]):
        for j in range(i + 1, dataset_x.shape[1]):
            coeff = np.corrcoef(dataset_x[:, i], dataset_x[:, j])[0, 1]
            if np.abs(coeff) > 0.8:
                # dataset_x[:, j] = np.random.rand(dataset_x.shape[0])
                print(f"{i=} {j=} {coeff=}")


def interpret_forest_ds(
    dataset: Dataset,
    forest,
    fi_impurity_path: str,
    fi_permutation_path: str,
    random_state=None,
    stratify_classes: bool = False,
    cv=None,
):
    dataset_x, dataset_y = dataset.as_table()

    check_correlation(dataset_x)

    imbalance = check_class_imbalance(dataset_y)

    # split dataset
    stratify = dataset_y if stratify_classes else None
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset_x,
        dataset_y,
        test_size=0.33,
        random_state=random_state,
        stratify=stratify,
    )

    cv_scores = evaluate_cv(forest, dataset_x, dataset_y, cv)

    compare_score_imbalance(np.min(cv_scores), imbalance)

    forest.fit(X_train, Y_train)

    if fi_impurity_path:
        feature_importance_forest_mdi(
            forest, fi_impurity_path, dataset.attribute_names(), 10
        )

    if fi_permutation_path:
        feature_importance_model_mds(
            forest,
            X_test,
            Y_test,
            fi_permutation_path,
            dataset.attribute_names(),
            10,
            random_state,
        )

def interpret_forest_binned(
    image_zarr_path: str,
    cls_dict: Dict[str, npt.NDArray[np.dtype("bool")]],
    bin_lo: npt.NDArray,
    bin_hi: npt.NDArray,
    y_slice: slice,
    x_slice: slice,
    fi_impurity_path: str,
    fi_permutation_path: str,
    roi_mask: Optional[npt.NDArray[np.dtype("bool")]] = None,
    append_background_cls: bool = False,
    random_state=None,
    stratify_classes: bool = False,
    model_choice: str = "extra_trees",
):
    dataset = ZarrProcessedBinned(
        image_zarr_path,
        cls_dict,
        bin_lo,
        bin_hi,
        roi_mask,
        append_background_cls,
        y_slice,
        x_slice,
    )

    interpret_forest_ds(
        dataset=dataset,
        forest=build_model(model_choice),
        fi_impurity_path=fi_impurity_path,
        fi_permutation_path=fi_permutation_path,
        random_state=random_state,
        stratify_classes=stratify_classes,
    )


def interpret_forest_nonbinned(
    image_zarr_path: str,
    cls_dict: Dict[str, npt.NDArray[np.dtype("bool")]],
    y_slice: slice,
    x_slice: slice,
    fi_impurity_path: str,
    fi_permutation_path: str,
    roi_mask: Optional[npt.NDArray[np.dtype("bool")]] = None,
    append_background_cls: bool = False,
    random_state=None,
    stratify_classes: bool = False,
    model_choice: str = "extra_trees",
):
    dataset = ZarrContinuousNonBinned(
        data_zarr_path=image_zarr_path,
        classes=cls_dict,
        roi_mask=roi_mask,
        background_class=append_background_cls,
        y_slice=x_slice,
        x_slice=y_slice,
    )

    interpret_forest_ds(
        dataset=dataset,
        forest=build_model(model_choice),
        fi_impurity_path=fi_impurity_path,
        fi_permutation_path=fi_permutation_path,
        random_state=random_state,
        stratify_classes=stratify_classes,
    )
