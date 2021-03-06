from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from msi_zarr_analysis.ml.dataset import (
    Dataset,
    ZarrContinuousNonBinned,
    ZarrProcessedBinned,
)
from scipy.stats import ttest_ind
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


from .utils import (
    compare_score_imbalance,
    evaluate_cv,
    feature_importance_forest_mdi,
    feature_importance_model_mds,
    get_feature_importance_forest_mdi,
    get_feature_importance_model_mda,
)


def build_model(description: str):
    common_kwargs = {
        "n_jobs": 4,
    }
    if description == "dt":
        return DecisionTreeClassifier(max_depth=1)
    if description == "extra_trees":
        return ExtraTreesClassifier(**common_kwargs)
    if description == "random_forests":
        return RandomForestClassifier(**common_kwargs)
    raise ValueError(f"invalid choice: {description}")


def interpret_forest_mdi(
    dataset: Dataset,
    forest,
    cv=None,
    return_mean_cv_score: bool = False,
):
    # load an check dataset
    _, imbalance = dataset.check_dataset(cache=True, print_=True, print_once=True)
    dataset_x, dataset_y = dataset.as_table()

    # estimate performance & warn if too bad
    cv_scores = evaluate_cv(forest, dataset_x, dataset_y, cv)
    # compare_score_imbalance(np.median(cv_scores), imbalance)

    print(f"{imbalance = :.3f}")
    print(f"{1.5 * imbalance = :.3f}")
    print(f"{np.min(cv_scores) = :.3f}")
    print(f"{np.mean(cv_scores) = :.3f}")
    print(f"{np.median(cv_scores) = :.3f}")
    print(f"{np.max(cv_scores) = :.3f}")

    # train on full dataset
    forest.fit(dataset_x, dataset_y)

    importances = get_feature_importance_forest_mdi(forest)

    if return_mean_cv_score:
        return importances, np.mean(cv_scores)

    return importances


def interpret_model_mda(
    dataset: Dataset,
    model,
    test_size: float = 0.33,
    random_state=None,
    stratify_classes: bool = False,
    n_repeat: int = 5,
    return_testset_score: bool = False,
):
    # load an check dataset
    _, imbalance = dataset.check_dataset(cache=True, print_=True, print_once=True)
    dataset_x, dataset_y = dataset.as_table()

    # split dataset
    stratify = dataset_y if stratify_classes else None
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset_x,
        dataset_y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # train on training data
    model.fit(X_train, Y_train)

    score = model.score(X_test, Y_test)

    print(f"{imbalance = :.3f}")
    print(f"{1.5 * imbalance = :.3f}")
    print(f"{score = :.3f}")

    # estimate performance on left out data & warn if too bad
    # compare_score_imbalance(model.score(X_test, Y_test), imbalance)

    # evaluate importance based on left out data
    importances = get_feature_importance_model_mda(
        model, X_test, Y_test, n_repeat=n_repeat, random_state=random_state
    )

    if return_testset_score:
        return importances, score

    return importances


def interpret_ttest(
    dataset: Dataset,
    random_state=None,
    /,
    correction = None,
):
    """statistical independence test to select important feature

    Args:
        dataset (Dataset): dataset from which features should be selected
        random_state (, optional): sklearn random state. Defaults to None.
        correction (str, optional): correction, either "bonferroni" or None. Defaults to None.

    Returns:
        np.ndarray, np.ndarray: p values and statistics for the test
    """
    # load an check dataset
    dataset.check_dataset(cache=True, print_=True, print_once=True)
    dataset_x, dataset_y = dataset.as_table()

    classes = np.unique(dataset_y)
    if classes.size != 2:
        raise ValueError(f"only binary classification supported, {classes=}")

    # do statistical tests
    mask_0 = dataset_y == classes[0]
    mask_1 = ~mask_0

    statistics, p_values = ttest_ind(
        dataset_x[mask_0, :],
        dataset_x[mask_1, :],
        equal_var=False,
        random_state=random_state,
    )
    
    if correction == "bonferroni":
        p_values *= dataset_x.shape[1]
    elif correction is not None:
        raise ValueError(F"invalid {correction=!r}")

    return p_values, statistics


def _prettify(mean, std) -> str:
    text = f"{mean:.3f}"
    if std is not None:
        text += f" $\\pm$ {std:.3f}"
    return text


def _get_best_indices(
    *values,
    limit: int,
) -> set:

    n_feature = len(values[0][1])
    indices = set(range(n_feature))

    if len(values) == 1:
        return set()

    for _, mean, _, mode in values:
        rank = np.argsort(mean)

        if mode == "max":
            rank = rank[::-1]
        elif mode != "min":
            raise ValueError(f"invalid {mode=}")

        indices.intersection_update(rank[:limit])

    return indices


def joint_presentation(*values, limit: int, labels: npt.NDArray, sep: str = " & "):
    # not a great idea after all...

    indices = _get_best_indices(*values, limit=limit)

    # c0 & c1 & ... & cn \\

    print("label" + sep + sep.join(tpl[0] for tpl in values) + r"\\")

    for idx in indices:
        text = labels[idx]

        for _, mean, std, _ in values:
            mean_ = mean[idx]
            std_ = std[idx] if std is not None else None
            text += sep + _prettify(mean_, std_)

        print(text, end="\\\\\n")


def present_disjoint(
    *values,
    limit: int,
    labels: List[str],
    sep: str = " & ",
    limit_bold: int = 5,
):
    def _format_number(val: float) -> str:
        if abs(val) < 1e-3:
            return f"{val:.0E}"
        return f"{val:.3f}"

    bf_idx = ()  # empty tuple: nothing in bold
    if limit_bold < len(labels):
        bf_idx = _get_best_indices(*values, limit=limit_bold)

    for label, mean, std, mode in values:
        idx = np.argsort(mean)
        if mode == "max":
            idx = idx[::-1]
        elif mode != "min":
            raise ValueError(f"invalid {mode=}")

        idx = idx[:limit]

        # header
        print("\n\\begin{table}[hbp]")
        print("\t\\centering")
        print("\t\\begin{tabular}{l|r}")

        print("\t\tLipid" + sep + label + r" \\")
        print("\t\t\\hline")

        # data
        for i in idx:
            label_ = labels[i]
            if i in bf_idx:
                label_ = f"\\textbf{{{label_}}}"

            value_text = _format_number(mean[i])
            if std is not None:
                value_text += f" $\\pm$ " + _format_number(std[i])

            print("\t\t" + label_ + sep + value_text + r" \\")

        # footer
        print("\t\\end{tabular}")
        print(f"\t\\caption{{{label}}}")
        print(f"\t\\label{{tab:{label}}}")
        print("\\end{table}")


def present_p_values(
    *values,
    limit: int,
    labels: List[str],
    sep: str = " & ",
    p_value_limit: float = 1e-3,
):
    def _format_number(val: float) -> str:
        if abs(val) < p_value_limit:
            return f"{val:.0E}"
        return f"{val:.3f}"

    for label, mean, mode in values:
        idx = np.argsort(mean)
        if mode == "max":
            idx = idx[::-1]
        elif mode != "min":
            raise ValueError(f"invalid {mode=}")

        idx = idx[:limit]

        # header
        print("\n\\begin{table}[hbp]")
        print("\t\\centering")
        print("\t\\begin{tabular}{l|r}")

        print("\t\tLipid" + sep + label + r" \\")
        print("\t\t\\hline")

        # data
        for i in idx:
            label_ = labels[i]
            if mean[i] < 5e-2:
                label_ = f"\\textbf{{{label_}}}"

            value_text = _format_number(mean[i])

            print("\t\t" + label_ + sep + value_text + r" \\")

        # footer
        print("\t\\end{tabular}")
        print(f"\t\\caption{{{label}}}")
        print(f"\t\\label{{tab:{label}}}")
        print("\\end{table}")


def interpret_forest_ds(
    dataset: Dataset,
    forest,
    fi_impurity_path: str,
    fi_permutation_path: str,
    random_state=None,
    stratify_classes: bool = False,
    cv=None,
):
    # load an check dataset
    _, imbalance = dataset.check_dataset(cache=True, print_=True, print_once=True)
    dataset_x, dataset_y = dataset.as_table()

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


def save_ds_heatmap(
    dataset: Dataset,
    title: str,
    path: str,
):
    ds_x, ds_y = dataset.as_table()
    ds_x = np.where(ds_x == 0.0, np.nan, ds_x)

    classes = np.unique(ds_y)

    fig, axes = plt.subplots(len(classes), 1, figsize=(70, 60))

    axes[0].set_title(title or "heat map of the attributes over the dataset")
    axes[0].set_ylabel("objects of the dataset")
    axes[0].set_yticks(())
    axes[0].set_xlabel("attributes")
    axes[0].set_xticks(())

    for ax, val in zip(axes, classes):
        mask = ds_y == val
        ax.matshow(ds_x[mask], aspect="auto")

    fig.tight_layout()
    # plt.show()
    fig.savefig(path or "heatmap.png")


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
