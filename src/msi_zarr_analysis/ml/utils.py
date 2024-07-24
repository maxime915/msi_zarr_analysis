import time
import warnings
from typing import List, Tuple
import joblib
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import clone

from msi_zarr_analysis.ml.dataset import Dataset


def compare_score_imbalance(score: float, imbalance: float):
    if score < 1.5 * imbalance:
        warnings.warn(
            f"{score = :.3f} is below {1.5*imbalance:.3f}, results may not be "
            f"indicative (class_{imbalance = :.3f})"
        )
    else:
        print(f"{score = :.3f} ({imbalance = :.3f})")


# def evaluate_train_test(model, X_train, Y_train, X_test, Y_test, imbalance):
#     print("building classifier...")

#     start_time = time.time()
#     model.fit(X_train, Y_train)
#     elapsed_time = time.time() - start_time

#     print(f"classifier is done: {elapsed_time:.3f} seconds !")

#     start_time = time.time()
#     acc = accuracy_score(model.predict(X_test), Y_test)
#     elapsed_time = time.time() - start_time

#     if acc < 1.5 * imbalance:
#         warnings.warn(
#             f"accuracy = {acc} is below {1.5*imbalance}, results may not be "
#             f"indicative ({imbalance=}) ({elapsed_time=:.3f} seconds)"
#         )
#     else:
#         print(
#             f"accuracy on the test set: {acc} ({imbalance=}) ({elapsed_time=:.3f} seconds)"
#         )


def evaluate_testset(model, X_test, Y_test):

    start_time = time.time()
    score = accuracy_score(model.predict(X_test), Y_test)
    elapsed_time = time.time() - start_time

    print(f"test set score: {score} (in {elapsed_time:.3f} seconds)")

    return score


def evaluate_cv(model, dataset_x, dataset_y, cv=None):

    start_time = time.time()
    scores = cross_val_score(model, dataset_x, dataset_y, n_jobs=4, cv=cv)
    elapsed_time = time.time() - start_time

    # print(f"mean CV score: {np.mean(scores):.3f} (in {elapsed_time:.3f} seconds)")

    return scores


def is_path_csv(path: str):
    return path[-4:].lower() == ".csv"


def is_path_image(path: str):
    ext5 = path[:-5].lower()
    if ext5 in [".tiff", ".jpeg"]:
        return True
    ext4 = ext5[1:]
    return ext4 in [
        ".png",
        ".jpg",
    ]


def present_feature_importance(
    importances,
    std,
    fig_title: str,
    fig_ylabel: str,
    store_at: str,
    feature_labels: npt.NDArray,
    print_n_most_important: int = 0,
):
    max_len = 0
    to_print = []

    for i in importances.argsort()[::-1][:print_n_most_important]:
        text_len = len(feature_labels[i])
        if text_len > max_len:
            max_len = text_len
        to_print.append(
            (feature_labels[i], text_len, f"{importances[i]:.3f} +/- {std[i]:.3f}")
        )

    for label, length, value in to_print:
        print(" " * (max_len - length) + label + " " + value)

    if not store_at:
        return

    if is_path_csv(store_at):
        pd.DataFrame(
            {
                "importances": importances,
                "std": std,
                "feature": feature_labels,
            }
        ).sort_values("importances", inplace=False, ascending=False).to_csv(store_at)

    elif is_path_image(store_at):
        forest_importances = pd.Series(importances, index=feature_labels)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title(fig_title)
        ax.set_ylabel(fig_ylabel)
        fig.tight_layout()
        fig.savefig(store_at)

    else:
        raise ValueError(f"invalid value for {store_at=}, neither an image nor a CSV")


def get_feature_importance_forest_mdi(forest) -> Tuple[npt.NDArray, npt.NDArray]:
    mean = forest.feature_importances_

    if hasattr(forest, "estimators_"):
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    else:
        std = np.full_like(mean, np.nan)

    return mean, std


def feature_importance_forest_mdi(
    forest,
    store_at: str,
    feature_labels: npt.NDArray,
    print_n_most_important: int = 0,
):
    """compute the importance for the dataset's feature by mean decrease in
    impurity of a training forest model.

    Parameters
    ----------
    forest : Model
        must implement fit(X, y), feature_importances_, estimators_
        see sklearn.ensemble.ExtraTreesClassifier, sklearn.ensemble.RandomForestClassifier
    store_at : str
        path to store the result (either image or csv), nothing is saved if falsy
    feature_labels : List[str]
        name for each feature
    print_n_most_important : int, optional
        number of most important feature to print, by default 0
    """
    if not hasattr(forest, "feature_importances_"):
        raise ValueError(
            "model may not be a tree ensemble, missing 'feature_importances_' attribute"
        )
    # if not hasattr(forest, "estimators_"):
    #     raise ValueError(
    #         "model may not be a tree ensemble, missing 'estimators_' attribute"
    #     )

    print("computing importance based on impurity...")

    print("computing importance based on impurity...")
    start_time = time.time()
    importances = forest.feature_importances_
    if hasattr(forest, "estimators_"):
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    else:
        std = np.full_like(importances, np.nan)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    present_feature_importance(
        importances,
        std,
        fig_title="Feature importances using MDI",
        fig_ylabel="Mean decrease in impurity (MDI)",
        store_at=store_at,
        feature_labels=feature_labels,
        print_n_most_important=print_n_most_important,
    )


def get_feature_importance_model_mda(
    model,
    X_set: npt.NDArray,
    Y_set: npt.NDArray,
    n_repeat=5,
    random_state=None,
):
    result = permutation_importance(
        model, X_set, Y_set, n_repeats=n_repeat, random_state=random_state, n_jobs=4
    )

    return result.importances_mean, result.importances_std


def feature_importance_model_mds(
    model,
    X_set: npt.NDArray,
    Y_set: npt.NDArray,
    store_at: str,
    feature_labels: List[str],
    print_n_most_important: int = 0,
    random_state=None,
):
    """compute the importance for the dataset's features by mean decrase in
    accuracy over random feature permutations.

    Parameters
    ----------
    model : Model
        must implement fit(X, y)
    X_set : npt.NDarray
        features
    Y_set : npt.NDArray
        classes
    store_at : str
        path to store the result (either image or csv), nothing is saved if falsy
    feature_labels : List[str]
        name for each feature
    print_n_most_important : int, optional
        number of most important feature to print, by default 0
    random_state : optional
        see sklearn manual, by default None
    """

    print("computing importance based on permutation importance...")

    start_time = time.time()
    result = permutation_importance(
        model, X_set, Y_set, n_repeats=10, random_state=random_state, n_jobs=4
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    present_feature_importance(
        result.importances_mean,
        result.importances_std,
        fig_title="Feature importances using MDA over permutations",
        fig_ylabel="Mean Decrease in Accuracy",
        store_at=store_at,
        feature_labels=feature_labels,
        print_n_most_important=print_n_most_important,
    )


def show_datasize_learning_curve(
    dataset: Dataset,
    model,
    cv=None,
    save_to: str = "",
    show: bool = False,
):
    """plot the accuracy of the model at different dataset sizes

    Args:
        dataset (Dataset): dataset to train on
        model (sklearn estimator): model from sci-kit learn
        cv (int, optional): number of CV fold. Defaults to None.
        save_to (str, optional): path to store the image. Defaults to "".
        show (bool, optional): plt.show(). Defaults to False.
    """

    dataset_x, dataset_y = dataset.as_table()

    if not save_to and not show:
        raise ValueError(f"at least one of {save_to=}, {show=} should be set")

    percentage = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def _do_percentage():
        "get scores at all percentages values"

        x_train, x_test, y_train, y_test = train_test_split(
            dataset_x, dataset_y, test_size=0.2
        )
        n_elem = x_train.shape[0]

        scores = []

        for p in percentage:
            model_ = clone(model)

            r_mask = np.full((n_elem,), False)
            r_mask[: int(np.ceil(p * n_elem))] = True
            np.random.shuffle(r_mask)

            ds_x = x_train[r_mask, :]
            ds_y = y_train[r_mask]

            model_.fit(ds_x, ds_y)

            scores.append(model_.score(x_test, y_test))

        return scores

    fn = joblib.delayed(_do_percentage)
    scores = joblib.Parallel(n_jobs=-1)(fn() for _ in range(cv))
    scores = np.array(scores)

    ys_ = 100 * np.mean(scores, axis=0)
    yer = 100 * np.std(scores, ddof=1, axis=0)
    y1s = ys_ - yer
    y2s = ys_ + yer
    xs_ = 100 * percentage

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    fig.suptitle("Estimated Accuracy as a Function of the Dataset Size")

    ax.set_ylim((0, 100))
    ax.set_xlim((0, 100))

    ax.set_ylabel("Estimated Accuracy (%)")
    ax.set_xlabel("Relative size of the dataset (%)")

    ax.fill_between(xs_, y1s, y2s, alpha=0.4, color="b")
    ax.plot(xs_, ys_, color="b")

    if save_to:
        fig.savefig(save_to)

    if show:
        plt.show()
