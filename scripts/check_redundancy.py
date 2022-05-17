# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
# https://orbi.uliege.be/bitstream/2268/155642/1/louppe13.pdf
# https://proceedings.neurips.cc/paper/2019/file/702cafa3bb4c9c86e4a3b6834b45aedd-Paper.pdf
# https://indico.cern.ch/event/443478/contributions/1098668/attachments/1157598/1664920/slides.pdf

import time
import warnings
from collections import defaultdict
from typing import Callable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate_cv(model, dataset_x, dataset_y, cv=None):

    start_time = time.time()
    scores = cross_val_score(model, dataset_x, dataset_y, cv=cv)
    elapsed_time = time.time() - start_time

    # print(f"mean CV score: {np.mean(scores):.3f} (in {elapsed_time:.3f} seconds)")

    return scores


def check_class_imbalance(dataset_y):
    _, occurrences = np.unique(dataset_y, return_counts=True)

    print(f"{occurrences = }")

    # highest fraction of class over samples
    imbalance = np.max(occurrences / dataset_y.size)
    print(f"{np.max(occurrences / dataset_y.size) = :.4f}")
    print(f"{np.min(occurrences / dataset_y.size) = :.4f}")
    print(f". . . . . . . . . . . . 1 / #classes = {1/(np.max(dataset_y)+1):.4f}")

    return imbalance


def compare_score_imbalance(score: float, imbalance: float):
    if score < 1.5 * imbalance:
        warnings.warn(
            f"{score = :.3f} is below {1.5*imbalance:.3f}, results may not be "
            f"indicative (class_{imbalance = :.3f})"
        )
    else:
        print(f"{score = :.3f} ({imbalance = :.3f})")


def check_correlation(dataset_x):

    for i in range(dataset_x.shape[1]):
        for j in range(i + 1, dataset_x.shape[1]):
            coeff = np.corrcoef(dataset_x[:, i], dataset_x[:, j])[0, 1]
            if np.abs(coeff) > 0.8:
                # dataset_x[:, j] = np.random.rand(dataset_x.shape[0])
                print(f"{i=} {j=} {coeff=}")


def new_model(
    random_state,
    n_estimators: int = 1000,
    max_features: int = None,
    max_depth: int = None,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=random_state,
    )


def get_feature_idx(dataset_x, dataset_y, start=(), random_state=48):
    cv = 5

    def get_score_partial_features(indices: tuple):
        partial_x = dataset_x[:, indices]
        # model = new_model(random_state)
        # model = new_model(random_state=random_state)
        model = ExtraTreesClassifier(random_state=random_state)

        return indices[-1], np.mean(evaluate_cv(model, partial_x, dataset_y, cv))

    delayed_score = joblib.delayed(get_score_partial_features)

    last_score = 0.0

    selected = tuple(start)
    candidates = list(set(range(dataset_x.shape[1])) - set(selected))

    while True:
        results = joblib.Parallel(n_jobs=-1)(
            delayed_score(selected + (c,)) for c in candidates
        )

        best_idx, best_score = results[0]
        for idx_, score_ in results[1:]:
            if score_ > best_score:
                best_score = score_
                best_idx = idx_

        if best_score - last_score < 0.01:
            break

        selected += (best_idx,)
        candidates.remove(best_idx)

        print(f"{best_score=:.3f} {selected=}")

        last_score = best_score

    return selected


def add_input_noise(dataset_x: np.ndarray, rel_scale: float):
    scale = rel_scale * np.mean(np.abs(dataset_x), axis=1)

    # numpy needs the first axis to be the same as the scale
    size = dataset_x.shape[::-1]
    noise = np.random.normal(scale=scale, size=size).T

    return dataset_x + noise


def do_plot(dataset_x, dataset_y, stratify_classes=True, random_state=48):

    model = new_model(random_state)
    cv = 10

    # check_correlation(dataset_x)

    # imbalance = check_class_imbalance(dataset_y)

    # split dataset
    stratify = dataset_y if stratify_classes else None
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset_x,
        dataset_y,
        test_size=0.33,
        random_state=random_state,
        stratify=stratify,
    )

    # cv_scores = evaluate_cv(model, dataset_x, dataset_y, cv)

    # print(
    #     f"{np.min(cv_scores)=}",
    #     f"{np.mean(cv_scores)=}",
    #     f"{np.median(cv_scores)=}",
    #     f"{np.max(cv_scores)=}",
    # )

    # compare_score_imbalance(np.mean(cv_scores), imbalance)

    model.fit(X_train, Y_train)

    ts_score = model.score(X_test, Y_test)

    print(f"{ts_score=}")

    feature_names = list(map(str, range(dataset_x.shape[1])))

    # find the most important features (see sklearn doc)

    # result = permutation_importance(
    #     model, X_train, Y_train, n_repeats=10, random_state=42
    # )
    # perm_sorted_idx = result.importances_mean.argsort()

    # tree_importance_sorted_idx = np.argsort(model.feature_importances_)
    # tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # ax1.barh(
    #     tree_indices, model.feature_importances_[tree_importance_sorted_idx], height=0.7
    # )
    # ax1.set_yticks(tree_indices)
    # ax1.set_yticklabels([feature_names[i] for i in tree_importance_sorted_idx])
    # ax1.set_ylim((0, len(model.feature_importances_)))
    # ax2.boxplot(
    #     result.importances[perm_sorted_idx].T,
    #     vert=False,
    #     labels=[feature_names[i] for i in perm_sorted_idx],
    # )
    # fig.tight_layout()
    # plt.show()

    # find the correlated features

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    corr = spearmanr(dataset_x).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
    )
    # dendro_idx = np.arange(0, len(dendro["ivl"]))

    # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    # ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()

    # for threshold in  [3.5, 2.5, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
    for threshold in [0.4]:

        cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")

        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

        X_train_sel = X_train[:, selected_features]
        X_test_sel = X_test[:, selected_features]

        clf_sel = new_model(random_state=random_state)
        clf_sel.fit(X_train_sel, Y_train)
        score = clf_sel.score(X_test_sel, Y_test)
        print(f"{threshold=:.3f} {score=:.3f} {len(selected_features)=}")

        print(f"{selected_features=}")


def get_mdi_importance(ds_x, ds_y, model):
    model.fit(ds_x, ds_y)
    try:
        importances = model.feature_importances_
        if hasattr(model, "estimators_"):
            std = np.std(
                [tree.feature_importances_ for tree in model.estimators_], axis=0
            )
        else:
            std = np.full_like(importances, np.nan)

        return importances, std
    except AttributeError as _:
        return None


def get_permutation_importance(ds_x, ds_y, model, random_state):

    # permutation importance
    X_train, X_test, Y_train, Y_test = train_test_split(
        ds_x, ds_y, test_size=0.33, random_state=random_state
    )

    model.fit(X_train, Y_train)

    result = permutation_importance(
        model, X_test, Y_test, random_state=random_state, n_repeats=10, n_jobs=-1,
    )

    return (result.importances_mean, result.importances_std)


def get_feature_importances(ds_x, ds_y, model_fn: Callable, random_state):

    return (
        get_permutation_importance(ds_x, ds_y, model_fn(random_state), random_state),
        get_mdi_importance(ds_x, ds_y, model_fn(random_state)),
    )


def study_model(ds_x, ds_y, random_state):

    model_builders = [
        lambda: ExtraTreesClassifier(
            n_estimators=1000, max_features=None, n_jobs=-1, random_state=random_state
        ),
        lambda: RandomForestClassifier(
            n_estimators=1000, max_features=None, n_jobs=-1, random_state=random_state
        ),
        lambda: MLPClassifier(hidden_layer_sizes=(128, 128), random_state=random_state),
    ]

    df = pd.DataFrame()

    # TODO


def add_features(
    dataset_x: np.ndarray,
    *,
    n_comb_lin_droppout: int = 0,
    n_noise: int = 0,
    n_lin_comb: int = 0,
    n_redundant: int = 0,
) -> np.ndarray:
    """add some correlated or noisy features to a dataset.

    Args:
        dataset_x (np.ndarray): original dataset
        n_comb_lin_droppout (int): first apply a 30% dropout to the dataset and
            then apply a linear combination with a small noise (scale=0.1*std)
        n_noise (int): number of gaussian noise features to add (scale=1.0)
        n_lin_comb (int): linear combination of the features with added
            gaussian noise (scale=0.1*std) to add
        n_redundant (int): number of redundant features to add with a gaussian
            noise (scale=0.1*std)

    Returns:
        np.ndarray: the dataset, columns are added in order, at the right edge
    """

    def _dropout() -> np.ndarray:
        "compute one correlated noisy feature column"
        weights = np.random.normal(loc=0, scale=1, size=(dataset_x.shape[1], 1))

        dropout = np.copy(dataset_x)
        dropout[np.random.rand(*dropout.shape) < 0.3] = 0

        feature = np.dot(dropout, weights)

        return feature + 0.1 * np.std(feature) * _noise()

    def _noise() -> np.ndarray:
        "compute one complete noise feature column"
        return np.random.normal(size=(dataset_x.shape[0], 1))

    def _lin_comb() -> np.ndarray:

        weights = np.random.normal(loc=0, scale=1, size=(dataset_x.shape[1], 1))
        feature = np.dot(dataset_x, weights)

        return feature + 0.1 * np.std(feature) * _noise()

    def _redundant() -> np.ndarray:
        idx = np.random.randint(dataset_x.shape[1])
        feature = dataset_x[:, idx : idx + 1]
        return feature + 0.1 * np.std(feature) * _noise()

    feature_columns = [dataset_x]
    feature_columns.extend(_dropout() for _ in range(n_comb_lin_droppout))
    feature_columns.extend(_noise() for _ in range(n_noise))
    feature_columns.extend(_lin_comb() for _ in range(n_lin_comb))
    feature_columns.extend(_redundant() for _ in range(n_redundant))

    merged = np.concatenate(feature_columns, axis=1)

    assert (
        dataset_x.shape[0] == merged.shape[0]
    ), "invalid number of objects after transformation"
    assert (
        dataset_x.shape[1] + n_comb_lin_droppout + n_noise + n_lin_comb + n_redundant
        == merged.shape[1]
    ), "invalid number of features after transformation"

    return merged


def _compare_mdi_perm(
    dataset_x, dataset_y, feature_names, model_fn, random_state
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # two series: one for MDI, one for MDA
    (mda_mean, mda_std), (mdi_mean, mdi_std) = get_feature_importances(
        dataset_x, dataset_y, model_fn, random_state
    )

    mean = pd.DataFrame()
    mean["MDI"] = pd.Series(mdi_mean, index=feature_names)
    mean["Perm"] = pd.Series(mda_mean, index=feature_names)

    std = pd.DataFrame()
    std["MDI"] = pd.Series(mdi_std, index=feature_names)
    std["Perm"] = pd.Series(mda_std, index=feature_names)

    return mean, std


def _compare_extra_features(
    dataset_x,
    dataset_y,
    noisy_offset: int,
    feature_names,
    model_fn,
    random_state,
    title: str = "",
    save_to: str = "",
):
    def extra_nan(df: pd.DataFrame):
        extras = pd.DataFrame()

        for col in df.columns:
            extras[col] = pd.Series(
                [np.nan] * (len(feature_names) - noisy_offset),
                index=feature_names[noisy_offset:],
            )

        return extras

    base = _compare_mdi_perm(
        dataset_x=dataset_x[:, :noisy_offset],
        dataset_y=dataset_y,
        feature_names=feature_names[:noisy_offset],
        model_fn=model_fn,
        random_state=random_state,
    )

    # add NaNs so the two dataset are aligned on the plot
    base = [pd.concat((df, extra_nan(df))) for df in base]

    full = _compare_mdi_perm(
        dataset_x=dataset_x,
        dataset_y=dataset_y,
        feature_names=feature_names,
        model_fn=model_fn,
        random_state=random_state,
    )

    # put the two in comparable plots

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True)

    if title:
        fig.suptitle(title)

    base[0].plot.barh(xerr=base[1], ax=axes[0])
    full[0].plot.barh(xerr=full[1], ax=axes[1])

    axes[0].grid(axis="x", which="both")
    axes[1].grid(axis="x", which="both")

    fig.tight_layout()
    plt.show()

    if save_to:
        fig.savefig(save_to)


def study_pure_noise(ds_x, ds_y, model_fn, feature_labels, random_state):
    "add pure noise features to the dataset and show its impact on feature selection"

    n_noise = 3

    ds_x_extra = add_features(ds_x, n_noise=n_noise)

    feature_labels = feature_labels + [f"noise_{i+1}" for i in range(n_noise)]

    _compare_extra_features(
        ds_x_extra,
        ds_y,
        ds_x.shape[1],
        feature_labels,
        model_fn,
        random_state,
        title="Effect of purely random features",
        save_to="tmp_noise.png",
    )


def study_duplicates(ds_x, ds_y, model_fn, feature_labels, random_state):
    "add duplicate features to the dataset and show its impact on feature selection"

    n_redundant = 3

    ds_x_extra = add_features(ds_x, n_redundant=n_redundant)

    feature_labels = feature_labels + [f"noise_{i+1}" for i in range(n_redundant)]

    _compare_extra_features(
        ds_x_extra,
        ds_y,
        ds_x.shape[1],
        feature_labels,
        model_fn,
        random_state,
        title="Effect of redundant random features",
        save_to="tmp_redundant.png",
    )


def study_duplicates_gn(ds_x, ds_y, model_fn, feature_labels, random_state):
    "add duplicate features with gaussian noise to the dataset and show its impact on feature selection"

    n_lin_comb = 3

    ds_x_extra = add_features(ds_x, n_lin_comb=n_lin_comb)

    feature_labels = feature_labels + [f"noise_{i+1}" for i in range(n_lin_comb)]

    _compare_extra_features(
        ds_x_extra,
        ds_y,
        ds_x.shape[1],
        feature_labels,
        model_fn,
        random_state,
        title="Effect of correlated random features (linear combination)",
        save_to="tmp_lin_comb.png",
    )


def study_duplicates_correlated(ds_x, ds_y, model_fn, feature_labels, random_state):
    "add duplicate features with correlated (linear combination then dropout) to the dataset and show its impact on feature selection"

    n_lin_comb_dropout = 3

    ds_x_extra = add_features(ds_x, n_comb_lin_droppout=n_lin_comb_dropout)

    feature_labels = feature_labels + [
        f"noise_{i+1}" for i in range(n_lin_comb_dropout)
    ]

    _compare_extra_features(
        ds_x_extra,
        ds_y,
        ds_x.shape[1],
        feature_labels,
        model_fn,
        random_state,
        title="Effect of correlated random features (linear combination with dropout)",
        save_to="tmp_lin_comb_dropout.png",
    )


def show_datasize_learning_curve(
    dataset_x: np.ndarray,
    dataset_y: np.ndarray,
    model,
    cv=None,
    save_to: str = "",
    show: bool = False,
):

    if not save_to and not show:
        raise ValueError(f"at least one of {save_to=}, {show=} should be set")

    # 0.1, 0.2, ..., 0.9, 1.0
    percentage = np.arange(0.2, 1.2, 0.2)

    def _get_score_for_percentage(p: float):
        "get the score using p percent of the data available"
        r_mask = np.random.rand(dataset_x.shape[0]) < p

        ds_x = dataset_x[r_mask, :]
        ds_y = dataset_y[r_mask]

        cv_scores = evaluate_cv(model, ds_x, ds_y, cv)

        return np.mean(cv_scores)

    fn = joblib.delayed(_get_score_for_percentage)
    scores = joblib.Parallel(n_jobs=-1)(fn(p) for p in percentage)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    fig.suptitle("Estimated Accuracy as a Function of the Dataset Size")

    ax.set_ylabel("Estimated Accuracy (%)")
    ax.set_xlabel("Relative size of the dataset (%)")

    ax.plot(percentage, scores)

    if save_to:
        fig.savefig(save_to)

    if show:
        plt.show()


def main():

    np.random.seed(5876)
    ds_x = add_input_noise(np.load("ds_x.npy"), rel_scale=2)
    ds_y = np.load("ds_y.npy")

    # selected = get_feature_idx(ds_x, ds_y, random_state=48)
    # print(f"{selected=}")
    selected = (73, 61, 67)  # save time
    ds_x = ds_x[:, selected]

    kwargs = {
        "ds_x": ds_x,
        "ds_y": ds_y,
        "model_fn": new_model,
        "feature_labels": [f"f_{i}" for i in selected],
        "random_state": 33,
    }

    # evaluate the impact of (model, noisy features, correlated features)
    study_duplicates(**kwargs)
    study_duplicates_gn(**kwargs)
    study_duplicates_correlated(**kwargs)
    study_pure_noise(**kwargs)

    # do_plot(ds_x, ds_y, random_state=48)


if __name__ == "__main__":
    main()
