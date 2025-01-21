# %%

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple
from warnings import warn

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import sklearn
from matplotlib import cm
from scipy.stats import rankdata, mannwhitneyu, false_discovery_control
from scipy.ndimage import maximum_filter1d
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import auc, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, LeaveOneGroupOut, cross_val_score, cross_val_predict, cross_validate


# %%


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    annotation_overlap: np.ndarray
    annotation_idx: np.ndarray
    bin_l: np.ndarray
    bin_r: np.ndarray
    regions: np.ndarray
    coord_y: np.ndarray
    coord_x: np.ndarray

    def select_region(self, region: int | float):
        "returns a new dataset only containing the given region"

        mask = self.regions == region
        return Tabular(
            self.dataset_x[mask].copy(),
            self.annotation_overlap[mask, :].copy(),
            self.annotation_idx[mask, :].copy(),
            self.bin_l.copy(),
            self.bin_r.copy(),
            self.regions[mask].copy(),
            self.coord_y[mask].copy(),
            self.coord_x[mask].copy(),
        )

    def max_pool(self):
        return Tabular(
            maximum_filter1d(self.dataset_x, 3, axis=-1, mode="constant", cval=0),
            self.annotation_overlap.copy(),
            self.annotation_idx.copy(),
            self.bin_l.copy(),
            self.bin_r.copy(),
            self.regions.copy(),
            self.coord_y.copy(),
            self.coord_x.copy(),
        )


# %%


def selection(
    models: list[tuple[BaseEstimator, dict[str, Any]]],
    X_: np.ndarray,
    y_: np.ndarray,
    g_: np.ndarray,
    w_: np.ndarray,
    scorer_: _BaseScorer,
    n_splits: int,
):
    "return the best fitted estimator on the dataset"

    best_: tuple[BaseEstimator | None, float] = (None, -1.0)

    n_groups = len(np.unique(g_))
    if n_groups < n_splits:
        warn(f"{n_splits=} is decreased to {n_groups=}")
        n_splits = n_groups
    split_ = StratifiedGroupKFold(n_splits, shuffle=True, random_state=32)
    folds = list(split_.split(X_, y_, g_))

    for model_, a_dct_ in models:
        search = GridSearchCV(model_, a_dct_, scoring=scorer_, cv=folds, n_jobs=-1)
        search.fit(X_, y_, sample_weight=w_)  # w_ is split automatically

        if np.isnan(search.best_score_):
            raise RuntimeError(f"encountered NaN score for {model_=}, {a_dct_=}")

        if search.best_score_ > best_[1]:
            best_ = (search.best_estimator_, search.best_score_)

    if best_[0] is None:
        raise RuntimeError("no model yielded a non-NaN score")

    return best_[0]


def asses_selection(
    models: list[tuple[BaseEstimator, dict[str, Any]]],
    X_: np.ndarray,
    y_: np.ndarray,
    g_: np.ndarray,
    w_: np.ndarray,
    scorer_: _BaseScorer,
    n_splits: int,
    *,
    decrease_split_select: bool = False,
):
    "evaluate the selection procedure"

    n_groups = len(np.unique(g_))
    if n_groups < n_splits:
        warn(f"{n_splits=} is decreased to {n_groups=}")
        n_splits = n_groups
    split_ = StratifiedGroupKFold(n_splits, shuffle=True, random_state=32)

    scores = np.empty((n_splits,), dtype=float)
    if decrease_split_select:
        n_splits -= 1

    for idx, (tr_, te_) in enumerate(split_.split(X_, y_, g_)):
        model = selection(models, X_[tr_], y_[tr_], g_[tr_], w_[tr_], scorer_, n_splits)
        scores[idx] = scorer_(model, X_[te_], y_[te_], w_[te_])

    return scores


def fit_and_eval(
    models: list[tuple[BaseEstimator, dict[str, Any]]],
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["none", "annotations", "regions"],
    weighting: bool,
    scorer_: _BaseScorer,
    n_splits: int,
):
    """fit_and_eval: select the best model across `models` on `dataset_`.

    If `grouping` is "regions" and `n_splits` is the number of regions, it will
    be adjusted for the nested selection of the assessment automatically.

    Here are the labels for the different `problem`
        - 'ls+/ls-': 0 -> 'ls-', 1 -> 'ls+'
        - 'sc+/sc-': 0 -> 'sc-', 1 -> 'sc+'
        - 'ls/sc'  : 0 -> 'sc' , 1 -> 'ls'
        - '+/-'    : 0 -> '-'  , 1 -> '+'

    Returns
        - the selected models
        - the CV-scores on each fold of *the selection procedure* (evaluated models may differ)
    """

    X, y, w, groups = _ds_to_Xy(dataset_, problem, grouping, "highest")

    if not weighting:
        w.fill(1.0)

    scores = asses_selection(
        models,
        X,
        y,
        groups,
        w,
        scorer_,
        n_splits,
        decrease_split_select=(grouping == "regions" and n_splits == 3),
    )
    model = selection(models, X, y, groups, w, scorer_, n_splits)

    return model, scores


def cv_score(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["none", "annotations", "regions"],
    weighting: Literal[False],
    scorer_: _BaseScorer,
    cv,
):
    """fit and evaluate a model on `dataset_`.

    If `grouping` is "regions" and `n_splits` is the number of regions, it will
    be adjusted for the nested selection of the assessment automatically.

    Here are the labels for the different `problem`
        - 'ls+/ls-': 0 -> 'ls-', 1 -> 'ls+'
        - 'sc+/sc-': 0 -> 'sc-', 1 -> 'sc+'
        - 'ls/sc'  : 0 -> 'sc' , 1 -> 'ls'
        - '+/-'    : 0 -> '-'  , 1 -> '+'

    Returns
        - the CV-score
    """

    X, y, _, groups = _ds_to_Xy(dataset_, problem, grouping, "highest")
    assert weighting is False

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        scoring=scorer_,
        cv=cv,
        n_jobs=-1,
    )

    return scores


def cv_logo_roc_auc(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["annotations", "regions"],
    weighting: Literal[False],
):

    X, y, _, g = _ds_to_Xy(dataset_, problem, grouping, "highest")
    assert weighting is False

    y_pred = cross_val_predict(
        model,
        X,
        y,
        groups=g,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
        method="predict_proba",
    )

    return roc_auc_score(y, y_pred[:, 1]), (y, y_pred)


def cv_logo_detailed_acc(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["annotations", "regions"],
    weighting: Literal[False],
    *,
    return_train_score: bool = True,
):
    """cv_logo_detailed_acc: share some details about grouped cross validation

    This function returns the training and validation accuracy on each of the folds, as well as the indices in of them.
    """

    X, y, _, g, row_idx, col_idx = _ds_to_Xy_with_idx(dataset_, problem, grouping, "highest")
    assert weighting is False

    infos = {
        "X": X,
        "y": y,
        "g": g,
        "groups": dataset_.annotation_idx[row_idx, col_idx],
        "regions": dataset_.regions[row_idx],
        "coords_y": dataset_.coord_y[row_idx],
        "coords_x": dataset_.coord_x[row_idx],
    }

    return cross_validate(
        model,
        X,
        y,
        groups=g,
        scoring="accuracy",
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
        return_train_score=return_train_score,
        return_estimator=False,
        return_indices=True,  # type: ignore
    ), infos


# %%s


def draw_2d_detailed_results(
    title: str,
    results: dict[str, Any],
    infos: dict[str, np.ndarray],
):
    if not np.allclose(infos["g"], infos["groups"]):
        raise ValueError("Only supported if grouping='groups'")

    regions = np.unique(infos["regions"])

    # x and y axis must be shared by column only
    fig, axes = plt.subplots(
        2 if "train_score" in results else 1,
        len(regions),
        sharex="col",
        sharey="col",
        squeeze=False,
        figsize=(12, 6)
    )

    c_val = 2.0 * infos["y"] - 1.0

    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cmap = cm.get_cmap("RdBu")
    cmap.set_bad("lightgray")

    ys, xs = infos["coords_y"], infos["coords_x"]

    for col, reg in enumerate(regions):
        height = ys[infos["regions"] == reg].max() + 1
        width = xs[infos["regions"] == reg].max() + 1
        # tr_map is computed even if it's not used: not an issue.
        tr_map = np.zeros((height, width), float)
        vl_map = np.zeros_like(tr_map)

        tr_map[:] = np.nan
        vl_map[:] = np.nan
        masks[reg] = (tr_map, vl_map)

    for fold_idx, vl_score in enumerate(results["test_score"]):
        # which region is it
        vl_idx = results["indices"]["test"][fold_idx]
        reg_idx = infos["regions"][vl_idx]
        assert len(np.unique(reg_idx)) == 1, "expected a single region per fold"

        if reg_idx[0] not in masks:
            continue
        tr_map, vl_map = masks[reg_idx[0]]
        vl_map[ys[vl_idx], xs[vl_idx]] = vl_score * c_val[vl_idx]

        if "train_score" in results:
            tr_score = results["train_score"][fold_idx]
            tr_map[ys[vl_idx], xs[vl_idx]] = tr_score * c_val[vl_idx]

    pcm: cm.ScalarMappable | None = None
    for col, reg in enumerate(regions):
        tr_map, vl_map = masks[reg]
        min_y = ys[infos["regions"] == reg].min()
        min_x = xs[infos["regions"] == reg].min()

        if col == 0:
            axes[0, 0].set_ylabel("Validation score")
        pcm = axes[0, col].pcolormesh(vl_map[min_y:, min_x:], cmap=cmap, vmin=-1.0, vmax=1.0)
        if "train_score" in results:
            if col == 0:
                axes[1, 0].set_ylabel("Training score")
            axes[1, col].pcolormesh(tr_map[min_y:, min_x:], cmap=cmap, vmin=-1.0, vmax=1.0)
    assert pcm is not None, "no graph to color"

    fig.suptitle(title)
    fig.tight_layout()

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes((0.98, 0.1, 0.02, 0.8))
    fig.colorbar(pcm, cax=cbar_ax)

    return fig, axes


def show_detail_cv(res_info_tpl):
    "show bars to represent each fold"
    res, infos = res_info_tpl

    colors = ["tab:orange", "tab:blue", "tab:green"]
    test_indices = res["indices"]["test"]
    region_per_fold = [np.unique(infos["regions"][i]) for i in test_indices]
    if any(len(r) > 1 for r in region_per_fold):
        raise RuntimeError("multiple regions in a fold")
    group_per_fold = [np.unique(infos["groups"][i]) for i in test_indices]
    if any(len(r) > 1 for r in group_per_fold):
        raise RuntimeError("multiple groups in a fold")

    color_per_fold = [colors[r[0]] for r in region_per_fold]
    label = [g[0] for g in group_per_fold]

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].bar(
        label,
        res["test_score"],
        color=color_per_fold,
    )
    axes[0].set_ylabel("test score")
    axes[0].set_ylim((0, 1))

    axes[1].bar(
        label,
        res["train_score"],
        color=color_per_fold,
    )
    axes[1].set_ylabel("train score")
    axes[1].set_ylim((0, 1))

    return fig, axes


# %%


def show_average_groupped_roc_auc(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["annotations", "regions"],
    weighting: Literal[False],
    n_splits: int,
):

    X, y, _, g = _ds_to_Xy(dataset_, problem, grouping, "highest")
    assert weighting is False

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y, g)):
        model.fit(X[train], y[train])  # type: ignore
        viz = RocCurveDisplay.from_estimator(
            model,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)  # type: ignore
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)  # type: ignore

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability",
    )
    ax.legend(loc="lower right")

    return fig, ax


# %%


def cv_score_by_region(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc"],
    weighting: Literal[False],
    scorer_: _BaseScorer,
):
    X, y, _, regions = _ds_to_Xy(dataset_, problem, "regions", "highest")
    assert weighting is False

    indices = np.arange(len(y))

    # manual folds
    region_values = np.unique(regions)
    assert list(region_values) == [0, 1, 2]

    folds = [
        (indices[regions != r], indices[regions == r])
        for r in region_values
    ]

    scores = np.zeros((len(folds),), dtype=float)
    for i, (tr_idx, vl_idx) in enumerate(folds):
        model.fit(X[tr_idx], y[tr_idx])  # type: ignore
        scores[i] = scorer_(model, X[vl_idx], y[vl_idx])

    return scores


def cv_score_by_region_2d(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc"],
    weighting: Literal[False],
    scorer_: _BaseScorer,
):
    X, y, _, regions = _ds_to_Xy(dataset_, problem, "regions", "highest")
    assert weighting is False

    indices = np.arange(len(y))

    # manual folds
    region_values = np.unique(regions)
    assert list(region_values) == [0, 1, 2]

    masks = [indices[regions == r] for r in region_values]

    scores = np.zeros((len(region_values), len(region_values)), dtype=float)
    for reg_l in region_values:
        for reg_r in region_values:
            tr = masks[reg_l]
            vl = masks[reg_r]
            model.fit(X[tr], y[tr])  # type: ignore
            scores[reg_l, reg_r] = scorer_(model, X[vl], y[vl])

    return scores


# %%

def u_test(
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    *,
    fdr: float | None = None,
    num_features: int | None = None,
    fdr: float = 5e-2,
    *,
    fdr: float | None = None,
    num_features: int | None = None,
):

    if fdr is None and num_features is None:
        raise ValueError(f"at least one of {fdr=} and {num_features=} must be given")
    if fdr is not None and num_features is not None:
        raise ValueError(f"{fdr=} and {num_features=} may not be both given")

    if fdr is None and num_features is None:
        raise ValueError(f"at least one of {fdr=} and {num_features=} must be given")
    if fdr is not None and num_features is not None:
        raise ValueError(f"{fdr=} and {num_features=} may not be both given")

    X, y, _, _ = _ds_to_Xy(dataset_, problem, "none", "highest")
    m = X.shape[1]

    # Mann-Whitney U test
    p_values = np.ones(shape=(m,), dtype=float)
    for feature_idx in range(m):
        # split attribute into two
        ds_pos = X[y == 1, feature_idx]
        ds_neg = X[y == 0, feature_idx]

        _, p_values[feature_idx] = mannwhitneyu(ds_pos, ds_neg)
        if feature_idx % 1000 == 0:
            print(f"{feature_idx=}")

    # apply Benjamini-Hochberg procedure
    s_idx = np.argsort(p_values)
    sorted_p_values = p_values[s_idx]

    # apply Benjamini-Hochberg procedure to control p-values
    p_values = false_discovery_control(p_values[s_idx], method="bh")

    if fdr is not None:
        # reject until the rate is fdr
        right_end = np.argmax(p_values > fdr)
        if not p_values[right_end] > fdr:
            right_end = -1
    else:
        assert num_features is not None
        right_end = min(num_features, len(s_idx))
    # find highest item such that P[k] <= (k + 1) * fdr / m
    check = sorted_p_values <= (1 + np.arange(m)) * fdr / m
    max_idx = m - 1 - np.argmax(check[::-1])
    if not check.any():
        max_idx = 0
    # apply Benjamini-Hochberg procedure to control p-values
    p_values = false_discovery_control(p_values[s_idx], method="bh")

    if fdr is not None:
        # reject until the rate is fdr
        right_end = np.argmax(p_values > fdr)
        if not p_values[right_end] > fdr:
            right_end = -1
    else:
        assert num_features is not None
        right_end = min(num_features, len(s_idx))

    valid_idx = s_idx[:right_end]
    valid_p_values = p_values[:right_end]
    # selected features
    valid_idx = s_idx[:max_idx]
    valid_idx = s_idx[:right_end]
    valid_p_values = p_values[:right_end]
    bin_c = 0.5 * (dataset_.bin_l + dataset_.bin_r)

    return bin_c[valid_idx], valid_p_values
    return bin_c[valid_idx], p_values[valid_idx]
    return bin_c[valid_idx], valid_p_values

# %%


def _ds_to_Xy_with_idx(
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["none", "annotations", "regions"],
    label_mode: Literal["soft", "highest", "drop"],
    *,
    min_weight: float | None = 0.2,
):
    """_ds_to_Xy: prepare a Tabular dataset for a binary classification

    Args:
        dataset_ (Tabular): dataset to use
        problem (Literal[&quot;ls+/ls-&quot;, &quot;sc+/sc-&quot;, &quot;ls/sc&quot;, &quot;+/-&quot;]): problem to find the labels
        grouping (Literal[&quot;none&quot;, &quot;annotations&quot;, &quot;regions&quot;]): kind of grouping to use
        label_mode (Literal[&quot;soft&quot;, &quot;highest&quot;, &quot;drop&quot;]): how to handle spectra which have an overlap with more than one annotations: `"soft"` keeps all possibilities, `"highest"` only keep the label with the highest overlap, `"drop"` will drop any spectrum which has more than one possible label
        min_weight (float | None, optional): minimal value for the overlap, values lower than that are dropped. Defaults to 0.2.

    Raises:
        ValueError: on any invalid parameter

    Returns:
        tuple[ndarray, ...]: X, y, w, g, row_idx, col_idx
    """
    pos_labels: tuple[int, ...]
    neg_labels: tuple[int, ...]
    match problem:
        case "ls+/ls-":
            pos_labels, neg_labels = (0,), (1,)
        case "sc+/sc-":
            pos_labels, neg_labels = (2,), (3,)
        case "ls/sc":
            pos_labels, neg_labels = (0, 1), (2, 3)
        case "+/-":
            pos_labels, neg_labels = (0, 2), (1, 3)
        case _:
            raise ValueError(f"unsupported value for {problem=!r}")

    # no row without label
    assert dataset_.annotation_overlap.max(axis=1).min() > 0.0

    # find relevant items of the dataset
    if label_mode == "highest":
        row_idx = np.arange(len(dataset_.dataset_x))
        col_idx = dataset_.annotation_overlap.argmax(axis=1)
    elif label_mode == "drop":
        mask = (dataset_.annotation_overlap > 0).sum(axis=1) == 1
        row_idx = np.arange(len(dataset_.dataset_x))[mask]
        col_idx = dataset_.annotation_overlap.argmax(axis=1)[mask]
    elif label_mode == "soft":
        row_idx, col_idx = dataset_.annotation_overlap.nonzero()
    else:
        raise ValueError(f"unsupported value for {label_mode=!r}")

    # select only rows relevant to the classification problem
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(col_idx, neg_labels + pos_labels)
    row_idx = row_idx[cls_mask]
    col_idx = col_idx[cls_mask]

    # weighting
    w = dataset_.annotation_overlap[row_idx, col_idx]
    if min_weight:
        valid = w >= min_weight
        row_idx = row_idx[valid]
        col_idx = col_idx[valid]
        w = w[valid]

    # split between positive and negative (be consistant with old implementation !)
    y = np.where(np.isin(col_idx, pos_labels), 1, 0)

    # obtain grouping & weights
    match grouping:
        case "annotations":
            g = dataset_.annotation_idx[row_idx, col_idx]
        case "regions":
            g = dataset_.regions[row_idx]
        case "none":
            g = np.arange(y.size)
        case _:
            raise ValueError(f"unsupported value for {grouping=!r}")

    return dataset_.dataset_x[row_idx], y, w, g, row_idx, col_idx


def _ds_to_Xy(
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["none", "annotations", "regions"],
    label_mode: Literal["soft", "highest", "drop"],
    *,
    min_weight: float | None = 0.2,
):
    """_ds_to_Xy: prepare a Tabular dataset for a binary classification

    Args:
        dataset_ (Tabular): dataset to use
        problem (Literal[&quot;ls+/ls-&quot;, &quot;sc+/sc-&quot;, &quot;ls/sc&quot;, &quot;+/-&quot;]): problem to find the labels
        grouping (Literal[&quot;none&quot;, &quot;annotations&quot;, &quot;regions&quot;]): kind of grouping to use
        label_mode (Literal[&quot;soft&quot;, &quot;highest&quot;, &quot;drop&quot;]): how to handle spectra which have an overlap with more than one annotations: `"soft"` keeps all possibilities, `"highest"` only keep the label with the highest overlap, `"drop"` will drop any spectrum which has more than one possible label
        min_weight (float | None, optional): minimal value for the overlap, values lower than that are dropped. Defaults to 0.2.

    Raises:
        ValueError: on any invalid parameter

    Returns:
        tuple[ndarray, ...]: X, y, w, g
    """
    X, y, w, g, _, _ = _ds_to_Xy_with_idx(dataset_, problem, grouping, label_mode, min_weight=min_weight)
    return X, y, w, g


# %%

base_dir = Path("/home/mamodei/datasets/slim-deisotoping-2.0e-03-binned")
assert base_dir.is_dir()

# %%

norm: Literal["no", "317"] = "317"
bin_method: Literal["sum", "integration"] = "sum"
dataset = Tabular(*np.load(base_dir / f"binned_{bin_method}_{norm}norm.npz").values())

# %%


def mono(p_values):
    """Enforce monotonicity of the values"""

    pvalues_mono = np.zeros(len(p_values))
    pvalues_mono[0] = p_values[0]

    if -555 in p_values:
        nb_feat = list(p_values).index(-555)
        for i in range(1, nb_feat):
            pvalues_mono[i] = max(pvalues_mono[i - 1], p_values[i])
        pvalues_mono[nb_feat:] = -555
    else:
        for i in range(1, len(p_values)):
            pvalues_mono[i] = max(pvalues_mono[i - 1], p_values[i])

    return pvalues_mono


def cer_fdr(
    mprobes_results: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    model: ForestClassifier,
    nb_perm: int,
    n_to_compute: int | Literal["all"],
    thresh_stop: float,
):
    "wrapper around the CER_FDR function"

    def vImp_attr(x_: np.ndarray, y_: np.ndarray, _):
        m_: ForestClassifier = sklearn.base.clone(model)
        m_.fit(x_, y_)
        return m_.feature_importances_

    # sort the whole dataframe by importance
    results = mprobes_results.sort_values("Imp", ascending=False)
    del mprobes_results  # make sure we don't modify the given dataframe
    var_imp_noperm = results.Imp.to_numpy()

    # ranks are integers, ties are broken based on position in the array
    rank_noperm = rankdata(var_imp_noperm, method="ordinal") - 1

    CER, FDR, eFDR = _cer_fdr__inner(
        X,
        Y,
        vImp_attr,
        (None,),
        nb_perm,
        n_to_compute,
        thresh_stop,
        rank_noperm,
        var_imp_noperm,
    )

    # NaN for those which weren't computed (removes the -555.0)
    CER[CER == -555.0] = np.nan
    FDR[FDR == -555.0] = np.nan
    eFDR[eFDR == -555.0] = np.nan

    results["CER"] = CER
    results["FDR"] = FDR
    results["eFDR"] = eFDR

    return results


@nb.jit("void(f8[:], f8[:], f8[:])", nopython=True)
def __incr_fdr(fdr: np.ndarray, imp_orig: np.ndarray, imp_perm: np.ndarray):
    """increment FDR count. imp_orig and imp_perm are left unchanged.

    Args:
        fdr (np.ndarray): output fdr array
        imp_orig (np.ndarray): importance of the original data (sorted in dec.)
        imp_perm (np.ndarray): importance after permutation (sorted in dec.)
    """

    if not np.all(imp_orig[:-1] >= imp_orig[1:]):
        raise ValueError("imp_orig should be sorted in decreasing order")
    if not np.all(imp_perm[:-1] >= imp_perm[1:]):
        raise ValueError("imp_perm should be sorted in decreasing order")
    if len(set([fdr.shape, imp_orig.shape, imp_perm.shape])) != 1:
        raise ValueError("shape mismatch")

    count: int = 0
    it_perm: int = 0
    for it_orig in range(imp_orig.size):
        # the importance of the current variable
        thresh = imp_orig[it_orig]

        while it_perm < imp_perm.size:  # don't count past the end
            # if the random importance is higher, count up and move to the next one
            if imp_perm[it_perm] >= thresh:
                count += 1
                it_perm += 1
            else:
                # else keep this value for the next feature
                break

        # update count in the array
        fdr[it_orig] += count


@nb.jit("void(f8[:], f8[:], f8[:])", nopython=True)
def __incr_fdr_baseline(fdr: np.ndarray, imp_orig: np.ndarray, imp_perm: np.ndarray):
    "this is the trivial O(N^2) implementation, used for regression checking"

    for it_orig in range(imp_orig.size):
        for it_perm in range(imp_perm.size):
            if imp_perm[it_perm] >= imp_orig[it_orig]:
                fdr[it_orig] += 1


def check(size: int = 1024):
    fdr_1 = np.random.rand(size)
    fdr_2 = fdr_1.copy()

    orig = np.sort(np.random.rand(size))[::-1]
    perm = np.random.rand(size)

    __incr_fdr_baseline(fdr_1, orig, perm)
    __incr_fdr(fdr_2, orig, np.sort(perm)[::-1])

    assert np.allclose(fdr_1, fdr_2), f"{np.abs(fdr_2 - fdr_1).max()=!r}"


def _cer_fdr__inner(
    X: np.ndarray,
    Y: np.ndarray,
    get_vImp_method: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    param: Any,
    nb_perm: int,
    n_to_compute: int | Literal["all"],
    thresh_stop: float,
    rank_noperm: np.ndarray,
    var_imp_noperm: np.ndarray,
):
    """
    adapted from: https://academic.oup.com/bioinformatics/article/28/13/1766/234473

    Original author: Vân Anh Huynh-Thu

    CER,FDR,eFDR
    = cer_fdr(X,Y,param,nb_perm,n_to_compute,thresh_stop,rank_noperm,var_imp_noperm)
    Compute CER, FDR, and eFDR
    - X : inputs
    - Y : outputs
    - get_vImp_method: the function that computes a feature ranking
    - param : parameters of the feature ranking method
    - nb_perm : number of permutations for each variable
    - n_to_compute: number of variables for which to compute the CER and eFDR
    - thresh_stop: algorithm stops when both CER and eFDR become higher than thresh_stop
    - rank_noperm : original ranking
    - var_imp_noperm: original variable relevance scores, in decreasing order


    return :
    - CER for each subset of top-ranked variables
    - FDR for each subset of top-ranked variables
    - eFDR for each subset of top-ranked variables
    """

    # if get_vImp_method == get_vimp_tree:
    #     param.verbose = False

    nb_obj = X.shape[0]
    nb_feat = X.shape[1]

    if Y.shape[0] != nb_obj:
        raise ValueError(f"{Y.shape[0]=!r} != {X.shape[0]=}")

    if len(rank_noperm) != nb_feat:
        raise ValueError(f"{len(rank_noperm)=} != {X.shape[1]=}")

    if n_to_compute == "all":
        n_to_compute = nb_feat
    if n_to_compute <= 0 or n_to_compute > nb_feat:
        raise ValueError(f"{n_to_compute=!r} not in [1, {nb_feat=}]")

    FDR = np.zeros(nb_feat) - 555
    CER = np.zeros(nb_feat) - 555
    eFDR = np.zeros(nb_feat) - 555

    # Compute FDR for all variables and compute CER and eFDR for the top-ranked variable
    FDR = np.zeros(nb_feat)
    CER[0] = 0
    eFDR[0] = 0

    # Permutation of the output values
    print("feature 1...")
    for t in range(nb_perm):
        print("feature 1 - permutation %d..." % (t + 1))
        Y_perm = Y.copy()
        np.random.shuffle(Y_perm)

        var_imp_perm = get_vImp_method(X, Y_perm, param)

        # CER
        if max(var_imp_perm) >= var_imp_noperm[0]:
            CER[0] += 1

        # # FDR
        # for i in range(len(var_imp_noperm)):
        #     FDR[i] += np.sum(var_imp_perm >= var_imp_noperm[i])

        # NOTE: this is O(N*log(N)) instead of O(N^2), which matters a lot here
        var_imp_perm_sorted = np.sort(var_imp_perm)[::-1]
        __incr_fdr(FDR, var_imp_noperm, var_imp_perm_sorted)

        # # eFDR
        # var_imp_perm_sort = var_imp_perm.copy()
        # var_imp_perm_sort.sort(axis=0)
        # var_imp_perm_sort = np.flipud(var_imp_perm_sort)
        # R1_e = nb_feat
        # for ii in range(len(var_imp_perm_sort)):
        #     if var_imp_perm_sort[ii] < var_imp_noperm[ii]:
        #         R1_e = ii
        #         break

        # NOTE updated -> numpy
        R1_e = int(np.argmax(var_imp_perm_sorted < var_imp_noperm))
        if var_imp_perm_sorted[R1_e] >= var_imp_noperm[R1_e]:
            R1_e = nb_feat

        if R1_e > 0:
            eFDR[0] += 1

    # for i in range(len(FDR)):
    #     FDR[i] = FDR[i] / nb_perm
    #     FDR[i] = FDR[i] / (i + 1)
    #     if FDR[i] > 1:
    #         FDR[i] = 1

    # NOTE updated -> numpy
    FDR /= np.arange(1, FDR.size + 1)
    FDR = np.minimum(FDR / nb_perm, 1.0)

    CER[0] = CER[0] / nb_perm
    eFDR[0] = eFDR[0] / nb_perm

    # Compute CER and eFDR for the remaining variables
    for i_current in range(1, n_to_compute):
        # Stop if CER and eFDR are higher than thresh_stop
        if CER[i_current - 1] > thresh_stop and eFDR[i_current - 1] > thresh_stop:
            break

        print("feature %d..." % (i_current + 1))

        # If FDR = 0, then CER and eFDR = 0
        if FDR[i_current] == 0:
            CER[i_current] = 0
            eFDR[i_current] = 0
        # To save time, as soon as a variable has a CER and eFDR = 1, the CER and eFDR of the variables ranked below are also set to 1
        elif CER[i_current - 1] == 1 and eFDR[i_current - 1] == 1:
            CER[i_current] = 1
            eFDR[i_current] = 1
        else:
            CER[i_current] = 0
            eFDR[i_current] = 0

            for t in range(nb_perm):
                print("feature %d - permutation %d..." % (i_current + 1, t + 1))

                order_perm = np.random.permutation(nb_obj)
                Y_perm = Y[order_perm].copy()
                X_perm = X.copy()
                for i in range(i_current):
                    X_perm[:, rank_noperm[i]] = X_perm[order_perm, rank_noperm[i]]

                print("getting vImp...", end="")
                var_imp_perm = get_vImp_method(X_perm, Y_perm, param)
                print("\b\b\b: done!")

                # Remove un-permuted variables
                var_imp_perm = np.delete(var_imp_perm, rank_noperm[:i_current], axis=0)
                var_imp_perm_sort = var_imp_perm.copy()
                var_imp_perm_sort.sort(axis=0)
                var_imp_perm_sort = np.flipud(var_imp_perm_sort)

                # CER
                if var_imp_perm_sort[0] >= var_imp_noperm[i_current]:
                    CER[i_current] += 1

                # # eFDR
                # Ri_e = nb_feat - i_current
                # for ii in range(len(var_imp_perm_sort)):
                #     if var_imp_perm_sort[ii] < var_imp_noperm[i_current + ii]:
                #         Ri_e = ii
                #         break

                # NOTE updated
                Ri_e = int(np.argmax(var_imp_perm_sort < var_imp_noperm[i_current:]))
                if var_imp_perm_sort[Ri_e] >= var_imp_noperm[i_current + Ri_e]:
                    Ri_e = nb_feat - i_current

                fi_e = float(Ri_e) / (Ri_e + i_current)
                eFDR[i_current] += fi_e

            CER[i_current] = CER[i_current] / nb_perm
            eFDR[i_current] = eFDR[i_current] / nb_perm

    CER = mono(CER)
    FDR = mono(FDR)
    eFDR = mono(eFDR)

    return CER, FDR, eFDR


def mprobes(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    model: RandomForestClassifier,
    nb_perm: int,
    join_permute: bool = False,
    verbose: bool = True,
):
    """
    adapted from: https://github.com/tuxette/ecas23_randomforest/blob/e67842949bcc5950da7c49753a4e718e2c786884/pierre/vimp/ECAS.py#L56
    changes: type hint annotations, formatting, removal of unused parameter.

    Original author: Vân Anh Huynh-Thu
    p,rank_mprobes = mprobes(X,Y,model)

    Args:
        X (np.ndarray | pd.DataFrame): inputs (needs to be a numpy array or a panda DataFrame of size (nsamples, nfeat))
        Y (np.ndarray | pd.DataFrame): outputs (needs to be a numpy array or a panda DataFrame of size (nsamples,))
        model (RandomForestClassifier): a scikit-learn model that fill in model.feature_importances_
        nb_perm (int): number of iterations
        join_permute (bool, optional): whether the contrast variables are permuted jointly or not. Defaults to False.
        verbose (bool, optional): if False, no messages are printed. Defaults to True.

    Returns:
        pd.DataFrame: a pandas DataFrame with original importance scores and computed FWER
    """

    if verbose:
        print("Compute the initial ranking...")

    nb_obj = X.shape[0]
    nb_feat = X.shape[1]

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns.values)
        X = X.values[range(nb_obj)]
        assert isinstance(Y, pd.DataFrame)
        Y = Y.values[range(nb_obj)]
    else:
        feature_names = ["f" + str(i) for i in range(nb_feat)]

    model.fit(X, Y)
    vimp0 = model.feature_importances_

    nb_obj = X.shape[0]
    nb_feat = X.shape[1]

    if Y.shape[0] != nb_obj:
        raise ValueError("X and Y must have the same number of objects.")

    # data
    X_full = np.zeros((nb_obj, nb_feat * 2), dtype=np.float32)
    X_full[:, :nb_feat] = X  # removed useless copy

    p = np.zeros(nb_feat)

    if verbose:
        print("Compute the permutation rankings...")

    for _ in range(nb_perm):
        if verbose:
            print(".", end="")
        # Artificial contrasts
        if join_permute:
            rand_ind = np.random.permutation(nb_obj)
            X_full[:, nb_feat:] = X[rand_ind, :]
        else:
            for i in range(nb_feat):
                feat_shuffle = X[:, i].copy()
                np.random.shuffle(feat_shuffle)
                X_full[:, i + nb_feat] = feat_shuffle

        # Learn an ensemble of tree and compute variable relevance scores
        var_imp = model.fit(X_full, Y).feature_importances_

        # Highest relevance score among all contrasts
        contrast_imp_max = np.max(var_imp[nb_feat : nb_feat * 2])

        # Original variables with a score lower than the highest score among all contrasts
        irr_var_idx = np.where(var_imp[:nb_feat] <= contrast_imp_max)
        p[irr_var_idx] += 1

    p = p / nb_perm

    R = np.zeros((nb_feat, 2))
    R[:, 0] = vimp0
    R[:, 1] = p
    Rd = pd.DataFrame(R, index=feature_names, columns=["Imp", "FWER"])

    return Rd


# %%


def run_lasso_param_sweep() -> None:

    class LassoClsCfg(NamedTuple):
        coeff: float
        problem: Literal["ls/sc", "ls+/ls-", "sc+/sc-"]
        bin_method: Literal["sum", "integration", "max-pool"]

    coeffs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e2, 1e3]
    problems: list[Literal["ls/sc", "ls+/ls-", "sc+/sc-"]] = [
        "ls/sc",
        "ls+/ls-",
        "sc+/sc-",
    ]
    bin_methods: list[Literal["sum", "integration", "max-pool"]] = ["sum", "integration", "max-pool"]

    configs = [LassoClsCfg(c_c, p_c, b_c) for p_c, c_c, b_c in product(problems, coeffs, bin_methods)]

    res_dir = Path(__file__).parent.parent / "res-exp-cv-untargeted-scores-lasso"
    res_dir.mkdir(exist_ok=True)

    exp_dir = res_dir / datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
    exp_dir.mkdir()

    for idx, cfg in enumerate(configs):
        cfg_dir = exp_dir / f"cfg-{idx}"
        cfg_dir.mkdir()

        with open(cfg_dir / "info.txt", "w", encoding="utf8") as cfg_info_f:
            print(f"{cfg.problem=} {cfg.bin_method=} {cfg.coeff=}", file=cfg_info_f)

        bm = cfg.bin_method
        if cfg.bin_method == "max-pool":
            bm = "sum"

        ds = Tabular(*np.load(base_dir / f"binned_{bm}_317norm.npz").values())
        if cfg.bin_method == "max-pool":
            ds = ds.max_pool()

        score = cv_logo_roc_auc(
            LogisticRegression(
                C=cfg.coeff,
                penalty="l1",
                solver="liblinear",
                random_state=0,
                # n_jobs=-2,  # no effect with liblinear and I don't like warnings
            ),
            dataset_=ds,
            problem=cfg.problem,
            grouping="annotations",
            weighting=False,
        )

        with open(cfg_dir / "score.txt", "w", encoding="utf8") as cfg_score_f:
            print(str(score), file=cfg_score_f)

# %%


def run_tree_param_sweep() -> None:

    class Configuration(NamedTuple):
        n_trees: int
        max_feat: Literal["sqrt"] | None
        max_depth: int | None
        problem: Literal["ls/sc", "ls+/ls-", "sc+/sc-"]
        bin_method: Literal["sum", "integration", "max-pool"]

    tree_configs: list[tuple[int, Literal["sqrt"] | None, int | None]] = [
        (100, None, 2),
        (100, None, 3),
        (1000, None, 3),
        (100, None, 5),
        (100, None, 10),
        (100, None, None),
        (10000, "sqrt", 2),
        (10000, "sqrt", 3),
        # (10, "sqrt", 1),
    ]
    problems: list[Literal['ls/sc', 'ls+/ls-', 'sc+/sc-']] = [
        "ls/sc",
        "ls+/ls-",
        "sc+/sc-",
    ]
    bin_methods: list[Literal['sum', 'integration', 'max-pool']] = [
        "sum",
        "integration",
        "max-pool",
    ]

    configs_ = [
        Configuration(*t_c, p_c, b_c)
        for t_c, p_c, b_c in product(tree_configs, problems, bin_methods)
    ]

    res_dir = Path(__file__).parent.parent / "res-exp-cv-untargeted-scores"
    res_dir.mkdir(exist_ok=True)

    exp_dir = res_dir / datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
    exp_dir.mkdir()

    for idx, cfg in enumerate(configs_):
        cfg_dir = exp_dir / f"cfg-{idx}"
        cfg_dir.mkdir()

        with open(cfg_dir / "info.txt", "w", encoding="utf8") as cfg_info_f:
            print(f"{cfg.problem=} cfg.bin_method=max-pool {cfg.n_trees=} {cfg.max_feat=} {cfg.max_depth=}", file=cfg_info_f)

        bm = cfg.bin_method
        if cfg.bin_method == "max-pool":
            bm = "sum"

        ds = Tabular(*np.load(base_dir / f"binned_{bm}_317norm.npz").values())
        if cfg.bin_method == "max-pool":
            ds = ds.max_pool()

        score = cv_logo_roc_auc(
            RandomForestClassifier(
                cfg.n_trees,
                max_features=cfg.max_feat,  # type:ignore
                max_depth=cfg.max_depth,
                random_state=0,
                n_jobs=-2,
            ),
            dataset_=ds,
            problem=cfg.problem,
            grouping="annotations",
            weighting=False,
        )

        with open(cfg_dir / "score.txt", "w", encoding="utf8") as cfg_score_f:
            print(str(score), file=cfg_score_f)

# %%
