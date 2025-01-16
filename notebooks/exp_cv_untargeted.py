# %%

import argparse
from datetime import datetime
from itertools import product  # noqa
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Protocol
from warnings import warn

import numba as nb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata, mannwhitneyu
from scipy.ndimage import maximum_filter1d
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier  # noqa:F401
from sklearn.ensemble._forest import ForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics._scorer import _BaseScorer, roc_auc_scorer  # noqa
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, LeaveOneGroupOut, cross_val_score, cross_val_predict, cross_validate

# %% protocols and stuff


class FitPred(Protocol):
    "A (restrictive) protocol for most useful classifiers in scikit-learn"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "FitPred": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# %%


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    dataset_y: np.ndarray
    groups: np.ndarray
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
            self.dataset_y[mask].copy(),
            self.groups[mask].copy(),
            self.bin_l.copy(),
            self.bin_r.copy(),
            self.regions[mask].copy(),
            self.coord_y[mask].copy(),
            self.coord_x[mask].copy(),
        )

    def max_pool(self):
         return Tabular(
            maximum_filter1d(self.dataset_x, 3, axis=-1, mode="constant", cval=0),
            self.dataset_y.copy(),
            self.groups.copy(),
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
    grouping: Literal["none", "groups", "regions"],
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

    flat_w = dataset_.dataset_y.flatten()  # [N * L]
    re_idx = (
        np.expand_dims(np.arange(dataset_.dataset_y.shape[0]), axis=1)
        .repeat(dataset_.dataset_y.shape[1], axis=1)
        .flatten()
    )
    re_cls = (
        np.expand_dims(np.arange(dataset_.dataset_y.shape[1]), axis=0)
        .repeat(dataset_.dataset_y.shape[0], axis=0)
        .flatten()
    )

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(re_cls, neg_labels + pos_labels) & (flat_w > 0)
    mask_bin_cls = re_idx[cls_mask]

    y = np.where(np.isin(re_cls[cls_mask], pos_labels), 1, 0)
    w = flat_w[cls_mask]
    X = dataset_.dataset_x[mask_bin_cls, :]

    if grouping == "groups":
        groups = dataset_.groups[mask_bin_cls]
    elif grouping == "none":
        groups = np.arange(y.size)
    elif grouping == "regions":
        assert dataset_.regions is not None
        groups = dataset_.regions[mask_bin_cls]
    else:
        raise ValueError(f"unexpected value for {grouping=!r}")

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
    grouping: Literal["none", "groups", "regions"],
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
    groups: np.ndarray | None = dataset_.groups[cls_mask]

    match grouping:
        case "none":
            groups = None
        case "groups" | "regions":
            if isinstance(cv, int) or cv is None:
                raise ValueError(f"{grouping=} will be ignored for {cv=}. Use a sklearn GroupCV object.")
        case _:
            raise ValueError(f"{grouping=!r} is not supported")

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
    grouping: Literal["groups", "regions"],
    weighting: Literal[False],
):
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
    match grouping:
        case "groups":
            g = dataset_.groups[cls_mask]
        case "regions":
            g = dataset_.regions[cls_mask]
        case _:
            raise ValueError(f"{grouping=!r} is invalid")

    y_pred = cross_val_predict(
        model,
        X,
        y,
        groups=g,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
    )

    return roc_auc_score(y, y_pred)


def cv_logo_detailed_acc(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["groups", "regions"],
    weighting: Literal[False],
):
    """cv_logo_detailed_acc: share some details about grouped cross validation

    This function returns the training and validation accuracy on each of the folds, as well as the indices in of them.
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
    match grouping:
        case "groups":
            g = dataset_.groups[cls_mask]
        case "regions":
            g = dataset_.regions[cls_mask]
        case _:
            raise ValueError(f"{grouping=!r} is invalid")

    infos = {
        "X": X,
        "y": y,
        "g": g,
        "groups": dataset_.groups[cls_mask],
        "regions": dataset_.regions[cls_mask],
    }

    return cross_validate(
        model,
        X,
        y,
        groups=g,
        scoring="accuracy",
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
        return_train_score=True,
        return_estimator=False,
        return_indices=True,
    ), infos


def cv_score_by_region(
    model: BaseEstimator,
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc"],
    weighting: Literal[False],
    scorer_: _BaseScorer,
):
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
    regions = dataset_.regions[cls_mask]
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
    regions = dataset_.regions[cls_mask]
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
    fdr: float = 5e-2,
):
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

    y = np.argmax(dataset_.dataset_y, axis=1)

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(y, neg_labels + pos_labels)

    y = np.where(np.isin(y[cls_mask], pos_labels), 1, 0)
    X = dataset_.dataset_x[cls_mask, :]
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

    # find highest item such that P[k] <= (k + 1) * fdr / m
    check = sorted_p_values <= (1 + np.arange(m)) * fdr / m
    max_idx = m - 1 - np.argmax(check[::-1])
    if not check.any():
        max_idx = 0

    # selected features
    valid_idx = s_idx[:max_idx]
    bin_c = 0.5 * (dataset_.bin_l + dataset_.bin_r)

    return bin_c[valid_idx], p_values[valid_idx]

# %%

def _ds_to_Xy(
    dataset_: Tabular,
    problem: Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"],
    grouping: Literal["none", "groups", "regions"],
    weighting: bool,
):

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

    flat_w = dataset_.dataset_y.flatten()  # [N * L]
    re_idx = (
        np.expand_dims(np.arange(dataset_.dataset_y.shape[0]), axis=1)
        .repeat(dataset_.dataset_y.shape[1], axis=1)
        .flatten()
    )
    re_cls = (
        np.expand_dims(np.arange(dataset_.dataset_y.shape[1]), axis=0)
        .repeat(dataset_.dataset_y.shape[0], axis=0)
        .flatten()
    )

    # only rows which have the right class and a positive weight
    assert not set(neg_labels).intersection(pos_labels)
    cls_mask = np.isin(re_cls, neg_labels + pos_labels) & (flat_w > 0)
    mask_bin_cls = re_idx[cls_mask]

    y = np.where(np.isin(re_cls[cls_mask], pos_labels), 1, 0)
    w = flat_w[cls_mask]
    X = dataset_.dataset_x[mask_bin_cls, :]

    if grouping == "groups":
        groups = dataset_.groups[mask_bin_cls]
    elif grouping == "none":
        groups = np.arange(y.size)
    elif grouping == "regions":
        assert dataset_.regions is not None
        groups = dataset_.regions[mask_bin_cls]
    else:
        raise ValueError(f"unexpected value for {grouping=!r}")

    if not weighting:
        w.fill(1.0)

    return X, y, w, groups


# %%

base_dir = Path("/home/maxime/datasets/slim-deisotoping-2.0e-03-binned")
assert base_dir.is_dir()

# %%

# problems: tuple[Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"], ...] = (
#     "ls+/ls-",
#     "sc+/sc-",
#     "ls/sc",
# )
# norms = ["317", "no"]

# for prob_, norm_ in product(problems, norms):
#     _, scores = fit_and_eval(
#         [
#             (
#                 RandomForestClassifier(random_state=0, n_jobs=-1),
#                 {
#                     "n_estimators": [100],
#                     "max_depth": [20],
#                     "max_features": ["sqrt"],
#                 },
#             ),
#         ],
#         dataset_=Tabular(
#             *np.load(
#                 base_dir / f"binned_{norm_}norm.npz"
#             ).values()
#         ),
#         problem=prob_,
#         grouping="regions",
#         weighting=True,
#         scorer_=roc_auc_scorer,
#         n_splits=3,
#     )

#     mean_: float = scores.mean()
#     std_: float = scores.std(ddof=1.0)
#     lo_ = mean_ - std_
#     print(f"{prob_=} {norm_=} {mean_=:.3f} {std_=:.3f} {lo_=:.3f}")


bin_method: Literal["sum", "integration"] = "integration"
dataset = Tabular(*np.load(base_dir / f"binned_{bin_method}_{norm}norm.npz").values())
# prob_='ls+/ls-' norm_='317' mean_=0.512 std_=0.049 lo_=0.463
# prob_='ls+/ls-' norm_='no' mean_=0.474 std_=0.027 lo_=0.447
# prob_='sc+/sc-' norm_='317' mean_=0.635 std_=0.031 lo_=0.604
# prob_='sc+/sc-' norm_='no' mean_=0.586 std_=0.083 lo_=0.503
# prob_='ls/sc' norm_='317' mean_=0.793 std_=0.047 lo_=0.746
# prob_='ls/sc' norm_='no' mean_=0.847 std_=0.079 lo_=0.768
# """

# %%

norm: Literal["no", "317"] = "317"
bin_method: Literal["sum", "integration"] = "integration"
dataset = Tabular(*np.load(base_dir / f"binned_{bin_method}_{norm}norm.npz").values())
# model, _ = fit_and_eval(
#     [
#         (
#             RandomForestClassifier(random_state=0, n_jobs=-1),
#             {
#                 "n_estimators": [10],
#                 # "max_depth": [20],
#                 # "max_features": ["sqrt"],
#             },
#         ),
#     ],
#     dataset,
#     "ls/sc",
#     grouping="regions",
#     weighting=True,
#     scorer_=roc_auc_scorer,
#     n_splits=3
# )

# # %%

# assert isinstance(model, RandomForestClassifier)
# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# df = pd.DataFrame({
#     "bin_center": 0.5 * (dataset.bin_l + dataset.bin_r),
#     "bin_width": (dataset.bin_r - dataset.bin_l),
#     "importances": importances,
#     "std": std
# })

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
        grouping="groups",
        weighting=False,
    )

    with open(cfg_dir / "score.txt", "w", encoding="utf8") as cfg_score_f:
        print(str(score), file=cfg_score_f)

# %%


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
problems = [
    "ls/sc",
    "ls+/ls-",
    "sc+/sc-",
]
bin_methods = [
    "sum",
    # "integration",
]

configs = [Configuration(*t_c, p_c, b_c) for t_c, p_c, b_c in product(tree_configs, problems, bin_methods)]

res_dir = Path(__file__).parent.parent / "res-exp-cv-untargeted-scores"
res_dir.mkdir(exist_ok=True)

exp_dir = res_dir / datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
exp_dir.mkdir()

for idx, cfg in enumerate(configs):
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
            max_features=cfg.max_feat,  # type:ignore -> doc issue in sklearn, None is actually fine
            max_depth=cfg.max_depth,
            random_state=0,
            n_jobs=-2,
        ),
        dataset_=ds,
        problem=cfg.problem,
        grouping="groups",
        weighting=False,
    )

    with open(cfg_dir / "score.txt", "w", encoding="utf8") as cfg_score_f:
        print(str(score), file=cfg_score_f)

# %%

max_feat: int | Literal["log2", "sqrt"]
problem: Literal["ls/sc", "ls+/ls-", "sc+/sc-", "+/-"]
try:
    # detect if the cell is ran in a notebook and manually defines values
    get_ipython()  # type:ignore
    n_trees = 1000
    max_feat = "sqrt"
    n_perms = 50
    joint = True
    problem = "ls/sc"
except NameError:
    # else, assume a script and parse arguments
    parser = argparse.ArgumentParser("exp_cv_binned")
    parser.add_argument("--n-trees", type=int)
    parser.add_argument("--max-feat", help="'log2', 'none', 'sqrt', or an integer")
    parser.add_argument("--n-perms", type=int)
    parser.add_argument("--joint", choices=["yes", "no"])
    parser.add_argument("--problem", choices=["ls/sc", "ls+/ls-", "sc+/sc-"])

    args = parser.parse_args()
    n_trees = int(args.n_trees)
    if args.max_feat in ["log2", "sqrt"]:
        max_feat = args.max_feat  # type:ignore
    elif args.max_feat == "none":
        # all features
        max_feat = dataset.dataset_x.shape[1]
    else:
        max_feat = int(args.max_feat)
    n_perms = int(args.n_perms)
    joint = {"yes": True, "no": False}[args.joint]
    problem = args.problem

# %%

X, y, w, g = _ds_to_Xy(dataset, problem, "regions", True)

# %%

res_dir = Path(__file__).parent.parent / "res-exp-cv-untargeted"
res_dir.mkdir(exist_ok=True)

exp_dir = res_dir / datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
exp_dir.mkdir()
# TODO this sucks. create a new directory inside res_dir with a datetime
#   add all meaningfull info inside a log file
#   add the importance into a simply-named file

with open(exp_dir / "info.csv", "w", encoding="utf8") as exp_info_f:
    print("norm,problem,n_trees,max_feat,n_perms,joint", file=exp_info_f)
    print(f"{norm},{problem},{n_trees},{max_feat},{n_perms},{joint}", file=exp_info_f)

# %%

results = mprobes(
    X,
    y,
    model=RandomForestClassifier(n_trees, max_features=max_feat, n_jobs=-1),
    nb_perm=n_perms,
    join_permute=joint,
    verbose=False,
)
results["bin_center"] = 0.5 * (dataset.bin_l + dataset.bin_r)
results["bin_width"] = dataset.bin_r - dataset.bin_l

# %%

# results = cer_fdr(
#     results,
#     X,
#     y,
#     model=RandomForestClassifier(n_trees, max_features=max_feat, n_jobs=-1),
#     nb_perm=n_perms,
#     n_to_compute="all",
#     thresh_stop=0.4,
# )

# %%

results.to_csv(exp_dir / "results.csv")
