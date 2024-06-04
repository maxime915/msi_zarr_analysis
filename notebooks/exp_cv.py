# %%

from itertools import product
from pathlib import Path
from typing import NamedTuple, Literal, Protocol, Any
from warnings import warn

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier  # noqa:F401
from sklearn.ensemble import (  # noqa:F401
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.metrics._scorer import (
    roc_auc_scorer,
    accuracy_scorer,
    _BaseScorer,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.tree import (  # noqa:F401
    DecisionTreeClassifier,
)


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
    regions: np.ndarray


# %%

base_dir = Path(__file__).parent.parent / "datasets"

# %%


# def selection_hpt(
#     models: list[tuple[BaseEstimator, dict[str, Any]]],
#     X_: np.ndarray,
#     y_: np.ndarray,
#     e_w_: np.ndarray,
#     scorer_: _BaseScorer,
#     n_splits: int,
#     tune_groups: tuple[np.ndarray, ...],
#     tune_weights: tuple[np.ndarray, ...],
# ):
#     "model, scorer, groups, weights"

#     # todo does it make sense to compute different weighting metrics ?
#     # The more I think about it, the less I'm convinced it does...

#     best_: tuple[BaseEstimator, tuple[np.ndarray, np.ndarray], float] | None = None
#     hp_iter = product(tune_groups, tune_weights)

#     for hp_g_, hp_w_ in hp_iter:
#         n_groups = len(np.unique(hp_g_))
#         if n_groups < n_splits:
#             warn(f"{n_splits=} is decreased to {n_groups=}")
#             n_splits = n_groups
#         split_ = StratifiedGroupKFold(n_splits, shuffle=True, random_state=32)
#         folds = list(split_.split(X_, y_, hp_g_))

#         for model_, a_dct_ in models:
#             search = GridSearchCV(model_, a_dct_, scoring=scorer_, cv=folds, n_jobs=-1)
#             search.fit(X_, y_, sample_weight=hp_w_)

#             # todo redo evaluation using e_w_, e_scorer_
#             score = search.best_score_

#             if np.isnan(search.best_score_):
#                 raise RuntimeError(f"encountered NaN score for {model_=}, {a_dct_=}")

#             if best_ is None or best_[2] < score:
#                 best_ = (search.best_estimator_, (hp_g_, hp_w_), score)

#     if best_ is None:
#         raise RuntimeError("no model yielded a non-NaN score")

#     return best_[:1] + best_[1]


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

    # y: np.ndarray = dataset_.dataset_y.argmax(axis=1)
    # w: np.ndarray = dataset_.dataset_y.max(axis=1)
    # assert not set(neg_labels).intersection(pos_labels)
    # mask_bin_cls = np.isin(y, neg_labels + pos_labels)

    # X = dataset_.dataset_x[mask_bin_cls, :]
    # w = w[mask_bin_cls]
    # y = np.where(np.isin(y[mask_bin_cls], pos_labels), 1, 0)

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


# def cv_score(
#     model: FitPred,
#     dataset_: Tabular,
#     metric: Literal["acc", "roc_auc"],
#     neg_label: int,
#     pos_label: int,
#     grouping: Literal["none", "groups", "regions"],
#     weighting: bool,
#     n_splits: int,
# ):
#     y: np.ndarray = dataset_.dataset_y.argmax(axis=1)
#     w: np.ndarray = dataset_.dataset_y.max(axis=1)
#     selection: np.ndarray = (y == neg_label) | (y == pos_label)

#     X = dataset_.dataset_x[selection, :]
#     w = w[selection]
#     y = np.where(y[selection] == neg_label, 0, 1)

#     if grouping == "groups":
#         groups = dataset_.groups[selection]
#     elif grouping == "none":
#         groups = np.arange(y.size)
#     elif grouping == "regions":
#         assert dataset_.regions is not None
#         groups = dataset_.regions[selection]
#     else:
#         raise ValueError(f"unexpected value for {grouping=!r}")

#     if not weighting:
#         w.fill(1.0)

#     n_groups = len(np.unique(groups))
#     if n_groups < n_splits:
#         warn(f"{n_splits=} is decreased to {n_groups=}")
#         n_splits = n_groups

#     scores = np.empty((n_splits,), dtype=float)
#     cv_ = StratifiedGroupKFold(n_splits, shuffle=True, random_state=2)
#     for i, (train_, test_) in enumerate(cv_.split(X, y, groups)):
#         model.fit(X[train_], y[train_], w[train_])

#         if metric == "acc":
#             y_pred = model.predict(X[test_])
#             scores[i] = accuracy_score(y[test_], y_pred, sample_weight=w[test_])
#         elif metric == "roc_auc":
#             y_scores = model.predict_proba(X[test_])[:, 1]
#             scores[i] = roc_auc_score(y[test_], y_scores, sample_weight=w[test_])

#     return scores.mean(), list(scores)

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

file_idx = 0

files = [
    "saved_msi_merged_nonslim_317norm.npz",
    "saved_msi_merged_nonslim_nonorm.npz",
    "saved_msi_merged_slim_317norm.npz",
    "saved_msi_merged_slim_nonorm.npz)",
]

dataset = Tabular(*np.load(base_dir / files[file_idx]).values())


# %%

# cv_score(
#     ExtraTreesClassifier(
#         n_estimators=100,
#         max_depth=1,
#         random_state=34,
#         n_jobs=-1
#     ),
#     dataset,
#     "roc_auc",
#     0,
#     1,
#     "regions",
#     weighting=False,
#     n_splits=10,
# )

# %%

problems: tuple[Literal["ls+/ls-", "sc+/sc-", "ls/sc", "+/-"], ...] = (
    "ls+/ls-",
    "sc+/sc-",
    "ls/sc",
    "+/-",
)
norms = ["317", "no"]
slims = ["non", ""]

for prob_, norm_, slim_ in product(problems, norms, slims):
    break
    _, scores = fit_and_eval(
        [
            (
                ExtraTreesClassifier(random_state=0),
                {
                    "n_estimators": [200, 500, 1000],
                    "max_depth": [1, 20, None],
                    "max_features": ["sqrt", None],
                },
            ),
            (
                RandomForestClassifier(random_state=0),
                {
                    "n_estimators": [200, 500, 1000],
                    "max_depth": [1, 20, None],
                    "max_features": ["sqrt", None],
                },
            ),
            (DecisionTreeClassifier(random_state=32), {"max_depth": [1, 20, None]}),
            # (DecisionTreeClassifier(random_state=32), {"max_depth": [1]}),
        ],
        dataset_=Tabular(
            *np.load(
                base_dir / f"saved_msi_merged_{slim_}slim_{norm_}norm.npz"
            ).values()
        ),
        problem=prob_,
        grouping="regions",
        weighting=True,
        scorer_=roc_auc_scorer,
        n_splits=3,
    )

    mean_: float = scores.mean()
    std_: float = scores.std(ddof=1.0)
    lo_ = mean_ - std_
    print(f"{prob_=} {norm_=} {slim_=} {mean_=:.3f} {std_=:.3f} {lo_=:.3f}")

# %%

model, scores = fit_and_eval(
    [
        # (
        #     ExtraTreesClassifier(random_state=0),
        #     {
        #         "n_estimators": [200, 500, 1000],
        #         "max_depth": [1, 20, None],
        #         "max_features": ["sqrt", None],
        #     },
        # ),
        # (
        #     RandomForestClassifier(random_state=0),
        #     {
        #         "n_estimators": [200, 500, 1000],
        #         "max_depth": [1, 20, None],
        #         "max_features": ["sqrt", None],
        #     },
        # ),
        # (DecisionTreeClassifier(random_state=32), {"max_depth": [1, 5, 20]}),
        (DummyClassifier(), {}),
        # (DecisionTreeClassifier(random_state=32), {"max_depth": [1]}),
    ],
    dataset_=Tabular(*np.load(base_dir / "saved_msi_merged_slim_317norm.npz").values()),
    problem="ls+/ls-",
    grouping="regions",
    weighting=True,
    scorer_=roc_auc_scorer,
    n_splits=3,
)

print(scores.mean(), list(scores))
print(type(model), model.get_params())

# %%

print(type(model), model.get_params())

# %%

"""
Does it make sense to tune hyperparameters of the selection procedure ?

Is it possible to do assessment, model selection, and hyperparameters tuning
with only two folds ?

- Because the score has a different meaning in each case (weighted or not,
  CV grouping, metric) it doesn't make much sense to compare the different
  scores across these parameters. Another CV loop would allow to select the best
  hyper-parameters that maximize the assessment procedure (with a fixed score)
  and another one would assess to whole stuff. Can we do three ? I don't know.
- If we need at least three layers of CV, region-grouping is infeasible and that
  defeats the whole purpose
- We choose to fix: grouping=region, weighted=True, score=roc_auc


SELECTION

* repeat GridSearchCV for all hyper-parameters (weighting, scorer, groups) available, in additions to all models in the dict
* compute the score of GridSearchCV.best_model_ using fixed (weighting=True, scorer=roc_auc) without retraining GridSearchCV.best_model_
* report the highest score using fixed (weighting=True, scorer=roc_auc) and associated models + configuration (weighting, groups, scorer)


+ we would have a better model
- there is only 3 regions, so at most 2 levels of CV are possible
* comparing scorer ?
    * use the hyperparameter scorer to select each configuration of each model
    * use a fixed scorer to re-evaluate the selected model on the same fold

"""

# %%

lines = """
prob_='ls+/ls-' norm_='317' slim_='non' scores.mean()=0.49460130378521266 std=0.02938302799823606
prob_='ls+/ls-' norm_='317' slim_='' scores.mean()=0.5634438698708896 std=0.016484448736343076

prob_='ls+/ls-' norm_='no' slim_='non' scores.mean()=0.5245225012114204 std=0.049998858203845906
prob_='ls+/ls-' norm_='no' slim_='' scores.mean()=0.46917676527733326 std=0.04808173043647598


prob_='sc+/sc-' norm_='317' slim_='non' scores.mean()=0.5027275793037028 std=0.013707082996208375
prob_='sc+/sc-' norm_='317' slim_='' scores.mean()=0.5287171458008307 std=0.10543474131705928

prob_='sc+/sc-' norm_='no' slim_='non' scores.mean()=0.5169422849909094 std=0.0760104227506919
prob_='sc+/sc-' norm_='no' slim_='' scores.mean()=0.5007328437929215 std=0.08030827600931234


prob_='ls/sc' norm_='317' slim_='non' scores.mean()=0.7981595195952685 std=0.17510966303768097
prob_='ls/sc' norm_='317' slim_='' scores.mean()=0.6430177215311402 std=0.14756859255099603

prob_='ls/sc' norm_='no' slim_='non' scores.mean()=0.8274687524026292 std=0.14137631141507137
prob_='ls/sc' norm_='no' slim_='' scores.mean()=0.6804147850006267 std=0.1513472989454603


prob_='+/-' norm_='317' slim_='non' scores.mean()=0.6017816804471746 std=0.08960412659784388
prob_='+/-' norm_='317' slim_='' scores.mean()=0.5712706910612213 std=0.09225980757622551

prob_='+/-' norm_='no' slim_='non' scores.mean()=0.5429243196978026 std=0.07852311495073701
prob_='+/-' norm_='no' slim_='' scores.mean()=0.5613556041111077 std=0.08235893340081994
"""

for line_ in lines.split("\n"):
    pattern_score_mean = "scores.mean()="
    idx = line_.find(pattern_score_mean)
    if idx == -1:
        print()
        continue
    idx += len(pattern_score_mean)
    print(line_[:idx], end="")

    sep = " std="
    mean_, std_ = line_[idx:].split(sep)
    mean, std = float(mean_), float(std_)
    print(f"{mean:.3f} ± {std:.3f}")

# %%

# after correcting the weights
# (only the highest weight was considered for each pixel, which is incorrect if
# more than one weight achieves the maximal value, and suboptimal if more than
# one weight has a non-zero value)
"""
prob_='ls+/ls-' norm_='317' slim_='non' mean_=0.513 std_=0.014 lo_=0.498
prob_='ls+/ls-' norm_='no' slim_='non' mean_=0.540 std_=0.070 lo_=0.470
prob_='ls+/ls-' norm_='317' slim_='' mean_=0.565 std_=0.039 lo_=0.526
prob_='ls+/ls-' norm_='no' slim_='' mean_=0.466 std_=0.087 lo_=0.380

prob_='sc+/sc-' norm_='317' slim_='non' mean_=0.499 std_=0.019 lo_=0.480
prob_='sc+/sc-' norm_='no' slim_='non' mean_=0.524 std_=0.038 lo_=0.486
prob_='sc+/sc-' norm_='317' slim_='' mean_=0.496 std_=0.053 lo_=0.442
prob_='sc+/sc-' norm_='no' slim_='' mean_=0.505 std_=0.064 lo_=0.441

prob_='ls/sc' norm_='317' slim_='non' mean_=0.794 std_=0.162 lo_=0.632
prob_='ls/sc' norm_='no' slim_='non' mean_=0.843 std_=0.116 lo_=0.727
prob_='ls/sc' norm_='317' slim_='' mean_=0.662 std_=0.139 lo_=0.524
prob_='ls/sc' norm_='no' slim_='' mean_=0.755 std_=0.014 lo_=0.741

prob_='+/-' norm_='317' slim_='non' mean_=0.608 std_=0.073 lo_=0.535
prob_='+/-' norm_='no' slim_='non' mean_=0.546 std_=0.062 lo_=0.484
prob_='+/-' norm_='317' slim_='' mean_=0.568 std_=0.076 lo_=0.493
prob_='+/-' norm_='no' slim_='' mean_=0.578 std_=0.074 lo_=0.504
"""

# %%

# DTC(max_depth=1) to mimic the statistical tests

"""
prob_='ls+/ls-' norm_='317' slim_='non' mean_=0.506 std_=0.020 lo_=0.486
prob_='ls+/ls-' norm_='no' slim_='non' mean_=0.503 std_=0.005 lo_=0.498
prob_='ls+/ls-' norm_='317' slim_='' mean_=0.498 std_=0.002 lo_=0.495
prob_='ls+/ls-' norm_='no' slim_='' mean_=0.489 std_=0.032 lo_=0.457

prob_='sc+/sc-' norm_='317' slim_='non' mean_=0.494 std_=0.010 lo_=0.484
prob_='sc+/sc-' norm_='no' slim_='non' mean_=0.526 std_=0.036 lo_=0.491
prob_='sc+/sc-' norm_='317' slim_='' mean_=0.522 std_=0.099 lo_=0.423
prob_='sc+/sc-' norm_='no' slim_='' mean_=0.470 std_=0.024 lo_=0.447

prob_='ls/sc' norm_='317' slim_='non' mean_=0.678 std_=0.122 lo_=0.555
prob_='ls/sc' norm_='no' slim_='non' mean_=0.705 std_=0.098 lo_=0.608
prob_='ls/sc' norm_='317' slim_='' mean_=0.609 std_=0.078 lo_=0.530
prob_='ls/sc' norm_='no' slim_='' mean_=0.658 std_=0.028 lo_=0.630

prob_='+/-' norm_='317' slim_='non' mean_=0.506 std_=0.018 lo_=0.488
prob_='+/-' norm_='no' slim_='non' mean_=0.531 std_=0.045 lo_=0.486
prob_='+/-' norm_='317' slim_='' mean_=0.537 std_=0.072 lo_=0.465
prob_='+/-' norm_='no' slim_='' mean_=0.530 std_=0.054 lo_=0.476
"""

# %%

np.count_nonzero(
    Tabular(
        *np.load(base_dir / "saved_msi_merged_slim_317norm.npz").values()
    ).dataset_y.max(axis=1)
    == 1.0
)

# %%

# reproducing methods from the Master's Thesis

problems = ("ls+/ls-", "sc+/sc-", "ls/sc")
norms = ["317", "no"]
slims = ["non"]

for prob_, norm_, slim_ in product(problems, norms, slims):
    _, scores = fit_and_eval(
        [
            (
                ExtraTreesClassifier(random_state=0),
                {
                    "n_estimators": [1000],
                    "max_depth": [1, 20, None],
                    "max_features": ["sqrt", None],
                },
            ),
        ],
        dataset_=Tabular(
            *np.load(
                base_dir / f"saved_msi_merged_{slim_}slim_{norm_}norm.npz"
            ).values()
        ),
        problem=prob_,
        grouping="regions",
        weighting=True,
        scorer_=accuracy_scorer,
        n_splits=3,
    )

    mean_ = scores.mean()
    std_ = scores.std(ddof=1.0)
    lo_ = mean_ - std_
    print(f"{prob_=} {norm_=} {slim_=} {mean_=:.3f} {std_=:.3f} {lo_=:.3f}")

# %%

# How to assess if the weighting is useful ?

# 1. Split the dataset into _test_(w == 1.0) and _train_(w < 1.0 + maybe a few w == 1.0)
# 2. Select and train on _train_ *with* and *without* weighting
# 3. Evaluate both approaches on _test_

for prob_, norm_, slim_ in product(problems, norms, slims):
    X__, y__, w__, g__ = _ds_to_Xy(
        Tabular(
            *np.load(
                base_dir / f"saved_msi_merged_{slim_}slim_{norm_}norm.npz"
            ).values()
        ),
        prob_,
        "groups",
        True,
    )

    # split train/test
    X_te__ = X__[w__ == 1.0]
    y_te__ = y__[w__ == 1.0]

    X_tr__ = X__[(w__ < 1.0) & (w__ >= 0.5)]
    y_tr__ = y__[(w__ < 1.0) & (w__ >= 0.5)]
    w_tr__ = w__[(w__ < 1.0) & (w__ >= 0.5)]
    g_tr__ = g__[(w__ < 1.0) & (w__ >= 0.5)]

    # train
    model = selection(
        [
            (
                ExtraTreesClassifier(random_state=0),
                {
                    "n_estimators": [1000],
                    "max_depth": [1, 20, None],
                    "max_features": ["sqrt", None],
                },
            ),
        ],
        X_tr__,
        y_tr__,
        g_tr__,
        w_tr__,
        roc_auc_scorer,
        3,
    )

    score_ = roc_auc_scorer(model, X_te__, y_te__)
    print(f"{prob_=} {norm_=} {slim_=} {score_=:.3f}")

# %%

problems = (
    "ls+/ls-",
    "sc+/sc-",
    "ls/sc",
    "+/-",
)
norms = ["317", "no"]
slims = [""]

for prob_, norm_, slim_ in product(problems, norms, slims):
    _, scores = fit_and_eval(
        [
            # (
            #     ExtraTreesClassifier(random_state=0),
            #     {
            #         "n_estimators": [5000],
            #         "max_depth": [1, 20, None],
            #         "max_features": ["sqrt", None],
            #     },
            # ),
            # (DecisionTreeClassifier(random_state=32), {"max_depth": [1, 20, None]}),
            (DecisionTreeClassifier(random_state=32), {"max_depth": [1]}),
        ],
        dataset_=Tabular(
            *np.load(
                base_dir / f"saved_msi_merged_{slim_}slim_{norm_}norm_v2.npz"
            ).values()
        ),
        problem=prob_,
        grouping="regions",
        weighting=True,
        scorer_=roc_auc_scorer,
        n_splits=3,
    )

    mean_ = scores.mean()
    std_ = scores.std(ddof=1.0)
    lo_ = mean_ - std_
    print(f"{prob_=} {norm_=} {slim_=} {mean_=:.3f} {std_=:.3f} {lo_=:.3f}")

# %%
