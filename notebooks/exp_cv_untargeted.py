# %%

import argparse
from itertools import product  # noqa
from pathlib import Path
from typing import NamedTuple, Literal, Protocol, Any
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier  # noqa:F401
from sklearn.metrics._scorer import roc_auc_scorer, _BaseScorer   # noqa
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV


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


# """
# prob_='ls+/ls-' norm_='317' mean_=0.512 std_=0.049 lo_=0.463
# prob_='ls+/ls-' norm_='no' mean_=0.474 std_=0.027 lo_=0.447
# prob_='sc+/sc-' norm_='317' mean_=0.635 std_=0.031 lo_=0.604
# prob_='sc+/sc-' norm_='no' mean_=0.586 std_=0.083 lo_=0.503
# prob_='ls/sc' norm_='317' mean_=0.793 std_=0.047 lo_=0.746
# prob_='ls/sc' norm_='no' mean_=0.847 std_=0.079 lo_=0.768
# """

# %%

dataset = Tabular(*np.load(base_dir / "binned_nonorm.npz").values())
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


# TODO grouping and weights ?
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
        print('Compute the initial ranking...')

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
        raise ValueError('X and Y must have the same number of objects.')

    # data
    X_full = np.zeros((nb_obj, nb_feat * 2), dtype=np.float32)
    X_full[:, :nb_feat] = X  # removed useless copy

    p = np.zeros(nb_feat)

    if verbose:
        print('Compute the permutation rankings...')

    for _ in range(nb_perm):
        if verbose:
            print('.', end='')
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
        contrast_imp_max = max(var_imp[nb_feat:nb_feat * 2])

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

parser = argparse.ArgumentParser("exp_cv_binned")
parser.add_argument("--n-trees", type=int)
parser.add_argument("--max-feat", type=int)
parser.add_argument("--n-perms", type=int)
parser.add_argument("--joint", choices=["yes", "no"])

args = parser.parse_args()
n_trees = int(args.n_trees)
max_feat = int(args.max_feat)
n_perms = int(args.n_perms)
joint = {"yes": True, "no": False}[args.joint]

X, y, w, g = _ds_to_Xy(dataset, "ls/sc", "regions", True)
results = mprobes(
    X,
    y,
    model=RandomForestClassifier(n_trees, max_features=5, n_jobs=-1),
    nb_perm=n_perms,
    join_permute=True,
    verbose=True,
)
results["bin_center"] = 0.5 * (dataset.bin_l + dataset.bin_r)
results["bin_width"] = (dataset.bin_r - dataset.bin_l)

print(results.to_csv())

# %%
