# %%

from pathlib import Path
from typing import NamedTuple, Literal

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import (
    roc_auc_scorer,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.tree import ( # noqa:F401
    DecisionTreeClassifier,
)


# %%


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    dataset_y: np.ndarray
    groups: np.ndarray
    regions: np.ndarray


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

backup = BaseEstimator.__getattribute__


# %%


def wrapped(self: BaseEstimator, _name: str):
    print(f"{type(self)}(<{id(self)}>).{_name}")
    value = backup(self, _name)
    return value


BaseEstimator.__getattribute__ = wrapped

# %%


model_ = DecisionTreeClassifier(random_state=0)
a_dct_ = {"max_depth": [1]}
X_, y_, w_, g_ = _ds_to_Xy(
    Tabular(*np.load(Path(__file__).parent.parent / "datasets" / "saved_msi_merged_slim_317norm.npz").values()),
    "ls/sc",
    "regions",
    True,
)
scorer_ = roc_auc_scorer

# model_.fit(X_, y_, sample_weight=w_)
split_ = StratifiedGroupKFold(3, shuffle=True, random_state=32)
folds = list(split_.split(X_, y_, g_))

search = GridSearchCV(model_, a_dct_, scoring=scorer_, cv=folds)
search.fit(X_, y_, sample_weight=w_)  # w_ is split automatically

# %%
