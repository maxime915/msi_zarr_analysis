# %%

from itertools import product
from pathlib import Path
from typing import NamedTuple, Literal, Protocol, Any
from warnings import warn

import numpy as np
import torch.nn as nn
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics._scorer import (
    roc_auc_scorer,
    _BaseScorer,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.tree import (  # noqa:F401
    DecisionTreeClassifier,
)
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from torch.nn.functional import sigmoid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


# %% protocols and stuff


class FitPred(Protocol):
    "A (restrictive) protocol for most useful classifiers in scikit-learn"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "FitPred": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def set_params(self, **kwargs) -> "FitPred": ...


# %%

INV_SQRT_2_PI = 0.3989422804014326779


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


class IntensitiesWS(nn.Module):
    "computes a weighted sum of intensities based on the masses"

    def __init__(
        self,
        n_vals: int,
        mz_min: float,
        mz_max: float,
        std_dev: float | None = None,
    ) -> None:
        super().__init__()
        if std_dev is None:
            std_dev = 0.5 * (mz_max - mz_min) / n_vals
        self.n_vals = n_vals
        self.mu = nn.Parameter(
            torch.linspace(mz_min, mz_max, n_vals, dtype=torch.float32),
        )
        self.lv = nn.Parameter(
            torch.zeros_like(self.mu) + 2.0 * torch.log(torch.tensor(std_dev))
        )

    def forward(self, masses: torch.Tensor, intensities: torch.Tensor):
        mz_ = masses.unsqueeze(-2)  # [B?, 1, N]
        int_ = intensities.unsqueeze(-2)  # [B?, 1, N]
        mu_ = self.mu.view(1, self.mu.shape[0], 1)  # [1, n_vals, 1]
        lv_ = self.lv.view(1, self.lv.shape[0], 1)
        if mz_.ndim == 2:
            mu_ = mu_.squeeze(0)  # [n_vals, 1]
            lv_ = lv_.squeeze(0)

        diff2 = torch.square(mu_ - mz_)  # [B?, n_vals, N]
        sigma2i = torch.exp(-lv_)  # [B?, n_vals, 1]
        exp = torch.exp(-0.5 * diff2 * sigma2i)  # [B?, n_vals, N]

        # do un-normalized weighted sum (to avoid useless multiplication)
        ws_ = torch.sum(exp * int_, -1)  # [B?, n_vals]
        # NOTE where the values are clipped should be revisited
        ws_ = torch.clamp(ws_, min=1e-4)  # avoid propagating useless gradients

        # normalize after the sum reduced most of the inputs
        std_dev_i = torch.exp(-0.5 * lv_.squeeze(-1))  # [B?, n_vals]
        norm_ws = INV_SQRT_2_PI * (std_dev_i * ws_)  # [B?, n_vals]

        return self.mu.detach(), norm_ws

    def self_similarity(self):
        # every possible combination, as pairs of vectors of shape (N*(N-1)/2,)
        lv_l, lv_r = torch.chunk(torch.combinations(self.lv, r=2), 2, dim=1)
        mu_l, mu_r = torch.chunk(torch.combinations(self.mu, r=2), 2, dim=1)

        # 2 * s^2_1 * s^2_2
        prod_s2 = 2 * torch.exp(lv_l + lv_r)

        # (m_1 - m_2)^2
        diff2_m = torch.square(mu_l - mu_r)

        # s^2_1 + s^2_2
        sum_s2 = torch.exp(lv_l) + torch.exp(lv_r)

        # s^4_1 + s^4_2
        sum_s4 = torch.exp(2.0 * lv_l) + torch.exp(2.0 * lv_r)

        # 1 / (1 + JSD)
        sim = prod_s2 / (sum_s4 + diff2_m * sum_s2)
        return sim.mean()


class ClsSingle(nn.Module):
    def __init__(self, n_masses: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones((n_masses,), dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((n_masses,), dtype=dtype))

    def forward(self, m_ws: tuple[torch.Tensor, torch.Tensor]):
        return m_ws[1] * self.weights + self.bias


class ClsMultiple(nn.Module):
    def __init__(self, n_masses: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.ln = nn.Linear(n_masses, n_masses, True, dtype=dtype)

    def forward(self, m_ws: tuple[torch.Tensor, torch.Tensor]):
        return self.ln(m_ws[1])


"""# What do I want ?

## Using only 1 mass (multiple heads to train them in parallel)

* Looking at a single mass seems closer to the research question (according to me)
* Prediction (per head): linear function of a single intensity
* Training: train each head on the same label (individually)
* Prediction (evaluation): predict_proba as an average of all masses

## Using multiple masses
That's more likely to have a higher accuracy
"""

CSV_MZS = torch.tensor(
    [
        205.19508,
        243.26824,
        285.27881,
        305.24751,
        317.21112,
        321.24242,
        335.22168,
        337.23734,
        353.23225,
        355.2479,
        361.23734,
        371.24282,
        377.23226,
        380.25603,
        496.33976,
        524.37107,
        546.35541,
        550.35033,
        552.40237,
        594.37654,
        596.33469,
        610.37146,
        650.43915,
        664.41841,
        666.43406,
        734.56943,
        758.56943,
        782.56943,
        786.60073,
        798.56435,
        806.56943,
        810.52796,
        810.60073,
        812.54361,
        814.55926,
        828.53852,
        832.56983,
    ],
    dtype=torch.float32,
)


class ModelTrainer(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_masses: int,
        mass_min: float,
        mass_max: float,
        bin_over_m2_c: float,
        reg_bin_size: float,
        reg_lasso: float,
        feature_mode: Literal["single", "multiple"],
        optimizer: Literal["adam"],
        # https://scikit-learn.org/1.4/developers/develop.html#optional-arguments
        n_iter: int,  # n_iter instead of n_epochs, see above link
        batch_size: int,
        lr: float,
        device: Literal["cpu", "cuda:0"],
        random_state: int | None = None,
    ) -> None:
        self.n_masses = n_masses
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.bin_over_m2_c = bin_over_m2_c
        self.reg_bin_size = reg_bin_size
        self.reg_lasso = reg_lasso
        self.feature_mode = feature_mode
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state

    def check_args(self):
        if not isinstance(self.n_masses, int) or self.n_masses <= 0:
            raise ValueError(f"{self.n_masses=!r} should be a positive integer")
        if not isinstance(self.mass_min, float) or self.mass_min <= 0.0:
            raise ValueError(f"{self.mass_min=!r} should be a positive float")
        if not isinstance(self.mass_max, float) or self.mass_max <= 0.0:
            raise ValueError(f"{self.mass_max=!r} should be a positive float")
        if self.mass_min >= self.mass_max:
            raise ValueError(f"{self.mass_min=!r} >= {self.mass_max}")
        if self.feature_mode not in ["single", "multiple"]:
            raise ValueError(f"invalid: {self.feature_mode=!r}")
        if self.optimizer not in ["adam"]:
            raise ValueError(f"invalid: {self.optimizer}")
        if not isinstance(self.n_iter, int) or self.n_iter <= 0:
            raise ValueError(f"{self.n_iter=!r} should be a positive integer")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"{self.batch_size=!r} should be a positive integer")
        if not isinstance(self.lr, float) or self.lr <= 0.0:
            raise ValueError(f"{self.lr} should be a positive float")

    @staticmethod
    def split_features(X: np.ndarray):
        # TODO in the future, the bottom implementation should be used.
        # for now, we assume that we know the mz values
        ints = torch.from_numpy(X)
        mzs = CSV_MZS.clone().to(ints.dtype)
        mzs = torch.unsqueeze(mzs, 0).repeat((len(X), 1))
        return mzs, torch.log1p(ints)

        mzs, ints = torch.chunk(torch.from_numpy(X), 2, dim=1)
        return mzs, torch.log1p(ints)

    @staticmethod
    def _forward(
        ws: IntensitiesWS, head: nn.Module, mzs: torch.Tensor, ints: torch.Tensor
    ):
        return sigmoid(head(ws(mzs, ints)))

    @staticmethod
    def _train(
        ws: IntensitiesWS,
        head: nn.Module,
        bin_size_c: float,
        bin_size_reg: float,
        lasso_reg: float,
        dataset: TensorDataset,
        optim: torch.optim.Optimizer,
        n_epochs: int,
        batch_size: int,
    ):
        # is the seed worker necessary ? not sure, but it doesn't hurt
        dl = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=seed_worker)

        for _ in range(n_epochs):
            for mzs, ints, lbl, weights in dl:
                optim.zero_grad(True)

                pred_prob = ModelTrainer._forward(ws, head, mzs, ints)

                # pred_prob is [B, N] and lbl is [B] -> we need to repeat lbl
                # because N is the number of "predictors" rather than the number of classes
                loss = nn.functional.binary_cross_entropy(
                    pred_prob,
                    torch.unsqueeze(lbl, -1).expand(-1, pred_prob.shape[1]),
                    reduction="none",
                )
                loss = (loss * torch.unsqueeze(weights, -1)).mean()
                loss += lasso_reg * ws.self_similarity()
                # FIXME numerically not stable at all
                target_bin_size = bin_size_c * ws.mu.detach() ** 2
                bin_size_diff = target_bin_size - 4 * torch.exp(0.5 * ws.lv)
                loss += bin_size_reg * (bin_size_diff).abs().mean()

                loss.backward()
                optim.step()

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        self.check_args()
        X, y = check_X_y(X, y)
        random_state = check_random_state(self.random_state)
        self.classes_ = unique_labels(y)

        torch.manual_seed(random_state.randint(2**32))

        # build module
        self.ws_ = IntensitiesWS(self.n_masses, self.mass_min, self.mass_max, None)
        if self.feature_mode == "single":
            self.head_: nn.Module = ClsSingle(self.n_masses, torch.float32)
        elif self.feature_mode == "multiple":
            self.head_ = ClsMultiple(self.n_masses, dtype=torch.float32)
        else:
            raise ValueError(self.feature_mode)

        self.ws_ = self.ws_.to(self.device)
        self.head_ = self.head_.to(self.device)

        if self.optimizer == "adam":

            def params():
                yield from self.ws_.parameters()
                yield from self.head_.parameters()

            optimizer = torch.optim.Adam(params(), lr=self.lr)
        else:
            raise ValueError(self.optimizer)

        # launch training
        dataset = TensorDataset(
            *ModelTrainer.split_features(X),
            torch.from_numpy(y).to(torch.float32),
            torch.from_numpy(sample_weight),
        )
        dataset.tensors = tuple(t.to(self.device) for t in dataset.tensors)
        ModelTrainer._train(
            self.ws_,
            self.head_,
            self.bin_over_m2_c,
            self.reg_bin_size,
            self.reg_lasso,
            dataset,
            optimizer,
            self.n_iter,
            self.batch_size,
        )

        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray):
        check_is_fitted(self, ("ws_", "head_"))

        mzs, ints = ModelTrainer.split_features(X)
        # average over heads
        pred_prob = torch.mean(
            ModelTrainer._forward(self.ws_, self.head_, mzs, ints), dim=1
        ).numpy()
        return np.stack([1.0 - pred_prob, pred_prob], axis=1)

    def predict(self, X: np.ndarray):
        y_prob = self.predict_proba(X)
        return y_prob.argmax(axis=1)


# %%


class Tabular(NamedTuple):
    "Afterthought: that could have been a pandas dataframe"

    dataset_x: np.ndarray
    dataset_y: np.ndarray
    groups: np.ndarray
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
        search = GridSearchCV(
            model_, a_dct_, scoring=scorer_, cv=folds, n_jobs=-1, error_score="raise"
        )
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
    # dataset_.dataset_y[re_idx[k], re_cls[k]] == flat_w[k] for all k
    re_idx = (  # dataset row for each flat_w
        np.expand_dims(np.arange(dataset_.dataset_y.shape[0]), axis=1)
        .repeat(dataset_.dataset_y.shape[1], axis=1)
        .flatten()
    )
    re_cls = (  # dataset class index for each flat_w
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
    X, y, w, groups = _ds_to_Xy(dataset_, problem, grouping, weighting)

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

file_idx = 0

files = [
    "saved_msi_merged_nonslim_317norm.npz",
    "saved_msi_merged_nonslim_nonorm.npz",
    "saved_msi_merged_slim_317norm.npz",
    "saved_msi_merged_slim_nonorm.npz)",
]


base_dir = Path(__file__).parent.parent / "datasets"


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
    _, scores = fit_and_eval(
        [
            (
                DecisionTreeClassifier(random_state=32),
                {"max_depth": [1, 20, None]},
            ),
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
        (DecisionTreeClassifier(random_state=32), {"max_depth": [1]}),
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

custom_model = ModelTrainer(
    n_masses=20,
    mass_min=300.0,
    mass_max=1000.0,
    bin_over_m2_c=1.9e-9,  # experimentally found for this dataset
    reg_bin_size=0.0,
    reg_lasso=0.0,
    feature_mode="single",
    optimizer="adam",
    n_iter=50,
    batch_size=128,
    lr=1e-3,
    device="cpu",
    random_state=0,
)

model, scores = fit_and_eval(
    [
        # (DecisionTreeClassifier(random_state=32), {"max_depth": [1]}),
        (custom_model, {"n_iter": [50], "lr": [1e-2], "reg_lasso": [0.1]})
    ],
    dataset_=Tabular(*np.load(base_dir / "saved_msi_merged_slim_317norm.npz").values()),
    problem="ls/sc",
    grouping="regions",
    weighting=True,
    scorer_=roc_auc_scorer,
    n_splits=3,
)

print(scores.mean(), list(scores))
print(type(model), model.get_params())

# %%

assert isinstance(model, ModelTrainer)
print(model.ws_.mu)
print(torch.exp(0.5 * model.ws_.lv))
# how close is each value to the closest one in the CSV file ?
print((model.ws_.mu.unsqueeze(1) - CSV_MZS.unsqueeze(0)).abs().min(1)[0])


# %%
