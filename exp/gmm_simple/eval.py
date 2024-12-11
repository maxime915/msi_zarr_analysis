"""eval: evaluation CLI and utils

NB: most values here are Negative Log-Likelihood -> - log P(D | H)
"""

import argparse
import dataclasses
import pathlib
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from msi_zarr_analysis.ml.gmm import GMM1DCls
from msi_zarr_analysis.ml.msi_ds import (
    FlattenedDataset,
    MSIDataset,
    split_to_mass_groups,
)
from runexp import config_file

from config import PSConfig
from train import train_model, load_dataset, split_dataset


class ImportanceScorer(Protocol):
    def __call__(
        self, nll_n: torch.Tensor, nll_p: torch.Tensor, p_ratio: float
    ) -> torch.Tensor:
        """Compute the importance score based of the negative log-likelihood of the negative and positive hypotheses

        Args:
            nll_n (torch.Tensor): float32[N,] NLL assuming negative class { log P(x | n) }
            nll_p (torch.Tensor): float32[N,] NLL assuming positive class { log P(x | p) }
            p_ratio (float): prior for negative / prior for positive { P(n) / P(p) }

        Returns:
            torch.Tensor: float32[N,] importance score for all measured samples
        """
        ...


@dataclasses.dataclass
class Experiment:
    save_to: pathlib.Path
    config: PSConfig
    model: GMM1DCls
    device: torch.device
    mz_vals: torch.Tensor
    nll_n: torch.Tensor
    nll_p: torch.Tensor


class DiscardLogger:
    def log(self, values: dict[str, Any], commit: bool = False):
        pass

    def get_url(self) -> str | None:
        return None

    @property
    def name(self) -> str:
        return "DiscardLogger"


def get_dir():
    res_dir = pathlib.Path(__file__).parent / "res-eval"
    if not res_dir.is_dir():
        raise ValueError(f"{res_dir} not found")
    return res_dir


def prob_ratio(int_neg: torch.Tensor, int_pos: torch.Tensor):
    t_int_n = int_neg.sum()
    t_int_p = int_pos.sum()

    # p_n = t_int_n / (t_int_n + t_int_p)
    # p_p = t_int_p / (t_int_n + t_int_p)
    # ratio = p_n / p_p

    return float((t_int_n / t_int_p).item())


def get_prob_ratio(cfg: PSConfig, ds: MSIDataset):
    "P(c=n)/P(c=p)"

    ds_neg, ds_pos = split_to_mass_groups(
        ds.mzs_,
        ds.int_,
        ds.y,
        filter_mz_lo=cfg.mz_min,
        filter_mz_hi=cfg.mz_max,
        filter_int_lo=cfg.int_min,
    )

    return prob_ratio(ds_neg.int_, ds_pos.int_)


def predict(exp: Experiment, values: np.ndarray | torch.Tensor):
    "(-log P(x | c=n), -log P(x | c=p))"

    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    values = values.to(torch.float32).to(exp.device)

    nll_p = exp.model.pos_head.neg_log_likelihood(values)
    nll_n = exp.model.neg_head.neg_log_likelihood(values)

    return nll_n, nll_p


def classification_prob(nll_n: torch.Tensor, nll_p: torch.Tensor, p_ratio: float):
    "P(c=p | x)"
    lh_ratio = torch.exp(nll_p - nll_n)  # P_(x | n) / P_(x | p)
    cls_prob = 1.0 / (1.0 + lh_ratio * p_ratio)
    return cls_prob


def ratio_min_max(nll_n: torch.Tensor, nll_p: torch.Tensor, p_ratio: float):
    diff_llh = nll_p - nll_n  # log {P_(x | n) / P_(x | p)}
    ratio = torch.exp(diff_llh) * p_ratio  # P_(x, n) / P_(x, p)
    ratio_inv = torch.exp(-diff_llh) / p_ratio  # P_(x, p) / P_(x, n)

    ratio_max = torch.maximum(ratio, ratio_inv)
    ratio_min = 1.0 - torch.minimum(ratio, ratio_inv)

    return ratio_min, ratio_max


def load_experiment(res_dir: pathlib.Path, override_dataset: str | None):
    config_path = res_dir / "config.yml"

    with open(config_path, "r", encoding="utf8") as config_fp:
        config_yaml = yaml.safe_load(config_fp)
        # allow overriding the dataset path in case the evaluation is on another computer
        if override_dataset is not None:
            config_yaml["data_dir_s"] = override_dataset
        cfg = config_file.try_cast(config_yaml, PSConfig)

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    model = GMM1DCls(
        cfg.components,
        cfg.mz_min,
        cfg.mz_max,
    ).to(device)

    saved_files = {p.name: p for p in res_dir.iterdir() if p != config_path}
    expected_files = sorted(["model.pth", "mz_vals.npy", "nll_n.npy", "nll_p.npy"])
    assert (
        sorted(saved_files) == expected_files
    ), f"{sorted(saved_files)=} VS {expected_files=}"

    model.load_state_dict(torch.load(saved_files["model.pth"], weights_only=True))

    mz_vals = torch.from_numpy(np.load(saved_files["mz_vals.npy"]))
    nll_n = torch.from_numpy(np.load(saved_files["nll_n.npy"]))
    nll_p = torch.from_numpy(np.load(saved_files["nll_p.npy"]))

    save_to = get_dir() / res_dir.name
    if not save_to.is_dir():
        save_to.mkdir()
    save_to /= str(len(list(save_to.iterdir())))
    save_to.mkdir()

    return Experiment(
        save_to,
        cfg,
        model,
        device,
        mz_vals,
        nll_n,
        nll_p,
    )


def show_ratio(exp: Experiment, ds: MSIDataset, mz_min: float, mz_max: float):
    fig, axes = plt.subplots(3, 1, squeeze=False, sharex=True)

    # restrict value for the plot
    mz_vals = exp.mz_vals.clone()
    mz_vals = mz_vals[(mz_vals > mz_min) & (mz_vals < mz_max)]

    nll_n, nll_p = predict(exp, mz_vals)
    prob_ratio = get_prob_ratio(exp.config, ds)
    cls_prob = classification_prob(nll_n, nll_p, prob_ratio)
    ratio_min, ratio_max = ratio_min_max(nll_n, nll_p, prob_ratio)

    x_axis = mz_vals.numpy()

    axes[0, 0].plot(x_axis, ratio_max.cpu().numpy())
    axes[0, 0].set_title("ratio max")
    axes[1, 0].plot(x_axis, ratio_min.cpu().numpy())
    axes[1, 0].set_title("ratio min")
    axes[2, 0].plot(x_axis, cls_prob.cpu().numpy())
    axes[2, 0].set_title("prob to be of positive class")

    fig.tight_layout()
    # fig.show()
    # FIXME this should save the image in a nice PNG file
    plt.show()


def false_positive_rate(
    exp: Experiment,
    n_iter: int,
    importance_scorer: ImportanceScorer,
    observed_score: torch.Tensor,
):
    dataset = load_dataset(exp.config)

    false_positives = torch.zeros_like(observed_score, dtype=torch.int64)
    importance_score_lst: list[torch.Tensor] = []

    for iter_idx in range(n_iter):
        gen = torch.Generator().manual_seed(iter_idx)
        baseline = GMM1DCls(exp.config.components, exp.config.mz_min, exp.config.mz_max)
        rnd_dataset = dataset.shuffled_copy(gen)
        tr_neg, tr_pos, vl_neg, vl_pos = split_dataset(
            exp.config, exp.device, rnd_dataset, generator=gen
        )
        nll_n, nll_p = train_model(
            exp.config,
            exp.device,
            baseline,
            exp.mz_vals,
            tr_neg,
            tr_pos,
            vl_neg,
            vl_pos,
            DiscardLogger(),
        )

        # compute importance score for mz_val
        p_ratio = get_prob_ratio(exp.config, rnd_dataset)
        score = importance_scorer(nll_n[-1], nll_p[-1], p_ratio)
        importance_score_lst.append(score)

        # count false positives
        false_positives[score > observed_score] += 1

    return exp.mz_vals, false_positives / n_iter


def _mz_bounds(*ds: FlattenedDataset):
    lo_hi = [(d.mzs_.min(), d.mzs_.max()) for d in ds]
    lo = min(float(tpl[0]) for tpl in lo_hi)
    hi = max(float(tpl[1]) for tpl in lo_hi)
    return lo, hi


def _m_probes_iter(
    exp: Experiment,
    dataset: MSIDataset,
    gen: torch.Generator,
):
    shuffled = dataset.shuffled_copy(gen)

    split_gen = torch.Generator().manual_seed(
        int(torch.randint(2**31 - 1, (), generator=gen))
    )

    # because we use gen.clone_state on the first one, both are guarantee to have the same split index so it's ok
    orig_tr, orig_vl = dataset.random_split(0.7, generator=split_gen.clone_state())

    split_arg = (exp.config.mz_min, exp.config.mz_max, exp.config.int_min)
    neg_orig_tr, pos_orig_tr = orig_tr.split_to_mass_groups(*split_arg)
    neg_orig_vl, pos_orig_vl = orig_vl.split_to_mass_groups(*split_arg)

    # the filtering and flattening don't look at labels, so the shift is computed right
    mz_min, mz_max = _mz_bounds(neg_orig_tr, pos_orig_tr, neg_orig_vl, pos_orig_vl)
    shift_by = 1.1 * (
        mz_max - mz_min
    )  # NOTE there is a 10% margin in case some peaks slightly deviate

    cfg = PSConfig(**config_file.to_dict(exp.config))
    # cfg.components -> not updated, see below for model construction
    cfg.mz_min = mz_min + shift_by
    cfg.mz_max = mz_max + shift_by

    # shift shuffled dataset and split it the exact same way (generator=split_gen)
    shuffled.mzs_ += shift_by
    rand_tr, rand_vl = shuffled.random_split(0.7, generator=split_gen)
    split_arg = (cfg.mz_min, cfg.mz_max, cfg.int_min)
    neg_rand_tr, pos_rand_tr = rand_tr.split_to_mass_groups(*split_arg)
    # neg_rand_vl, pos_rand_vl = rand_vl.split_to_mass_groups(*split_arg)

    # cat the datasets
    tr_neg = neg_orig_tr.cat(neg_rand_tr)
    tr_pos = pos_orig_tr.cat(pos_rand_tr)
    # vl_neg = neg_orig_vl.cat(neg_rand_vl)
    # vl_pos = pos_orig_vl.cat(pos_rand_vl)

    # model
    # model_l = GMM1DCls(cfg.components, mz_min, mz_max, generator=gen)
    # model_r = GMM1DCls(cfg.components, cfg.mz_min, cfg.mz_max, generator=gen)

    # TODO merge left and right to form the complete model
    model = GMM1DCls(
        cfg.components,
        cfg.mz_min,
        cfg.mz_max,
        generator=gen,
    )

    # _, _ = train_model(
    #     cfg,
    #     exp.device,
    #     model,
    #     None,
    #     tr_neg,
    #     tr_pos,
    #     vl_neg,
    #     vl_pos,
    #     DiscardLogger(),
    #     generator=gen,
    # )

    return (
        model,
        prob_ratio(tr_neg.int_, tr_pos.int_),
        (mz_min, mz_max),
        (mz_min + shift_by, mz_max + shift_by),
    )


def interpolate_linspace(linspace: torch.Tensor, factor: int):
    # each semi-open interval is multiplied by factor, and the last point is re-added
    target_len = (len(linspace) - 1) * factor + 1

    # [-1, 1] coordinate for grid_sample fn
    grid_y = torch.linspace(0, len(linspace) - 1, target_len) / (len(linspace) - 1)
    grid_y = 2.0 * grid_y - 1.0
    # this axis is never interpolated
    grid_x = torch.zeros_like(grid_y)

    # joint grid
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.view(1, 1, *grid.shape)

    # sample the piecewise linear interpolation
    interpolated_linspace = torch.nn.functional.grid_sample(
        linspace.view(1, 1, -1, 1),
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(-1)

    return interpolated_linspace


@torch.enable_grad
def maximum_importance(
    model: GMM1DCls,
    mz_min: float,
    mz_max: float,
    p_ratio: float,
    importance_scorer: ImportanceScorer,
):
    """Find the local maxima of the importance function withing a range

    Args:
        model (GMM1DCls): model to estimate the probabilities
        mz_min (float): lower bound of the domain
        mz_max (float): higher bound of the domain
        p_ratio (float): correction factor for the likelihood ratio
        importance_scorer (ImportanceScorer): importance function

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (arg_max, max)
    """

    def _score(p: torch.Tensor):
        nll_n = model.neg_head.neg_log_likelihood(p)
        nll_p = model.pos_head.neg_log_likelihood(p)
        return importance_scorer(nll_n, nll_p, p_ratio)

    points = torch.cat(
        (
            torch.tensor([mz_min, mz_max], device=model.pos_head.mu.device),
            model.pos_head.mu,
            model.neg_head.mu,
        )
    )
    points.sort()
    points = interpolate_linspace(points, 10)
    score = _score(points)

    # find approximate maxima (need to check edges later)
    local_maxima = torch.zeros_like(points, dtype=torch.bool)
    local_maxima[1:-1] = (score[1:-1] > score[:-2]) & (score[1:-1] > score[2:])
    points_lm = points[local_maxima]

    # empty check
    if not torch.any(local_maxima):
        return torch.cat((points[:1], points[-1:])), torch.cat((score[:1], score[-1:]))

    class TmpModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidates = torch.nn.Parameter(points_lm.clone())

        def forward(self):
            importance = _score(self.candidates)
            self.last_importance = importance.detach().clone()
            return importance

    o_mod = TmpModule()
    optim = torch.optim.adam.Adam(o_mod.parameters())

    last_candidates = o_mod.candidates.data.detach().clone()

    # gradient ascent to improve local maxima
    while True:
        optim.zero_grad()
        loss = -1.0 * torch.sum(o_mod())
        loss.backward()
        optim.step()

        # fix bounds
        with torch.no_grad():
            o_mod.candidates.data.clamp_(mz_min, mz_max)

        diff = torch.abs(last_candidates - o_mod.candidates)
        if diff.max() < 1e-4:
            break
        last_candidates = o_mod.candidates.data.detach().clone()

    # discard any value that is not in the range
    valid = (last_candidates >= mz_min) & (last_candidates <= mz_max)
    last_candidates = last_candidates[valid]
    last_importance = o_mod.last_importance[valid]

    # re-add bounds
    last_candidates = torch.cat((points[:1], last_candidates, points[-1:]))
    last_importance = torch.cat((score[:1], last_importance, score[-1:]))

    return last_candidates, last_importance


@torch.enable_grad
def find_intervals_above(
    model: GMM1DCls,
    mz_min: float,
    mz_max: float,
    p_ratio: float,
    importance_scorer: ImportanceScorer,
    threshold: float,
):

    # idea: create a very fine grid (50x interpolation of components + edges)
    #   - find leading and trailing points by comparison
    #       - can we have a mismatch of those ?
    #          YES: assuming the threshold is low enough, the leading edge may just be the first bound (similar for the trailing edge)
    #   - do binary search between those until convergence
    # yay no need for a module anymore

    # how do we do a piecewise linear interpolation ?

    def _score(p: torch.Tensor):
        nll_n = model.neg_head.neg_log_likelihood(p)
        nll_p = model.pos_head.neg_log_likelihood(p)
        return importance_scorer(nll_n, nll_p, p_ratio)

    points = torch.cat(
        [torch.tensor([mz_min, mz_max]), model.pos_head.mu, model.neg_head.mu]
    )
    points.sort()
    points = interpolate_linspace(points, 10)
    score = _score(points)

    # find the indices before the rising and the falling edges
    before_re = torch.nonzero((score[1:] > threshold) & (score[:-1] < threshold))[:, 0]
    before_fe = torch.nonzero((score[1:] < threshold) & (score[:-1] > threshold))[:, 0]

    lo_re, hi_re = points[before_re], points[before_re + 1]
    lo_fe, hi_fe = points[before_fe], points[before_fe + 1]

    # missing first rising edge
    if before_fe[0] < before_re[0]:
        lo_re = torch.cat((torch.tensor([mz_min]), lo_re))
        hi_re = torch.cat((torch.tensor([mz_min]), hi_re))
    # missing the last falling edge
    if before_re[-1] < before_fe[-1]:
        lo_fe = torch.cat((lo_fe, torch.tensor([mz_max])))
        hi_fe = torch.cat((hi_fe, torch.tensor([mz_max])))

    # do binary search on each interval (rising edge)
    while True:
        mid_point_re = 0.5 * (lo_re + hi_re)
        mp_score = _score(mid_point_re)
        is_hi = mp_score > threshold
        hi_re[is_hi] = mid_point_re[is_hi]
        lo_re[~is_hi] = mid_point_re[~is_hi]

        widths = lo_re - hi_re
        if widths.max() < 1e-4:
            break

    # do binary search on each interval (falling edge)
    while True:
        mid_point_fe = 0.5 * (lo_fe + hi_fe)
        mp_score = _score(mid_point_fe)
        is_hi = mp_score < threshold
        hi_fe[is_hi] = mid_point_fe[is_hi]
        lo_fe[~is_hi] = mid_point_fe[~is_hi]

        widths = lo_fe - hi_fe
        if widths.max() < 1e-4:
            break

    return mid_point_re, mid_point_fe


# this would work a lot better with numba... almost impossible to do with torch alone I think
# it's possible to do it by broadcasting first and have a 2D tensor, but what if it's too large to fit in memory ?
# it's possible to do it iteratively but it will be slow

# it's also possible to do another approach where we compute it as a "multi-steps" function which is probably more useful if we want all intervals such that FWER <= 0.05
# yeah, just make a function which returns all interval with 1.0-FWER >= 0.95 (or some other values)


def _family_wise_error_rate(
    intervals: list[tuple[torch.Tensor, torch.Tensor]], sampling: torch.Tensor
):
    """"""

    # the FWER is computed for all values in sampling


def m_probes(
    exp: Experiment,
    n_iter: int,
    importance_scorer: ImportanceScorer,
):
    dataset = load_dataset(exp.config)

    for iter_idx in range(n_iter):
        gen = torch.Generator().manual_seed(iter_idx)
        (*_,) = _m_probes_iter(exp, dataset, gen)

        # evaluate the max importance on the "random" range

        # select all intervals higher than this threshold

    # make a function that computes the largest collection of intervals such that FWER <= thresh
    # using the intervals computed above
    # (this can be done crudely: the bounds are approximative anyway)


# TODO also need to measure the accuracy of the classifier (because the feature-importance of a bad classifier don't matter)


def run_eval(args: argparse.Namespace):
    """run_eval: start the evaluation procedure. `args` should have the following attributes.

    - res-dir (str): the directory containing the results of the training
    - mz_min (float): a minimal bound to run the evaluation
    - mz_max (float): a maximal bound to run the evaluation
    - override-dataset (str | None): a path to use instead of the saved dataset. Ignored if None (default).
    - fpr_iter (int | None): if None, the FPR is not performed. The number of iteration to compute the FPR of the importance values for all saved m/z values during training. mz_min and mz_max are ignored by this analysis.
    """

    res_dir = pathlib.Path(getattr(args, "res-dir"))
    mz_min = float(getattr(args, "mz_min"))
    mz_max = float(getattr(args, "mz_max"))
    if mz_max <= mz_min:
        raise ValueError(f"{mz_max=!r} <= {mz_min=!r}")
    override_dataset: str | None = args.override_dataset
    if override_dataset is not None:
        if not pathlib.Path(override_dataset).is_dir():
            raise ValueError(f"{override_dataset=} not found")
    fpr_iter = 0
    if args.fpr_iter is not None:
        fpr_iter = int(args.fpr_iter)
        if fpr_iter < 0:
            raise ValueError(f"{fpr_iter=} may not be negative")
        if fpr_iter > 1e3:
            print(f"warning, {fpr_iter=} is very high")

    exp = load_experiment(res_dir, override_dataset)
    dataset = load_dataset(exp.config)

    def scorer(nll_n: torch.Tensor, nll_p: torch.Tensor, p_ratio: float):
        return ratio_min_max(nll_n, nll_p, p_ratio)[0]

    # TODO update fpr.csv (local file) to also have the imp, that way we can have nice stats
    if fpr_iter > 0:
        p_ratio = get_prob_ratio(exp.config, dataset)
        imp = scorer(exp.nll_n[-1], exp.nll_p[-1], p_ratio)
        m_mzs, m_fpr = false_positive_rate(exp, fpr_iter, scorer, imp)
        # FIXME these are float32 and not float64 -> this is NOT enough for the csv file (different)
        fpr_df = pd.DataFrame({"mzs": m_mzs, "imp": imp, "fpr": m_fpr})
        fpr_df.to_csv(exp.save_to / "fpr.csv")

    with torch.no_grad():
        show_ratio(exp, dataset, mz_min, mz_max)


if __name__ == "__main__":
    from run_eval import get_parser

    run_eval(get_parser().parse_args())
