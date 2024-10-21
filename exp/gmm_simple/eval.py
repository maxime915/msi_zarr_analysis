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
from msi_zarr_analysis.ml.msi_ds import MSIDataset, split_to_mass_groups
from runexp import config_file

from config import PSConfig
from train import train_model, load_dataset


def random_mass_group(
    mzs_: torch.Tensor,
    int_: torch.Tensor,
    y: torch.Tensor,
    filter_mz_lo: float = 0.0,
    filter_mz_hi: float | None = None,
    filter_int_lo: float | None = None,
    *,
    generator: torch.Generator | None = None,
):
    "return [negative mass group, positive mass group]"
    perm = torch.randperm(len(y), generator=generator)
    return split_to_mass_groups(
        mzs_, int_, y[perm], filter_mz_lo, filter_mz_hi, filter_int_lo
    )


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
    return cls_prob.cpu()


def ratio_min_max(nll_n: torch.Tensor, nll_p: torch.Tensor, p_ratio: float):
    diff_llh = nll_p - nll_n  # log {P_(x | n) / P_(x | p)}
    ratio = torch.exp(diff_llh) * p_ratio  # P_(x, n) / P_(x, p)
    ratio_inv = torch.exp(-diff_llh) / p_ratio  # P_(x, p) / P_(x, n)

    ratio_max = torch.maximum(ratio, ratio_inv).cpu()
    ratio_min = 1.0 - torch.minimum(ratio, ratio_inv).cpu()

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

    x_axis = mz_vals.cpu().numpy()

    axes[0, 0].plot(x_axis, ratio_max.numpy())
    axes[0, 0].set_title("ratio max")
    axes[1, 0].plot(x_axis, ratio_min.numpy())
    axes[1, 0].set_title("ratio min")
    axes[2, 0].plot(x_axis, cls_prob)
    axes[2, 0].set_title("prob to be of positive class")

    fig.tight_layout()
    # fig.show()
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
        ds_neg, ds_pos = random_mass_group(
            dataset.mzs_,
            dataset.int_,
            dataset.y,
            exp.config.mz_min,
            exp.config.mz_max,
            exp.config.int_min,
            generator=gen,
        )
        # NOTE the mz_vals here are used for validation -> we need the full group, even if the experiment doesn't care
        nll_n, nll_p = train_model(
            exp.config,
            exp.device,
            baseline,
            exp.mz_vals,
            ds_neg,
            ds_pos,
            DiscardLogger(),
        )

        # compute importance score for mz_val
        p_ratio = prob_ratio(ds_neg.int_, ds_pos.int_)
        score = importance_scorer(nll_n[-1], nll_p[-1], p_ratio)
        importance_score_lst.append(score)

        # count false positives
        false_positives[score > observed_score] += 1

    return exp.mz_vals, false_positives / n_iter


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

    if fpr_iter > 0:
        p_ratio = get_prob_ratio(exp.config, dataset)
        imp = scorer(exp.nll_n[-1], exp.nll_p[-1], p_ratio)
        m_mzs, m_fpr = false_positive_rate(exp, fpr_iter, scorer, imp)
        fpr_df = pd.DataFrame({"mzs": m_mzs, "imp": imp, "fpr": m_fpr})
        fpr_df.to_csv(exp.save_to / "fpr.csv")

    with torch.no_grad():
        show_ratio(exp, dataset, mz_min, mz_max)


if __name__ == "__main__":
    from run_eval import get_parser

    run_eval(get_parser().parse_args())
