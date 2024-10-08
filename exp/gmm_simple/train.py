import pathlib
from functools import partial
from typing import Any, Protocol, runtime_checkable

import numpy as np
import omezarrmsi as ozm
import runexp
import torch
import yaml
from config import PSConfig
from runexp import env
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

import wandb
from msi_zarr_analysis.ml.gmm import GMM1DCls
from msi_zarr_analysis.ml.msi_ds import Axis, MSIDataset, split_to_mass_groups, FlattenedDataset
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run


@runtime_checkable
class Logger(Protocol):
    def log(self, values: dict[str, Any], commit: bool = False) -> None: ...

    def get_url(self) -> str | None: ...

    @property
    def name(self) -> str: ...


class StdOutLogger:
    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}

    def log(self, values: dict[str, Any], commit: bool = False):
        self.kv.update(values)
        if commit:
            print(f"{self.kv!r}")
            self.kv.clear()

    def get_url(self) -> str | None:
        return None

    @property
    def name(self) -> str:
        return "SilentLogger"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def load_dataset(cfg: PSConfig):
    files = sorted(cfg.data_dir.iterdir())
    assert len(files) == 6

    # NOTE in this case it only makes sense to use the 317 norm
    # without normalisation there may be a bias which would invalidate the results
    files = [f for f in files if "317norm" in f.name]
    assert len(files) == 3, files

    ozm_datasets = {f.stem[6:8]: ozm.OMEZarrMSI(f, mode="r") for f in files}
    assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

    match cfg.problem:
        case "ls-/ls+":
            load_fn = partial(MSIDataset.load, label_neg=["ls-"], label_pos=["ls+"])
        case "sc-/sc+":
            load_fn = partial(MSIDataset.load, label_neg=["sc-"], label_pos=["sc+"])
        case "sc/ls":
            load_fn = partial(
                MSIDataset.load, label_neg=["sc-", "sc+"], label_pos=["ls-", "ls+"]
            )
        case _:
            raise ValueError(f"{cfg.problem=!r} is not recognized")

    ozm_ds = ozm_datasets[cfg.region]
    dataset = load_fn(ds=ozm_ds, min_len_hint=ozm_ds.int_shape[Axis.C])

    return dataset


def prep_train(cfg: PSConfig, dataset: MSIDataset):

    ds_neg, ds_pos = split_to_mass_groups(
        dataset.mzs_,
        dataset.int_,
        dataset.y,
        filter_mz_lo=cfg.mz_min,
        filter_mz_hi=cfg.mz_max,
        filter_int_lo=cfg.int_min,
    )

    model = GMM1DCls(cfg.components, cfg.mz_min, cfg.mz_max)

    res_dir = pathlib.Path(__file__).parent / "res"
    if not res_dir.is_dir():
        raise ValueError(f"{res_dir} not found")
    out = res_dir / env.execution_key()
    out.mkdir()

    run: Logger
    run_ = wandb.init(
        name=cfg.wandb_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        config=runexp.config_file.to_dict(cfg),
    )
    if cfg.wandb_mode == "disabled" and isinstance(run_, RunDisabled):
        run = StdOutLogger()
    elif isinstance(run_, Run):
        run = run_
    else:
        raise ValueError("failed to log in to wandb")

    # save a copy of the config so it's easier to parse back
    with open(out / "config.yml", "w", encoding="utf8") as out_cfg:
        cfg_dict = runexp.config_file.to_dict(cfg)
        cfg_dict["wandb_name"] = run.get_url() or run.name
        yaml.safe_dump(cfg_dict, out_cfg)

    return model, ds_neg, ds_pos, run, out


def train_model(
    cfg: PSConfig,
    device: torch.device,
    model: GMM1DCls,
    dataset_neg: FlattenedDataset,
    dataset_pos: FlattenedDataset,
    run: Logger,
):
    "update `model` to fit the given dataset"

    model = model.to(device)

    dl_neg = DataLoader(
        dataset_neg.to(device), batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    dl_pos = DataLoader(
        dataset_pos.to(device),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    optim = Adam(model.parameters(), lr=cfg.lr)

    mz_vals = torch.linspace(cfg.mz_min, cfg.mz_max, 50 * cfg.components, device=device)
    ratio_min_lst = [model.ratio_min(mz_vals).detach().cpu()]
    ratio_max_lst = [model.ratio_max(mz_vals).detach().cpu()]

    for _ in range(cfg.epochs):
        tr_losses = torch.tensor(2 * [0.0], device=device)
        model.train(True)

        num_batches = 0
        for (m_n, i_n), (m_p, i_p) in zip(dl_neg, dl_pos, strict=False):
            optim.zero_grad()
            nll_n = model.neg_head.ws_neg_log_likelihood(
                m_n, cfg.norm_intensity * i_n / len(m_n)
            )
            nll_p = model.pos_head.ws_neg_log_likelihood(
                m_p, cfg.norm_intensity * i_p / len(m_p)
            )
            (nll_n + nll_p).backward()
            optim.step()

            tr_losses += torch.stack([nll_n, nll_p]).detach()
            num_batches += 1

        ratio_min_lst.append(model.ratio_min(mz_vals).detach().cpu())
        ratio_max_lst.append(model.ratio_max(mz_vals).detach().cpu())

        tr_losses /= num_batches
        run.log({"train/nll_n": float(tr_losses[0])}, commit=False)
        run.log({"train/nll_p": float(tr_losses[1])})

    return mz_vals, torch.stack(ratio_min_lst), torch.stack(ratio_max_lst), model


def train(cfg: PSConfig):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    model, ds_neg, ds_pos, run, out = prep_train(cfg, load_dataset(cfg))
    mz_vals, ratio_min, ratio_max, model = train_model(cfg, device, model, ds_neg, ds_pos, run)

    # save ratio & model
    np.save(out / "mz_vals", mz_vals.numpy())
    np.save(out / "ratio_min", ratio_min.numpy())
    np.save(out / "ratio_max", ratio_max.numpy())
    torch.save(model.state_dict(), out / "model.pth")


if __name__ == "__main__":
    runexp.runexp_main(train)
