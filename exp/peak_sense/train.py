import pathlib
from functools import partial
from typing import Any, Protocol, runtime_checkable

import numpy as np
import omezarrmsi as ozm
import runexp
import torch
import wandb
from runexp import env
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import binary_cross_entropy
from torcheval.metrics import BinaryAUROC
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled

from msi_zarr_analysis.ml.msi_ds import Axis, MSIDataset

from config import PSConfig
from model import Model


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

    files = [f for f in files if "317norm" in f.name]
    assert len(files) == 3, files

    ozm_datasets = {f.stem[6:8]: ozm.OMEZarrMSI(f, mode="r") for f in files}
    assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

    # define padding length: less memory copies if we compute it first
    seq_max_size = max(ds.int_shape[Axis.C] for ds in ozm_datasets.values())

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

    return {
        key: load_fn(ds=ozm_ds, min_len_hint=seq_max_size)
        for key, ozm_ds in ozm_datasets.items()
    }


def prep_train(cfg: PSConfig, datasets: dict[str, MSIDataset]):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    val_dl = DataLoader(
        datasets["14"].to(device),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    train_dl = DataLoader(
        MSIDataset.cat(datasets["13"], datasets["15"]).to(device),
        # torch_dataset["13"].to(device),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    model = Model(
        cfg.n_peaks,
        cfg.mz_min,
        cfg.mz_max,
        cfg.norm_type,
        cfg.head_type,
    ).to(device)

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

    return device, model, train_dl, val_dl, run, out


def train(cfg: PSConfig):
    device, model, tr_dl, vl_dl, run, out = prep_train(cfg, load_dataset(cfg))
    optim = Adam(model.parameters(), lr=cfg.lr)
    f = 2.013e-9  # 2.013*10^-3 / (10^3)^2

    traces_mu_lst = [model.mu.detach().cpu()]
    traces_s1_lst = [model.s1.detach().cpu()]
    traces_w_lst = [model.peak_importance().detach_().cpu()]

    for _ in range(cfg.epochs):
        tr_losses = torch.tensor(4 * [0.0], device=device)
        model.train(True)
        for b_mzs, b_int, b_y, b_w in tr_dl:
            optim.zero_grad()
            prob, overlap, width, reg_l1 = model.train_step(b_mzs, b_int, cfg.alpha, f)
            ce = binary_cross_entropy(prob, b_y.float(), b_w)

            loss = (
                ce
                + cfg.c_overlap * overlap
                + cfg.c_width * width
                + cfg.c_weights * reg_l1
            )
            loss.backward()
            optim.step()

            tr_losses += torch.stack([ce, overlap, width, reg_l1]).detach()

        traces_mu_lst.append(model.mu.detach().cpu())
        traces_s1_lst.append(model.s1.detach().cpu())
        traces_w_lst.append(model.peak_importance().detach_().cpu())

        tr_losses /= len(tr_dl)
        run.log({"train/ce": float(tr_losses[0])}, commit=False)
        run.log({"train/overlap": float(tr_losses[1])}, commit=False)
        run.log({"train/width": float(tr_losses[2])}, commit=False)
        run.log({"train/weights": float(tr_losses[3])}, commit=False)

        model.train(False)
        with torch.no_grad():
            vl_loss = torch.tensor(0.0, device=device)
            metric = BinaryAUROC().to(device)
            for b_mzs, b_int, b_y, b_w in vl_dl:
                prob = model.cls_prob(b_mzs, b_int)
                metric.update(prob, b_y, b_w)
                vl_loss += binary_cross_entropy(prob, b_y.float(), b_w)

            vl_loss /= len(vl_dl)
            run.log({"val/ce": float(vl_loss)}, commit=False)
            run.log({"val/auroc": float(metric.compute())}, commit=True)

    # save traces, model, ...
    np.save(out / "traces_mu", torch.stack(traces_mu_lst).numpy())
    np.save(out / "traces_s1", torch.stack(traces_s1_lst).numpy())
    np.save(out / "traces_w", torch.stack(traces_w_lst).numpy())
    torch.save(model.state_dict(), out / "model.pth")


if __name__ == "__main__":
    runexp.runexp_main(train)
