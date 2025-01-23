import pathlib
from functools import partial
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import omezarrmsi as ozm
import runexp
import torch
import yaml
from config import PSConfig
from runexp import env
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torcheval.metrics import Mean, BinaryAUROC

import wandb
from msi_zarr_analysis.ml.gmm import GMM1DClsShared
from msi_zarr_analysis.ml.msi_ds import Axis, MSIDataset
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run


@runtime_checkable
class Logger(Protocol):
    def log(self, values: dict[str, Any], commit: bool = True) -> None: ...

    def get_url(self) -> str | None: ...

    @property
    def name(self) -> str: ...


class StdOutLogger:
    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}

    def log(self, values: dict[str, Any], commit: bool = True):
        self.kv.update(values)
        if commit:
            print(f"{self.kv!r}")
            self.kv.clear()

    def get_url(self) -> str | None:
        return None

    @property
    def name(self):
        return str(type(self))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def load_dataset(cfg: PSConfig) -> MSIDataset:
    files = sorted(cfg.data_dir.iterdir())
    assert len(files) == 6

    # NOTE in this case it only makes sense to use the 317 norm
    # without normalisation there may be a bias which would invalidate the results
    files = [f for f in files if "317norm" in f.name]
    assert len(files) == 3, files

    ozm_datasets = {f.stem[6:8]: ozm.OMEZarrMSI(f, mode="r") for f in files}
    regions = ["13", "14", "15"]
    assert sorted(ozm_datasets.keys()) == regions

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

    if cfg.region == "all":
        min_len_hint = max(ozm_datasets[r].int_shape[Axis.C] for r in regions)
        dataset_parts = [load_fn(ds=ozm_datasets[r], min_len_hint=min_len_hint) for r in regions]
        dataset = MSIDataset.cat(*dataset_parts)
    else:
        ozm_ds = ozm_datasets[cfg.region]
        dataset = load_fn(ds=ozm_ds, min_len_hint=ozm_ds.int_shape[Axis.C])

    return dataset


def prep_train(cfg):
    model = GMM1DClsShared(cfg.components, cfg.mz_min, cfg.mz_max)

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

    return model, run, out


class ModelMetadata(NamedTuple):
    prior_pos: float
    int_log_mean: torch.Tensor
    int_log_std: torch.Tensor

    def scale(self, values):
        return torch.exp((torch.log(values) - self.int_log_mean) / self.int_log_std)

    @property
    def prior_neg(self):
        return 1.0 - self.prior_pos


def train_model(
    cfg: PSConfig,
    device: torch.device,
    model: GMM1DClsShared,
    tr_ds: MSIDataset,
    vl_ds: MSIDataset,
    run: Logger,
):
    model = model.to(device)
    optim = Adam(model.parameters(), lr=cfg.lr)

    mask_tr = (tr_ds.mzs_ > 0.0) & (tr_ds.int_ > 0.0)
    mask_vl = (vl_ds.mzs_ > 0.0) & (vl_ds.int_ > 0.0)

    metadata = ModelMetadata(
        float(tr_ds.y.float().mean().item()),
        *torch.std_mean(torch.log(tr_ds.int_[mask_tr]))[::-1]
    )

    # normalize the intensities
    tr_ds = tr_ds.clone()
    vl_ds = vl_ds.clone()
    tr_ds.int_[mask_tr] = metadata.scale(tr_ds.int_[mask_tr])
    vl_ds.int_[mask_vl] = metadata.scale(vl_ds.int_[mask_vl])

    # NOTE I don't know why but this fails with num_workers > 0 with a CUDA error
    tr_dl = DataLoader(tr_ds.to(device), batch_size=cfg.batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker)
    vl_dl = DataLoader(vl_ds.to(device), batch_size=cfg.batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker)

    last_loss = torch.inf
    for _ in range(cfg.max_epochs):
        model.train(True)

        tr_loss = Mean().to(device)
        for (b_mzs, b_int, b_y, _) in tr_dl:
            optim.zero_grad()
            # NOTE the NLL is only computed up to the prior over the number of
            # items in a spectrum, which is assumed constant for simplicity.

            pos_mask = (b_y == 1)
            valid_mask = (b_mzs > 0.0)

            nll_n = model.neg_head.ws_neg_log_likelihood(b_mzs[pos_mask][valid_mask[pos_mask]], b_int[pos_mask][valid_mask[pos_mask]])
            nll_p = model.pos_head.ws_neg_log_likelihood(b_mzs[~pos_mask][valid_mask[~pos_mask]], b_int[~pos_mask][valid_mask[~pos_mask]])
            loss = torch.mean(nll_n) + torch.mean(nll_p)
            loss.backward()
            optim.step()
            tr_loss.update(loss, weight=b_mzs.shape[0])

        with torch.no_grad():
            model.eval()
            vl_loss_n, vl_loss_p, vl_loss = Mean().to(device), Mean().to(device), Mean().to(device)
            vl_auroc = BinaryAUROC().to(device)
            for (b_mzs, b_int, b_y, _) in vl_dl:
                pos_mask = (b_y == 1)
                valid_mask = (b_mzs > 0.0)

                nll_n = model.neg_head.ws_neg_log_likelihood(b_mzs[pos_mask][valid_mask[pos_mask]], b_int[pos_mask][valid_mask[pos_mask]])
                nll_p = model.pos_head.ws_neg_log_likelihood(b_mzs[~pos_mask][valid_mask[~pos_mask]], b_int[~pos_mask][valid_mask[~pos_mask]])
                nll_n = nll_n.mean()
                nll_p = nll_p.mean()

                # small note about why we don't worry about b_mzs == 0.0 here
                # in those cases, intensity is also 0.0 so the weighted likelihood will be unchanged
                # and because we are in no-grad, we're sure it won't impact the model
                pred = model.predict_proba(b_mzs, b_int, metadata.prior_pos)[:, 1]
                vl_auroc.update(pred, b_y)

                vl_loss_n.update(nll_n, weight=pos_mask.shape[0] - pos_mask.sum())
                vl_loss_p.update(nll_p, weight=pos_mask.sum())
                vl_loss.update(nll_n + nll_p)

        loss_vl = vl_loss.compute().item()
        run.log({
            "tr/nll": float(tr_loss.compute().item()),
            "val/nll_n": float(vl_loss_n.compute().item()),
            "val/nll_p": float(vl_loss_p.compute().item()),
            "val/nll": loss_vl,
            "val/auroc": float(vl_auroc.compute().item()),
        })

        if abs(last_loss - loss_vl) < cfg.convergence_threshold:
            break

        last_loss = loss_vl

    return metadata


def train(cfg: PSConfig):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    model, run, out = prep_train(cfg)
    dataset = load_dataset(cfg)
    tr_ds, vl_ds = dataset.random_split(0.7, generator=torch.Generator().manual_seed(0))
    metadata = train_model(cfg, device, model, tr_ds, vl_ds, run=run)

    torch.save(model.state_dict(), out / "model.pth")
    torch.save(metadata, out / "metadata.pth")


if __name__ == "__main__":
    runexp.runexp_main(train)
