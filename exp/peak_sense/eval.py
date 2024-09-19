import argparse
import dataclasses
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import omezarrmsi as ozm
import torch
import yaml
from PIL import Image
from runexp import config_file, env

from config import PSConfig
from model import Model
from train import MSIDataset, load_dataset as tr_load_dataset, Axis


# TODO show the distribution of the width w.r.t. the target width (like log (actual / expected))
#   - for every mz, measure the width and compare it to f * mz**2


@dataclasses.dataclass
class Experiment:
    save_to: pathlib.Path
    config: PSConfig
    model: Model
    device: torch.device
    traces_mu: np.ndarray
    traces_s1: np.ndarray
    traces_w: np.ndarray


def get_dir():
    res_dir = pathlib.Path(__file__).parent / "res-eval"
    if not res_dir.is_dir():
        raise ValueError(f"{res_dir} not found")
    return res_dir


def load_experiment(res_dir: pathlib.Path):
    # path.name() is the execution key
    # find the invocation key from runexp
    # go to .runexp/train/INVOCATION_KEY and search for sweep-1-XXX.yml for the info

    execution_key = res_dir.name
    invocation_key = env._derive_inv_key(execution_key)
    execution_arg = execution_key.removeprefix(invocation_key).removeprefix(env._SEP)

    # very specific to this, not great --> a copy of the config should have been saved :/
    runexp_dir = pathlib.Path(
        "/home/maxime/repos/msi_zarr_analysis/copy-runexp-cyto/train"
    )
    config_path = runexp_dir / invocation_key / f"sweep-1-{execution_arg}.yml"

    with open(config_path, "r", encoding="utf8") as config_fp:
        config_yaml = yaml.safe_load(config_fp)
        cfg = config_file.try_cast(config_yaml["base_config"], PSConfig)

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    model = Model(
        cfg.n_peaks,
        cfg.mz_min,
        cfg.mz_max,
        cfg.norm_type,
        cfg.head_type,
    ).to(device)

    saved_files = {p.name: p for p in res_dir.iterdir() if p != config_path}
    assert sorted(saved_files) == ["model.pth", "traces_mu.npy", "traces_s1.npy", "traces_w.npy"]

    model.load_state_dict(torch.load(saved_files["model.pth"], weights_only=True))

    traces_mu = np.load(saved_files["traces_mu.npy"])
    traces_s1 = np.load(saved_files["traces_s1.npy"])
    traces_w = np.load(saved_files["traces_w.npy"])

    save_to = get_dir() / execution_key
    if not save_to.is_dir():
        save_to.mkdir()
    save_to /= str(len(list(save_to.iterdir())))
    save_to.mkdir()

    return Experiment(
        save_to,
        cfg,
        model,
        device,
        traces_mu,
        traces_s1,
        traces_w,
    )


def show_peak_traces(exp: Experiment, mz_min: float, mz_max: float):
    "evolution of the position and width of each peak during training"

    # [0, ..., nSteps]
    v_range = np.arange(exp.traces_mu.shape[0])

    mask = np.any((exp.traces_mu >= mz_min) & (exp.traces_mu <= mz_max), axis=0)
    indices, = np.nonzero(mask)

    fig, axP = plt.subplots(figsize=(8, int(0.1 * exp.traces_mu.shape[0])))

    axP.set_title("Peak with width")
    for idx in indices:
        mu_ = exp.traces_mu[:, idx]
        s1_ = exp.traces_s1[:, idx]
        p = axP.plot(mu_, v_range)
        axP.fill_betweenx(v_range, mu_ - s1_, mu_ + s1_, alpha=0.3, color=p[0].get_color())
    axP.invert_yaxis()

    fig.tight_layout()

    # save fig to disk
    fig.savefig(exp.save_to / "peak_traces.png")


def show_weight_traces(exp: Experiment, mz_min: float, mz_max: float):
    "evolution of the position and *weight* of each peak during training"

    # [0, ..., nSteps]
    v_range = np.arange(exp.traces_mu.shape[0])

    mask = np.any((exp.traces_mu >= mz_min) & (exp.traces_mu <= mz_max), axis=0)
    indices, = np.nonzero(mask)

    fig, axW = plt.subplots(figsize=(8, int(0.1 * exp.traces_mu.shape[0])))

    # the width must be visible on the plot, so
    #   -> not so thin that all peaks are invisible
    #   -> not so large that one peak covers everything
    #   -> w = 1.0 should refer to W/N ?
    coef = (mz_max - mz_min) / np.count_nonzero(mask)

    axW.set_title("Peak with importance (weight)")
    for idx in indices:
        mu_ = exp.traces_mu[:, idx]
        hw_ = coef * exp.traces_w[:, idx]
        p = axW.plot(mu_, v_range)
        axW.fill_betweenx(v_range, mu_ - hw_, mu_ + hw_, alpha=0.3, color=p[0].get_color())
    axW.invert_yaxis()

    fig.tight_layout()

    # save fig to disk
    fig.savefig(exp.save_to / "peak_importances.png")


def load_dataset(cfg: PSConfig):
    # TODO pass the new dataset as an argument to the script to make it more general
    # modify dataset path because the devices are different
    data_dir = pathlib.Path(cfg.data_dir_s)
    if data_dir.is_relative_to("/home/mamodei/"):
        data_dir = "/home/maxime/" / data_dir.relative_to("/home/mamodei")
    cfg = PSConfig(**vars(cfg))  # copy
    cfg.data_dir_s = str(data_dir)
    del data_dir

    files = sorted(cfg.data_dir.iterdir())
    assert len(files) == 6

    files = [f for f in files if "317norm" in f.name]
    assert len(files) == 3, files

    ozm_datasets = {f.stem[6:8]: ozm.OMEZarrMSI(f, mode="r") for f in files}
    assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

    dataset_full = {
        key: MSIDataset.load_full_images(ds=ozm_ds)
        for key, ozm_ds in ozm_datasets.items()
    }

    dataset_annotated = tr_load_dataset(cfg)
    return dataset_full, dataset_annotated


def show_segmentation_masks(exp: Experiment):
    "segmentation masks for all 3 regions (+ ground truth)"

    exp.model.train(False)

    ds_full, ds_ann = load_dataset(exp.config)
    for key, (a_mzs, a_int, ys, xs) in ds_full.items():
        # make a single batch of the dataset (small enough in this case)
        prob = exp.model.cls_prob(
            a_mzs.to(exp.device),
            a_int.to(exp.device),
        )

        width = xs.max() + 1
        height = ys.max() + 1
        prob_spatial = np.zeros((height, width), np.float64)
        prob_spatial[ys, xs] = prob.numpy(force=True)

        colored = np.zeros((height, width, 3), dtype=np.uint8)
        # red channel = prob_spatial
        # blue channel = prob_spatial
        # background = [0.6]  # light gray
        colored[..., 0] = np.round(255.0 * prob_spatial)
        colored[..., 2] = 255 - colored[..., 0]
        background = np.full((height, width), True, dtype=np.bool_)
        background[ys, xs] = False
        colored[background, :] = 150  # light gray

        Image.fromarray(colored, mode="RGB").save(exp.save_to / f"mask_{key}_pred.png")

        # add annotations with their weight
        colored *= 0
        colored[background, :] = 150  # background is the same

        ys = np.array([c[Axis.Y] for c in ds_ann[key].coords])
        xs = np.array([c[Axis.X] for c in ds_ann[key].coords])
        c1 = np.asarray(ds_ann[key].y == 1)
        w_ = np.asarray(ds_ann[key].w)

        colored[ys[c1], xs[c1], 0] = np.round(255.0 * w_[c1])
        colored[ys[~c1], xs[~c1], 2] = np.round(255.0 * w_[~c1])

        Image.fromarray(colored, mode="RGB").save(exp.save_to / f"mask_{key}_gt.png")


def run_all(exp: Experiment, mz_min: float, mz_max: float):
    show_peak_traces(exp, mz_min, mz_max)
    show_weight_traces(exp, mz_min, mz_max)
    show_segmentation_masks(exp)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("res-dir", type=str)
    parser.add_argument("--mz-min", type=float, default=778.0)
    parser.add_argument("--mz-max", type=float, default=786.0)

    args = parser.parse_args()

    res_dir = pathlib.Path(getattr(args, "res-dir"))
    mz_min = float(getattr(args, "mz_min"))
    mz_max = float(getattr(args, "mz_max"))
    if mz_max <= mz_min:
        raise ValueError(f"{mz_max=!r} <= {mz_min=!r}")

    with torch.no_grad():
        run_all(load_experiment(res_dir), mz_min, mz_max)


if __name__ == "__main__":
    main()
