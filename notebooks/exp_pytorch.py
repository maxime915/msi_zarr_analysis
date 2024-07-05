# %%
import pathlib
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import omezarrmsi as ozm
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torcheval.metrics import BinaryAUROC

import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from msi_zarr_analysis.ml.msi_ds import MSIDataset, Axis  # noqa:E402
from msi_zarr_analysis.ml.peak_sense import PeakSense, ThreshCounter  # noqa:E402

# %%


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


# %%
de_tol = 2e-3  # tolerance of the deisotoping function
slim_dir = pathlib.Path.home() / "datasets" / "COMULIS-slim-msi"
dest_dir = slim_dir.parent / f"slim-deisotoping-{de_tol:.1e}"

files = sorted(dest_dir.iterdir())
assert len(files) == 6

files = [f for f in files if "317norm" in f.name]
assert len(files) == 3, files

ozm_datasets = {
    f.stem[6:8]: ozm.OMEZarrMSI(f, mode="r") for f in files
}
assert sorted(ozm_datasets.keys()) == ["13", "14", "15"]

# define padding length: less memory copies if we compute it first
seq_max_size = max(ds.int_shape[Axis.C] for ds in ozm_datasets.values())

torch_dataset = {
    key: MSIDataset.load(ozm_ds, label_neg=["sc-", "sc+"], label_pos=["ls-", "ls+"], min_len_hint=seq_max_size)
    for key, ozm_ds in ozm_datasets.items()
}

# %%

# estimate f
mz_ref = 1e3
bin_ref = 2.013e-3
f = bin_ref / mz_ref / mz_ref  # 2.013e-9

# how many intervals to consider for overlapping
alpha = 2.0

# %%

assert torch.cuda.is_available()
device = torch.device("cuda:0")
batch_size = 128
epochs = 100

test_dl = DataLoader(
    torch_dataset["14"].to(device),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    worker_init_fn=seed_worker,
)

train_dl = DataLoader(
    MSIDataset.cat(torch_dataset["13"], torch_dataset["15"]).to(device),
    # torch_dataset["13"].to(device),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
)

n_peaks = 1000
mz_min = 300.0  # 496.0  # 300.0  # 150.0
mz_max = 1000.0  # 497.0  # 1000.0  # 1150.0

peak_att = PeakSense(n_peaks, mz_min, mz_max).to(device)
thresh = ThreshCounter(n_peaks).to(device)
head = nn.Linear(n_peaks, 2).to(device)

optim = Adam(
    chain(peak_att.parameters(), thresh.parameters(), head.parameters()),
    lr=1e-2,
)


def predict_logits(mzs_: torch.Tensor, int_: torch.Tensor) -> torch.Tensor:
    int_ = torch.log1p(int_)
    peaks_ = peak_att(mzs_, int_)
    peaks_ = thresh(peaks_)
    return head(peaks_)


# %%

traces_mu_lst: list[torch.Tensor] = [peak_att.mu.cpu()]
traces_s1_lst: list[torch.Tensor] = [torch.exp(0.5 * peak_att.lv).cpu()]

for epoch in range(epochs):
    train_loss, train_batches = 0.0, 0
    for b_mzs, b_int, b_y, b_w in train_dl:
        optim.zero_grad(set_to_none=True)

        logits = predict_logits(b_mzs, b_int)

        ce = cross_entropy(logits, b_y)
        loss = torch.mean(ce * b_w)

        # # no overlap
        # loss += 1e-2 * peak_att.overlap_metric(alpha)
        # appropriate bin size
        loss += 1e-1 * peak_att.width_metric(alpha, f)
        # L1 regularization on linear head
        loss += 1e-1 * head.weight.abs().mean()

        loss.backward()
        optim.step()

        train_loss += float(loss.item())
        train_batches += 1

    traces_mu_lst.append(peak_att.mu.cpu())
    traces_s1_lst.append(torch.exp(0.5 * peak_att.lv).cpu())

    print(f"{epoch=: 4d}: train loss: {train_loss/train_batches:.2e}")

    # compute roc auc
    with torch.no_grad():
        metric = BinaryAUROC().to(device)
        for b_mzs, b_int, b_y, b_w in test_dl:
            logits = predict_logits(b_mzs, b_int)
            pred = torch.argmax(logits, dim=1)
            metric.update(pred, b_y, b_w)
        print(f"{epoch=: 4d}: \t\t\ttest auroc: {float(metric.compute()):.3e}")

# %%

traces_mu = torch.stack(traces_mu_lst, dim=1).detach().numpy()  # [N, T]
traces_s1 = torch.stack(traces_s1_lst, dim=1).detach().numpy()  # [N, T]
v_range = np.arange(traces_mu.shape[1])

fig, ax = plt.subplots(figsize=(8, int(0.2 * len(v_range))))

k = 100
r_idx = slice(k, k + 10, 1)
traces_mu = traces_mu[r_idx]
traces_s1 = traces_s1[r_idx]

for mu_, s1_ in zip(traces_mu, traces_s1, strict=True):
    ...
    p = ax.plot(mu_, v_range)
    ax.fill_betweenx(v_range, mu_ - s1_, mu_ + s1_, alpha=0.3, color=p[0].get_color())

ax.invert_yaxis()
fig.tight_layout()

print(list(np.abs(traces_mu[:, -1] - traces_mu[:, 0])))

# TODO nice graphs
#   - how do mu_i,sigma_i evolve with the training ? record them and show an nice image
#   - what does the output look like on each region ?

# %%
