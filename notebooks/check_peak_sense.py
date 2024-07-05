import pathlib

import torch
from torch import autograd

import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from msi_zarr_analysis.ml.peak_sense import _PeakSenseFn  # noqa:E402


def peak_sense(
    mu: torch.Tensor,
    lv: torch.Tensor,
    mz: torch.Tensor,
    iv: torch.Tensor,
) -> torch.Tensor:
    return _PeakSenseFn.apply(mu, lv, mz, iv, False)  # type:ignore


mu = torch.rand((2**6,), requires_grad=True, device="cuda:0", dtype=torch.float64)
lv = torch.rand((2**6,), requires_grad=True, device="cuda:0", dtype=torch.float64)
mz = torch.rand((2**7, 2**8,), device="cuda:0", dtype=torch.float64)
iv = torch.rand((2**7, 2**8,), device="cuda:0", dtype=torch.float64)

# with autograd.detect_anomaly():
peak_sense(mu, lv, mz, iv).sum().backward()
for t in [mu, lv, mz, iv]:
    t.grad = None

autograd.gradcheck(
    func=peak_sense,
    inputs=(mu, lv, mz, iv),
)
