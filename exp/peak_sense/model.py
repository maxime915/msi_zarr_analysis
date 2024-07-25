from math import ceil
from typing import cast, Literal

import torch
from torch import nn

from msi_zarr_analysis.ml.peak_sense import PeakSense, ThreshCounter


class Model(nn.Module):
    def __init__(
        self,
        n_peaks: int,
        mz_min: float,
        mz_max: float,
        peak_norm: Literal["batch-norm", "thresh-counter"],
        head: Literal["linear", "mlp"],
    ) -> None:
        super().__init__()

        self.peak_sense = PeakSense(n_peaks, mz_min, mz_max)

        self.peak_norm_type: Literal["batch-norm"] | Literal["thresh-counter"] = (
            peak_norm
        )
        if peak_norm == "batch-norm":
            self.peak_norm = nn.BatchNorm1d(
                n_peaks, affine=False, track_running_stats=True
            )
        else:
            assert peak_norm == "thresh-counter"
            self.peak_norm = ThreshCounter(n_peaks)

        self.head_type: Literal["linear"] | Literal["mlp"] = head
        self.head: nn.Module
        if head == "linear":
            self.head = nn.Linear(n_peaks, 1)
        else:
            assert head == "mlp"
            self.head = nn.Sequential(
                nn.Linear(n_peaks, ceil(n_peaks ** (2 / 3))),
                nn.ReLU(),
                nn.Linear(ceil(n_peaks ** (2 / 3)), ceil(n_peaks ** (1 / 3))),
                nn.ReLU(),
                nn.Linear(ceil(n_peaks ** (1 / 3)), 1),
            )

    def forward(self, b_mzs: torch.Tensor, b_int: torch.Tensor):
        peaks = self.peak_sense(b_mzs, torch.log1p(b_int))
        logits = self.head(self.peak_norm(peaks))
        assert logits.shape[-1] == 1
        return torch.sigmoid(logits[..., 0])

    def cls_prob(self, b_mzs: torch.Tensor, b_int: torch.Tensor):
        assert not self.training
        return cast(torch.Tensor, self(b_mzs, b_int))

    @property
    def mu(self):
        "mean of each peak"
        return self.peak_sense.mu

    @property
    def s1(self):
        "standard deviation of each peak"
        return torch.exp(0.5 * self.peak_sense.lv)

    def _first_weight(self):
        "return the first weight matrix[C, N]"
        if self.head_type == "linear":
            assert isinstance(self.head, nn.Linear)
            return self.head.weight
        else:
            assert self.head_type == "mlp"
            assert isinstance(self.head, nn.Sequential)
            linear = self.head[0]
            assert isinstance(linear, nn.Linear)
            return linear.weight

    def peak_importance(self, indices: torch.Tensor | None = None):
        "indices: int64[N] -> float32[N]"
        weights = self._first_weight()
        if indices is not None:
            # all classes, only the selected peaks
            weights = weights[:, indices]
        return weights.abs().mean(dim=0)

    def train_step(
        self,
        b_mzs: torch.Tensor,
        b_int: torch.Tensor,
        alpha: float,
        mzs_f: float,
    ):
        """compute the necessary tensor to complete a training step.

        Args:
            b_mzs (torch.Tensor[B, L]): 0-padded masses
            b_int (torch.Tensor[B, L]): 0-padded intensities, matching to b_mzs
            alpha (float): how many standard deviation makes the bin size
            mzs_f (float): the ratio between the bin size and the square of the mass

        Returns:
            prob (torch.Tensor[B]): the probability that each sample belongs to the positive class
            overlap (torch.Tensor[]): a scalar score for the overlapping of peaks
            width (torch.Tensor[]): a scalar score for the mismatch of the peak width w.r.t. the theoretical formula
            reg_l1 (torch.Tensor[]): a scalar score for the L1 regularization of the first weights
        """

        assert self.training
        prob = cast(torch.Tensor, self(b_mzs, b_int))

        # other useful metrics
        overlap = self.peak_sense.overlap_metric(alpha)
        width = self.peak_sense.width_metric(alpha, mzs_f)
        reg_l1 = self._first_weight().abs().mean()

        return (
            prob,
            overlap,
            width,
            reg_l1,
        )
