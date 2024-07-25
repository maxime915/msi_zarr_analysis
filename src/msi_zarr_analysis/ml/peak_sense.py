from math import exp, log

import torch
import torch.nn as nn
from numba import cuda
from torch.nn.functional import sigmoid


THREADS_PER_BLOCK: tuple[int, int] = (8, 16)
THRESH_LOW_WEIGHT = -10.0


def kern(shape: tuple[int, int]):
    blocks_per_grid = tuple(
        (a + (b - 1)) // b for a, b in zip(shape, THREADS_PER_BLOCK, strict=True)
    )
    return blocks_per_grid, THREADS_PER_BLOCK


# TODO there is still some possible optimization.
#   - rather than compute dx_i/dz and multiply with the chain rule afterward,
#       we can pass dC/dx_i in and compute dC/dz directly.
#   - using the previous optimization, it might be possible to compute the gradient
#       w.r.t. the inputs as well using a reasonable amount of memory and time


@cuda.jit(
    [
        "void(f4[:], f4[:], f4[:, :], f4[:, :], f4[:, :])",
        "void(f8[:], f8[:], f8[:, :], f8[:, :], f8[:, :])",
    ]
)
def _nb_forward(mu, inv_s2, mz, iv, out):
    # batch index
    b = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # type:ignore
    # peak index
    p = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  # type:ignore

    if b < out.shape[0] and p < out.shape[1]:
        ws = 0.0
        for i in range(mz.shape[1]):
            arg = -0.5 * (mz[b, i] - mu[p]) ** 2 * inv_s2[p]
            if arg >= THRESH_LOW_WEIGHT:
                ws += exp(arg) * iv[b, i]

        out[b, p] = ws


@cuda.jit(
    [
        "void(f4[:], f4[:], f4[:, :], f4[:, :], f4[:, :], f4[:, :])",
        "void(f8[:], f8[:], f8[:, :], f8[:, :], f8[:, :], f8[:, :])",
    ]
)
def _nb_backward_param_only(mu, inv_s1, mz, iv, g_mu, g_s1):
    "computes the partial derivatives of mu and s1 wrt x"

    # batch index
    b = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # type:ignore
    # peak index
    p = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  # type:ignore

    if b < g_mu.shape[0] and p < g_mu.shape[1]:
        ws_mu = 0.0
        ws_s1 = 0.0
        for i in range(mz.shape[1]):
            arg = -0.5 * (mz[b, i] - mu[p]) ** 2 * inv_s1[p] ** 2
            if arg >= THRESH_LOW_WEIGHT:
                wi = exp(arg) * iv[b, i]
                ws_mu -= (mu[p] - mz[b, i]) * inv_s1[p] ** 2 * wi
                ws_s1 += (mu[p] - mz[b, i]) ** 2 * inv_s1[p] ** 3 * wi

        g_mu[b, p] = ws_mu
        g_s1[b, p] = ws_s1


class _PeakSenseFn(torch.autograd.Function):
    "see peak_sense()"

    @staticmethod
    def forward(*a, **kw):
        mu: torch.Tensor
        lv: torch.Tensor
        mz: torch.Tensor
        iv: torch.Tensor
        mu, lv, mz, iv, _ = a  # additional arg for compute_grad_spectra
        assert not kw, "no keyword argument accepted to forward"

        # use numba/cuda for a parallel and memory efficient computation
        peaks = torch.empty(
            (mz.shape[0], mu.shape[0]), dtype=iv.dtype, device=iv.device
        )

        _nb_forward[kern(peaks.shape)](  # type:ignore
            cuda.as_cuda_array(mu.detach()),
            cuda.as_cuda_array(torch.exp(-lv.detach())),
            cuda.as_cuda_array(mz.detach()),
            cuda.as_cuda_array(iv.detach()),
            cuda.as_cuda_array(peaks),
        )

        return peaks

    @staticmethod
    def setup_context(ctx, inputs: tuple, output):
        mu, lv, mz, iv, compute_grad_spectra = inputs
        ctx.save_for_backward(mu, lv, mz, iv)
        ctx.compute_grad_spectra = compute_grad_spectra  # saved for backward

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *a, **kw):
        g_peaks: torch.Tensor  # [B, N]
        (g_peaks,) = a
        assert not kw, "no keyword argument accepted to backward"

        mu: torch.Tensor  # [N]
        lv: torch.Tensor  # [N]
        mz: torch.Tensor  # [B, L]
        iv: torch.Tensor  # [B, L]
        mu, lv, mz, iv = ctx.saved_tensors

        if ctx.compute_grad_spectra:
            raise NotImplementedError("full gradient hasn't been implemented yet")

        d__x__mu = torch.empty(  # [B, N]
            (mz.shape[0], mu.shape[0]), dtype=iv.dtype, device=iv.device
        )
        d__x__s1 = torch.empty_like(d__x__mu)  # [B, N]

        _nb_backward_param_only[kern(d__x__mu.shape)](  # type:ignore
            cuda.as_cuda_array(mu.detach()),
            cuda.as_cuda_array(torch.exp(-0.5 * lv).detach()),
            cuda.as_cuda_array(mz.detach()),
            cuda.as_cuda_array(iv.detach()),
            cuda.as_cuda_array(d__x__mu),
            cuda.as_cuda_array(d__x__s1),
        )

        # compute grad for mu and lv
        g_mu = torch.sum(g_peaks * d__x__mu, dim=0)
        g_s1 = torch.sum(g_peaks * d__x__s1, dim=0)

        # grad should be w.r.t. lv , not w.r.t. s1
        g_lv = 0.5 * torch.exp(0.5 * lv) * g_s1

        # mu, lv, mz, iv, compute_grad_spectra
        return g_mu, g_lv, None, None, None


def peak_sense(
    mu: torch.Tensor,
    lv: torch.Tensor,
    mz: torch.Tensor,
    iv: torch.Tensor,
    *,
    compute_grad_spectra: bool,
) -> torch.Tensor:
    """Backbone of the Peak Sense module to wrap the numba kernel.

    Args:
        mu (torch.Tensor float32[N]): the location of the peaks
        lv (torch.Tensor float32[N]): the (log of the square of the) width of the peaks
        mz (torch.Tensor float32[B, L]): the masses for each spectra in the batch
        iv (torch.Tensor float32[B, L]): the intensities for each spectra in the batch
        compute_grad_spectra (bool): if True, computes the gradient w.r.t. mz and \
            iv, otherwise return None. Passing True raises a NotImplementedError, \
                the default value (False) saves a lot of memory.

    Raises:
        NotImplementedError if compute_grad_spectra is True

    Returns:
        torch.Tensor float32[B, N]: the value of the selected peaks for each spectra in the batch
    """
    return _PeakSenseFn.apply(mu, lv, mz, iv, compute_grad_spectra)  # type:ignore


class PeakSense(nn.Module):
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

        if n_vals < 1:
            raise ValueError(f"{n_vals=!r} < 1 is invalid")

        means = torch.linspace(mz_min, mz_max, n_vals, dtype=torch.float32)
        if n_vals == 1:
            # in the case of a single value, prefer it to be centered
            means.fill_(0.5 * (mz_min + mz_max))

        # add a small amount of noise to the means
        noise = torch.empty_like(means)
        noise.normal_(0.0, 0.5 * std_dev)
        means.add_(noise)

        log_vars = torch.empty_like(means)
        log_vars.fill_(2.0 * log(std_dev))

        self.mu = nn.Parameter(means)
        self.lv = nn.Parameter(log_vars)

    def forward(self, masses: torch.Tensor, intensities: torch.Tensor):
        has_batch = masses.ndim == 2
        if not has_batch:
            masses = masses.unsqueeze(0)
            intensities = intensities.unsqueeze(0)

        peaks = peak_sense(
            self.mu,
            self.lv,
            masses,
            intensities,
            compute_grad_spectra=False,  # avoid computing gradients wrt the inputs (too expensive and useless)
        )

        if not has_batch:
            peaks = peaks.squeeze(0)
        return peaks

    # pure Pytorch code below, very memory inefficient (diff2[B, N, L])

    # def forward(self, masses: torch.Tensor, intensities: torch.Tensor):
    #     mz_ = masses.unsqueeze(-2)  # [B?, 1, L]
    #     int_ = intensities.unsqueeze(-2)  # [B?, 1, L]
    #     mu_ = self.mu.view(1, self.mu.shape[0], 1)  # [1, N, 1]
    #     lv_ = self.lv.view(1, self.lv.shape[0], 1)
    #     if mz_.ndim == 2:
    #         mu_ = mu_.squeeze(0)  # [N, 1]
    #         lv_ = lv_.squeeze(0)

    #     diff2 = torch.square(mu_ - mz_)  # [B?, N, L]
    #     sigma2i = torch.exp(-lv_)  # [B?, N, 1]
    #     exp = torch.exp(-0.5 * diff2 * sigma2i)  # [B?, N, L]

    #     # prevent gradient flow for low weights
    #     weighted_int = torch.where(exp >= 1e-4, exp * int_, 0.0)  # [B?, N, L]
    #     weighted_sum = torch.sum(weighted_int, dim=-1)  # [B?, N]

    #     return weighted_sum

    def overlap_metric(self, alpha: float):
        "compute the average overlap between selected peaks"

        if self.mu.nelement() == 1:
            return torch.mean(self.mu * 0.0)

        # only neighboring values
        mu_sorted, sort_idx = torch.sort(self.mu)
        as_sorted = alpha * torch.exp(0.5 * self.lv[sort_idx])
        mu_l, mu_r = mu_sorted[:-1], mu_sorted[1:]
        as_l, as_r = as_sorted[:-1], as_sorted[1:]

        # end of left - start of right, if positive
        overlap = torch.clamp((mu_l + as_l) - (mu_r - as_r), min=0.0)
        return torch.mean(overlap)

    # def overlap_metric(self, alpha: float):
    #     "compute the average overlap between selected peaks"

    #     if self.mu.nelement() == 1:
    #         return torch.mean(self.mu * 0.0)

    #     # every possible combination, as pairs of vectors of shape (N*(N-1)/2,)
    #     mu_l, mu_r = torch.chunk(torch.combinations(self.mu, r=2), 2, dim=1)
    #     lv_l, lv_r = torch.chunk(torch.combinations(self.lv, r=2), 2, dim=1)
    #     as_l, as_r = alpha * torch.exp(0.5 * lv_l), alpha * torch.exp(0.5 * lv_r)

    #     # lowest ends - highest start, if positive
    #     overlap = torch.clamp(
    #         torch.min(mu_l + as_l, mu_r + as_r) - torch.max(mu_l - as_l, mu_r - as_r),
    #         min=0.0,
    #     )

    #     return torch.mean(overlap)

    def width_metric(self, alpha: float, f: float):
        pred = 2 * alpha * torch.exp(0.5 * self.lv)
        gt = f * torch.square(self.mu.detach())
        return torch.abs(pred - gt).mean()


class ThreshCounter(nn.Module):
    """This layer applies a learnable threshold on the peaks, it should be placed
    before any classifying head."""

    def __init__(self, n_peaks: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((n_peaks,), dtype=torch.float32))

    def forward(self, peak_values: torch.Tensor):
        return sigmoid(peak_values - self.bias)
