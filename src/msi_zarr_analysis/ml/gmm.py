"""
gmm: Gaussian Mixture Models for MSI & associated utilities
"""

import math
import warnings  # noqa: F401

import torch
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.core.errors import NumbaPerformanceWarning  # noqa:F401
from torch import nn
from torch.nn.functional import softmax

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# I honestly have no idea of how to fix that, this gives good enough perf on the laptop
THREADS_PER_BLOCK: int = 128
HALF_LOG_2_PI: float = 0.5 * math.log(2 * math.pi)
INV_SQRT_2_PI: float = math.pow(2 * math.pi, -0.5)


def kern(size: int):
    "build the invocation tuple for a given size with THREADS_PER_BLOCK"
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"{size=!r} must be a non negative integer")
    block_per_grid = (size + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    return block_per_grid, THREADS_PER_BLOCK


def _prob_to_nll(prob: torch.Tensor):
    "map a likelihood probability to a negative log likelihood"
    return -torch.log(1e-6 + prob)


@cuda.jit(
    [
        "void(f4[:], f4[:], f4[:], f4[:], f4[:])",
        "void(f8[:], f8[:], f8[:], f8[:], f8[:])",
    ]
)
def _nb_gmm_prob(
    pi: DeviceNDArray,
    mu: DeviceNDArray,
    lv: DeviceNDArray,
    values: DeviceNDArray,
    prob_out: DeviceNDArray,
):
    "compute sum_i pi_i p(v | mu_i, s2_i)"

    tid: int = cuda.threadIdx.x  # type: ignore
    idx: int = cuda.blockIdx.x * cuda.blockDim.x + tid  # type: ignore

    if idx < prob_out.shape[0]:
        ws = 0.0
        for i in range(mu.shape[0]):
            arg = -0.5 * (mu[i] - values[idx]) ** 2 * math.exp(-lv[i])
            ws += pi[i] * INV_SQRT_2_PI * math.exp(arg) * math.exp(-0.5 * lv[i])
        prob_out[idx] = ws


@cuda.jit(
    [
        "void(f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], f4[:])",
        "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])",
    ]
)
def _nb_gmm_prob_backward(
    pi: DeviceNDArray,
    mu: DeviceNDArray,
    lv: DeviceNDArray,
    values: DeviceNDArray,
    g_prob: DeviceNDArray,
    g_pi_out: DeviceNDArray,
    g_mu_out: DeviceNDArray,
    g_lv_out: DeviceNDArray,
):
    "compute partial(L, z) given partial(L, p) and z for z in [pi, mu, lv]"

    tid: int = cuda.threadIdx.x  # type: ignore
    idx: int = cuda.blockIdx.x * cuda.blockDim.x + tid  # type: ignore

    if idx < pi.shape[0]:
        ws_pi = ws_mu = ws_lv = 0.0
        for i in range(values.shape[0]):
            arg = -0.5 * (mu[idx] - values[i]) ** 2 * math.exp(-lv[idx])
            density = INV_SQRT_2_PI * math.exp(arg) * math.exp(-0.5 * lv[idx])

            ws_pi += g_prob[i] * density
            ws_mu -= (
                g_prob[i]
                * density
                * pi[idx]
                * (mu[idx] - values[i])
                * math.exp(-lv[idx])
            )
            ws_lv += (
                g_prob[i]
                * density
                * pi[idx]
                * ((mu[idx] - values[i]) ** 2 * math.exp(-lv[idx]) - 1.0)
            )

        g_pi_out[idx] = ws_pi
        g_mu_out[idx] = ws_mu
        g_lv_out[idx] = 0.5 * ws_lv


class _GMMProbFn(torch.autograd.Function):
    "torch autograd Function to compute the prob density using a GMM in 1D"

    @staticmethod
    def forward(
        pi: torch.Tensor,
        mu: torch.Tensor,
        lv: torch.Tensor,
        mz: torch.Tensor,
        _: bool,  # see setup context
    ):

        prob = torch.empty(mz.shape, dtype=pi.dtype, device=pi.device)

        # parallel over the number of items
        _nb_gmm_prob[kern(prob.shape[0])](  # type: ignore
            cuda.as_cuda_array(pi.detach()),
            cuda.as_cuda_array(mu.detach()),
            cuda.as_cuda_array(lv.detach()),
            cuda.as_cuda_array(mz.detach()),
            cuda.as_cuda_array(prob),
        )

        return prob

    @staticmethod
    def setup_context(ctx, inputs: tuple, output):
        pi: torch.Tensor
        mu: torch.Tensor
        lv: torch.Tensor
        mz: torch.Tensor
        pi, mu, lv, mz, compute_grad_spectra = inputs

        ctx.save_for_backward(pi, mu, lv, mz)
        ctx.compute_grad_spectra = compute_grad_spectra

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_prob: torch.Tensor):
        pi: torch.Tensor
        mu: torch.Tensor
        lv: torch.Tensor
        mz: torch.Tensor
        pi, mu, lv, mz = ctx.saved_tensors

        if ctx.compute_grad_spectra:
            raise NotImplementedError("full gradient hasn't been implemented yet")

        g_pi = torch.empty(pi.shape, dtype=pi.dtype, device=pi.device)
        g_mu = torch.empty(mu.shape, dtype=mu.dtype, device=mu.device)
        g_lv = torch.empty(lv.shape, dtype=lv.dtype, device=lv.device)

        # parallel over the number of components
        _nb_gmm_prob_backward[kern(pi.shape[0])](  # type: ignore
            cuda.as_cuda_array(pi.detach()),
            cuda.as_cuda_array(mu.detach()),
            cuda.as_cuda_array(lv.detach()),
            cuda.as_cuda_array(mz.detach()),
            cuda.as_cuda_array(g_prob.detach()),
            cuda.as_cuda_array(g_pi),
            cuda.as_cuda_array(g_mu),
            cuda.as_cuda_array(g_lv),
        )

        return g_pi, g_mu, g_lv, None, None, None


def gmm_prob(
    pi: torch.Tensor,
    mu: torch.Tensor,
    lv: torch.Tensor,
    mz: torch.Tensor,
    *,
    compute_grad_inputs: bool = False,
) -> torch.Tensor:
    """Compute the PDF of a GMM

    Args:
        pi (torch.Tensor): the weight of each component
        mu (torch.Tensor): the mean of each component
        lv (torch.Tensor): the log of the variance
        mz (torch.Tensor): the inputs to compute the PDF for
        compute_grad_inputs (bool, optional): keeping track of mz for gradient computation. Defaults to False.
    """
    return _GMMProbFn.apply(pi, mu, lv, mz, compute_grad_inputs)  # type: ignore


class GMM1D(nn.Module):
    # TODO doc
    def __init__(
        self,
        components: int,
        mz_min: float,
        mz_max: float,
        std_dev: float | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ):
        super().__init__()

        if std_dev is None:
            std_dev = 0.5 * (mz_max - mz_min) / components

        self.components = components

        means = torch.linspace(mz_min, mz_max, components, dtype=dtype)
        if components == 1:
            # in the case of a single value, prefer it to be centered
            means.fill_(0.5 * (mz_min + mz_max))

        # add a small amount of noise to the means
        noise = torch.empty_like(means)
        noise.normal_(0.0, 0.5 * std_dev, generator=generator)
        means.add_(noise)

        log_vars = torch.empty_like(means)
        log_vars.fill_(2.0 * math.log(std_dev))

        pi_logits = torch.zeros_like(means)

        self.mu = nn.Parameter(means)
        self.lv = nn.Parameter(log_vars)
        self.pi_l = nn.Parameter(pi_logits)

    def prob(self, inputs: torch.Tensor):
        return gmm_prob(
            softmax(self.pi_l, 0),
            self.mu,
            self.lv,
            inputs,
            compute_grad_inputs=False,
        )

    def neg_log_likelihood(self, inputs: torch.Tensor) -> torch.Tensor:
        "log likelihood of a batch of inputs"

        return _prob_to_nll(self.prob(inputs))

    def ws_neg_log_likelihood(
        self, inputs: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        "weighted sum of the negative log likelihood for all inputs"

        return torch.sum(weights * self.neg_log_likelihood(inputs))


class GMM1DCls(nn.Module):
    # TODO doc
    def __init__(
        self,
        components: int,
        mz_min: float,
        mz_max: float,
        std_dev: float | None = None,
    ):
        super().__init__()

        self.pos_head = GMM1D(components, mz_min, mz_max, std_dev)
        self.neg_head = GMM1D(components, mz_min, mz_max, std_dev)

    def ratio_max(self, batch: torch.Tensor):
        # TODO doc
        llh_pos = self.pos_head.neg_log_likelihood(batch)
        llh_neg = self.neg_head.neg_log_likelihood(batch)

        return torch.exp(torch.abs(llh_pos - llh_neg))

    def ratio_min(self, batch: torch.Tensor):
        # TODO doc
        llh_pos = self.pos_head.neg_log_likelihood(batch)
        llh_neg = self.neg_head.neg_log_likelihood(batch)

        return 1.0 - torch.exp(-torch.abs(llh_pos - llh_neg))
