"""We need to integrate something like this in omezarrmsi,
map_ds: (OMEZarrMSI, (mzs, int) -> (mzs, int)) -> OMEZarrMSI
    - work chunk by chunks (different chunks in parallel ?)
    - support for processed and continuous modes (?)
    - accepts upper bound on C-length (for storage) + auto-resize C axis  after
    - could be used for normalization, deisotoping, binning, ...
    - look at the latest binning function, might provide insight for something generic
"""

# %%

import pathlib
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import omezarrmsi as omz
import pyopenms as oms
from omezarrmsi.plots.mz_slice import mz_slice
from omezarrmsi.utils.axial import AxialMapping, Axis
from scipy.interpolate import interp1d

# %%

slim_dir = pathlib.Path.home() / "datasets" / "COMULIS-slim-msi"
files = {
    "r13": ["region13_317norm_sample.zarr", "region13_nonorm_sample.zarr"],
    "r14": ["region14_317norm_sample.zarr", "region14_nonorm_sample.zarr"],
    "r15": ["region15_317norm_sample.zarr", "region15_nonorm_sample.zarr"],
}
zf = omz.OMEZarrMSI(slim_dir / files["r13"][0], mode="r")

for n1, n2 in files.values():
    assert (slim_dir / n1).is_dir()
    assert (slim_dir / n2).is_dir()

# %%


def nth_coordinates(y: np.ndarray, x: np.ndarray, *, n: int = 20):
    for idx, (y_, x_) in enumerate(zip(y, x)):
        if idx == n:
            break
        yield AxialMapping({Axis.Z: 0, Axis.Y: int(y_), Axis.X: int(x_)})


def avg_ms_i(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    mz_lo_: float,
    mz_hi_: float,
    bin_size: float,
    kind="linear",
):
    "this is an approximation that should be fast enough and useful enough"
    domain = np.linspace(mz_lo_, mz_hi_, int(np.ceil((mz_hi_ - mz_lo_) / bin_size)))
    sum_ = np.zeros_like(domain)

    for s_mzs_, s_int_ in spectra:
        if s_mzs_.size == 0:
            continue
        sum_ += interp1d(
            s_mzs_,
            s_int_,
            kind=kind,
            copy=False,
            assume_sorted=True,
            bounds_error=False,
        )(domain)

    return domain, sum_ / len(spectra)


def avg_ms_ws(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    mz_lo_: float,
    mz_hi_: float,
    bin_size: float,
    std_dev: float,
):
    "this assumes a small domain (mz_hi_ - mz_lo_) because the bin size is constant"
    domain = np.linspace(mz_lo_, mz_hi_, int(np.ceil((mz_hi_ - mz_lo_) / bin_size)))
    sum_ = np.zeros_like(domain)

    # stuff for Gaussian PDF
    pre_f = 1 / (std_dev * np.sqrt(2 * np.pi))
    exp_f = -0.5 / std_dev**2
    mu_ = np.expand_dims(domain, -1)

    def _process(s_mzs__, s_int__):
        s_mzs__ = np.expand_dims(s_mzs__, -2)
        diff2 = (s_mzs__ - mu_) ** 2
        weights = pre_f * np.exp(exp_f * diff2)
        return np.dot(weights, s_int__)

    for s_mzs_, s_int_ in spectra:
        mask_ = (s_mzs_ >= mz_lo_) & (s_mzs_ <= mz_hi_)
        ws_int = _process(s_mzs_[mask_], s_int_[mask_])
        sum_ += ws_int

    return domain, sum_ / len(spectra)


# %%


def avg_ms_hist(
    spectra: list[tuple[np.ndarray, np.ndarray]],
    mz_lo_: float,
    mz_hi_: float,
    bin_size: float,
):
    """Use binning to approximate the average spectrum of the dataset

    this assumes a small domain (mz_hi_ - mz_lo_) because the bin size is constant

    Returns two vectors of the same length
        - center of each bin
        - right edge of each bin
        - value of each bin
    """
    n_bins = int(np.ceil((mz_hi_ - mz_lo_) / bin_size))
    domain = np.linspace(mz_lo_, mz_hi_, n_bins)
    sum_ = np.zeros_like(domain)
    total_ = np.zeros_like(domain)

    for s_mzs_, s_int_ in spectra:
        mask_ = (s_mzs_ >= mz_lo_) & (s_mzs_ <= mz_hi_)
        bin_indices = np.digitize(s_mzs_[mask_], domain)
        # no check on bin_indices, we assume it's all good
        sum_[bin_indices] += s_int_[mask_]
        total_[bin_indices] += 1.0

    # sum_[0] is the sum of all ions with m/z < domain[0]
    # sum_[i] is the sum of all ions with m/z in the range [domain[i-1], domain[i]]

    # domain[i] is the right bound of the bin (i) with value sum_[i]

    return domain - 0.5 * bin_size, domain, sum_ / np.maximum(total_, 1e-5)


# %%


def deisotoping(s_mzs_: np.ndarray, s_int_: np.ndarray, tol: float):
    s = oms.MSSpectrum()
    s.set_peaks((s_mzs_, s_int_))  # type:ignore
    oms.Deisotoper.deisotopeAndSingleCharge(  # type:ignore
        spectra=s,
        fragment_tolerance=tol,  # less than 100 if ppm, less than 0.1 if Da
        fragment_unit_ppm=False,  # True -> ppm ; False -> Da
        min_charge=1,
        max_charge=1,
        keep_only_deisotoped=False,
        min_isopeaks=3,
        max_isopeaks=10,
        make_single_charged=True,
        annotate_charge=False,
        annotate_iso_peak_count=False,
        use_decreasing_model=True,
        start_intensity_check=2,
        add_up_intensity=False,
    )
    d_: tuple[np.ndarray, np.ndarray] = s.get_peaks()  # type:ignore
    return d_


# %%

masses = [732.5545, 733.55831, 734.56083, 734.56974]
mz_tol = 2e-3

# %%

fig, axes = plt.subplots(len(masses), 1)

for idx, mass in enumerate(masses):
    bg, ion_img = mz_slice(zf, mass - mz_tol, mass + mz_tol)
    ion_img[~bg] = np.nan
    axes[idx].imshow(ion_img)
    axes[idx].set_axis_off()

fig.tight_layout()

# %%

mz_lo, mz_hi = min(masses) - 0.1, max(masses) + 0.1
coords = AxialMapping({Axis.Z: 0, Axis.X: 247, Axis.Y: 19})
coords = AxialMapping({Axis.Z: 0, Axis.X: 84, Axis.Y: 15})

# select coordinates of interest
bg, ion_img = mz_slice(zf, masses[0] - mz_tol, masses[0] + mz_tol)

fig, ax = plt.subplots(4, figsize=(11, 8))
ion_img[~bg] = np.nan
ax[0].imshow(ion_img)
ax[0].plot([coords[Axis.X]], [coords[Axis.Y]], "r-x")

((s_mzs, s_int),) = zf.fetch_spectra(coords)
markerline, _, __ = ax[1].stem(s_mzs, s_int)
markerline.set_markerfacecolor("none")
ax[1].set_yscale("log")
ax[1].set_xlim((mz_lo, mz_hi))
ax[1].set_ylim((1e2, 5e7))

d_mzs, d_int = deisotoping(s_mzs, s_int, 1e-3)

markerline, _, __ = ax[2].stem(d_mzs, d_int, "green")
markerline.set_markerfacecolor("none")
ax[2].set_yscale("log")
ax[2].set_xlim((mz_lo, mz_hi))
ax[2].set_ylim((1e2, 5e7))

# find the removed peaks

removed = ~np.in1d(s_mzs, d_mzs)

markerline, _, __ = ax[3].stem(s_mzs[removed], s_int[removed], "red")
markerline.set_markerfacecolor("none")
ax[3].set_yscale("log")
ax[3].set_xlim((mz_lo, mz_hi))
ax[3].set_ylim((1e2, 5e7))

fig.tight_layout()

# %%

print(f"{s_mzs[removed & (s_mzs > 734.0) & (s_mzs < 735.0)]=}")


# %%

# select coordinates of interest
bg, ion_img = mz_slice(zf, masses[0] - mz_tol, masses[0] + mz_tol)
y, x = np.unravel_index(np.argsort(-ion_img, axis=None), ion_img.shape)
coords_lst = list(nth_coordinates(y, x, n=-1))
spectra_lst = zf.fetch_spectra(*coords_lst)

# %%

de_tol = 1.5e-3  # tolerance of the deisotoping function
d_spectra_lst = [deisotoping(s_mzs_, s_int_, de_tol) for s_mzs_, s_int_ in spectra_lst]

# %%

bin_size = 2e-3
avg = partial(
    avg_ms_hist,
    mz_lo_=min(masses) - 0.5,
    mz_hi_=max(masses) + 0.5,
    bin_size=bin_size,
)

f_avg_mzs, bins_, f_avg_int = avg(spectra_lst)
d_avg_mzs, _, d_avg_int = avg(d_spectra_lst)

assert np.allclose(f_avg_mzs, d_avg_mzs)
diff = f_avg_int - d_avg_int

# assert (diff >= 0.0).all(), "deisotoping increased avg spectrum"
# # clip by 1.0 so that anything less doesn't affect the scale
# diff = np.maximum(diff, 1.0)


# %%

for xlim in [
    # {"left": min(masses) - 0.25, "right": max(masses) + 0.25},
    {"left": 732.54, "right": 732.57},  # around the 1st isotope
    {"left": 733.54, "right": 733.58},  # around the 2nd isotope
    {"left": 734.54, "right": 734.59},  # around the 3rd isotope + next (close) mass
]:
    fig, ax = plt.subplots(3, figsize=(11, 6), sharex=True, sharey=True)
    ax[0].plot(f_avg_mzs, f_avg_int, alpha=0.5)
    ax[0].set_title("Before desitotoping")

    ax[1].plot(d_avg_mzs, d_avg_int, "green", alpha=0.5)
    ax[1].set_title(f"After deisotoping (tol={de_tol:.1e})")

    ax[2].plot(f_avg_mzs, diff, "red", alpha=0.5)
    ax[2].set_title(f"Removed amount (tol={de_tol:.1e})")

    for ax_ in ax:
        # ax_.set_yscale("log")
        ax_.set_xlim(**xlim)

        # show a grid with all bins ticks if the range is not too large
        ticks = bins_[(f_avg_mzs > xlim["left"]) & (f_avg_mzs < xlim["right"])]
        # if ticks.size < 15:
        ax_.set_xticks(ticks)
        for label in ax_.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax_.grid(which="both", axis="x")

        # no "+7.3405e2"
        ax_.ticklabel_format(axis="x", useOffset=False)

        # show the masses
        # for m in masses:
        #     ax_.axvline(m, 0.0, 0.25, c="black")

    fig.tight_layout()

"""
The value fragment_tolerance=2e-3 was found by adjusting the parameter until
the 2nd and 3rd isotopes of m^1_1=732.5545 (m^1_2=733.55831 and m^1_3=734.56083,
respectively) were removed while the closest ion to the third isotope, m^2=734.56974,
remained in the dataset.
"""

# %% store the data after deisotoping

dest_dir = slim_dir.parent / f"slim-deisotoping-{de_tol:.1e}"
dest_dir.mkdir(exist_ok=False, parents=False)  # don't override implicitly

for r_key, r_files in files.items():
    for f_ in r_files:
        zf = omz.OMEZarrMSI(slim_dir / f_, mode="r")
        y, x = zf.z_len[0, 0].nonzero()
        coords_lst = list(nth_coordinates(y, x, n=-1))
        spectra_lst = zf.fetch_spectra(*coords_lst)
        de_spec_lst = [deisotoping(m, i, de_tol) for m, i in spectra_lst]

        # get new C-axis
        max_len = max(m.size for m, _ in de_spec_lst)
        dst = omz.OMEZarrMSI.create(
            dest_dir / f_,
            zf.int_shape.with_update(c=max_len),
            zf.z_int.dtype,
            zf.z_mzs.dtype,
            omz.ImzMLBinaryMode.PROCESSED,
            # leave the rest to defaults
        )

        # no need to update the metadata yet

        # copy data
        dst.set_spectra(dict(zip(coords_lst, de_spec_lst, strict=True)))

        # copy labels
        for lbl in zf.labels():
            dst.add_label(lbl, zf.get_label(lbl)[...], *zf.int_shape.keys())

# %%
