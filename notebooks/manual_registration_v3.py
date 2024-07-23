# %%

import pathlib
from functools import partial
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cytomine import Cytomine
from matplotlib import colors
from omezarrmsi import OMEZarrMSI

# from omezarrmsi.plots.tic import tic_array
from omezarrmsi.plots.mz_slice import mz_slice
from rasterio.features import rasterize
from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import convex_hull_image
from shapely import Polygon, affinity
from skimage.color import rgba2rgb
from valis.warp_tools import warp_img_inv

import sys

sys.path.insert(0, str((pathlib.Path(__file__).parent.parent).resolve()))

from msi_zarr_analysis.ml.dataset.translate_annotation import (  # noqa: E402
    ParsedAnnotation,
)
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (  # noqa: E402
    get_overlay_annotations,
)


# %%


def _read_tiff(path: pathlib.Path, page_idx: int) -> np.ndarray:
    img = tifffile.TiffFile(path)
    return img.pages[page_idx].asarray().transpose((1, 2, 0)) / 255.0  # type:ignore


def _read_png(path: pathlib.Path) -> np.ndarray:
    return np.asarray(rgba2rgb(Image.open(path)))


def up_scaler(img_: np.ndarray, up: int = 16):
    return img_.repeat(up, axis=0).repeat(up, axis=1)


# %%

v1_parent = pathlib.Path(__file__).parent.parent / "datasets" / "masks-images"

imgs_v1 = {
    "r13": _read_tiff(v1_parent / "overlay-r13.tif", 6),  # PLPC
    "r14": _read_tiff(v1_parent / "mask-r14.tif", 8),  # PLPC
    "r15": _read_tiff(v1_parent / "mask-r15.tif", 8),  # PLPC
}

# %%

fig, axes = plt.subplots(1, len(imgs_v1), squeeze=False, figsize=(9, 9))
for idx, (key, image) in enumerate(imgs_v1.items()):
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()


# %%

v2_parent = pathlib.Path.home() / "datasets" / "comulis-masks-v2"

imgs_v2 = {
    "r13": _read_png(v2_parent / "region13_purplemask_v2.png"),
    "r14": _read_png(v2_parent / "region14_purplemask_v2.png"),
    "r15": _read_png(v2_parent / "region15_purplemask_v2.png"),
}

# %%

fig, axes = plt.subplots(1, len(imgs_v2), squeeze=False, figsize=(9, 9))
for idx, (key, image) in enumerate(imgs_v2.items()):
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()

# %%


def _filter_purple(img: np.ndarray) -> np.ndarray:
    "img[H, W, RGB]"
    # select [0.2627, 0.0, 0.3255]
    return (
        (img[..., 0] > 0.25)
        & (img[..., 0] < 0.27)  # noqa:W503
        & (img[..., 1] < 0.02)  # noqa:W503
        & (img[..., 2] > 0.32)  # noqa:W503
        & (img[..., 2] < 0.34)  # noqa:W503
    )


def mask_ms_shadow(img: np.ndarray):
    filled_mask: np.ndarray = binary_fill_holes(_filter_purple(img))  # type:ignore
    # count labels
    labels: np.ndarray
    labels, _ = label(filled_mask)  # type:ignore
    label_val, label_cnt = np.unique(labels, return_counts=True)
    for idx in range(len(label_cnt)):
        if not filled_mask[labels == idx].any():
            label_cnt[idx] = 0
    sample_idx = np.argmax(label_cnt)

    return labels == label_val[sample_idx]


# %%

fig, axes = plt.subplots(
    1,
    len(imgs_v2),
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, image) in enumerate(imgs_v2.items()):
    image = image.copy()
    image[~mask_ms_shadow(image), :] = 0.3
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()


# %% get gray background in v1


def _filter_grey(img: np.ndarray) -> np.ndarray:
    "img[H, W, RGB]"

    # get min(green, blue) > 0.2
    return img[..., 1] > 0.3


def mask_bf_background(img: np.ndarray):
    filled_mask: np.ndarray = convex_hull_image(_filter_grey(img))  # type:ignore
    return filled_mask > 0


# %%

fig, axes = plt.subplots(
    1,
    len(imgs_v1),
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, image) in enumerate(imgs_v1.items()):
    image = image.copy()
    image[~mask_bf_background(image), :] = 0.3
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()

# %%

fig, axes = plt.subplots(
    1,
    len(imgs_v2),
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, image) in enumerate(imgs_v2.items()):
    image = image.copy()
    image[~mask_bf_background(image), :] = 0.3
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()


# %%

ms_ds_dir = pathlib.Path("/home/maxime/datasets/COMULIS-slim-msi")

ms_dict = {
    "r13": OMEZarrMSI(ms_ds_dir / "region13_nonorm_sample.zarr"),
    "r14": OMEZarrMSI(ms_ds_dir / "region14_nonorm_sample.zarr"),
    "r15": OMEZarrMSI(ms_ds_dir / "region15_nonorm_sample.zarr"),
}

assert sorted(ms_dict) == sorted(imgs_v2)

# %%


def mask_ms_footprint(ms: OMEZarrMSI):
    val_len: np.ndarray = ms.z_len[0, 0] > 0  # type:ignore
    return val_len


# %%

fig, axes = plt.subplots(
    len(ms_dict),
    1,
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, ms) in enumerate(ms_dict.items()):
    image = mask_ms_footprint(ms)
    axes[idx, 0].imshow(image)
    axes[idx, 0].set_title(key)
    axes[idx, 0].set_axis_off()
fig.tight_layout()

# %%

plt.imshow(imgs_v1["r13"])
plt.gca().set_axis_off()
plt.figure()
plt.imshow(imgs_v2["r13"])
plt.gca().set_axis_off()
plt.figure()
plt.imshow(mask_ms_footprint(ms_dict["r13"]))
plt.gca().set_axis_off()

# %% compute transform


def _nz_dims(mask: np.ndarray):
    y, x = mask.nonzero()
    assert y.size > 0, "empty mask"

    return (
        y.max() + 1 - y.min(),
        x.max() + 1 - x.min(),
    )


def _nz_offset(mask: np.ndarray):
    y, x = mask.nonzero()
    assert y.size > 0, "empty mask"

    return y.min(), x.min()


def scale_factor(src: np.ndarray, dst: np.ndarray):
    src_h, src_w = _nz_dims(src)
    dst_h, dst_w = _nz_dims(dst)

    if dst_w > dst_h:
        return dst_w / src_w
    return dst_h / src_h


# %% transform matrices to be composted


def matrix_from_list(m_list: list[float]):
    assert len(m_list) == 6
    matrix = [
        [m_list[0], m_list[1], m_list[4]],
        [m_list[2], m_list[3], m_list[5]],
        [0, 0, 1],
    ]
    return np.array(matrix, np.float64)


def matrix_to_list(matrix: np.ndarray):
    "prepare for argument list to shapely.affinity.affine_transform"
    return [
        matrix[0, 0],
        matrix[0, 1],
        matrix[1, 0],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    ]


def matrix_offset(xoff: float, yoff: float):
    return matrix_from_list([1, 0, 0, 1, xoff, yoff])


def matrix_scale(f: float):
    return matrix_from_list([f, 0, 0, f, 0, 0])

# %%


def get_map_matrix(
    bf_mask_v1: np.ndarray,
    bf_mask_v2: np.ndarray,
    shadow_v2: np.ndarray,
    footprint_msi: np.ndarray,
):

    bf_mask_v1_rf = np.rot90(np.flip(bf_mask_v1, axis=1), k=-1, axes=(0, 1))
    del bf_mask_v1  # make sure I'm not using it again
    s_1_to_2 = scale_factor(bf_mask_v1_rf, bf_mask_v2)
    s_sha_to_msi = scale_factor(shadow_v2, footprint_msi)

    y0_1m, x0_1m = _nz_offset(bf_mask_v1_rf)

    y0_2s, x0_2s = _nz_offset(shadow_v2)
    y0_2m, x0_2m = _nz_offset(bf_mask_v2)

    # the steps should be read from bottom-up as the last item correspond to
    # the rightmost factor in the matrix product.
    matrix = (
        # scale for MSI axes
        matrix_scale(s_sha_to_msi)
        # offset (border-to-shadow minus border-to-background)
        @ matrix_offset(-(x0_2s - x0_2m), -(y0_2s - y0_2m))
        # scale for v2 axes
        @ matrix_scale(s_1_to_2)
        # offset for the bright-field background
        @ matrix_offset(-x0_1m, -y0_1m)
        # flip and rotate
        @ matrix_from_list(
            [0, -1, -1, 0, bf_mask_v1_rf.shape[1], bf_mask_v1_rf.shape[0]]
        )
    )

    return matrix


def map_geometry_fn_v2(
    bf_mask_v1: np.ndarray,
    bf_mask_v2: np.ndarray,
    shadow_v2: np.ndarray,
    footprint_msi: np.ndarray,
):
    return partial(
        affinity.affine_transform,
        matrix=matrix_to_list(
            get_map_matrix(bf_mask_v1, bf_mask_v2, shadow_v2, footprint_msi)
        ),
    )


# %% load overlays

overlays_dict = {
    "r13": _read_tiff(v1_parent / "overlay-r13.tif", 6),  # PLPC
    "r14": _read_tiff(v1_parent / "overlay-r14.tif", 8),  # PLPC
    "r15": _read_tiff(v1_parent / "mask-r15.tif", 8),  # PLPC
}

assert sorted(overlays_dict.keys()) == sorted(imgs_v1.keys())

if any(overlay.shape != imgs_v1[key].shape for key, overlay in overlays_dict.items()):
    raise ValueError("size mismatch")

# %% show mapped polygon

annotation = Polygon(
    [(275, 350), (275, 600), (375, 600), (375, 550), (325, 550), (325, 350)]
)

key = "r13"

fig, axes = plt.subplots(1, 3, squeeze=False)

axes[0, 0].imshow(rasterize([annotation], imgs_v1[key].shape[:2]))

fn_1 = map_geometry_fn_v2(
    mask_bf_background(imgs_v1[key]),
    mask_bf_background(imgs_v2[key]),
    mask_ms_shadow(imgs_v2[key]),
    mask_ms_footprint(ms_dict[key]),
)

axes[0, 1].imshow(rasterize([fn_1(annotation)], mask_ms_footprint(ms_dict[key]).shape))

matrix = get_map_matrix(
    mask_bf_background(imgs_v1[key]),
    mask_bf_background(imgs_v2[key]),
    mask_ms_shadow(imgs_v2[key]),
    mask_ms_footprint(ms_dict[key]),
)

fn_2 = partial(affinity.affine_transform, matrix=matrix_to_list(matrix))

axes[0, 2].imshow(rasterize([fn_2(annotation)], mask_ms_footprint(ms_dict[key]).shape))

fig.tight_layout()

# %% upscale after registration : poor quality images due to the low resolution

upscale = 10
wrapped_v1: np.ndarray = warp_img_inv(  # type:ignore
    imgs_v1[key],
    M=matrix,
    src_shape_rc=mask_ms_footprint(ms_dict[key]).shape,
    interp_method="nearest",
)
wrapped_v1 = up_scaler(wrapped_v1, upscale)

fig = plt.figure(frameon=False, figsize=(24, 24))
fig.set_size_inches(wrapped_v1.shape[1] / 50, wrapped_v1.shape[0] / 50)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(wrapped_v1)
ax.set_axis_off()

# %% something similar with the lipids

fig = plt.figure(frameon=False, figsize=(24, 24))
fig.set_size_inches(wrapped_v1.shape[1] / 50, wrapped_v1.shape[0] / 50)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

mz_mean = 758.56943
mz_tol = 0.00266
_, data = mz_slice(ms_dict[key], mz_mean - mz_tol, mz_mean + mz_tol)
data[~mask_ms_footprint(ms_dict[key])] = np.nan

ax.imshow(data)
ax.set_axis_off()


# %% register with up-scaled MSI

key = "r15"

matrix = get_map_matrix(
    mask_bf_background(imgs_v1[key]),
    mask_bf_background(imgs_v2[key]),
    mask_ms_shadow(imgs_v2[key]),
    up_scaler(mask_ms_footprint(ms_dict[key])),
)

wrapped_v1 = warp_img_inv(  # type:ignore
    overlays_dict[key],
    M=matrix,
    src_shape_rc=up_scaler(mask_ms_footprint(ms_dict[key])).shape,
    interp_method="nearest",
)


fig = plt.figure(frameon=False, figsize=(24, 24))
fig.set_size_inches(wrapped_v1.shape[1] / 50, wrapped_v1.shape[0] / 50)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(wrapped_v1)
ax.set_axis_off()


mz_mean = 758.56943
mz_tol = 0.00266
_, data = mz_slice(ms_dict[key], mz_mean - mz_tol, mz_mean + mz_tol)
data[~mask_ms_footprint(ms_dict[key])] = np.nan

fig = plt.figure(frameon=False, figsize=(24, 24))
fig.set_size_inches(wrapped_v1.shape[1] / 50, wrapped_v1.shape[0] / 50)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(data)
ax.set_axis_off()

# %% fetch annotations

try:
    annotation_dict  # type:ignore
    raise RuntimeError("this should not be ran twice !")
except NameError:
    pass

project_id = 542576374
classes = {
    "ls+": [544926081],
    "ls-": [544926097],
    "sc+": [544924846],
    "sc-": [544926052],
}

image_id_dict = {"r13": 545025763, "r14": 548365416, "r15": 548365463}

# fetch annotations, rasterize them on the original image, translate them, rasterize them on the TIC
with Cytomine(
    "https://research.cytomine.be",
    "9245ed4e-980b-497c-b305-6c24c3143c3b",
    "e78600fa-a541-4ec5-8e78-c75ea4fd4fc0",
):
    annotation_dict = {
        key: get_overlay_annotations(
            project_id=project_id,
            image_id=image_id,
            classes=classes,
            select_users=(),
        )
        for key, image_id in image_id_dict.items()
    }

# %% map annotation dict

mapped_ann_dict = {
    r_key: {
        cls_key: [
            ParsedAnnotation(p_a.annotation, map_fn_(p_a.geometry)) for p_a in ann_lst
        ]
        for cls_key, ann_lst in ann_dict.items()
    }
    for r_key, ann_dict in annotation_dict.items()
    if (
        map_fn_ := map_geometry_fn_v2(
            mask_bf_background(imgs_v1[r_key]),
            mask_bf_background(imgs_v2[r_key]),
            mask_ms_shadow(imgs_v2[r_key]),
            mask_ms_footprint(ms_dict[r_key]),
        )
    )
}

# %% prepare data for plots

key = "r13"
ms_mask_: np.ndarray = mask_ms_footprint(ms_dict[key])
image_size = (imgs_v2[key].shape[1] / 200.0, imgs_v2[key].shape[0] / 200.0)
ms_size = (
    mask_ms_footprint(ms_dict[key]).shape[1] / 20,
    mask_ms_footprint(ms_dict[key]).shape[0] / 20,
)
# _, ms_data = tic_array(ms_dict[key])
_, ms_data = mz_slice(ms_dict[key], 758.56943 - 0.00166, 758.56943 + 0.00166)

# %%

mz_mean = 758.56943
mz_tol = 0.00266
_, ms_data = mz_slice(ms_dict[key], mz_mean - mz_tol, mz_mean + mz_tol)
plt.imshow(ms_data)


# %% colors

color_lst = [
    np.array(colors.to_rgb("tab:olive")),
    np.array(colors.to_rgb("tab:red")),
    np.array(colors.to_rgb("tab:brown")),
    np.array(colors.to_rgb("tab:pink")),
]

# %% plot overlay for the image with rasterized annotations

fig = plt.figure(frameon=False)
fig.set_size_inches(*image_size)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(imgs_v1[key])

for idx, ann_lst in enumerate(annotation_dict[key].values()):
    raster = rasterize([a.geometry for a in ann_lst], imgs_v1[key].shape[:2])
    raster_rgba = np.concatenate(
        [
            np.multiply.outer(raster, color_lst[idx]),
            np.stack([raster * 0.8], axis=-1),
        ],
        axis=-1,
    )
    ax.imshow(raster_rgba)

# %% plot overlay for the ms with rasterized annotations

fig = plt.figure(frameon=False)
fig.set_size_inches(*ms_size)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

# shown_ms_data = np.stack(3 * [ms_data], axis=-1)
shown_ms_data = ms_data.copy()
shown_ms_data /= ms_data[ms_mask_].max()
shown_ms_data[~ms_mask_] = np.nan
ax.imshow(shown_ms_data)

for idx, ann_lst in enumerate(mapped_ann_dict[key].values()):
    raster = rasterize([a.geometry for a in ann_lst], ms_mask_.shape)
    raster_rgba = np.concatenate(
        [
            np.multiply.outer(raster, color_lst[idx]),
            np.stack([raster * 0.8], axis=-1),
        ],
        axis=-1,
    )
    ax.imshow(raster_rgba)


# %%

super_sampling: int = 16


def raster_float(
    ann_lst: list[ParsedAnnotation], shape: tuple[int, ...], s: int
) -> np.ndarray:
    geometries = [a.geometry for a in ann_lst]

    h, w = shape
    s_shape = (s * h, s * w)
    s_geoms = [affinity.affine_transform(g, [s, 0, 0, s, 0, 0]) for g in geometries]
    s_raster = rasterize(s_geoms, s_shape)  # [s*h, s*w]

    # down-sample
    assert s_raster.flags.c_contiguous
    return np.reshape(s_raster, (h, s, w, s)).mean(axis=(1, 3))


# %%

# also try : https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7

color_lst = [
    np.array(colors.to_rgb("#F0E442")),
    np.array(colors.to_rgb("#009E73")),
    np.array(colors.to_rgb("#56B4E9")),
    np.array(colors.to_rgb("#0072B2")),
]

fig = plt.figure(frameon=False, figsize=(24, 24))
fig.set_size_inches(*ms_size)
ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

# ax.imshow(np.where(ms_mask_, ms_data, np.nan))
# background = np.zeros(ms_mask_.shape + (3,))
# background[~ms_mask_, :] = 1.0
# ax.imshow(background)

shown_ms_data = ms_data.copy()
shown_ms_data /= ms_data[ms_mask_].max()
shown_ms_data[~ms_mask_] = np.nan
ax.imshow(shown_ms_data, cmap="copper")

for idx, ann_lst in enumerate(mapped_ann_dict[key].values()):
    raster = raster_float(ann_lst, ms_mask_.shape, super_sampling)
    # raster[raster < .5] = .0
    raster_rgba = np.concatenate(
        [
            np.multiply.outer(raster, color_lst[idx]),
            np.stack([raster * 1.0], axis=-1),
        ],
        axis=-1,
    )
    ax.imshow(raster_rgba)

# fig.tight_layout()


# %% show comparison between old and new labels : where are the differences, and by how much ?

rasterized_masks_dict = {
    key_: {
        cls: raster_float(
            ann_lst, mask_ms_footprint(ms_dict[key_]).shape, super_sampling
        )
        for cls, ann_lst in ann_cls_dict_.items()
    }
    for key_, ann_cls_dict_ in mapped_ann_dict.items()
}

fig, axes = plt.subplots(len(rasterized_masks_dict) * 4, 1, figsize=(12, 24))

for row_, (region_, mask_dict_) in enumerate(rasterized_masks_dict.items()):
    print(f"{region_}:")
    for col_, (cls_, new_) in enumerate(mask_dict_.items()):
        old_ = ms_dict[region_].get_label(cls_)[0, 0]
        diff_ = new_ - old_
        print(f"\t{cls_}: min={diff_.min():.3} max={diff_.max():.3}")
        diff_[~mask_ms_footprint(ms_dict[region_])] = np.nan
        im_ = axes[row_ * 4 + col_].imshow(diff_, vmin=-1.0, vmax=+1.0, cmap="coolwarm")
        axes[row_ * 4 + col_].set_axis_off()
        axes[row_ * 4 + col_].set_title(f"{region_}: {cls_}")

fig.colorbar(im_, ax=axes.ravel().tolist())

# %%

pkg_dir = pathlib.Path(__file__).parent.parent
home_ds_dir = pathlib.Path("/home/maxime/datasets/COMULIS-slim-msi")

all_version_ms_dict = {
    "r13": (
        OMEZarrMSI(pkg_dir / "tmp_r13.zarr", mode="r+"),
        OMEZarrMSI(pkg_dir / "tmp_r13_n317.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region13_nonorm_sample.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region13_317norm_sample.zarr", mode="r+"),
    ),
    "r14": (
        OMEZarrMSI(pkg_dir / "tmp_r14.zarr", mode="r+"),
        OMEZarrMSI(pkg_dir / "tmp_r14_n317.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region14_nonorm_sample.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region14_317norm_sample.zarr", mode="r+"),
    ),
    "r15": (
        OMEZarrMSI(pkg_dir / "tmp_r15.zarr", mode="r+"),
        OMEZarrMSI(pkg_dir / "tmp_r15_n317.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region15_nonorm_sample.zarr", mode="r+"),
        OMEZarrMSI(home_ds_dir / "region15_317norm_sample.zarr", mode="r+"),
    ),
}
assert sorted(ms_dict.keys()) == sorted(all_version_ms_dict.keys())
assert all(len(tpl) == 4 for tpl in all_version_ms_dict.values())

# %%

rasterized_masks = {
    cls: raster_float(ann_lst, ms_mask_.shape, super_sampling)
    for cls, ann_lst in mapped_ann_dict[key].items()
}

fig, axes = plt.subplots(len(rasterized_masks), 1, figsize=(12, 8))
for idx, (cls_, mask_) in enumerate(rasterized_masks.items()):
    axes[idx].imshow(mask_)
    axes[idx].set_title(cls_)
fig.tight_layout()

# %% add rasterization to dataset

for dataset in all_version_ms_dict[key]:
    try:
        for cls in rasterized_masks.keys():
            dataset.delete_label(cls)
    except KeyError as err:
        print(f"{err=!r}")
        pass

    for cls, label_ in rasterized_masks.items():
        print(f"{label_.shape=!r} ({cls=!r})")
        label_ = np.expand_dims(label_, (0, 1))
        print(f"{label_.shape=!r} ({cls=!r})")
        dataset.add_label(cls, label_, *"czyx")

# %%

list(ms_dict[key]._group_at("/labels").keys())
