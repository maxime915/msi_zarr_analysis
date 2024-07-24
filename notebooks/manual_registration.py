# %%

import pathlib
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
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
from shapely import Polygon, affinity
from skimage.color import rgba2rgb

from msi_zarr_analysis.ml.dataset.translate_annotation import ParsedAnnotation
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import get_overlay_annotations


# %%

def _read_tiff(path: pathlib.Path, page_idx: int) -> np.ndarray:
    img = tifffile.TiffFile(path)
    return img.pages[page_idx].asarray().transpose((1, 2, 0)) / 255.0  # type:ignore


def _read_png(path: pathlib.Path) -> np.ndarray:
    return np.asarray(rgba2rgb(Image.open(path)))


msk_images_dir = pathlib.Path(__file__).parent.parent / "datasets" / "masks-images"

images = {
    "r13": _read_png(msk_images_dir / "mask-r13.png"),
    "r14": _read_tiff(msk_images_dir / "mask-r14.tif", 8),  # PLPC
    "r15": _read_tiff(msk_images_dir / "mask-r15.tif", 8),  # PLPC
}

# %%

fig, axes = plt.subplots(1, len(images), squeeze=False, figsize=(9, 9))
for idx, (key, image) in enumerate(images.items()):
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
fig.tight_layout()

# %%


def stats(img: np.ndarray):
    "img[H, W, RGB]"

    return f"{img.shape=} {img.max()=} {img.min()=} {img.mean()=}"


print({k: stats(v) for k, v in images.items()})

# %%


def mask(img: np.ndarray) -> np.ndarray:
    "img[H, W, RGB]"
    return (img[..., 0] > 0.4) & (img[..., 1:].max(axis=2) < 0.4)


def mask_and_fill(img: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask(img))  # type:ignore


def mask_fill_sample(img: np.ndarray):
    filled_mask = mask_and_fill(img)
    # count labels
    labels: np.ndarray
    labels, _ = label(filled_mask)  # type:ignore
    label_val, label_cnt = np.unique(labels, return_counts=True)
    for idx in range(len(label_cnt)):
        if not filled_mask[labels == idx].any():
            label_cnt[idx] = 0
    sample_idx = np.argmax(label_cnt)

    return labels == label_val[sample_idx]


fig, axes = plt.subplots(
    1,
    len(images),
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, image) in enumerate(images.items()):
    image = image.copy()
    image[~mask_fill_sample(image), :] = 0.3
    axes[0, idx].imshow(image)
    axes[0, idx].set_title(key)
    axes[0, idx].set_axis_off()
fig.tight_layout()

# %%

ms_ds_dir = pathlib.Path(__file__).parent.parent

ms_dict = {
    "r13": OMEZarrMSI(ms_ds_dir / "tmp_r13.zarr"),
    "r14": OMEZarrMSI(ms_ds_dir / "tmp_r14.zarr"),
    "r15": OMEZarrMSI(ms_ds_dir / "tmp_r15.zarr"),
}

# ms_ds_dir = pathlib.Path("/home/maxime/datasets/COMULIS-slim-msi")

# ms_dict = {
#     "r13": OMEZarrMSI(ms_ds_dir / "region13_nonorm_sample.zarr"),
#     "r14": OMEZarrMSI(ms_ds_dir / "region14_nonorm_sample.zarr"),
#     "r15": OMEZarrMSI(ms_ds_dir / "region15_nonorm_sample.zarr"),

assert sorted(ms_dict) == sorted(images)

# %%


def ms_mask(ms: OMEZarrMSI):
    val_len: np.ndarray = ms.z_len[0, 0] > 0  # type:ignore
    return val_len


fig, axes = plt.subplots(
    len(ms_dict),
    1,
    squeeze=False,
    figsize=(9, 9),
)
for idx, (key, ms) in enumerate(ms_dict.items()):
    image = ms_mask(ms)
    axes[idx, 0].imshow(image)
    axes[idx, 0].set_title(key)
    axes[idx, 0].set_axis_off()
fig.tight_layout()

# %% compute transform


def compute_offset(img_mask: np.ndarray) -> tuple[int, int]:
    y, x = img_mask.nonzero()
    assert y.size > 0
    return y.min(), x.min()


def scale_factor(img_mask: np.ndarray, ms_mask: np.ndarray):
    # transform img mask
    img_mask = np.flip(img_mask, axis=1)
    img_mask = np.rot90(img_mask, k=-1, axes=(0, 1))

    # get dimensions in image_mask
    y, x = img_mask.nonzero()
    assert y.size > 0, "empty img_mask"

    img_height: int = y.max() + 1 - y.min()
    img_width: int = x.max() + 1 - x.min()

    y, x = ms_mask.nonzero()
    assert y.size > 0, "empty ms_mask"

    ms_height: int = y.max() + 1 - y.min()
    ms_width: int = x.max() + 1 - x.min()

    if img_width > img_height:
        return img_width / ms_width
    return img_height / ms_height


for key, ms in ms_dict.items():
    image = images[key]
    s = scale_factor(mask_fill_sample(image), ms_mask(ms))
    print(f"{key}: {s=!r}")

# %%

annotation = Polygon([(275, 350), (275, 600), (375, 600), (375, 550), (325, 550), (325, 350)])

key = "r13"
copy_img = images[key].copy()
copy_img_mask = mask_fill_sample(copy_img)
copy_ms = ms_mask(ms_dict[key])
annotation_mask = rasterize([annotation], copy_img_mask.shape)

# fig, axes = plt.subplots(1, 2, squeeze=False)
# axes[0, 0].imshow(copy_img_mask)
# axes[0, 1].imshow(copy_ms)
# fig.tight_layout()


def map_geometry_fn(img_mask: np.ndarray, ms_mask: np.ndarray):
    "get a mapping from polygons in img_mask to ms_mask"

    # get dimensions in image_mask
    y, x = img_mask.nonzero()
    assert y.size > 0, "empty img_mask"

    height = y.max() + 1 - y.min()
    width = x.max() + 1 - x.min()

    s = 1.0 / scale_factor(img_mask, ms_mask)

    def _fn(plg: Polygon) -> Polygon:
        "apply mapping from image to ms"

        # offset y, x to remove padding in img
        plg = affinity.affine_transform(plg, [1, 0, 0, 1, -x.min(), -y.min()])

        # flip axis X
        plg = affinity.affine_transform(plg, [-1, 0, 0, 1, width, 0])

        # non centered rotation (rotation then shift)
        plg = affinity.affine_transform(plg, [0, -1, 1, 0, height, 0])

        # scale down
        plg = affinity.affine_transform(plg, [s, 0, 0, s, 0, 0])

        # no need to offset for padding in ms_mask: there is none

        return plg

    return _fn


plot_image = np.stack([
    copy_img_mask.astype(float),
    rasterize([annotation], copy_img_mask.shape),
    np.ones_like(copy_img_mask, dtype=float),
], axis=-1)

fn = map_geometry_fn(copy_img_mask, copy_ms)
plot_ms = np.stack([
    copy_ms.astype(float),
    rasterize([fn(annotation)], copy_ms.shape),
    np.ones_like(copy_ms, dtype=float),
], axis=-1)


fig, axes = plt.subplots(1, 2, squeeze=False)
axes[0, 0].imshow(plot_image)
axes[0, 1].imshow(plot_ms)
fig.tight_layout()


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
with Cytomine("https://research.cytomine.be", "9245ed4e-980b-497c-b305-6c24c3143c3b", "e78600fa-a541-4ec5-8e78-c75ea4fd4fc0"):
    annotation_dict = {
        key: get_overlay_annotations(
            project_id=project_id,
            image_id=image_id,
            classes=classes,
            select_users=(),
        ) for key, image_id in image_id_dict.items()
    }

# %% map annotation dict

mapped_ann_dict = {
    r_key: {
        cls_key: [
            ParsedAnnotation(p_a.annotation, map_fn_(p_a.geometry))
            for p_a in ann_lst
        ]
        for cls_key, ann_lst in ann_dict.items()
    }
    for r_key, ann_dict in annotation_dict.items()
    if (map_fn_ := map_geometry_fn(mask_fill_sample(images[r_key]), ms_mask(ms_dict[r_key])))
}

# %% load overlays

overlays_dict = {
    "r13": _read_tiff(msk_images_dir / "overlay-r13.tif", 6),  # PLPC
    "r14": _read_tiff(msk_images_dir / "mask-r14.tif", 8),  # PLPC
    "r15": _read_tiff(msk_images_dir / "mask-r15.tif", 8),  # PLPC
}

assert sorted(overlays_dict.keys()) == sorted(images.keys())

if any(overlay.shape != images[key].shape for key, overlay in overlays_dict.items()):
    raise ValueError("size mismatch")


# %% prepare data for plots

key = "r15"
ms_mask_: np.ndarray = ms_mask(ms_dict[key])
image_size = (images[key].shape[1] / 200.0, images[key].shape[0] / 200.0)
ms_size = (ms_mask(ms_dict[key]).shape[1] / 20, ms_mask(ms_dict[key]).shape[0] / 20)
# _, ms_data = tic_array(ms_dict[key])
_, ms_data = mz_slice(ms_dict[key], 758.56943 - 0.00166, 758.56943 + 0.00166)

# %%

mz_mean = 546.56943
mz_tol = 0.00166
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

# fig = plt.figure(frameon=False)
# fig.set_size_inches(*image_size)
# ax = plt.Axes(fig, (0., 0., 1., 1.))  # type:ignore
# ax.set_axis_off()
# fig.add_axes(ax)

# ax.imshow(overlays_dict[key])

# for idx, ann_lst in enumerate(annotation_dict[key].values()):
#     raster = rasterize([a.geometry for a in ann_lst], overlays_dict[key].shape[:2])
#     raster_rgba = np.concatenate(
#         [
#             np.multiply.outer(raster, color_lst[idx]),
#             np.stack([raster * 0.8], axis=-1),
#         ],
#         axis=-1,
#     )
#     ax.imshow(raster_rgba)

# %% plot overlay for the ms with rasterized annotations

# fig = plt.figure(frameon=False)
# fig.set_size_inches(*ms_size)
# ax = plt.Axes(fig, (0., 0., 1., 1.))  # type:ignore
# ax.set_axis_off()
# fig.add_axes(ax)

# # ax.imshow(np.where(ms_mask_, ms_data, np.nan))
# background = np.zeros(ms_mask_.shape + (3,))
# background[~ms_mask_, :] = 1.0
# ax.imshow(background)

# for idx, ann_lst in enumerate(mapped_ann_dict[key].values()):
#     raster = rasterize([a.geometry for a in ann_lst], ms_mask_.shape)
#     raster_rgba = np.concatenate(
#         [
#             np.multiply.outer(raster, color_lst[idx]),
#             np.stack([raster * 0.8], axis=-1),
#         ],
#         axis=-1,
#     )
#     ax.imshow(raster_rgba)


# %%

super_sampling: int = 16


def raster_float(ann_lst: list[ParsedAnnotation], shape: tuple[int, ...], s: int) -> np.ndarray:
    geometries = [a.geometry for a in ann_lst]

    h, w = shape
    s_shape = (s * h, s * w)
    s_geoms = [affinity.affine_transform(g, [s, 0, 0, s, 0, 0]) for g in geometries]
    s_raster = rasterize(s_geoms, s_shape)  # [s*h, s*w]

    # down-sample
    assert s_raster.flags.c_contiguous
    return np.reshape(s_raster, (h, s, w, s)).mean(axis=(1, 3))


fig = plt.figure(frameon=False)
fig.set_size_inches(*ms_size)
ax = plt.Axes(fig, (0., 0., 1., 1.))  # type:ignore
ax.set_axis_off()
fig.add_axes(ax)

# ax.imshow(np.where(ms_mask_, ms_data, np.nan))
background = np.zeros(ms_mask_.shape + (3,))
background[~ms_mask_, :] = 1.0
ax.imshow(background)

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
    )
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

# make a dummy label

target_ds = OMEZarrMSI(pkg_dir / "tmp_r13.zarr", mode="r+")
dummy_label = np.fromfunction(lambda i, j: (i + 1) * (j + 1), ms_mask_.shape)
dummy_label = np.expand_dims(dummy_label, (0, 1))
print(f"{dummy_label.shape=!r}")
target_ds.add_label("dummy-label", dummy_label, "c", "z", "y", "x")
print(f"{dummy_label.shape=!r}")
print(f"{target_ds.get_label('dummy-label').shape=}")
print(f"{target_ds.get_label('dummy-label')[...].shape=}")
target_ds.delete_label('dummy-label')
del target_ds

# %%

list(ms_dict[key]._group_at("/labels").keys())

# %%

# ms_dict[key].get_label("ls+").shape

# %%

ms_dict[key]._arr_at("/labels/ls-/0").shape

# %%
