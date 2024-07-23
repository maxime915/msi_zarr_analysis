# interactive registration plot


# take as input \
# - the CSV for the lipids
# - the Cytomine ID for the overlay
# - the local path for the overlay
# - the local path for the zarr image

import argparse
import functools
import pathlib
import sys
import typing

import cytomine
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

try:
    import msi_zarr_analysis
except (ModuleNotFoundError, ImportError):
    # add msi_zarr_analysis to the path if it was not imported
    package_path = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    if package_path not in sys.path:
        sys.path.append(package_path)
    import msi_zarr_analysis

from msi_zarr_analysis.ml.dataset.translate_annotation import (
    TemplateTransform,
    colorize_data,
    load_ms_template_multi,
    load_tiff_file_multi,
    scale_image,
)
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import mmmm

CSV_PATH = "mz value + lipid name.csv"
CYT_HOST = "https://research.cytomine.be"
CYT_PUB_KEY = "9245ed4e-980b-497c-b305-6c24c3143c3b"
CYT_PRV_KEY = "e78600fa-a541-4ec5-8e78-c75ea4fd4fc0"

parser = argparse.ArgumentParser()
parser.add_argument("id_image", type=int)
parser.add_argument("overlay_path", type=str)
parser.add_argument("ms_path", type=str)

args = parser.parse_args(sys.argv[1:])

fig, ax = plt.subplots()
fig.subplots_adjust(0.3)


def pos(*, left=0.12, bottom=0.0, width=0.1, height=0.075):
    return [left, bottom, width, height]


alpha_b = TextBox(ax=fig.add_axes(pos(bottom=0.05)), label="Alpha", initial="0.5")

scale_b = TextBox(ax=fig.add_axes(pos(bottom=0.15)), label="Scale", initial="1.0")

y_b = TextBox(ax=fig.add_axes(pos(bottom=0.25)), label="Y", initial="0")

x_b = TextBox(ax=fig.add_axes(pos(bottom=0.35)), label="X", initial="0")

lr_b = TextBox(ax=fig.add_axes(pos(bottom=0.45)), label="Flip LR", initial="False")

ud_b = TextBox(ax=fig.add_axes(pos(bottom=0.55)), label="Flip UD", initial="False")

rot_b = TextBox(ax=fig.add_axes(pos(bottom=0.65)), label="Rot (90°)", initial="0")

lipid_b = TextBox(ax=fig.add_axes(pos(bottom=0.75)), label="Lipid (idx)", initial="0")

T = typing.TypeVar("T")


def read(button: TextBox, conv: typing.Callable[[str], T] = int, default="0") -> T:
    try:
        return conv(button.text)
    except ValueError:
        button.set_val(default)
        return conv(default)


def conv_bool(any: str):
    any = any.strip().lower()
    if any in ["1", "true", "yes"]:
        return True
    if any in ["0", "false", "no"]:
        return False
    raise ValueError(f"{any=!r} is not a bool")


old_res = None


def state_changed():
    global old_res

    x = read(x_b, int, "0")
    y = read(y_b, int, "0")
    scale = read(scale_b, float, "0.0")
    alpha = read(alpha_b, float, "0.0")
    lr = read(lr_b, conv_bool, "False")
    ud = read(ud_b, conv_bool, "False")
    rot = read(rot_b, int, "0")
    lipid_idx = read(lipid_b, int, "0")

    new_res = (x, y, scale, alpha, lr, ud, rot, lipid_idx)
    if new_res == old_res:
        return False, new_res

    old_res = new_res
    return True, new_res


@functools.lru_cache()
def load_base_data():

    with cytomine.Cytomine(CYT_HOST, CYT_PUB_KEY, CYT_PRV_KEY) as conn:
        page_dict, channel_dict, *_ = mmmm(args.id_image, CSV_PATH)

        common = set(channel_dict.keys()).intersection(page_dict.keys())
        channel_indices = tuple(channel_dict[lipid] for lipid in common)
        page_indices = tuple(page_dict[lipid] for lipid in common)

        ms_template_group = open_group_ro(args.ms_path)
        overlay_lst = load_tiff_file_multi(args.overlay_path, page_indices)
        crop_idx, ms_template_multi = load_ms_template_multi(
            ms_template_group, channel_indices
        )

    return overlay_lst, ms_template_multi, crop_idx

# x=120 y=65 scale=6.9
#transform=TemplateTransform(rotate_90=1, flip_ud=True, flip_lr=False)
#crop_idx=(slice(0, 31, None), slice(1, 272, None))

def get_data(state):
    (x, y, scale, alpha, lr, ud, rot, lipid_idx) = state

    overlay_lst, ms_template_lst, crop_idx = load_base_data()
    lipid_idx = lipid_idx % len(overlay_lst)
    overlay, ms_template = overlay_lst[lipid_idx], ms_template_lst[lipid_idx]

    transform = TemplateTransform(
        rot,
        ud,
        lr,
    )

    ms_template = transform.transform_template(ms_template)
    ms_template = colorize_data(ms_template)
    ms_template = scale_image(ms_template, scale)

    # TODO: compute overlay
    # https://stackoverflow.com/a/53942810/5770818

    # trim width and height
    # x + trimmed(ms_template.width) == overlay.width (if x + ms_template.width > overlay.width)
    # x + trimmed(ms_template.width) <= overlay.width (if x + ms_template.width <= overlay.width)
    max_width = ms_template.shape[1]
    if x + max_width > overlay.shape[1]:
        max_width = overlay.shape[1] - x
    max_height = ms_template.shape[0]
    if y + max_height > overlay.shape[0]:
        max_height = overlay.shape[0]

    copy = overlay.copy()
    copy[y : y + max_height, x : x + max_width] = (1 - alpha) * copy[
        y : y + max_height, x : x + max_width
    ] + alpha * ms_template[:max_height, :max_width]

    # log values
    print(f"{x=!r} {y=!r} {scale=!r}")
    print(f"{transform=!r}")
    print(f"{crop_idx=!r}")

    return copy


drawing = ax.matshow(get_data(state_changed()[1]))


def update(*_):
    changed, state = state_changed()
    if not changed:
        return

    drawing.set_data(get_data(state))
    fig.canvas.draw_idle()


for button in [alpha_b, scale_b, y_b, x_b, lr_b, ud_b, rot_b]:
    button.on_submit(update)

state_changed()
plt.show()
