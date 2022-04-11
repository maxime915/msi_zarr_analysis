"entry point module"

import json
import pathlib
from typing import Tuple

import click
from sklearn.ensemble import ExtraTreesClassifier

from ..ml import dataset
from ..ml.forests import interpret_forest_binned as interpret_trees_binned_
from ..ml.forests import interpret_forest_ds
from ..ml.forests import interpret_forest_nonbinned as interpret_trees_nonbinned_
from ..preprocessing.binning import bin_processed_lo_hi
from .utils import bins_from_csv, load_img_mask, uniform_bins


@click.group()
def main_group():
    pass


@main_group.command()
@click.option("--mz-low", type=float, default=200.0)
@click.option("--mz-high", type=float, default=850.0)
@click.option("--n-bins", type=int, default=100)
@click.option("--x-low", type=int, default=0)
@click.option("--x-high", type=int, default=-1)
@click.option("--y-low", type=int, default=0)
@click.option("--y-high", type=int, default=-1)
@click.option(
    "--background/--no-background",
    type=bool,
    default=False,
    help=(
        "If true, assign all blank pixel from the image to the background class."
        "Otherwise, ignore all pixels non included in the class masks, even those in the ROI"
    ),
)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
)
@click.argument("cls_mask_path", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--roi-mask-path", type=click.Path(exists=True))
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def interpret_trees_binned(
    image_zarr_path: click.Path,
    roi_mask_path: click.Path,
    cls_mask_path: Tuple[click.Path],
    mz_low: float,
    mz_high: float,
    n_bins: int,
    background: bool,
    bin_csv_path: click.Path,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
):
    """provide an interpretation using random forests for the detection of a mask
    in the image, shows details on which bins provide the most information about
    the decision.
    """

    # check arguments
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"{n_bins=} should be an int > 1")

    classes_dict = {
        pathlib.Path(cls_path).stem: load_img_mask(cls_path)
        for cls_path in cls_mask_path
    }

    # coordinates to be classified
    if roi_mask_path:
        roi_mask = load_img_mask(roi_mask_path)
    else:
        roi_mask = None

    base = pathlib.Path(image_zarr_path).stem

    if bin_csv_path:
        bin_lo, bin_hi = bins_from_csv(bin_csv_path)
    else:
        bin_lo, bin_hi = uniform_bins(mz_low, mz_high, n_bins)

    interpret_trees_binned_(
        image_zarr_path=image_zarr_path,
        cls_dict=classes_dict,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
        y_slice=slice(y_low, y_high, 1),
        x_slice=slice(x_low, x_high, 1),
        fi_impurity_path=base + "_fi_imp.csv",
        # fi_impurity_path=None,
        fi_permutation_path=base + "_fi_per.csv",
        # fi_permutation_path=None,
        roi_mask=roi_mask,
        append_background_cls=background,
        stratify_classes=True,
        model_choice="extra_trees",
    )


@main_group.command()
@click.option("--x-low", type=int, default=0)
@click.option("--x-high", type=int, default=-1)
@click.option("--y-low", type=int, default=0)
@click.option("--y-high", type=int, default=-1)
@click.option(
    "--background/--no-background",
    type=bool,
    default=False,
    help=(
        "If true, assign all blank pixel from the image to the background class."
        "Otherwise, ignore all pixels non included in the class masks, even those in the ROI"
    ),
)
@click.argument("cls_mask_path", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--roi-mask-path", type=click.Path(exists=True))
@click.option(
    "--model",
    type=click.Choice(["extra_trees", "random_forests"], case_sensitive=True),
    default="extra_trees",
)
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def interpret_trees_nonbinned(
    image_zarr_path: click.Path,
    roi_mask_path: click.Path,
    cls_mask_path: Tuple[click.Path],
    background: bool,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
    model: str,
):
    """provide an interpretation using random forests for the detection of a mask
    in the image, shows details on which bins provide the most information about
    the decision.
    """

    classes_dict = {
        pathlib.Path(cls_path).stem: load_img_mask(cls_path)
        for cls_path in cls_mask_path
    }

    # coordinates to be classified
    if roi_mask_path:
        roi_mask = load_img_mask(roi_mask_path)
    else:
        roi_mask = None

    base = pathlib.Path(image_zarr_path).stem

    interpret_trees_nonbinned_(
        image_zarr_path=image_zarr_path,
        cls_dict=classes_dict,
        y_slice=slice(y_low, y_high, 1),
        x_slice=slice(x_low, x_high, 1),
        fi_impurity_path=base + "_fi_imp.csv",
        # fi_impurity_path=None,
        fi_permutation_path=base + "_fi_per.csv",
        # fi_permutation_path=None,
        roi_mask=roi_mask,
        append_background_cls=background,
        stratify_classes=True,
        model_choice=model,
    )


@main_group.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members"
)
def cytomine_raw_example(
    config_path: str,
):
    from cytomine import Cytomine
    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    with Cytomine(host_url, pub_key, priv_key):
        
        ds = dataset.CytomineNonBinned(
            project_id=31054043,
            image_id=146726078,
            term_set={"Urothelium", "Stroma"},
        )

        interpret_forest_ds(
            ds,
            ExtraTreesClassifier(n_jobs=4),
            fi_impurity_path="cytomine_raw_live_fi_imp.csv",
            fi_permutation_path="cytomine_raw_live_fi_per.csv",
            stratify_classes=True,
        )


@main_group.command()
@click.option("--mz-low", type=float, default=200.0)
@click.option("--mz-high", type=float, default=850.0)
@click.option("--n-bins", type=int, default=100)
@click.option("--x-low", type=int, default=0)
@click.option("--x-high", type=int, default=-1)
@click.option("--y-low", type=int, default=0)
@click.option("--y-high", type=int, default=-1)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
)
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("destination_zarr_path", type=click.Path(exists=False))
def bin_dataset(
    image_zarr_path: click.Path,
    destination_zarr_path: click.Path,
    mz_low: float,
    mz_high: float,
    n_bins: int,
    bin_csv_path: click.Path,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
):
    # check arguments
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"{n_bins=} should be an int > 1")

    if bin_csv_path:
        bin_lo, bin_hi = bins_from_csv(bin_csv_path)
    else:
        bin_lo, bin_hi = uniform_bins(mz_low, mz_high, n_bins)

    bin_processed_lo_hi(
        image_zarr_path,
        destination_zarr_path,
        bin_lo,
        bin_hi,
        y_slice=slice(y_low, y_high),
        x_slice=slice(x_low, x_high),
    )
