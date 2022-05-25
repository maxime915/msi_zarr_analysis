"entry point module"

import json
import pathlib
from typing import Tuple

import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import (
    CytomineTranslated,
    CytomineTranslatedProgressiveBinningFactory,
)
from msi_zarr_analysis.ml.utils import get_feature_importance_forest_mdi, show_datasize_learning_curve
from msi_zarr_analysis.utils.check import open_group_ro
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from ..ml import dataset
from ..ml.forests import interpret_forest_binned as interpret_trees_binned_
from ..ml.forests import interpret_forest_ds, interpret_forest_mdi
from ..ml.forests import interpret_forest_nonbinned as interpret_trees_nonbinned_
from ..ml.forests import (
    interpret_model_mda,
    interpret_ttest,
    present_disjoint,
    present_p_values,
)
from ..preprocessing.binning import bin_processed_lo_hi
from ..preprocessing.normalize import normalize_array, valid_norms
from ..utils.cytomine_utils import get_lipid_dataframe, get_lipid_names, get_page_bin_indices
from .utils import (
    BinningParam,
    RegionParam,
    bins_from_csv,
    load_img_mask,
    parser_callback,
    split_csl,
    uniform_bins,
)


@click.group()
@click.option("--echo/--no-echo", default=False)
def main_group(echo):
    if echo:
        import sys

        click.echo(" ".join(sys.argv))


@main_group.command()
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
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@BinningParam.add_click_options
@RegionParam.add_click_options
def interpret_trees_binned(
    image_zarr_path: click.Path,
    roi_mask_path: click.Path,
    cls_mask_path: Tuple[click.Path],
    background: bool,
    **kwargs,
):
    """provide an interpretation using random forests for the detection of a mask
    in the image, shows details on which bins provide the most information about
    the decision.
    """

    # check arguments

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

    bin_lo, bin_hi = BinningParam(kwargs).validated().get_bins()
    region = RegionParam(kwargs).validated()

    interpret_trees_binned_(
        image_zarr_path=image_zarr_path,
        cls_dict=classes_dict,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
        y_slice=region.y_slice,
        x_slice=region.x_slice,
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
    type=click.Choice(["extra_trees", "random_forests", "dt"], case_sensitive=True),
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
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members",
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
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members",
    required=True,
)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
    required=True,
)
@click.option(
    "--lipid",
    type=str,
    default="LysoPPC",
)
@click.option(
    "--select-terms-id",
    type=str,
    default="",
    help="Cytomine identifier for the term to fetch. Expects a comma separated list of ID.",
)
@click.option(
    "--select-users-id",
    type=str,
    default="",
    help="Cytomine identifier for the users that did the annotations. Expects a comma separated list of ID.",
)
@click.option(
    "--et-max-depth",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-n-estimators",
    default=1000,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-max-features",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--cv-fold",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "overlay_tiff_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument("overlay_id", type=int, default=545025763)
@click.argument("annotated_image_id", type=int, default=545025783)
@click.argument("annotated_project_id", type=int, default=542576374)
def comulis_translated_example(
    config_path: str,
    bin_csv_path: str,
    lipid: str,
    select_terms_id: str,
    select_users_id: str,
    et_max_depth,
    et_n_estimators,
    et_max_features,
    cv_fold,
    image_zarr_path: str,
    overlay_tiff_path: str,
    overlay_id: int,
    annotated_image_id: int,
    annotated_project_id: int,
):
    from cytomine import Cytomine

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    model_ = lambda: ExtraTreesClassifier(
        n_estimators=et_n_estimators,
        max_depth=et_max_depth,
        max_features=et_max_features,
        n_jobs=4,
    )

    with Cytomine(host_url, pub_key, priv_key):

        page_idx, bin_idx, *_ = get_page_bin_indices(overlay_id, lipid, bin_csv_path)
        lipid_df = get_lipid_dataframe(bin_csv_path)

        lipids_of_interest = lipid_df[
            lipid_df.Name.isin(
                [
                    "DPPC",
                    "PLPC",
                    "LysoPPC",
                ]
            )
        ]

        ds = CytomineTranslated(
            annotated_project_id,
            annotated_image_id,
            image_zarr_path,
            bin_idx,
            overlay_tiff_path,
            page_idx,
            transform_template_rot90=1,
            transform_template_flip_ud=True,
            select_users=split_csl(select_users_id),
            select_terms=split_csl(select_terms_id),
            attribute_name_list=lipid_df.Name,
            save_image=True,
        )

        ds.check_dataset(print_=True)

        print(f"model: {model_()!r}")
        class_names = ds.class_names()
        print(f"terms: {class_names}")

        # pairplot
        ds_x, ds_y = ds.as_table()
        
        pair_df = pd.DataFrame(columns=lipids_of_interest.Name)
        pair_df["class"] = [class_names[idx] for idx in ds_y]

        for tpl in lipids_of_interest.itertuples():
            pair_df[tpl.Name] = ds_x[:, tpl.Index]

        sns.pairplot(pair_df, hue="class")
        plt.savefig("pair_plot_" + "_".join(class_names))

        mdi_m, mdi_s = interpret_forest_mdi(ds, model_(), cv_fold)
        mda_m, mda_s = interpret_model_mda(ds, model_(), cv_fold)
        p_value, _ = interpret_ttest(ds, correction="bonferroni")

        present_p_values(
            ("p-value", p_value, "min"),
            limit=None,
            labels=ds.attribute_names(),
        )

        present_disjoint(
            ("MDI", mdi_m, mdi_s, "max"),
            ("MDA", mda_m, mda_s, "max"),
            limit=None,
            limit_bold=10,
            labels=ds.attribute_names(),
        )

        show_datasize_learning_curve(
            ds,
            model_(),
            cv_fold,
            save_to="learning_curve_" + "_".join(ds.class_names()) + ".png",
        )


@main_group.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members",
    required=True,
)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
    required=True,
)
@click.option(
    "--lipid",
    type=str,
    default="LysoPPC",
)
@click.option(
    "--select-terms-id",
    type=str,
    default="",
    help="Cytomine identifier for the term to fetch. Expects a comma separated list of ID.",
)
@click.option(
    "--select-users-id",
    type=str,
    default="",
    help="Cytomine identifier for the users that did the annotations. Expects a comma separated list of ID.",
)
@click.option(
    "--et-max-depth",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-n-estimators",
    default=1000,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-max-features",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--cv-fold",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--mz-low",
    default=100.0,
    callback=parser_callback,
)
@click.option(
    "--mz-high",
    default=1150.0,
    callback=parser_callback,
)
@click.option(
    "--n-bins",
    default=20,
    callback=parser_callback,
)
@click.argument(
    "binned_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "processed_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "overlay_tiff_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument("overlay_id", type=int, default=545025763)
@click.argument("annotated_image_id", type=int, default=545025783)
@click.argument("annotated_project_id", type=int, default=542576374)
def comulis_translated_example_custombins(
    config_path: str,
    bin_csv_path: str,
    lipid: str,
    select_terms_id: str,
    select_users_id: str,
    et_max_depth,
    et_n_estimators,
    et_max_features,
    cv_fold,
    mz_low: float,
    mz_high: float,
    n_bins: int,
    binned_zarr_path: str,
    processed_zarr_path: str,
    overlay_tiff_path: str,
    overlay_id: int,
    annotated_image_id: int,
    annotated_project_id: int,
):
    from cytomine import Cytomine

    if not isinstance(n_bins, int):
        raise ValueError(f"{n_bins=!r} should be an int")
    if n_bins % 2 == 1:
        print(f"{n_bins=} should be even, an additional bin will be added")
        n_bins += 1

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    model_ = lambda: ExtraTreesClassifier(
        n_estimators=et_n_estimators,
        max_depth=et_max_depth,
        max_features=et_max_features,
        n_jobs=4,
    )
    print(f"model: {model_()!r}")

    with Cytomine(host_url, pub_key, priv_key):

        page_idx, bin_idx, *_ = get_page_bin_indices(overlay_id, lipid, bin_csv_path)

        factory = CytomineTranslatedProgressiveBinningFactory(
            annotation_project_id=annotated_project_id,
            annotation_image_id=annotated_image_id,
            zarr_binned_path=binned_zarr_path,
            bin_idx=bin_idx,
            tiff_path=overlay_tiff_path,
            tiff_page_idx=page_idx,
            transform_template_rot90=1,
            transform_template_flip_ud=True,
            select_users=split_csl(select_users_id),
            select_terms=split_csl(select_terms_id),
        )

        print(f"terms: {factory.term_names}")

        ms_group = open_group_ro(processed_zarr_path)

        # setup bins
        bin_lo, bin_hi = uniform_bins(mz_low, mz_high, n_bins)

        ds = factory.bin(ms_group, bin_lo, bin_hi)

        ds.check_dataset(print_=True)

        mdi_m, mdi_s = interpret_forest_mdi(ds, model_(), cv_fold)
        mda_m, mda_s = interpret_model_mda(ds, model_(), cv_fold)
        p_value, _ = interpret_ttest(ds, correction="bonferroni")

        present_p_values(
            ("p-value", p_value, "min"),
            limit=None,
            labels=ds.attribute_names(),
            p_value_limit=5e-2,
        )

        present_disjoint(
            ("MDI", mdi_m, mdi_s, "max"),
            ("MDA", mda_m, mda_s, "max"),
            limit=None,
            limit_bold=int(np.ceil(np.sqrt(n_bins))),
            labels=ds.attribute_names(),
        )

        show_datasize_learning_curve(
            ds,
            model_(),
            cv_fold,
            save_to="finer_learning_curve_" + "_".join(ds.class_names()) + ".png",
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


@main_group.command()
@click.option("--x-low", type=int, default=0)
@click.option("--x-high", type=int, default=-1)
@click.option("--y-low", type=int, default=0)
@click.option("--y-high", type=int, default=-1)
@click.option(
    "--norm",
    type=click.Choice(valid_norms(), case_sensitive=False),
    default="tic",
)
@click.argument(
    "image_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("destination_zarr_path", type=click.Path(exists=False))
def normalize(
    image_zarr_path: click.Path,
    destination_zarr_path: click.Path,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
    norm: str,
):

    # bad idea: it's better to normalize after binning (and can be done online)

    normalize_array(
        image_zarr_path,
        destination_zarr_path,
        norm_name=norm,
        y_slice=slice(y_low, y_high),
        x_slice=slice(x_low, x_high),
    )


@main_group.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members",
    required=True,
)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
    required=True,
)
@click.option(
    "--lipid",
    type=str,
    default="LysoPPC",
)
@click.option(
    "--select-terms-id",
    type=str,
    default="",
    help="Cytomine identifier for the term to fetch. Expects a comma separated list of ID.",
)
@click.option(
    "--select-users-id",
    type=str,
    default="",
    help="Cytomine identifier for the users that did the annotations. Expects a comma separated list of ID.",
)
@click.option(
    "--et-max-depth",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-n-estimators",
    default=1000,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-max-features",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--cv-fold",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--mz-low",
    default=100.0,
    callback=parser_callback,
)
@click.option(
    "--mz-high",
    default=1150.0,
    callback=parser_callback,
)
@click.option(
    "--n-bins",
    default=20,
    callback=parser_callback,
)
@click.option(
    "--min-bin-width",
    default=1.0,
    callback=parser_callback,
)
@click.argument(
    "binned_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "processed_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "overlay_tiff_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument("overlay_id", type=int, default=545025763)
@click.argument("annotated_image_id", type=int, default=545025783)
@click.argument("annotated_project_id", type=int, default=542576374)
def comulis_translated_progressive_binning(
    config_path: str,
    bin_csv_path: str,
    lipid: str,
    select_terms_id: str,
    select_users_id: str,
    et_max_depth,
    et_n_estimators,
    et_max_features,
    cv_fold,
    mz_low: float,
    mz_high: float,
    n_bins: int,
    min_bin_width: float,
    binned_zarr_path: str,
    processed_zarr_path: str,
    overlay_tiff_path: str,
    overlay_id: int,
    annotated_image_id: int,
    annotated_project_id: int,
):
    from cytomine import Cytomine

    if not isinstance(n_bins, int):
        raise ValueError(f"{n_bins=!r} should be an int")
    if n_bins % 2 == 1:
        print(f"{n_bins=} should be even, an additional bin will be added")
        n_bins += 1

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    model_ = lambda: ExtraTreesClassifier(
        n_estimators=et_n_estimators,
        max_depth=et_max_depth,
        max_features=et_max_features,
        n_jobs=4,
    )
    print(f"model: {model_()!r}")

    with Cytomine(host_url, pub_key, priv_key):

        page_idx, bin_idx, *_ = get_page_bin_indices(overlay_id, lipid, bin_csv_path)

        factory = CytomineTranslatedProgressiveBinningFactory(
            annotation_project_id=annotated_project_id,
            annotation_image_id=annotated_image_id,
            zarr_binned_path=binned_zarr_path,
            bin_idx=bin_idx,
            tiff_path=overlay_tiff_path,
            tiff_page_idx=page_idx,
            transform_template_rot90=1,
            transform_template_flip_ud=True,
            select_users=split_csl(select_users_id),
            select_terms=split_csl(select_terms_id),
        )

        print(f"terms: {factory.term_names}")

        ms_group = open_group_ro(processed_zarr_path)

        # setup bins
        bin_lo, bin_hi = uniform_bins(mz_low, mz_high, n_bins)

        collected_bin_lo = [bin_lo]
        collected_bin_hi = [bin_hi]
        collected_indices = []
        collected_scores = []

        while (current_width := min(bin_hi - bin_lo)) > min_bin_width:

            print(f"{current_width=}")
            print(f"{list(zip(bin_lo, bin_hi))=}")

            bin_lo_next = np.empty_like(bin_lo)
            bin_hi_next = np.empty_like(bin_hi)

            # build dataset (binning)
            ds = factory.bin(ms_group, bin_lo, bin_hi)

            # get n/2 most important features
            (mdi, _) = interpret_forest_mdi(ds, model_(), cv_fold)

            indices = np.argsort(mdi)
            indices = indices[n_bins // 2 :]

            # select the n/2 most important features
            bin_lo = bin_lo[indices]
            bin_hi = bin_hi[indices]

            # build refined bins
            midpoints = (bin_lo + bin_hi) / 2

            bin_lo_next[0::2] = bin_lo
            bin_lo_next[1::2] = midpoints

            bin_hi_next[0::2] = midpoints
            bin_hi_next[1::2] = bin_hi

            # next iteration
            bin_lo, bin_hi = bin_lo_next, bin_hi_next

            # this evaluation is biased, use comulis_translated_evaluate_progressive_binning
            collected_scores.append(np.nan)
            collected_indices.append(indices)
            collected_bin_lo.append(bin_lo)
            collected_bin_hi.append(bin_hi)

        np.savez(
            "fine_saved_prog_binning_" + "_".join(factory.term_names),
            collected_bin_hi=np.stack(collected_bin_hi),
            collected_bin_lo=np.stack(collected_bin_lo),
            collected_indices=np.stack(collected_indices),
            collected_scores=np.stack(collected_scores),
        )

        print("Done!")
        print(f"{list(zip(bin_lo, bin_hi))=}")


@main_group.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="path to a JSON file containing 'HOST_URL', 'PUB_KEY', 'PRIV_KEY' members",
    required=True,
)
@click.option(
    "--bin-csv-path",
    type=click.Path(exists=True),
    help=(
        "CSV file containing the m/Z values in the first column and the "
        "intervals' width in the second one. Overrides 'mz-low', 'mz-high' "
        "and 'b-bins'"
    ),
    required=True,
)
@click.option(
    "--lipid",
    type=str,
    default="LysoPPC",
)
@click.option(
    "--select-terms-id",
    type=str,
    default="",
    help="Cytomine identifier for the term to fetch. Expects a comma separated list of ID.",
)
@click.option(
    "--select-users-id",
    type=str,
    default="",
    help="Cytomine identifier for the users that did the annotations. Expects a comma separated list of ID.",
)
@click.option(
    "--et-max-depth",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-n-estimators",
    default=1000,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--et-max-features",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--cv-fold",
    default=None,
    help="see sci-kit learn documentation",
    callback=parser_callback,
)
@click.option(
    "--mz-low",
    default=100.0,
    callback=parser_callback,
)
@click.option(
    "--mz-high",
    default=1150.0,
    callback=parser_callback,
)
@click.option(
    "--n-bins",
    default=20,
    callback=parser_callback,
)
@click.option(
    "--min-bin-width",
    default=1.0,
    callback=parser_callback,
)
@click.argument(
    "binned_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "processed_zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "overlay_tiff_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument("overlay_id", type=int, default=545025763)
@click.argument("annotated_image_id", type=int, default=545025783)
@click.argument("annotated_project_id", type=int, default=542576374)
def comulis_translated_evaluate_progressive_binning(
    config_path: str,
    bin_csv_path: str,
    lipid: str,
    select_terms_id: str,
    select_users_id: str,
    et_max_depth,
    et_n_estimators,
    et_max_features,
    cv_fold,
    mz_low: float,
    mz_high: float,
    n_bins: int,
    min_bin_width: float,
    binned_zarr_path: str,
    processed_zarr_path: str,
    overlay_tiff_path: str,
    overlay_id: int,
    annotated_image_id: int,
    annotated_project_id: int,
):
    from cytomine import Cytomine

    if not isinstance(n_bins, int):
        raise ValueError(f"{n_bins=!r} should be an int")
    if n_bins % 2 == 1:
        print(f"{n_bins=} should be even, an additional bin will be added")
        n_bins += 1

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    model_ = lambda: ExtraTreesClassifier(
        n_estimators=et_n_estimators,
        max_depth=et_max_depth,
        max_features=et_max_features,
        n_jobs=4,
    )
    print(f"model: {model_()!r}")

    with Cytomine(host_url, pub_key, priv_key):

        page_idx, bin_idx, *_ = get_page_bin_indices(overlay_id, lipid, bin_csv_path)

        factory = CytomineTranslatedProgressiveBinningFactory(
            annotation_project_id=annotated_project_id,
            annotation_image_id=annotated_image_id,
            zarr_binned_path=binned_zarr_path,
            bin_idx=bin_idx,
            tiff_path=overlay_tiff_path,
            tiff_page_idx=page_idx,
            transform_template_rot90=1,
            transform_template_flip_ud=True,
            select_users=split_csl(select_users_id),
            select_terms=split_csl(select_terms_id),
        )

        print(f"terms: {factory.term_names}")

        ms_group = open_group_ro(processed_zarr_path)

        # setup bins
        bin_lo, bin_hi = uniform_bins(mz_low, mz_high, n_bins)

        # TODO this is not stratified, but classes relatively well balanced so OK

        # how many folds -> cv_fold
        # how many elements per fold ? ceil(total / folds)
        n_elem_per_fold = int(np.ceil(factory.dataset_rows / cv_fold))
        # assign a fold to each row
        assignment = np.repeat(np.arange(n_elem_per_fold), cv_fold)
        # shuffle it
        np.random.shuffle(assignment)
        # trim last fold to fit the dataset
        assignment = assignment[:factory.dataset_rows]

        def _score_evolution(test_mask) -> np.ndarray:
            lows_ = bin_lo.copy()
            highs_ = bin_hi.copy()
            
            scores_ = []
            
            train_mask = ~test_mask
            
            while  min(highs_ - lows_) > min_bin_width:
                
                lows_next_ = np.empty_like(lows_)
                highs_next_ = np.empty_like(highs_)

                ds = factory.bin(ms_group, lows_, highs_)
                
                ds_x, ds_y = ds.as_table()
                
                forest_ = model_()
                forest_.fit(ds_x[train_mask], ds_y[train_mask])
                scores_.append(forest_.score(ds_x[test_mask], ds_y[test_mask]))
                
                mdi, _ = get_feature_importance_forest_mdi(forest_)

                indices = np.argsort(mdi)
                indices = indices[n_bins // 2 :]
                
                # select the n/2 most important features
                lows_ = lows_[indices]
                highs_ = highs_[indices]
                
                # build refined bins
                midpoints = (lows_ + highs_) / 2

                lows_next_[0::2] = lows_
                lows_next_[1::2] = midpoints

                highs_next_[0::2] = midpoints
                highs_next_[1::2] = highs_

                # next iteration
                lows_, highs_ = lows_next_, highs_next_
            
            return np.array(scores_)
        
        scores = np.stack([_score_evolution(assignment == k) for k in range(cv_fold)])

        np.savez(
            "evaluation_prog_binning_" + "_".join(factory.term_names),
            scores=scores,
        )

        print("Done!")
