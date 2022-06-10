import contextlib
from datetime import datetime
import json
from multiprocessing import Process
import os
import pathlib
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from msi_zarr_analysis.ml.dataset import Dataset, MergedDS
from msi_zarr_analysis.ml.dataset.cytomine_ms_overlay import CytomineTranslated
from msi_zarr_analysis.ml.forests import (
    interpret_forest_mdi,
    interpret_model_mda,
    interpret_ttest,
    present_disjoint,
    present_p_values,
)
from msi_zarr_analysis.ml.utils import show_datasize_learning_curve
from msi_zarr_analysis.utils.cytomine_utils import (
    get_lipid_dataframe,
    get_page_bin_indices,
)
from sklearn.ensemble import ExtraTreesClassifier
from msi_zarr_analysis import VERSION


def get_cytomine_params(config_path: str) -> Tuple[str, str, str]:
    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    return host_url, pub_key, priv_key


class MLConfig(NamedTuple):
    et_max_depth: Optional[int]
    et_n_estimators: Optional[int]
    et_max_features: Optional[int]
    cv_fold: Optional[int]


class DSConfig(NamedTuple):
    image_id_overlay: int  # Cytomine ID for the overlay image
    local_overlay_path: str  # local path of the (downloaded) overlay
    lipid_tm: str  # name of the lipid to base the template matching on

    project_id: int  # project id
    annotated_image_id: int  # image with the annotations

    zarr_path: str  # path to the non-binned zarr image

    term_list: List[str]  # force order on the classes

    save_image: Union[bool, str] = False

    transform_rot90: int = 0
    transform_flip_ud: bool = False
    transform_flip_lr: bool = False

    annotation_users_id: Tuple[int] = ()  # select these users only
    annotation_terms_id: Tuple[int] = ()  # select these terms only

    zarr_template_path: str = None  # use another group for the template matching


def run(
    name: str,
    config_path: str,
    ml_config: MLConfig,
    bin_csv_path: str,
    *ds_config: DSConfig,
):
    if not ds_config:
        raise ValueError("a list of dataset configuration is required")

    from cytomine import Cytomine

    with open(config_path) as config_file:
        config_data = json.loads(config_file.read())
        host_url = config_data["HOST_URL"]
        pub_key = config_data["PUB_KEY"]
        priv_key = config_data["PRIV_KEY"]

    model_fun = lambda: ExtraTreesClassifier(
        n_estimators=ml_config.et_n_estimators,
        max_depth=ml_config.et_max_depth,
        max_features=ml_config.et_max_features,
        n_jobs=4,
    )

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

    # build ds
    with Cytomine(host_url, pub_key, priv_key):

        dataset_to_be_merged = []

        for ds_config_itm in ds_config:

            # build all datasets and merge them if there are more than one
            page_idx, bin_idx, *_ = get_page_bin_indices(
                ds_config_itm.image_id_overlay, ds_config_itm.lipid_tm, bin_csv_path
            )

            print(f"{ds_config_itm.save_image=}")

            ds = CytomineTranslated(
                annotation_project_id=ds_config_itm.project_id,
                annotation_image_id=ds_config_itm.annotated_image_id,
                zarr_path=ds_config_itm.zarr_path,
                bin_idx=bin_idx,
                tiff_path=ds_config_itm.local_overlay_path,
                tiff_page_idx=page_idx,
                transform_template_rot90=ds_config_itm.transform_rot90,
                transform_template_flip_ud=ds_config_itm.transform_flip_ud,
                transform_template_flip_lr=ds_config_itm.transform_flip_lr,
                select_users=ds_config_itm.annotation_users_id,
                select_terms=ds_config_itm.annotation_terms_id,
                attribute_name_list=lipid_df.Name,
                save_image=ds_config_itm.save_image,
                term_list=ds_config_itm.term_list,
                zarr_template_path=ds_config_itm.zarr_template_path,
            )

            dataset_to_be_merged.append(ds)

        if len(dataset_to_be_merged) == 1:
            ds = dataset_to_be_merged[0]
        else:
            ds = MergedDS(*dataset_to_be_merged)

    comulis_translated_example(
        ds,
        model_fun,
        name,
        lipids_of_interest,
        ml_config.cv_fold,
    )


def comulis_translated_example(
    ds: Dataset,
    model_fun: Callable[[], sklearn.base.BaseEstimator],
    name: str,
    lipids_of_interest: pd.DataFrame,
    cv_fold: Optional[int],
):

    ds.check_dataset(print_=True)

    print(f"model: {model_fun()!r}")
    class_names = ds.class_names()
    print(f"terms: {class_names}")

    # pairplot
    ds_x, ds_y = ds.as_table()

    pair_df = pd.DataFrame(columns=lipids_of_interest.Name)
    pair_df["class"] = [class_names[idx] for idx in ds_y]

    for tpl in lipids_of_interest.itertuples():
        pair_df[tpl.Name] = ds_x[:, tpl.Index]

    sns.pairplot(pair_df, hue="class")
    plt.savefig("pair_plot_" + name + "_" + "_".join(class_names))

    mdi_m, mdi_s = interpret_forest_mdi(ds, model_fun(), cv_fold)
    mda_m, mda_s = interpret_model_mda(ds, model_fun(), cv_fold)
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
        model_fun(),
        cv_fold,
        save_to="learning_curve_" + name + "_" + "_".join(ds.class_names()) + ".png",
    )


def do_all():
    print(f"{__file__=}")
    print(f"{VERSION=}")

    region_13_args = {
        "image_id_overlay": 545025763,
        "local_overlay_path": "Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
        "lipid_tm": "LysoPPC",
        "project_id": 542576374,
        "annotated_image_id": 545025783,
        "transform_rot90": 1,
        "transform_flip_ud": True,
        "transform_flip_lr": False,
        "annotation_users_id": (),
    }

    zarr_normalized_files = [
        "comulis13_binned.zarr",
        "comulis13_norm_2305_binned.zarr",
        "comulis13_norm_max_binned.zarr",
        "comulis13_norm_tic_binned.zarr",
        "comulis13_norm_vect_binned.zarr",
    ]

    prefix = os.path.commonprefix(zarr_normalized_files)

    classification_problems = {
        "SC_n_SC_p": {
            "term_list": ["SC negative AREA", "SC positive AREA"],
            "annotation_terms_id": (544926052, 544924846),
        },
        "LS_n_LS_p": {
            "term_list": ["LivingStrata negative AREA", "LivingStrata positive AREA"],
            "annotation_terms_id": (544926097, 544926081),
        },
        "LS_n_SC_n": {
            "term_list": ["LivingStrata negative AREA", "SC negative AREA"],
            "annotation_terms_id": (544926097, 544926052),
        },
        "LS_p_SC_p": {
            "term_list": ["LivingStrata positive AREA", "SC positive AREA"],
            "annotation_terms_id": (544926081, 544924846),
        },
    }

    for zarr_path in zarr_normalized_files:
        suffix = zarr_path.split(".")[0][len(prefix) :]

        for name, class_problem in classification_problems.items():
            base = suffix + "_" + name

            # build stdout.txt, stderr.txt
            with open(f"log_{base}_out.txt", mode="w") as out, open(
                f"log_{base}_err.txt", mode="w"
            ) as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):

                    print(f"{__file__=}")
                    print(f"{VERSION=}")
                    print(f"{datetime.now()=}")

                    ds_config = DSConfig(
                        **region_13_args,
                        **class_problem,
                        save_image=pathlib.Path(zarr_path).stem,
                        zarr_path=zarr_path,
                    )

                    # fix issue with Cytomine logger: use a different process
                    process = Process(
                        target=run,
                        args=(
                            base,
                            "config_cytomine.json",
                            MLConfig(
                                et_max_depth=None,
                                et_n_estimators=1000,
                                et_max_features=None,
                                cv_fold=10,
                            ),
                            "mz value + lipid name.csv",
                            ds_config,
                        ),
                    )

                    process.start()
                    process.join()

                    print(f"{process.exitcode=}")


def do_joint():

    region_13_args = {
        "image_id_overlay": 545025763,
        "local_overlay_path": "Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
        "lipid_tm": "LysoPPC",
        "project_id": 542576374,
        "annotated_image_id": 545025783,
        "transform_rot90": 1,
        "transform_flip_ud": True,
        "transform_flip_lr": False,
        "annotation_users_id": (),
    }

    region_14_args = {
        "image_id_overlay": 548365416,
        "local_overlay_path": "Region014-Viridis-stacked.ome.tif",
        "lipid_tm": "LysoPPC",
        "project_id": 542576374,
        "annotated_image_id": 548365416,
        "transform_rot90": 1,
        "transform_flip_ud": True,
        "transform_flip_lr": False,
        "annotation_users_id": (),
    }

    SC_n_SC_p = {
        "term_list": ["SC negative AREA", "SC positive AREA"],
        "annotation_terms_id": (544926052, 544924846),
    }

    run(
        "joint",
        "config_cytomine.json",
        MLConfig(
            et_max_depth=None,
            et_n_estimators=1000,
            et_max_features=None,
            cv_fold=10,
        ),
        "mz value + lipid name.csv",
        DSConfig(
            **region_13_args,
            **SC_n_SC_p,
            save_image="_r13",
            zarr_path="comulis13_binned.zarr",
        ),
        DSConfig(
            **region_14_args,
            **SC_n_SC_p,
            save_image="_r14",
            zarr_path="comulis14_binned.zarr",
        ),
    )


def do_all_joint():

    data_sources = [
        {
            "name": "region_13",
            "args": {
                "image_id_overlay": 545025763,
                "local_overlay_path": "Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 545025783,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "comulis13_binned.zarr",
            },
            "base": "comulis13",
        },
        {
            "name": "region_14",
            "args": {
                "image_id_overlay": 548365416,
                "local_overlay_path": "Region014-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365416,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "comulis14_binned.zarr",
            },
            "base": "comulis14",
        },
        {
            "name": "region_15",
            "args": {
                "image_id_overlay": 548365463,
                "local_overlay_path": "Region015-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365463,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "comulis15_binned.zarr",
            },
            "base": "comulis15",
        },
    ]

    normalizations = [
        "",
        "_norm_2305",
        "_norm_max",
        "_norm_tic",
        "_norm_vect",
    ]

    classification_problems = {
        "SC_n_SC_p": {
            "term_list": ["SC negative AREA", "SC positive AREA"],
            "annotation_terms_id": (544926052, 544924846),
        },
        "LS_n_LS_p": {
            "term_list": ["LivingStrata negative AREA", "LivingStrata positive AREA"],
            "annotation_terms_id": (544926097, 544926081),
        },
        "LS_n_SC_n": {
            "term_list": ["LivingStrata negative AREA", "SC negative AREA"],
            "annotation_terms_id": (544926097, 544926052),
        },
        "LS_p_SC_p": {
            "term_list": ["LivingStrata positive AREA", "SC positive AREA"],
            "annotation_terms_id": (544926081, 544924846),
        },
    }

    for normalization in normalizations:

        for name, class_problem in classification_problems.items():
            base = name + (normalization or "_no_norm")

            # build stdout.txt, stderr.txt
            with open(f"log_{base}_out.txt", mode="w") as out, open(
                f"log_{base}_err.txt", mode="w"
            ) as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):

                    print(f"{__file__=}")
                    print(f"{VERSION=}")
                    start = datetime.now()
                    print(f"{start=}")

                    ds_lst = []

                    for source in data_sources:
                        zarr_path = source["base"] + normalization + "_binned.zarr"
                        ds_lst.append(
                            DSConfig(
                                **source["args"],
                                **class_problem,
                                save_image=False,
                                zarr_path=zarr_path,
                            )
                        )

                    # fix issue with Cytomine logger: use a different process
                    process = Process(
                        target=run,
                        args=(
                            base,
                            "config_cytomine.json",
                            MLConfig(
                                et_max_depth=None,
                                et_n_estimators=1000,
                                et_max_features=None,
                                cv_fold=10,
                            ),
                            "mz value + lipid name.csv",
                            *ds_lst,
                        ),
                    )

                    process.start()
                    process.join()

                    print(f"{process.exitcode=}")
                    end = datetime.now()
                    print(f"{end=}")
                    print(f"{end-start=}")


if __name__ == "__main__":
    # do_all()
    # do_joint()
    do_all_joint()
