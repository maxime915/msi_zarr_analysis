import json
from typing import (
    Iterable,
    List,
    NamedTuple,
    Tuple,
    Union,
)

import numpy as np
from cytomine.models import AnnotationCollection, ImageInstance
from matplotlib import pyplot as plt

from msi_zarr_analysis.ml.dataset.translate_annotation import (
    TemplateTransform,
    load_annotation,
    load_ms_template,
    load_tif_file,
    match_template_ms_overlay,
    rasterize_annotation_dict,
    translate_annotation_dict,
)
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.cytomine_utils import (
    get_lipid_dataframe,
    get_page_bin_indices,
)
from msi_zarr_analysis.utils.iter_chunks import iter_loaded_chunks


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

    lipid_df = get_lipid_dataframe(bin_csv_path)

    # build ds
    with Cytomine(host_url, pub_key, priv_key):

        for ds_config_itm in ds_config:

            # build all datasets and merge them if there are more than one
            page_idx, bin_idx, *_ = get_page_bin_indices(
                ds_config_itm.image_id_overlay, ds_config_itm.lipid_tm, bin_csv_path
            )

            save_annotated_image(
                name,
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
                term_list=ds_config_itm.term_list,
                zarr_template_path=ds_config_itm.zarr_template_path,
            )

def save_annotated_image(
    name: str,
    annotation_project_id: int,
    annotation_image_id: int,
    zarr_path: str,
    bin_idx: int,
    tiff_path: str,
    tiff_page_idx: int,
    transform_template_rot90: int = 0,
    transform_template_flip_ud: bool = False,
    transform_template_flip_lr: bool = False,
    select_users: Iterable[int] = (),
    select_terms: Iterable[int] = (),
    attribute_name_list: List[str] = (),
    *,
    term_list: List[str] = None,
    zarr_template_path: str,
):

    transform_template = TemplateTransform(
        transform_template_rot90,
        transform_template_flip_ud,
        transform_template_flip_lr,
    )

    if "binned" in zarr_path:
        raise ValueError("zarr_path must not be the binned version")
    ms_group = open_group_ro(zarr_path)

    if "binned" not in zarr_template_path:
        raise ValueError("zarr_template_path must be binned dataset")
    ms_template_group = open_group_ro(zarr_template_path)

    if ms_group["/0"].shape[1:] != ms_template_group["/0"].shape[1:]:
        raise ValueError("inconsistent shape between template and value groups")

    # template matching between the template and overlay
    matching_result, crop_idx = match_template_ms_overlay(
        ms_group=ms_template_group,
        bin_idx=bin_idx,
        tiff_path=tiff_path,
        tiff_page_idx=tiff_page_idx,
        transform=transform_template,
    )
    # load additional data to make the images
    overlay = load_tif_file(page_idx=tiff_page_idx, disk_path=tiff_path)
    # _, ms_template = load_ms_template(ms_template_group, bin_idx=bin_idx)

    # fetch all annotations
    annotations = AnnotationCollection()
    annotations.project = annotation_project_id
    annotations.image = annotation_image_id
    annotations.users = list(select_users)
    annotations.terms = list(select_terms)
    annotations.showTerm = True
    annotations.showWKT = True
    annotations.fetch()
    image_instance = ImageInstance().fetch(id=annotation_image_id)

    # filter matching annotation & correct geometry
    annotation_dict = load_annotation(
        annotations,
        image_height=image_instance.height,
        term_list=term_list,
    )

    # class names
    term_names = term_list
    if not term_names:
        term_names = list(annotation_dict.keys())

    # attribute names
    attribute_name_list = list(attribute_name_list)
    if not attribute_name_list:
        raise ValueError("missing attribut names")

    # update values from the annotation dict
    translate_annotation_dict(
        annotation_dict,
        template_transform=transform_template,
        matching_result=matching_result,
        crop_idx=crop_idx,
    )

    # have a numpy array like the template for each annotation
    rasterized_dict = rasterize_annotation_dict(
        annotation_dict, ms_template_group["/0"].shape[-2:]
    )
    
    # per annotation rasterization on the overlay
    image_raster_dict = rasterize_annotation_dict(
        annotation_dict,
        overlay.shape[:2],
        key="image_geometry",
    )
    
    def _color(idx: int):
        _color_data = [
            [255, 0, 0], # red
            [255, 255, 255], # white
        ]
        return _color_data[idx % len(_color_data)]

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(overlay, interpolation="nearest")
    for idx, (term, lst) in enumerate(image_raster_dict.items()):
        selection = 0
        for mask in lst:
            selection = np.logical_or(mask, selection)
        mask = np.zeros(selection.shape + (4,), dtype=np.uint8)
        mask[selection, :3] = _color(idx)
        mask[selection, 3] = 150
        
        ax.imshow(mask, interpolation="nearest")

    ax.set_title("Annotation " + name + " overlay")

    fig.tight_layout()
    fig.savefig("dataset_annotation_" + name + "_overlayYou can adjust the subplot geometry in the very tight_layout call as follows:.png")
    plt.close(fig)

    # Different choices to use for the MS image (max intensity, spectrum length, any lipid, ...)
    z_ints = ms_group["/0"]
    z_lens = ms_group["/labels/lengths/0"][0, 0]
    
    # ms_data = np.zeros(z_ints.shape[-2:], dtype=float)
    # for cy, cx in iter_loaded_chunks(z_ints, *crop_idx, skip=2):
    #     max_len = z_lens[cy, cx].max()
    #     ms_data[cy, cx] = ms_group["/0"][:max_len, 0, cy, cx].max(axis=0)
    ms_data = z_lens.astype(float)
    valid_mask = (z_lens != 0)
    ms_data[valid_mask] -= ms_data[valid_mask].mean()
    norm_val = ms_data[valid_mask].std()
    if np.abs(norm_val) > 1e-7:
        ms_data[valid_mask] /= norm_val
    ms_data[~valid_mask] = None

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(ms_data, interpolation="nearest")
    for idx, (term, lst) in enumerate(rasterized_dict.items()):
        selection = 0
        for mask in lst:
            selection = np.logical_or(mask, selection)
        mask = np.zeros(selection.shape + (4,), dtype=np.uint8)
        mask[selection, :3] = _color(idx)
        mask[selection, 3] = 150
        
        ax.imshow(mask, interpolation="nearest")

    ax.set_ylim((crop_idx[0].stop, crop_idx[0].start))  # backward y axis
    ax.set_xlim((crop_idx[1].start, crop_idx[1].stop))
    ax.set_title("Annotation " + name + " overlay")

    fig.tight_layout()
    fig.savefig("dataset_annotation_" + name + "_mass_spec.png")
    plt.close(fig)


def main():
    data_sources = [
        {
            "name": "region_13",
            "args": {
                "image_id_overlay": 545025763,
                "local_overlay_path": "datasets/Adjusted_Cytomine_MSI_3103_Region013-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 545025783,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "datasets/comulis13_binned.zarr",
            },
            "base": "datasets/comulis13",
        },
        {
            "name": "region_14",
            "args": {
                "image_id_overlay": 548365416,
                "local_overlay_path": "datasets/Region014-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365416,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "datasets/comulis14_binned.zarr",
            },
            "base": "datasets/comulis14",
        },
        {
            "name": "region_15",
            "args": {
                "image_id_overlay": 548365463,
                "local_overlay_path": "datasets/Region015-Viridis-stacked.ome.tif",
                "lipid_tm": "LysoPPC",
                "project_id": 542576374,
                "annotated_image_id": 548365463,
                "transform_rot90": 1,
                "transform_flip_ud": True,
                "transform_flip_lr": False,
                "annotation_users_id": (),
                "zarr_template_path": "datasets/comulis15_binned.zarr",
            },
            "base": "datasets/comulis15",
        },
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

    for name, class_problem in classification_problems.items():

        for source in data_sources:
            zarr_path = source["base"] + ".zarr"

            run(
                source["base"].split("/")[-1] + "_" + name,
                "config_cytomine.json",
                "mz value + lipid name.csv",
                DSConfig(
                    **source["args"],
                    **class_problem,
                    save_image=False,
                    zarr_path=zarr_path,
                ),
            )


if __name__ == "__main__":
    main()
