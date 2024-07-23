import typing
from collections import defaultdict

import cytomine
import cytomine.models as cm
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads
from shapely.ops import transform

# given annotations in two images (of possibly different scales, auto-link them)
# given terms ? /api/project/542576374/term.json
# given users ?  /api/project/542576374/user.json
# given slices ?  -> nope, ask for an optional int id (or take the reference slice otherwise)


# TODO: this doesn't work...
#   - there may be extra annotations which aren't suitable for the match in both images
#   - another approach would be to
#       - normalize each polygon
#       - compute pairwise similarity measures (IoU...)
#       - try to greedily math based on similarity (remove row&col for best iteratively)
#       - stop when the similarity becomes too low (what is a good threshold in this case ?)


def get_image(id: int):
    img = cm.ImageInstance().fetch(id)
    if img is False:
        raise ValueError(f"unable to fetch image instance with {id=}")
    return img


def get_group(image: cm.ImageInstance):
    ig_ii_c = cm.ImageGroupImageInstanceCollection().fetch_with_filter(
        "imageinstance", image.id
    )
    if ig_ii_c is False:
        raise ValueError(f"unable to get image groups for {image.id=}")
    if len(ig_ii_c) != 1:
        raise ValueError(f"expected one image group for {image.id=} ({len(ig_ii_c)=})")
    group_id: int = ig_ii_c[0].group  # type: ignore

    group = cm.ImageGroup().fetch(group_id)
    if group is False:
        raise ValueError(f"could not fetch {group_id=}")
    return group


def get_annotations(image: cm.ImageInstance, terms: list[int], users: list[int]):
    an_coll = cm.AnnotationCollection()
    an_coll.project = image.project
    an_coll.image = image.id
    an_coll.terms = terms  # type: ignore
    an_coll.users = users  # type: ignore
    an_coll.showTerm = True  # type: ignore
    an_coll.showWKT = True  # type: ignore
    an_coll = an_coll.fetch()
    if an_coll is False:
        raise ValueError(f"could not fetch annotations for {image.id=}")
    
    return an_coll


def group_by_term(an_coll: cm.AnnotationCollection):

    grouped_by_term: typing.Dict[int, list[cm.Annotation]] = defaultdict(list)

    for an_ in an_coll:
        keys = an_.term or [-1]
        for key_ in keys:
            grouped_by_term[key_].append(an_)

    return grouped_by_term


_an_to_geom_bl_cache: typing.Dict[int, BaseGeometry] = {}


def get_bounds(an_coll: list[cm.Annotation], img: cm.ImageInstance):
    bounds: tuple[float, float, float, float] = None  # type: ignore

    for an_ in an_coll:
        geom_tl = loads(an_.location)
        geom_bl: BaseGeometry = affine_transform(geom_tl, [1, 0, 0, -1, 0, img.height])

        _an_to_geom_bl_cache[an_.id] = geom_bl  # type: ignore

        if bounds is None:
            bounds = geom_bl.bounds
            continue

        bounds = (
            min(bounds[0], geom_bl.bounds[0]),  # min x
            min(bounds[1], geom_bl.bounds[1]),  # min y
            max(bounds[2], geom_bl.bounds[2]),  # max x
            max(bounds[3], geom_bl.bounds[3]),  # max y
        )

    if bounds is None:
        raise ValueError(f"{an_coll=!r} is empty")

    return bounds


def get_bounds_by_term(
    grouped: typing.Mapping[int, list[cm.Annotation]],
    image: cm.ImageInstance,
):
    return {term: get_bounds(an_coll, image) for term, an_coll in grouped.items()}


def coordinate_array(bounds: list[tuple[float, float, float, float]], *, attr=3):
    ds = np.ones((4 * len(bounds), attr), float)
    for idx, (min_x, min_y, max_x, max_y) in enumerate(bounds):
        base = 4 * idx
        ds[base : base + 4, :2] = [
            [min_x, min_y],
            [min_x, max_y],
            [max_x, min_y],
            [max_x, max_y],
        ]

    return ds


class Mapper(typing.Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, z: None = None, /) -> np.ndarray:
        ...


def find_mapping(
    by_term_left: dict[int, tuple[float, float, float, float]],
    by_term_right: dict[int, tuple[float, float, float, float]],
) -> Mapper:
    "mapping from left(x, y) to right(x, y)"

    keys = list(by_term_left.keys())
    coord_left = coordinate_array([by_term_left[key_] for key_ in keys])
    coord_right = coordinate_array([by_term_right[key_] for key_ in keys], attr=2)

    matrix, *_ = np.linalg.lstsq(coord_left, coord_right, rcond=None)

    def mapping(x: np.ndarray, y: np.ndarray, z=None) -> np.ndarray:
        if z is not None:
            raise NotImplementedError("2D only")

        if isinstance(x, np.ndarray):
            _1 = np.ones_like(x)
        else:
            _1 = np.ones((1,), float)
        xy1 = np.stack([x, y, _1], axis=-1)
        xy1_ = xy1 @ matrix

        return xy1_[:, :2]

    return mapping


def get_center_by_term(by_term: dict[int, list[cm.Annotation]]):
    def _center_list(an_coll: list[cm.Annotation]) -> np.ndarray:
        return np.array(
            [_an_to_geom_bl_cache[an.id].centroid.coords[0] for an in an_coll]  # type: ignore
        )

    return {key: (an_coll, _center_list(an_coll)) for key, an_coll in by_term.items()}


def map_dict_arr(
    mapping: Mapper,
    centroids: dict[int, tuple[list[cm.Annotation], np.ndarray]],
):
    def _map_arr(arr: np.ndarray):
        return mapping(arr[:, 0], arr[:, 1])

    return {key: (an, _map_arr(arr)) for key, (an, arr) in centroids.items()}


def get_len_by_term(an_dict: dict[int, list[cm.Annotation]]):
    return {key: len(an_coll) for key, an_coll in an_dict.items()}


def check_lengths(
    left: dict[int, list[cm.Annotation]], right: dict[int, list[cm.Annotation]]
):
    left_count_by_term = get_len_by_term(left)
    right_count_by_term = get_len_by_term(right)

    if left_count_by_term != right_count_by_term:
        raise ValueError(f"{left_count_by_term} != {right_count_by_term}")


def associate(left_id: int, right_id: int, terms: list[int], users: list[int]):
    left_cyt = get_image(left_id)
    right_cyt = get_image(right_id)

    # make sure they are in the same image group
    image_group_left = get_group(left_cyt)
    image_group_right = get_group(right_cyt)
    if image_group_left.id != image_group_right.id:
        raise ValueError(f"{image_group_left.id=} != {image_group_right.id}: mismatch")

    # fetch annotations
    an_c_left = get_annotations(left_cyt, terms, users)
    an_c_right = get_annotations(right_cyt, terms, users)

    # make sure there are the same number of annotations, for each term, in each image
    left_by_term = group_by_term(an_c_left)
    right_by_term = group_by_term(an_c_right)
    check_lengths(left_by_term, right_by_term)

    # find bounding box for all annotations of given terms
    bounds_left = get_bounds_by_term(left_by_term, left_cyt)
    bounds_right = get_bounds_by_term(right_by_term, right_cyt)

    # guess linear mapping
    mapping = find_mapping(bounds_left, bounds_right)

    # find the center of each annotation (by term)
    center_left_RAW_dct = get_center_by_term(left_by_term)  # N, d
    center_left_MAP_dct = map_dict_arr(mapping, center_left_RAW_dct)  # N, d
    center_right_dct = get_center_by_term(right_by_term)  # N, d

    to_create: list[tuple[cm.Annotation, cm.Annotation]] = []

    # make a distance matrix between right and transformed left (by term)
    for term_ in center_right_dct.keys():
        an_coll_left, center_left = center_left_MAP_dct[term_]
        an_coll_right, center_right = center_right_dct[term_]

        dx2 = np.subtract.outer(center_left[:, 0], center_right[:, 0]) ** 2
        dy2 = np.subtract.outer(center_left[:, 1], center_right[:, 1]) ** 2
        distances = dx2 + dy2  # (N_left, N_right)

        left_to_right = np.argmin(distances, axis=1)  # (N_left,)
        right_to_left = np.argmin(distances, axis=0)  # (N_right,)

        i_ = np.arange(left_to_right.size)
        j_ = np.arange(right_to_left.size)

        mismatch = np.logical_or(
            (right_to_left[left_to_right[i_]] != i_),
            (left_to_right[right_to_left[j_]] != j_),
        )

        if mismatch.sum() != 0:
            bad_left, = np.nonzero(right_to_left[left_to_right[i_]] != i_)
            bad_right, = np.nonzero(left_to_right[right_to_left[j_]] != j_)
            
            print(f"{[an_coll_left[idx].id for idx in bad_left]=}")
            print(f"{[an_coll_right[idx].id for idx in bad_right]=}")
            raise ValueError(f"greedy approach failed for {term_=} ({mismatch=})")

        # we can now associate any annotation from the left to one on the right
        for i, an_left in enumerate(an_coll_left):
            j: int = left_to_right[i]
            an_right = an_coll_right[j]
            
            to_create.append((an_left, an_right))
    
    for an_left, an_right in to_create:
        # try to find one first !
        
        ag = cm.AnnotationGroup(left_cyt.project, image_group_left.id)
        if ag.save() is False:
            raise ValueError("unable to create annotation group")

        link_left = cm.AnnotationLink(
            id_annotation=an_left.id, id_annotation_group=ag.id
        )
        link_right = cm.AnnotationLink(
            id_annotation=an_right.id, id_annotation_group=ag.id
        )
        
        if link_left.save() is False:
            raise ValueError(f"could not link {an_left.id=}")
        if link_right.save() is False:
            raise ValueError(f"could not link {an_right.id=}")


# assign pairs to minimize the total IoU (or dice ?) between matching annotations

# -> naive pairing should produce IoU either very good or very bad

# -> performing k-means with k = the number of annotations for the current term
# should produce decent results when computing on the centers...

# -> not pairing any annotations is a trivial solution to that ! (and the best one at that...)
# -> how to include the number of mapped annotations in the mix ?

# this problem looks very hard...
# each annotation


CYT_HOST = "https://research.cytomine.be"
CYT_PUB_KEY = "9245ed4e-980b-497c-b305-6c24c3143c3b"
CYT_PRV_KEY = "e78600fa-a541-4ec5-8e78-c75ea4fd4fc0"

raise NotImplementedError("do not use this !")
# the grouping creation is wrong: if some annotations are already linked, it crashes
# the procedure ain't that great: some false negative, and requires exactly the same amount of annotations

with cytomine.Cytomine(CYT_HOST, CYT_PUB_KEY, CYT_PRV_KEY):
    associate(  # R13
        542625425,
        543034522,
        [544926081, 544926052, 544926097, 544924846],
        [542585245, 534530561],
    )

    associate(  # R14
        542625396,
        544126320,
        [544926081, 544926052, 544926097, 544924846],
        [542585245, 534530561],
    )

    # this one should be faster by hand...
    # associate(  # R15
    #     542625387,
    #     544126313,
    #     [544926081, 544926052, 544926097, 544924846],
    #     [542585245, 534530561],
    # )
