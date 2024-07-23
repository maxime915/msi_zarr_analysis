import typing

import cv2
import cytomine
import cytomine.models as cm
import numpy as np
import numpy.typing as npt
import pydantic
from shapely.ops import transform
from shapely.wkt import dumps, loads

CYT_HOST = "https://research.cytomine.be"
CYT_PUB_KEY = "9245ed4e-980b-497c-b305-6c24c3143c3b"
CYT_PRV_KEY = "e78600fa-a541-4ec5-8e78-c75ea4fd4fc0"


class ImageRegistration(pydantic.BaseModel):
    w_border_id: int  # the image with black borders
    clean_id: int  # the source image (where annotations should go)


def and_all(*seq: np.ndarray) -> npt.NDArray:
    assert len(seq) > 0
    first, *rest = seq
    for item in rest:
        first = np.logical_and(first, item)
    return first


def copy_annotations(
    images: ImageRegistration,
):
    border_cy = cm.ImageInstance().fetch(images.w_border_id)
    if border_cy is False:
        raise ValueError(f"unable to download {images.w_border_id=}")

    clean_cy = cm.ImageInstance().fetch(images.clean_id)
    if clean_cy is False:
        raise ValueError(f"unable to download {images.clean_id=}")

    if not border_cy.download(override=False):
        raise ValueError(f"failed to download {border_cy.id=}")
    # if not moving_cy.download(override=False):
    #     raise ValueError(f"failed to download {moving_cy.id=}")

    # see https://stackoverflow.com/a/13539194/5770818
    border_bgr = cv2.imread(border_cy.filename)
    border_np = border_bgr[:, :, 1]

    # find greyish
    greyish = (np.max(border_bgr, axis=2) - np.min(border_bgr, axis=2)) < 20
    light = np.min(border_bgr, axis=2) > 100
    thresh: npt.NDArray[np.uint8] = np.uint8(255 * np.logical_and(greyish, light))  # type: ignore
    cv2.imwrite(f"tmp-{border_cy.id}.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_cnt = cv2.boundingRect(contours[0])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > largest_cnt[2] * largest_cnt[3]:
            largest_cnt = (x, y, w, h)

    x, y, w, h = largest_cnt
    print(f"{y=} {x=} {h=} {w=} {thresh.shape=}")

    # scale detection
    scale_x = clean_cy.width / w
    scale_y = clean_cy.height / h
    print(f"{scale_y=:.3E} {scale_x=:.3E}")

    # compute translation (starting from top left)
    def to_raw_tl(x_: npt.ArrayLike, y_: npt.ArrayLike, z_=None):
        assert z_ is None
        return (x_ - x) * scale_x, (y_ - y) * scale_y

    # compute translation (starting from bottom left)
    def to_raw_bl(x_: npt.ArrayLike, y_: npt.ArrayLike, z_=None):
        assert z_ is None
        x_, y_ = to_raw_tl(x_, border_cy.height - y_)  # type: ignore
        return x_, clean_cy.height - y_  # type: ignore

    an_coll = cm.AnnotationCollection()
    an_coll.project = border_cy.project
    an_coll.image = border_cy.id
    an_coll.showTerm = True
    an_coll.showWKT = True
    an_coll.terms = [544926081, 544926052, 544926097, 544924846]
    an_coll = an_coll.fetch()
    if an_coll is False:
        raise ValueError("cannot fetch annotations")

    # group by slice to avoid duplication
    groups: typing.Dict[int, typing.Set[cm.Annotation]] = {}
    for an_ in an_coll:
        group_ = groups.get(an_.slice, set())
        group_.add(an_)
        groups[an_.slice] = group_

    _, an_unique = groups.popitem()

    dst_an_coll = cm.AnnotationCollection()
    for an_ in an_unique:
        geom = loads(an_.location)
        dst_an_coll.append(
            cm.Annotation(
                dumps(transform(to_raw_bl, geom)),
                clean_cy.id,
                an_.term,
                clean_cy.project,
            )
        )

    print(f"{len(dst_an_coll)=}")

    if dst_an_coll.save() is False:
        raise ValueError("some annotations could not be saved")


def cleanup_img(img_id: int):
    img = cm.ImageInstance().fetch(img_id)
    if img is False:
        raise ValueError("unable to fetch image")

    an_coll = cm.AnnotationCollection()
    an_coll.image = img_id
    an_coll.user = 534530561
    an_coll = an_coll.fetch()
    if an_coll is False:
        raise ValueError("unable to fetch annotations")

    an_: cm.Annotation
    for an_ in an_coll:
        an_.delete()


# with cytomine.Cytomine(CYT_HOST, CYT_PUB_KEY, CYT_PRV_KEY):
#     cleanup_img(542625425)

# with cytomine.Cytomine(CYT_HOST, CYT_PUB_KEY, CYT_PRV_KEY):
#     copy_annotations(ImageRegistration(w_border_id=545025763, clean_id=542625425))
#     copy_annotations(ImageRegistration(w_border_id=548365416, clean_id=542625396))
#     copy_annotations(ImageRegistration(w_border_id=548365463, clean_id=542625387))

raise NotImplementedError("uncomment something else first")
