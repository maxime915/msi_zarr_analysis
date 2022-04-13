import itertools

import numpy as np

from msi_zarr_analysis.ml.dataset.translated_t_m import TemplateTransform


def test_coordinates():

    source = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
        ]
    )

    rot90 = range(-5, 6)
    flip_ud = (False, True)
    flip_lr = (False, True)

    for r, ud, lr in itertools.product(rot90, flip_ud, flip_lr):
        transform = TemplateTransform(r, ud, lr)

        y_s, x_s = source.nonzero()

        transformed = transform.transform_template(source)

        y_t, x_t = transformed.nonzero()

        # y_p is a permutation of y_s
        y_p, x_p, _ = transform.inverse_transform_coordinate(
            y_t, x_t, transformed.shape
        )

        # build points and sort them to make the comparison
        pairs_s = sorted(list(zip(y_s, x_s)))
        pairs_p = sorted(list(zip(y_p, x_p)))

        assert pairs_s == pairs_p
