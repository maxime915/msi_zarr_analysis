from functools import partial
from typing import Literal, Iterable

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

import omezarrmsi as ozm
from omezarrmsi.utils.axial import Shape, AxialMapping, Axis


Label = Literal["ls+", "ls-", "sc+", "sc-"]


def _get_coordinates(mask: npt.NDArray, shape: Shape):
    nz_coords = mask.nonzero()

    tpl_coords: Iterable[tuple[int, ...]] = zip(*nz_coords)
    return [AxialMapping(dict(zip(shape.keys(), c))) for c in tpl_coords]


def _pad_to(tsr: torch.Tensor, size: int, axis: int):
    pad_r = size - tsr.shape[1]
    assert pad_r >= 0
    if not pad_r:
        return tsr

    pad_ = (0, 0) * axis + (0, pad_r) + (tsr.ndim - axis - 1) * (0, 0)
    assert len(pad_) == 2 * tsr.ndim

    tsr = pad(tsr, pad_[::-1])
    return tsr


def _pad_seq_to(size: int, arr: Iterable[np.ndarray]):
    tsr = pad_sequence(
        [torch.tensor(a, dtype=torch.float32) for a in arr],
        batch_first=True,
    )

    return _pad_to(tsr, size, axis=1)


class MSIDataset(Dataset):
    def __init__(
        self,
        mzs_: torch.Tensor,  # float[S, L]
        int_: torch.Tensor,  # float[S, L]
        y: torch.Tensor,  # int[S]
        w: torch.Tensor,  # float[S]
        coords: list[AxialMapping[int]],  # [S]
        ds_lst: list[ozm.OMEZarrMSI],  # [S]
        labels_pos: list[Label],
        labels_neg: list[Label],
    ) -> None:
        super().__init__()
        self.mzs_ = mzs_
        self.int_ = int_
        self.y = y
        self.w = w
        self.coords = coords
        self.ds_lst = ds_lst
        self.labels_pos = sorted(labels_pos)
        self.labels_neg = sorted(labels_neg)

    @staticmethod
    def load_full_images(
        ds: ozm.OMEZarrMSI,
        *,
        min_len_hint: int | None = None,
    ):
        min_len = ds.int_shape[Axis.C]
        if min_len_hint is not None:
            if min_len_hint < min_len:
                raise ValueError(f"{min_len_hint=!r} < computed {min_len}")
            min_len = min_len_hint

        z_len = ds.z_len
        assert len(z_len.shape) == 4
        assert z_len.shape[:2] == (1, 1)

        mask: np.ndarray = z_len[...] > 0  # type:ignore
        coords = _get_coordinates(mask, ds.int_shape)

        spectra = ds.fetch_spectra(*[c.without_axes(Axis.C) for c in coords])
        a_mzs = _pad_seq_to(min_len, (s_mzs for s_mzs, _ in spectra))
        a_int = _pad_seq_to(min_len, (s_int for _, s_int in spectra))

        _, _, ys, xs = mask.nonzero()

        return a_mzs, a_int, ys, xs

    @staticmethod
    def load(
        ds: ozm.OMEZarrMSI,
        *,
        label_pos: list[Label],
        label_neg: list[Label],
        min_len_hint: int | None = None,
    ):
        min_len = ds.int_shape[Axis.C]
        if min_len_hint is not None:
            if min_len_hint < min_len:
                raise ValueError(f"{min_len_hint=!r} < computed {min_len}")
            min_len = min_len_hint

        assert len(set(label_pos)) == len(label_pos), f"duplicates in {label_pos}"
        assert len(set(label_neg)) == len(label_neg), f"duplicates in {label_neg}"
        # union of all labels for data loading
        labels = list(set(label_pos).union(label_neg))
        assert len(labels) == len(label_pos) + len(label_neg), "overlap in labels"
        cls_per_label = [1 if lbl in label_pos else 0 for lbl in labels]

        masks = [ds.get_label(l_)[...] for l_ in labels]
        coords_per_mask = [_get_coordinates(m, ds.int_shape) for m in masks]

        # fetch the weights
        weights = [
            [m[c.tpl()] for c in c_]
            for m, c_ in zip(masks, coords_per_mask, strict=True)
        ]

        # 1-hot label
        class_idx = [
            [cls_per_label[idx]] * len(c_)  # all labels of this mask have this class
            for idx, c_ in enumerate(coords_per_mask)
        ]

        spectra = [
            ds.fetch_spectra(*[c.without_axes(Axis.C) for c in coords])
            for coords in coords_per_mask
        ]

        a_mzs = [
            _pad_seq_to(min_len, (s_mzs for s_mzs, _ in spec_lst))
            for spec_lst in spectra
        ]
        a_int = [
            _pad_seq_to(min_len, (s_int for _, s_int in spec_lst))
            for spec_lst in spectra
        ]

        return MSIDataset(
            mzs_=torch.cat(a_mzs, dim=0),
            int_=torch.cat(a_int, dim=0),
            y=torch.tensor(sum(class_idx, [])),
            w=torch.tensor(sum(weights, [])),
            coords=sum(coords_per_mask, []),
            ds_lst=[ds],
            labels_pos=label_pos,
            labels_neg=label_neg,
        )

    @staticmethod
    def cat(*datasets: "MSIDataset"):
        "concatenate multiple MSIDataset instances"

        assert datasets, "at least one must be provided"
        labels_p, labels_n = datasets[0].labels_pos, datasets[0].labels_neg
        if any(ds.labels_pos != labels_p or ds.labels_neg != labels_n for ds in datasets):
            raise ValueError(f"{datasets} have inconsistent labels")

        # add padding if necessary
        spec_len = max(d.mzs_.shape[1] for d in datasets)
        p = partial(_pad_to, size=spec_len, axis=1)

        return MSIDataset(
            mzs_=torch.cat([p(d.mzs_) for d in datasets], dim=0),
            int_=torch.cat([p(d.int_) for d in datasets], dim=0),
            y=torch.cat([d.y for d in datasets], dim=0),
            w=torch.cat([d.w for d in datasets], dim=0),
            coords=sum((d.coords for d in datasets), []),
            ds_lst=sum((d.ds_lst for d in datasets), []),
            labels_pos=labels_p,
            labels_neg=labels_n,
        )

    def to(self, device: torch.device):
        return MSIDataset(
            mzs_=self.mzs_.to(device),
            int_=self.int_.to(device),
            y=self.y.to(device),
            w=self.w.to(device),
            coords=self.coords,
            ds_lst=self.ds_lst,
            labels_pos=self.labels_pos,
            labels_neg=self.labels_neg,
        )

    def __len__(self):
        return len(self.mzs_)

    def __getitem__(self, index):
        return self.mzs_[index], self.int_[index], self.y[index], self.w[index]
