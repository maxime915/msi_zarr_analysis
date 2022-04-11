"dataset: defines some representations for data source from Zarr & Cytomine."

import abc
import pathlib
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from cytomine.models import (
    AnnotationCollection,
    SliceInstanceCollection,
    TermCollection,
)
from msi_zarr_analysis.cli.utils import load_img_mask
from msi_zarr_analysis.ml.dataset.utils import (
    bin_array_dataset,
    build_class_masks,
    nonbinned_array_dataset,
)
from msi_zarr_analysis.utils.check import open_group_ro
from msi_zarr_analysis.utils.iter_chunks import clean_slice_tuple


class Dataset(abc.ABC):
    @abc.abstractmethod
    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        """Iterate all rows of attribute and class from the dataset

        Yields
        ------
        Iterator[Tuple[npt.NDArray, npt.NDArray]]
            pair of (attributes, class)
        """
        ...

    @abc.abstractmethod
    def is_table_like(self) -> bool:
        "If true, the dataset has a fixed number of attribute for all rows."
        ...

    @abc.abstractmethod
    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Whole dataset as a pair of table of rows.
        May raise an error for datasets where the number of attribute is not
        constant. See Dataset.is_table_like .

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray]
            pair of (table of attributes, table of classes)
        """
        ...

    def as_train_test_tables(
        self, **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Whole dataset, splitted as train et test pairs.
        May raise an error for datasets where the number of attribute is not
        constant. See Dataset.is_table_like .

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
            X_train, X_test, y_train, y_test
        """
        return train_test_split(*self.as_table(), **kwargs)

    @abc.abstractmethod
    def attribute_names(self) -> List[str]:
        """List of names matching the attributes.
        May raise an error for datasets where the number of attribute is not
        constant. See Dataset.is_table_like .

        Returns
        -------
        List[str]
            a list of name, matching the attributes in order
        """
        ...


class ZarrAbstractDataset(Dataset):
    def __init__(
        self,
        data_zarr_path: str,
        classes: Union[
            Dict[str, npt.NDArray[np.dtype("bool")]],
            Dict[str, str],
            List[Union[str, pathlib.Path]],
        ],
        roi_mask: Union[str, None, npt.NDArray[np.dtype("bool")]] = None,
        background_class: bool = False,
        y_slice: slice = slice(None),
        x_slice: slice = slice(None),
    ) -> None:
        super().__init__()

        if not classes:
            raise ValueError("empty class set")

        if isinstance(classes, (list, tuple)):
            for idx, cls in enumerate(classes):
                if not isinstance(cls, (str, pathlib.Path)):
                    raise ValueError(
                        f"classes[{idx}] path has invalid type {type(cls)}"
                    )
            classes = {pathlib.Path(cls).stem: load_img_mask(cls) for cls in classes}
        elif isinstance(classes, dict):
            classes = classes.copy()
            for key, value in classes.items():
                if isinstance(value, (pathlib.Path, str)):
                    classes[key] = load_img_mask(value)
                elif isinstance(value, np.ndarray):
                    continue
                else:
                    raise ValueError(f"classes[{key}] has invalid type {type(value)}")

        else:
            raise ValueError(f"classes has invalid type {type(classes)}")

        if roi_mask:
            if not isinstance(roi_mask, (str, pathlib.Path, np.ndarray)):
                raise ValueError(f"roi_mask has invalid type {type(roi_mask)}")

        self.z = open_group_ro(data_zarr_path)

        if isinstance(roi_mask, np.ndarray):
            roi_mask = load_img_mask(roi_mask)

        self.cls_mask, self.roi_mask = build_class_masks(
            self.z, classes, roi_mask, background_class
        )

        self.y_slice, self.x_slice = clean_slice_tuple(
            self.z["/0"].shape[2:], y_slice, x_slice
        )

        self._cached_ds = None


class ZarrContinuousNonBinned(ZarrAbstractDataset):
    """Read an OME-Zarr MSI data from a path. The channels (or masses, m/Z) will
    be used as the attributes' keys while the intensities will be used as the
    attributes' values. No additional preprocessing step is applied to the data
    before interpretation.

    An ROI may be supplied : only coordinates (x, y) present in the ROI will be
    studied.

    Classes segmentation masks may be submitted via numpy mask arrays (or path
    to such arrays encoded as greyscale image format).

    Analysis space may be restraining to a rectangle using slices for the Y and
    X axes.
    """

    def __init__(
        self,
        data_zarr_path: str,
        classes: Union[
            Dict[str, npt.NDArray[np.dtype("bool")]],
            Dict[str, str],
            List[Union[str, pathlib.Path]],
        ],
        roi_mask: Union[str, None, npt.NDArray[np.dtype("bool")]] = None,
        background_class: bool = False,
        y_slice: slice = slice(None),
        x_slice: slice = slice(None),
    ) -> None:
        super().__init__(
            data_zarr_path, classes, roi_mask, background_class, y_slice, x_slice
        )
        binary_mode = self.z.attrs["pims-msi"]["binary_mode"]
        if binary_mode != "continuous":
            raise ValueError(f"invalid {binary_mode=}: expected 'continuous'")

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        dataset_x, dataset_y = self.as_table()

        for row_x, row_y in zip(dataset_x, dataset_y):
            yield row_x, row_y

    def is_table_like(self) -> bool:
        return True

    def __load_ds(self) -> Tuple[npt.NDArray, npt.NDArray]:
        "heavy lifting: call to utils"
        return nonbinned_array_dataset(
            self.z, self.roi_mask, self.cls_mask, self.y_slice, self.x_slice
        )

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_ds:
            self._cached_ds = self.__load_ds()
        return self._cached_ds

    def get_dataset_x(self) -> Tuple[npt.NDArray]:
        return self.as_table()[0]

    def get_dataset_y(self) -> Tuple[npt.NDArray]:
        return self.as_table()[1]

    def attribute_names(self) -> List[str]:
        return [str(v) for v in self.z["/labels/mzs/0"][:, 0, 0, 0]]


class ZarrProcessedBinned(ZarrAbstractDataset):
    """Read an OME-Zarr MSI data from a path. The bins for the channel (or
    masses, m/Z) will be used as the attributes' keys (formally, mean of bin ±
    half width of bin) while the sum of the intensities from that bin will be
    used as the attributes' values. No additional preprocessing step is applied
    to the data before interpretation.

    An ROI may be supplied : only coordinates (x, y) present in the ROI will be
    studied.

    Classes segmentation masks may be submitted via numpy mask arrays (or path
    to such arrays encoded as greyscale image format).

    Analysis space may be restraining to a rectangle using slices for the Y and
    X axes.
    """

    def __init__(
        self,
        data_zarr_path: str,
        classes: Union[
            Dict[str, npt.NDArray[np.dtype("bool")]],
            Dict[str, str],
            List[Union[str, pathlib.Path]],
        ],
        bin_lo: npt.NDArray,
        bin_hi: npt.NDArray,
        roi_mask: Union[str, None, npt.NDArray[np.dtype("bool")]] = None,
        background_class: bool = False,
        y_slice: slice = slice(None),
        x_slice: slice = slice(None),
    ) -> None:
        super().__init__(
            data_zarr_path, classes, roi_mask, background_class, y_slice, x_slice
        )
        self.bin_lo = bin_lo
        self.bin_hi = bin_hi

        binary_mode = self.z.attrs["pims-msi"]["binary_mode"]
        if binary_mode != "processed":
            raise ValueError(f"invalid {binary_mode=}: expected 'processed'")

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        dataset_x, dataset_y = self.as_table()

        for row_x, row_y in zip(dataset_x, dataset_y):
            yield row_x, row_y

    def is_table_like(self) -> bool:
        return True

    def __load_ds(self) -> Tuple[npt.NDArray, npt.NDArray]:
        "heavy lifting: call to utils"
        return bin_array_dataset(
            self.z,
            self.roi_mask,
            self.cls_mask,
            self.y_slice,
            self.x_slice,
            self.bin_hi,
            self.bin_hi,
        )

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_ds:
            self._cached_ds = self.__load_ds()
        return self._cached_ds

    def get_dataset_x(self) -> Tuple[npt.NDArray]:
        return self.as_table()[0]

    def get_dataset_y(self) -> Tuple[npt.NDArray]:
        return self.as_table()[1]

    def attribute_names(self) -> List[str]:
        return [
            f"{0.5*(lo + hi)} ± {0.5*(hi - lo)}"
            for lo, hi in zip(self.bin_lo, self.bin_hi)
        ]


class CytomineNonBinned(Dataset):
    def __init__(
        self,
        project_id: int,
        image_id: int,
        term_set: Optional[Set[str]] = None,
        cache_data: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        project_id : int
        image_id : int
        term_set : Optional[Set[str]], optional
            whitelist of term to load, by default None (all terms loaded)
        cache_data : bool, optional
            data must be tabular, by default True
        """
        super().__init__()
        self.project_id = project_id
        self.image_id = image_id
        self.term_set = term_set

        self.cache_data = bool(cache_data)
        self._cached_table = None

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        
        if self.cache_data:
            for row in zip(*self.as_table()):
                yield row
            return

        for profile, class_idx in self.__raw_iter():
            yield np.array(profile), class_idx

    def __raw_iter(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        term_collection = TermCollection().fetch_with_filter("project", self.project_id)
        annotations = AnnotationCollection(
            project=self.project_id, image=self.image_id, showTerm=True, showWKT=True
        ).fetch()

        # TODO is it true that an annotation always has 0 or 1 terms ?
        term_set = {
            term_collection.find_by_attribute("id", a.term[0]).name
            for a in annotations
            if a.term
        }
        term_lst = list(term_set)

        # if given, only consider given terms
        if self.term_set:
            term_set = self.term_set & term_set

        for annotation in annotations:
            if not annotation.term:
                continue
            term_name = term_collection.find_by_attribute("id", annotation.term[0]).name

            if term_name not in term_set:
                continue

            for profile in annotation.profile():
                yield profile["profile"], term_lst.index(term_name)

    def __load_ds(self) -> Tuple[npt.NDArray, npt.NDArray]:
        attributes, classes = zip(*self.iter_rows())
        dtype = type(attributes[0][0])
        return np.array(attributes, dtype=dtype), np.array(classes)

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_table:
            self._cached_table = self.__load_ds()
    
        if not self.cache_data:
            # remove cache if it shouldn't be there
            tmp, self._cached_table = self._cached_table, None
            return tmp

        return self._cached_table

    def is_table_like(self) -> bool:
        try:
            _ = self.as_table()
            return True
        except (ValueError, IndexError):
            return False

    def attribute_names(self) -> List[str]:
        return [
            xySlice.zName
            for xySlice in SliceInstanceCollection().fetch_with_filter(
                "imageinstance", self.image_id
            )
        ]
