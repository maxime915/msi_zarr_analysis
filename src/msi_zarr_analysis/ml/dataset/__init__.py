"dataset: defines some representations for data source from Zarr & Cytomine."

import abc
import logging
import pathlib
from typing import Dict, Iterator, NamedTuple, Optional, Set, Tuple, Union, Sequence

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
from msi_zarr_analysis.utils.cytomine_utils import iter_annotation_single_term
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
        return train_test_split(*self.as_table(), **kwargs)  # type: ignore

    @abc.abstractmethod
    def attribute_names(self) -> Sequence[str]:
        """List of names matching the attributes.
        May raise an error for datasets where the number of attribute is not
        constant. See Dataset.is_table_like .

        Returns
        -------
        List[str]
            a list of name, matching the attributes in order
        """
        ...

    @abc.abstractmethod
    def class_names(self) -> Sequence[str]:
        """List of names matching the classes.
        May return an empty list if no names are found.

        Returns:
            List[str]: list of classes, matching the class index in the targets
        """

    def __raw_check_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        ds_x, ds_y = self.as_table()
        n_features = ds_x.shape[1]

        assert ds_x.shape[0] == ds_y.shape[0], f"{ds_x.shape=} vs {ds_y.shape=}"
        logging.info("ds_x.shape=%s", ds_x.shape)

        # check X : look for correlations between the samples
        corr = np.zeros((n_features, n_features))
        # for i in range(1, n_features):
        #     for j in range(i + 1, n_features):
        #         corr[i, j] = np.corrcoef(ds_x[:, i], ds_x[:, j])[0, 1]
        corr[:] = np.nan

        # ensure symmetric matrix with unitary diagonal
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # check Y : look for class imbalance
        _, occurrences = np.unique(ds_y, return_counts=True)

        return corr, occurrences

    def check_dataset(
        self,
        cache: bool = False,
        print_: bool = False,
        print_once: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """identify danger of feature correlations and class imbalance in the
        tabular dataset.

        May raise an error for datasets where the number of attribute is not
        constant. See Dataset.is_table_like .

        Args:
            cache (bool, optional): cache the computation. Defaults to False.
            print_ (bool, optional): log results with INFO level. Defaults to False.
            print_once (bool, optional): subsequent calls to this function don't print more than once. Defaults to False.

        Returns:
            Tuple[np.ndarray, float]: the correlation matrix and the highest relative occurrence
        """

        cache_attr_name = "__cached_check_dataset"

        if cache and hasattr(self, cache_attr_name):
            corr, occurrences = getattr(self, cache_attr_name)
        else:
            corr, occurrences = self.__raw_check_dataset()
            if cache:
                setattr(self, cache_attr_name, (corr, occurrences))

        single_print_attr_name = "__cached_single_print"

        if print_ and not (print_once and hasattr(self, single_print_attr_name)):
            setattr(self, single_print_attr_name, True)

            # logging.info("checking inter-feature correlation:")
            # for i in range(corr.shape[0]):
            #     for j in range(i + 1, corr.shape[1]):
            #         if np.abs(corr[i, j]) > 0.8:
            #             logging.info("i=%02d, j=%02d, corr=%.4f", i, j, corr[i, j])

            logging.info("checking for class imbalance:")

            n_classes = occurrences.size
            n_items = np.sum(occurrences)

            # information about the occurrence of each class
            occurrence_per_class = dict(zip(self.class_names(), occurrences))
            logging.info("occurrence_per_class: %s", occurrence_per_class)
            logging.info("max rel. occurrence = %.4f", np.max(occurrences) / n_items)
            logging.info("min rel. occurrence = %.4f", np.min(occurrences) / n_items)
            logging.info(". . .  1 / #classes = %.4f", 1 / n_classes)

        # largest relative occurrence
        imbalance = np.max(occurrences) / np.sum(occurrences)

        return corr, imbalance


class Tabular(Dataset):
    def __init__(
        self,
        dataset_x: np.ndarray,
        dataset_y: np.ndarray,
        attributes_names: Sequence[str],
        classes_names: Sequence[str],
    ) -> None:
        super().__init__()

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.attribute_names_ = attributes_names
        self.classes_names_ = classes_names

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        yield from zip(self.dataset_x, self.dataset_y)

    def is_table_like(self) -> bool:
        return True

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.dataset_x, self.dataset_y

    def attribute_names(self) -> Sequence[str]:
        if not self.attribute_names_:
            raise ValueError("no attribute name found")
        return self.attribute_names_

    def class_names(self) -> Sequence[str]:
        if not self.classes_names_:
            raise ValueError("no class name found")
        return self.classes_names_


class GroupCollection(NamedTuple):
    "tuple of a dataset and mask that associate a group index for all samples"

    dataset: Tabular
    groups: np.ndarray

    @property
    def data(self):
        "inputs of the dataset, as a (n_sample, n_features) array"
        return self.dataset.dataset_x

    @property
    def target(self):
        "targets of the dataset, as a (n_sample, n_output) array or (n_sample,) array"
        return self.dataset.dataset_y

    @staticmethod
    def merge_datasets(*sets: Tabular) -> "GroupCollection":

        class_names = sets[0].class_names()
        attribute_names = sets[0].attribute_names()

        # check for coherence
        for idx, dataset in enumerate(sets[1:], start=1):
            if dataset.class_names() != class_names:
                raise ValueError(
                    f"incoherent class names at {idx=}: "
                    f"{class_names} VS {dataset.class_names()}"
                )
            if dataset.attribute_names() != attribute_names:
                raise ValueError(
                    f"incoherent attribute names at {idx=}: "
                    f"{attribute_names} VS {dataset.attribute_names()}"
                )

        pairs = [(s.dataset_x, s.dataset_y) for s in sets]

        lengths = []
        for idx, (x, y) in enumerate(pairs):
            lengths.append(len(x))
            if len(x) != len(y):
                raise ValueError(
                    f"inconsistent sizes at {idx=}: {len(x)=} VS {len(y)=}"
                )
        offsets = np.cumsum(lengths)

        groups = np.empty(dtype=int, shape=(offsets[-1],))

        lo = 0
        for idx, hi in enumerate(offsets):
            groups[lo:hi] = idx
            lo = hi

        xs, ys = zip(*pairs)
        dataset_x = np.concatenate(xs)
        dataset_y = np.concatenate(ys)

        return GroupCollection(
            Tabular(dataset_x, dataset_y, attribute_names, class_names),
            groups,
        )

    @staticmethod
    def merge_collections(*collections: "GroupCollection") -> "GroupCollection":

        # build collection with incorrect grouping
        collection = GroupCollection.merge_datasets(*[ds.dataset for ds in collections])

        # correct grouping
        start = 0
        baseline = 0
        for collection_ in collections:
            end = start + len(collection_.groups)
            collection.groups[start:end] = baseline + collection_.groups

            baseline += np.unique(collection_.groups).size
            start = end

        return collection


class MergedDS(Dataset):
    """MergedDS: (lazily) merge multiple datasets into one

    NOTE There could be an optimization to flatten the tree, in case one of the inner
    datasets is also a MergedDS. self.as_table() would probably benefit from this.
    """

    def __init__(
        self,
        *datasets: Dataset,
    ) -> None:

        super().__init__()

        if not datasets:
            raise ValueError("cannot merge en empty list of datasets")

        first, *rest = datasets
        attr_names = first.attribute_names()
        cls_names = first.class_names()

        # check for consistency
        for idx, item in enumerate(rest, start=1):
            if item.attribute_names() != attr_names:
                raise ValueError(
                    (
                        f"inconsistent attribute names at position {idx}: "
                        f"{item.attribute_names()} (expected {attr_names}"
                    )
                )

            if item.class_names() != cls_names:
                raise ValueError(
                    (
                        f"inconsistent class names at position {idx}: "
                        f"{item.class_names()} (expected {cls_names}"
                    )
                )

        self.attribute_names_ = attr_names
        self.class_names_ = cls_names
        self.datasets = datasets

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        for ds in self.datasets:
            yield from ds.iter_rows()

    def is_table_like(self) -> bool:
        return all(ds.is_table_like() for ds in self.datasets)

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        ds_x, ds_y = zip(*(ds.as_table() for ds in self.datasets))
        return np.concatenate(ds_x), np.concatenate(ds_y)

    def attribute_names(self) -> Sequence[str]:
        return self.attribute_names_

    def class_names(self) -> Sequence[str]:
        return self.class_names_


class ZarrAbstractDataset(Dataset):
    def __init__(
        self,
        data_zarr_path: str,
        classes: Union[
            Dict[str, npt.NDArray[np.bool_]],
            Dict[str, str],
            Sequence[Union[str, pathlib.Path]],
        ],
        roi_mask: Union[str, None, npt.NDArray[np.bool_]] = None,
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
            def _normalize(key: str, value: Union[str, npt.NDArray[np.bool_]]):
                if isinstance(value, (pathlib.Path, str)):
                    return load_img_mask(value)
                elif isinstance(value, np.ndarray):
                    return value
                else:
                    raise ValueError(f"classes[{key}] has invalid type={type(value)}")
            
            classes = {k: _normalize(k, v) for k, v in classes.items()}

        else:
            raise ValueError(f"classes has invalid type {type(classes)}")

        if roi_mask:
            if not isinstance(roi_mask, (str, pathlib.Path, np.ndarray)):
                raise ValueError(f"roi_mask has invalid type {type(roi_mask)}")

        self.z = open_group_ro(data_zarr_path)

        if isinstance(roi_mask, (str, pathlib.Path)):
            roi_mask = load_img_mask(roi_mask)

        self.cls_mask, _, self.class_names_ = build_class_masks(
            self.z, classes, roi_mask, background_class
        )

        self.y_slice, self.x_slice = clean_slice_tuple(
            self.z["/0"].shape[2:], y_slice, x_slice
        )

        self._cached_ds = None

    def class_names(self) -> Sequence[str]:
        return self.class_names_


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
            Dict[str, npt.NDArray[np.bool_]],
            Dict[str, str],
            Sequence[Union[str, pathlib.Path]],
        ],
        roi_mask: Union[str, None, npt.NDArray[np.bool_]] = None,
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
            self.z, self.cls_mask, self.y_slice, self.x_slice
        )

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_ds:
            self._cached_ds = self.__load_ds()
        return self._cached_ds

    def get_dataset_x(self) -> npt.NDArray:
        return self.as_table()[0]

    def get_dataset_y(self) -> npt.NDArray:
        return self.as_table()[1]

    def attribute_names(self) -> Sequence[str]:
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
            Dict[str, npt.NDArray[np.bool_]],
            Dict[str, str],
            Sequence[Union[str, pathlib.Path]],
        ],
        bin_lo: npt.NDArray,
        bin_hi: npt.NDArray,
        roi_mask: Union[str, None, npt.NDArray[np.bool_]] = None,
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
            self.cls_mask,
            self.y_slice,
            self.x_slice,
            self.bin_lo,
            self.bin_hi,
        )

    def as_table(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if not self._cached_ds:
            self._cached_ds = self.__load_ds()
        return self._cached_ds

    def get_dataset_x(self) -> npt.NDArray:
        return self.as_table()[0]

    def get_dataset_y(self) -> npt.NDArray:
        return self.as_table()[1]

    def attribute_names(self) -> Sequence[str]:
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
        self.term_set = term_set or set()

        self.cache_data = bool(cache_data)
        self._cached_table = None

        self.cached_term_lst = None

    def iter_rows(self) -> Iterator[Tuple[npt.NDArray, int]]:

        if self.cache_data:
            for row in zip(*self.as_table()):
                yield row
            return

        for profile, class_idx in self.__raw_iter():
            yield np.array(profile), class_idx

    def __raw_iter(self) -> Iterator[Tuple[npt.NDArray, int]]:
        term_collection = TermCollection().fetch_with_filter("project", self.project_id)
        if term_collection is False:
            raise ValueError("cannot fetch term collection")
        annotations = AnnotationCollection(
            project=self.project_id, image=self.image_id, showTerm=True, showWKT=True
        ).fetch()
        if annotations is False:
            raise ValueError("cannot fetch annotations")

        term_lst: Sequence[str] = []

        for annotation, term in iter_annotation_single_term(
            annotations,
            term_collection,
        ):
            term_name: str = term.name  # type: ignore

            if term_name not in self.term_set:
                continue

            try:
                term_idx = term_lst.index(term_name)
            except ValueError:
                term_idx = len(term_lst)
                term_lst.append(term_name)

            profile_ = annotation.profile()
            if profile_ is False:
                raise ValueError("cannot get profile")
            for profile in profile_:
                yield profile["profile"], term_idx

        self.cached_term_lst = term_lst

    def __load_ds(self) -> Tuple[npt.NDArray, npt.NDArray]:
        attributes, classes = zip(*self.__raw_iter())
        dtype = type(attributes[0][0])  # type: ignore
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

    def attribute_names(self) -> Sequence[str]:
        slices = SliceInstanceCollection().fetch_with_filter("imageinstance", self.image_id)
        if slices is False:
            raise ValueError("cannot fetch slices of image")
        return [xySlice.zName for xySlice in slices]

    def class_names(self) -> Sequence[str]:
        if self.cached_term_lst is None:
            for _ in self.__raw_iter():
                pass
            return self.class_names()

        return self.cached_term_lst
