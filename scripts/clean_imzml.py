"""
- read an imzml file
- find the length (in values) of each spectra
- find the largest *connected-component* and remove the rest
- find "outliers" in lengths (first spot has more values) and remove it
- create a new imzml file with the remaining spectra
"""

import collections
import pathlib
import re
import typing

import numpy as np
import pandas as pd
from scipy.ndimage import label

"""
moralité : s'assurer d'avoir les données avant de se préoccuper de l'implémentation

- à refaire en zarr
- probablement moins chiant de virer le reste et adapter les métadonnées
- chunking à refaire :'(

"""


def find_unique_accessors(
    text: str, *accessor_re: str
) -> typing.Mapping[str, re.Match]:

    if not accessor_re:
        return collections.OrderedDict({})

    re_lst = {
        pattern: re.compile(rf'<cvParam [^>]*accession="{pattern}"[^>]*>')
        for pattern in accessor_re
    }

    matches = {key: must_search(pattern, text) for key, pattern in re_lst.items()}

    def sort_key(match_item: tuple[str, re.Match]):
        if not match_item[1]:
            return -1
        return match_item[1].start(0)

    return collections.OrderedDict(sorted(matches.items(), key=sort_key))


def get_encoding(path: pathlib.Path) -> str:
    # figure out encodings
    with open(path, mode="rb") as source_file:
        line = next(source_file, b"")
        if not line:
            raise ValueError(f"empty file {path}")

        prefix = b'encoding="'
        idx_start = line.index(prefix) + len(prefix)

        idx_end = line.index(b'"', idx_start)

        encoding = line[idx_start:idx_end].decode("ASCII")

        return encoding


def must_search(pattern: str | re.Pattern[str], data: str, pos: int = 0):
    if isinstance(pattern, re.Pattern):
        match = pattern.search(data, pos)
        pattern = pattern.pattern
    elif isinstance(pattern, str):
        if pos != 0:
            raise ValueError(f"{pos=!r} can only be specified with re.Pattern object")
        match = re.search(pattern, data)
    else:
        raise TypeError(f"unsupported {pattern=!r}")

    if not match:
        raise ValueError(f"could not find {pattern}")

    return match


class Spectrum(typing.NamedTuple):
    imzml_pos: int
    imzml_index: int
    coord_y: int
    coord_x: int
    length: int


def find_all_spectra(imzml_data: str):
    ""
    spectrum_start_re = re.compile(r"<spectrum ")
    spectrum_end_re = re.compile(r"</spectrum>")
    index_re = re.compile(r'index="(\d+)"')
    pos_y_re = re.compile(r'<cvParam [^>]*accession="IMS:1000051"[^>]*>')
    pos_x_re = re.compile(r'<cvParam [^>]*accession="IMS:1000050"[^>]*>')
    s_len_re = re.compile(r'<cvParam [^>]*accession="IMS:1000103"[^>]*>')
    value_re = re.compile(r'value="(\d+)"')

    def _get(pattern: re.Pattern[str], pos: int):
        attr = must_search(pattern, imzml_data, pos)
        return must_search(value_re, attr.string).string[7:-1]

    def _extract_at(pos: int):
        return Spectrum(
            pos,
            int(must_search(index_re, imzml_data, pos).string[7:-1]),
            int(_get(pos_y_re, pos)),
            int(_get(pos_x_re, pos)),
            int(_get(s_len_re, pos)),
        )

    values: list[Spectrum] = []

    idx = 0
    while True:
        start = spectrum_start_re.search(imzml_data, pos=idx)
        if not start:
            break
        end = must_search(spectrum_end_re, imzml_data, start.end())
        values.append(_extract_at(start.end()))
        idx = end.end()

    return pd.DataFrame.from_records(values, columns=Spectrum._fields)


def build_mask(df: pd.DataFrame):
    ""

    df_cpy = df.copy()
    height: int = df_cpy["coord_y"].max()
    width: int = df_cpy["coord_y"].max()
    # 1 based indexing
    df_cpy["coord_y"] -= 1
    df_cpy["coord_x"] -= 1

    len_np = np.zeros((height, width), dtype=int)
    len_np[df_cpy["coord_y"], df_cpy["coord_x"]] = df_cpy["length"]

    # find the largest connected components
    labeled, nComponents = label(len_np > 0)

    # which one is the highest
    value, counts = np.unique(labeled, return_counts=True)
    mask = [value == value[np.argmax(counts)]]

    # only keep the highest components
    len_np = len_np[mask]

    # find outliers
    len_med = np.median(len_np[len_np > 0])
    diff_to_med = np.where(len_np > 0, np.abs(len_np - len_med), 0)
    med_dev = np.median(diff_to_med[len_np > 0])

    outliers_mask = (diff_to_med > 2.0 * med_dev)
    ys, xs = np.nonzero(outliers_mask)
    for y, x in zip(ys, xs):
        print(f"outlier at {y=} {x=} with {len_np[y, x]=}")
    print("done")


def clean_imzml(
    source: pathlib.Path,
):
    ""
    try:
        with open(
            source,
            mode="r",
            encoding=get_encoding(source),
        ) as source_file:
            content = source_file.read()
    except MemoryError as err:
        raise ValueError("XML file is too large") from err

    # TODO build a mask of all x-y values
    spectra = find_all_spectra(content)
    _ = build_mask(spectra)


class Clipper:
    def __init__(self, clip_x: int, clip_y: int) -> None:
        self.idx = 0
        self.data = ""
        self.clip_x = clip_x
        self.clip_y = clip_y

        self.counted = 0

        if not isinstance(clip_x, int) or clip_x < 1:
            raise ValueError(f"{clip_x=} should be a positive integer")
        if not isinstance(clip_y, int) or clip_y < 1:
            raise ValueError(f"{clip_y=} should be a positive integer")

    def parse(self, source: pathlib.Path, dest: pathlib.Path):

        # open text file & load it in memory at once
        try:
            encoding = get_encoding(source)
            with open(source, mode="r", encoding=encoding) as source_file:
                self.data = source_file.read()
        except MemoryError as err:
            raise ValueError("XML file is too large") from err

        # remove spectra
        with open(dest, mode="w", encoding=encoding) as dest_file:
            for part in self.content():
                dest_file.write(part)

        # re-read data
        with open(dest, mode="r", encoding=encoding) as dest_file:
            self.data = dest_file.read()

        # write data with fixed counts
        self.idx = 0
        with open(dest, mode="w", encoding=encoding) as dest_file:
            for part in self.fix_count():
                dest_file.write(part)
            for part in self.iter_footer():
                dest_file.write(part)

    def _yield_until_match(self, match: re.Match | None):
        "make sure all text before match has been yielded"
        if not match:
            return
        low = match.start(0)
        if low != self.idx:
            yield self.data[self.idx: low]
            self.idx = low

    def fix_count(self):
        # get spectrumList definition
        spectrum_list = must_search(r"<spectrumList [^>]*>", self.data)
        yield from self._yield_until_match(spectrum_list)

        count_kv = must_search(r'count="(\d+)"', spectrum_list[0])[0]

        self.idx = spectrum_list.end(0)
        yield spectrum_list[0].replace(count_kv, f'count="{self.counted}"')

    def iter_header(self):
        """iterate and update header accessor regarding: max count pixel x \
        (IMS:1000042), max count pixel y (IMS:1000043)"""

        def update_width(match: re.Match[str]) -> str:
            """
            - expects that self.idx == match.start(0)
            - update match[0] to set the width to *= self.repeat_x
            - yield the updated accessor
            - set self.idx == match.end(0)
            """
            attr = match[0]
            width_kv = must_search(r'value="(\d+)"', attr)[0]

            self.idx = match.end(0)
            return attr.replace(width_kv, f'value="{self.clip_x}"')

        def update_height(match: re.Match[str]) -> str:
            """
            - expects that self.idx == match.start(0)
            - update match[0] to set the height to *= self.repeat_y
            - yield the updated accessor
            - set self.idx == match.end(0)
            """
            attr = match[0]
            height_kv = must_search(r'value="(\d+)"', attr)[0]

            self.idx = match.end(0)
            return attr.replace(height_kv, f'value="{self.clip_y}"')

        param_update = {
            "IMS:1000042": update_width,
            "IMS:1000043": update_height,
        }

        matches = find_unique_accessors(self.data, *param_update.keys())

        for match in matches.items():
            yield from self._yield_until_match(match[1])
            yield param_update[match[0]](match[1])

    def iter_spectra(self):

        spectrum_start_re = re.compile(r"<spectrum ")
        spectrum_end_re = re.compile(r"</spectrum>")
        index_re = re.compile(r'index="(\d+)"')
        pos_x_re = re.compile(r'<cvParam [^>]*accession="IMS:1000050"[^>]*>')
        pos_y_re = re.compile(r'<cvParam [^>]*accession="IMS:1000051"[^>]*>')
        value_re = re.compile(r'value="(\d+)"')

        def yield_filtered_spectrum(original: str):
            # find idx
            index_kv = must_search(index_re, original)[0]
            old_idx = int(index_kv[7:-1])

            # find x, y
            x_attr = must_search(pos_x_re, original)[0]
            x_kv = must_search(value_re, x_attr)[0]
            old_x = int(x_kv[7:-1])

            y_attr = must_search(pos_y_re, original)[0]
            y_kv = must_search(value_re, y_attr)[0]
            old_y = int(y_kv[7:-1])

            if old_x > self.clip_x:
                return
            if old_y > self.clip_y:
                return

            spectrum = original.replace(index_kv, f'value="{self.counted}"')
            spectrum = spectrum.replace(
                f"spectrum={old_idx}", f"spectrum={self.counted}"
            )

            self.counted += 1

            yield spectrum

        while True:
            start = spectrum_start_re.search(self.data, pos=self.idx)
            end = spectrum_end_re.search(self.data, self.idx)

            if start is None:
                break
            if end is None:
                raise ValueError(f"{spectrum_end_re.pattern!r} not found in ImzML data after {self.idx}")

            yield from self._yield_until_match(start)

            # get spectrum
            spectrum = self.data[start.start(0): end.end(0)]

            yield from yield_filtered_spectrum(spectrum)

            self.idx = end.end(0)

    def iter_footer(self):
        yield self.data[self.idx:]
        self.idx = -1

    def content(self):
        yield from self.iter_header()
        yield from self.iter_spectra()
        yield from self.iter_footer()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('imzml_in', type=str)
    # parser.add_argument('imzml_out', type=str)
    # parser.add_argument('clip_x', type=int)
    # parser.add_argument('clip_y', type=int)

    args = parser.parse_args()

    clean_imzml(args.imzml_in)
    # Clipper(args.clip_x, args.clip_y).parse(args.imzml_in, args.imzml_out)
