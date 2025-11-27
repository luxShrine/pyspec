from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger
import numpy as np
import polars as pl

from pyspectral.config import READY_DATA_DIR
from pyspectral.core import Cube, FlatMap, assert_same_grid
from pyspectral.data.preprocessing import (
    PeakNormConfig,
    PreConfig,
    PreprocStats,
    SameGridCubes,
    preprocess_cube,
)
from pyspectral.types import (
    Arr1DF,
    Arr2DF,
    ArrayF,
    ArrayI,
)

# -- data helpers

type AnyDict = dict[str, Any]


def extract_arrays[T](
    obj: T, prefix: str = ""
) -> tuple[AnyDict | T | list[Any], AnyDict]:
    arrays: AnyDict = {}
    meta: AnyDict | T | list[Any] = obj

    if is_dataclass(obj):
        meta = {}
        for field in obj.__dataclass_fields__.keys():
            value = getattr(obj, field)
            sub_meta, sub_arrays = extract_arrays(value, f"{prefix}{field}.")
            meta[field] = sub_meta
            arrays.update(sub_arrays)

    elif isinstance(obj, np.ndarray):
        key = prefix.rstrip(".")
        arrays[key] = obj
        meta = {"__array__": key, "shape": obj.shape, "dtype": str(obj.dtype)}

    elif isinstance(obj, list):
        new_meta = []
        for i, v in enumerate(obj):
            sub_meta, sub_arrays = extract_arrays(v, f"{prefix}{i}.")
            new_meta.append(sub_meta)
            arrays.update(sub_arrays)
        meta = new_meta

    elif isinstance(obj, dict):
        new_meta = {}
        for k, v in obj.items():
            sub_meta, sub_arrays = extract_arrays(v, f"{prefix}{k}.")
            new_meta[k] = sub_meta
            arrays.update(sub_arrays)
        meta = new_meta

    return meta, arrays


def save_outer(obj: Any, path: Path) -> None:
    path = Path(path)
    meta, arrays = extract_arrays(obj)

    # save arrays
    np.savez_compressed(path.with_suffix(".npz"), **arrays)

    # save metadata (including array keys)
    with path.with_suffix(".json").open("w") as f:
        json.dump(meta, f, indent=2)


def restore_arrays(meta: Any, arrays: AnyDict) -> list[Any] | AnyDict:
    if isinstance(meta, dict) and "__array__" in meta:
        key = meta["__array__"]
        return arrays[key]

    if isinstance(meta, list):
        return [restore_arrays(v, arrays) for v in meta]

    if isinstance(meta, dict):
        return {k: restore_arrays(v, arrays) for k, v in meta.items()}

    return meta


@dataclass
class SafeData:
    df: pl.DataFrame
    optional_col: bool
    base_dir: Path
    data_file: Path

    def __iter__(self) -> Generator[AnyDict, Any, None]:
        yield from self.df.iter_rows(named=True)

    @classmethod
    def create(
        cls, base_dir: Path | str, csv_file: Path | str, optional_col: set[str]
    ) -> SafeData:
        base = Path(base_dir) if isinstance(base_dir, str) else base_dir
        csv = Path(csv_file) if isinstance(csv_file, str) else csv_file
        if not base.is_dir():
            raise FileNotFoundError(f"Base directory not found: {base}")
        if not csv.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv}")

        df: pl.DataFrame = pl.read_csv(csv)
        needed_col = {"raw_path", "presence"}
        if not needed_col.issubset(df.columns):
            raise ValueError(f"CSV must contain: {needed_col}.\nFound: {df.columns}")

        # check optional cols.
        get_optional = optional_col.issubset(df.columns)
        return cls(df=df, optional_col=get_optional, base_dir=base, data_file=csv)


class PresenceType:
    def __init__(self, value: int | float | bool):
        self.v: np.float64 = self.create(value)

    @classmethod
    def create(cls, potential: int | float | bool) -> np.float64:
        if potential < 0:
            raise ValueError(
                f"Input passed to create presence type was negative: {potential}"
            )
        if potential > 3:
            raise ValueError(
                f"Input passed to create presence type was greater than three: {potential}"
            )
        return np.float64(potential)


@dataclass(frozen=True, slots=True)
class Presence:
    """
    mean: mean presence value across the map
    map: presence values per pixel of the map
    """

    mean: np.float64
    map: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Presence:
        mean = PresenceType(d["mean"])
        map_arr = np.asarray(d["map"], dtype=np.float64)
        if (map_arr < 0).any() or (map_arr > 3).any():
            raise ValueError("Presence map values must be in [0, 3]")
        return cls(mean=mean.v, map=map_arr)

    @staticmethod
    def create_map(h: int, w: int, presence: int) -> np.ndarray[tuple[int, int]]:
        """
        Project out a single value, by labeling each point on the map as
        equal to this value
        """
        return np.repeat([presence], (h * w)).reshape((h, w))


class PreArts(TypedDict):
    hw: list[tuple[int, int]]
    wls: list[Arr1DF]
    pre_stats: list[PreprocStats]
    flats: list[Arr2DF]
    presences: list[Presence]


@dataclass(frozen=True)
class DataArtifacts:
    """
    scene_ids: integer array shape (S,), length is the number of scenes
    preprocess_stats: list of PreprocStats, one per scene
    pre_config: the preprocessing parameters used
    wls: array of floats, denoting common wl per scene
    presences: Presence maps denoting presence per scene
    lengths: integer array (H*W) per-scene pixel counts
    hw: integer array shape (S,2): (H,W) per scene
    slices: slices per scene into flat arrays
    pixel_to_scene: integer array of length N, maps global pixel -> scene index [0..S]
    coords: integer array shape (N,2), (row,col) for each pixel
    """

    scene_ids: ArrayI
    preprocess_stats: list[PreprocStats]
    pre_config: PreConfig
    wls: list[Arr1DF]
    presences: list[Presence]
    lengths: ArrayI
    hw: ArrayI
    slices: list[slice]
    pixel_to_scene: np.ndarray[tuple[int], np.dtype[np.intp]]
    coords: np.ndarray[tuple[int, int], np.dtype[np.int64]]

    def scene_slice(self, scene_idx: int) -> slice:
        return self.slices[scene_idx]

    def scene_of(self, global_idx: int) -> ArrayI:
        x: ArrayI = self.pixel_to_scene[global_idx]
        return x

    def reshape_scene(self, arr_flat: ArrayF, scene_idx: int) -> ArrayF:
        sl = self.slices[scene_idx]
        H, W = self.hw[scene_idx]
        return arr_flat[sl].reshape(H, W)


def build_artifacts(
    pre_config: PreConfig,
    pre_art: PreArts,
    scene_ids: list[int] | None = None,
) -> DataArtifacts:
    hw_list = pre_art["hw"]
    scenes = len(hw_list)
    if scene_ids is None:
        scene_ids = list(range(scenes))
    lens = np.array([H * W for (H, W) in hw_list], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lens)[:-1]))
    slices = [slice(o, o + L) for o, L in zip(offsets, lens)]

    # pixel -> scene map
    s_arr = np.arange(scenes, dtype=np.int64)
    pixel_to_scene: np.ndarray[tuple[int], np.dtype[np.intp]] = np.repeat(s_arr, lens)

    # per-pixel (row,col)
    coords_per_scene = []
    for H, W in hw_list:
        # 0, ..., H*W - 1 -> (H,W) grid, where does each pixel lie
        arr_pixels = np.arange(H * W, dtype=np.int64)
        rr, cc = np.unravel_index(arr_pixels, (H, W))
        coords_per_scene.append(np.stack([rr, cc], axis=1))
    coords = np.vstack(coords_per_scene).astype(np.int64)

    hw = np.array(hw_list, dtype=np.int64)
    scene_ids_arr = np.array(scene_ids, dtype=np.int64)

    return DataArtifacts(
        scene_ids=scene_ids_arr,
        preprocess_stats=pre_art["pre_stats"],
        pre_config=pre_config,
        wls=pre_art["wls"],
        presences=pre_art["presences"],
        lengths=lens,
        hw=hw,
        slices=slices,
        pixel_to_scene=pixel_to_scene,
        coords=coords,
    )


# -- HSI


@dataclass(frozen=True, slots=True)
class HSIMap:
    """Contains:
    wl: (M,) float64 wavelengths
    xy: (N, 2) float64 stage coords
    spectra: (N, C) float32 intensities
    cube: (H, W, M) float32 reshaped spectral cube
    presence: Boolean of whether the sample contains alginate, or Presence obj
    _version: float corresponding to HSI schema
    """

    wl: ArrayF
    xy: ArrayF
    spectra: FlatMap
    cube: Cube
    presence: Presence
    _version: float = 0.3

    def check_same_wavelength_grid(
        self,
        prc: HSIMap,
        ref_wl: ArrayF | None = None,
    ) -> SameGridCubes:
        cube_x = self.cube
        cube_y = prc.cube
        ref = self.wl if ref_wl is None else ref_wl
        if assert_same_grid(self.wl, ref) is not None:
            cube_x = cube_x.resample_cube(self.wl, ref)
        if assert_same_grid(prc.wl, ref) is not None:
            cube_y = cube_y.resample_cube(prc.wl, ref)

        return SameGridCubes(cube_x, cube_y, ref)

    @classmethod
    def from_txt(
        cls,
        txt_path: Path,
        presence: Presence,
        acq_time_s: float | None = None,
        accumulation: int | None = None,
    ) -> HSIMap:
        """
        Returns HSIMap containing:
            wl: (M,) float64 wavelengths in cm^-1
            xy: (N, 2) float64 stage coords (assumed µm)
            spectra: (N, M) float32 intensities
            cube: (H, W, M) float32 reshaped spectral cube (row-major by sorted y,x)
        """
        with txt_path.open("r") as f:
            # first non-empty line is wavelengths (M values)
            wl = np.fromstring(next(line for line in f if line.strip()), sep="\t")
            # remaining lines: x, y, I_1, ..., I_M
            rows = [np.fromstring(r_line, sep="\t") for r_line in f if r_line.strip()]
        xy_and_spectra = np.vstack(rows)  # N stages, x, y, and M wavelenghts (N, M+2)

        # get all values of the spatial x, y, which are the first two cols of each row (0: x, 1: y)
        xy = xy_and_spectra[:, :2]
        # after col2 are the intensity values per wl
        spectra = xy_and_spectra[:, 2:].astype(np.float32)
        assert spectra.shape[1] == wl.size, (
            f"band count mismatch {spectra.shape[1]} != {wl.size}"
        )

        if acq_time_s and accumulation:
            # normalize spectra to the aqcuisiton time
            spectra /= float(acq_time_s) * int(accumulation)

        # Get the unique positions, and sort x & y
        xs, _x_indices = np.unique(xy[:, 0], return_inverse=True)
        ys, _y_indices = np.unique(xy[:, 1], return_inverse=True)
        # infer grid and reshape to (H,W,M)
        H, W = ys.size, xs.size
        N = xy.shape[0]
        assert H * W == N, f"grid not rectangular ({H * W=} != {N=})"
        # order indices by (y,x) row-major
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        cube = Cube.from_flat(
            flat_cube=spectra[order], height=H, width=W, spec_bands=wl.size
        )
        return cls(wl, xy, FlatMap.make(spectra), cube, presence)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HSIMap:
        cube = Cube.from_dict(d["cube"])
        try:
            ver = float(d["_version"])
            logger.debug(f"Version found as: {ver}")
            if ver == 0.2:
                # presence is only a boolean
                logger.debug("Version does not include boolean map.")
                presence = d["presence"]
                h, w = cube.H, cube.W
                mapping = Presence.create_map(h, w, presence)
                presence = Presence(mean=np.float64(d["presence"]), map=mapping)
            elif ver == 0.3:
                # presence is a map or boolean
                presence = Presence.from_dict(d["presence"])
            else:
                raise NotImplementedError
        except KeyError as e:
            logger.exception(f"key error loading processed HSI map: {e}")
            raise e

        return cls(
            cube=cube,
            wl=d["wl"],
            xy=d["xy"],
            spectra=FlatMap.from_dict(d["spectra"]),
            presence=presence,
            _version=d["_version"],
        )

    @classmethod
    def from_processed(cls, path: Path) -> HSIMap | None:
        arrays = np.load(path.with_suffix(".npz"))
        with path.with_suffix(".json").open() as f:
            meta = json.load(f)

        restored = restore_arrays(meta, arrays)
        # reconstruct as nested dict/list/scalars
        if isinstance(restored, dict):
            return cls.from_dict(restored)
        return None

    def get_artifacts(self) -> tuple[Presence, ArrayF, tuple[int, int]]:
        return (self.presence, self.wl, (self.cube.H, self.cube.W))


def read_class(
    csv_file: Path, base_dir: Path
) -> Generator[tuple[HSIMap, AnyDict], Any, None]:
    """Read annotations file to retrieve HSIMaps for purpose of classification."""

    # NOTE: currently discard proc_path
    optional_col = {"proc_path", "acq_s", "accum"}
    class_data = SafeData.create(base_dir, csv_file, optional_col)
    base = class_data.base_dir
    opt = class_data.optional_col

    # materialize rows
    for r in class_data:
        hsi_map = HSIMap.from_txt(
            txt_path=base / r["raw_path"],
            presence=Presence.from_dict(r["presence"]),
            accumulation=int(r["accum"]) if opt else None,
            acq_time_s=float(r["acq_s"]) if opt else None,
        )
        yield hsi_map, r


# -- Pair rows


@dataclass(frozen=True)
class PairRow:
    raw_path: Path
    proc_path: Path
    presence: bool
    accumulation: int | None = None
    acquisition: float | None = None

    def retrieve_maps(self) -> tuple[HSIMap, HSIMap]:
        """Load pair of HSIMaps, (raw, processed)."""
        # setup by assuming dimensions, we dont have the dimensions atm.
        h = w = 8
        pres_map = Presence.create_map(h, w, self.presence)
        pres = Presence(np.float64(self.presence), pres_map)

        x_map = HSIMap.from_txt(
            self.raw_path,
            pres,
            self.acquisition,
            self.accumulation,
        )
        y_map = HSIMap.from_txt(
            self.proc_path,
            pres,
            self.acquisition,
            self.accumulation,
        )
        if err := assert_same_grid(x_map.xy, y_map.xy):
            raise RuntimeError(err)

        # match presence to grid properly with known dimensions
        h, w = x_map.cube.H, x_map.cube.W
        pres_map = Presence.create_map(h, w, self.presence)
        pres = Presence(np.float64(self.presence), pres_map)
        x_map = replace(x_map, presence=pres)
        y_map = replace(y_map, presence=pres)
        return (x_map, y_map)


# TODO: convert to using FlatMap?
@dataclass(frozen=True)
class SpectraPair:
    X_raw: ArrayF  # (N, C)
    Y_proc: ArrayF  # (N, C)

    @staticmethod
    def from_annotations(
        rows: list[PairRow],
        ref_wl: None | ArrayF = None,
        peak_cfg: PeakNormConfig | None = None,
        pre_config: PreConfig | None = None,
    ) -> tuple[SpectraPair, DataArtifacts]:
        x_list, y_list = [], []
        pre_arts: PreArts = {
            "hw": [],
            "wls": [],
            "pre_stats": [],
            "flats": [],
            "presences": [],
        }

        if pre_config is None:
            pre_config = PreConfig.make_min()

        for r in rows:
            raw_map, prc_map = r.retrieve_maps()
            same_grid_cubes = raw_map.check_same_wavelength_grid(prc_map, ref_wl)

            cube_x, cube_y, train_stats = same_grid_cubes.pre_process(
                pre_config, peak_cfg
            )

            xpix = cube_x.flatten().get().astype(np.float64)
            ypix = cube_y.flatten().get().astype(np.float64)
            x_list.append(xpix)
            y_list.append(ypix)

            pre_arts["hw"].append((cube_x.H, cube_x.W))
            pre_arts["wls"].append(same_grid_cubes.common_wl)
            pre_arts["presences"].append(raw_map.presence)
            pre_arts["flats"].append(xpix)
            pre_arts["pre_stats"].append(train_stats)

        x_all = np.vstack(x_list, dtype=np.float64)
        y_all = np.vstack(y_list, dtype=np.float64)
        arts = build_artifacts(
            pre_config,
            pre_art=pre_arts,
            scene_ids=list(range(len(pre_arts["hw"]))),
        )
        return SpectraPair(x_all, y_all), arts


def read_pairs(csv_file: Path | str, base_dir: Path | str) -> list[PairRow]:
    """Read annotations file to retrieve PairRows."""

    # check optional cols.
    optional_col = {"acq_s", "accum"}
    pair_rows = SafeData.create(base_dir, csv_file, optional_col)
    base = pair_rows.base_dir
    opt = pair_rows.optional_col

    # materialize rows
    rows: list[PairRow] = []
    for r in pair_rows:
        pair_row = PairRow(
            raw_path=base / Path(r["raw_path"]),
            proc_path=base / Path(r["proc_path"]),
            presence=bool(r["presence"]),
            accumulation=int(r["accum"]) if opt else None,
            acquisition=float(r["acq_s"]) if opt else None,
        )
        rows.append(pair_row)

    return rows


# -- helper for converting raw text files to structured data


def convert_raw_class(csv: str | Path, base: str | Path) -> None:
    """Save HSI maps to files in raw_path."""
    csv = Path(csv) if isinstance(csv, str) else csv
    base = Path(base) if isinstance(base, str) else base

    for map, df_row in read_class(csv_file=csv, base_dir=base):
        READY_DATA_DIR.mkdir(exist_ok=True)
        new_map_path = READY_DATA_DIR / Path(df_row["raw_path"]).name
        try:
            save_outer(map, new_map_path)
        except IOError as e:
            logger.warning(e)


# -- helper for classification building


@dataclass(frozen=True, slots=True)
class ClassPair:
    all_flatmaps: Arr2DF
    arts: DataArtifacts


def build_classification(
    base: str | Path,
    *,
    csv: str | Path | None = None,
    peak_cfg: PeakNormConfig | None = None,
    pre_config: PreConfig | None = None,
) -> ClassPair:
    """
    Build pair of data, labels and raw spectra.

    Args:
        base: base directory containing training data.
        csv: optional path to csv file, used only for raw data

    Returns:
        ClassPair object that contains the DataArtifacts and Spectra
    """
    base = Path(base) if isinstance(base, str) else base
    pre_config = PreConfig.make_min() if pre_config is None else pre_config

    if csv is not None:
        csv = Path(csv) if isinstance(csv, str) else csv
        gen = read_class(csv, base)
        data = map(lambda g: g[0], gen)  # grab the hsi-maps only
    else:
        # search for .npz, each should correspond to a json
        paths = base.glob("*.npz")
        data = map(lambda x: HSIMap.from_processed(x), paths)

    data = (y for y in data if y is not None)

    pre_arts: PreArts = {
        "hw": [],
        "wls": [],
        "pre_stats": [],
        "flats": [],
        "presences": [],
    }

    for h in data:
        presence, wl, hw = h.get_artifacts()
        cube, stats = preprocess_cube(h.cube, wl, pre_config, peak_cfg)
        flat = cube.flatten()

        pre_arts["hw"].append(hw)
        pre_arts["wls"].append(wl)
        pre_arts["presences"].append(presence)
        pre_arts["flats"].append(flat.get())
        pre_arts["pre_stats"].append(stats)

    x_all: Arr2DF = np.vstack(pre_arts["flats"], dtype=np.float64)

    arts = build_artifacts(
        pre_config,
        pre_art=pre_arts,
        scene_ids=list(range(len(pre_arts["hw"]))),
    )

    return ClassPair(x_all, arts)
