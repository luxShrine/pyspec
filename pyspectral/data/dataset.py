from __future__ import annotations

from collections.abc import Generator
import warnings
from typing import override

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from pyspectral.core import FlatMap
from pyspectral.data.features import FoldStat, RegionSet, create_specband_feats
from pyspectral.data.io import ClassPair, DataArtifacts, SpectraPair
from pyspectral.types import (
    Arr1DF,
    Arr1DF32,
    Arr2DF,
    Arr2DF32,
    ArrayF,
    ArrayF32,
    ArrayI,
    ClassPixIndices,
    KfoldPixIndices,
    SplitIter,
)

type CrossValidator = KFold | GroupKFold | StratifiedGroupKFold | StratifiedKFold


class KFolds:
    def __init__(
        self,
        n_splits: int,
        raw_data: ArrayF | ArrayF32,
        random_state: None | int = 42,
    ) -> None:
        self.n_splits: int = n_splits
        self.raw: ArrayF | ArrayF32 = raw_data
        self.random_state: None | int = random_state

    @staticmethod
    def _make_grid_groups(
        H: int,
        W: int,
        *,
        tiles_h: int | None,
        tiles_w: int | None,
        pixels_per_tile: int | None,
    ) -> np.ndarray:
        """
        Returns group IDs of shape (H*W,) assigning each pixel to a tile.
        - If tiles_h/tiles_w are given, divides into that many tiles.
        - Else if pixels_per_tile is given, tries fit that number into tile.
          (e.g., H=8, W=8, ppt=4 -> ~2x2 tiles each)
        Handles ragged edges; last tiles may be smaller.
        """

        if tiles_h is None or tiles_w is None:
            if not pixels_per_tile:
                pixels_per_tile = 4  # default ~4 pixels per group
            # choose tile dimensions close to square
            tile_h = max(1, int(round(np.sqrt(pixels_per_tile))))
            tile_w = max(1, pixels_per_tile // tile_h or 1)
            tiles_h = int(np.ceil(H / tile_h))
            tiles_w = int(np.ceil(W / tile_w))
            # bins by block size (ragged last bins)
            size_h = (np.arange(H)) // tile_h
            size_w = (np.arange(W)) // tile_w
        else:
            # divide into a fixed number of tiles along each axis
            size_h = (np.arange(H) * tiles_h) // H
            size_w = (np.arange(W) * tiles_w) // W

        h_bins = np.minimum(size_h, tiles_h - 1)
        w_bins = np.minimum(size_w, tiles_w - 1)

        # tile_id = (h_bin * n_tiles_w) + w_bin
        G = ((h_bins[:, None] * tiles_w) + w_bins[None, :]).astype(np.int32)
        return G.ravel()

    def _scene_splitter(
        self, y_scene: np.ndarray[tuple[int], np.dtype[np.int64]]
    ) -> tuple[CrossValidator, SplitIter]:
        num_scenes = int(y_scene.shape[0])
        if num_scenes < 2:
            raise ValueError(
                "Scene cross-validation requires at least 2 scenes; "
                + f"got {num_scenes}."
            )

        n_splits = min(self.n_splits, num_scenes)
        if n_splits != self.n_splits:
            warnings.warn(
                "Reducing scene CV fold count from "
                + f"{self.n_splits} to {n_splits} due to limited scenes.",
                stacklevel=2,
            )

        classes, counts = np.unique(y_scene, return_counts=True)
        min_class_count = int(counts.min()) if counts.size else 0
        can_stratify = classes.size > 1 and min_class_count >= 2

        if can_stratify:
            strat_splits = min(n_splits, min_class_count)
            if strat_splits != n_splits:
                warnings.warn(
                    "Reducing scene CV fold count from "
                    + f"{n_splits} to {strat_splits} to satisfy class balance.",
                    stacklevel=2,
                )
            skf = StratifiedKFold(
                n_splits=strat_splits, shuffle=True, random_state=self.random_state
            )
            return skf, skf.split(np.zeros(num_scenes), y_scene)

        warnings.warn(
            "Falling back to non-stratified KFold for scene CV; "
            + f"class counts are {counts.tolist()}.",
            stacklevel=2,
        )
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return kfold, kfold.split(np.zeros(num_scenes))

    def get_splits(
        self,
        hw: tuple[int, int] | None = None,
        *,
        labels: np.ndarray | None = None,
        tiles_h: int | None = None,
        tiles_w: int | None = None,
        pixels_per_tile: int | None = None,
    ) -> tuple[CrossValidator, SplitIter]:
        x = self.raw
        if hw is None:
            kfold = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            return kfold, kfold.split(x)

        h, w = hw
        groups = self._make_grid_groups(
            h, w, tiles_h=tiles_h, tiles_w=tiles_w, pixels_per_tile=pixels_per_tile
        )
        kfold = GroupKFold(n_splits=self.n_splits)

        if labels is None:
            return kfold, kfold.split(x, groups=groups)

        splitter = StratifiedGroupKFold(
            self.n_splits, shuffle=True, random_state=self.random_state
        )
        return splitter, splitter.split(X=x, y=labels, groups=groups)

    def scene_splits(
        self, arts: DataArtifacts
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        y_scene = np.asarray(
            [int(float(p.mean) >= 0.75) for p in arts.presences], dtype=np.int64
        )  # (S,)
        _, split_iter = self._scene_splitter(y_scene)
        for tr_s, va_s in split_iter:
            yield tr_s.astype(np.int64), va_s.astype(np.int64)

    def scene_cv_pixel_indices(
        self, arts: DataArtifacts
    ) -> Generator[KfoldPixIndices, None, None]:
        # binary stratifying on presence mean >= 0.5
        # could also be p.map.any(), which would find if the map
        # has any presences
        strat = [int(p.map.mean() > 0.5) for p in arts.presences]
        y_scene = np.asarray(strat, dtype=np.int64)  # shape (Scenes,)
        _, split_iter = self._scene_splitter(y_scene)
        for tr_s, va_s in split_iter:
            tr_idx: np.ndarray[tuple[int], np.dtype[np.int64]] = np.concatenate(
                [
                    np.arange(sl.start, sl.stop)
                    for s in tr_s
                    for sl in [arts.scene_slice(s)]
                ]
            )
            va_idx: np.ndarray[tuple[int], np.dtype[np.int64]] = np.concatenate(
                [
                    np.arange(sl.start, sl.stop)
                    for s in va_s
                    for sl in [arts.scene_slice(s)]
                ]
            )
            yield tr_idx, va_idx, tr_s, va_s


# -- Spectra to Spectra

type SpectralData = tuple[torch.Tensor, torch.Tensor]


class SpecSpecDataset(Dataset[SpectralData]):
    """Dataset for operating on individual spectra data, pixel by pixel."""

    def __init__(self, X: ArrayF32, Y: ArrayF32) -> None:
        assert X.shape == Y.shape, "X and Y shapes differ, must be same shaped arrays"
        self.raw: FlatMap = FlatMap.make(X.astype(np.float32, copy=False))
        self.prc: FlatMap = FlatMap.make(Y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.raw.shape[0]

    @override
    def __getitem__(self, idx: int) -> SpectralData:
        X = torch.from_numpy(self.raw.get()[idx]).float().contiguous()
        Y = torch.from_numpy(self.prc.get()[idx]).float().contiguous()
        return X, Y


def build_spec_datasets(
    n_splits: int,
    data: SpectraPair,
    *,
    arts: DataArtifacts | None = None,
    pixels_per_tile: int | None = None,
) -> Generator[
    tuple[
        tuple[SpecSpecDataset, np.ndarray],
        tuple[SpecSpecDataset, np.ndarray],
        FoldStat,
    ],
    None,
    None,
]:
    # per-pixel features
    raw = data.X_raw.astype(np.float32)  # (N,C)
    prc = data.Y_proc.astype(np.float32)  # (N,C)

    if arts is None:
        kfolds = KFolds(n_splits, raw)
        _cv, split_iter = kfolds.get_splits()
    else:
        groups_list = []
        group_offset = 0
        for H, W in arts.hw:
            g = KFolds._make_grid_groups(
                int(H),
                int(W),
                tiles_h=None,
                tiles_w=None,
                pixels_per_tile=pixels_per_tile,
            )
            g = g + group_offset
            groups_list.append(g)
            group_offset += int(g.max()) + 1

        groups = np.concatenate(groups_list).astype(np.int32)
        if groups.shape[0] != raw.shape[0]:
            raise RuntimeError(
                "Group/pixel mismatch: groups length "
                + f"{groups.shape[0]} != data length {raw.shape[0]}"
            )

        kfold = GroupKFold(n_splits=n_splits)
        split_iter = kfold.split(raw, groups=groups)

    for tr_s, te_s in split_iter:
        # take out the sections of the arrays by index & normalize around average
        fold_stat = FoldStat.from_subset(raw[tr_s], prc[tr_s], raw[te_s], prc[te_s])

        # must recreate model, datasets & loaders each fold
        train_ds = SpecSpecDataset(fold_stat.tr_x_znorm.z, fold_stat.tr_y_znorm.z)
        test_ds = SpecSpecDataset(fold_stat.te_x_znorm.z, fold_stat.te_y_znorm.z)

        yield (train_ds, tr_s), (test_ds, te_s), fold_stat


# -- Spectra to Classificaton

type ClassSpectralData = tuple[torch.Tensor, torch.Tensor, int]


class PixelSpectraDataset(Dataset[ClassSpectralData]):
    """
    X_i: (D,)               float32, pixel features for pixel i
    y_i: ()                 float32 scalar {0,0.5,1}
    pixel_ids (M,):         indicies into feats
    scene_of_pixel (N,):    maps global pixel -> scene_idx

    getitem[i] -> (X_i, y_i, scene_idx)
    """

    def __init__(
        self,
        feats: np.ndarray[tuple[int]],
        arts: DataArtifacts,
        pixel_ids: np.ndarray[tuple[int]],
        scene_of_pixel: np.ndarray[tuple[int]],
    ) -> None:
        self.feats: np.ndarray[tuple[int], np.dtype[np.float32]] = feats.astype(
            np.float32, copy=False
        )
        self.arts: DataArtifacts = arts
        self.pixel_ids: np.ndarray[tuple[int]] = pixel_ids
        self.scene_of_pixel: np.ndarray[tuple[int]] = scene_of_pixel

    def __len__(self) -> int:
        return self.pixel_ids.size

    @override
    def __getitem__(self, idx: int) -> ClassSpectralData:
        # global pixel index
        pixel: int = self.pixel_ids[idx]
        # which scene this pixel is in
        scene_index: int = self.scene_of_pixel[pixel]

        # local index within that scene via slice
        slice = self.arts.slices[scene_index]
        local_index = pixel - slice.start

        pixel_presence = float(
            self.arts.presences[scene_index].map.flatten()[local_index]
        )
        if not (
            np.isclose(pixel_presence, 0.0)
            or np.isclose(pixel_presence, 0.5)
            or np.isclose(pixel_presence, 1.0)
        ):
            raise ValueError(f"Unsupported pixel presence value: {pixel_presence}")

        X = torch.from_numpy(self.feats[pixel]).float().contiguous()
        y = torch.tensor(pixel_presence, dtype=torch.float32)
        return X, y, scene_index


def pixel_collate(
    batch: list[ClassSpectralData],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for pixel-level spectral datasets.

    Args:
        batch: List of (features, target, scene_idx) tuples where features
            have shape (D,).

    Returns:
        feats: Tensor of shape (B, 1, D) suitable for MIL models.
        mask: Boolean tensor of shape (B, 1) marking valid positions.
        y: Tensor of shape (B,) with presence targets.
        scene_idx: Tensor of shape (B,) with scene indices.
    """
    feats = torch.stack([b[0] for b in batch], dim=0)  # (B, D)
    feats = feats.unsqueeze(1)  # (B, 1, D)
    y = torch.stack([b[1] for b in batch], dim=0)  # (B,)
    scene_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    mask = torch.ones((feats.shape[0], feats.shape[1]), dtype=torch.bool)
    return feats, mask, y, scene_idx


class SceneSpectralDataset(Dataset[ClassSpectralData]):
    """
    getitem[i] -> (X_i, y_i, scene_idx)
    X_i : (T_i, D) float32, all pixel features for scene i
    y_i : ()        float32 scalar {0,0.5,1}
    scene_idx : int index into arts.scene_ids
    """

    def __init__(self, feats: ArrayF32, arts: DataArtifacts, scene_ids: ArrayI) -> None:
        self.feats: ArrayF32 = feats.astype(np.float32, copy=False)
        self.arts: DataArtifacts = arts
        self.scenes: ArrayI = scene_ids

    def __len__(self) -> int:
        return self.scenes.size

    @override
    def __getitem__(self, idx: int) -> ClassSpectralData:
        scene: int = self.scenes[idx]
        slice = self.arts.scene_slice(scene)
        X = torch.from_numpy(self.feats[slice]).float().contiguous()
        scene_pres_avg = self.arts.presences[scene].mean
        y = torch.tensor(scene_pres_avg, dtype=torch.float32)
        return X, y, scene


def scene_collate(
    batch: list[ClassSpectralData],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each differently sized batch to largest batch size.

    Args:
        batch: list[tuple[torch.Tensor, torch.Tensor, int]] of length (B,) each
        with shape of (T_i*D, 1, scene_index)
        - T_i:         the ith feature tensor
        - D:           dimensionality of features
        - scene_index: the corresponding int id of the scene


    Returns:
        padded_features: Features padded to shape of (B, T, D)
        mask:
        y:
        scene_idx:
    """
    # batch: list of (X, y, scene_idx): b[0].shape[0] = X.shape[0] = T_i*D
    lengths = [b[0].shape[0] for b in batch]
    D = batch[0][0].shape[1]
    B = len(batch)
    T = max(lengths)

    padded_features = batch[0][0].new_zeros((B, T, D))  # padded features
    mask = torch.zeros((B, T), dtype=torch.bool)
    y = torch.stack([b[1] for b in batch], dim=0)  # (B,)
    scene_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)

    for i, (xi, _, _) in enumerate(batch):
        t = xi.shape[0]
        padded_features[i, :t] = xi
        mask[i, :t] = True  # True for valid positions

    return padded_features, mask, y, scene_idx


type TrainSceneDSIndex = tuple[SceneSpectralDataset, ArrayI]
type TestSceneDSIndex = tuple[SceneSpectralDataset, ArrayI]


def expand_indices(
    scene_slices: list[slice], split: np.ndarray
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    scene_indices = [
        np.arange(scene_slices[s].start, scene_slices[s].stop) for s in split
    ]
    return np.concatenate(scene_indices, dtype=np.int64)


def build_scene_datasets(
    n_splits: int,
    data: ClassPair,  # has all_flatmaps (N,C) and arts
    region_set: RegionSet,
) -> Generator[tuple[TrainSceneDSIndex, TestSceneDSIndex, DataArtifacts], None, None]:
    arts = data.arts
    # NOTE: assume all scenes share a common wavelength grid
    wl: Arr1DF = arts.wls[0]

    # per-pixel features
    X_pix: Arr2DF = data.all_flatmaps  # (N, C)
    # get features for each n ∈ N, n is each pixel,
    feats = [create_specband_feats(xi, wl, region_set) for xi in X_pix]
    feats = np.vstack(feats).astype(np.float32, copy=False)  # (N, D)

    # splits correspond to each scene
    splits = KFolds(n_splits, X_pix)
    for tr_s, te_s in splits.scene_splits(arts):
        # fit scaler on training pixels
        tr_pix = expand_indices(arts.slices, tr_s)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(feats[tr_pix])
        # apply fit to all pixels
        F = np.asarray(scaler.transform(feats))

        # construct datasets with only scene indices (ragged per item)
        ds_tr = SceneSpectralDataset(F, arts, tr_s)
        ds_te = SceneSpectralDataset(F, arts, te_s)
        yield (ds_tr, tr_s), (ds_te, te_s), arts


# -- Pixel dataset helpers
type TrainPixDSIndex = tuple[PixelSpectraDataset, ClassPixIndices, ClassPixIndices]
type TestPixDSIndex = tuple[PixelSpectraDataset, ClassPixIndices, ClassPixIndices]


def make_scene_of_pixel(
    arts: DataArtifacts, n_pixels: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    scene_of_pixel = np.empty(n_pixels, dtype=np.int64)
    for scene_idx, slice in enumerate(arts.slices):
        # assign the scene index to each slice of pixels
        scene_of_pixel[slice] = scene_idx
    return scene_of_pixel


def apply_scaler(feats: Arr2DF32, tr_pix_idx: ClassPixIndices) -> Arr1DF32:
    """Fit scaler on training pixels, and apply this fit to all pixels."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(feats[tr_pix_idx])

    # apply fit to all pixels
    F = np.asarray(scaler.transform(feats), np.float32)
    return F


def build_pix_scene_datasets(
    n_splits: int,
    data: ClassPair,  # has all_flatmaps (N,C) and arts
    region_set: RegionSet,
    random_state: int = 47,
) -> Generator[tuple[TrainPixDSIndex, TestPixDSIndex, DataArtifacts], None, None]:
    arts = data.arts
    # NOTE: assume all scenes share a common wavelength grid
    wl: Arr1DF = arts.wls[0]

    # per-pixel features
    X_pix: Arr2DF = data.all_flatmaps  # (N, C)
    scene_pixel = make_scene_of_pixel(arts, X_pix.shape[0])
    # get features for each n ∈ N, n is each pixel
    flatfeats = [create_specband_feats(xi, wl, region_set) for xi in X_pix]
    # (N, D)
    feats: Arr2DF32 = np.vstack(flatfeats).astype(np.float32, copy=False)

    # splits correspond to each scene
    splits = KFolds(n_splits, X_pix, random_state)
    for tr_pix_idx, te_pix_idx, tr_s, te_s in splits.scene_cv_pixel_indices(arts):
        F = apply_scaler(feats, tr_pix_idx)

        ds_tr = PixelSpectraDataset(F, arts, tr_pix_idx, scene_of_pixel=scene_pixel)
        ds_te = PixelSpectraDataset(F, arts, te_pix_idx, scene_of_pixel=scene_pixel)

        yield (ds_tr, tr_pix_idx, tr_s), (ds_te, te_pix_idx, te_s), arts
