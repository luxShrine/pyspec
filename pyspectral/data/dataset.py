from __future__ import annotations

from collections.abc import Generator
from typing import Any, override

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from pyspectral.config import ArrayF, ArrayF32, ArrayI, IndexArray
from pyspectral.core import FlatMap
from pyspectral.data.features import FoldStat, RegionSet, create_specband_feats
from pyspectral.data.io import ClassPair, DataArtifacts, SpectraPair

type SplitIter = Generator[tuple[IndexArray, IndexArray]]
type CrossValidator = KFold | GroupKFold | StratifiedGroupKFold | StratifiedKFold


class KFolds:
    def __init__(
        self,
        n_splits: int,
        raw_data: ArrayF | ArrayF32,
        random_state: None | int = 42,
    ):
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

    def scene_splits(self, arts: DataArtifacts):
        y_scene = np.asarray(arts.presences, dtype=int)  # (S,)
        num_scenes = len(arts.scene_ids)
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for tr_s, va_s in skf.split(np.zeros(num_scenes), y_scene):
            yield tr_s.astype(np.int64), va_s.astype(np.int64)

    def scene_cv_pixel_indices(self, arts: DataArtifacts):
        y_scene = np.asarray(arts.presences, dtype=int)  # shape (Scenes,)
        num_scenes = len(arts.scene_ids)
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for tr_s, va_s in skf.split(np.zeros(num_scenes), y_scene):
            tr_idx = np.concatenate(
                [
                    np.arange(sl.start, sl.stop)
                    for s in tr_s
                    for sl in [arts.scene_slice(s)]
                ]
            )
            va_idx = np.concatenate(
                [
                    np.arange(sl.start, sl.stop)
                    for s in va_s
                    for sl in [arts.scene_slice(s)]
                ]
            )
            yield tr_idx, va_idx, tr_s, va_s


# -- Spectra to Spectra

type SpectralData = tuple[torch.Tensor, torch.Tensor]


class PixelSpectraDataset(Dataset[SpectralData]):
    """Dataset for operating on individual spectra data, pixel by pixel."""

    def __init__(self, X: ArrayF32, Y: ArrayF32):
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


def build_spec_datasets(n_splits: int, data: SpectraPair, arts: DataArtifacts):
    wl = arts.wls[0]

    # per-pixel features
    raw = data.X_raw.astype(np.float32)  # (N,C)
    prc = data.Y_proc.astype(np.float32)  # (N,C)

    # TODO: calculate this instead of magic
    h = w = 8
    ppt = int(h / 2)

    kfolds = KFolds(n_splits, raw)
    _cv, split_iter = kfolds.get_splits((h, w), pixels_per_tile=ppt)

    for tr_s, te_s in split_iter:
        # take out the sections of the arrays by index & normalize around average
        fold_stat = FoldStat.from_subset(raw[tr_s], prc[tr_s], raw[te_s], prc[te_s])

        # create baseline to compare to.
        yhat_te_std, yhat_te_orig = fold_stat.get_baseline()

        # must recreate model, datasets & loaders each fold
        train_ds = PixelSpectraDataset(fold_stat.train_raw_z, fold_stat.train_prc_z)
        test_ds = PixelSpectraDataset(fold_stat.test_raw_z, fold_stat.test_prc_z)

        yield (train_ds, tr_s), (test_ds, te_s), fold_stat


# -- Spectra to Classificaton

type SceneSpectralData = tuple[torch.Tensor, torch.Tensor, int]


class SceneSpectralDataset(Dataset[SceneSpectralData]):
    """
    __getitem__(i) -> (X_i, y_i, scene_idx)
       X_i : (T_i, D) float32, all pixel features for scene i
       y_i : ()        float32 scalar {0,1}
       scene_idx : int index into arts.scene_ids
    """

    def __init__(self, feats: ArrayF32, arts: DataArtifacts, scene_ids: ArrayI):
        self.feats: ArrayF32 = feats.astype(np.float32, copy=False)
        self.arts: DataArtifacts = arts
        self.scenes: ArrayI = scene_ids

    def __len__(self) -> int:
        return self.scenes.size

    @override
    def __getitem__(self, idx: int):
        scene: int = self.scenes[idx]
        slice = self.arts.scene_slice(scene)
        X = torch.from_numpy(self.feats[slice]).float().contiguous()
        y = torch.tensor(float(self.arts.presences[scene]), dtype=torch.float32)
        return X, y, scene


def scene_collate(batch: list[SceneSpectralData]):
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


def expand_indices(scene_slices: list[slice], split: np.ndarray):
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
    wl = arts.wls[0]

    # per-pixel features
    X_pix = data.all_flatmaps  # (N, C) float64
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
