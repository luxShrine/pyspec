from __future__ import annotations

from collections.abc import Generator
import warnings

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
import torch

from pyspectral.data.io import DataArtifacts
from pyspectral.types import (
    Arr1DF32,
    Arr2DF32,
    ArrayF,
    ArrayF32,
    ClassPixIndices,
    KfoldPixIndices,
    SplitIter,
)

# -- types & helpers

type CrossValidator = KFold | GroupKFold | StratifiedGroupKFold | StratifiedKFold
type SpectralData = tuple[torch.Tensor, torch.Tensor]
type ClassSpectralData = tuple[torch.Tensor, torch.Tensor, int]


def expand_indices(
    scene_slices: list[slice], split: np.ndarray
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    scene_indices = [
        np.arange(scene_slices[s].start, scene_slices[s].stop) for s in split
    ]
    return np.concatenate(scene_indices, dtype=np.int64)


def apply_scaler(feats: Arr2DF32, tr_pix_idx: ClassPixIndices) -> Arr1DF32:
    """Fit scaler on training pixels, and apply this fit to all pixels."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(feats[tr_pix_idx])

    # apply fit to all pixels
    F = np.asarray(scaler.transform(feats), np.float32)
    return F


def make_scene_of_pixel(
    arts: DataArtifacts, n_pixels: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    scene_of_pixel = np.empty(n_pixels, dtype=np.int64)
    for scene_idx, slice in enumerate(arts.slices):
        # assign the scene index to each slice of pixels
        scene_of_pixel[slice] = scene_idx
    return scene_of_pixel


# -- classes -------------------------------------------------------


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
