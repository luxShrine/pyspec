from __future__ import annotations

from collections.abc import Generator
from typing import override

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from pyspectral.data.features import RegionSet, create_specband_feats
from pyspectral.data.io import ClassPair, DataArtifacts
from pyspectral.data.shared import (
    ClassSpectralData,
    KFolds,
    apply_scaler,
    expand_indices,
    make_scene_of_pixel,
)
from pyspectral.types import (
    Arr1DF,
    Arr2DF,
    Arr2DF32,
    ArrayF32,
    ArrayI,
    ClassPixIndices,
)

# types

type TrainPixDSIndex = tuple[PixelSpectraDataset, ClassPixIndices, ClassPixIndices]
type TestPixDSIndex = tuple[PixelSpectraDataset, ClassPixIndices, ClassPixIndices]
type TrainSceneDSIndex = tuple[SceneSpectralDataset, ArrayI]
type TestSceneDSIndex = tuple[SceneSpectralDataset, ArrayI]

# -- Spectra to Classificaton


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
