from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyspectral.core import EpochLoss
from pyspectral.data.features import FoldStat
from pyspectral.data.io import DataArtifacts, Presence
from pyspectral.types import (
    Arr1DF,
    ArrayF,
    ArrayF32,
    ArrayI,
    IndexArray,
)


def _resize_presence_map(
    presence_map: np.ndarray[Any, np.dtype[np.float64]],
    target_hw: tuple[int, int],
) -> ArrayF32:
    """Resize presence map to scene shape using nearest-neighbor sampling."""
    target_h, target_w = target_hw
    expected_len = int(target_h * target_w)
    map_arr = np.asarray(presence_map, dtype=np.float32)

    if map_arr.size == expected_len:
        return map_arr.reshape(target_h, target_w)

    if map_arr.ndim == 2 and map_arr.shape[0] > 0 and map_arr.shape[1] > 0:
        src_h, src_w = map_arr.shape
        row_idx = np.minimum(
            (np.arange(target_h, dtype=np.int64) * src_h) // max(1, target_h),
            src_h - 1,
        )
        col_idx = np.minimum(
            (np.arange(target_w, dtype=np.int64) * src_w) // max(1, target_w),
            src_w - 1,
        )
        resized: ArrayF32 = np.asarray(
            map_arr[np.ix_(row_idx, col_idx)], dtype=np.float32
        )
        return resized

    flat_map = map_arr.reshape(-1)
    if flat_map.size == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    flat_idx = np.minimum(
        (np.arange(expected_len, dtype=np.int64) * flat_map.size) // expected_len,
        flat_map.size - 1,
    )
    resized_flat: ArrayF32 = np.asarray(
        flat_map[flat_idx].reshape(target_h, target_w), dtype=np.float32
    )
    return resized_flat


@dataclass(frozen=True)
class FoldLoss:
    train: ArrayF  # shape: (folds, Epoch count)
    test: ArrayF
    frac_improved: float
    best_test: float

    def repr(self) -> str:
        out = (
            f"train loss={self.train.mean():.3g} | "
            + f"test loss={self.test.mean():.3g} | "
            + f"best test loss={self.best_test:.3g} | "
            + f"frac improved vs identity={self.frac_improved:.3f}"
        )
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoldLoss:
        return cls(
            train=d["train"],
            test=d["test"],
            frac_improved=d["frac_improved"],
            best_test=d["best_test"],
        )


@dataclass
class EpochLosses:
    # Mean epoch loss across folds
    loss: EpochLoss = field(init=False)

    train: list[ArrayF] = field(init=False)
    test: list[ArrayF] = field(init=False)
    # Loss, epoch data for each fold, ignore from repr this is not meant to be used
    _fold_loss: list[FoldLoss] = field(repr=False)

    def __post_init__(self) -> None:
        """Average across each fold at same epoch at the end for per epoch metric"""
        folds = len(self._fold_loss)
        self.train = [f.train / folds for f in self._fold_loss]
        self.test = [f.test / folds for f in self._fold_loss]

        # collapse from fold: (epoch_idx: Loss) -> epoch_idx: avg_fold(Loss)
        tr_all_folds_ep = np.einsum("ij->j", np.vstack(self.train))
        te_all_folds_ep = np.einsum("ij->j", np.vstack(self.test))
        self.loss = EpochLoss(train=tr_all_folds_ep, test=te_all_folds_ep)

    def test_mean(self) -> Arr1DF:
        return self.loss.test

    def train_mean(self) -> Arr1DF:
        return self.loss.train

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpochLosses:
        _fold_loss = [FoldLoss.from_dict(i) for i in d["_fold_loss"]]
        return cls(_fold_loss=_fold_loss)


class Preds:
    def __init__(self, true_values: np.ndarray) -> None:
        self.epoch_loss: list[FoldLoss] = []
        self.pred: np.ndarray | None = None
        self.true: np.ndarray = true_values

    def _ensure_storage(self, pred_dim: int) -> None:
        if self.pred is None:
            n_pixels = self.true.shape[0]
            self.pred = np.zeros((n_pixels, pred_dim), dtype=np.float32)

    def store(
        self, preds: np.ndarray, indices: np.ndarray, fold_loss: FoldLoss
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if preds.shape[0] != indices.size:
            raise ValueError(
                "Prediction count mismatch "
                + f"(preds={preds.shape[0]} vs indices={indices.size})."
            )
        self._ensure_storage(preds.shape[1] if preds.ndim > 1 else 1)
        assert self.pred is not None  # for type checker
        self.pred[indices] = preds
        self.epoch_loss.append(fold_loss)

    def get_loss(self) -> EpochLosses:
        return EpochLosses(_fold_loss=self.epoch_loss)

    def get_pred(
        self, idx: tuple[slice, int] | tuple[int, ...] | tuple[slice, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.pred is None:
            raise RuntimeError("Attempted to retrieve None type predictions")

        if idx is not None:
            return self.true[idx], self.pred[idx]
        return self.true, self.pred


class Stats:
    """Store per fold stats."""

    def __init__(self, prc_spectra: ArrayF32 | ArrayF, artifacts: DataArtifacts):
        # all shapes are the same: (N,C)
        prc_spectra = prc_spectra.astype(np.float32, copy=False)
        self.pred_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.true_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.diag_std: ArrayF32 = np.zeros_like(prc_spectra)
        self.diag_orig: ArrayF32 = np.zeros_like(prc_spectra)

        # to compute RMSE in original units we’ll store per-sample inverse later
        self.pred_orig: ArrayF32 = np.zeros_like(prc_spectra)
        self._prc: ArrayF32 = prc_spectra

        # training & test loss stats
        self.epoch_loss: list[FoldLoss] = []

        self.artifacts: DataArtifacts = artifacts

    def store(
        self,
        fold_stat: FoldStat,
        best_preds_fold: list[ArrayF32] | None,
        test_idx: IndexArray,
        fold_epoch_loss: FoldLoss,
    ) -> None:
        """Update records of out of fold stats each fold."""
        if best_preds_fold is None:
            raise RuntimeError(
                "best_preds_fold is None; did training produce any predictions?"
            )
        preds_fold = np.concatenate(best_preds_fold, axis=0)
        self.pred_std[test_idx] = preds_fold
        self.true_std[test_idx] = fold_stat.te_y_znorm.z
        # inverse transform to original Y units: y = y_std*sd + mu
        y_pred_orig = preds_fold * fold_stat.tr_y_znorm.std + fold_stat.tr_y_znorm.mean
        self.pred_orig[test_idx] = y_pred_orig

        self.epoch_loss.append(fold_epoch_loss)

    def get_loss(self) -> EpochLosses:
        return EpochLosses(_fold_loss=self.epoch_loss)

    def get_pred(self) -> tuple[ArrayF32, ArrayF32]:
        assert self.pred_orig is not None
        return self._prc, self.pred_orig


class ClassStats:
    """Per-scene diagnostics for classification cross-validation."""

    def __init__(self, artifacts: DataArtifacts):
        self.artifacts: DataArtifacts = artifacts
        self.scene_ids: ArrayI = np.asarray(artifacts.scene_ids)
        # binary presence labels stored for convenience
        pres = [1.0 if p.mean >= 0.75 else 0.0 for p in artifacts.presences]
        true: ArrayF32 = np.asarray(pres, dtype=np.float32).reshape(-1, 1)
        self.stats: Preds = Preds(true)

    def store(
        self,
        scene_indices: IndexArray,
        best_preds_fold: list[ArrayF32] | None,
        fold_loss: FoldLoss,
    ) -> None:
        if best_preds_fold is None:
            raise RuntimeError(
                "best_preds_fold is None; did the classification fold run evaluation?"
            )
        preds = np.concatenate(best_preds_fold, axis=0).astype(np.float32, copy=False)
        if preds.ndim == 1:
            preds = preds[:, None]
        elif preds.ndim > 2:
            preds = preds.reshape(preds.shape[0], -1)

        self.stats.store(preds, scene_indices, fold_loss)


class PxlStats:
    """Per-pixel diagnostics for classification cross-validation."""

    def __init__(self, artifacts: DataArtifacts):
        self.artifacts: DataArtifacts = artifacts
        n_pixels = int(artifacts.pixel_to_scene.shape[0])

        true = np.zeros((n_pixels, 1), dtype=np.float32)
        for scene_idx, presence in enumerate(artifacts.presences):
            sl = artifacts.slices[scene_idx]
            target_h = int(artifacts.hw[scene_idx, 0])
            target_w = int(artifacts.hw[scene_idx, 1])
            aligned_map = _resize_presence_map(presence.map, (target_h, target_w))
            true[sl, 0] = aligned_map.reshape(-1)

            if presence.map.shape != aligned_map.shape:
                # Keep artifact labels aligned so downstream pixel datasets train safely.
                artifacts.presences[scene_idx] = Presence(
                    mean=np.float64(aligned_map.mean()),
                    map=aligned_map.astype(np.float64, copy=False),
                )

        self.stats: Preds = Preds(true)

    def store(
        self,
        pixel_indices: IndexArray,
        best_preds_fold: list[ArrayF32] | None,
        fold_loss: FoldLoss,
    ) -> None:
        if best_preds_fold is None:
            raise RuntimeError(
                "best_preds_fold is None; did the pixel "
                + "classification fold run evaluation?"
            )
        preds = np.concatenate(best_preds_fold, axis=0).astype(np.float32, copy=False)
        if preds.ndim == 1:
            preds = preds[:, None]
        elif preds.ndim > 2:
            preds = preds.reshape(preds.shape[0], -1)

        self.stats.store(preds, pixel_indices, fold_loss)
