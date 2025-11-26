from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyspectral.core import EpochLoss
from pyspectral.data.features import FoldStat
from pyspectral.data.io import DataArtifacts
from pyspectral.types import (
    Arr1DF,
    ArrayF,
    ArrayF32,
    ArrayI,
    IndexArray,
)


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

    def store(self, preds: np.ndarray, indices: np.ndarray, fold_loss: FoldLoss):
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
    ):
        if self.pred is None:
            raise RuntimeError("Attempted to retrieve None type predictions")

        if idx is not None:
            return self.true[idx], self.true[idx]
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

    def get_pred(self):
        assert self.pred_orig is not None
        return self._prc, self.pred_orig


class ClassStats:
    """Per-scene diagnostics for classification cross-validation."""

    def __init__(self, artifacts: DataArtifacts):
        self.artifacts: DataArtifacts = artifacts
        self.scene_ids: ArrayI = np.asarray(artifacts.scene_ids)
        # binary presence labels stored for convenience
        true: ArrayF32 = np.asarray(artifacts.presences, dtype=np.float32).reshape(
            -1, 1
        )
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
            flat_map = presence.map.reshape(-1).astype(np.float32, copy=False)
            expected_len = sl.stop - sl.start
            if flat_map.shape[0] != expected_len:
                raise ValueError(
                    "Presence map size mismatch "
                    + f"for scene {scene_idx}: {flat_map.shape[0]} vs {expected_len}."
                )
            true[sl, 0] = flat_map

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
