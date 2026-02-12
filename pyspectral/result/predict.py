from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm
import torch
from tqdm.auto import tqdm

from pyspectral.core import Cube, FlatMap, TruePredPair
from pyspectral.data.dataset import CrossValidator, KFolds
from pyspectral.data.features import (
    FoldStat,
    RegionSet,
    create_specband_feats,
    fit_diag_affine,
    predict_diag_affine,
)
from pyspectral.data.io import HSIMap, Presence, read_pairs
from pyspectral.data.preprocessing import find_peak_in_window, preprocess_cube
from pyspectral.modeling.models import MILMeanHead
from pyspectral.modeling.oof import Stats
from pyspectral.modeling.train import compute_iou_from_masks, pred_to_numpy
from pyspectral.types import Arr1DF, Arr2DF32, ArrayF, ArrayF32, UnitFloat

# -- general helpers -------------------------------------------------------


@dataclass
class MaskedValues:
    pos: np.ndarray
    maybe: np.ndarray
    neg: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from (self.pos, self.maybe, self.neg)

    def __getitem__(self, idx: int) -> np.ndarray:
        col = [self.pos, self.maybe, self.neg]
        return col[idx]

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        arr = np.concat([self.pos, self.maybe, self.neg], dtype=dtype)
        return arr

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaskedValues:
        return cls(
            pos=np.asarray(d["pos"]),
            maybe=np.asarray(d["maybe"]),
            neg=np.asarray(d["neg"]),
        )

    @staticmethod
    def concat(mask_list: list[MaskedValues]) -> MaskedValues:
        """Concatenate a list of masked values into a single MaskedValue object."""
        pos_joined = np.concat([m[0] for m in mask_list])
        maybe_joined = np.concat([m[1] for m in mask_list])
        neg_joined = np.concat([m[2] for m in mask_list])
        return MaskedValues(
            pos_joined,
            maybe_joined,
            neg_joined,
        )

    @staticmethod
    def get_mask(true_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        legacy_ternary = np.isclose(true_arr, 2.0).any()
        if legacy_ternary:
            mask_pos = np.isclose(true_arr, 2.0)
            mask_maybe = np.isclose(true_arr, 1.0)
            mask_neg = np.isclose(true_arr, 0.0)
        else:
            mask_pos = np.isclose(true_arr, 1.0)
            mask_maybe = np.isclose(true_arr, 0.5)
            mask_neg = np.isclose(true_arr, 0.0)
        return mask_pos, mask_maybe, mask_neg

    @classmethod
    def build_with_mask(
        cls,
        arr: np.ndarray,
        mask_pos: np.ndarray,
        mask_maybe: np.ndarray,
        mask_neg: np.ndarray,
    ) -> MaskedValues:
        return cls(arr[mask_pos], arr[mask_maybe], arr[mask_neg])

    @classmethod
    def build(cls, arr: np.ndarray, true_arr: np.ndarray) -> MaskedValues:
        mask_pos, mask_maybe, mask_neg = cls.get_mask(true_arr)
        return cls(arr[mask_pos], arr[mask_maybe], arr[mask_neg])

    def get_positive_negative_mask(
        self, *, threshold: float | None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        """Collapse masked labels into binary positives/negatives."""
        arr = np.concatenate([self.pos, self.neg])
        if threshold is None:
            labeled = np.isclose(arr.flatten(), 1.0) | np.isclose(arr.flatten(), 2.0)
        else:
            labeled = arr.flatten() >= threshold
        return labeled.astype(np.int8, copy=False)


@dataclass
class PredCompare:
    true: MaskedValues
    pred: MaskedValues
    iou: float
    pred_prob: np.ndarray
    threshold: UnitFloat = UnitFloat(0.8)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredCompare:
        return cls(
            true=MaskedValues.from_dict(d["true"]),
            pred=MaskedValues.from_dict(d["pred"]),
            iou=d["iou"],
            pred_prob=d["pred_prob"],
            threshold=d["threshold"],
        )

    @classmethod
    def from_predictions(
        cls,
        true: np.ndarray,
        pred_mask: npt.NDArray[np.bool],
        baseline: npt.NDArray[np.bool],
        pred_prob: np.ndarray,
        threshold: float,
    ):
        true_obj = MaskedValues.build(true, true)
        pred_obj = MaskedValues.build(pred_mask, true)
        iou = compute_iou_from_masks(baseline, pred_mask)
        return cls(true_obj, pred_obj, iou, pred_prob, UnitFloat(threshold))

    def get_true_pred(self) -> TruePredPair:
        return TruePredPair(np.asarray(self.true), np.asarray(self.pred))
