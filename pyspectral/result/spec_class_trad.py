from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from pyspectral.result.predict import PredCompare
from pyspectral.types import Arr1DF, Arr2DF32, ArrayF, ArrayF32, UnitFloat


def svm_class_predict_pixel(
    spectra: FlatMap,
    presence: Presence,
    pipeline: Pipeline,
    *,
    threshold: float = 0.8,
) -> PredCompare:
    """
    Evaluate a fitted PCA+SVM pipeline on a new HSI map at pixel level.
    Assumes the SVC inside pipeline was trained with binary labels {0,1},
    where 1 denotes the positive class (alginate).
    """
    # Features must match training-time features exactly
    pixels = spectra.get_pixels()
    true = presence.map.flatten()

    svc = pipeline.named_steps["svc"]
    classes = svc.classes_

    # assert [0 1] == classes  # should be [0 1]
    # p_pos = pipeline.predict_proba(pixels)[:, 1]
    # print(p_pos.min(), p_pos.max())

    if classes.size != 2:
        raise ValueError(f"Expected binary SVM, got classes {classes!r}")

    # assume the larger class is the positive one (0/1 -> 1)
    pos_class_val = classes.max()
    pos_idx = int(np.where(classes == pos_class_val)[0][0])

    prob = pipeline.predict_proba(pixels)[:, pos_idx].astype(np.float32)

    # binary mask for IoU consistent
    # baseline = np.isclose(true, 1.0) | np.isclose(true, 2.0)
    baseline = np.isclose(true, 1.0)
    pred_mask = prob >= threshold

    return PredCompare.from_predictions(
        true, pred_mask, baseline, prob, threshold=threshold
    )
