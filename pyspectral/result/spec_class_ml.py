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
from pyspectral.result.predict import MaskedValues, PredCompare
from pyspectral.types import Arr1DF, Arr2DF32, ArrayF, ArrayF32, UnitFloat

# -- spec to class -------------------------------------------------------


def create_svm_pipeline(k: int, RANDOM_STATE: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=k, svd_solver="full")),
            (
                "svc",
                svm.SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


@dataclass
class SVCPred:
    k: int
    w: Arr1DF
    b: float
    X: np.ndarray
    y: np.ndarray
    p_neg: np.ndarray[tuple[int, int]]
    p_pos: np.ndarray[tuple[int, int]]
    true_neg: np.ndarray[tuple[int, int]]
    true_pos: np.ndarray[tuple[int, int]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SVCPred:
        return cls(
            k=d["k"],
            w=d["w"],
            b=d["b"],
            X=d["X"],
            y=d["y"],
            p_neg=d["p_neg"],
            p_pos=d["p_pos"],
            true_neg=d["true_neg"],
            true_pos=d["true_pos"],
        )

    @classmethod
    def from_pipeline(
        cls,
        k: int,
        svc_model: svm.SVC,
        x_pca: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
        presence_maps: Arr1DF,
        svc_prob: np.ndarray[tuple[int, int]],
        svc_pos_probs: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ) -> SVCPred:
        return cls(
            k=k,
            w=svc_model.coef_[0],  # pyright: ignore[reportIndexIssue]
            b=float(svc_model.intercept_[0]),
            X=x_pca,
            y=presence_maps,
            p_neg=svc_prob[:, 0],
            p_pos=svc_pos_probs,
            true_neg=x_pca[:, 0],
            true_pos=x_pca[:, 1],
        )


# TODO: not currently used
def pred_SVC(
    X_pca_tr: np.ndarray,
    X_pca_te: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 10,
    drop_maybe: bool = True,
) -> SVCPred:
    y_train_bin = np.isclose(y_train, 1.0).astype(int)
    _y_test_bin = np.isclose(y_test, 1.0).astype(int)

    clf = svm.SVC(kernel="linear", probability=True, class_weight="balanced")
    clf.fit(X_pca_tr[:, :k], y_train_bin)

    w: Arr1DF = clf.coef_[0]  # shape (k,)  # pyright: ignore[reportIndexIssue]
    b: float = clf.intercept_[0]  # scalar

    # X: (N, C) -> X[:, :k]: (N, k) -> predict probability:
    # -> P(class): Shape (N, k) -> P(n): Shape (N, 1)
    p_class: np.ndarray[tuple[int, int]] = clf.predict_proba(X_pca_te[:, :k])
    p_neg: np.ndarray[tuple[int, int]] = p_class[:, 0]
    p_pos: np.ndarray[tuple[int, int]] = p_class[:, 1]
    neg: np.ndarray[tuple[int, int]] = X_pca_te[:, 0]
    pos: np.ndarray[tuple[int, int]] = X_pca_te[:, 1]

    if drop_maybe:
        pca_mask_pos = np.isclose(y_test, 1.0)
        pca_mask_neg = np.isclose(y_test, 0.0)
        pca_mask_maybe = np.isclose(y_test, 0.5)
        filtered_xtest = MaskedValues.build_with_mask(
            X_pca_te, pca_mask_pos, pca_mask_maybe, pca_mask_neg
        )
        filtered_ytest = MaskedValues.build_with_mask(
            y_test, pca_mask_pos, pca_mask_maybe, pca_mask_neg
        )

        drop_maybe_y_te = np.concat([filtered_ytest.pos, filtered_ytest.neg])
        drop_maybe_x_te = np.concat([filtered_xtest.pos, filtered_xtest.neg])

        X_pca_te = drop_maybe_x_te
        y_test = drop_maybe_y_te

    return SVCPred(
        k=k,
        w=w,
        b=b,
        X=X_pca_te,
        y=y_test,
        p_neg=p_neg,
        p_pos=p_pos,
        true_neg=neg,
        true_pos=pos,
    )


def ml_class_predict_pixel(
    spectra: FlatMap, wl: np.ndarray, presence: Presence, *, threshold: float = 0.8
) -> PredCompare:
    region = RegionSet()
    threshold = UnitFloat(threshold)
    flatfeats = [create_specband_feats(xi, wl, region) for xi in iter(spectra)]
    feats: Arr2DF32 = np.vstack(flatfeats).astype(np.float32, copy=False)

    device = torch.device("cpu")
    feat_tensor = torch.from_numpy(feats).unsqueeze(1)  # (Npix, 1, D)
    mask = torch.ones((feat_tensor.shape[0], feat_tensor.shape[1]), dtype=torch.bool)

    d_in = feat_tensor.shape[-1]
    model = MILMeanHead.load(d_in=d_in).to(device=device)
    model.eval()

    with torch.no_grad():
        pred = model.pred((feat_tensor, mask))
        prob = pred_to_numpy(pred)  # (Npix,)

    preds_arr = prob.astype(np.float32)
    true = presence.map.flatten()

    pred_mask = preds_arr >= threshold
    baseline = np.isclose(true, 1.0) | np.isclose(true, 2.0)

    pred_compare = PredCompare.from_predictions(
        true, pred_mask, baseline, pred_prob=preds_arr, threshold=threshold
    )
    return pred_compare
