from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm
import torch

from pyspectral.core import FlatMap
from pyspectral.data.features import (
    RegionSet,
    create_specband_feats,
)
from pyspectral.data.io import Presence
from pyspectral.modeling.models import MILMeanHead
from pyspectral.modeling.train import pred_to_numpy
from pyspectral.result.predict import PredCompare
from pyspectral.types import Arr1DF, Arr2DF32, UnitFloat

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


def ml_class_predict_pixel(
    spectra: FlatMap, wl: np.ndarray, presence: Presence, *, threshold: float = 0.8
) -> PredCompare:
    region = RegionSet()
    threshold = UnitFloat(threshold)
    flatfeats = [create_specband_feats(xi, wl, region) for xi in iter(spectra)]
    feats: Arr2DF32 = np.vstack(flatfeats).astype(np.float32, copy=False)
    # WARN:
    print(">>Debug output:")
    print("feats shape:", feats.shape)
    print("per-feature std mean:", feats.std(axis=0).mean())
    print("per-sample std mean :", feats.std(axis=1).mean())
    print("min/max:", feats.min(), feats.max())

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
    baseline = np.isclose(true, 1.0)

    pred_compare = PredCompare.from_predictions(
        true, pred_mask, baseline, pred_prob=preds_arr, threshold=threshold
    )
    return pred_compare
