from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm

from pyspectral.config import READY_DATA_DIR
from pyspectral.core import FlatMap
from pyspectral.data.dataset import KFolds
import pyspectral.data.io as io
import pyspectral.data.preprocessing as prep
import pyspectral.modeling.oof as oof
from pyspectral.modeling.train import (
    compute_iou_from_masks,
    train_pixel,
)
from pyspectral.result.class_ml import (
    SVCPred,
    create_svm_pipeline,
    ml_class_predict_pixel,
)
from pyspectral.result.class_trad import svm_class_predict_pixel
from pyspectral.result.predict import (
    MaskedValues,
    PredCompare,
)
from pyspectral.types import (
    Arr2DF,
    ArrayF,
    SameDimensionArrays,
    SameFirstDimensionArrays,
    UnitFloat,
)

# -- types

type LossCompare = tuple[list[FoldPlot], ClassicalPredict]

# -- check polarity


def _positive_presence_mask(
    labels: np.ndarray,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    flat = np.asarray(labels).reshape(-1)
    mask = np.isclose(flat, 1.0) | np.isclose(flat, 2.0)
    return np.asarray(mask, dtype=np.bool_)


def _flatten_positive_probability(
    probs: np.ndarray,
):
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError(
                "Expected probability matrix with at least two columns; "
                + f"got shape {arr.shape}."
            )
        return np.asarray(arr[:, 1], dtype=np.float32)
    return np.asarray(arr.reshape(-1), dtype=np.float32)


def _check_prediction_polarity(
    name: str,
    true_labels: np.ndarray,
    pred_prob: np.ndarray,
    threshold: UnitFloat,
    *,
    strict: bool = False,
) -> bool:
    true_pos = _positive_presence_mask(true_labels)
    probs = _flatten_positive_probability(pred_prob)
    if true_pos.shape[0] != probs.shape[0]:
        raise ValueError(
            f"Cannot compare {name} polarity; "
            + f"label count {true_pos.shape[0]} != probability count {probs.shape[0]}."
        )

    pos_count = int(true_pos.sum())
    neg_count = int((~true_pos).sum())
    if pos_count == 0 or neg_count == 0:
        logger.warning(
            "Skipping polarity check for {} due to missing classes "
            + "(pos_count={}, neg_count={}).",
            name,
            pos_count,
            neg_count,
        )
        return False

    pos_mean = float(np.mean(probs[true_pos]))
    neg_mean = float(np.mean(probs[~true_pos]))
    pred_pos = probs >= float(threshold)
    acc = float(np.mean(pred_pos == true_pos))
    acc_flipped = float(np.mean((~pred_pos) == true_pos))

    appears_flipped = (pos_mean < neg_mean) or (acc_flipped > (acc + 0.05))
    if appears_flipped:
        msg = (
            f"Potential polarity inversion in {name}: "
            + f"mean_pos={pos_mean:.4f}, mean_neg={neg_mean:.4f}, "
            + f"acc@{float(threshold):.2f}={acc:.4f}, "
            + f"acc_if_flipped={acc_flipped:.4f}."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.debug(
            "{} polarity check passed: mean_pos={:.4f}, mean_neg={:.4f}, "
            + "acc@{:.2f}={:.4f}.",
            name,
            pos_mean,
            neg_mean,
            float(threshold),
            acc,
        )
    return appears_flipped


def _warn_if_models_anticorrelate(
    first_name: str,
    first_prob: np.ndarray,
    second_name: str,
    second_prob: np.ndarray,
) -> None:
    p_first = _flatten_positive_probability(first_prob)
    p_second = _flatten_positive_probability(second_prob)
    if p_first.shape[0] != p_second.shape[0]:
        logger.warning(
            "Skipping model-to-model polarity check: {} count {} != {} count {}.",
            first_name,
            p_first.shape[0],
            second_name,
            p_second.shape[0],
        )
        return
    if np.isclose(float(np.std(p_first)), 0.0) or np.isclose(
        float(np.std(p_second)), 0.0
    ):
        logger.warning(
            "Skipping model-to-model polarity check: near-constant probabilities "
            + "for {} or {}.",
            first_name,
            second_name,
        )
        return

    corr = float(np.corrcoef(p_first, p_second)[0, 1])
    corr_flipped = float(np.corrcoef(1.0 - p_first, p_second)[0, 1])
    if corr_flipped > (corr + 0.2):
        logger.warning(
            "{} and {} appear anti-correlated "
            + "(corr={:.4f}, corr_if_first_flipped={:.4f}).",
            first_name,
            second_name,
            corr,
            corr_flipped,
        )


# -- Compare pre-processing -------------------------------------------------------
@dataclass
class ClassicalPredict:
    pcr_rmse: float
    elasticnet_rmse: float


class FoldPlot:
    def __init__(self, train_loss: ArrayF, test_loss: ArrayF, label: str) -> None:
        self.train_loss: ArrayF = train_loss
        self.test_loss: ArrayF = test_loss
        self.label: str = label


@dataclass
class MLSVMPlot:
    ml_cmp: PredCompare
    ml_loss: oof.EpochLosses
    pca_cmp: PredCompare
    svc_preds: list[SVCPred]

    def get_mse(self) -> tuple[float, float, float, float]:
        ml_pred = self.ml_cmp.pred
        pca_pred = self.pca_cmp.pred
        ML_MSE_pos: float = ((self.ml_cmp.true.pos - ml_pred.pos) ** 2).mean()
        ML_MSE_neg: float = ((self.ml_cmp.true.neg - ml_pred.neg) ** 2).mean()
        PCA_MSE_pos: float = ((self.pca_cmp.true.pos - pca_pred.pos) ** 2).mean()
        PCA_MSE_neg: float = ((self.pca_cmp.true.neg - pca_pred.neg) ** 2).mean()
        return (ML_MSE_pos, ML_MSE_neg, PCA_MSE_pos, PCA_MSE_neg)

    def to_dict(self) -> dict[str, Any]:
        mlsvm_dict = {
            "ml_cmp": self.ml_cmp,
            "ml_loss": self.ml_loss,
            "pca_cmp": self.pca_cmp,
            "svc_preds": self.svc_preds,
        }
        return mlsvm_dict

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MLSVMPlot:
        svc_preds = [SVCPred.from_dict(s) for s in d["svc_preds"]]
        return cls(
            ml_cmp=PredCompare.from_dict(d["ml_cmp"]),
            ml_loss=oof.EpochLosses.from_dict(d["ml_loss"]),
            pca_cmp=PredCompare.from_dict(d["pca_cmp"]),
            svc_preds=svc_preds,
        )

    def save_to_file(self, path: Path) -> None:
        io.save_outer(self, path=path)

    @classmethod
    def from_file(cls, path: Path) -> MLSVMPlot | None:
        arrays = np.load(path.with_suffix(".npz"))
        with path.with_suffix(".json").open() as f:
            meta = json.load(f)

        restored = io.restore_arrays(meta, arrays)
        # reconstruct as nested dict/list/scalars
        if isinstance(restored, dict):
            return cls.from_dict(restored)
        return None


def _do_ml_train(
    class_pair: io.ClassPair,
    epochs: int,
    threshold: UnitFloat,
    N_SPLITS: int,
    RANDOM_STATE: int,
    *,
    strict_polarity: bool = False,
) -> tuple[PredCompare, oof.EpochLosses, np.ndarray[tuple[int], np.dtype[np.float32]]]:
    oof_stats = train_pixel(
        class_pair,
        n_splits=N_SPLITS,
        epochs=epochs,
        verbose=True,
        random_state=RANDOM_STATE,
    )
    true, pred = oof_stats.stats.get_pred()
    true_flat = true.flatten()
    ml_loss = oof_stats.stats.get_loss()

    ml_true_masked = MaskedValues.build(true, true)
    ml_pred_masked = MaskedValues.build(pred, true)

    # ml_cm = _make_confusion_matrix(ml_true_masked, ml_pred_masked, threshold)

    ml_probs = pred.flatten()
    _check_prediction_polarity(
        "ML OOF", true_flat, ml_probs, threshold, strict=strict_polarity
    )
    baseline = _positive_presence_mask(true_flat)
    ml_mask = ml_probs >= threshold
    ml_iou = compute_iou_from_masks(baseline, ml_mask)
    ml_cmp = PredCompare(
        true=ml_true_masked, pred=ml_pred_masked, iou=ml_iou, pred_prob=ml_probs
    )
    return ml_cmp, ml_loss, true_flat


def _get_fitted_pipeline(
    pipeline: Pipeline,
    arrays: SameFirstDimensionArrays,
) -> tuple[StandardScaler, PCA, svm.SVC, Pipeline]:
    pipeline.fit(arrays.arr_1, arrays.arr_2)
    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    svc: svm.SVC = pipeline.named_steps["svc"]
    return scaler, pca, svc, pipeline


def _fit_predict_feature_count(pipeline: Pipeline) -> int:
    scaler = pipeline.named_steps.get("scaler")
    features = getattr(scaler, "n_features_in_", None)
    if features is None:
        raise RuntimeError(
            "SVM pipeline scaler is not fitted; expected fitted n_features_in_."
        )
    return int(features)


def _align_pixel_features(
    pixels: Arr2DF,
    expected_features: int,
    *,
    source_wl: ArrayF | None = None,
    target_wl: ArrayF | None = None,
) -> Arr2DF:
    """Resample feature width to match fitted scaler expectations."""
    n_pixels, actual_features = pixels.shape
    if actual_features == expected_features:
        return pixels

    aligned = np.empty((n_pixels, expected_features), dtype=np.float64)
    if (
        source_wl is not None
        and target_wl is not None
        and source_wl.size == actual_features
        and target_wl.size == expected_features
    ):
        for i in range(n_pixels):
            aligned[i] = np.interp(target_wl, source_wl, pixels[i])
        return aligned

    src = np.linspace(0.0, 1.0, num=actual_features, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=expected_features, dtype=np.float64)
    for i in range(n_pixels):
        aligned[i] = np.interp(dst, src, pixels[i])
    return aligned


def _ensure_pipeline_feature_compatibility(
    pixels: Arr2DF,
    pipeline: Pipeline,
    *,
    source_wl: ArrayF | None = None,
    target_wl: ArrayF | None = None,
) -> Arr2DF:
    expected_features = _fit_predict_feature_count(pipeline)
    actual_features = int(pixels.shape[1])
    if actual_features == expected_features:
        return pixels

    logger.warning(
        "Pipeline feature mismatch detected; aligning pixel features "
        + f"from {actual_features} to {expected_features}."
    )
    return _align_pixel_features(
        pixels,
        expected_features,
        source_wl=source_wl,
        target_wl=target_wl,
    )


def _get_binary_map(arts: io.DataArtifacts) -> SameDimensionArrays:
    presence_maps = np.concatenate([x.map.flatten() for x in arts.presences])
    return SameDimensionArrays(
        (np.isclose(presence_maps, 1.0) | np.isclose(presence_maps, 2.0)).astype(
            np.int32, copy=False
        ),
        presence_maps,
    )


def _do_pca_svm_predict(
    X_pix: Arr2DF,
    arts: io.DataArtifacts,
    k: int,
    true_flat: np.ndarray[tuple[int]],
    threshold: UnitFloat,
    N_SPLITS: int,
    RANDOM_STATE: int,
    *,
    strict_polarity: bool = False,
) -> tuple[PredCompare, list[SVCPred]]:
    y_binary, presence_maps = _get_binary_map(arts).tup()
    unique_labels = np.unique(y_binary)
    if unique_labels.size < 2:
        raise ValueError(
            "SVM classification requires at least two classes; "
            + f"got labels {unique_labels.tolist()}."
        )

    splits = KFolds(N_SPLITS, X_pix, RANDOM_STATE)
    cv_indices = [
        (tr_idx, te_idx) for tr_idx, te_idx, _, _ in splits.scene_cv_pixel_indices(arts)
    ]

    pipeline = create_svm_pipeline(k, RANDOM_STATE)
    valid_training_folds = [
        np.unique(y_binary[tr_idx]).size >= 2 for tr_idx, _ in cv_indices
    ]
    can_cross_validate = len(cv_indices) > 1 and all(valid_training_folds)

    if can_cross_validate:
        svc_prob: np.ndarray[tuple[int, int]] = cross_val_predict(  # pyright: ignore[reportAssignmentType]
            pipeline, X=X_pix, y=y_binary, cv=cv_indices, method="predict_proba"
        )
    else:
        class_counts = np.bincount(np.asarray(y_binary, dtype=np.int64), minlength=2)
        bad_folds = len(valid_training_folds) - sum(valid_training_folds)
        logger.warning(
            "Scene-wise SVM cross-validation is not feasible "
            + f"(folds={len(cv_indices)}, invalid_train_folds={bad_folds}, "
            + f"class_counts={class_counts.tolist()}). "
            + "Using fitted in-sample probabilities instead."
        )
        pipeline.fit(X_pix, y_binary)
        svc_prob = np.asarray(pipeline.predict_proba(X_pix))

    svc_pos_probs = _flatten_positive_probability(svc_prob)
    _check_prediction_polarity(
        "SVM OOF", true_flat, svc_pos_probs, threshold, strict=strict_polarity
    )

    # masks from true labels
    svc_true_masked = MaskedValues.build(true_flat, true_flat)
    svc_pred_masked = MaskedValues.build(svc_pos_probs, true_flat)

    svc_mask = svc_pos_probs >= float(threshold)
    svc_iou = compute_iou_from_masks(_positive_presence_mask(true_flat), svc_mask)

    pca_cmp = PredCompare(
        true=svc_true_masked, pred=svc_pred_masked, iou=svc_iou, pred_prob=svc_prob
    )
    scaler, pca, svc_model, _ = _get_fitted_pipeline(
        pipeline, SameFirstDimensionArrays(X_pix, y_binary)
    )

    x_scaled = scaler.transform(X_pix)
    x_pca: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.asarray(
        pca.transform(x_scaled), dtype=np.float32
    )

    svc_pred = SVCPred.from_pipeline(
        k, svc_model, x_pca, presence_maps, svc_prob, svc_pos_probs
    )

    return pca_cmp, [svc_pred]


def class_ml_svm(
    epochs: int = 10,
    *,
    threshold: float = 0.5,
    k: int = 10,
    strict_polarity: bool = False,
) -> MLSVMPlot:
    N_SPLITS: int = 4
    RANDOM_STATE: int = 47
    threshold = UnitFloat(threshold)
    class_pair = io.build_classification(READY_DATA_DIR)
    arts = class_pair.arts
    X_pix: Arr2DF = class_pair.all_flatmaps  # (N, C)

    # -- ML -------------------------------------------------------

    ml_cmp, ml_loss, true_flat = _do_ml_train(
        class_pair,
        epochs,
        threshold,
        N_SPLITS,
        RANDOM_STATE,
        strict_polarity=strict_polarity,
    )

    # -- PCA -------------------------------------------------------

    pca_cmp, svc_preds = _do_pca_svm_predict(
        X_pix,
        arts,
        k,
        true_flat,
        threshold,
        N_SPLITS,
        RANDOM_STATE,
        strict_polarity=strict_polarity,
    )
    _warn_if_models_anticorrelate(
        "ML OOF", ml_cmp.pred_prob, "SVM OOF", pca_cmp.pred_prob
    )

    return MLSVMPlot(
        ml_cmp=ml_cmp, ml_loss=ml_loss, pca_cmp=pca_cmp, svc_preds=svc_preds
    )


def class_pixel(
    path_to_map: Path, *, threshold: float = 0.9, strict_polarity: bool = False
) -> tuple[PredCompare, PredCompare]:
    threshold = UnitFloat(threshold)
    hsi_map = io.HSIMap.from_processed(path_to_map)
    if hsi_map is None:
        raise RuntimeError

    pre_config = prep.PreConfig.make_als()
    wl = hsi_map.wl
    presence = hsi_map.presence
    cube, _ = prep.preprocess_cube(hsi_map.cube, wl, pre_config)
    flat = cube.flatten()

    ml_prediction = ml_class_predict_pixel(flat, wl, presence, threshold=threshold)
    true_flat = np.asarray(presence.map.flatten(), dtype=np.float32)
    _check_prediction_polarity(
        "ML single-map",
        true_flat,
        ml_prediction.pred_prob,
        threshold,
        strict=strict_polarity,
    )

    # fit pipeline on all data
    class_pair = io.build_classification(
        READY_DATA_DIR, pre_config=pre_config, ref_wl=wl
    )
    arts = class_pair.arts
    y = _get_binary_map(arts).arr_1
    x: Arr2DF = class_pair.all_flatmaps  # (N, C)
    pipeline = create_svm_pipeline(k=10, RANDOM_STATE=47)
    _, _, _, pipeline = _get_fitted_pipeline(pipeline, SameFirstDimensionArrays(x, y))

    ref_wl = arts.wls[0] if len(arts.wls) else None
    flat_pixels = np.asarray(flat.get(), dtype=np.float64)
    aligned_pixels = _ensure_pipeline_feature_compatibility(
        flat_pixels,
        pipeline,
        source_wl=wl,
        target_wl=ref_wl,
    )
    aligned_flat = FlatMap.make(aligned_pixels)
    # predict on the particular pixels from the map
    svc_prediction = svm_class_predict_pixel(
        aligned_flat, presence, pipeline, threshold=threshold
    )
    _check_prediction_polarity(
        "SVM single-map",
        true_flat,
        svc_prediction.pred_prob,
        threshold,
        strict=strict_polarity,
    )
    _warn_if_models_anticorrelate(
        "ML single-map",
        ml_prediction.pred_prob,
        "SVM single-map",
        svc_prediction.pred_prob,
    )
    return ml_prediction, svc_prediction
