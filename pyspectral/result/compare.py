from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm

from pyspectral.config import READY_DATA_DIR
from pyspectral.core import TruePredPair
from pyspectral.data.dataset import KFolds
from pyspectral.data.io import (
    ClassPair,
    DataArtifacts,
    SpectraPair,
    build_classification,
    read_pairs,
    restore_arrays,
    save_outer,
)
import pyspectral.data.preprocessing as prep
from pyspectral.modeling.models import BandPenalty
import pyspectral.modeling.oof as oof
from pyspectral.modeling.train import (
    compute_iou_from_masks,
    cv_train_model,
    train_pixel,
)
from pyspectral.result.predict import (
    MaskedValues,
    SVCPred,
    multitask_elasticnet_predict,
    pcr_predict,
)
from pyspectral.types import (
    Arr2DF,
    ArrayF,
    BaselinePolynomialDegree,
    SpecModelType,
    UnitFloat,
)

# -- General Compare -------------------------------------------------------


def confusion_at_threshold(
    true: np.ndarray, pred: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    y_hat = (pred >= threshold).astype(np.float32)

    tp = np.sum((true == 1.0) & (y_hat == 1.0))
    fp = np.sum((true == 0.0) & (y_hat == 1.0))
    tn = np.sum((true == 0.0) & (y_hat == 0.0))
    fn = np.sum((true == 1.0) & (y_hat == 0.0))

    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# -- Compare pre-processing -------------------------------------------------------
@dataclass
class ClassicalPredict:
    pcr_rmse: float
    elasticnet_rmse: float


class FoldPlot:
    def __init__(self, oof_stats: oof.Stats, label: str) -> None:
        all_loss = oof_stats.get_loss()
        self.train_loss: ArrayF = all_loss.loss.train
        self.test_loss: ArrayF = all_loss.loss.test
        self.label: str = label


type LossCompare = tuple[list[FoldPlot], ClassicalPredict]


def compare_models(csv: Path, data: Path, epochs: int = 10) -> LossCompare:
    """Spec to Spec Pre-processing determiner."""
    # compare across epochs, ranks, complexity
    normal_off_diag = 1e-5
    normal_bias = 1e-2

    ranks = [12, 64]
    lrs = [2e-4]
    lam_ids = [0.0, 1e-2]
    off_diag = [0.0, normal_off_diag]
    bands = [1]
    biases = [0.0, 1e-4, normal_bias]
    n_splits = [4]
    pre_processing = [
        prep.PreConfig(
            smoothing=None,
            spike_kernel_size=1,
            baseline=BaselinePolynomialDegree(1),
        ),
        prep.PreConfig(smoothing=None, baseline=BaselinePolynomialDegree(2)),
        prep.PreConfig(
            smoothing=prep.SmoothCfg(), baseline=BaselinePolynomialDegree(2)
        ),
    ]
    models = [SpecModelType.LRSM, SpecModelType.LSM]
    # prevent training with different penalties that won't apply to respective types
    lrsm_mp = list(product([models[0]], lam_ids, off_diag, [bands[0]], [biases[0]]))
    lsm_mp = list(product([models[1]], [lam_ids[0]], [off_diag[0]], bands, biases))
    model_penalties = lrsm_mp + lsm_mp

    peak_config = prep.PeakNormConfig(mode=prep.GlobalPeakNorm())

    loss_plot_data = []
    permutations = list(product(ranks, lrs, n_splits, model_penalties, pre_processing))
    logger.debug(f"Number of configs comparing: {len(permutations)}")
    rows = read_pairs(csv, data)
    # iterate across each rank, and then CNN then against classical method
    for r, lr, n, mp, pre in permutations:
        mt, li, od, ba, bi = mp
        train_settings = f"{r=}|{lr=}|{li=}|{od=}|{ba=}|{bi=}|{n=}|{pre}|{mt}"
        print(train_settings)
        spectra, arts = SpectraPair.from_annotations(
            rows, peak_cfg=peak_config, pre_config=pre
        )

        band_penalty = BandPenalty(id=li, off_diag=od, band=ba, bias=bi)

        oof_stats = cv_train_model(
            spectral_pairs=spectra,
            arts=arts,
            epochs=epochs,
            band_penalty=band_penalty,
            lr=lr,
            n_splits=n,
            model_type=mt,
            rank=r,
        )
        loss_plot_data.append(FoldPlot(oof_stats, train_settings))

        print(20 * "-")

    # PCR/ElasticNet comparison
    classical, _ = SpectraPair.from_annotations(rows)
    true = classical.Y_proc
    cv, _split_iter = KFolds(
        n_splits=n_splits[0], raw_data=classical.X_raw
    ).get_splits()
    enet = multitask_elasticnet_predict(classical.X_raw, classical.Y_proc, cv)
    pcr = pcr_predict(classical.X_raw, classical.Y_proc, cv=cv)
    enet_rmse = root_mean_squared_error(true, enet)
    pcr_rmse = root_mean_squared_error(true, pcr)

    return loss_plot_data, ClassicalPredict(pcr_rmse, enet_rmse)


# -- Compare Spec to Class -------------------------------------------------------


@dataclass
class PredCompare:
    true: MaskedValues
    pred: MaskedValues
    iou: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredCompare:
        return cls(
            true=MaskedValues.from_dict(d["true"]),
            pred=MaskedValues.from_dict(d["pred"]),
            iou=d["iou"],
        )

    def get_true_pred(self) -> TruePredPair:
        return TruePredPair(np.asarray(self.true), np.asarray(self.pred))


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
        save_outer(self, path=path)

    @classmethod
    def from_file(cls, path: Path) -> MLSVMPlot | None:
        arrays = np.load(path.with_suffix(".npz"))
        with path.with_suffix(".json").open() as f:
            meta = json.load(f)

        restored = restore_arrays(meta, arrays)
        # reconstruct as nested dict/list/scalars
        if isinstance(restored, dict):
            return cls.from_dict(restored)
        return None


def _do_ml_predict(
    class_pair: ClassPair,
    epochs: int,
    threshold: UnitFloat,
    N_SPLITS: int,
    RANDOM_STATE: int,
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
    baseline = true_flat > 1
    ml_mask = ml_probs >= threshold
    ml_iou = compute_iou_from_masks(baseline, ml_mask)
    ml_cmp = PredCompare(true=ml_true_masked, pred=ml_pred_masked, iou=ml_iou)
    return ml_cmp, ml_loss, true_flat


def _do_pca_svm_predict(
    X_pix: Arr2DF,
    arts: DataArtifacts,
    k: int,
    true_flat: np.ndarray[tuple[int]],
    threshold: UnitFloat,
    N_SPLITS: int,
    RANDOM_STATE: int,
) -> tuple[PredCompare, list[SVCPred]]:
    presence_maps = np.concatenate([x.map.flatten() for x in arts.presences])
    y_binary = (presence_maps == 2.0).astype(np.int32, copy=False)

    splits = KFolds(N_SPLITS, X_pix, RANDOM_STATE)
    cv_indices = [
        (tr_idx, te_idx) for tr_idx, te_idx, _, _ in splits.scene_cv_pixel_indices(arts)
    ]

    pipeline = Pipeline(
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

    svc_prob: np.ndarray[tuple[int, int]] = cross_val_predict(
        pipeline, X=X_pix, y=y_binary, cv=cv_indices, method="predict_proba"
    )
    svc_pos_probs = np.asarray(svc_prob[:, 1], dtype=np.float32)

    # masks from true labels
    svc_true_masked = MaskedValues.build(true_flat, true_flat)
    svc_pred_masked = MaskedValues.build(svc_pos_probs, true_flat)

    # confusion matrix from thresholded probs
    # pca_cm = _make_confusion_matrix(svc_true_masked, svc_pred_masked, threshold)

    svc_mask = svc_pos_probs >= float(threshold)
    svc_iou = compute_iou_from_masks((true_flat > 1), svc_mask)

    pca_cmp = PredCompare(true=svc_true_masked, pred=svc_pred_masked, iou=svc_iou)

    pipeline.fit(X_pix, y_binary)
    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    svc_model: svm.SVC = pipeline.named_steps["svc"]

    x_scaled = scaler.transform(X_pix)
    x_pca: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.asarray(
        pca.transform(x_scaled), dtype=np.float32
    )

    svc_pred = SVCPred.from_pipeline(
        k, svc_model, x_pca, presence_maps, svc_prob, svc_pos_probs
    )

    return pca_cmp, [svc_pred]


def class_ml_svm(epochs: int = 10, *, threshold: float = 0.5, k: int = 10) -> MLSVMPlot:
    N_SPLITS: int = 4
    RANDOM_STATE: int = 47
    threshold = UnitFloat(threshold)
    class_pair = build_classification(READY_DATA_DIR)
    arts = class_pair.arts
    X_pix: Arr2DF = class_pair.all_flatmaps  # (N, C)

    # -- ML -------------------------------------------------------

    ml_cmp, ml_loss, true_flat = _do_ml_predict(
        class_pair, epochs, threshold, N_SPLITS, RANDOM_STATE
    )

    # -- PCA -------------------------------------------------------

    pca_cmp, svc_preds = _do_pca_svm_predict(
        X_pix, arts, k, true_flat, threshold, N_SPLITS, RANDOM_STATE
    )

    return MLSVMPlot(
        ml_cmp=ml_cmp, ml_loss=ml_loss, pca_cmp=pca_cmp, svc_preds=svc_preds
    )
