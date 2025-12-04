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
import sklearn.svm as svm
from tqdm.auto import tqdm

from pyspectral.core import Cube
from pyspectral.data.dataset import CrossValidator, KFolds
from pyspectral.data.features import FoldStat, fit_diag_affine, predict_diag_affine
from pyspectral.data.io import read_pairs
from pyspectral.data.preprocessing import (
    find_peak_in_window,
)
from pyspectral.modeling.oof import Stats
from pyspectral.types import Arr1DF, ArrayF

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
            pos=d["pos"],
            maybe=d["maybe"],
            neg=d["neg"],
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
        mask_pos = true_arr == 2.0
        mask_neg = true_arr == 0.0
        mask_maybe = true_arr == 1.0
        return mask_pos, mask_neg, mask_maybe

    @classmethod
    def build_with_mask(
        cls,
        arr: np.ndarray,
        mask_pos: np.ndarray,
        mask_neg: np.ndarray,
        mask_maybe: np.ndarray,
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
            labeled = arr.flatten() > 1
        else:
            labeled = arr.flatten() >= threshold
        return labeled.astype(np.int8, copy=False)


# -- Spec to Spec Clasical Predict -------------------------------------------------------


def diagonal_affine_predict(x: np.ndarray, y: np.ndarray, cv: CrossValidator) -> ArrayF:
    yhat = np.empty_like(y)
    for tr_idx, te_idx in cv.split(x, y):
        # take out the sections of the arrays by index & normalize around average
        fold_stat = FoldStat.from_subset(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx])
        a, b = fit_diag_affine(fold_stat.tr_x_znorm.z, fold_stat.tr_y_znorm.z)
        yhat[te_idx] = predict_diag_affine(fold_stat.te_x_znorm.z, a, b)
    return yhat


def pcr_predict(x: np.ndarray, y: np.ndarray, cv: CrossValidator) -> ArrayF:
    ncomp = max(2, min(x.shape[0] // 2, 32))
    yhat = np.empty_like(y)
    for tr_idx, te_idx in cv.split(x, y):
        fold_stat = FoldStat.from_subset(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx])
        pipe = Pipeline(
            [
                # ("x_scaler", StandardScaler()),
                ("pca", PCA(n_components=ncomp, svd_solver="full", whiten=False)),
                (
                    "ridge",
                    RidgeCV(alphas=np.logspace(-4, 3, 20), fit_intercept=True),
                ),
            ]
        )
        pipe.fit(fold_stat.tr_x_znorm.z, fold_stat.tr_y_znorm.z)
        yhat[te_idx] = pipe.predict(fold_stat.te_x_znorm.z)
    return yhat


def multitask_elasticnet_predict(
    x: np.ndarray, y: np.ndarray, cv: CrossValidator
) -> ArrayF:
    yhat = np.empty_like(y)
    for tr_idx, te_idx in tqdm(cv.split(x, y), total=cv.get_n_splits()):
        fold_stat = FoldStat.from_subset(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx])
        model = MultiTaskElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 1.0],  # pyright: ignore[reportArgumentType]
            alphas=10,  # def 100 # pyright: ignore[reportArgumentType]
            eps=5e-3,
            cv=4,
            max_iter=5_000,
            tol=5e-3,
            selection="random",
            n_jobs=-1,
            fit_intercept=True,
            # verbose=1,
        )
        model.fit(fold_stat.tr_x_znorm.z, fold_stat.tr_y_znorm.z)
        Yte_std = model.predict(fold_stat.te_x_znorm.z)
        yhat[te_idx] = (Yte_std * fold_stat.tr_y_znorm.std) + fold_stat.tr_y_znorm.mean
    return yhat


def eval(x: np.ndarray, y: np.ndarray, n_splits: int = 4) -> None:
    cv, _ = KFolds(n_splits, x).get_splits()

    rmse_id = root_mean_squared_error(y, x)
    print(f"Identity RMSE: {rmse_id:.6f}")

    yhat_diag = diagonal_affine_predict(x, y, cv)  # vectorized per-band slopes
    rmse_diag = root_mean_squared_error(y, yhat_diag)
    print(f"Diagonal affine RMSE (oof): {rmse_diag:.6f}")

    ncomp = max(2, min(x.shape[0] // 2, 32))
    yhat_pcr = pcr_predict(x, y, cv=cv)
    rmse_pcr = root_mean_squared_error(y, yhat_pcr)
    print(f"PCR({ncomp}) RMSE (oof): {rmse_pcr:.6f}")

    yhat_mten = multitask_elasticnet_predict(x, y, cv=cv)
    rmse_mten = root_mean_squared_error(y, yhat_mten)
    print(f"MultiTaskElasticNet RMSE (oof): {rmse_mten:.6f}")


# -- Spec to Spec ML predict -------------------------------------------------------


@dataclass
class PredictCubeData:
    pred_cube: Cube
    proc_cube: Cube
    raw_cube: Cube
    wl: ArrayF
    # scene_stats: list[PreprocStats]
    num_center: float
    den_center: float = field(init=False)

    def __post_init__(self) -> None:
        Y = self.proc_cube.flatten()
        den_c = find_peak_in_window(
            spectrum=np.median(Y.get(), axis=0),  # or mean Y.mean(axis=0)
            wl=self.wl,
            window=(1240.0, 1460.0),
        )
        self.den_center = den_c


def predict_cube(
    idx: int, stats: Stats, base_dir: Path, csv_path: Path
) -> PredictCubeData:
    """
    Grab the indices belonging to that scene, and reshape stats.oof_pred_orig[scene_idx] to (H,W,M).

    """
    # Find the per-scene slice in the flattened arrays
    rows = read_pairs(csv_path, base_dir)
    shapes, wls, lens, raw_cubes, prc_cubes, pre_stats = [], [], [], [], [], []
    # compute cumulative lengths (H*W per scene)
    for i, r in enumerate(rows):
        raw_map, prc_map = r.retrieve_maps()
        same_grid_cubes = raw_map.check_same_wavelength_grid(prc_map, ref_wl=None)

        _scene_stats = stats.artifacts.preprocess_stats[i]
        pre_config = stats.artifacts.pre_config
        cube_x_pre, cube_y_pre, pre_stat = same_grid_cubes.pre_process(pre_config)
        height, width, m = cube_y_pre.shape

        shapes.append((height, width, m))
        wls.append(same_grid_cubes.common_wl)
        lens.append(height * width)
        pre_stats.append(pre_stat)
        prc_cubes.append(cube_y_pre)  # store the preprocessed version
        raw_cubes.append(cube_x_pre)

    # Reshape ML OOF predictions for this scene back to one cube
    start, n_rows = np.cumsum([0] + lens[:-1])[idx], lens[idx]
    H, W, M = shapes[idx]
    wl = wls[idx]
    # (N,C) -> (H*W,M) -> (H,W,M)
    pred_cube = Cube.from_flat(stats.pred_orig[start : (start + n_rows)], H, W, M)
    if (nc := pre_stats[idx].ref_center_cm1) is not None:
        return PredictCubeData(
            raw_cube=raw_cubes[idx],
            proc_cube=prc_cubes[idx],
            pred_cube=pred_cube,
            wl=wl,
            # scene_stats=stats.artifacts.scene_stats,
            num_center=nc,
        )
    else:
        raise ValueError(
            "Preprocessing stats for this scene do not include a reference center."
        )


# -- spec to class -------------------------------------------------------


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
    y_train_bin = (y_train == 2).astype(int)
    _y_test_bin = (y_test == 2).astype(int)

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
        pca_mask_pos = y_test == 2.0
        pca_mask_neg = y_test == 0.0
        pca_mask_maybe = y_test == 1.0
        filtered_xtest = MaskedValues.build_with_mask(
            X_pca_te, pca_mask_pos, pca_mask_neg, pca_mask_maybe
        )
        filtered_ytest = MaskedValues.build_with_mask(
            y_test, pca_mask_pos, pca_mask_neg, pca_mask_maybe
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
