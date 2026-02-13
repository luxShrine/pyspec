from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from pyspectral.data.dataset import CrossValidator, KFolds
from pyspectral.data.features import (
    FoldStat,
    fit_diag_affine,
    predict_diag_affine,
)
from pyspectral.types import ArrayF

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
