from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from pyspectral.config import ArrayF, SpecModelType
from pyspectral.core import Cube
from pyspectral.data.dataset import CrossValidator, KFolds
from pyspectral.data.features import FoldStat, fit_diag_affine, predict_diag_affine
from pyspectral.data.io import SpectraPair, build_artifacts, read_pairs
from pyspectral.data.preprocessing import (
    BaselinePolynomialDegree,
    CubeStats,
    GlobalPeakNorm,
    PeakNormConfig,
    PreConfig,
    SmoothCfg,
    find_peak_in_window,
    preprocess_cube,
)
from pyspectral.modeling.models import BandPenalty
from pyspectral.modeling.train import OOFStats, cv_train_model

type LossCompare = tuple[list[FoldPlot], ClassicalPredict]


class FoldPlot:
    def __init__(self, oof_stats: OOFStats, label: str):
        all_loss = oof_stats.get_loss()
        self.train_loss: ArrayF = all_loss.epoch_loss_store.train
        self.test_loss: ArrayF = all_loss.epoch_loss_store.test
        self.label: str = label


@dataclass
class ClassicalPredict:
    pcr_rmse: float
    elasticnet_rmse: float


@dataclass
class PredictData:
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


# -- Clasical Predict -------------------------------------------------------


def diagonal_affine_predict(x: np.ndarray, y: np.ndarray, cv: CrossValidator) -> ArrayF:
    yhat = np.empty_like(y)
    for tr_idx, te_idx in cv.split(x, y):
        # take out the sections of the arrays by index & normalize around average
        fold_stat = FoldStat.from_subset(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx])
        a, b = fit_diag_affine(fold_stat.train_raw_z, fold_stat.train_prc_z)
        yhat[te_idx] = predict_diag_affine(fold_stat.test_raw_z, a, b)
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
        pipe.fit(fold_stat.train_raw_z, fold_stat.train_prc_z)
        yhat[te_idx] = pipe.predict(fold_stat.test_raw_z)
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
        model.fit(fold_stat.train_raw_z, fold_stat.train_prc_z)
        Yte_std = model.predict(fold_stat.test_raw_z)
        yhat[te_idx] = (Yte_std * fold_stat.y_std) + fold_stat.y_mean
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


# -- ML predict -------------------------------------------------------


def predict_cube(
    idx: int, stats: OOFStats, base_dir: Path, csv_path: Path
) -> PredictData:
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

        scene_stats = stats.artifacts.preprocess_stats[i]
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
        return PredictData(
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


def compare_models(csv: Path, data: Path, epochs: int = 10) -> LossCompare:
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
        PreConfig(
            smoothing=None, spike_kernel_size=1, baseline=BaselinePolynomialDegree(1)
        ),
        PreConfig(smoothing=None, baseline=BaselinePolynomialDegree(2)),
        PreConfig(smoothing=SmoothCfg(), baseline=BaselinePolynomialDegree(2)),
    ]
    models = [SpecModelType.LRSM, SpecModelType.LSM]
    # prevent training with different penalties that won't apply to respective types
    lrsm_mp = list(product([models[0]], lam_ids, off_diag, [bands[0]], [biases[0]]))
    lsm_mp = list(product([models[1]], [lam_ids[0]], [off_diag[0]], bands, biases))
    model_penalties = lrsm_mp + lsm_mp

    peak_config = PeakNormConfig(mode=GlobalPeakNorm())

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
