from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.metrics import root_mean_squared_error

from pyspectral.config import ArrayF, Cube, ModelType
from pyspectral.dataset import Annotations, KFolds, SpectraPair
from pyspectral.features import (
    CubeStats,
    PreConfig,
    PreprocStats,
    SmoothCfg,
    find_peak_in_window,
    preprocess_cube,
)
from pyspectral.modeling.train import BandPenalty, OOFStats, cv_train_model

type LossCompare = tuple[list[FoldPlot], ClassicalPredict]


class FoldPlot:
    def __init__(self, oof_stats: OOFStats, label: str):
        all_loss = oof_stats.get_loss()
        self.train_loss = all_loss.epoch_loss_store.train
        self.test_loss = all_loss.epoch_loss_store.test
        self.label: str = label


@dataclass
class ClassicalPredict:
    pcr_rmse: float
    elasticnet_rmse: float


@dataclass
class PredictData:
    pred_cube: Cube
    proc_cube: Cube
    wl: ArrayF
    scene_stats: list[PreprocStats]
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
    idx: int, stats: OOFStats, base_dir: Path, csv_path: Path
) -> PredictData:
    """
    Grab the indices belonging to that scene, and reshape stats.oof_pred_orig[scene_idx] to (H,W,M).

    """
    # Find the per-scene slice in the flattened arrays
    values = Annotations.read(csv_path, base_dir)
    shapes, wls, lens, prc_cubes, pre_stats = [], [], [], [], []
    # compute cumulative lengths (H*W per scene)
    for i, r in enumerate(values.rows):
        raw_map, prc_map = r.retrieve_maps()
        _cube_x, cube_y, common_wl = raw_map.check_same_wavelength_grid(
            prc_map, ref_wl=None
        )
        height, width, m = cube_y.shape
        shapes.append((height, width, m))
        wls.append(common_wl)
        lens.append(height * width)
        prc_cubes.append(cube_y)

        scene_stats = stats.artifacts.scene_stats[i]
        pre_config = stats.artifacts.pre_config
        cube_y_pre, pre_stat = preprocess_cube(
            cube_maybe=CubeStats(cube_y, scene_stats),
            wl_cm1=common_wl,
            pre_config=pre_config,
        )
        prc_cubes.append(cube_y_pre)  # store the preprocessed version
        pre_stats.append(pre_stat)

    # Reshape ML OOF predictions for this scene back to one cube
    start, n_rows = np.cumsum([0] + lens[:-1])[idx], lens[idx]
    H, W, M = shapes[idx]
    wl = wls[idx]
    # (N,C) -> (H*W,M) -> (H,W,M)
    pred_cube = Cube.from_flat(stats.pred_orig[start : (start + n_rows)], H, W, M)
    if (nc := pre_stats[idx].ref_center_cm1) is not None:
        return PredictData(
            pred_cube=pred_cube,
            proc_cube=prc_cubes[idx],
            wl=wl,
            scene_stats=stats.artifacts.scene_stats,
            num_center=nc,
        )
    else:
        raise ValueError(
            "Preprocessing stats for this scene do not include a reference center."
        )


# TODO: move to training/plot? <luxShrine >
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
        PreConfig(smoothing=None, spike_kernel_size=1, baseline_poly=1),
        PreConfig(smoothing=None, spike_kernel_size=7, baseline_poly=2),
        PreConfig(smoothing=SmoothCfg(), spike_kernel_size=7, baseline_poly=2),
    ]
    models = [ModelType.LRSM, ModelType.LSM]
    # prevent training with different penalties that won't apply to respective types
    lrsm_mp = list(product([models[0]], lam_ids, off_diag, [bands[0]], [biases[0]]))
    lsm_mp = list(product([models[1]], [lam_ids[0]], [off_diag[0]], bands, biases))
    model_penalties = lrsm_mp + lsm_mp

    loss_plot_data = []
    permutations = list(product(ranks, lrs, n_splits, model_penalties, pre_processing))
    logger.debug(f"Number of configs comparing: {len(permutations)}")
    # iterate across each rank, and then CNN then against classical method
    for r, lr, n, mp, pre in permutations:
        mt, li, od, ba, bi = mp
        train_settings = f"{r=}|{lr=}|{li=}|{od=}|{ba=}|{bi=}|{n=}|{pre}|{mt}"
        s_poly = pre.smoothing.poly if pre.smoothing is not None else None
        s_window = pre.smoothing.window if pre.smoothing is not None else None
        print(train_settings)
        spectra = SpectraPair.from_annotations(
            csv,
            data,
            spike_k=pre.spike_kernel_size,
            base_poly=pre.baseline_poly,
            s_poly=s_poly,
            s_window=s_window,
        )

        band_penalty = BandPenalty(id=li, off_diag=od, band=ba, bias=bi)

        oof_stats = cv_train_model(
            spectral_pairs=spectra[0],
            arts=spectra[1],
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
    classical, _ = SpectraPair.from_annotations(csv, data)
    true = classical.Y_proc
    cv, _split_iter = KFolds(
        n_splits=n_splits[0], raw_data=classical.X_raw
    ).get_splits()
    enet = classical.multitask_elasticnet_predict(cv)
    pcr = classical.pcr_predict(cv=cv)
    enet_rmse = root_mean_squared_error(true, enet)
    pcr_rmse = root_mean_squared_error(true, pcr)

    return loss_plot_data, ClassicalPredict(pcr_rmse, enet_rmse)
