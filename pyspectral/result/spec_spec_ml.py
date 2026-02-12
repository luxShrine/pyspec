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
