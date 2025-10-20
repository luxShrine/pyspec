from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pyspectral.config import RNG, ArrayF, ArrayF32, FlatMap
from pyspectral.features import (
    CubeStats,
    DataArtifacts,
    FoldStat,
    PairRow,
    PreConfig,
    fit_diag_affine,
    predict_diag_affine,
    preprocess_cube,
)

type SpectralData = tuple[torch.Tensor, torch.Tensor]
type IndexArray = npt.NDArray[np.intp]  # indices dtype
type SplitIter = Generator[tuple[IndexArray, IndexArray]]


class KFolds:
    def __init__(
        self,
        n_splits: int,
        raw_data: ArrayF | ArrayF32,
        groups: None | Iterable[int] = None,
        random_state: None | int = 42,
    ):
        self.n_splits: int = n_splits
        self.raw: ArrayF | ArrayF32 = raw_data
        self.groups: None | Iterable[int] = groups
        self.random_state: None | int = 42

    def get_splits(
        self,
    ) -> tuple[(KFold | GroupKFold), SplitIter]:
        if self.groups is None:
            cv = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
        else:
            cv = GroupKFold(n_splits=self.n_splits)

        return cv, cv.split(self.raw, groups=self.groups)

    @staticmethod
    def create_groups(H: int, W: int, tiles_h: int = 2, tiles_w: int = 2):
        g = np.zeros((H, W), dtype=np.int32)
        for r in range(H):
            for c in range(W):
                g[r, c] = (r * tiles_h // H) * tiles_w + (c * tiles_w // W)
        return g.ravel()


@dataclass
class Annotations:
    rows: list[PairRow]
    c_out: int
    c_in: int
    hw: tuple[int, int]

    @classmethod
    def read(cls, csv_file: Path | str, base_dir: Path | str) -> Annotations:
        """Read annotations file to retrieve PairRows."""
        base = Path(base_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"Base directory not found: {base}")

        # check csv
        df: pl.DataFrame = pl.read_csv(csv_file)
        needed_col = {"raw_path", "proc_path"}
        if not needed_col.issubset(df.columns):
            raise ValueError(f"CSV must contain: {needed_col}.\nFound: {df.columns}")

        # check optional cols.
        optional_col = {"acq_s", "accum"}
        get_optional = optional_col.issubset(df.columns)

        # materialize rows
        rows: list[PairRow] = []
        for r in df.iter_rows(named=True):
            pair_row = PairRow(
                raw_path=base / Path(r["raw_path"]),
                proc_path=base / Path(r["proc_path"]),
                accumulation=int(r["accum"]) if get_optional else None,
                acquisition=float(r["acq_s"]) if get_optional else None,
            )

            rows.append(pair_row)

        # Return channel counts and spatial shape using the (possibly updated) maps
        map_raw, map_prc = rows[0].retrieve_maps()
        c_out = map_prc.cube.shape[2]  # processed channels
        c_in = map_raw.cube.shape[2]  # raw channels
        hw = map_raw.cube.shape[:2]  # (H, W)
        return cls(rows=rows, c_out=c_out, c_in=c_in, hw=hw)


# datasets


# TODO: convert to using FlatMap
@dataclass(frozen=True)
class SpectraPair:
    X_raw: ArrayF  # (N, C)
    Y_proc: ArrayF  # (N, C)

    def eval(self, n_splits: int = 4) -> None:
        cv, _ = KFolds(n_splits, self.X_raw).get_splits()

        rmse_id = root_mean_squared_error(self.Y_proc, self.X_raw)
        print(f"Identity RMSE: {rmse_id:.6f}")

        yhat_diag = self.diagonal_affine_predict(cv)  # vectorized per-band slopes
        rmse_diag = root_mean_squared_error(self.Y_proc, yhat_diag)
        print(f"Diagonal affine RMSE (oof): {rmse_diag:.6f}")

        ncomp = max(2, min(self.X_raw.shape[0] // 2, 32))
        yhat_pcr = self.pcr_predict(cv=cv)
        rmse_pcr = root_mean_squared_error(self.Y_proc, yhat_pcr)
        print(f"PCR({ncomp}) RMSE (oof): {rmse_pcr:.6f}")

        yhat_mten = self.multitask_elasticnet_predict(cv=cv)
        rmse_mten = root_mean_squared_error(self.Y_proc, yhat_mten)
        print(f"MultiTaskElasticNet RMSE (oof): {rmse_mten:.6f}")

    def diagonal_affine_predict(self, cv: KFold | GroupKFold) -> ArrayF:
        X, Y = self.X_raw, self.Y_proc
        yhat = np.empty_like(Y)
        for tr_idx, te_idx in cv.split(X):
            # take out the sections of the arrays by index & normalize around average
            fold_stat = FoldStat.from_subset(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx])
            a, b = fit_diag_affine(fold_stat.train_raw_z, fold_stat.train_prc_z)
            yhat[te_idx] = predict_diag_affine(fold_stat.test_raw_z, a, b)
        return yhat

    def pcr_predict(self, cv: KFold | GroupKFold) -> ArrayF:
        X, Y = self.X_raw, self.Y_proc
        ncomp = max(2, min(X.shape[0] // 2, 32))
        yhat = np.empty_like(Y)
        for tr_idx, te_idx in cv.split(X):
            fold_stat = FoldStat.from_subset(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx])
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

    def multitask_elasticnet_predict(self, cv: KFold | GroupKFold) -> ArrayF:
        X, Y = self.X_raw, self.Y_proc
        yhat = np.empty_like(Y)
        for tr_idx, te_idx in tqdm(cv.split(X), total=cv.get_n_splits()):
            fold_stat = FoldStat.from_subset(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx])
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

    @staticmethod
    def from_annotations(
        csv_path: str | Path,
        base_dir: str | Path,
        ref_wl: None | ArrayF = None,
        max_pixels_per_scene: int | None = 20_000,
        spike_k: int | None = 7,
        base_poly: int | None = 2,
        s_poly: int | None = 3,
        s_window: int | None = 15,
    ) -> tuple[SpectraPair, DataArtifacts]:
        values = Annotations.read(csv_path, base_dir)
        x_list, y_list = [], []
        stats_per_scene, wls_per_scene, lens = [], [], []

        pre_config = PreConfig.make(
            spike_kernel_size=spike_k,
            baseline_poly=base_poly,
            s_poly=s_poly,
            s_window=s_window,
        )
        for r in values.rows:
            raw_map, prc_map = r.retrieve_maps()

            cube_x, cube_y, common_wl = raw_map.check_same_wavelength_grid(
                prc_map, ref_wl
            )

            cube_x, train_stats = preprocess_cube(
                cube_maybe=cube_x,
                wl_cm1=common_wl,
                pre_config=pre_config,
            )
            cube_y, _ = preprocess_cube(
                cube_maybe=CubeStats(cube_y, train_stats),
                wl_cm1=common_wl,
                pre_config=pre_config,
            )

            xpix = cube_x.get().reshape(-1, cube_x.M).astype(np.float64)
            ypix = cube_y.get().reshape(-1, cube_x.M).astype(np.float64)

            if (
                max_pixels_per_scene is not None
                and xpix.shape[0] > max_pixels_per_scene
            ):
                # if max pixels N surpassed, ensure that final array is size N
                idx = RNG.choice(
                    xpix.shape[0], size=max_pixels_per_scene, replace=False
                )
                xpix, ypix = xpix[idx], ypix[idx]

            x_list.append(xpix)
            y_list.append(ypix)
            stats_per_scene.append(train_stats)
            wls_per_scene.append(common_wl)
            lens.append(cube_x.H * cube_x.W)

        x_all = np.vstack(x_list, dtype=np.float64)
        y_all = np.vstack(y_list, dtype=np.float64)
        arts = DataArtifacts(stats_per_scene, wls_per_scene, lens, pre_config)
        return SpectraPair(x_all, y_all), arts


class PixelSpectraDataset(Dataset[SpectralData]):
    """Dataset for operating on individual spectra data, pixel by pixel."""

    def __init__(self, X: ArrayF32, Y: ArrayF32):
        assert X.shape == Y.shape, "X and Y shapes differ, must be same shaped arrays"
        self.raw: FlatMap = FlatMap.make(X.astype(np.float32, copy=False))
        self.prc: FlatMap = FlatMap.make(Y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.raw.shape[0]

    @override
    def __getitem__(self, idx: int) -> SpectralData:
        X = torch.from_numpy(self.raw.get()[idx]).view(-1, 1, 1)
        Y = torch.from_numpy(self.prc.get()[idx]).view(-1, 1, 1)
        return X, Y
