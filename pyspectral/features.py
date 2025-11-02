from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum, auto
from pathlib import Path
from typing import NewType

from loguru import logger
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

from pyspectral.config import ArrayF, ArrayF32, Cube, FlatMap, MeanArray, StdArray

ALGINATE_PEAK_CM = 1003.0  # 1,410 cm^-1 peaks, other potentials: 3250, 2940, 1600, 1020


class SpectralMode(StrEnum):
    RAMAN = auto()
    REFLECTANCE = auto()  # alternative method not used


def fit_diag_affine(x_std: ArrayF32, y_std: ArrayF32) -> tuple[ArrayF, ArrayF]:
    """
    Fit per-band y ~ a*x + b on train (both x,y are z-scored per-band using training stats).
    Args: x_std, y_std: (n_train, C)
    Returns: a, b: (C,), (C,)
    """
    x_std_m = x_std.mean(axis=0)  # (C,)
    y_std_m = y_std.mean(axis=0)  # (C,)
    x_center = x_std - x_std_m  # (n,C)
    y_center = y_std - y_std_m  # (n,C)
    x2 = np.sum(x_center * x_center, axis=0)  # Var * n (C,)
    xy = np.sum(x_center * y_center, axis=0)  # Cov * n (C,)
    a = xy / np.clip(x2, 1e-12, None)  # (C,) slope per band
    b = y_std_m - a * x_std_m  # (C,) intercept per band
    return a, b


def predict_diag_affine(x_std: ArrayF32, a: ArrayF, b: ArrayF) -> ArrayF:
    """
    Apply per-band yhat = a*x + b in standardized space.
    Args: x_std: (n_test, C), a,b: (C,)
    Returns: yhat_std: (n_test, C)
    """
    return (a * x_std) + b


def assert_same_grid(a: ArrayF, b: ArrayF, tolerance: float = 1e-9) -> None | str:
    """Check that array 'a' and 'b' are the same shape and each element is within a tolerance."""
    # add to a single string so that we can see both issues potentially
    output: str = ""
    if a.shape != b.shape:
        output += f"Grid {a.shape=} not equal to {b.shape=}.\n"
    if not np.allclose(a, b, atol=tolerance):
        # np.allclose checks if the element by element array are the samef within tolerance
        output += f"Grids differ by supplied tolerance: {tolerance}."
    if output != "":
        return output
    return None


@dataclass(frozen=True, slots=True)
class FoldStat:
    x_mean: MeanArray  # (C,)
    y_mean: MeanArray
    x_std: StdArray  # (C,)
    y_std: StdArray
    train_raw_z: ArrayF32
    train_prc_z: ArrayF32
    test_raw_z: ArrayF32
    test_prc_z: ArrayF32

    def get_baseline(self) -> tuple[ArrayF, ArrayF]:
        # find the slope & intercept for linear relation between raw and correct z weights; shape: (C,), (C,)
        a, b = fit_diag_affine(self.train_raw_z, self.train_prc_z)
        # using the train fit, predict the normalized result
        yhat_test_std = predict_diag_affine(self.test_raw_z, a, b)  # (n_test, C)
        # inverse to original prediction units for reporting; shape: (n_test, C)
        yhat_test_orig = (yhat_test_std * self.y_std) + self.y_mean
        return yhat_test_std, yhat_test_orig

    @staticmethod
    def apply_z(y: ArrayF32, mean: MeanArray, std: StdArray) -> ArrayF32:
        return ((y - mean) / std).astype(np.float32)

    @classmethod
    def from_subset(
        cls,
        raw_train_subset: ArrayF32 | ArrayF,
        prc_train_subset: ArrayF32 | ArrayF,
        raw_test_subset: ArrayF32 | ArrayF,
        prc_test_subset: ArrayF32 | ArrayF,
    ) -> FoldStat:
        # Per-fold std deviation & mean (fit on train only)
        raw_mean = MeanArray(raw_train_subset.mean(axis=0).astype(np.float32))
        prc_mean = MeanArray(prc_train_subset.mean(axis=0).astype(np.float32))
        raw_std = StdArray(
            raw_train_subset.std(axis=0).astype(np.float32) + 1e-8
        )  # prevent zero
        prc_std = StdArray(prc_train_subset.std(axis=0).astype(np.float32) + 1e-8)

        # normalize around average
        tr_raw_z = cls.apply_z(
            raw_train_subset.astype(np.float32, copy=False), raw_mean, raw_std
        )  # (n_test, C)
        tr_prc_z = cls.apply_z(
            prc_train_subset.astype(np.float32, copy=False), prc_mean, prc_std
        )
        te_raw_z = cls.apply_z(
            raw_test_subset.astype(np.float32, copy=False), raw_mean, raw_std
        )
        te_prc_z = cls.apply_z(
            prc_test_subset.astype(np.float32, copy=False), prc_mean, prc_std
        )
        return cls(
            x_mean=raw_mean,
            y_mean=prc_mean,
            x_std=raw_std,
            y_std=prc_std,
            train_raw_z=tr_raw_z,
            train_prc_z=tr_prc_z,
            test_raw_z=te_raw_z,
            test_prc_z=te_prc_z,
        )


@dataclass(frozen=True, slots=True)
class HSIMap:
    """Contains:
    wl: (M,) float64 wavelengths
    xy: (N, 2) float64 stage coords
    spectra: (N, C) float32 intensities
    cube: (H, W, M) float32 reshaped spectral cube
    """

    wl: ArrayF
    xy: ArrayF
    spectra: FlatMap
    cube: Cube

    def check_same_wavelength_grid(
        self,
        prc: HSIMap,
        ref_wl: ArrayF | None = None,
    ) -> tuple[Cube, Cube, ArrayF]:
        cube_x = self.cube
        cube_y = prc.cube
        ref = self.wl if ref_wl is None else ref_wl
        if assert_same_grid(self.wl, ref) is not None:
            cube_x = resample_cube(self.cube, self.wl, ref)
        if assert_same_grid(prc.wl, ref) is not None:
            cube_y = resample_cube(prc.cube, prc.wl, ref)

        return cube_x, cube_y, ref

    @classmethod
    def load(
        cls,
        txt_path: Path,
        acq_time_s: float | None = None,
        accumulation: int | None = None,
    ) -> HSIMap:
        """
        Returns HSIMap containing:
            wl: (M,) float64 wavelengths in cm^-1
            xy: (N, 2) float64 stage coords (assumed µm)
            spectra: (N, M) float32 intensities
            cube: (H, W, M) float32 reshaped spectral cube (row-major by sorted y,x)
        """
        with txt_path.open("r") as f:
            # first non-empty line is wavelengths (M values)
            wl = np.fromstring(next(line for line in f if line.strip()), sep="\t")
            # remaining lines: x, y, I_1, ..., I_M
            rows = [np.fromstring(r_line, sep="\t") for r_line in f if r_line.strip()]
        xy_and_spectra = np.vstack(rows)  # N stages, x, y, and M wavelenghts (N, M+2)

        # get all values of the spatial x, y, which are the first two cols of each row (0: x, 1: y)
        xy = xy_and_spectra[:, :2]
        # after col2 are the intensity values per wl
        spectra = xy_and_spectra[:, 2:].astype(np.float32)
        assert spectra.shape[1] == wl.size, (
            f"band count mismatch {spectra.shape[1]} != {wl.size}"
        )

        if acq_time_s and accumulation:
            # normalize spectra to the aqcuisiton time
            spectra /= float(acq_time_s) * int(accumulation)

        # Get the unique positions, and sort x & y
        xs, _x_indices = np.unique(xy[:, 0], return_inverse=True)
        ys, _y_indices = np.unique(xy[:, 1], return_inverse=True)
        # infer grid and reshape to (H,W,M)
        H, W = ys.size, xs.size
        N = xy.shape[0]
        assert H * W == N, f"grid not rectangular ({H * W=} != {N=})"
        # order indices by (y,x) row-major
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        cube = Cube.from_flat(
            flat_cube=spectra[order], height=H, width=W, spec_bands=wl.size
        )
        return cls(wl, xy, FlatMap.make(spectra), cube)


@dataclass(frozen=True)
class PairRow:
    raw_path: Path
    proc_path: Path
    accumulation: int | None = None
    acquisition: float | None = None

    def retrieve_maps(self) -> tuple[HSIMap, HSIMap]:
        """Load pair of HSIMaps, (raw, processed)."""
        x_map = HSIMap.load(
            self.raw_path,
            self.acquisition,
            self.accumulation,
        )
        y_map = HSIMap.load(
            self.proc_path,
            self.acquisition,
            self.accumulation,
        )
        if err := assert_same_grid(x_map.xy, y_map.xy):
            raise RuntimeError(err)
        return (x_map, y_map)


# -- Preproccessing Helpers -------------------------------------------------------


def resample_cube(cube: Cube, wl_src: ArrayF, wl_dst: ArrayF) -> Cube:
    """Resample (H,W,M_src) -> (H,W,M_dst) by 1D linear interp per pixel."""
    flat = cube.get().reshape(-1, cube.M)
    out = np.empty((flat.shape[0], wl_dst.size), dtype=np.float32)
    for i in range(flat.shape[0]):  # NOTE: could vectorize this process
        out[i] = np.interp(wl_dst, wl_src, flat[i])
    return Cube.from_flat(
        flat_cube=out, height=cube.H, width=cube.W, spec_bands=wl_dst.size
    )


def _median_spike_removal(y: FlatMap, k: int | None) -> ArrayF32:
    """Apply 1D median filter of window k to each spectrum (rows of y)."""
    if k is None:
        logger.debug("No median spike removal pre-processing done.")
        return y
    if k % 2 == 0:
        k += 1
        logger.warning(f"Adjusted kernel size to be odd: {k=}")
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = medfilt(y[i], kernel_size=k)
    return out


def _poly_baseline_subtract(y: ArrayF32, wl: ArrayF, degree: int | None) -> ArrayF32:
    """Fit polynomial baseline per spectrum; subtract it."""
    if degree is None:
        logger.debug("No baseline subtraction done.")
        return y
    x = wl.astype(np.float64)
    out = np.empty_like(y)
    V = np.vander(x, N=degree + 1, increasing=False)  # (M, deg+1)
    # Precompute pseudo-inverse
    V_pinv = np.linalg.pinv(
        V
    )  # or use np.linalg.lstsq(V, y[i], rcond=None)[0] in the loop
    for i in range(y.shape[0]):
        # least squares solution of: x = (y_i - [(V^-1 * y_i) * V])
        coeff = V_pinv @ y[i].astype(np.float64)
        baseline = V @ coeff
        out[i] = (y[i] - baseline).astype(np.float32)
    return out


def _normalize_to_peak(
    y: ArrayF32,
    wl: ArrayF,
    center: float,
    halfwidth: float,
) -> ArrayF32:
    """Divide by max value in a small window around `center`."""
    mask = (wl >= (center - halfwidth)) & (wl <= (center + halfwidth))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        # check that mask is not all false
        raise ValueError(
            f"No wavelengths in window [{center - halfwidth}, {center + halfwidth}] "
            + f"given wl range [{wl.min()}, {wl.max()}]. Check units and peak center."
        )
    denom = y[:, mask].max(axis=1, keepdims=True) + 1e-8
    return (y / denom).astype(np.float32)  # type: ignore


def _refine_peak_center(wl: ArrayF, y: ArrayF, i: int) -> float:
    """Refine peak index i to sub-sample center using neighbouring points."""
    i0 = max(i - 1, 0)
    i2 = min(i + 1, y.size - 1)
    if i2 - i0 < 2:
        return float(wl[i])
    x = wl[i0 : i2 + 1]
    y = y[i0 : i2 + 1]
    # Fit y = ax^2 + bx + c on 3 points, vertex at -b/(2a)
    # Solve with np.polyfit which is stable for size-3.
    a, b, _ = np.polyfit(x, y, 2)
    if abs(a) < 1e-20:
        return float(x[1])  # looks flat; fall back to center sample
    return float(-b / (2 * a))


def find_peak_in_window(
    wl: ArrayF,
    spectrum: ArrayF,
    window: tuple[float, float],
    min_prom: float = 0.0,
    min_width: int = 1,
) -> float:
    """Pick the most prominent local max in a wl-window and refine center."""
    low, hi = window
    mask = (wl >= low) & (wl <= hi)
    if not np.any(mask):
        raise RuntimeError("No valid points found in window")
    y = spectrum[mask]
    idx, props = find_peaks(y, prominence=min_prom, width=min_width)
    if idx.size == 0:
        logger.warning("No peak found, falling back to argmax in window")
        k = int(np.argmax(y))
        return float(np.mean(wl[mask][max(k - 1, 0) : min(k + 2, y.size)]))
    # Choose the peak with max prominence
    prom_peak = int(np.argmax(props["prominences"]))  # pyright: ignore[reportTypedDictNotRequiredAccess]
    i_win = int(idx[prom_peak])

    # Map back to full indices
    i_full = int(np.nonzero(mask)[0][0]) + i_win
    return _refine_peak_center(wl, spectrum, i_full)


@dataclass(frozen=True)
class SmoothCfg:
    poly: int = 3
    window: int = 11

    def __post_init__(self) -> None:
        if self.window <= 5:
            raise RuntimeError(
                f"Smoothing window must be greater than (5), found {self.window=}"
            )
        if self.poly < 1 or self.poly >= 10:
            raise RuntimeError(
                f"Smooth polynomial value must be within range of (1-10), found {self.poly=}"
            )

    def smooth(self, y) -> ArrayF32:
        return savgol_filter(
            y, window_length=self.window, polyorder=self.poly, axis=1
        ).astype(np.float32)


BaselinePolynomialDegree = NewType("BaselinePolynomialDegree", int)


@dataclass
class ALS:
    """
    Parameters:
    lam : float
        Smoothness (λ). Larger => smoother baseline. Typical Raman 1e5–1e7.
    p : float
        Asymmetry. Small p (~0.001–0.05) downweights positive residuals, typically peaks.
    niter : int
        Max iterations of reweighting.
    tol : float
        Relative change tolerance for early stop.
    """

    smoothness: float = 1e5
    asymmetry: float = 0.01
    n_iter: int = 20
    tolerance: float = 1e-4

    def remove_baseline(
        self,
        y: np.ndarray,
    ) -> tuple[np.ndarray, FlatMap]:
        """
        Robust baseline via asymmetric least squares (ALS).

        Args:
        y : array
            1D spectrum (C,) or 2D array (N, C). Baseline is computed along `axis`.

        Returns:
        baseline : array
            Estimated baseline, same shape as `y`.
        corrected : array
            y - baseline

        Notes:
        Solves for z: argmin_z  (sum_i^C w_i (y_i - z_i)^2 + λ ||D^2 z||^2_2) ,
        with asymmetric weights w_i updated as:
            w_i = p   if (y_i - z_i) > 0  (peak region)
                = 1-p otherwise           (baseline region)

        # References
        Inspired by https://github.com/charlesll/Spectra.jl/blob/master/src/baseline.jl

        """
        # Percentile for mask. Values above this are considered peaks.
        MASK_PERCENTILE: float = 95.0
        # Dilate the mask by this many points on each side.
        MASK_HALF_WIDTH: int = 6
        axis: int = -1  # Spectral axis.
        p = self.asymmetry
        lam = self.smoothness

        y = np.asarray(y)
        y = np.moveaxis(y, axis, -1)  # operate on last axis
        orig_shape = y.shape
        C = y.shape[-1]
        Y = y.reshape(-1, C)  # (M, C) where M = prod(other dims)

        # Build second-difference operator (D^2), a second derivative to act on z
        # d_op acts as a penalty to non-smooth curves, these high slope regions
        # are likely to be peaks, not a part of the baseline
        d_op = sparse.diags([1, -2, 1], [0, 1, 2], shape=(C - 2, C), format="csc")
        dt_d = (d_op.T @ d_op).tocsc()

        # Construct peak mask (shared across rows)
        threshold = np.nanpercentile(Y, MASK_PERCENTILE, axis=-1, keepdims=True)
        mask = (Y > threshold).astype(int)
        kernel = np.ones(2 * MASK_HALF_WIDTH + 1, dtype=int)
        # apply: (a * v)_n = \sum_{m = -\infty}^{\infty} a_m v_{n - m} to the rows
        # The convolution finds the overlap between the kernel and the mask of the signal
        mask = np.apply_along_axis(
            lambda a: np.convolve(a=a, v=kernel, mode="same"), -1, mask
        )
        # only keep the matches
        mask_positive = mask > 0
        peak_mask_batch = mask_positive.astype(bool)

        # Iterate per-row, finding the penalized least square for some z with
        # the current weight w
        Z = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            yi = Y[i]
            wi = np.ones(C, dtype=float)  # reset every loop

            # Downweight on peak regions, using mask
            p_local = np.full(C, p, dtype=float)
            DOWN_FACTOR = 0.1
            p_local[peak_mask_batch[i]] = min(p * DOWN_FACTOR, 0.0005)

            zi_prev = None
            for it in range(self.n_iter):
                # Construct A = W + λ D^T D, and solve for z in Az = Wy
                W = sparse.diags(wi, 0, shape=(C, C), format="csc")
                A = W + (lam * dt_d)
                b = wi * yi  # W y
                zi = spsolve(A, b)

                # check for convergence
                if zi_prev is not None:
                    denom = max(np.linalg.norm(zi_prev), 1e-12)
                    if (np.linalg.norm(zi - zi_prev) / denom) < self.tolerance:
                        break
                # if not, continue on iterating
                zi_prev = zi
                # Update asymmetric weights
                residual = yi - zi
                positive_resid = residual > 0
                # determine if in peak region: p_local, else: 1 - p_local
                wi = np.where(positive_resid, p_local, 1.0 - p_local)
                # Prevent weights being zero
                wi = np.clip(wi, 1e-6, 1.0)

            Z[i] = zi

        baseline = Z.reshape(orig_shape)
        baseline = np.moveaxis(baseline, -1, axis)
        corrected = np.moveaxis(y.reshape(orig_shape), -1, axis) - baseline
        return baseline, FlatMap.make(corrected)


type BaselineMethod = BaselinePolynomialDegree | ALS | None


@dataclass
class PreConfig:
    smoothing: SmoothCfg | None = None
    mode: SpectralMode = SpectralMode.RAMAN
    spike_kernel_size: int | None = 7
    baseline: BaselineMethod = 2

    def __post_init__(self) -> None:
        if self.smoothing and ((w := self.smoothing.window) % 2 == 0):
            self.smoothing = replace(self.smoothing, window=(w + 1))
        if isinstance(self.baseline, int) and (
            self.baseline < 1 or self.baseline >= 10
        ):
            raise RuntimeError(
                f"Baseline polynomial variable must be within range of (1-10), found {self.baseline=}"
            )
        if self.spike_kernel_size and (
            self.spike_kernel_size < 1 or self.spike_kernel_size >= 20
        ):
            raise RuntimeError(
                f"Spike kernel variable must be within range of (1-20), found {self.spike_kernel_size=}"
            )

    @classmethod
    def make_min(cls, mode: SpectralMode = SpectralMode.RAMAN) -> PreConfig:
        """Provide a minimal pre-processing config."""
        return cls(None, mode, None, None)

    @classmethod
    def make_als(
        cls,
        mode: SpectralMode = SpectralMode.RAMAN,
        smoothness: float = 1e5,
        asymmetry: float = 0.01,
        n_iter: int = 20,
        tolerance: float = 1e-4,
    ) -> PreConfig:
        """Provide a minimal pre-processing config."""
        return cls(
            None,
            mode,
            ALS(
                smoothness=smoothness,
                asymmetry=asymmetry,
                n_iter=n_iter,
                tolerance=tolerance,
            ),
            None,
        )

    @classmethod
    def make_poly(
        cls,
        spike_kernel_size: int | None,
        baseline_poly: BaselinePolynomialDegree | None,
        s_poly: int | None = None,
        s_window: int | None = None,
    ) -> PreConfig:
        # create smoothing config
        if s_poly is not None and s_window is not None:
            smooth_cfg = SmoothCfg(s_poly, s_window)
        elif s_poly is not None:
            smooth_cfg = SmoothCfg(
                poly=s_poly,
            )
        elif s_window is not None:
            smooth_cfg = SmoothCfg(window=s_window)
        else:
            smooth_cfg = None

        if smooth_cfg is None and spike_kernel_size is None and baseline_poly is None:
            return cls.make_min()
        else:
            return cls(
                smoothing=smooth_cfg,
                mode=SpectralMode.RAMAN,
                spike_kernel_size=spike_kernel_size,
                baseline=baseline_poly,
            )


@dataclass(frozen=True)
class GuidedPeakNorm:
    window_cm1: tuple[float, float]
    mode_type: str = "Guided"


@dataclass(frozen=True)
class NonePeakNorm:
    mode_type: str = "None"


@dataclass(frozen=True)
class GlobalPeakNorm:
    mode_type: str = "Global"

    @staticmethod
    def normalize(
        y: ArrayF32,
    ) -> ArrayF32:
        """Preform a global normalization."""
        min = np.min(y)
        num = y - min
        den = np.max(y) - min
        return num / den


@dataclass(frozen=True)
class FixedPeakNorm:
    center_cm1: float
    mode_type: str = "Fixed"


type PeakNormMode = GuidedPeakNorm | NonePeakNorm | FixedPeakNorm | GlobalPeakNorm


@dataclass(frozen=True)
class PreprocStats:
    ref_center_cm1: float | None = None
    ref_halfwidth_cm1: float | None = None

    def get_center(self) -> float:
        if self.ref_center_cm1 is None:
            raise RuntimeError("Cannot get center of preprocess stats as it is none")
        else:
            return self.ref_center_cm1


type NormResult = tuple[
    ArrayF32, PreprocStats | None, int | float | None, int | float | None
]


@dataclass(frozen=True)
class PeakNormConfig:
    mode: PeakNormMode
    # used in all modes where a center exists
    halfwidth_cm1: float = 12.0

    def fit_preproc(
        self,
        Y_train: ArrayF32,  # (N_train, M), after spike/baseline/smoothing
        wl_cm1: ArrayF,  # (M,)
    ) -> PreprocStats:
        match self.mode:
            case GuidedPeakNorm():
                med_spec = np.median(Y_train, axis=0)
                center = find_peak_in_window(
                    wl=wl_cm1, spectrum=med_spec, window=self.mode.window_cm1
                )
                return PreprocStats(center, self.halfwidth_cm1)
            case FixedPeakNorm():
                return PreprocStats(self.mode.center_cm1, self.halfwidth_cm1)
            case _:
                return PreprocStats(None, None)

    def normalize(
        self, cube_maybe: Cube | CubeStats, y: ArrayF, wl_cm1: ArrayF
    ) -> NormResult:
        ref_center = None
        ref_halfw = None
        stats = None
        if isinstance(self.mode, GlobalPeakNorm):
            stats = self.fit_preproc(y, wl_cm1)
            return GlobalPeakNorm.normalize(y), stats, ref_center, ref_halfw
        elif not isinstance(self.mode, NonePeakNorm):
            if isinstance(cube_maybe, Cube):
                stats = self.fit_preproc(y, wl_cm1)
                ref_center = stats.get_center()
                ref_halfw = self.halfwidth_cm1
            elif (c := cube_maybe.stats.ref_center_cm1) is not None:
                # use the stats provided
                ref_center = c
                ref_halfw = cube_maybe.stats.ref_halfwidth_cm1 or self.halfwidth_cm1
            else:
                raise RuntimeError(
                    f"Invalid cube and stats combination {cube_maybe=}, {stats=}"
                )
            y = _normalize_to_peak(y, wl_cm1, ref_center, ref_halfw)
            return y, stats, ref_center, ref_halfw

        stats = self.fit_preproc(y, wl_cm1)
        return y, stats, ref_center, ref_halfw


@dataclass(frozen=True)
class DataArtifacts:
    scene_stats: list[PreprocStats]  # one per scene
    scene_wls: list[ArrayF]  # common wl per scene
    scene_lens: list[int]  # H*W per scene
    pre_config: PreConfig  # the preproc knobs used


@dataclass
class CubeStats:
    cube: Cube
    stats: PreprocStats


def preprocess_cube(
    cube_maybe: Cube | CubeStats,
    wl_cm1: ArrayF,  # (M,)
    pre_config: PreConfig | None = None,
    peak_cfg: PeakNormConfig | None = None,
) -> tuple[Cube, PreprocStats]:
    """
    Apply spike removal, baseline, optional peak normalization, smoothing, and z-score.
    If stats is None, z-score stats are computed on this cube.
    Returns (cube_processed, stats_used).
    """
    pre_config = PreConfig() if pre_config is None else pre_config
    peak_cfg = (
        PeakNormConfig(
            mode=GuidedPeakNorm(window_cm1=(1000.0, 1150.0))
            # mode=FixedPeakNorm( center_cm1=1004.0))
        )
        if peak_cfg is None
        else peak_cfg
    )

    cube = cube_maybe if isinstance(cube_maybe, Cube) else cube_maybe.cube

    y_flat = cube.flatten()  # (N, M)
    # remove spikes
    y = _median_spike_removal(y_flat, pre_config.spike_kernel_size)

    # find & remove baseline
    baseline_method = pre_config.baseline
    if isinstance((degree := baseline_method), int):
        y = _poly_baseline_subtract(y, wl_cm1, degree=degree)
    elif isinstance(baseline_method, ALS):
        _baseline, y = baseline_method.remove_baseline(y)

    # smoothing
    if (s := pre_config.smoothing) is not None:
        y = s.smooth(y)

    # normalization to peak
    if pre_config.mode == SpectralMode.RAMAN:
        y, stats, ref_center, ref_halfw = peak_cfg.normalize(cube_maybe, y, wl_cm1)

    # compute stats for testing set
    stats = stats if stats is not None else PreprocStats(ref_center, ref_halfw)

    if isinstance(y, FlatMap):
        return Cube.from_flat(y.get(), cube.H, cube.W, cube.M), stats
    return Cube.make(y.reshape(cube.H, cube.W, cube.M)), stats
