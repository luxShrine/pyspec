"""Preproccessing Helpers"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from loguru import logger
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

from pyspectral.core import Cube, FlatMap
from pyspectral.types import (
    Arr1DF,
    Arr2DF,
    ArrayF,
    ArrayF32,
    BaselinePolynomialDegree,
    SpectralMode,
)

type BaselineMethod = BaselinePolynomialDegree | ALS | None
DEFAULT_BASE_POLY = BaselinePolynomialDegree(2)


def _median_spike_removal(y: FlatMap, k: int | None) -> Arr2DF:
    """Apply 1D median filter of window k to each spectrum (rows of y)."""
    if k is None:
        logger.debug("No median spike removal pre-processing done.")
        return y.get()
    if k % 2 == 0:
        k += 1
        logger.warning(f"Adjusted kernel size to be odd: {k=}")
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = medfilt(y[i], kernel_size=k)
    return out


def poly_baseline_subtract(
    y: ArrayF, wl: ArrayF, degree: int | None
) -> tuple[ArrayF, ArrayF]:
    """Fit polynomial baseline per spectrum; subtract it."""
    out = np.empty_like(y)
    baseline = np.empty_like(y)
    if degree is None:
        logger.debug("No baseline subtraction done.")
        return baseline, y
    x = wl.astype(np.float64)
    V = np.vander(x, N=degree + 1, increasing=False)  # (M, deg+1)
    # Precompute pseudo-inverse
    V_pinv = np.linalg.pinv(V)
    # TODO: try to use np.linalg.lstsq(V, y[i], rcond=None)[0] in the loop
    for i in range(y.shape[0]):
        # least squares solution of: x = (y_i - [(V^-1 * y_i) * V])
        coeff = V_pinv @ y[i].astype(np.float64)
        base = V @ coeff
        baseline[i] = base
        out[i] = y[i] - base
    return baseline, out


def normalize_to_peak(
    y: np.ndarray,
    wl: ArrayF,
    center: float,
    halfwidth: float,
) -> ArrayF:
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
    return np.asarray(y / denom, dtype=np.float64)


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
    prom_peak = int(np.argmax(props["prominences"]))
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

    def smooth(self, y: np.ndarray) -> ArrayF:
        sm = savgol_filter(y, window_length=self.window, polyorder=self.poly, axis=1)
        return sm.astype(np.float64)


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

    def _solve_for_z(
        self,
        dt_d: sparse.csc_matrix,
        wi: Arr1DF,
        yi: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        C: int,
    ) -> Arr1DF:
        """Construct A = W + λ D^T D, and solve for z in Az = Wy."""
        lam = self.smoothness

        W: sparse.csc_matrix = sparse.diags(wi, 0, shape=(C, C), format="csc")
        A: sparse.csc_matrix = W + (lam * dt_d)
        b = wi * yi  # W y
        zi: Arr1DF = spsolve(A, b)
        return zi

    def remove_baseline(
        self,
        y: Arr1DF | Arr2DF,
    ) -> tuple[np.ndarray, ArrayF]:
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

        y = np.moveaxis(y, axis, -1)  # operate on last axis
        orig_shape = y.shape
        C = y.shape[-1]
        Y = y.reshape(-1, C)  # (M, C) where M = prod(other dims)

        # Build second-difference operator (D^2), a second derivative to act on z
        # d_op acts as a penalty to non-smooth curves, these high slope regions
        # are likely to be peaks, not a part of the baseline
        d_op: sparse.csc_matrix = sparse.diags(
            [1, -2, 1],
            [0, 1, 2],
            shape=(C - 2, C),
            format="csc",
        )
        dt_d: sparse.csc_matrix = (d_op.T @ d_op).tocsc()

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
            yi: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = Y[i]
            wi = np.ones(C, dtype=float)  # reset every loop

            # Downweight on peak regions, using mask
            p_local = np.full(C, p, dtype=float)
            DOWN_FACTOR = 0.1
            p_local[peak_mask_batch[i]] = min(p * DOWN_FACTOR, 0.0005)

            zi = None
            zi_prev: None | np.ndarray = None
            for _it in range(self.n_iter):
                zi = np.asarray(self._solve_for_z(dt_d, wi, yi, C), dtype=float)

                # check for convergence
                if zi_prev is not None:
                    denom = cast(float, max(np.linalg.norm(zi_prev), 1e-12))
                    numer = np.asarray(zi - zi_prev)
                    if (np.linalg.norm(numer) / denom) < self.tolerance:
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
        return baseline, corrected.astype(np.float64)


@dataclass
class PreConfig:
    smoothing: SmoothCfg | None = None
    mode: SpectralMode = SpectralMode.RAMAN
    spike_kernel_size: int | None = 7
    baseline: BaselineMethod = DEFAULT_BASE_POLY

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
        spike_removal: bool = False,
        mode: SpectralMode = SpectralMode.RAMAN,
        smoothness: float = 1e5,
        asymmetry: float = 0.01,
        n_iter: int = 20,
        tolerance: float = 1e-4,
    ) -> PreConfig:
        """Provide a minimal pre-processing config."""
        spike_kernel_size = 7 if spike_removal else None
        return cls(
            smoothing=None,
            mode=mode,
            spike_kernel_size=spike_kernel_size,
            baseline=ALS(
                smoothness=smoothness,
                asymmetry=asymmetry,
                n_iter=n_iter,
                tolerance=tolerance,
            ),
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
        y: ArrayF,
    ) -> ArrayF:
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
    ArrayF, PreprocStats | None, int | float | None, int | float | None
]


@dataclass(frozen=True)
class PeakNormConfig:
    mode: PeakNormMode
    # used in all modes where a center exists
    halfwidth_cm1: float = 16.0

    def fit_preproc(
        self,
        Y_train: ArrayF,  # (N_train, M), after spike/baseline/smoothing
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
        self, cube_maybe: Cube | CubeStats, y: np.ndarray, wl_cm1: ArrayF
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
            y = normalize_to_peak(y, wl_cm1, ref_center, ref_halfw)
            return y, stats, ref_center, ref_halfw

        stats = self.fit_preproc(y, wl_cm1)
        return y, stats, ref_center, ref_halfw


@dataclass
class CubeStats:
    cube: Cube
    stats: PreprocStats


def preprocess_cube(
    cube_maybe: Cube | CubeStats,
    wl_cm1: ArrayF,  # (C,)
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

    cube = cube_maybe.cube if not isinstance(cube_maybe, Cube) else cube_maybe

    y_flat = cube.flatten()  # (N, M)
    # remove spikes
    y = _median_spike_removal(y_flat, pre_config.spike_kernel_size)

    # find & remove baseline
    baseline_method = pre_config.baseline
    if isinstance((degree := baseline_method), int):
        _, y = poly_baseline_subtract(y, wl_cm1, degree=degree)
    elif isinstance(baseline_method, ALS):
        _baseline, y = baseline_method.remove_baseline(y)

    # smoothing
    if (s := pre_config.smoothing) is not None:
        y = s.smooth(y)

    # normalization to peak
    if pre_config.mode == SpectralMode.RAMAN:
        y, stats, ref_center, ref_halfw = peak_cfg.normalize(cube_maybe, y, wl_cm1)
    else:
        raise NotImplementedError()

    # compute stats for testing set
    stats = stats if stats is not None else PreprocStats(ref_center, ref_halfw)

    if isinstance(y, FlatMap):
        return Cube.from_flat(y.get(), cube.H, cube.W, cube.M), stats
    return Cube.make(y.reshape(cube.H, cube.W, cube.M)), stats


@dataclass
class SameGridCubes:
    xcube: Cube
    ycube: Cube
    common_wl: ArrayF

    def pre_process(
        self, pre_config: PreConfig, peak_cfg: PeakNormConfig | None = None
    ) -> tuple[Cube, Cube, PreprocStats]:
        cube_x, train_stats = preprocess_cube(
            cube_maybe=self.xcube,
            wl_cm1=self.common_wl,
            pre_config=pre_config,
            peak_cfg=peak_cfg,
        )
        cube_y, _ = preprocess_cube(
            cube_maybe=CubeStats(self.ycube, train_stats),
            wl_cm1=self.common_wl,
            pre_config=pre_config,
            peak_cfg=peak_cfg,
        )
        return cube_x, cube_y, train_stats
