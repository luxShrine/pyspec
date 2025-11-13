from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pyspectral.config import REF_PHE, ArrayF, ArrayF32, MeanArray, StdArray


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


# TODO:
def get_concentration(absorption: ArrayF, k: ArrayF):
    # Regression, we can use least squares to measure concentration:
    # minimizing function: S = \sum_i=1^n (y_i - f_i)^2
    # => c = (K^T A) / (K^T K)
    # with K being the absorbance coefficient & A the light absorption
    # with variance of:
    # σ^2 ≈ S/(N-1)
    numerator = k.T @ absorption
    denom = np.linalg.inv(k.T @ k)
    return denom @ numerator


@dataclass
class RegionSet:
    """Characteristic spectral features in 800–1000 cm⁻¹, 1300–1500 cm⁻¹, and 1500–1800 cm⁻¹"""

    low: float = 900.0
    mid: float = 1400.0
    high: float = 1650.0
    lo_window: tuple[float, float] = field(init=False)
    mid_window: tuple[float, float] = field(init=False)
    hi_window: tuple[float, float] = field(init=False)
    lo_distance: float = field(init=False)
    mid_distance: float = field(init=False)
    hi_distance: float = field(init=False)

    # float (0, 1): percent half-width
    # int (0 <= ): absolute half-width (cm^-1)
    window_range: float | int = 250

    def __post_init__(self) -> None:
        def hw(center: float) -> float:
            win_range = float(self.window_range)
            return win_range * center if 0.0 < win_range <= 1.0 else win_range

        r1 = hw(self.low)
        r2 = hw(self.mid)
        r3 = hw(self.high)
        # prevent negative region, could also add high boundaries
        self.lo_window = (max(self.low - r1, 0), self.low + r1)
        self.mid_window = (max(self.mid - r2, 0), self.mid + r2)
        self.hi_window = (max(self.high - r3, 0), self.high + r3)
        # adjust if overlap greater than 0.1
        while (0.9 * self.lo_window[1]) > self.mid_window[0]:
            r1 = r1 * 0.9
            r2 = r2 * 0.9
            self.lo_window = (max(self.low - r1, 0), self.low + r1)
            self.mid_window = (max(self.mid - r2, 0), self.mid + r2)
        while (0.9 * self.mid_window[1]) > self.hi_window[0]:
            r2 = r2 * 0.9
            r3 = r3 * 0.9
            self.mid_window = (max(self.mid - r2, 0), self.mid + r2)
            self.hi_window = (max(self.high - r3, 0), self.high + r3)

        self.lo_distance = 2 * r1
        self.mid_distance = 2 * r2
        self.hi_distance = 2 * r3

    def __iter__(self):
        win = [self.lo_window, self.mid_window, self.hi_window]
        dist = [self.lo_distance, self.mid_distance, self.hi_distance]
        yield from zip(win, dist)

    def get_low(self) -> tuple[float, float]:
        return self.lo_window

    def get_mid(self) -> tuple[float, float]:
        return self.mid_window

    def get_high(self) -> tuple[float, float]:
        return self.hi_window

    def match(self, sample: float) -> Literal["low", "middle", "high"] | None:
        """Find if sample is found within expected features, if so, return which feature."""
        if self.lo_window[0] <= sample and sample <= self.lo_window[1]:
            return "low"
        elif self.mid_window[0] <= sample and sample <= self.mid_window[1]:
            return "middle"
        elif self.hi_window[0] <= sample and sample <= self.hi_window[1]:
            return "high"
        return None


def _filter_array(
    array: ArrayF, wl: ArrayF, lower_bound: float, upper_bound: float
) -> tuple[ArrayF, ArrayF]:
    mask = (wl >= lower_bound) & (wl <= upper_bound)
    if not np.any(mask):
        return np.empty(0, dtype=array.dtype), np.empty(0, dtype=array.dtype)
    return array[..., mask], wl[mask]


def _area_under_curve(y: ArrayF, x: ArrayF) -> float:
    if y.size < 2:
        return np.nan
    return float(np.trapezoid(y, x))


def _log_ratio_area(
    roi: ArrayF,
    baseline_region: ArrayF,
    wl_roi: ArrayF,
    wl_base: ArrayF,
) -> float:
    # return nan if nan
    ar = _area_under_curve(roi, wl_roi)
    ab = _area_under_curve(baseline_region, wl_base)
    if not np.isfinite(ar) or not np.isfinite(ab):
        return np.nan
    # prevent negative with small min
    min = np.median(roi) * 0.05
    roi_area = max(ar, min)
    base_area = max(ab, min)
    ratio = np.log(base_area / roi_area)
    return ratio


def _get_dim_in_region(roi: ArrayF, wl_roi: ArrayF):
    elements = roi.size
    if elements == 0:
        return np.nan, np.nan
    i = int(np.nanargmax(roi))
    peak = float(roi[i])
    if not np.isfinite(peak) or elements < 3 or peak <= 0:
        return peak, np.nan

    half_max = 0.5 * peak
    # left crossing (<= hp) just before peak
    left_idx = np.where(roi[:i] <= half_max)[0]
    right_idx = np.where(roi[i:] <= half_max)[0]

    if left_idx.size == 0 and right_idx.size == 0:
        return peak, np.nan
    else:
        xl, xr = 0, 0
        if left_idx.size != 0:
            li = left_idx[-1]  # last index <= hm on the left
            xl = np.interp(li, xp=roi, fp=wl_roi)
        if right_idx.size != 0:
            ri = i + right_idx[0]  # first index <= hm on the right
            xr = np.interp(ri, xp=roi, fp=wl_roi)

        # full width or half width
        return peak, float(abs(xr - xl))


@dataclass(frozen=True, slots=True)
class PresenceMap:
    lo: np.float64
    mid: np.float64
    hi: np.float64

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None):
        arr = np.asarray([self.lo, self.mid, self.hi], dtype=dtype)
        return arr.copy if copy else arr

    @classmethod
    def from_ratios(cls, ratios: list[np.float64]) -> PresenceMap:
        if (b := len(ratios)) > 3:
            raise ValueError(f"Maps contain more bands than expected: {b}")
        elif (b := len(ratios)) < 3:
            raise ValueError(f"Maps contain less bands than expected: {b}")
        else:
            lo = ratios[0]
            mid = ratios[1]
            hi = ratios[2]
            return PresenceMap(lo, mid, hi)


type Dimensions = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


@dataclass
class SpectraBandFeatures:
    presence_map: PresenceMap
    dims: Dimensions

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None):
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")

        presence_map = np.asarray(self.presence_map, dtype=np.float64)
        dims = np.asarray(self.dims, dtype=np.float64).ravel()

        return np.concat([dims, presence_map], dtype=dtype)


def _disjoint_baseline_window(
    band_win: tuple[float, float], base_center: float, half_width: float
) -> tuple[float, float]:
    lo, hi = base_center - half_width, base_center + half_width
    # shift baseline window away if overlapping
    if not (hi <= band_win[0] or lo >= band_win[1]):
        # prefer shifting upward
        shift = (band_win[1] - lo) + 10.0  # 10 cm^-1 buffer
        lo, hi = lo + shift, hi + shift
    return lo, hi


def create_specband_feats(
    spectra: ArrayF,
    wl: ArrayF,
    regions: RegionSet,
    baseline_point: float = REF_PHE,
) -> SpectraBandFeatures:
    dims_list: list[tuple[float, float]] = []
    ratios: list[np.float64] = []

    for band_win, dist in regions:
        roi, wl_roi = _filter_array(spectra, wl, band_win[0], band_win[1])
        base_lo, base_hi = _disjoint_baseline_window(band_win, baseline_point, dist / 2)
        baseline, wl_base = _filter_array(spectra, wl, base_lo, base_hi)

        region_ratio = _log_ratio_area(roi, baseline, wl_roi, wl_base)
        ratios.append(np.float64(region_ratio))

        h, w = _get_dim_in_region(roi, wl_roi)
        dims_list.append((h, w))

    presence_map = PresenceMap.from_ratios(ratios)
    dims: Dimensions = np.asarray(dims_list, dtype=np.float64)  # (3,2)
    return SpectraBandFeatures(presence_map, dims)
