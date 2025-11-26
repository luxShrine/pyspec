from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
import numpy as np
import numpy.typing as npt
from numpy.typing import DTypeLike
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyspectral.config import REF_PHE
from pyspectral.core import apply_z
from pyspectral.types import (
    Arr1DF,
    ArrayF,
    ArrayF32,
    MeanArray,
    StdArray,
    UnitFloat,
)

# -- general helpers -------------------------------------------------------


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


# -- per fold details -------------------------------------------------------


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


class NormalizedData:
    """
    Input Args:
        array: shape (N, C)

    Fields:
        original: shape (N, C)
        mean: shape (C,)
        std: shape (C,)
        z: shape (N, C), normalized original array


    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        axis: int | None = None,
        dtype: DTypeLike | None = None,
        mean: MeanArray | None = None,
        std: StdArray | None = None,
    ):
        """Optionally provide an overiding std and mean"""
        self.original: np.ndarray = array
        self.mean: MeanArray = (
            MeanArray(array.mean(axis=axis, dtype=dtype)) if mean is None else mean
        )
        self.std: StdArray = (
            StdArray(array.std(axis=axis, dtype=dtype) + 1e-8) if std is None else std
        )  # prevent zero std
        self.z: np.ndarray = apply_z(array, self.mean, self.std)

    def get_stats(self) -> tuple[MeanArray, StdArray]:
        return self.mean, self.std


@dataclass(frozen=True, slots=True)
class FoldStat:
    tr_x_znorm: NormalizedData
    tr_y_znorm: NormalizedData
    te_x_znorm: NormalizedData
    te_y_znorm: NormalizedData

    def get_baseline(self) -> tuple[ArrayF, ArrayF]:
        x_std = self.tr_x_znorm.z
        # find the slope & intercept for linear relation between raw and correct z weights; shape: (C,), (C,)
        a, b = fit_diag_affine(x_std, self.tr_y_znorm.z)
        # using the train fit, predict the normalized result
        yhat_test_std = predict_diag_affine(x_std, a, b)  # (n_test, C)
        # inverse to original prediction units for reporting; shape: (n_test, C)
        yhat_test_orig = (yhat_test_std * self.tr_y_znorm.std) + self.tr_y_znorm.mean
        return yhat_test_std, yhat_test_orig

    @classmethod
    def from_subset(
        cls,
        raw_train_subset: np.ndarray,
        prc_train_subset: np.ndarray,
        raw_test_subset: np.ndarray,
        prc_test_subset: np.ndarray,
    ) -> FoldStat:
        # Per-fold std deviation & mean (fit on train only)
        tr_raw_z = NormalizedData(raw_train_subset, axis=0)
        tr_prc_z = NormalizedData(prc_train_subset, axis=0)
        raw_mean, raw_std = tr_raw_z.get_stats()
        prc_mean, prc_std = tr_prc_z.get_stats()
        te_raw_z = NormalizedData(raw_test_subset, mean=raw_mean, std=raw_std)
        te_prc_z = NormalizedData(prc_test_subset, mean=prc_mean, std=prc_std)

        return cls(
            tr_x_znorm=tr_raw_z,
            tr_y_znorm=tr_prc_z,
            te_x_znorm=te_raw_z,
            te_y_znorm=te_prc_z,
        )


# -- pixel to class -------------------------------------------------------


def convert_to_class(pres: float, *, maybe_value: float | None = 0.5):
    """converts presence from ∈ {0,2} to ∈ {0,1}.

    Args:
        maybe_value: float or None, if None, we drop the value.
    """
    if pres == 0:
        return UnitFloat(0.0)
    elif pres == 2:
        return UnitFloat(1.0)
    else:
        if maybe_value is None:
            return maybe_value
        return UnitFloat(maybe_value)


def convert_arr_to_class(
    arr: np.ndarray[tuple[int]],
    *,
    dtype: None | npt.DTypeLike = None,
    maybe_value: float | None = 0.5,
) -> np.ndarray[tuple[int]]:
    return np.asarray(
        [convert_arr_to_class(i, maybe_value=maybe_value) for i in arr], dtype=dtype
    )


def make_pca_features(X_train: np.ndarray, X_test: np.ndarray, n_components: int = 10):
    # standardize per-band
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    pca = PCA(n_components=n_components, svd_solver="full")
    Z_train = pca.fit_transform(X_train_sc)
    Z_test = pca.transform(X_test_sc)
    return Z_train.astype("float32"), Z_test.astype("float32"), pca, scaler


# -- per pixel details -------------------------------------------------------


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


def _log_ratio_area(
    roi: ArrayF,
    baseline_region: ArrayF,
    wl_roi: ArrayF,
    wl_base: ArrayF,
) -> float:
    ar = _area_under_curve(roi, wl_roi)
    ab = _area_under_curve(baseline_region, wl_base)

    # if we cant compute area, treat it as no contrast, zero
    if not np.isfinite(ar) or not np.isfinite(ab):
        return 0.0

    # prevent negative with small min
    min = np.median(roi) * 0.05
    roi_area = max(ar, min)
    base_area = max(ab, min)

    # clip ratio if above/below certain extremes
    raw_ratio = base_area / roi_area
    clip_ratio = np.clip(raw_ratio, 1e-5, 1e5)
    if raw_ratio != clip_ratio:
        logger.debug(f"Clipping ratio {raw_ratio} -> {clip_ratio}")
    return np.log(clip_ratio)


def _get_dim_in_region(roi: ArrayF, wl_roi: ArrayF) -> tuple[float, float]:
    elements = roi.size
    if elements == 0:
        return 0.0, 0.0

    i = int(np.nanargmax(roi))
    peak = float(roi[i])
    if not np.isfinite(peak) or elements < 3 or peak <= 0:
        return peak, 0.0

    half_max = 0.5 * peak
    # left crossing (<= hm) just before peak
    left_idx = np.where(roi[:i] <= half_max)[0]
    right_idx = np.where(roi[i:] <= half_max)[0]

    if left_idx.size == 0 and right_idx.size == 0:
        return peak, 0.0

    xl, xr = 0, 0
    if left_idx.size != 0:
        li = left_idx[-1]  # last index <= hm on the left
        xl = np.interp(li, xp=roi, fp=wl_roi)
    if right_idx.size != 0:
        ri = i + right_idx[0]  # first index <= hm on the right
        xr = np.interp(ri, xp=roi, fp=wl_roi)

    width = float(abs(xr - xl))
    # full width or half width
    return peak, width


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


# -- helper to build spectral band features ------------------------------------------------------


def create_specband_feats(
    spectra: Arr1DF,
    wl: Arr1DF,
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
