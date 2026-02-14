from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from loguru import logger
import numpy as np
import numpy.typing as npt

from pyspectral.config import REF_PHE
from pyspectral.types import (
    Arr1DF,
    ArrayF,
)

# -- general helpers -------------------------------------------------------

type Dimensions = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


def _filter_array(
    to_filter_array: ArrayF, basis_array: ArrayF, lower_bound: float, upper_bound: float
) -> tuple[ArrayF, ArrayF]:
    """Filter an array based on the values of the basis array."""
    mask = (basis_array >= lower_bound) & (basis_array <= upper_bound)
    if not np.any(mask):
        return np.empty(0, dtype=to_filter_array.dtype), np.empty(
            0, dtype=to_filter_array.dtype
        )
    return to_filter_array[..., mask], basis_array[mask]


def _area_under_curve(y: ArrayF, x: ArrayF) -> float:
    if y.size < 2:
        return np.nan
    return float(np.trapezoid(y, x))


# -- pixel to class -------------------------------------------------------


# TODO: move to core?
def convert_to_class(pres: float, *, maybe_value: float | None = 0.5) -> float | None:
    """Normalize a presence scalar to {0.0, 0.5, 1.0}.

    Args:
        maybe_value: float or None, if None, we drop the value.
    """
    if np.isclose(pres, 0.0):
        return 0.0
    if np.isclose(pres, 1.0) or np.isclose(pres, 2.0):
        return 1.0
    if np.isclose(pres, 0.5):
        return 0.5
    return maybe_value


def convert_arr_to_class(
    arr: np.ndarray[tuple[int]],
    *,
    dtype: None | npt.DTypeLike = None,
    maybe_value: float | None = 0.5,
) -> np.ndarray[tuple[int]]:
    return np.asarray(
        [convert_to_class(i, maybe_value=maybe_value) for i in arr], dtype=dtype
    )


# -- per pixel details -------------------------------------------------------


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
    window_range: float | int = 0.01

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

    def get_window(self, center: float) -> float:
        """Potentially convert window range to distance from relative precentage."""
        win_range = float(self.window_range)
        win_range_is_percent = 0.0 < win_range <= 1.0
        return (win_range * center) if win_range_is_percent else win_range

    def __iter__(self) -> Iterator[tuple[tuple[float, float], float]]:
        win = [self.lo_window, self.mid_window, self.hi_window]
        dist = [self.lo_distance, self.mid_distance, self.hi_distance]
        yield from zip(win, dist)

    def get_dims(
        self, spectra: ArrayF, wl: ArrayF
    ) -> tuple[Dimensions, list[tuple[ArrayF, ArrayF]]]:
        """Returns a (3,2) dimension indicating feature, along with the ROI filtered arrays."""
        # filters each region, for the intensities and wl
        roi_filtered_arr = list(
            map(lambda filt: _filter_array(spectra, wl, filt[0][0], filt[0][1]), self)
        )
        dims_list: list[tuple[float, float]] = [
            _get_dim_in_region(roi, wl_roi) for (roi, wl_roi) in roi_filtered_arr
        ]
        dims: Dimensions = np.asarray(dims_list)
        return (dims, roi_filtered_arr)


@dataclass(frozen=True, slots=True)
class PresenceWindows:
    lo: np.float64
    mid: np.float64
    hi: np.float64

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        arr = np.asarray([self.lo, self.mid, self.hi], dtype=dtype)
        return arr.copy() if copy else arr

    @classmethod
    def from_ratios(cls, ratios: list[np.float64]) -> PresenceWindows:
        if (b := len(ratios)) > 3:
            raise ValueError(f"Maps contain more bands than expected: {b}")
        elif (b := len(ratios)) < 3:
            raise ValueError(f"Maps contain less bands than expected: {b}")
        else:
            lo = ratios[0]
            mid = ratios[1]
            hi = ratios[2]
            return cls(lo, mid, hi)


@dataclass(frozen=True, slots=True)
class FullPresenceWindows:
    """Alternative to the PresenceWindows, this contains the full spectra contained in the windows."""

    lo: ArrayF
    mid: ArrayF
    hi: ArrayF

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        arr = np.concat([self.lo, self.mid, self.hi], dtype=dtype)
        return arr.copy() if copy else arr

    @classmethod
    def from_regions(cls, arrays: list[ArrayF]) -> FullPresenceWindows:
        if (b := len(arrays)) > 3:
            raise ValueError(f"Maps contain more bands than expected: {b}")
        elif (b := len(arrays)) < 3:
            raise ValueError(f"Maps contain less bands than expected: {b}")
        else:
            lo = arrays[0]
            mid = arrays[1]
            hi = arrays[2]
            return cls(lo, mid, hi)


@dataclass
class SpectraBandFeatures:
    """Spectral features around particular bands.

    presence_map: (Full)PresenceWindows corresponding to the intensity spectrum or ratios of regions examined
    dims: Dimensions, a 3x2 feature array for the sprectal bands
    """

    presence_map: FullPresenceWindows | PresenceWindows
    dims: Dimensions

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")

        presence_map = np.asarray(self.presence_map, dtype=np.float64)
        dims = np.asarray(self.dims, dtype=np.float64).ravel()

        return np.concat([dims, presence_map], dtype=dtype)


def _log_ratio_area(
    roi: ArrayF,
    baseline_region: ArrayF,
    wl_roi: ArrayF,
    wl_base: ArrayF,
) -> float:
    roi_area = _area_under_curve(roi, wl_roi)
    base_area = _area_under_curve(baseline_region, wl_base)

    # if we cant compute area, treat it as no contrast, zero
    if not np.isfinite(roi_area) or not np.isfinite(base_area):
        return 0.0

    # prevent negative with small min
    min = np.median(roi) * 0.05
    roi_area = max(roi_area, min)
    base_area = max(base_area, min)

    # clip ratio if above/below certain extremes
    raw_ratio = base_area / roi_area
    clip_ratio = np.clip(raw_ratio, 1e-5, 1e5)
    if raw_ratio != clip_ratio:
        logger.debug(f"Clipping ratio {raw_ratio} -> {clip_ratio}")
    return float(np.log(clip_ratio))


def _get_dim_in_region(roi: ArrayF, wl_roi: ArrayF) -> tuple[float, float]:
    n = roi.size
    if n == 0:
        return 0.0, 0.0

    i = int(np.nanargmax(roi))
    peak = float(roi[i])
    if not np.isfinite(peak) or n < 3 or peak <= 0:
        return peak, 0.0

    half = 0.5 * peak

    # left crossing: last point <= half before peak
    left = np.where(roi[:i] <= half)[0]
    # right crossing: first point <= half after peak
    right = np.where(roi[i:] <= half)[0]

    if left.size == 0 or right.size == 0:
        return peak, 0.0

    li = int(left[-1])
    li2 = min(li + 1, i)
    xpl = [roi[li], roi[li2]]  # using the intensity mappings as the x values
    fpl = [wl_roi[li], wl_roi[li2]]  # interpreting wls as they are the desired y values
    xl = float(np.interp(half, xpl, fpl))

    ri = int(i + right[0])
    ri1 = max(ri - 1, i)
    xpr = [roi[ri1], roi[ri]]
    fpr = [wl_roi[ri1], wl_roi[ri]]
    xr = float(np.interp(half, xpr, fpr))

    return peak, float(abs(xr - xl))


def _disjoint_baseline_window(
    band_win: tuple[float, float], base_center: float, half_width: float
) -> tuple[float, float]:
    """Shift the band window away from the baseline if overlapping."""
    lo, hi = base_center - half_width, base_center + half_width
    # shift baseline window away if overlapping
    if not (hi <= band_win[0] or lo >= band_win[1]):
        # prefer shifting upward
        shift = (band_win[1] - lo) + 10.0  # 10 cm^-1 buffer
        lo, hi = lo + shift, hi + shift
    return lo, hi


def create_specband_feats(
    spectra: Arr1DF,
    wl: Arr1DF,
    regions: RegionSet,
    baseline_point: float = REF_PHE,
    window_full: bool = False,
) -> SpectraBandFeatures:
    """Helper to build spectral band features."""

    (dims, roi_list) = regions.get_dims(spectra, wl)

    secondary_feature: FullPresenceWindows | PresenceWindows
    if window_full:
        # use full window
        roi_arr = [roi_intensity for (roi_intensity, _) in roi_list]
        secondary_feature = FullPresenceWindows.from_regions(roi_arr)
    else:
        # use ratios
        ratios: list[np.float64] = []
        for i, ((lo_band_win, hi_band_win), dist) in enumerate(regions):
            intensity_roi, wl_roi = roi_list[i][0], roi_list[i][1]
            base_lo, base_hi = _disjoint_baseline_window(
                (lo_band_win, hi_band_win), baseline_point, dist / 2
            )
            intensity_base, wl_base = _filter_array(spectra, wl, base_lo, base_hi)
            region_ratio = _log_ratio_area(
                intensity_roi, intensity_base, wl_roi, wl_base
            )
            ratios.append(np.float64(region_ratio))
        secondary_feature = PresenceWindows.from_ratios(ratios)

    return SpectraBandFeatures(secondary_feature, dims)
