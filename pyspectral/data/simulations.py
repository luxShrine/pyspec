from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.signal import savgol_filter

from pyspectral.core import Cube
from pyspectral.types import Arr1DF, Arr2DF, ThirdCoeff


def find_nearest(array: np.ndarray, point: float) -> np.intp:
    """Find the nearest value's index in array to desired point."""
    idx: np.intp = np.abs(array - point).argmin()
    return idx


def get_fwhm(
    nu: Arr1DF,  # wavenumber grid, shape (C,)
    i_mean: Arr1DF,  # intensity, shape (C,)
    idx_peak: np.intp,
) -> float:
    """Estimate FWHM around a single peak index."""
    peak_val = i_mean[idx_peak]
    half_peak = peak_val / 2.0

    # Left side: indices [0, idx_peak]
    l_idx = idx_peak + 1
    left_int = i_mean[:l_idx]
    left_nu = nu[:l_idx]

    # Right side: indices [idx_peak, ...]
    right_int = i_mean[idx_peak:]
    right_nu = nu[idx_peak:]

    # Find nearest points to half_peak on each side
    l_idx_local = find_nearest(left_int, half_peak)
    r_idx_local = find_nearest(right_int, half_peak)

    # FWHM in cm^-1
    return float(right_nu[r_idx_local] - left_nu[l_idx_local])


@dataclass
class PeakDef:
    center: float  # nu_k
    width: float  # sigma_k

    @classmethod
    def from_distribution(cls, nu: Arr1DF, intensity: Arr1DF, peak_point: float):
        idx_peak = find_nearest(nu, peak_point)
        width = get_fwhm(nu, intensity, idx_peak)
        return cls(center=peak_point, width=width)

    def gauss(self, nu: Arr1DF) -> Arr1DF:
        return np.exp(-0.5 * ((nu - self.center) / self.width) ** 2)


@dataclass
class DistributionPeak:
    mean: float
    std: float

    def sample(self) -> float:
        return float(np.random.normal(self.mean, self.std))


@dataclass
class PolyCoeffStats:
    mean: Arr1DF  # shape (3,), (p0, p1, p2)
    cov: Arr2DF  # shape (3, 3)

    def sample(self) -> Arr1DF:
        return np.random.multivariate_normal(self.mean, self.cov)

    @classmethod
    def fit(cls, nus: list[Arr1DF], bases: list[Arr1DF]):
        """
        Fit a 2nd-order polynomial to each (nu, baseline) pair and
        return multivariate Gaussian stats for [p0, p1, p2].
        """
        coeffs: list[ThirdCoeff] = []

        for nu, b in zip(nus, bases):
            p2, p1, p0 = np.polyfit(nu, b, deg=2)
            coeffs.append((p0, p1, p2))

        coeff_arr = np.asarray(coeffs, dtype=np.float64)  # shape (N, 3)

        mean = coeff_arr.mean(axis=0)  # (3,)
        if coeff_arr.shape[0] < 2:
            cov = np.zeros((3, 3), dtype=np.float64)
        else:
            cov = np.cov(coeff_arr.T, ddof=1, dtype=np.float64)  # (3,3)

        return cls(mean=mean, cov=cov)


@dataclass
class NoiseStats:
    mean: float
    std: float

    def sample(self, shape: tuple[int, ...]) -> Arr1DF:
        return np.random.normal(self.mean, self.std, size=shape)

    @classmethod
    def from_intensities(cls, intensities: list[Arr1DF]):
        """
        Estimate noise mean and std across a list of 1D spectra.
        """
        all_residuals: list[np.ndarray] = []

        for i in intensities:
            n_points = i.shape[0]
            if n_points <= 3:
                I_smooth = i
            else:
                window = min(11, n_points if n_points % 2 == 1 else n_points - 1)
                if window <= 3:
                    I_smooth = i
                else:
                    I_smooth = savgol_filter(i, window_length=window, polyorder=3)
            res = i - I_smooth
            all_residuals.append(res)

        residuals = np.concatenate(all_residuals)
        return cls(residuals.mean(), residuals.std())


class AmpDistPeak(TypedDict):
    """
    Amplitude distributions derived from intensities.
    """

    samples: list[DistributionPeak]
    negatives: list[DistributionPeak]


def fit_amp_dists(
    peak_points: tuple[float, ...],
    wls: Arr1DF,
    i_sample: list[Arr1DF],
    i_base: list[Arr1DF],
) -> AmpDistPeak:
    amp_neg: list[DistributionPeak] = []
    amp_pos: list[DistributionPeak] = []

    for peak in peak_points:
        peak_index = find_nearest(wls, peak)

        val_pos = np.asarray([i[peak_index] for i in i_sample])
        val_neg = np.asarray([j[peak_index] for j in i_base])

        dist_peak_neg = DistributionPeak(mean=val_neg.mean(), std=val_neg.std())
        dist_peak_pos = DistributionPeak(mean=val_pos.mean(), std=val_pos.std())
        amp_neg.append(dist_peak_neg)
        amp_pos.append(dist_peak_pos)

    return {"samples": amp_pos, "negatives": amp_neg}


@dataclass
class ClassSimStats:
    peaks: list[PeakDef]
    amps: list[DistributionPeak]  # one per peak

    @classmethod
    def fit(
        cls,
        nu: Arr1DF,
        class_spectra: list[Arr1DF],
        peak_points: tuple[float, ...],
    ):
        arr = np.asarray(class_spectra)  # (N_class, C)
        avg = arr.mean(axis=0)  # (C,)

        peaks: list[PeakDef] = []
        amps: list[DistributionPeak] = []

        for peak_point in peak_points:
            # Peak shape from average Arr1DF
            peak_def = PeakDef.from_distribution(nu, avg, peak_point)
            peaks.append(peak_def)

            # Amplitude distribution from per-pixel variation
            idx = find_nearest(nu, peak_def.center)
            # intensities at that band across class pixels
            vals = arr[:, idx]
            amps.append(DistributionPeak(mean=vals.mean(), std=vals.std()))

        return cls(peaks=peaks, amps=amps)


def simulate_Arr1DF_single(
    nu: Arr1DF,
    cls_stats: ClassSimStats,
    poly_stats: PolyCoeffStats,
    noise_stats: NoiseStats,
) -> Arr1DF:
    I_clean = np.zeros_like(nu)

    for peak_def, amp_dist in zip(cls_stats.peaks, cls_stats.amps):
        A_k = amp_dist.sample()
        I_clean += A_k * peak_def.gauss(nu)

    # Baseline
    p0, p1, p2 = poly_stats.sample()
    baseline = p0 + p1 * nu + p2 * (nu**2)
    I_clean += baseline

    # Noise
    noise = noise_stats.sample(nu.shape)
    return I_clean + noise


def simulate_map_from_labels(
    labels: np.ndarray[tuple[int, int]],  # (H, W), 0/1/2
    nu: Arr1DF,  # (C,)
    stats_by_class: dict[str, ClassSimStats],
    poly_stats: PolyCoeffStats,
    noise_stats: NoiseStats,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
    H, W = labels.shape
    C = nu.shape[0]
    sim_cube = np.zeros((H, W, C), dtype=np.float32)

    id_to_name = {0: "negatives", 1: "maybe", 2: "samples"}

    for i in range(H):
        for j in range(W):
            cls_name = id_to_name[int(labels[i, j])]
            cls_stats = stats_by_class[cls_name]
            sim_cube[i, j, :] = simulate_Arr1DF_single(
                nu, cls_stats, poly_stats, noise_stats
            )

    return sim_cube


def simulate_from_single_hsi(
    cube: Cube,  # (H, W, C)
    nu: Arr1DF,  # (C,)
    labels: np.ndarray,  # (H, W) with 0/1/2
    peak_points: tuple[float, ...],  # e.g. (1003.0, 1410.0, 1600.0)
) -> Cube:
    # Split spectra by class
    spectra_by_class = cube.split_cube_by_label(labels)

    # Fit per-class spectral stats
    stats_by_class: dict[str, ClassSimStats] = {}
    specs: list[Arr1DF]
    for cls_name, specs in spectra_by_class.items():  # pyright: ignore[reportAssignmentType]
        if len(specs) == 0:
            raise RuntimeError(f"No pixels for class {cls_name}")
        stats_by_class[cls_name] = ClassSimStats.fit(nu, specs, peak_points)

    # Fit global baseline & noise
    flat = cube.flatten()
    pixel_spec = flat.get_pixels()
    nus = [nu] * flat.N
    # WARN: without explicit baselines, treating full spectra as “baseline + peaks”
    # TODO: use baseline estimate from pre-processing here instead
    poly_stats = PolyCoeffStats.fit(nus, bases=pixel_spec)
    noise_stats = NoiseStats.from_intensities(pixel_spec)

    # Simulate new cube with same label layout
    sim_cube = simulate_map_from_labels(
        labels=labels,
        nu=nu,
        stats_by_class=stats_by_class,
        poly_stats=poly_stats,
        noise_stats=noise_stats,
    )
    return Cube.make(sim_cube)
