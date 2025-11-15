import numpy as np
import numpy.testing as npt
import pytest

from pyspectral.core import Cube, assert_same_grid
from pyspectral.data.preprocessing import (
    SmoothCfg,
    find_peak_in_window,
    normalize_to_peak,
)


def test_assert_same_grid_detects_mismatch() -> None:
    base = np.linspace(0.0, 1.0, 4)
    shifted = base + 1e-3
    not_err = assert_same_grid(base, base.copy())
    assert not_err is None

    expected_err = assert_same_grid(base, shifted, tolerance=1e-5)
    assert expected_err is not None
    assert "Grids differ" in expected_err


def test_resample_cube_interpolates_along_wavelength_axis() -> None:
    src_wl = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    dst_wl = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    cube = Cube.make(np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32))

    resampled = cube.resample_cube(src_wl, dst_wl)

    assert resampled.shape == (1, 1, dst_wl.size)
    npt.assert_allclose(
        resampled.get()[0, 0], np.interp(dst_wl, src_wl, cube.get()[0, 0])
    )


def test_normalize_to_peak_scales_each_row() -> None:
    wl = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=np.float64)
    spectra = np.array(
        [
            [1.0, 2.0, 8.0, 4.0],
            [2.0, 8.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_to_peak(spectra, wl, center=1010.0, halfwidth=5.0)
    mask = (wl >= 1005.0) & (wl <= 1015.0)
    denom = spectra[:, mask].max(axis=1, keepdims=True) + 1e-8

    npt.assert_allclose(normalized, spectra / denom)


def test_normalize_to_peak_raises_on_empty_window() -> None:
    wl = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=np.float64)
    spectra = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        normalize_to_peak(spectra, wl, center=4000.0, halfwidth=5.0)


def test_find_peak_in_window_prefers_prominent_peak() -> None:
    # align peak value in window to 1004.0
    wl = np.array([1000.0, 1002.0, 1004.0, 1006.0, 1008.0], dtype=np.float64)
    spectrum = np.array([0.1, 0.3, 1.5, 0.2, 0.1], dtype=np.float64)

    peak = find_peak_in_window(wl, spectrum, window=(1000.0, 1008.0), min_prom=0.1)

    assert 1003.5 < peak < 1004.5


def test_smoothcfg_validates_and_smooths() -> None:
    # check that invalid config raises error
    with pytest.raises(RuntimeError):
        SmoothCfg(window=5)
    with pytest.raises(RuntimeError):
        SmoothCfg(poly=0)

    cfg = SmoothCfg(window=7, poly=3)
    # create a 2 dimensional array to mimic spectra
    sample_spectra = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    spectra = np.tile(sample_spectra, (2, 1))
    smoothed = cfg.smooth(spectra)

    assert smoothed.shape == spectra.shape
    assert smoothed.dtype == np.float32
