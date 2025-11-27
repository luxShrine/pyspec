import numpy as np
import pytest

from pyspectral.config import RNG
import pyspectral.core as pc
import pyspectral.result.plots as plot


@pytest.mark.slow
def test_plot_boundary():
    NUM_CENTER: float = 1654.0  # example Raman peak
    DEN_CENTER: float = 1446.0
    HALFWIDTH: float = 12.0
    SMOOTH_PX: int = 1

    # std = np.abs(np.diff([NUM_CENTER, DEN_CENTER]))
    low = int(DEN_CENTER - (HALFWIDTH * 2))
    high = int(NUM_CENTER + (HALFWIDTH * 2))

    arr = (RNG.random() * np.ones((2, 2, 50))).astype(np.float32)
    cube = pc.Cube.make(arr)
    wl = RNG.integers(low=low, high=high, size=arr.shape[-1])
    ratio_map = plot.ratio_map(cube, wl, NUM_CENTER, DEN_CENTER, HALFWIDTH, SMOOTH_PX)

    otsu_bound = plot.Boundary.create_otsu_mask(ratio_map)
    hys_bound = plot.Boundary.create_hysteresis_mask(ratio_map)
    assert isinstance(otsu_bound.boundary, np.ndarray)
    assert isinstance(hys_bound.boundary, np.ndarray)


# TODO:
# def test_plot_window():
#     assert False
