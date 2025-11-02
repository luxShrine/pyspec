import numpy as np
import numpy.typing as npt

from pyspectral.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
import pyspectral.dataset as pyd
import pyspectral.features as pyf
import pyspectral.plots as plot

# makes this test agnostic to data names, just grab txt files and a csv
raw_paths = RAW_DATA_DIR.glob("*.txt")
raw_first = next(raw_paths)
prc_paths = PROCESSED_DATA_DIR.glob("*.txt")
prc_first = next(prc_paths)
csv_path = next(DATA_DIR.glob("*.csv"))

hsi_raw = pyf.HSIMap.load(raw_first, acq_time_s=10.0, accumulation=2)
hsi_prc = pyf.HSIMap.load(prc_first, acq_time_s=10.0, accumulation=2)

wl, xy, spectra, cube = (hsi_raw.wl, hsi_raw.xy, hsi_raw.spectra, hsi_raw.cube)
wl_gt, xy_gt, spectra_gt, cube_gt = (
    hsi_prc.wl,
    hsi_prc.xy,
    hsi_prc.spectra,
    hsi_prc.cube,
)

# TODO: grab this value instead of hardcoding it
M = 950


def test_loading_hsi_map():
    print(wl.shape, xy.shape, spectra.shape, cube.shape)
    print(wl_gt.shape, xy_gt.shape, spectra_gt.shape, cube_gt.shape)
    # expect (M,), (64,2), (64,M), (8,8,M)
    assert wl.shape == wl_gt.shape == (M,)
    assert xy.shape == xy_gt.shape == (64, 2)
    assert spectra.shape == spectra_gt.shape == (64, M)
    assert cube.shape == cube_gt.shape == (8, 8, M)


def test_spectral_pair():
    from sklearn.model_selection import KFold

    pair = pyd.SpectraPair(
        spectra.get().astype(np.float64), spectra_gt.get().astype(np.float64)
    )
    ncomp = max(2, min(pair.X_raw.shape[0] // 2, 32))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = pair.pcr_predict(cv)

    assert isinstance(pred, npt.ArrayLike)


def test_plot_boundary():
    num_center: float = 1654.0  # example Raman peak (adjust per sample)
    den_center: float = 1446.0  # example Raman peak (adjust per sample)
    ratio_map = plot.ratio_map(cube, wl, num_center, den_center)

    otsu_bound = plot.Boundary.create_otsu_mask(ratio_map)
    hys_bound = plot.Boundary.create_hysteresis_mask(ratio_map)
    assert isinstance(otsu_bound.boundary, npt.ArrayLike)
    assert isinstance(hys_bound.boundary, npt.ArrayLike)


def test_oof_stats():
    from pyspectral.modeling.train import OOFStats

    pair, arts = pyd.SpectraPair.from_annotations(csv_path, DATA_DIR)
    _raw = pair.X_raw.astype(np.float32)  # (N,C)
    prc = pair.Y_proc.astype(np.float32)  # (N,C)
    oof_stats = OOFStats(prc, arts)
    assert isinstance(oof_stats, OOFStats)


# TODO:
# def test_plot_window():
#     assert False
