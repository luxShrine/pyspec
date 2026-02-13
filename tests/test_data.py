import numpy as np

from pyspectral.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
import pyspectral.data.io as pdi

# makes this test agnostic to data names, just grab txt files and a csv
raw_paths = RAW_DATA_DIR.glob("*.txt")
raw_first = next(raw_paths)
prc_paths = PROCESSED_DATA_DIR.glob("*.txt")
prc_first = next(prc_paths)
csv_path = next(DATA_DIR.glob("*.csv"))

raw_p = pdi.Presence.create_map(8, 8, 1)
prc_p = pdi.Presence.create_map(8, 8, 1)
raw_p = pdi.Presence(np.float64(1), raw_p)
prc_p = pdi.Presence(np.float64(1), prc_p)

hsi_raw = pdi.HSIMap.from_txt(
    raw_first, acq_time_s=10.0, accumulation=2, presence=raw_p
)
hsi_prc = pdi.HSIMap.from_txt(
    prc_first, acq_time_s=10.0, accumulation=2, presence=prc_p
)

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
