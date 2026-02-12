import numpy as np
import pytest

from pyspectral.data.io import Presence, build_artifacts
from pyspectral.data.preprocessing import PreConfig
from pyspectral.modeling.oof import PxlStats


@pytest.mark.unit
def test_pxlstats_init_handles_mismatched_presence_shapes() -> None:
    pre_art = {
        "hw": [(2, 3), (3, 2)],
        "wls": [np.array([1000.0], dtype=np.float64), np.array([1000.0])],
        "pre_stats": [None, None],
        "flats": [
            np.zeros((6, 1), dtype=np.float64),
            np.zeros((6, 1), dtype=np.float64),
        ],
        "presences": [
            Presence(np.float64(1.0), np.ones((2, 2), dtype=np.float64)),
            Presence(np.float64(0.0), np.zeros((4, 4), dtype=np.float64)),
        ],
    }
    arts = build_artifacts(
        pre_config=PreConfig.make_min(),
        pre_art=pre_art,
        scene_ids=[0, 1],
    )

    pxl_stats = PxlStats(arts)

    assert pxl_stats.stats.true.shape == (12, 1)
    assert np.allclose(pxl_stats.stats.true[:6], 1.0)
    assert np.allclose(pxl_stats.stats.true[6:], 0.0)

    assert arts.presences[0].map.shape == (2, 3)
    assert arts.presences[1].map.shape == (3, 2)
