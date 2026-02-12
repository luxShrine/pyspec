import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyspectral.result.compare import _ensure_pipeline_feature_compatibility
from pyspectral.result.predict import MaskedValues


def test_positive_negative_mask_thresholds_predictions() -> None:
    true = np.array([2.0, 0.0, 2.0, 0.0, 1.0], dtype=np.float32)
    pred = np.array([0.9, 0.1, 0.2, 0.8, 0.4], dtype=np.float32)

    mask_pos, mask_neg, mask_maybe = MaskedValues.get_mask(true)
    true_masked = MaskedValues.build_with_mask(true, mask_pos, mask_neg, mask_maybe)
    pred_masked = MaskedValues.build_with_mask(pred, mask_pos, mask_neg, mask_maybe)

    y_true = true_masked.get_positive_negative_mask()
    y_pred = pred_masked.get_positive_negative_mask(threshold=0.5)

    assert np.array_equal(y_true, np.array([1, 1, 0, 0], dtype=np.int8))
    assert np.array_equal(y_pred, np.array([1, 0, 0, 1], dtype=np.int8))

    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [1, 1]]


@pytest.mark.unit
def test_ensure_pipeline_feature_compatibility_resamples_with_wavelengths() -> None:
    pipeline = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True))])
    x_train = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    pipeline.fit(x_train)

    pixels = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    src_wl = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    dst_wl = np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64)

    aligned = _ensure_pipeline_feature_compatibility(
        pixels, pipeline, source_wl=src_wl, target_wl=dst_wl
    )

    assert aligned.shape == (2, 5)
    assert np.allclose(aligned[0], np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert np.allclose(aligned[1], np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
