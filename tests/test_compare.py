import numpy as np
from sklearn.metrics import confusion_matrix

from pyspectral.result.compare import _get_positive_negative_mask
from pyspectral.result.predict import MaskedValues


def test_positive_negative_mask_thresholds_predictions() -> None:
    true = np.array([2.0, 0.0, 2.0, 0.0, 1.0], dtype=np.float32)
    pred = np.array([0.9, 0.1, 0.2, 0.8, 0.4], dtype=np.float32)

    mask_pos, mask_neg, mask_maybe = MaskedValues.get_mask(true)
    true_masked = MaskedValues.build_with_mask(true, mask_pos, mask_neg, mask_maybe)
    pred_masked = MaskedValues.build_with_mask(pred, mask_pos, mask_neg, mask_maybe)

    y_true = _get_positive_negative_mask(true_masked)
    y_pred = _get_positive_negative_mask(pred_masked, threshold=0.5)

    assert np.array_equal(y_true, np.array([1, 1, 0, 0], dtype=np.int8))
    assert np.array_equal(y_pred, np.array([1, 0, 0, 1], dtype=np.int8))

    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [1, 1]]
