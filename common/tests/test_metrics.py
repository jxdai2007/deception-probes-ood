import numpy as np
import pytest
from jrp_common.metrics import auroc, catch_rate_at_fpr

def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0

def test_auroc_random_is_half():
    scores = np.array([0.1, 0.9, 0.1, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(auroc(scores, labels) - 0.5) < 1e-9

def test_catch_rate_at_fpr_perfect():
    # positives all score above negatives -> at 1% FPR we still catch all positives
    scores = np.array([0.0, 0.1, 0.2, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert catch_rate_at_fpr(scores, labels, fpr=0.01) == 1.0

def test_catch_rate_at_fpr_overlap():
    scores = np.array([0.0, 0.5, 0.5, 0.9])
    labels = np.array([0, 0, 1, 1])
    # threshold must keep FPR<=0; only score>0.5 (the 0.9) clears -> TPR=0.5
    assert catch_rate_at_fpr(scores, labels, fpr=0.0) == 0.5

def test_catch_rate_keeps_collinear_operating_points():
    # drop_intermediate=False is required to preserve collinear ROC points that are real operating points.
    scores = np.array([0.9, 0.9, 0.6, 0.6, 0.3, 0.3])
    labels = np.array([1, 0, 1, 0, 1, 0])
    assert catch_rate_at_fpr(scores, labels, fpr=0.67) == pytest.approx(2/3)

def test_catch_rate_raises_on_all_positive():
    scores = np.array([0.1, 0.9])
    labels = np.array([1, 1])
    with pytest.raises(ValueError, match="labels must contain both 0 and 1"):
        catch_rate_at_fpr(scores, labels, fpr=0.01)

def test_catch_rate_raises_on_all_negative():
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 0])
    with pytest.raises(ValueError, match="labels must contain both 0 and 1"):
        catch_rate_at_fpr(scores, labels, fpr=0.01)

def test_catch_rate_accepts_list_inputs():
    # Regression: function must accept plain-list inputs, not just ndarrays.
    scores_list = [0.1, 0.9, 0.2, 0.8]
    labels_list = [0, 0, 1, 1]
    scores_array = np.array(scores_list)
    labels_array = np.array(labels_list)
    result_from_list = catch_rate_at_fpr(scores_list, labels_list, fpr=0.01)
    result_from_array = catch_rate_at_fpr(scores_array, labels_array, fpr=0.01)
    assert result_from_list == result_from_array
