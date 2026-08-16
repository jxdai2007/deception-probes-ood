"""Synthetic-activation tests for Tasks 2-5 (IMPLEMENTATION.md micro-steps)."""

import numpy as np
import pytest

from p04.gate import check_gate
from p04.transfer_models import released_scores
from p04.stats import bootstrap_auroc_se
from jrp_common.probes import fit_probe
from jrp_common.metrics import auroc


def make_synth(n=200, d=64, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    y = np.repeat([0, 1], n // 2).astype(np.int8)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[y == 1] += sep * w
    return X, y, w


def test_check_gate_range():
    assert check_gate(0.97, 0.96, 0.999) is True
    assert check_gate(0.95, 0.96, 0.999) is True  # within -0.02 tolerance
    assert check_gate(0.70, 0.96, 0.999) is False


def test_released_scores_known_auroc():
    """Fixed linear probe on synthetic separated data -> near-1 AUROC."""
    X, y, w = make_synth()
    probe = {
        "directions": w[None, :].astype(np.float32),
        "normalize": False,
        "scaler_mean": None,
        "scaler_scale": None,
    }
    s = released_scores(probe, X)
    assert auroc(s, y) > 0.95


def test_released_scores_normalize_changes_scores():
    X, y, w = make_synth()
    raw = {"directions": w[None, :].astype(np.float32), "normalize": False,
           "scaler_mean": None, "scaler_scale": None}
    norm = {"directions": w[None, :].astype(np.float32), "normalize": True,
            "scaler_mean": np.full((1, X.shape[1]), 0.5, np.float32),
            "scaler_scale": np.full((1, X.shape[1]), 2.0, np.float32)}
    assert not np.allclose(released_scores(raw, X), released_scores(norm, X))


def test_coordinate_restriction_handles_dim_mismatch():
    X, y, w = make_synth(d=64)
    probe = {"directions": np.concatenate([w, np.zeros(36)])[None, :],
             "normalize": False, "scaler_mean": None, "scaler_scale": None}
    s = released_scores(probe, X)  # probe dim 100 > act dim 64
    assert s.shape == (len(y),)
    assert auroc(s, y) > 0.95


def test_same_dataset_beats_cross_when_signal_differs():
    """Fit on signal-A data, eval on signal-B data -> lower than same-dist."""
    Xa, ya, _ = make_synth(seed=1)
    Xb, yb, _ = make_synth(seed=99)  # different direction
    probe = fit_probe(Xa, ya)
    same = auroc(probe.decision_function(Xa), ya)
    cross = auroc(probe.decision_function(Xb), yb)
    assert same > 0.95
    assert cross < same


def test_bootstrap_se_positive_and_small():
    X, y, w = make_synth()
    s = released_scores(
        {"directions": w[None, :], "normalize": False,
         "scaler_mean": None, "scaler_scale": None}, X)
    se = bootstrap_auroc_se(s, y, n_boot=200)
    assert 0.0 < se < 0.1
