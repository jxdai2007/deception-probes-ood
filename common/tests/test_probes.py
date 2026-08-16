import numpy as np
from jrp_common.probes import fit_probe, eval_probe, layer_sweep

def _sep_data(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0]*(n//2) + [1]*(n//2))
    X = rng.normal(size=(n, d))
    X[y == 1, 0] += 3.0  # class 1 separable along dim 0
    return X, y

def test_fit_eval_separable():
    X, y = _sep_data()
    probe = fit_probe(X, y)
    out = eval_probe(probe, X, y)
    assert out["auroc"] > 0.95
    assert 0.0 <= out["catch_at_1pct"] <= 1.0

def test_layer_sweep_picks_separable_layer():
    Xs, y = _sep_data(seed=1)
    Xn = np.random.default_rng(2).normal(size=Xs.shape)  # noise layer
    train = {5: (Xs, y), 9: (Xn, y)}
    val = {5: (Xs, y), 9: (Xn, y)}
    res = layer_sweep(train, val)
    assert res["best_layer"] == 5
    assert set(res["per_layer"]) == {5, 9}

def test_fit_probe_centering_learned_from_train():
    # Structural: the pipeline's centering step must learn its mean from the training
    # data passed to fit_probe, not re-derive it from whatever array is passed later.
    X_train, y = _sep_data()
    probe = fit_probe(X_train, y)
    assert np.allclose(probe.named_steps["center"].mean_, X_train.mean(axis=0))

def test_eval_probe_uses_train_fitted_centering_not_eval_set_mean():
    # Pins: centering is learned once at fit time and reused at eval time, so a
    # validation set offset by a large constant scores differently than the training
    # set. Under per-call re-centering (the brief's original _normalize), the offset
    # would be silently subtracted away and the two AUROCs would come out identical.
    X_train, y = _sep_data()
    probe = fit_probe(X_train, y)
    X_val = X_train + 50.0
    out_train = eval_probe(probe, X_train, y)
    out_val = eval_probe(probe, X_val, y)
    assert out_train["auroc"] != out_val["auroc"]
