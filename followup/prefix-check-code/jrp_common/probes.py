import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from .metrics import auroc, catch_rate_at_fpr


def fit_probe(X: np.ndarray, y: np.ndarray) -> Pipeline:
    # Mean-center then unit-norm per row, then logistic regression. A Pipeline so the
    # centering mean is learned on X here and reused (not re-derived) at eval time.
    #
    # Returns a fitted sklearn Pipeline, not a bare classifier. The probe direction
    # (for steering work) lives at probe.named_steps["lr"].coef_, in the
    # centered/L2-normalized space produced by the "center" and "unitnorm" steps --
    # not in the original activation space.
    probe = Pipeline([
        ("center", StandardScaler(with_mean=True, with_std=False)),
        ("unitnorm", Normalizer(norm="l2")),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    probe.fit(X, y)
    return probe


def eval_probe(probe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    scores = probe.decision_function(X)
    return {"auroc": auroc(scores, y), "catch_at_1pct": catch_rate_at_fpr(scores, y, 0.01)}


def layer_sweep(layer_to_Xy: dict, val_layer_to_Xy: dict) -> dict:
    per_layer = {}
    for layer, (X, y) in layer_to_Xy.items():
        probe = fit_probe(X, y)
        Xv, yv = val_layer_to_Xy[layer]
        per_layer[layer] = eval_probe(probe, Xv, yv)["auroc"]
    best = max(per_layer, key=per_layer.get)
    return {"best_layer": best, "per_layer": per_layer}
