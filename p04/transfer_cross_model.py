"""Task 5 — cross-model transfer at matched layer fraction.

All activations were extracted at layer_frac 0.275, so layer matching is
inherent. Every model pair here has a different hidden width, so transfer
uses coordinate restriction to min(d_a, d_b) — the same explicitly-naive rule
as Task 3, recorded per cell (IMPLEMENTATION's cross-dim guard). Fit on model
A's restricted activations, score model B's; diagonal (A==B) uses the held-out
split as the within-model control.
"""

import numpy as np

from jrp_common.probes import fit_probe
from jrp_common.metrics import auroc

from p04.loader import load_pooled_acts
from p04.stats import bootstrap_auroc_se
from p04.transfer_datasets import _split


def fit_a_eval_b(model_a: str, model_b: str, dataset: str,
                 acts_dir=None) -> dict:
    Xa, ya, _ = load_pooled_acts(model_a, dataset, acts_dir)
    ka = ya >= 0
    Xa, ya = Xa[ka], ya[ka]

    if model_a == model_b:
        train, val = _split(ya)
        d = Xa.shape[1]
        probe = fit_probe(Xa[train], ya[train])
        scores = probe.decision_function(Xa[val])
        ye = ya[val]
    else:
        Xb, yb, _ = load_pooled_acts(model_b, dataset, acts_dir)
        kb = yb >= 0
        Xb, yb = Xb[kb], yb[kb]
        d = min(Xa.shape[1], Xb.shape[1])
        probe = fit_probe(Xa[:, :d], ya)
        scores = probe.decision_function(Xb[:, :d])
        ye = yb

    return {
        "fit_model": model_a,
        "eval_model": model_b,
        "dataset": dataset,
        "auroc": float(auroc(scores, ye)),
        "auroc_se": bootstrap_auroc_se(scores, ye),
        "dims_used": int(d),
        "coordinate_restriction": model_a != model_b,
        "n_eval": int(len(ye)),
    }
