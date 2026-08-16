"""Task 4 — cross-dataset shift matrix, per model.

Fit a fresh probe (jrp_common.fit_probe: center + L2 + LR — OUR instrument,
identical everywhere) on one dataset's pooled activations, evaluate on
another, same model. Diagonal cells use the gate's 80/20 held-out split so
same-dataset numbers are honest; off-diagonal cells fit on all of the fit
dataset and evaluate on all of the eval dataset.
"""

import numpy as np

from jrp_common.probes import fit_probe

from p04.loader import load_pooled_acts
from p04.stats import bootstrap_auroc_se
from jrp_common.metrics import auroc


def _split(y, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    npv, nnv = max(1, len(pos) // 5), max(1, len(neg) // 5)
    val = np.concatenate([pos[:npv], neg[:nnv]])
    train = np.concatenate([pos[npv:], neg[nnv:]])
    return train, val


def fit_on_eval_on(model_short: str, fit_dataset: str, eval_dataset: str,
                   acts_dir=None) -> dict:
    Xf, yf, _ = load_pooled_acts(model_short, fit_dataset, acts_dir)
    kf = yf >= 0
    Xf, yf = Xf[kf], yf[kf]

    if fit_dataset == eval_dataset:
        train, val = _split(yf)
        probe = fit_probe(Xf[train], yf[train])
        Xe, ye = Xf[val], yf[val]
    else:
        probe = fit_probe(Xf, yf)
        Xe, ye, _ = load_pooled_acts(model_short, eval_dataset, acts_dir)
        ke = ye >= 0
        Xe, ye = Xe[ke], ye[ke]

    scores = probe.decision_function(Xe)
    return {
        "model": model_short,
        "fit_dataset": fit_dataset,
        "eval_dataset": eval_dataset,
        "auroc": float(auroc(scores, ye)),
        "auroc_se": bootstrap_auroc_se(scores, ye),
        "n_eval": int(len(ye)),
    }
