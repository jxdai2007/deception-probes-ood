"""Bootstrap SE for AUROC (plan-pinned: 1000 resamples, seed 42)."""

import numpy as np

from jrp_common.metrics import auroc


def bootstrap_auroc_se(scores, y, n_boot=1000, seed=42):
    """Stratified bootstrap SE of AUROC over dialogues."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        i = np.concatenate(
            [rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        )
        vals.append(auroc(scores[i], y[i]))
    return float(np.std(vals))
