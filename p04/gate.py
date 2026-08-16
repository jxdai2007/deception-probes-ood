"""Task 2 — replication gate: same-distribution AUROC on a model we can run.

70B is impractical on this cluster; per IMPLEMENTATION.md we gate on
Llama-3.1-8B: fit a fresh linear probe on roleplaying activations (train
split), score held-out same-distribution data, and require the AUROC to land
in the paper's cited range (with the documented -0.02 tolerance on the low
end). Paper's Table AUROCs for the residual-stream LR probes span 0.96-0.999
on same-distribution eval; Gate A re-derived those exactly from shipped
scores.

If this gate fails: STOP all transfer work and debug extraction/normalization.
"""

import numpy as np

from jrp_common.probes import fit_probe, eval_probe

from p04.loader import load_pooled_acts

GATE_MODEL = "Meta-Llama-3.1-8B-Instruct"
GATE_DATASET = "roleplaying"
PUBLISHED_LOW, PUBLISHED_HIGH = 0.96, 0.999


def check_gate(auroc: float, published_low: float, published_high: float) -> bool:
    return auroc >= published_low - 0.02


def reproduce_auroc_per_token(model_short: str = GATE_MODEL,
                              dataset: str = GATE_DATASET,
                              acts_dir=None, seed: int = 42) -> float:
    """Apollo-faithful protocol: fit on TOKENS (dialogue label broadcast),
    score val tokens, mean per dialogue, AUROC over dialogues. Split is at
    dialogue level so no dialogue leaks between train and val."""
    from pathlib import Path
    from p04.loader import ACTS_DIR
    from jrp_common.metrics import auroc as _auroc

    d = Path(acts_dir) if acts_dir else ACTS_DIR
    stem = f"{model_short}__{dataset}"
    acts = np.load(d / f"{stem}__acts.npy").astype(np.float32)
    lengths = np.load(d / f"{stem}__lengths.npy")
    labels = np.load(d / f"{stem}__labels.npy")

    spans, y = [], []
    off = 0
    for ln, lab in zip(lengths, labels):
        if ln > 0 and lab >= 0:
            spans.append((off, off + int(ln)))
            y.append(int(lab))
        off += int(ln)
    y = np.array(y, np.int8)

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    npv, nnv = max(1, len(pos) // 5), max(1, len(neg) // 5)
    val = np.concatenate([pos[:npv], neg[:nnv]])
    train = np.concatenate([pos[npv:], neg[nnv:]])

    Xtr = np.concatenate([acts[spans[i][0]:spans[i][1]] for i in train])
    ytr = np.concatenate([np.full(spans[i][1] - spans[i][0], y[i]) for i in train])
    probe = fit_probe(Xtr, ytr)

    dial_scores = np.array([
        probe.decision_function(acts[spans[i][0]:spans[i][1]]).mean() for i in val
    ])
    return float(_auroc(dial_scores, y[val]))


def reproduce_auroc_cv(model_short: str = GATE_MODEL,
                       dataset: str = GATE_DATASET,
                       acts_dir=None, seed: int = 42, k: int = 5) -> dict:
    """Gate protocol: 5-fold CV at dialogue level, per-token fit (Apollo's
    protocol), every dialogue scored exactly once out-of-fold, single AUROC
    over all held-out dialogue scores. Replaces the single 80/20 split, whose
    ~17-dialogue honest arm made the gate seed-dominated (0.88-0.97 across
    seeds); documented in the gate record."""
    from pathlib import Path
    from p04.loader import ACTS_DIR
    from jrp_common.metrics import auroc as _auroc

    d = Path(acts_dir) if acts_dir else ACTS_DIR
    stem = f"{model_short}__{dataset}"
    acts = np.load(d / f"{stem}__acts.npy").astype(np.float32)
    lengths = np.load(d / f"{stem}__lengths.npy")
    labels = np.load(d / f"{stem}__labels.npy")

    spans, y = [], []
    off = 0
    for ln, lab in zip(lengths, labels):
        if ln > 0 and lab >= 0:
            spans.append((off, off + int(ln)))
            y.append(int(lab))
        off += int(ln)
    y = np.array(y, np.int8)

    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), np.int32)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k

    oof = np.zeros(len(y), np.float64)
    for f in range(k):
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        Xtr = np.concatenate([acts[spans[i][0]:spans[i][1]] for i in tr])
        ytr = np.concatenate(
            [np.full(spans[i][1] - spans[i][0], y[i]) for i in tr])
        probe = fit_probe(Xtr, ytr)
        for i in va:
            oof[i] = probe.decision_function(
                acts[spans[i][0]:spans[i][1]]).mean()

    return {"auroc": float(_auroc(oof, y)), "n": int(len(y)),
            "n_honest": int((y == 0).sum()),
            "n_deceptive": int((y == 1).sum()), "k_folds": k}


def reproduce_auroc(model_short: str = GATE_MODEL, dataset: str = GATE_DATASET,
                    acts_dir=None, seed: int = 42) -> float:
    X, y, _ = load_pooled_acts(model_short, dataset, acts_dir)
    keep = y >= 0
    X, y = X[keep], y[keep]

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val, n_neg_val = max(1, len(pos) // 5), max(1, len(neg) // 5)
    val = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]])
    train = np.concatenate([pos[n_pos_val:], neg[n_neg_val:]])

    probe = fit_probe(X[train], y[train])
    return float(eval_probe(probe, X[val], y[val])["auroc"])
