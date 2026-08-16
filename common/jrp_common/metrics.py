import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))

def catch_rate_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.01) -> float:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("labels must contain both 0 and 1")
    fprs, tprs, _ = roc_curve(labels, scores, drop_intermediate=False)  # preserves collinear operating points
    ok = fprs <= fpr
    return float(tprs[ok].max()) if ok.any() else 0.0
