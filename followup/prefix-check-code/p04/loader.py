"""Probe and dataset loaders for the Apollo deception-detection extension.

Probes are read from the converted .npy/.json artifacts produced by
p04/convert_detector.py — never from the raw detector.pt pickles. The one
audited pickle execution lives in convert_detector.py.
"""

import json
from pathlib import Path

import numpy as np

from p04.pooling import parse_pooling, pool_response_rows

PROJECT_ROOT = Path(__file__).parent.parent
PROBES_DIR = PROJECT_ROOT / "results" / "probes"
ROLLOUTS_DIR = (
    PROJECT_ROOT / "vendor" / "deception-detection" / "data" / "rollouts"
)

# Transfer datasets. insider_trading substitution: the pinned repo's
# insider_trading__upscale__llama-70b-3.3.json is a stub ("SEE OTHER FILE");
# the real graded rollouts ship as insider_trading__onpolicy__llama-70b.json.
# Logged in docs/plans/extension-sprint-questions.md.
DATASET_FILES = {
    "roleplaying": "roleplaying__plain__llama-70b-3.3.json",
    "insider_trading": "insider_trading__onpolicy__llama-70b.json",
    "sandbagging": "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
}

# Label integers used across the pipeline (AUROC uses 0/1 only; -1 rows are
# excluded from every metric — the Gate A lesson: never fold AMBIGUOUS/SKIP
# into the honest arm).
HONEST, DECEPTIVE, EXCLUDED = 0, 1, -1


def layer_frac_to_index(depth: int, frac: float) -> int:
    """Convert layer fraction to absolute layer index.

    Matches the paper's 22/80 on the 70B at frac=0.275.

    Raises:
        ValueError: If frac not in [0, 1]
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("layer_frac must be between 0 and 1")
    if frac == 1.0:
        return depth - 1
    return int(frac * depth)


def load_released_probe(name: str) -> dict:
    """Load a converted released probe.

    Returns dict with keys: layers (list[int]), directions [layer, hidden],
    normalize (bool), scaler_mean/scaler_scale ([layer, hidden] or None),
    manifest (the full manifest dict).

    Raises:
        FileNotFoundError: If the converted manifest is missing.
        ValueError: If the probe is excluded from transfer (sae_rp).
    """
    manifest_path = PROBES_DIR / f"{name}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Converted probe manifest not found: {manifest_path} "
            "(run p04/convert_detector.py first)"
        )
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("excluded_from_transfer"):
        raise ValueError(
            f"Probe {name!r} is excluded from transfer: "
            f"{manifest.get('exclusion_reason')}"
        )

    def artifact_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    out = {
        "layers": manifest["layers"],
        "directions": np.load(
            artifact_path(manifest["outputs"]["directions_npy"])
        ),
        "normalize": manifest.get("normalize", False),
        "scaler_mean": None,
        "scaler_scale": None,
        "manifest": manifest,
    }
    if "scaler_mean_npy" in manifest["outputs"]:
        out["scaler_mean"] = np.load(
            artifact_path(manifest["outputs"]["scaler_mean_npy"])
        )
        out["scaler_scale"] = np.load(
            artifact_path(manifest["outputs"]["scaler_scale_npy"])
        )
    if out["normalize"]:
        assert out["scaler_mean"] is not None, (
            f"{name}: normalize=True but scaler arrays missing — reconvert"
        )
    return out


ACTS_DIR = PROJECT_ROOT / "results" / "activations"


def _validate_pooled_artifacts(acts, lengths, labels, meta, stem):
    if acts.ndim != 2 or not np.issubdtype(acts.dtype, np.number):
        raise ValueError(f"activations for {stem} must be a numeric 2D array")
    if not np.isfinite(acts).all():
        raise ValueError(f"non-finite activations for {stem}")
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(f"activation lengths for {stem} must be a 1D integer array")
    if np.any(lengths < 0):
        raise ValueError(f"negative activation length for {stem}")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels for {stem} must be a 1D integer array")
    if len(lengths) != len(labels):
        raise ValueError(f"row-count mismatch for {stem}")
    if int(lengths.sum()) != len(acts):
        raise ValueError(f"activation-length mismatch for {stem}")
    if not set(np.asarray(labels, dtype=int).tolist()).issubset({-1, 0, 1}):
        raise ValueError(f"unexpected label value for {stem}")
    if not isinstance(meta, dict):
        raise ValueError(f"metadata for {stem} must be a JSON object")
    if "n_dialogues" in meta and int(meta["n_dialogues"]) != len(labels):
        raise ValueError(f"metadata row-count mismatch for {stem}")
    if "hidden_size" in meta and int(meta["hidden_size"]) != acts.shape[1]:
        raise ValueError(f"metadata hidden-size mismatch for {stem}")


def load_pooled_acts(
    model_short: str,
    dataset: str,
    acts_dir=None,
    pooling="full_response",
):
    """Load cached activations, pooled per dialogue by a frozen rule.

    Returns (X [n, hidden] float32, y [n] int8, meta dict). Rows whose response
    was fully truncated away (length 0) are dropped and counted in
    meta["n_zero_length"]. The default remains full-response mean pooling;
    fixed final-token windows additionally record coverage over the binary
    probe-fit population in metadata.
    """
    d = Path(acts_dir) if acts_dir else ACTS_DIR
    stem = f"{model_short}__{dataset}"
    acts = np.load(d / f"{stem}__acts.npy")
    lengths = np.load(d / f"{stem}__lengths.npy")
    labels = np.load(d / f"{stem}__labels.npy")
    meta = json.loads((d / f"{stem}__meta.json").read_text())
    _validate_pooled_artifacts(acts, lengths, labels, meta, stem)
    mode, _ = parse_pooling(pooling)
    if mode == "answer_span":
        raise ValueError(
            "answer_span pooling is unavailable for legacy activation artifacts"
        )
    if mode == "full_response" and not np.any(lengths > 0):
        X = np.zeros((0, acts.shape[1]), dtype=np.float32)
        kept = np.array([], dtype=int)
        coverage = None
    else:
        X, kept, coverage = pool_response_rows(
            acts,
            lengths,
            pooling=pooling,
        )
        if coverage is not None:
            binary = (lengths > 0) & np.isin(labels, [0, 1])
            n_total = int(binary.sum())
            n_full = int((binary & (lengths >= coverage["k"])).sum())
            coverage = {
                "k": coverage["k"],
                "n_total": n_total,
                "n_full": n_full,
                "n_partial": n_total - n_full,
                "full_fraction": n_full / n_total if n_total else 0.0,
            }
    y = labels[kept]
    meta["n_zero_length"] = int((lengths == 0).sum())
    if coverage is not None:
        meta["pooling_coverage"] = coverage
    return X, y, meta


def _label_from_judgement(judgement, dataset: str) -> int:
    """Mirror the vendor's Label mapping (data/base.py + data/roleplaying.py).

    roleplaying grades are ints 1-7: >=6 DECEPTIVE, <=2 HONEST, else excluded.
    Other datasets use the string rule: honest/deceptive keep their label;
    unknown/ambiguous/refusal are excluded.
    """
    if dataset == "roleplaying":
        j = int(judgement)
        if j >= 6:
            return DECEPTIVE
        if j <= 2:
            return HONEST
        return EXCLUDED
    if isinstance(judgement, str):
        return {"honest": HONEST, "deceptive": DECEPTIVE}.get(
            judgement, EXCLUDED
        )
    return EXCLUDED


def make_example_id(
    dataset: str, rollout_index: int, completion_index: int
) -> str:
    """Return the stable source-row identity used by paired conditions."""
    return f"{dataset}:{rollout_index}:{completion_index}"


def load_rollouts(dataset: str) -> list[dict]:
    """Load graded rollout dialogues for one dataset.

    Returns a list of dicts: {example_id, input_messages, output_str, label} —
    one entry per completion, label per _label_from_judgement (EXCLUDED rows
    retained so downstream can count them; metrics must filter to
    HONEST/DECEPTIVE).
    """
    if dataset not in DATASET_FILES:
        raise KeyError(f"Unknown dataset {dataset!r}; know {list(DATASET_FILES)}")
    path = ROLLOUTS_DIR / DATASET_FILES[dataset]
    data = json.loads(path.read_text())
    rows = []
    for rollout_index, item in enumerate(data["rollouts"]):
        for completion_index, output_str in enumerate(item["output_strs"]):
            rows.append(
                {
                    "example_id": make_example_id(
                        dataset, rollout_index, completion_index
                    ),
                    "input_messages": item["input_messages"],
                    "output_str": output_str.rstrip(),
                    "label": _label_from_judgement(
                        item["graded_deception"][completion_index], dataset
                    ),
                }
            )
    return rows
