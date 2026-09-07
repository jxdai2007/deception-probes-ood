"""Artifact naming, validation, pooling, and row metadata for conditions."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np

from p04.pooling import WINDOW_KS, parse_pooling, pool_response_rows
from p04.rendering import RENDERINGS


POOLINGS = (
    "full_response",
    "answer_span",
    *(f"final_{window}" for window in WINDOW_KS),
)
_ROW_FIELDS = {
    "example_id",
    "immutable_hash",
    "selected_answer",
    "label",
    "truncated",
    "answer_token_span",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ConditionBatch:
    """One validated and pooled condition artifact."""

    X: np.ndarray
    y: np.ndarray
    example_ids: np.ndarray
    immutable_hashes: np.ndarray
    selected_answers: np.ndarray
    exclusions: dict[str, int] = field(default_factory=dict)
    response_lengths: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    pooling_coverage: dict | None = None


def condition_stem(model_short: str, rendering: str) -> str:
    """Return a condition stem that cannot collide with legacy artifacts."""
    if rendering not in RENDERINGS:
        raise ValueError(f"unknown rendering {rendering!r}; expected {RENDERINGS}")
    return f"{model_short}__sandbagging__{rendering}"


def _validate_artifacts(acts, lengths, labels, rows, stem):
    if acts.ndim != 2:
        raise ValueError(f"activations for {stem} must be a 2D array")
    if not np.isfinite(acts).all():
        raise ValueError(f"non-finite activations for {stem}")
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(f"activation lengths for {stem} must be a 1D integer array")
    if np.any(lengths < 0):
        raise ValueError(f"negative activation length for {stem}")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels for {stem} must be a 1D integer array")
    if not isinstance(rows, list):
        raise ValueError(f"row manifest for {stem} must be a JSON list")
    if not (len(lengths) == len(labels) == len(rows)):
        raise ValueError(f"row-count mismatch for {stem}")
    if int(lengths.sum()) != len(acts):
        raise ValueError(f"activation-length mismatch for {stem}")
    if not set(np.asarray(labels, dtype=int).tolist()).issubset({-1, 0, 1}):
        raise ValueError(f"unexpected label value for {stem}")

    example_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError(f"malformed row manifest entry {index} for {stem}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"invalid example ID at row {index} for {stem}")
        example_ids.append(example_id)
        if not isinstance(row["immutable_hash"], str) or not _SHA256_RE.fullmatch(
            row["immutable_hash"]
        ):
            raise ValueError(f"invalid immutable hash for {example_id}")
        if row["selected_answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid selected answer for {example_id}")
        if type(row["label"]) is not int or row["label"] != int(labels[index]):
            raise ValueError(f"row/label mismatch for {example_id}")
        if type(row["truncated"]) is not bool:
            raise ValueError(f"invalid truncation flag for {example_id}")
        span = row["answer_token_span"]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
        ):
            raise ValueError(f"malformed answer span for {example_id}")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError(f"duplicate example IDs for {stem}")


def _parse_exclusion_count(meta, rows, stem):
    required = {
        "n_dialogues",
        "n_source_dialogues",
        "n_parse_excluded",
        "parse_exclusions",
    }
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError(f"parse-exclusion metadata missing for {stem}")
    records = meta["parse_exclusions"]
    if not isinstance(records, list):
        raise ValueError(f"parse-exclusion metadata invalid for {stem}")
    ids = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "example_id",
            "output_sha256",
            "reason",
        }:
            raise ValueError(f"parse-exclusion metadata invalid for {stem}")
        if not isinstance(record["example_id"], str) or not record["example_id"]:
            raise ValueError(f"parse-exclusion metadata invalid for {stem}")
        if not isinstance(record["output_sha256"], str) or not _SHA256_RE.fullmatch(
            record["output_sha256"]
        ):
            raise ValueError(f"parse-exclusion metadata invalid for {stem}")
        if record["reason"] != "invalid_selected_answer_tag":
            raise ValueError(f"parse-exclusion metadata invalid for {stem}")
        ids.append(record["example_id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"parse-exclusion metadata invalid for {stem}")
    retained_ids = {row["example_id"] for row in rows}
    overlap = sorted(retained_ids.intersection(ids))
    if overlap:
        raise ValueError(
            f"parse-excluded example is retained for {stem}: {overlap}"
        )
    values = (
        meta["n_dialogues"],
        meta["n_source_dialogues"],
        meta["n_parse_excluded"],
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"parse-exclusion metadata invalid for {stem}")
    if (
        meta["n_dialogues"] != len(rows)
        or meta["n_parse_excluded"] != len(records)
        or meta["n_source_dialogues"] != len(rows) + len(records)
    ):
        raise ValueError(f"parse-exclusion metadata inconsistent for {stem}")
    return len(records)


def load_condition_batch(
    model_short,
    rendering,
    pooling,
    acts_dir,
    drop_truncated=True,
):
    """Load, validate, filter, and pool one rendering artifact."""
    parse_pooling(pooling)
    root = Path(acts_dir)
    stem = condition_stem(model_short, rendering)
    acts = np.load(root / f"{stem}__acts.npy").astype(np.float32)
    lengths = np.load(root / f"{stem}__lengths.npy")
    labels = np.load(root / f"{stem}__labels.npy")
    rows = json.loads((root / f"{stem}__rows.json").read_text())
    meta = json.loads((root / f"{stem}__meta.json").read_text())
    _validate_artifacts(acts, lengths, labels, rows, stem)
    parse_invalid = _parse_exclusion_count(meta, rows, stem)

    exclusions = {
        "zero_length": 0,
        "truncated": 0,
        "excluded_label": 0,
        "parse_invalid": parse_invalid,
    }
    include = np.ones(len(rows), dtype=bool)
    for index, (length_value, row) in enumerate(zip(lengths, rows)):
        length = int(length_value)
        if length == 0:
            exclusions["zero_length"] += 1
            include[index] = False
            continue
        if row["truncated"] and drop_truncated:
            exclusions["truncated"] += 1
            include[index] = False
            continue
        if int(labels[index]) not in (0, 1):
            exclusions["excluded_label"] += 1
            include[index] = False
            continue
    answer_spans = np.asarray(
        [row["answer_token_span"] for row in rows], dtype=np.int64
    )
    try:
        X, kept, coverage = pool_response_rows(
            acts,
            lengths,
            pooling=pooling,
            answer_spans=answer_spans,
            include=include,
        )
    except ValueError as error:
        if str(error) == f"no valid rows for {pooling}":
            raise ValueError(f"no valid rows for {stem}/{pooling}") from error
        raise
    return ConditionBatch(
        X=X,
        y=labels[kept],
        example_ids=np.array([rows[index]["example_id"] for index in kept]),
        immutable_hashes=np.array(
            [rows[index]["immutable_hash"] for index in kept]
        ),
        selected_answers=np.array(
            [rows[index]["selected_answer"] for index in kept]
        ),
        exclusions=exclusions,
        response_lengths=np.asarray(lengths[kept], dtype=np.int32),
        pooling_coverage=coverage,
    )
