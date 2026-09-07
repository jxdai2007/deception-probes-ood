"""Aligned score records and frozen statistics for the pooling audit."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from p04.pooling import WINDOW_KS


@dataclass(frozen=True)
class ScoreSeries:
    """One score vector with immutable pairing and source-length metadata."""

    example_ids: np.ndarray
    immutable_hashes: np.ndarray
    y: np.ndarray
    scores: np.ndarray
    source_lengths: np.ndarray


def _validated_series(item):
    if not isinstance(item, ScoreSeries):
        raise TypeError("score series entries must be ScoreSeries instances")
    arrays = {
        "example_ids": np.asarray(item.example_ids),
        "immutable_hashes": np.asarray(item.immutable_hashes),
        "y": np.asarray(item.y),
        "scores": np.asarray(item.scores, dtype=float),
        "source_lengths": np.asarray(item.source_lengths),
    }
    lengths = {len(value) for value in arrays.values() if value.ndim == 1}
    if any(value.ndim != 1 for value in arrays.values()) or len(lengths) != 1:
        raise ValueError("score-series fields must be aligned 1D arrays")
    ids = arrays["example_ids"].tolist()
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("example IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example IDs within a score series")
    hashes = arrays["immutable_hashes"].tolist()
    if any(not isinstance(value, str) or not value for value in hashes):
        raise ValueError("immutable hashes must be non-empty strings")
    if not set(arrays["y"].tolist()).issubset({0, 1}):
        raise ValueError("score-series labels must be binary")
    if not np.isfinite(arrays["scores"]).all():
        raise ValueError("score series require finite scores")
    if (
        not np.issubdtype(arrays["source_lengths"].dtype, np.integer)
        or np.any(arrays["source_lengths"] < 0)
    ):
        raise ValueError("source lengths must be nonnegative integers")
    return ScoreSeries(**arrays)


def align_score_series(series):
    """Align score vectors on common IDs in first-series order."""
    if not series:
        raise ValueError("at least one score series is required")
    validated = [_validated_series(item) for item in series]
    common = set(validated[0].example_ids.tolist())
    for item in validated[1:]:
        common &= set(item.example_ids.tolist())
    ordered_ids = [
        value for value in validated[0].example_ids.tolist() if value in common
    ]
    if not ordered_ids:
        raise ValueError("score series have no common example IDs")

    aligned = []
    for item in validated:
        lookup = {
            value: index for index, value in enumerate(item.example_ids.tolist())
        }
        indices = np.asarray([lookup[value] for value in ordered_ids], dtype=int)
        aligned.append(
            ScoreSeries(
                example_ids=item.example_ids[indices],
                immutable_hashes=item.immutable_hashes[indices],
                y=item.y[indices],
                scores=item.scores[indices],
                source_lengths=item.source_lengths[indices],
            )
        )

    reference = aligned[0]
    for item in aligned[1:]:
        if not np.array_equal(item.y, reference.y):
            raise ValueError("label mismatch across score series")
        if not np.array_equal(item.immutable_hashes, reference.immutable_hashes):
            raise ValueError("immutable hash mismatch across score series")
    if set(reference.y.tolist()) != {0, 1}:
        raise ValueError("aligned score series must contain both binary classes")
    return aligned


def paired_bootstrap_auroc_contrasts(
    y,
    scores,
    contrasts,
    *,
    n_boot=10000,
    seed=42,
):
    """Estimate linear AUROC contrasts with one paired draw per replicate."""
    y = np.asarray(y)
    if y.ndim != 1 or set(y.tolist()) != {0, 1}:
        raise ValueError("contrasts require both binary classes")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("at least one score cell is required")
    cells = {name: np.asarray(values, dtype=float) for name, values in scores.items()}
    if any(
        values.shape != y.shape or not np.isfinite(values).all()
        for values in cells.values()
    ):
        raise ValueError("score cells must be finite and aligned")
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(contrasts, dict) or not contrasts:
        raise ValueError("at least one contrast is required")

    normalized = {}
    for contrast_name, coefficients in contrasts.items():
        if not isinstance(contrast_name, str) or not contrast_name:
            raise ValueError("contrast names must be non-empty strings")
        if (
            not isinstance(coefficients, dict)
            or not coefficients
            or not set(coefficients).issubset(cells)
        ):
            raise ValueError("contrast references an unknown score cell")
        normalized[contrast_name] = {}
        for cell, coefficient in coefficients.items():
            try:
                value = float(coefficient)
            except (TypeError, ValueError) as error:
                raise ValueError("contrast coefficients must be finite") from error
            if not np.isfinite(value):
                raise ValueError("contrast coefficients must be finite")
            normalized[contrast_name][cell] = value

    point_auc = {
        name: float(roc_auc_score(y, values)) for name, values in cells.items()
    }
    draws = {
        name: np.empty(n_boot, dtype=np.float64) for name in normalized
    }
    rng = np.random.default_rng(seed)
    by_label = [np.flatnonzero(y == label) for label in (0, 1)]
    for draw_index in range(n_boot):
        indices = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in by_label]
        )
        draw_auc = {
            name: float(roc_auc_score(y[indices], values[indices]))
            for name, values in cells.items()
        }
        for contrast_name, coefficients in normalized.items():
            draws[contrast_name][draw_index] = sum(
                coefficient * draw_auc[cell]
                for cell, coefficient in coefficients.items()
            )

    result = {}
    for contrast_name, coefficients in normalized.items():
        values = draws[contrast_name]
        result[contrast_name] = {
            "estimate": float(
                sum(
                    coefficient * point_auc[cell]
                    for cell, coefficient in coefficients.items()
                )
            ),
            "se": float(np.std(values, ddof=1 if n_boot > 1 else 0)),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "n_boot": n_boot,
            "seed": seed,
        }
    return result


def _validated_window_records(records, coverage):
    expected = set(WINDOW_KS)
    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError(f"window records must contain exactly {WINDOW_KS}")
    if not isinstance(coverage, dict) or set(coverage) != expected:
        raise ValueError(f"window coverage must contain exactly {WINDOW_KS}")
    validated_records = {}
    validated_coverage = {}
    for window in WINDOW_KS:
        record = records[window]
        if not isinstance(record, dict) or not {"estimate", "se", "ci95"}.issubset(
            record
        ):
            raise ValueError(f"invalid statistical record for final_{window}")
        estimate = float(record["estimate"])
        se = float(record["se"])
        ci = np.asarray(record["ci95"], dtype=float)
        if (
            not np.isfinite([estimate, se]).all()
            or se < 0.0
            or ci.shape != (2,)
            or not np.isfinite(ci).all()
            or ci[0] > ci[1]
        ):
            raise ValueError(f"invalid statistical record for final_{window}")
        coverage_record = coverage[window]
        if not isinstance(coverage_record, dict) or "full_fraction" not in coverage_record:
            raise ValueError(f"invalid coverage record for final_{window}")
        full_fraction = float(coverage_record["full_fraction"])
        if not np.isfinite(full_fraction) or not 0.0 <= full_fraction <= 1.0:
            raise ValueError(f"invalid coverage record for final_{window}")
        validated_records[window] = {
            "estimate": estimate,
            "se": se,
            "ci95": ci.tolist(),
        }
        validated_coverage[window] = full_fraction
    return validated_records, validated_coverage


def select_window(records, coverage, min_coverage=0.95):
    """Apply the frozen one-SE selector and four discovery gate checks."""
    min_coverage = float(min_coverage)
    if not np.isfinite(min_coverage) or not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1]")
    records, coverage_fraction = _validated_window_records(records, coverage)
    eligible = [
        window
        for window in WINDOW_KS
        if coverage_fraction[window] >= min_coverage
    ]
    positive = [window for window in eligible if records[window]["estimate"] > 0.0]

    k_max = None
    threshold = None
    selected = None
    if positive:
        largest = max(records[window]["estimate"] for window in positive)
        k_max = min(
            window for window in positive if records[window]["estimate"] == largest
        )
        threshold = records[k_max]["estimate"] - records[k_max]["se"]
        candidates = [
            window
            for window in positive
            if records[window]["estimate"] >= threshold
        ]
        if candidates:
            selected = min(candidates)

    gate_checks = {
        "at_least_three_eligible": len(eligible) >= 3,
        "selected_interval_above_zero": (
            selected is not None and records[selected]["ci95"][0] > 0.0
        ),
        "at_least_three_positive": len(positive) >= 3,
        "no_interval_entirely_below_zero": not any(
            records[window]["ci95"][1] < 0.0 for window in eligible
        ),
    }
    return {
        "min_coverage": min_coverage,
        "eligible_windows": eligible,
        "ineligible_windows": [
            window for window in WINDOW_KS if window not in eligible
        ],
        "positive_eligible_windows": positive,
        "k_max": k_max,
        "one_se_threshold": float(threshold) if threshold is not None else None,
        "selected_k": selected,
        "gate_checks": gate_checks,
        "proceed": all(gate_checks.values()),
    }


def length_quartiles(lengths, y=None):
    """Assign deterministic within-model length quartiles, retaining gaps."""
    lengths = np.asarray(lengths)
    if (
        lengths.ndim != 1
        or len(lengths) == 0
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths < 0)
    ):
        raise ValueError("lengths must be a non-empty nonnegative integer array")
    labels = None
    if y is not None:
        labels = np.asarray(y)
        if (
            labels.ndim != 1
            or len(labels) != len(lengths)
            or not set(labels.tolist()).issubset({0, 1})
        ):
            raise ValueError("quartile labels must be aligned and binary")

    cut_points = np.quantile(
        lengths, [0.25, 0.5, 0.75], method="linear"
    )
    assignments = np.searchsorted(cut_points, lengths, side="right")
    strata = []
    for quartile in range(4):
        indices = np.flatnonzero(assignments == quartile)
        reason = None
        label_counts = None
        if len(indices) == 0:
            reason = "empty_stratum"
        elif labels is not None:
            observed = set(labels[indices].tolist())
            label_counts = {
                str(label): int(np.sum(labels[indices] == label))
                for label in (0, 1)
            }
            if observed != {0, 1}:
                reason = "one_class"
        strata.append(
            {
                "quartile": quartile + 1,
                "indices": indices.tolist(),
                "n": int(len(indices)),
                "label_counts": label_counts,
                "unestimable_reason": reason,
            }
        )
    return {
        "cut_points": cut_points.tolist(),
        "assignments": assignments.astype(int).tolist(),
        "strata": strata,
    }
