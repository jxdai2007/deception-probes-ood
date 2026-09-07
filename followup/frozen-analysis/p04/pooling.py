"""Frozen response-pooling rules for Project 04 activation artifacts."""

import numpy as np


WINDOW_KS = (1, 2, 4, 8)


def parse_pooling(name):
    """Parse a frozen pooling name into its mode and optional window size."""
    if name in {"full_response", "answer_span"}:
        return name, None
    if isinstance(name, str) and name.startswith("final_"):
        value = name.removeprefix("final_")
        if value.isdigit() and int(value) in WINDOW_KS:
            return "final_window", int(value)
    raise ValueError(f"unknown pooling {name!r}")


def pool_response_rows(
    acts,
    lengths,
    *,
    pooling,
    answer_spans=None,
    include=None,
):
    """Pool flattened response rows and report fixed-window coverage."""
    acts = np.asarray(acts, dtype=np.float32)
    lengths = np.asarray(lengths)
    mode, k = parse_pooling(pooling)
    if acts.ndim != 2 or not np.isfinite(acts).all():
        raise ValueError("activations must be a finite 2D array")
    if (
        lengths.ndim != 1
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths < 0)
        or int(lengths.sum()) != len(acts)
    ):
        raise ValueError("flattened activations and lengths are inconsistent")

    if include is None:
        mask = np.ones(len(lengths), dtype=bool)
    else:
        mask = np.asarray(include)
        if mask.ndim != 1 or len(mask) != len(lengths):
            raise ValueError("include mask must align with response lengths")
        mask = mask.astype(bool, copy=False)

    spans = None
    if mode == "answer_span":
        if answer_spans is None:
            raise ValueError("answer spans are required for answer_span pooling")
        spans = np.asarray(answer_spans)
        if (
            spans.shape != (len(lengths), 2)
            or not np.issubdtype(spans.dtype, np.integer)
        ):
            raise ValueError("answer spans must be an aligned integer [n, 2] array")

    pooled = []
    kept = []
    full_window = []
    offset = 0
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        row = acts[offset : offset + length]
        offset += length
        if not mask[index] or length == 0:
            continue
        if mode == "full_response":
            selected = row
        elif mode == "answer_span":
            start, end = map(int, spans[index])
            if start < 0 or end <= start or end > length:
                raise ValueError(
                    f"invalid answer span at row {index}: "
                    f"{[start, end]} with length {length}"
                )
            selected = row[start:end]
        else:
            selected = row[max(0, length - k) :]
            full_window.append(length >= k)
        pooled.append(selected.mean(axis=0))
        kept.append(index)

    if not pooled:
        raise ValueError(f"no valid rows for {pooling}")
    X = np.stack(pooled).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite pooled activations for {pooling}")

    coverage = None
    if mode == "final_window":
        n_full = int(sum(full_window))
        coverage = {
            "k": k,
            "n_total": len(full_window),
            "n_full": n_full,
            "n_partial": len(full_window) - n_full,
            "full_fraction": n_full / len(full_window),
        }
    return X, np.asarray(kept, dtype=int), coverage
