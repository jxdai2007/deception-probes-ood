"""Freeze and analyze the held-out pooling-mechanism confirmation."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from sklearn.metrics import roc_auc_score

from jrp_common.probes import fit_probe
from p04.condition_io import condition_stem, load_condition_batch
from p04.loader import DATASET_FILES, ROLLOUTS_DIR, load_pooled_acts
from p04.pooling import WINDOW_KS
from p04.pooling_mechanism import (
    ScoreSeries,
    align_score_series,
    paired_bootstrap_auroc_contrasts,
)


CONFIRMATION_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
CONFIRMATION_MODEL_SHORT = "Llama-3.2-3B-Instruct"
DISCOVERY_MODEL = "Meta-Llama-3.1-8B-Instruct"
PARITY_TARGET = 0.22073635209479472
PARITY_TOLERANCE = 0.020
N_BOOT = 10000
BOOTSTRAP_SEED = 42
MIN_COVERAGE = 0.95
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXTRACTION_SOURCE_PATHS = (
    "projects/04-deception-probes/p04/condition_io.py",
    "projects/04-deception-probes/p04/extract_activations.py",
    "projects/04-deception-probes/p04/loader.py",
    "projects/04-deception-probes/p04/rendering.py",
    "projects/04-deception-probes/slurm/extract_format_geometry.sbatch",
    "projects/04-deception-probes/slurm/run_format_geometry.sh",
)


class ConfirmationIntegrityError(ValueError):
    """An input failed a frozen confirmation integrity requirement."""


def _canonical_sha256(record):
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_protocol_sha256(record):
    """Hash a protocol using its frozen canonical JSON representation."""
    if not isinstance(record, dict):
        raise TypeError("confirmation protocol must be a JSON object")
    return _canonical_sha256(record)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_component_sign(value):
    value = float(value)
    if value == 0.0 or not np.isfinite(value):
        raise ValueError("confirmation component sign is undefined")
    return 1 if value > 0.0 else -1


def observed_component_sign(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("observed confirmation component is non-finite")
    if value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def resolve_confirmation_model(model_id, *, api=None):
    """Resolve the one approved held-out model to an immutable Hub commit."""
    if model_id != CONFIRMATION_MODEL_ID:
        raise ValueError("unexpected held-out model")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.model_info(model_id)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("held-out model did not resolve to an immutable revision")
    return {
        "model_id": model_id,
        "model_short": CONFIRMATION_MODEL_SHORT,
        "resolved_model_revision": revision,
        "resolved_tokenizer_revision": revision,
    }


def _window(mapping, window):
    if window in mapping:
        return mapping[window]
    return mapping[str(window)]


def _require_sha_mapping(value, description):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{description} must be a non-empty hash mapping")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
        for key, digest in value.items()
    ):
        raise ValueError(f"{description} contains an invalid SHA-256")


def build_confirmation_protocol(discovery, model_resolution, source_identity):
    """Build the frozen protocol, refusing any unfrozen or ineligible input."""
    if not isinstance(discovery, dict) or discovery.get("schema_version") != 1:
        raise ValueError("discovery result has an invalid schema")
    if discovery.get("status") != "exploratory":
        raise ValueError("discovery result is not exploratory")
    if discovery.get("models") != [
        "Meta-Llama-3.1-8B-Instruct",
        "gemma-2-9b-it",
    ]:
        raise ValueError("discovery models do not match the frozen design")
    if discovery.get("bootstrap") != {
        "n_resamples": N_BOOT,
        "seed": BOOTSTRAP_SEED,
    }:
        raise ValueError("discovery bootstrap does not match the frozen design")

    selection = discovery.get("selection")
    if not isinstance(selection, dict) or selection.get("proceed") is not True:
        raise ValueError("discovery gate did not pass")
    gate_checks = selection.get("gate_checks")
    expected_gate_checks = {
        "at_least_three_eligible",
        "selected_interval_above_zero",
        "at_least_three_positive",
        "no_interval_entirely_below_zero",
    }
    if (
        not isinstance(gate_checks, dict)
        or set(gate_checks) != expected_gate_checks
        or any(value is not True for value in gate_checks.values())
    ):
        raise ValueError("discovery gate checks did not all pass")
    selected_k = selection.get("selected_k")
    if type(selected_k) is not int or selected_k not in WINDOW_KS:
        raise ValueError("discovery selected window is invalid")
    try:
        endpoint = _window(
            discovery["window_endpoints"][DISCOVERY_MODEL], selected_k
        )["r2"]
        full_delta = float(endpoint["components"]["full_delta"])
        window_delta = float(endpoint["components"]["window_delta"])
        endpoint_estimate = float(endpoint["estimate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("discovery endpoint is incomplete") from error
    component_signs = {
        "full": frozen_component_sign(full_delta),
        "final_k": frozen_component_sign(window_delta),
    }
    if (
        not np.isfinite(endpoint_estimate)
        or not np.isclose(endpoint_estimate, full_delta - window_delta)
        or full_delta <= window_delta
    ):
        raise ValueError("discovery endpoint and component ordering disagree")

    expected_resolution_keys = {
        "model_id",
        "model_short",
        "resolved_model_revision",
        "resolved_tokenizer_revision",
    }
    if (
        not isinstance(model_resolution, dict)
        or set(model_resolution) != expected_resolution_keys
        or model_resolution.get("model_id") != CONFIRMATION_MODEL_ID
        or model_resolution.get("model_short") != CONFIRMATION_MODEL_SHORT
    ):
        raise ValueError("unexpected held-out model resolution")
    for key in ("resolved_model_revision", "resolved_tokenizer_revision"):
        if not _REVISION_RE.fullmatch(str(model_resolution.get(key, ""))):
            raise ValueError(f"held-out {key} is not immutable")

    if not isinstance(source_identity, dict):
        raise ValueError("source identity is missing")
    if source_identity.get("dirty") is not False:
        raise ValueError("confirmation requires a clean execution-code tree")
    if not _REVISION_RE.fullmatch(str(source_identity.get("git_commit", ""))):
        raise ValueError("analysis-code commit is not immutable")
    _require_sha_mapping(source_identity.get("source_sha256"), "source identity")
    if "confirmation_analysis_source_sha256" in source_identity:
        _require_sha_mapping(
            source_identity["confirmation_analysis_source_sha256"],
            "confirmation analysis source identity",
        )
    _require_sha_mapping(source_identity.get("dataset_sha256"), "dataset identity")
    if set(source_identity["dataset_sha256"]) != {"roleplaying", "sandbagging"}:
        raise ValueError("dataset identity must freeze roleplaying and sandbagging")
    if not _SHA256_RE.fullmatch(
        str(source_identity.get("requirements_lock_sha256", ""))
    ):
        raise ValueError("requirements lock identity is invalid")

    provenance = discovery.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("discovery provenance is missing")
    _require_sha_mapping(
        provenance.get("analysis_source_sha256"),
        "discovery analysis source identity",
    )
    _require_sha_mapping(
        provenance.get("preserved_artifact_sha256"),
        "preserved artifact identity",
    )
    if not _SHA256_RE.fullmatch(
        str(provenance.get("preservation_manifest_sha256", ""))
    ):
        raise ValueError("discovery preservation identity is invalid")
    evidence = provenance.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("discovery evidence identity is missing")

    protocol = {
        "schema_version": 1,
        "experiment": "p04_pooling_mechanism_confirmation",
        "discovery_sha256": _canonical_sha256(discovery),
        "discovery_analysis_source_sha256": dict(
            provenance["analysis_source_sha256"]
        ),
        "discovery_preservation_manifest_sha256": provenance[
            "preservation_manifest_sha256"
        ],
        "discovery_preserved_artifact_sha256": dict(
            provenance["preserved_artifact_sha256"]
        ),
        "discovery_evidence": evidence,
        "selected_k": selected_k,
        "selection_record": selection,
        "discovery_endpoint": endpoint,
        "primary_endpoint": (
            "delta_r2_r0_full_response_minus_"
            f"delta_r2_r0_final_{selected_k}"
        ),
        "primary_formula": (
            "(AUROC_R2_full-AUROC_R0_full)-"
            f"(AUROC_R2_final_{selected_k}-AUROC_R0_final_{selected_k})"
        ),
        "component_signs": component_signs,
        "ordering": "full_delta_gt_final_k_delta",
        "min_full_window_coverage": MIN_COVERAGE,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOTSTRAP_SEED},
        "parity": {
            "target_auroc": PARITY_TARGET,
            "inclusive_tolerance": PARITY_TOLERANCE,
        },
        "renderings": ["r0", "r1", "r2"],
        "inclusion_rules": {
            "labels": [0, 1],
            "pairing": "common_example_id_and_immutable_hash",
            "zero_length": "exclude",
            "truncated": "exclude",
            "short_final_windows": "retain_and_report_coverage",
            "eos_token": "excluded",
            "prompt_tokens": "excluded",
            "invalid_selected_answer": "exclude_and_record",
        },
        "model": dict(model_resolution),
        "instrument": {"layer_frac": 0.275, "max_tokens": 2048},
        "dataset_sha256": dict(source_identity["dataset_sha256"]),
        "requirements_lock_sha256": source_identity[
            "requirements_lock_sha256"
        ],
        "analysis_code_commit": source_identity["git_commit"],
        "source_sha256": dict(source_identity["source_sha256"]),
        "artifact_schema_version": 1,
    }
    if "confirmation_analysis_source_sha256" in source_identity:
        protocol["confirmation_analysis_source_sha256"] = dict(
            source_identity["confirmation_analysis_source_sha256"]
        )
    return protocol


def primary_decision(endpoint, full_delta, window_delta, protocol):
    """Apply only the frozen primary CI, component signs, and ordering."""
    ci_low = float(endpoint["ci95"][0])
    signs_match = (
        observed_component_sign(full_delta)
        == protocol["component_signs"]["full"]
        and observed_component_sign(window_delta)
        == protocol["component_signs"]["final_k"]
    )
    ordered = full_delta > window_delta
    return (
        "confirmed"
        if ci_low > 0.0 and signs_match and ordered
        else "not_confirmed"
    )


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _current_confirmation_analysis_hashes():
    repo_root = _repo_root()
    project_root = Path(__file__).resolve().parent.parent
    paths = (
        Path(__file__).resolve(),
        project_root / "p04" / "condition_io.py",
        project_root / "p04" / "loader.py",
        project_root / "p04" / "pooling.py",
        project_root / "p04" / "pooling_mechanism.py",
        Path(inspect.getsourcefile(fit_probe) or ""),
    )
    return {
        str(path.relative_to(repo_root)): _sha256_file(path) for path in paths
    }


def _current_source_identity():
    repo_root = _repo_root()
    extraction_hashes = {
        relative: _sha256_file(repo_root / relative)
        for relative in _EXTRACTION_SOURCE_PATHS
    }
    lock_path = repo_root / "common" / "env" / "requirements.lock.txt"
    dataset_hashes = {
        dataset: _sha256_file(ROLLOUTS_DIR / DATASET_FILES[dataset])
        for dataset in ("roleplaying", "sandbagging")
    }
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "common",
            "projects/04-deception-probes/p04",
            "projects/04-deception-probes/slurm",
        ],
        text=True,
    )
    return {
        "git_commit": commit,
        "dirty": bool(status.strip()),
        "source_sha256": extraction_hashes,
        "confirmation_analysis_source_sha256": (
            _current_confirmation_analysis_hashes()
        ),
        "requirements_lock_sha256": _sha256_file(lock_path),
        "dataset_sha256": dataset_hashes,
    }


def _load_json_object(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmationIntegrityError(
            f"invalid {description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ConfirmationIntegrityError(f"{description} must be a JSON object")
    return value


def _artifact_paths(acts_dir, model_short):
    root = Path(acts_dir)
    stems = {
        "roleplaying": f"{model_short}__roleplaying",
        "historical": f"{model_short}__sandbagging",
        **{
            rendering: condition_stem(model_short, rendering)
            for rendering in ("r0", "r1", "r2")
        },
    }
    output = {}
    for name, stem in stems.items():
        suffixes = ["acts.npy", "lengths.npy", "labels.npy", "meta.json"]
        if name in {"r0", "r1", "r2"}:
            suffixes.append("rows.json")
        output[name] = [root / f"{stem}__{suffix}" for suffix in suffixes]
    return output


def _expected_protocol_identity(protocol):
    return {
        "schema_version": 1,
        "selected_k": protocol["selected_k"],
        "model_short": protocol["model"]["model_short"],
        "resolved_model_revision": protocol["model"][
            "resolved_model_revision"
        ],
        "resolved_tokenizer_revision": protocol["model"][
            "resolved_tokenizer_revision"
        ],
    }


def _validate_artifact_identity(protocol, digest, acts_dir):
    paths = _artifact_paths(acts_dir, protocol["model"]["model_short"])
    identity = _expected_protocol_identity(protocol)
    hashes = {}
    metas = {}
    for leg, leg_paths in paths.items():
        for path in leg_paths:
            if not path.is_file():
                raise ConfirmationIntegrityError(f"missing artifact: {path.name}")
            hashes[path.name] = _sha256_file(path)
        meta_path = next(
            path for path in leg_paths if path.name.endswith("__meta.json")
        )
        meta = _load_json_object(meta_path, f"{leg} metadata")
        metas[leg] = meta
        if meta.get("experiment_protocol_sha256") != digest:
            raise ConfirmationIntegrityError(
                f"protocol identity mismatch for {leg}"
            )
        if meta.get("experiment_protocol_identity") != identity:
            raise ConfirmationIntegrityError(
                f"protocol identity mismatch for {leg}"
            )
        if meta.get("model") != protocol["model"]["model_id"]:
            raise ConfirmationIntegrityError(f"model identity mismatch for {leg}")
        if (
            meta.get("resolved_model_revision")
            != protocol["model"]["resolved_model_revision"]
        ):
            raise ConfirmationIntegrityError(
                f"model revision mismatch for {leg}"
            )
        if (
            meta.get("resolved_tokenizer_revision")
            != protocol["model"]["resolved_tokenizer_revision"]
        ):
            raise ConfirmationIntegrityError(
                f"tokenizer revision mismatch for {leg}"
            )
        if meta.get("layer_frac") != protocol["instrument"]["layer_frac"]:
            raise ConfirmationIntegrityError(f"layer mismatch for {leg}")
        if meta.get("max_tokens") != protocol["instrument"]["max_tokens"]:
            raise ConfirmationIntegrityError(f"sequence length mismatch for {leg}")
        if meta.get("limit") is not None:
            raise ConfirmationIntegrityError(f"limited artifact for {leg}")
        expected_dataset = "roleplaying" if leg == "roleplaying" else "sandbagging"
        if meta.get("dataset") != expected_dataset:
            raise ConfirmationIntegrityError(f"dataset mismatch for {leg}")
        expected_rendering = leg if leg in {"r0", "r1", "r2"} else None
        if meta.get("rendering") != expected_rendering:
            raise ConfirmationIntegrityError(f"rendering mismatch for {leg}")
        provenance = meta.get("extraction_provenance")
        if not isinstance(provenance, dict):
            raise ConfirmationIntegrityError(f"source provenance missing for {leg}")
        if provenance.get("source_sha256") != protocol["source_sha256"]:
            raise ConfirmationIntegrityError(f"source identity mismatch for {leg}")
        if (
            provenance.get("requirements_lock_sha256")
            != protocol["requirements_lock_sha256"]
        ):
            raise ConfirmationIntegrityError(
                f"requirements lock mismatch for {leg}"
            )
    return hashes, metas


def _binary_probe(X, y, description):
    X = np.asarray(X)
    y = np.asarray(y)
    keep = np.isin(y, [0, 1])
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ConfirmationIntegrityError(f"invalid {description} activation artifact")
    if not np.isfinite(X).all() or set(y[keep].tolist()) != {0, 1}:
        raise ConfirmationIntegrityError(
            f"{description} must contain finite rows from both classes"
        )
    return fit_probe(X[keep], y[keep]), keep


def load_treatment_batch(model_short, rendering, pooling, acts_dir):
    """Load one condition batch; kept as the parity load-boundary seam."""
    return load_condition_batch(model_short, rendering, pooling, acts_dir)


def _score_series(probe, batch):
    return ScoreSeries(
        example_ids=np.asarray(batch.example_ids),
        immutable_hashes=np.asarray(batch.immutable_hashes),
        y=np.asarray(batch.y),
        scores=np.asarray(probe.decision_function(batch.X), dtype=float),
        source_lengths=np.asarray(batch.response_lengths),
    )


def _ids_sha256(ids):
    encoded = json.dumps(
        np.asarray(ids).tolist(), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aligned_endpoint(full_probe, window_probe, batches):
    named = {
        "full_r0": _score_series(full_probe, batches["full_r0"]),
        "full_treatment": _score_series(full_probe, batches["full_treatment"]),
        "window_r0": _score_series(window_probe, batches["window_r0"]),
        "window_treatment": _score_series(
            window_probe, batches["window_treatment"]
        ),
    }
    names = list(named)
    aligned_values = align_score_series([named[name] for name in names])
    aligned = dict(zip(names, aligned_values))
    return aligned_values, aligned


def _endpoint(aligned_values, aligned, protocol):
    scores = {name: value.scores for name, value in aligned.items()}
    endpoint = paired_bootstrap_auroc_contrasts(
        aligned_values[0].y,
        scores,
        {
            "effect": {
                "full_treatment": 1.0,
                "full_r0": -1.0,
                "window_treatment": -1.0,
                "window_r0": 1.0,
            }
        },
        n_boot=protocol["bootstrap"]["n_resamples"],
        seed=protocol["bootstrap"]["seed"],
    )["effect"]
    cell_aurocs = {
        name: float(roc_auc_score(aligned_values[0].y, values))
        for name, values in scores.items()
    }
    full_delta = cell_aurocs["full_treatment"] - cell_aurocs["full_r0"]
    window_delta = (
        cell_aurocs["window_treatment"] - cell_aurocs["window_r0"]
    )
    endpoint.update(
        {
            "n_common": int(len(aligned_values[0].y)),
            "common_ids_sha256": _ids_sha256(aligned_values[0].example_ids),
            "cell_aurocs": cell_aurocs,
        }
    )
    return endpoint, {
        "full_delta": float(full_delta),
        "window_delta": float(window_delta),
    }, aligned


def _secondary_endpoint(full_probe, window_probe, batches):
    """Compute a point-only diagnostic after the primary decision is frozen."""
    aligned_values, aligned = _aligned_endpoint(
        full_probe, window_probe, batches
    )
    cell_aurocs = {
        name: float(roc_auc_score(aligned_values[0].y, item.scores))
        for name, item in aligned.items()
    }
    full_delta = cell_aurocs["full_treatment"] - cell_aurocs["full_r0"]
    window_delta = (
        cell_aurocs["window_treatment"] - cell_aurocs["window_r0"]
    )
    components = {
        "full_delta": float(full_delta),
        "window_delta": float(window_delta),
    }
    return {
        "estimate": float(full_delta - window_delta),
        "inference": "secondary_point_diagnostic",
        "n_common": int(len(aligned_values[0].y)),
        "common_ids_sha256": _ids_sha256(aligned_values[0].example_ids),
        "cell_aurocs": cell_aurocs,
        "components": components,
    }, aligned


def _coverage(role_meta, aligned, window):
    role_coverage = role_meta.get("pooling_coverage")
    if not isinstance(role_coverage, dict):
        raise ConfirmationIntegrityError("roleplaying window coverage is missing")
    paired = (
        (aligned["window_r0"].source_lengths >= window)
        & (aligned["window_treatment"].source_lengths >= window)
    )
    n_total = int(len(paired))
    n_full = int(paired.sum())
    paired_fraction = n_full / n_total if n_total else 0.0
    eligible = min(float(role_coverage["full_fraction"]), paired_fraction)
    return {
        "roleplaying_fit": dict(role_coverage),
        "paired": {
            "k": window,
            "n_total": n_total,
            "n_full": n_full,
            "n_partial": n_total - n_full,
            "full_fraction": paired_fraction,
        },
        "eligible_full_fraction": eligible,
    }


def _failure_result(protocol, digest, status, integrity, parity):
    return {
        "schema_version": 1,
        "status": status,
        "primary_decision": "not_tested",
        "protocol": protocol,
        "protocol_sha256": digest,
        "integrity": integrity,
        "parity": parity,
    }


def _validate_protocol_record(protocol):
    if protocol.get("schema_version") != 1:
        raise ConfirmationIntegrityError("confirmation protocol schema mismatch")
    if protocol.get("experiment") != "p04_pooling_mechanism_confirmation":
        raise ConfirmationIntegrityError("confirmation protocol identity mismatch")
    model = protocol.get("model")
    if (
        not isinstance(model, dict)
        or model.get("model_id") != CONFIRMATION_MODEL_ID
        or model.get("model_short") != CONFIRMATION_MODEL_SHORT
    ):
        raise ConfirmationIntegrityError("confirmation model mismatch")
    if any(
        not _REVISION_RE.fullmatch(str(model.get(key, "")))
        for key in ("resolved_model_revision", "resolved_tokenizer_revision")
    ):
        raise ConfirmationIntegrityError("confirmation revision mismatch")
    selected_k = protocol.get("selected_k")
    if selected_k not in WINDOW_KS:
        raise ConfirmationIntegrityError("confirmation window mismatch")
    if protocol.get("primary_endpoint") != (
        "delta_r2_r0_full_response_minus_"
        f"delta_r2_r0_final_{selected_k}"
    ):
        raise ConfirmationIntegrityError("confirmation endpoint mismatch")
    component_signs = protocol.get("component_signs")
    if (
        not isinstance(component_signs, dict)
        or set(component_signs) != {"full", "final_k"}
        or set(component_signs.values()) - {-1, 1}
    ):
        raise ConfirmationIntegrityError("confirmation component signs mismatch")
    if protocol.get("ordering") != "full_delta_gt_final_k_delta":
        raise ConfirmationIntegrityError("confirmation ordering mismatch")
    if protocol.get("min_full_window_coverage") != MIN_COVERAGE:
        raise ConfirmationIntegrityError("confirmation coverage rule mismatch")
    if protocol.get("bootstrap") != {
        "n_resamples": N_BOOT,
        "seed": BOOTSTRAP_SEED,
    }:
        raise ConfirmationIntegrityError("confirmation bootstrap mismatch")
    if protocol.get("parity") != {
        "target_auroc": PARITY_TARGET,
        "inclusive_tolerance": PARITY_TOLERANCE,
    }:
        raise ConfirmationIntegrityError("confirmation parity rule mismatch")
    if protocol.get("renderings") != ["r0", "r1", "r2"]:
        raise ConfirmationIntegrityError("confirmation renderings mismatch")
    if protocol.get("instrument") != {
        "layer_frac": 0.275,
        "max_tokens": 2048,
    }:
        raise ConfirmationIntegrityError("confirmation instrument mismatch")
    try:
        _require_sha_mapping(protocol.get("source_sha256"), "source identity")
        _require_sha_mapping(protocol.get("dataset_sha256"), "dataset identity")
    except ValueError as error:
        raise ConfirmationIntegrityError(str(error)) from error
    if set(protocol["dataset_sha256"]) != {"roleplaying", "sandbagging"}:
        raise ConfirmationIntegrityError("confirmation dataset identity mismatch")
    if not _SHA256_RE.fullmatch(
        str(protocol.get("requirements_lock_sha256", ""))
    ):
        raise ConfirmationIntegrityError("requirements lock identity mismatch")
    frozen_analysis = protocol.get("confirmation_analysis_source_sha256")
    if frozen_analysis is not None:
        try:
            _require_sha_mapping(
                frozen_analysis, "confirmation analysis source identity"
            )
        except ValueError as error:
            raise ConfirmationIntegrityError(str(error)) from error
        if frozen_analysis != _current_confirmation_analysis_hashes():
            raise ConfirmationIntegrityError(
                "confirmation analysis source identity mismatch"
            )


def analyze_confirmation(protocol_path, acts_dir):
    """Validate and run the parity-first held-out confirmation analysis."""
    protocol = {}
    digest = ""
    parity = {"passed": False, "status": "not_tested"}
    try:
        protocol = _load_json_object(protocol_path, "confirmation protocol")
        digest = canonical_protocol_sha256(protocol)
        _validate_protocol_record(protocol)
        artifact_hashes, _ = _validate_artifact_identity(
            protocol, digest, acts_dir
        )
        integrity = {
            "passed": True,
            "checks": {
                "protocol_binding": True,
                "model_tokenizer_identity": True,
                "source_identity": True,
                "artifact_sha256": artifact_hashes,
            },
        }
    except (ConfirmationIntegrityError, KeyError, TypeError, ValueError) as error:
        return _failure_result(
            protocol,
            digest,
            "integrity_failed",
            {"passed": False, "error": str(error)},
            parity,
        )

    model_short = protocol["model"]["model_short"]
    try:
        role_X, role_y, _ = load_pooled_acts(
            model_short, "roleplaying", acts_dir, pooling="full_response"
        )
        full_probe, _ = _binary_probe(role_X, role_y, "roleplaying")
        historical_X, historical_y, _ = load_pooled_acts(
            model_short, "sandbagging", acts_dir, pooling="full_response"
        )
        historical_keep = np.isin(historical_y, [0, 1])
        if set(historical_y[historical_keep].tolist()) != {0, 1}:
            raise ConfirmationIntegrityError(
                "historical parity requires both binary classes"
            )
        historical_scores = full_probe.decision_function(
            historical_X[historical_keep]
        )
        r0_full_batch = load_condition_batch(
            model_short, "r0", "full_response", acts_dir
        )
        observed_parity = float(
            roc_auc_score(historical_y[historical_keep], historical_scores)
        )
        target = float(protocol["parity"]["target_auroc"])
        tolerance = float(protocol["parity"]["inclusive_tolerance"])
        parity = {
            "passed": abs(observed_parity - target) <= tolerance,
            "observed_auroc": observed_parity,
            "target_auroc": target,
            "inclusive_tolerance": tolerance,
            "inclusive_interval": [target - tolerance, target + tolerance],
            "n_binary": int(historical_keep.sum()),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _failure_result(
            protocol,
            digest,
            "integrity_failed",
            {"passed": False, "error": str(error)},
            parity,
        )
    if not parity["passed"]:
        return _failure_result(
            protocol, digest, "parity_failed", integrity, parity
        )

    try:
        selected_k = protocol["selected_k"]
        window_pooling = f"final_{selected_k}"
        role_window_X, role_window_y, role_window_meta = load_pooled_acts(
            model_short, "roleplaying", acts_dir, pooling=window_pooling
        )
        window_probe, _ = _binary_probe(
            role_window_X, role_window_y, f"roleplaying {window_pooling}"
        )
        primary_batches = {
            "full_r0": r0_full_batch,
            "full_treatment": load_treatment_batch(
                model_short, "r2", "full_response", acts_dir
            ),
            "window_r0": load_treatment_batch(
                model_short, "r0", window_pooling, acts_dir
            ),
            "window_treatment": load_treatment_batch(
                model_short, "r2", window_pooling, acts_dir
            ),
        }
        aligned_values, aligned = _aligned_endpoint(
            full_probe, window_probe, primary_batches
        )
        selected_coverage = _coverage(role_window_meta, aligned, selected_k)
        if (
            selected_coverage["eligible_full_fraction"]
            < protocol["min_full_window_coverage"]
        ):
            raise ConfirmationIntegrityError(
                "selected-window coverage is below the frozen minimum"
            )
        endpoint, components, aligned = _endpoint(
            aligned_values, aligned, protocol
        )
        decision = primary_decision(
            endpoint,
            components["full_delta"],
            components["window_delta"],
            protocol,
        )

        r1_batches = {
            "full_r0": primary_batches["full_r0"],
            "full_treatment": load_treatment_batch(
                model_short, "r1", "full_response", acts_dir
            ),
            "window_r0": primary_batches["window_r0"],
            "window_treatment": load_treatment_batch(
                model_short, "r1", window_pooling, acts_dir
            ),
        }
        r1_endpoint, r1_aligned = _secondary_endpoint(
            full_probe, window_probe, r1_batches
        )
        r1_endpoint["coverage"] = _coverage(
            role_window_meta, r1_aligned, selected_k
        )

        other_windows = {}
        for window in WINDOW_KS:
            if window == selected_k:
                continue
            pooling = f"final_{window}"
            role_other_X, role_other_y, role_other_meta = load_pooled_acts(
                model_short, "roleplaying", acts_dir, pooling=pooling
            )
            other_probe, _ = _binary_probe(
                role_other_X, role_other_y, f"roleplaying {pooling}"
            )
            other_batches = {
                "full_r0": primary_batches["full_r0"],
                "full_treatment": primary_batches["full_treatment"],
                "window_r0": load_treatment_batch(
                    model_short, "r0", pooling, acts_dir
                ),
                "window_treatment": load_treatment_batch(
                    model_short, "r2", pooling, acts_dir
                ),
            }
            other_endpoint, other_aligned = _secondary_endpoint(
                full_probe, other_probe, other_batches
            )
            other_endpoint["coverage"] = _coverage(
                role_other_meta, other_aligned, window
            )
            other_windows[str(window)] = other_endpoint

        final_hashes, _ = _validate_artifact_identity(protocol, digest, acts_dir)
        if final_hashes != artifact_hashes:
            raise ConfirmationIntegrityError(
                "artifact hashes changed during confirmation analysis"
            )
    except (
        OSError,
        ConfirmationIntegrityError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return _failure_result(
            protocol,
            digest,
            "integrity_failed",
            {"passed": False, "error": str(error)},
            parity,
        )

    return {
        "schema_version": 1,
        "status": decision,
        "primary_decision": decision,
        "protocol": protocol,
        "protocol_sha256": digest,
        "integrity": integrity,
        "parity": parity,
        "primary_endpoint": endpoint,
        "components": components,
        "coverage": selected_coverage,
        "secondary": {"r1": r1_endpoint, "other_windows": other_windows},
        "limitations": [
            "All intervals condition on fitted probes.",
            "R1 and non-selected windows are secondary and do not alter "
            "the primary decision.",
        ],
    }


def _exclusive_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve-model")
    resolve.add_argument("--model", required=True)
    resolve.add_argument("--out", required=True)

    build = subparsers.add_parser("build-protocol")
    build.add_argument("--discovery", required=True)
    build.add_argument("--model-resolution", required=True)
    build.add_argument("--out", required=True)

    hash_parser = subparsers.add_parser("hash-protocol")
    hash_parser.add_argument("--protocol", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--protocol", required=True)
    analyze.add_argument("--acts-dir", required=True)
    analyze.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "resolve-model":
        return _exclusive_json(
            args.out, resolve_confirmation_model(args.model)
        )
    if args.command == "build-protocol":
        discovery = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
        resolution = json.loads(
            Path(args.model_resolution).read_text(encoding="utf-8")
        )
        return _exclusive_json(
            args.out,
            build_confirmation_protocol(
                discovery, resolution, _current_source_identity()
            ),
        )
    if args.command == "hash-protocol":
        protocol = json.loads(
            Path(args.protocol).read_text(encoding="utf-8")
        )
        digest = canonical_protocol_sha256(protocol)
        print(digest)
        return digest
    return _exclusive_json(
        args.out, analyze_confirmation(args.protocol, args.acts_dir)
    )


if __name__ == "__main__":
    main()
