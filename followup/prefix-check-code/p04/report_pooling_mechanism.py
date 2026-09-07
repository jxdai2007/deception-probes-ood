"""Write deterministic internal reports for the pooling-mechanism audit."""

import argparse
import hashlib
import json
from pathlib import Path


POOLING_ORDER = (
    "full_response",
    "answer_span",
    "final_1",
    "final_2",
    "final_4",
    "final_8",
)
WINDOWS = (1, 2, 4, 8)


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window(mapping, window):
    return mapping[window] if window in mapping else mapping[str(window)]


def _validate_discovery(result):
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("discovery result has an invalid schema")
    if result.get("status") != "exploratory":
        raise ValueError("discovery result must be labeled exploratory")
    required = {
        "models",
        "model_results",
        "heterogeneity",
        "window_endpoints",
        "coverage",
        "selection",
        "length_diagnostic",
        "provenance",
        "limitations",
    }
    if not required.issubset(result):
        raise ValueError("discovery result is missing required fields")
    provenance = result["provenance"]
    if (
        not isinstance(provenance.get("evidence"), dict)
        or not provenance["evidence"]
        or not isinstance(provenance.get("analysis_source_sha256"), dict)
        or not provenance["analysis_source_sha256"]
        or not isinstance(provenance.get("preservation_manifest_sha256"), str)
    ):
        raise ValueError("discovery provenance is incomplete")


def _validate_confirmation(discovery, confirmation):
    _validate_discovery(discovery)
    if not isinstance(confirmation, dict) or confirmation.get("schema_version") != 1:
        raise ValueError("confirmation result has an invalid schema")
    protocol = confirmation.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("confirmation protocol is missing")
    if confirmation.get("protocol_sha256") != _canonical_sha256(protocol):
        raise ValueError("confirmation protocol hash mismatch")
    if protocol.get("discovery_sha256") != _canonical_sha256(discovery):
        raise ValueError("confirmation discovery hash mismatch")
    if protocol.get("discovery_analysis_source_sha256") != discovery[
        "provenance"
    ]["analysis_source_sha256"]:
        raise ValueError("confirmation discovery analysis source mismatch")
    if protocol.get("selected_k") != discovery["selection"].get("selected_k"):
        raise ValueError("confirmation selected window mismatch")
    expected_decisions = {
        "confirmed": "confirmed",
        "not_confirmed": "not_confirmed",
        "parity_failed": "not_tested",
        "integrity_failed": "not_tested",
    }
    status = confirmation.get("status")
    if status not in expected_decisions:
        raise ValueError("unknown confirmation status")
    if confirmation.get("primary_decision") != expected_decisions[status]:
        raise ValueError("confirmation status and primary decision disagree")
    if status in {"confirmed", "not_confirmed"}:
        if "primary_endpoint" not in confirmation:
            raise ValueError("evaluated confirmation is missing its primary endpoint")
    elif "primary_endpoint" in confirmation:
        raise ValueError("failed integrity/parity record contains treatment metrics")


def _exclusive_write(path, text):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _stat_text(record):
    return (
        f"estimate={float(record['estimate']):.6f}; "
        f"95% interval={list(map(float, record['ci95']))}; "
        f"SE={float(record['se']):.6f}; n_common={record.get('n_common')}"
    )


def write_discovery_report(result, out_path, date):
    """Write the complete exploratory internal evidence report."""
    _validate_discovery(result)
    lines = [
        "# Project 04 pooling-mechanism discovery",
        "",
        f"**Date:** {date}",
        "**Status:** exploratory internal evidence; not publication prose",
        f"**Phase A protocol:** {result.get('phase_a_protocol_sha256')}",
        "",
        "## Provenance",
        "",
        f"- evidence: {json.dumps(result['provenance']['evidence'], sort_keys=True)}",
        "- preservation_manifest_sha256: "
        f"{result['provenance']['preservation_manifest_sha256']}",
        "- analysis_source_sha256: "
        f"{json.dumps(result['provenance']['analysis_source_sha256'], sort_keys=True)}",
        "",
        "## R2 effects by pooling",
        "",
        "| Model | Pooling | Delta(R2-R0) | 95% interval |",
        "|---|---|---:|---|",
    ]
    for model in result["models"]:
        for pooling in POOLING_ORDER:
            effect = result["model_results"][model]["poolings"][pooling][
                "effects"
            ]["r2"]
            lines.append(
                f"| {model} | {pooling} | {effect['estimate']:.6f} | "
                f"{effect['ci95']} |"
            )

    lines.extend(
        [
            "",
            "## Model heterogeneity",
            "",
            "| Pooling | Treatment | H(p) | 95% interval |",
            "|---|---|---:|---|",
        ]
    )
    for pooling in POOLING_ORDER:
        for treatment in ("r1", "r2"):
            effect = result["heterogeneity"][pooling][treatment]
            lines.append(
                f"| {pooling} | {treatment.upper()} | "
                f"{effect['estimate']:.6f} | {effect['ci95']} |"
            )

    lines.extend(
        [
            "",
            "## Frozen-window audit",
            "",
            "| Model | K | P_R1(K) | P_R2(K) | eligibility coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in result["models"]:
        for window in WINDOWS:
            endpoints = _window(result["window_endpoints"][model], window)
            coverage = _window(result["coverage"][model], window)
            lines.append(
                f"| {model} | {window} | {endpoints['r1']['estimate']:.6f} | "
                f"{endpoints['r2']['estimate']:.6f} | "
                f"{coverage['eligible_full_fraction']:.6f} |"
            )
            lines.extend(
                [
                    "",
                    f"Coverage detail for {model} / K={window}:",
                    "",
                    "```json",
                    json.dumps(coverage, sort_keys=True),
                    "```",
                ]
            )

    selection = result["selection"]
    lines.extend(
        [
            "",
            "## K* selection and discovery gate",
            "",
            f"- K*: {selection['selected_k']}",
            f"- K_max: {selection.get('k_max')}",
            f"- one-SE threshold: {selection.get('one_se_threshold')}",
            "- directionally coherent discovery gate: "
            f"{'passed' if selection['proceed'] else 'failed'}.",
        ]
    )
    for name, passed in selection["gate_checks"].items():
        lines.append(f"- {name}: {passed}")

    diagnostic = result["length_diagnostic"]
    lines.extend(["", "## Conditional length diagnostic", ""])
    if diagnostic["status"] == "skipped":
        lines.append(f"Skipped: {diagnostic['reason']}")
    else:
        lines.extend(["```json", json.dumps(diagnostic, sort_keys=True), "```"])

    lines.extend(["", "## Required limitations", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    return _exclusive_write(out_path, "\n".join(lines) + "\n")


def write_confirmation_report(discovery, confirmation, out_path, date):
    """Write an internal held-out report without recomputing its decision."""
    _validate_confirmation(discovery, confirmation)
    status_labels = {
        "confirmed": "Confirmation success",
        "not_confirmed": "Primary endpoint not confirmed",
        "parity_failed": "Historical parity failed",
        "integrity_failed": "Integrity failure",
    }
    lines = [
        "# Project 04 pooling-mechanism confirmation",
        "",
        f"**Date:** {date}",
        "**Status:** internal evidence; not publication prose",
        f"**Outcome:** {status_labels[confirmation['status']]}",
        f"**protocol_sha256:** {confirmation['protocol_sha256']}",
        "",
        "## Integrity",
        "",
        json.dumps(confirmation["integrity"], sort_keys=True),
        "",
        "## Historical parity",
        "",
        json.dumps(confirmation["parity"], sort_keys=True),
        "",
        "## Primary endpoint",
        "",
        f"primary_decision: {confirmation['primary_decision']}",
    ]
    if "primary_endpoint" in confirmation:
        lines.append(_stat_text(confirmation["primary_endpoint"]))
        lines.append(
            f"components: {json.dumps(confirmation.get('components'), sort_keys=True)}"
        )
    else:
        lines.append("Primary endpoint not tested; no treatment metrics recorded.")
    lines.extend(
        [
            "",
            "## Secondary diagnostics",
            "",
            json.dumps(confirmation.get("secondary", {}), sort_keys=True),
            "",
            "Secondary diagnostics do not alter the recorded primary decision.",
        ]
    )
    return _exclusive_write(out_path, "\n".join(lines) + "\n")


def _load_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discovery_parser = subparsers.add_parser("discovery")
    discovery_parser.add_argument("--result", required=True)
    discovery_parser.add_argument("--out", required=True)
    discovery_parser.add_argument("--date", required=True)
    confirmation_parser = subparsers.add_parser("confirmation")
    confirmation_parser.add_argument("--discovery", required=True)
    confirmation_parser.add_argument("--confirmation", required=True)
    confirmation_parser.add_argument("--out", required=True)
    confirmation_parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    if args.command == "discovery":
        return write_discovery_report(
            _load_json(args.result), args.out, args.date
        )
    return write_confirmation_report(
        _load_json(args.discovery),
        _load_json(args.confirmation),
        args.out,
        args.date,
    )


if __name__ == "__main__":
    main()
