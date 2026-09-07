"""Render deterministic internal figures for the pooling-mechanism audit."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "Meta-Llama-3.1-8B-Instruct": "#2166ac",
    "gemma-2-9b-it": "#b2182b",
    "Llama-3.2-3B-Instruct": "#4d9221",
}
POOLING_MARKERS = {
    "full_response": "o",
    "answer_span": "s",
    "final_1": "^",
    "final_2": "v",
    "final_4": "D",
    "final_8": "P",
}
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
        "window_endpoints",
        "selection",
        "provenance",
    }
    if not required.issubset(result):
        raise ValueError("discovery result is missing required fields")
    source_hashes = result["provenance"].get("analysis_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("discovery analysis source hashes are missing")


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


def _exclusive_save(fig, out_path):
    out_path = Path(out_path)
    with out_path.open("xb") as handle:
        fig.savefig(
            handle,
            format="png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return out_path


def make_discovery_figure(result, out_path):
    """Plot R2 effects and pooling-sensitivity curves from discovery JSON."""
    _validate_discovery(result)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)

    pooling_order = (
        "answer_span",
        "full_response",
        "final_1",
        "final_2",
        "final_4",
        "final_8",
    )
    labels = ("answer", "full", "K=1", "K=2", "K=4", "K=8")
    fig, (ax_delta, ax_sensitivity) = plt.subplots(1, 2, figsize=(11, 4.4))
    for model in result["models"]:
        color = MODEL_COLORS[model]
        deltas = [
            result["model_results"][model]["poolings"][pooling]["effects"][
                "r2"
            ]["estimate"]
            for pooling in pooling_order
        ]
        x = np.arange(len(pooling_order))
        ax_delta.plot(x, deltas, color=color, linewidth=1.4, label=model)
        for index, pooling in enumerate(pooling_order):
            ax_delta.scatter(
                index,
                deltas[index],
                color=color,
                marker=POOLING_MARKERS[pooling],
                s=45,
                zorder=3,
            )

        endpoints = result["window_endpoints"][model]
        values = [_window(endpoints, window)["r2"]["estimate"] for window in WINDOWS]
        lows = [_window(endpoints, window)["r2"]["ci95"][0] for window in WINDOWS]
        highs = [_window(endpoints, window)["r2"]["ci95"][1] for window in WINDOWS]
        ax_sensitivity.plot(
            WINDOWS,
            values,
            color=color,
            linewidth=1.4,
            label=model,
        )
        for index, window in enumerate(WINDOWS):
            ax_sensitivity.errorbar(
                window,
                values[index],
                yerr=[
                    [values[index] - lows[index]],
                    [highs[index] - values[index]],
                ],
                color=color,
                marker=POOLING_MARKERS[f"final_{window}"],
                capsize=3,
                linewidth=1.2,
            )

    ax_delta.axhline(0.0, color="#666666", linewidth=0.8)
    ax_sensitivity.axhline(0.0, color="#666666", linewidth=0.8)
    ax_delta.set_title("A. R2 effect by pooling")
    ax_delta.set_ylabel("Delta(R2-R0) AUROC")
    ax_delta.set_xticks(range(len(labels)), labels)
    ax_sensitivity.set_title("B. Pooling sensitivity")
    ax_sensitivity.set_ylabel("P(K) AUROC contrast")
    ax_sensitivity.set_xlabel("Final-token window K")
    ax_sensitivity.set_xticks(WINDOWS)
    for axis in (ax_delta, ax_sensitivity):
        axis.spines[["top", "right"]].set_visible(False)
    ax_delta.legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return _exclusive_save(fig, out_path)


def make_confirmation_figure(discovery, confirmation, out_path):
    """Compare the frozen discovery endpoint with its held-out estimate."""
    _validate_confirmation(discovery, confirmation)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    selected_k = discovery["selection"]["selected_k"]
    discovery_endpoint = _window(
        discovery["window_endpoints"]["Meta-Llama-3.1-8B-Instruct"],
        selected_k,
    )["r2"]
    records = [("Discovery", discovery_endpoint, MODEL_COLORS["Meta-Llama-3.1-8B-Instruct"])]
    if "primary_endpoint" in confirmation:
        records.append(
            (
                "Held-out",
                confirmation["primary_endpoint"],
                MODEL_COLORS["Llama-3.2-3B-Instruct"],
            )
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for index, (label, record, color) in enumerate(records):
        estimate = float(record["estimate"])
        low, high = map(float, record["ci95"])
        ax.errorbar(
            index,
            estimate,
            yerr=[[estimate - low], [high - estimate]],
            color=color,
            marker="o",
            capsize=4,
            markersize=7,
        )
    ax.set_xticks(range(len(records)), [record[0] for record in records])
    ax.set_ylabel(f"Frozen P(K={selected_k}) AUROC contrast")
    ax.set_title("Pooling-mechanism endpoint comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return _exclusive_save(fig, out_path)


def _load_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discovery_parser = subparsers.add_parser("discovery")
    discovery_parser.add_argument("--result", required=True)
    discovery_parser.add_argument("--out", required=True)
    confirmation_parser = subparsers.add_parser("confirmation")
    confirmation_parser.add_argument("--discovery", required=True)
    confirmation_parser.add_argument("--confirmation", required=True)
    confirmation_parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "discovery":
        return make_discovery_figure(_load_json(args.result), args.out)
    return make_confirmation_figure(
        _load_json(args.discovery),
        _load_json(args.confirmation),
        args.out,
    )


if __name__ == "__main__":
    main()
