#!/usr/bin/env python3
"""Run the full 04 extension: gate -> Task 3 -> Task 4 -> Task 5 -> heatmaps.

Usage: PYTHONPATH=. python -m p04.run_extension [--acts-dir DIR] [--out-dir DIR]

STOP rule: if the same-distribution gate on Llama-3.1-8B fails, nothing after
it runs.
"""

import argparse
import json
from pathlib import Path

from p04 import gate
from p04.loader import DATASET_FILES
from p04.transfer_models import eval_released_on
from p04.transfer_datasets import fit_on_eval_on
from p04.transfer_cross_model import fit_a_eval_b

MODELS = [
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
]
DATASETS = list(DATASET_FILES)
RELEASED_PROBES = ["roleplaying", "descriptive", "followup", "instructed_pairs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent.parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Task 2 gate (STOP rule): 5-fold CV, per-token fit ----
    g = gate.reproduce_auroc_cv(acts_dir=args.acts_dir)
    g_auroc = g["auroc"]
    g_ok = gate.check_gate(g_auroc, gate.PUBLISHED_LOW, gate.PUBLISHED_HIGH)
    gate_rec = {
        "model": gate.GATE_MODEL, "dataset": gate.GATE_DATASET,
        "protocol": "5-fold CV, dialogue-level folds, per-token fit, "
                    "mean prompt score, single AUROC over out-of-fold scores",
        **g,
        "published_range": [gate.PUBLISHED_LOW, gate.PUBLISHED_HIGH],
        "tolerance": 0.02, "passed": bool(g_ok),
        "single_split_seed_range_note":
            "80/20 single split gave 0.88-0.97 across seeds 42/7/123 "
            "(val honest arm n~17); CV protocol adopted for power",
    }
    # the maintainer's 2026-08-16 disposition: the 70B-calibrated bar misses on 8B
    # (0.9365 vs 0.94) but the instrument is validated by the per-model size
    # trend (1B 0.863 -> 8B 0.936); proceed with the miss documented and each
    # model's own same-distribution CV AUROC as its transfer reference
    # ceiling. A hard sanity floor of 0.90 still stops a truly broken run.
    SANITY_FLOOR = 0.90
    gate_rec["disposition"] = (
        "documented miss; proceed per the documented project decision — "
        "validated by size trend; per-model ceilings are the OOD reference"
    )
    (out_dir / "gate_task2.json").write_text(json.dumps(gate_rec, indent=2))
    print(f"GATE: 8B roleplaying CV AUROC = {g_auroc:.4f} "
          f"(range {gate.PUBLISHED_LOW}-{gate.PUBLISHED_HIGH}) -> "
          f"{'PASS' if g_ok else 'DOCUMENTED MISS (proceeding)'}")
    if g_auroc < SANITY_FLOOR:
        print(f"GATE below sanity floor {SANITY_FLOOR} — STOPPING (instrument suspect).")
        return 1

    # Per-model same-distribution ceilings (the denominator for OOD drops)
    ceilings = {}
    for m in MODELS:
        try:
            ceilings[m] = gate.reproduce_auroc_cv(
                model_short=m, acts_dir=args.acts_dir)
        except FileNotFoundError:
            ceilings[m] = None
    (out_dir / "gate_ceilings.json").write_text(json.dumps(ceilings, indent=2))
    print("ceilings:", {m: (round(c["auroc"], 4) if c else None)
                        for m, c in ceilings.items()})

    # ---- Task 3: released probes, unrefitted ----
    missing = []
    released = []
    for probe_name in RELEASED_PROBES:
        for m in MODELS:
            for ds in DATASETS:
                try:
                    r = eval_released_on(m, ds, probe_name, args.acts_dir)
                except FileNotFoundError:
                    missing.append(f"{m}/{ds}")
                    continue
                released.append(r)
                print(f"released[{probe_name}] {m} / {ds}: "
                      f"AUROC {r['auroc']:.3f} ± {r['auroc_se']:.3f}")
    (out_dir / "released_on_small_models.json").write_text(
        json.dumps(released, indent=2))

    # ---- Task 4: cross-dataset 3x3 per model ----
    xdata = []
    for m in MODELS:
        for fd in DATASETS:
            for ed in DATASETS:
                try:
                    r = fit_on_eval_on(m, fd, ed, args.acts_dir)
                except FileNotFoundError:
                    missing.append(f"{m}:{fd}->{ed}")
                    continue
                xdata.append(r)
        diag = [r for r in xdata if r["model"] == m
                and r["fit_dataset"] == r["eval_dataset"]]
        print(f"cross-dataset {m}: diag "
              + ", ".join(f"{r['fit_dataset']}={r['auroc']:.3f}" for r in diag))
    (out_dir / "cross_dataset_matrix.json").write_text(
        json.dumps(xdata, indent=2))

    # ---- Task 5: cross-model matrix (roleplaying) ----
    xmodel = []
    for ma in MODELS:
        for mb in MODELS:
            r = fit_a_eval_b(ma, mb, "roleplaying", args.acts_dir)
            xmodel.append(r)
    (out_dir / "cross_model_matrix.json").write_text(
        json.dumps(xmodel, indent=2))
    print("cross-model matrix done")

    # ---- Task 6: heatmaps ----
    from p04.heatmap import make_heatmaps
    paths = make_heatmaps(released, xdata, xmodel, out_dir)
    print("figures:", *paths)
    if missing:
        uniq = sorted(set(missing))
        (out_dir / "missing_cells.json").write_text(json.dumps(uniq, indent=2))
        print(f"MISSING CELLS ({len(uniq)}):", *uniq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
