#!/usr/bin/env python3
"""Publication figures for the 04 post (distinct from research heatmaps).

Design decisions, all deliberate:
- Diverging colormap CENTERED AT 0.5 (chance). AUROC is not a 0-1 quantity
  where 0 is "bad" — 0.5 is the null and 0.2 is as informative as 0.8, in the
  opposite direction. A sequential/RdYlGn 0-1 map renders chance as a mid
  color and hides exactly the contrast this post is about.
- Blue/red diverging (RdBu_r), not red/green: ~8% of men cannot separate
  red from green.
- Opaque white background: LessWrong renders dark mode; a transparent PNG
  leaves black axis text on a dark page.
- Fonts sized for ~700px display width on LW/Substack, not for a paper column.

Usage: PYTHONPATH=. python -m p04.figures_post
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RESULTS = Path(__file__).parent.parent / "results"
FIGS = RESULTS / "figures-post"

# Display order: smallest to largest, so size trends read left-to-right
MODEL_ORDER = [
    "Llama-3.2-1B-Instruct", "gemma-2-2b-it", "Llama-3.2-3B-Instruct",
    "gemma-2-9b-it", "Meta-Llama-3.1-8B-Instruct",
]
MODEL_LABEL = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "gemma-2-2b-it": "Gemma-2-2B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "gemma-2-9b-it": "Gemma-2-9B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
DS_ORDER = ["roleplaying", "insider_trading", "sandbagging"]
DS_LABEL = {"roleplaying": "roleplaying", "insider_trading": "insider trading",
            "sandbagging": "sandbagging"}

NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
CMAP = "RdBu_r"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
})


def _load(name):
    return json.loads((RESULTS / name).read_text())


def fig1_collapse(xd, ceilings, out):
    """Money figure: same-dataset ceiling vs cross-dataset transfer, per model.

    One row per model; ceiling as a filled marker, each of the six transfer
    cells as a dot; chance line at 0.5. The inverted direction is called out
    by color so the reader sees the finding without reading a matrix.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = np.arange(len(MODEL_ORDER))[::-1]

    for y, m in zip(ys, MODEL_ORDER):
        rows = [r for r in xd if r["model"] == m]
        off = [r for r in rows if r["fit_dataset"] != r["eval_dataset"]]
        inv = [r for r in off
               if r["fit_dataset"] == "roleplaying"
               and r["eval_dataset"] == "sandbagging"]
        other = [r for r in off if r not in inv]
        ceil = ceilings[m]["auroc"]

        ax.plot([r["auroc"] for r in other], [y] * len(other), "o",
                color="#7f8c9b", ms=7, alpha=0.85,
                label="cross-dataset transfer" if y == ys[0] else None)
        for r in inv:
            ax.errorbar(r["auroc"], y, xerr=1.96 * r["auroc_se"], fmt="o",
                        color="#c0392b", ms=9, capsize=3, zorder=3,
                        label="roleplaying → sandbagging"
                        if y == ys[0] else None)
        ax.plot(ceil, y, "D", color="#1f4e79", ms=9, zorder=3,
                label="same-dataset ceiling" if y == ys[0] else None)

    ax.axvline(0.5, color="k", lw=1.2, ls="--", zorder=1)
    ax.text(0.5, ys[-1] - 0.75, "chance", fontsize=10, ha="center", va="top")
    ax.set_yticks(ys, [MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ys[-1] - 1.15, ys[0] + 0.6)
    ax.set_xlabel("AUROC")
    ax.set_title("Deception probes: within-dataset skill does not transfer\n"
                 "between deception datasets at 1B–9B", loc="left", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _matrix_panel(ax, mat, xlabels, ylabels, title, se=None):
    ax.imshow(mat, norm=NORM, cmap=CMAP)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right",
                  fontsize=9)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            # white text on saturated cells, black on pale ones
            col = "white" if abs(v - 0.5) > 0.28 else "black"
            txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=col, fontweight="bold" if abs(v - 0.5) > 0.28 else "normal")


def fig2_cross_dataset(xd, out):
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(15, 3.6))
    for ax, m in zip(axes, MODEL_ORDER):
        mat = np.full((len(DS_ORDER), len(DS_ORDER)), np.nan)
        for r in xd:
            if r["model"] == m:
                mat[DS_ORDER.index(r["fit_dataset"]),
                    DS_ORDER.index(r["eval_dataset"])] = r["auroc"]
        _matrix_panel(ax, mat, [DS_LABEL[d] for d in DS_ORDER],
                      [DS_LABEL[d] for d in DS_ORDER] if m == MODEL_ORDER[0]
                      else [""] * len(DS_ORDER), MODEL_LABEL[m])
    axes[0].set_ylabel("probe fit on", fontsize=10)
    fig.suptitle("Cross-dataset transfer (row = fit, column = evaluated on). "
                 "White = chance; red = better than chance, blue = inverted.",
                 fontsize=12, y=1.06)
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("AUROC", fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_released(rel, out):
    probes = sorted({r["probe"] for r in rel})
    fig, axes = plt.subplots(1, len(probes), figsize=(14, 3.6))
    for ax, p in zip(axes, probes):
        mat = np.full((len(MODEL_ORDER), len(DS_ORDER)), np.nan)
        for r in rel:
            if r["probe"] == p:
                mat[MODEL_ORDER.index(r["model"]),
                    DS_ORDER.index(r["dataset"])] = r["auroc"]
        _matrix_panel(ax, mat, [DS_LABEL[d] for d in DS_ORDER],
                      [MODEL_LABEL[m] for m in MODEL_ORDER]
                      if p == probes[0] else [""] * len(MODEL_ORDER),
                      f"released probe: {p}")
    fig.suptitle("Apollo's released 70B probes applied to smaller models "
                 "(no refit, coordinate-restricted): chance everywhere",
                 fontsize=12, y=1.06)
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("AUROC", fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    xd = _load("cross_dataset_matrix.json")
    rel = _load("released_on_small_models.json")
    ceilings = _load("gate_ceilings.json")
    outs = [
        fig1_collapse(xd, ceilings, FIGS / "fig1_transfer_collapse.png"),
        fig2_cross_dataset(xd, FIGS / "fig2_cross_dataset.png"),
        fig3_released(rel, FIGS / "fig3_released_probes.png"),
    ]
    for o in outs:
        print(o)


if __name__ == "__main__":
    main()
