"""Task 6 — heatmap figures for the transfer results."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _grid(ax, mat, xlabels, ylabels, title):
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6)
    return im


def make_heatmaps(released, xdata, xmodel, out_dir):
    models = sorted({r["model"] for r in released})
    datasets = sorted({r["dataset"] for r in released})
    paths = []

    # Released probes: one panel per probe, models x datasets
    probes = sorted({r["probe"] for r in released})
    fig, axes = plt.subplots(1, len(probes), figsize=(4 * len(probes), 3.2))
    for ax, p in zip(np.atleast_1d(axes), probes):
        mat = np.full((len(models), len(datasets)), np.nan)
        for r in released:
            if r["probe"] == p:
                mat[models.index(r["model"]), datasets.index(r["dataset"])] = r["auroc"]
        _grid(ax, mat, datasets, models, f"released {p} (unrefit, coord-restricted)")
    fig.suptitle("Apollo released probes on small models — AUROC (chance 0.5)")
    fig.tight_layout()
    p1 = out_dir / "heatmap_released.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    paths.append(str(p1))

    # Cross-dataset: one 3x3 per model
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2))
    for ax, m in zip(np.atleast_1d(axes), models):
        mat = np.full((len(datasets), len(datasets)), np.nan)
        for r in xdata:
            if r["model"] == m:
                mat[datasets.index(r["fit_dataset"]),
                    datasets.index(r["eval_dataset"])] = r["auroc"]
        _grid(ax, mat, datasets, datasets, m)
    fig.suptitle("Cross-dataset transfer (fit row -> eval col), refit probes")
    fig.tight_layout()
    p2 = out_dir / "heatmap_cross_dataset.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    paths.append(str(p2))

    # Cross-model matrix
    mm = sorted({r["fit_model"] for r in xmodel})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    mat = np.full((len(mm), len(mm)), np.nan)
    for r in xmodel:
        mat[mm.index(r["fit_model"]), mm.index(r["eval_model"])] = r["auroc"]
    _grid(ax, mat, mm, mm, "cross-model (roleplaying, coord-restricted off-diag)")
    fig.tight_layout()
    p3 = out_dir / "heatmap_cross_model.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    paths.append(str(p3))

    plt.close("all")
    return paths
