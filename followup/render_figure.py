"""Grouped bars from frozen results; run with uv run --with matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "frozen-analysis/pooling_confirmation.public.json"
data = json.loads(SOURCE.read_text())
primary, r1 = data["primary_endpoint"], data["secondary"]["r1"]
assert r1["common_ids_sha256"] == primary["common_ids_sha256"]
assert r1["n_common"] == primary["n_common"] == 912
assert r1["inference"] == "secondary_point_diagnostic"
assert r1["coverage"]["paired"]["k"] == 4
p, s = primary["cell_aurocs"], r1["cell_aurocs"]
assert s["full_r0"] == p["full_r0"] and s["window_r0"] == p["window_r0"]
OUT = ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 4.9))
fig.subplots_adjust(left=.115, right=.96, bottom=.23, top=.82)
fig.text(.115, .935, "Llama-3.2-3B-Instruct · 912 paired examples", fontsize=11)
ax.set(xlim=(-.55, 1.55), ylim=(0, 1), ylabel="AUROC (source labels)",
       xticks=[0, 1], xticklabels=["Full-response pooling", "Final-four-token pooling"],
       yticks=[0, .25, .5, .75, 1])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
ax.spines[["top", "right"]].set_visible(False)
for key in ("left", "bottom"):
    ax.spines[key].set_color("#888888")
    ax.spines[key].set_linewidth(.7)
ax.tick_params(axis="both", length=3, width=.7, color="#888888", pad=7)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#EEEEEE", linewidth=.6)
for tick, gridline in zip(ax.get_yticks(), ax.get_ygridlines()):
    if tick == .5:
        gridline.set_visible(False)
series = [
    ("Original response", "#596A78", -.24, [p["full_r0"], p["window_r0"]]),
    ("Tags removed", "#BAC5CC", 0, [s["full_treatment"], s["window_treatment"]]),
    ("Fixed template", "#4D83A1", .24, [p["full_treatment"], p["window_treatment"]]),
]
for label, color, offset, values in series:
    bars = ax.bar([offset, 1+offset], values, width=.21, color=color,
                  edgecolor="#FFFFFF", linewidth=.4, label=label, zorder=2)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.6))
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.102,.902),
           ncol=3, frameon=False, fontsize=9.5, handlelength=1.3, columnspacing=1.6)
lo, hi = primary["ci95"]
fig.text(.115, .10,
         f"Difference in AUROC changes: {primary['estimate']:.3f}  (95% paired-bootstrap CI: {lo:.3f}–{hi:.3f})",
         fontsize=9, color="#333333")
fig.text(.115, .055, "Tags removed is a secondary diagnostic.", fontsize=9, color="#555555")
for extension in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"p04-pooling-followup.{extension}", dpi=240, facecolor="white")
print(json.dumps({"source": str(SOURCE.relative_to(ROOT)), "cohort_identity_matches": True,
                  "values": {a:v for a,_,_,v in series}, "primary_interaction": primary["estimate"]}))
