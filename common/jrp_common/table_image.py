"""Render a results table as a PNG.

Why this exists: Substack has no table support and strips raw <table> HTML
during sanitization, so a table-heavy post needs its tables as images there.
LessWrong has good native tables, so on LW use the real table and keep this
for Substack (or use the image on both to keep one asset in sync — the
tradeoff is losing selectable/accessible text on LW).

Style matches p04/figures_post.py: off-white background rather than pure
white, because LessWrong renders dark mode and does not adapt raster images;
a pure-white block is a glaring rectangle on a dark page.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FAFAFA"
HEADER_BG = "#E8ECF0"
GRID = "#C8D0D8"


def render_table(rows, headers, out_path, title=None, footnote=None,
                 col_widths=None, highlight_rows=(), highlight_cols=(),
                 align_left_first=True):
    """rows: list[list[str]] (pre-formatted strings — this does no rounding).

    highlight_rows / highlight_cols: indices to emphasise (bold + tinted).
    Column indices count the first column, so col 0 is the row-label column.
    """
    ncol = len(headers)
    nrow = len(rows)
    # Sized so the table fills the canvas; matplotlib tables otherwise float
    # in a large empty axes and the export is mostly whitespace.
    fig_w = min(1.6 + 1.25 * ncol, 16)
    fig_h = 0.42 * (nrow + 1) + (0.45 if title else 0.05)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)

    tbl = ax.table(
        cellText=rows, colLabels=headers, loc="center",
        cellLoc="center", colWidths=col_widths, bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        hot = (r - 1) in highlight_rows or c in highlight_cols
        if r == 0:
            cell.set_facecolor("#DCE6EE" if c in highlight_cols else HEADER_BG)
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FFF1E0" if hot else BG)
            if hot:
                cell.set_text_props(fontweight="bold")
        if c == 0 and align_left_first:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left",
                     y=1.0 + (0.40 / fig_h))
    if footnote:
        fig.text(0.01, -0.10 / fig_h, footnote, fontsize=9, va="top",
                 color="#444")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG,
                pad_inches=0.28)
    plt.close(fig)
    return out_path
