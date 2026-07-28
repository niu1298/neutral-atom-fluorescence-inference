"""Shared plotting style and figure helpers.

Kept separate from the analysis so that figures can be restyled without
touching any number, and so the README asset generator and the QC report look
like one project.

Two rules the whole project follows and this module enforces mechanically:

* No figure title, annotation or filename may contain an absolute path, an
  account name, a machine name or an internal run identifier.
* Any panel not derived from the measured data must carry a visible
  ``SYNTHETIC DEMONSTRATION`` banner.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ------------------------------------------------------------------ palette
INK = "#12161c"
MUTED = "#6b7684"
GRID = "#dfe3e8"
ACCENT = "#2f6f9f"          # frame 0 / primary
ACCENT_2 = "#c9622b"        # frame 1 / secondary
OK = "#2e7d5b"
WARN = "#b3452c"
PANEL = "#ffffff"
SOFT = "#f4f6f8"

FRAME_COLORS = {0: ACCENT, 1: ACCENT_2}
SEQ_CMAP = "magma"
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "fi_div", ["#2f6f9f", "#eef1f4", "#c9622b"])

#: patterns that must never appear in a public asset.
#: The drive-letter pattern needs the lookbehind: without it, `https://` matches
#: (the "s" before "://" reads as a drive letter) and every URL is rejected.
FORBIDDEN = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),       # windows absolute path
    re.compile(r"/(?:home|Users|mnt)/"),                 # posix absolute path
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),          # email address
    re.compile(r"\bmit\.edu\b", re.I),
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.size": 10.5,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.7,
    })


def assert_public_safe(text: str, where: str = "") -> None:
    """Raise when a caption or title would leak private information."""
    for pat in FORBIDDEN:
        if pat.search(text):
            raise ValueError(
                f"refusing to render text that matches {pat.pattern!r}"
                + (f" in {where}" if where else "")
            )


def synthetic_banner(ax: plt.Axes, text: str = "SYNTHETIC DEMONSTRATION") -> None:
    ax.text(0.5, 0.985, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.35", fc=WARN, ec="none", alpha=0.95),
            zorder=50)


def provisional_note(fig: plt.Figure, text: str) -> None:
    """Small footnote that keeps a provisional claim visible on the figure.

    Placed just below the figure box; ``savefig(bbox_inches="tight")`` grows the
    canvas to include it, so it never collides with an axis label.
    """
    assert_public_safe(text, "provisional note")
    fig.text(0.005, -0.015, text, fontsize=8.2, color=MUTED, ha="left", va="top")


def show_image(ax: plt.Axes, img: np.ndarray, *, percentile=(1.0, 99.5),
               cmap: str = SEQ_CMAP, extent: Iterable[float] | None = None,
               vlim: tuple[float, float] | None = None):
    lo, hi = vlim if vlim is not None else np.percentile(img, percentile)
    return ax.imshow(np.asarray(img), vmin=lo, vmax=hi, cmap=cmap,
                     extent=list(extent) if extent is not None else None,
                     interpolation="nearest", origin="upper")


def draw_roi_boxes(ax: plt.Axes, boxes, *, color: str = "#57d0ff",
                   lw: float = 0.7, labels: dict[int, str] | None = None,
                   label_color: str = "#ffffff", fontsize: float = 6.0) -> None:
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                   fill=False, ec=color, lw=lw))
        if labels and i in labels:
            ax.text(x1 - 0.2, y0 - 0.8, labels[i], color=label_color,
                    fontsize=fontsize, ha="left", va="bottom")


def colorbar(fig: plt.Figure, mappable, ax, label: str, **kw):
    cb = fig.colorbar(mappable, ax=ax, fraction=kw.pop("fraction", 0.046),
                      pad=kw.pop("pad", 0.03), **kw)
    cb.set_label(label, fontsize=9.5, color=INK)
    cb.ax.tick_params(labelsize=8.5, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(MUTED)
    return cb


def save(fig: plt.Figure, path: Path, *, dpi: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or plt.rcParams["savefig.dpi"])
    plt.close(fig)
    return path


def fmt_count(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}"


def ci_string(point: float, lo: float, hi: float, *, pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return f"{point * scale:.2f}{unit} [{lo * scale:.2f}, {hi * scale:.2f}]"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])
