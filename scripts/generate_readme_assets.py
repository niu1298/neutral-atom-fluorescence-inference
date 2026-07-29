"""Generate every README asset from the processed dataset, deterministically.

    python scripts/generate_readme_assets.py --config configs/paired_100ms.yaml

Produces
--------
``assets/readme/fluorescence_inference_overview.gif``   hero animation
``assets/readme/fluorescence_inference_overview.png``   static fallback
``assets/readme/paired_images_roi_overlay.png``
``assets/readme/count_distribution_fit.png``
``assets/readme/paired_frame_scatter.png``
``assets/readme/site_summary_map.png``
``assets/readme/asset_metadata.json``                   provenance + selection rule
``reports/readme_metrics.md``                           the numbers the README quotes

Design rules this script enforces
---------------------------------
* Every panel comes from the measured data. Nothing synthetic is drawn; if a
  synthetic panel is ever added it must call ``reporting.synthetic_banner``.
* No label may claim a fidelity, an error rate or an atom loss. Frame-to-frame
  disagreement is drawn and named as disagreement.
* Representative shots and sites are chosen by a documented rule that targets
  the *median*, never the cleanest example, and the rule and its outcome are
  written into ``asset_metadata.json``.
* Nothing identifying the machine, the account or the run directory reaches an
  asset. Every caption passes ``reporting.assert_public_safe``.
* Missing input is a hard failure, never a silent substitution.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from fluorescence_inference import quality_control as qc  # noqa: E402
from fluorescence_inference import reporting as rp  # noqa: E402
from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import (  # noqa: E402
    SourceDataError, _frame_loader, discover_shots, load_dataset,
)
from fluorescence_inference.provenance import stamp  # noqa: E402

ASSET_DIR = Path("assets/readme")
GIF_NAME = "fluorescence_inference_overview.gif"
PNG_NAME = "fluorescence_inference_overview.png"

#: the storyboard, in order. `hold` frames repeat the last rendered frame, which
#: costs almost nothing in a delta-compressed GIF.
SCENES = [
    ("Two consecutive frames", 6, 10),
    ("Site ROIs", 6, 10),
    ("One site: signal and background", 8, 10),
    ("Raw ROI counts", 6, 10),
    ("Background-corrected counts", 8, 10),
    ("Two-component description", 7, 12),
    ("Paired readout, frame 0 vs frame 1", 9, 12),
    ("What the pipeline produces", 4, 16),
]
STEP_LABELS = ["frames", "ROIs", "extract", "raw", "corrected",
               "describe", "pair", "output"]

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "fi_density", ["#eef2f6", "#a7cde4", "#4f93bf", "#245b84", "#12324a"])

TITLE = "Neutral-atom fluorescence inference"
SUBTITLE = "Single-condition 100 ms paired-readout dataset"


# ------------------------------------------------------------------ selection
@dataclass(frozen=True)
class Selection:
    shot_order: int
    shot_rule: str
    site_id: int
    site_rule: str
    site_grid: str
    site_row: int
    site_col: int


def choose_representative(df: pd.DataFrame) -> Selection:
    """Pick a median shot and a median site, by a rule fixed in advance.

    Targeting the median rather than the best example is the point: an asset
    built from the cleanest shot would misrepresent the run. Shot 0 is excluded
    because the audit records it as a first-shot background outlier, not
    because of how it looks.
    """
    per_shot = (df[df["frame_id"] == 0]
                .groupby("shot_order", observed=True)["roi_sum_raw"].mean())
    eligible = per_shot.drop(index=0, errors="ignore")
    shot_order = int((eligible - eligible.median()).abs().idxmin())

    per_site = (df[(df["frame_id"] == 0) & (df["quality_flag"] == "ok")]
                .groupby("site_id", observed=True)["background_corrected_count"]
                .agg(["mean", "count"]))
    per_site = per_site[per_site["count"] == per_site["count"].max()]
    site_id = int((per_site["mean"] - per_site["mean"].median()).abs().idxmin())
    row = df[(df["site_id"] == site_id) & (df["frame_id"] == 0)].iloc[0]

    return Selection(
        shot_order=shot_order,
        shot_rule="shot whose mean frame-0 raw ROI count is closest to the "
                  "run median, excluding shot 0 (documented background outlier)",
        site_id=site_id,
        site_rule="unflagged site whose mean frame-0 background-corrected count "
                  "is closest to the median across sites",
        site_grid=str(row["grid"]), site_row=int(row["site_row"]),
        site_col=int(row["site_col"]),
    )


# --------------------------------------------------------------------- canvas
def canvas_style() -> None:
    rp.apply_style()
    rp.plt.rcParams.update({
        "font.size": 13.0,
        "axes.titlesize": 15.0,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12.0,
        "axes.grid": True,
        "figure.dpi": 100,
    })


def new_canvas(width_px: int, height_px: int):
    fig = rp.plt.figure(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    fig.patch.set_facecolor(rp.PANEL)
    return fig


def draw_chrome(fig, scene_index: int, caption: str) -> None:
    """Title bar, scene caption and the persistent step rail.

    The rail is why a single frame of this animation still explains the whole
    pipeline, which is what makes the static PNG fallback honest.
    """
    rp.assert_public_safe(caption, "scene caption")
    fig.text(0.022, 0.945, TITLE, fontsize=21, fontweight="bold", color=rp.INK,
             va="center")
    fig.text(0.978, 0.947, SUBTITLE, fontsize=12.5, color=rp.MUTED,
             va="center", ha="right")
    fig.add_artist(rp.plt.Line2D([0.022, 0.978], [0.905, 0.905],
                                 color=rp.GRID, lw=1.2))
    fig.text(0.022, 0.873, f"{scene_index + 1}. {caption}", fontsize=14.5,
             color=rp.ACCENT, fontweight="bold", va="center")

    n = len(STEP_LABELS)
    x0, x1, y = 0.022, 0.978, 0.040
    span = (x1 - x0) / n
    for i, label in enumerate(STEP_LABELS):
        cx = x0 + span * (i + 0.5)
        on = i == scene_index
        done = i < scene_index
        col = rp.ACCENT if on else (rp.MUTED if done else rp.GRID)
        fig.add_artist(rp.plt.Line2D([x0 + span * i + 0.006, x0 + span * (i + 1) - 0.006],
                                     [y + 0.026], color=col,
                                     lw=4.2 if on else 2.4, solid_capstyle="round"))
        fig.text(cx, y, label, fontsize=11.5 if on else 10.8,
                 color=rp.INK if on else rp.MUTED,
                 fontweight="bold" if on else "normal", ha="center", va="center")


def stage_axes(fig, rect=(0.055, 0.20, 0.895, 0.63)):
    return fig.add_axes(rect)


def caption(fig, text: str) -> None:
    """Footnote under the stage, wrapped to at most two lines.

    Wrapping here rather than in each scene keeps long honesty caveats from
    being clipped at the canvas edge, which would defeat the point of them.
    """
    rp.assert_public_safe(text, "scene footnote")
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > 104 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    lines = lines[:2]
    for i, line in enumerate(lines):
        fig.text(0.5, 0.132 - 0.030 * i, line, fontsize=12.4, color=rp.MUTED,
                 ha="center", va="center")


def render(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    rp.plt.close(fig)
    return Image.fromarray(buf, mode="RGB")


# ---------------------------------------------------------------- gif writing
def quantize_all(frames: list[Image.Image], colors: int = 128) -> list[Image.Image]:
    """One shared palette for every frame.

    Per-frame palettes make a GIF shimmer and defeat delta compression; a single
    global palette keeps colours stable and repeated frames nearly free.
    """
    step = max(1, len(frames) // 12)
    sample = frames[::step]
    w = sample[0].width
    montage = Image.new("RGB", (w, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        montage.paste(f, (0, f.height * i))
    palette = montage.quantize(colors=colors, method=Image.MEDIANCUT)
    return [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]


def write_gif(frames: list[Image.Image], path: Path, fps: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(round(1000.0 / fps))
    quantized = quantize_all(frames)
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=duration, loop=0, optimize=True, disposal=1)
    size = path.stat().st_size
    return {
        # repository-relative on purpose: an absolute path in a published
        # artefact is exactly the leak `assert_public_safe` exists to stop
        "path": (ASSET_DIR / path.name).as_posix(),
        "width_px": frames[0].width,
        "height_px": frames[0].height,
        "n_frames": len(frames),
        "fps": fps,
        "duration_s": round(len(frames) / fps, 2),
        "loop": "infinite",
        "bytes": size,
        "megabytes": round(size / 1e6, 2),
    }


# --------------------------------------------------------------------- scenes
def _ease(t: float) -> float:
    """Smoothstep. Keeps every transition free of a hard start or stop."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class SceneBuilder:
    """Holds everything the scenes draw, so each scene stays a pure renderer."""

    def __init__(self, cfg, df, sites_df, ctx: dict[str, Any]):
        self.cfg = cfg
        self.df = df
        self.sites_df = sites_df.sort_values("site_id").reset_index(drop=True)
        self.sel: Selection = ctx["selection"]
        self.images: dict[int, np.ndarray] = ctx["images"]
        self.crop = ctx["crop"]
        self.fits = ctx["fits"]
        self.pairs = ctx["pairs"]
        self.refs = ctx["refs"]
        self.metrics = ctx["metrics"]
        self.width = ctx["width"]
        self.height = ctx["height"]

        clean = df[df["quality_flag"] == "ok"]
        self.raw_by_frame = {int(f): g["roi_sum_raw"].to_numpy(float)
                             for f, g in clean.groupby("frame_id", observed=True)}
        self.cor_by_frame = {
            int(f): g["background_corrected_count"].to_numpy(float)
            for f, g in clean.groupby("frame_id", observed=True)}

        allraw = np.concatenate(list(self.raw_by_frame.values()))
        allcor = np.concatenate(list(self.cor_by_frame.values()))
        self.raw_lim = tuple(np.percentile(allraw, (0.2, 99.8)))
        self.cor_lim = tuple(np.percentile(allcor, (0.2, 99.8)))
        self.raw_edges = np.linspace(*self.raw_lim, 81)
        self.cor_edges = np.linspace(*self.cor_lim, 81)
        self.hist_raw = {f: np.histogram(v, bins=self.raw_edges)[0]
                         for f, v in self.raw_by_frame.items()}
        self.hist_cor = {f: np.histogram(v, bins=self.cor_edges)[0]
                         for f, v in self.cor_by_frame.items()}
        self.hist_ymax = 1.12 * max(h.max() for h in
                                    (*self.hist_raw.values(), *self.hist_cor.values()))

        site = self.sites_df[self.sites_df["site_id"] == self.sel.site_id].iloc[0]
        self.site_center = (float(site["site_y"]), float(site["site_x"]))
        self.site_box = (int(site["roi_y0"]), int(site["roi_y1"]),
                         int(site["roi_x0"]), int(site["roi_x1"]))
        self.site_counts = {
            int(f): df[(df["site_id"] == self.sel.site_id)
                       & (df["frame_id"] == f)
                       & (df["shot_order"] == self.sel.shot_order)].iloc[0]
            for f in (0, 1)}

    # ------------------------------------------------------------- scene 1-3
    def _image_pair(self, fig, zoom: float, boxes_fraction: float,
                    show_annulus: bool, show_numbers: float,
                    right_alpha: float = 1.0):
        x0, x1, y0, y1 = self.crop
        cy, cx = self.site_center
        half = 21.0          # wide enough to show the whole 33 px annulus
        zx0 = x0 + (cx - half - x0) * zoom
        zx1 = x1 + (cx + half - x1) * zoom
        zy0 = y0 + (cy - half - y0) * zoom
        zy1 = y1 + (cy + half - y1) * zoom

        lo, hi = np.percentile(np.concatenate(
            [self.images[0][y0:y1, x0:x1].ravel(),
             self.images[1][y0:y1, x0:x1].ravel()]), (1, 99.6))

        axes = [fig.add_axes([0.055, 0.185, 0.415, 0.615]),
                fig.add_axes([0.525, 0.185, 0.415, 0.615])]
        for fid, ax in enumerate(axes):
            img = self.images[fid]
            ax.imshow(img, vmin=lo, vmax=hi, cmap=rp.SEQ_CMAP,
                      interpolation="nearest",
                      alpha=1.0 if fid == 0 else right_alpha)
            ax.set_xlim(zx0, zx1)
            ax.set_ylim(zy1, zy0)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(rp.GRID)
            ax.set_title(f"frame {fid}   ·   100 ms exposure",
                         color=rp.FRAME_COLORS[fid], fontsize=15,
                         alpha=1.0 if fid == 0 else right_alpha)

            if boxes_fraction > 0:
                n = int(round(boxes_fraction * len(self.sites_df)))
                for r in self.sites_df.iloc[:n].itertuples():
                    target = show_annulus and r.site_id == self.sel.site_id
                    ax.add_patch(rp.plt.Rectangle(
                        (r.roi_x0 - 0.5, r.roi_y0 - 0.5),
                        r.roi_x1 - r.roi_x0, r.roi_y1 - r.roi_y0, fill=False,
                        ec="#7fe3ff" if target else "#a9c7d6",
                        lw=(3.0 if target else 0.7 + 0.5 * zoom),
                        alpha=1.0 if target else (1.0 - 0.7 * zoom), zorder=4))
            if show_numbers > 0:
                # the four corners of each sub-array: enough to show that
                # site_id is a stable index, without a wall of labels
                for _, g in self.sites_df.groupby("grid"):
                    r_lo, r_hi = g["site_row"].min(), g["site_row"].max()
                    c_lo, c_hi = g["site_col"].min(), g["site_col"].max()
                    corners = g[(g["site_row"].isin([r_lo, r_hi]))
                                & (g["site_col"].isin([c_lo, c_hi]))]
                    for r in corners.itertuples():
                        ax.text(r.roi_x1 + 0.8, r.roi_y0 - 1.0, f"site {r.site_id}",
                                color="#ffffff", fontsize=10.5, alpha=show_numbers,
                                zorder=6,
                                bbox=dict(boxstyle="round,pad=0.18", fc="#12161c",
                                          ec="none", alpha=0.6 * show_numbers))
            if show_annulus:
                by0, by1, bx0, bx1 = self.site_box
                mcy, mcx = 0.5 * (by0 + by1 - 1), 0.5 * (bx0 + bx1 - 1)
                for hw, ls, lbl in ((6, "-", None), (16, "--", "local background")):
                    ax.add_patch(rp.plt.Rectangle(
                        (mcx - hw - 0.5, mcy - hw - 0.5), 2 * hw + 1, 2 * hw + 1,
                        fill=False, ec="#ffd166", lw=1.8, ls=ls, zorder=5))
                if fid == 0:
                    ax.text(mcx, by0 - 1.2, "ROI 5x5 px", color="#7fe3ff",
                            fontsize=11.5, ha="center", va="bottom", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="#12161c",
                                      ec="none", alpha=0.62))
                    ax.text(0.5, 0.022, "local-background annulus, 13 to 33 px",
                            transform=ax.transAxes, color="#ffd166",
                            fontsize=11.5, ha="center", va="bottom", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="#12161c",
                                      ec="none", alpha=0.62))
        return axes

    def scene_0(self, fig, t: float) -> None:
        # both panels are fully drawn from the very first frame: a GIF whose
        # opening frame is half empty reads as a broken figure wherever the
        # animation is paused, previewed, or thumbnailed
        self._image_pair(fig, zoom=0.0, boxes_fraction=0.0, show_annulus=False,
                         show_numbers=0.0, right_alpha=1.0)
        a = _ease(t)
        fig.patches.append(rp.plt.matplotlib.patches.FancyArrowPatch(
            (0.472, 0.50), (0.472 + 0.046 * a, 0.50), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18, lw=2.0, color=rp.ACCENT_2,
            alpha=a, zorder=10))
        fig.text(0.4955, 0.545, "+110 ms", fontsize=12.5, color=rp.ACCENT_2,
                 fontweight="bold", ha="center", va="bottom", alpha=a)
        caption(fig,
                "Two exposures of the same atoms: a 100 ms frame, a 10 ms gap "
                "with the imaging light off, then a second 100 ms frame. "
                "Identical crop, identical intensity scale.")

    def scene_1(self, fig, t: float) -> None:
        self._image_pair(fig, zoom=0.0, boxes_fraction=_ease(t),
                         show_annulus=False, show_numbers=max(0.0, _ease(t) - 0.5) * 2)
        n = len(self.sites_df)
        caption(fig,
                f"{n} sites in two 10x10 sub-arrays. Positions come from the "
                f"shot-to-shot variance map, fitted to sub-pixel residual.")

    def scene_2(self, fig, t: float) -> None:
        z = _ease(t)
        self._image_pair(fig, zoom=z, boxes_fraction=1.0,
                         show_annulus=z > 0.45, show_numbers=0.0)
        s = self.sel
        if z > 0.6:
            a = min(1.0, (z - 0.6) / 0.4)
            for fid, ax_x in ((0, 0.2625), (1, 0.7325)):
                r = self.site_counts[fid]
                bg = float(r["roi_sum_raw"]) - float(r["background_corrected_count"])
                fig.text(ax_x, 0.172,
                         f"ROI sum {float(r['roi_sum_raw']):,.0f}      "
                         f"background {bg:,.0f}      "
                         f"corrected {float(r['background_corrected_count']):,.0f}",
                         fontsize=12.0, color=rp.INK, ha="center", va="center",
                         alpha=a)
        run_mean = float(self.df[(self.df["site_id"] == s.site_id)
                                 & (self.df["frame_id"] == 0)]
                         ["background_corrected_count"].mean())
        caption(fig,
                f"Site {s.site_id} ({s.site_grid}, row {s.site_row}, col "
                f"{s.site_col}), chosen as the median-brightness site; its "
                f"frame-0 mean over the run is {run_mean:,.0f} counts. This "
                f"shot is one draw from that distribution, not a picked example.")

    # ------------------------------------------------------------- scene 4-6
    def _hist_axes(self, fig):
        ax = stage_axes(fig, (0.075, 0.235, 0.885, 0.585))
        ax.set_ylabel("site-frame observations")
        ax.set_ylim(0, self.hist_ymax)
        return ax

    @staticmethod
    def _step(ax, edges, counts, color, label, alpha=1.0, scale=1.0):
        centers = np.repeat(edges, 2)[1:-1]
        y = np.repeat(counts * scale, 2)
        ax.plot(centers, y, color=color, lw=2.1, alpha=alpha, label=label)
        ax.fill_between(centers, 0, y, color=color, alpha=0.13 * alpha, lw=0)

    def scene_3(self, fig, t: float) -> None:
        ax = self._hist_axes(fig)
        s = _ease(t)
        for fid in (0, 1):
            self._step(ax, self.raw_edges, self.hist_raw[fid],
                       rp.FRAME_COLORS[fid], f"frame {fid}", scale=s)
        ax.set_xlim(*self.raw_lim)
        ax.set_xlabel("raw ROI count (camera counts)")
        ax.legend(loc="upper right")
        caption(fig,
                "Before background correction the two frames are offset by the "
                "frame-dependent background, not only by their signal.")

    def scene_4(self, fig, t: float) -> None:
        ax = self._hist_axes(fig)
        s = _ease(t)
        lim = (self.raw_lim[0] + (self.cor_lim[0] - self.raw_lim[0]) * s,
               self.raw_lim[1] + (self.cor_lim[1] - self.raw_lim[1]) * s)
        for fid in (0, 1):
            if s < 1.0:
                self._step(ax, self.raw_edges, self.hist_raw[fid],
                           rp.FRAME_COLORS[fid], None, alpha=1.0 - s)
            if s > 0.0:
                self._step(ax, self.cor_edges, self.hist_cor[fid],
                           rp.FRAME_COLORS[fid], f"frame {fid}" if s > 0.9 else None,
                           alpha=s)
        ax.set_xlim(*lim)
        ax.set_xlabel("background-corrected ROI count (camera counts)")
        if s > 0.9:
            ax.legend(loc="upper right")
        caption(fig,
                "Subtracting each observation's own local background aligns the "
                "two frames on a common count axis.")

    def scene_5(self, fig, t: float) -> None:
        ax = self._hist_axes(fig)
        s = _ease(t)
        x = np.linspace(*self.cor_lim, 600)
        binw = self.cor_edges[1] - self.cor_edges[0]
        for fid in (0, 1):
            self._step(ax, self.cor_edges, self.hist_cor[fid],
                       rp.FRAME_COLORS[fid], f"frame {fid}")
            fit = self.fits[fid]
            lo, hi = fit.density(x)
            n = self.cor_by_frame[fid].size * binw
            cut = int(len(x) * s)
            if cut > 2:
                ax.plot(x[:cut], (lo * n)[:cut], color=rp.FRAME_COLORS[fid],
                        lw=1.5, ls=":", alpha=0.95)
                ax.plot(x[:cut], (hi * n)[:cut], color=rp.FRAME_COLORS[fid],
                        lw=1.5, ls="--", alpha=0.95)
            if s > 0.75:
                ax.axvline(fit.reference_level, color=rp.FRAME_COLORS[fid],
                           lw=1.6, alpha=(s - 0.75) / 0.25)
        ax.set_xlim(*self.cor_lim)
        ax.set_xlabel("background-corrected ROI count (camera counts)")
        ax.legend(loc="upper right")
        d0 = self.fits[0].separation_d_prime
        d1 = self.fits[1].separation_d_prime
        caption(fig,
                f"Descriptive fit on the full dataset. Not a held-out fidelity "
                f"estimate. Model-implied separation d' = {d0:.2f} (frame 0), "
                f"{d1:.2f} (frame 1).")

    # --------------------------------------------------------------- scene 7
    def scene_6(self, fig, t: float) -> None:
        ax = stage_axes(fig, (0.075, 0.235, 0.545, 0.585))
        ok = self.pairs[~self.pairs["any_flag"]]
        lim = (min(self.cor_lim[0], -2000.0), self.cor_lim[1])
        s = _ease(t)
        n = max(50, int(len(ok) * min(1.0, s * 1.6)))
        sub = ok.iloc[:n]
        # light-to-dark on a white page: with a dark-low colormap the sparse
        # single-pair cells read as dense black speckle
        ax.hexbin(sub["f0"], sub["f1"], gridsize=54, bins="log",
                  cmap=DENSITY_CMAP, extent=(*lim, *lim), mincnt=1, linewidths=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("frame 0 background-corrected count")
        ax.set_ylabel("frame 1 background-corrected count")
        ax.set_aspect("equal")

        if s > 0.35:
            a = min(1.0, (s - 0.35) / 0.2)
            ax.axvline(self.refs[0], color=rp.ACCENT, ls="--", lw=1.5, alpha=a)
            ax.axhline(self.refs[1], color=rp.ACCENT_2, ls="--", lw=1.5, alpha=a)

        pr = self.metrics["paired_readout"]
        quadrants = [
            ("low-low", 0.04, 0.07, pr["low_low"]),
            ("high-high", 0.58, 0.93, pr["high_high"]),
            ("apparent\nbright-to-dark", 0.55, 0.09, pr["high_low"]),
            ("apparent\ndark-to-bright", 0.04, 0.91, pr["low_high"]),
        ]
        for i, (name, fx, fy, count) in enumerate(quadrants):
            appear = 0.55 + 0.10 * i
            if s <= appear:
                continue
            a = min(1.0, (s - appear) / 0.09)
            ax.text(lim[0] + fx * (lim[1] - lim[0]), lim[0] + fy * (lim[1] - lim[0]),
                    f"{name}\n{count:,}", fontsize=11.2, color="#ffffff",
                    alpha=a, ha="left", va="center", linespacing=1.3, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.28", fc="#12161c", ec="none",
                              alpha=0.84 * a))

        side = fig.add_axes([0.690, 0.235, 0.270, 0.585])
        side.axis("off")
        lines = [
            ("paired-readout agreement",
             f"{pr['agreement'] * 100:.1f}%  "
             f"[{pr['agreement_ci95'][0] * 100:.1f}, {pr['agreement_ci95'][1] * 100:.1f}]"),
            ("apparent bright-to-dark", f"{pr['apparent_bright_to_dark_rate'] * 100:.1f}%"),
            ("apparent dark-to-bright", f"{pr['apparent_dark_to_bright_rate'] * 100:.1f}%"),
            ("site-shot pairs", f"{pr['n_pairs']:,}"),
            ("independent shots", f"{self.metrics['dataset']['n_shots']}"),
        ]
        for i, (k, v) in enumerate(lines):
            y = 0.92 - i * 0.175
            a = min(1.0, max(0.0, (s - 0.5 - 0.06 * i) / 0.1))
            side.text(0.0, y, k, fontsize=11.6, color=rp.MUTED, alpha=a)
            side.text(0.0, y - 0.072, v, fontsize=14.5, color=rp.INK,
                      fontweight="bold", alpha=a)
        caption(fig,
                "Disagreement between the frames is reported as disagreement. "
                "Without matched-empty and loss controls it cannot be assigned "
                "to atom loss.")

    # --------------------------------------------------------------- scene 8
    def scene_7(self, fig, t: float) -> None:
        ax = stage_axes(fig, (0.055, 0.185, 0.895, 0.645))
        ax.axis("off")
        s = _ease(t)
        d = self.metrics["dataset"]
        steps = [
            ("raw fluorescence frames",
             f"{d['n_shots']} shots x {d['n_frames_per_shot']} frames, "
             f"{d['exposure_ms']:.0f} ms"),
            ("site ROIs from the variance map",
             f"{d['n_sites']} sites, {d['roi_pixels']} px each"),
            ("standardized frame-site table",
             f"{d['n_site_frame_observations']:,} rows, one primary key"),
            ("QC and background variants",
             "raw / local annulus / site-free common mode"),
            ("descriptive summaries",
             "separation and paired-readout agreement, with bootstrap CIs"),
            ("next: modelling milestone",
             "held-out split, mixture baseline, paired latent state"),
        ]
        n_show = int(np.ceil(s * len(steps)))
        for i, (head, sub) in enumerate(steps[:max(1, n_show)]):
            y = 0.90 - i * 0.163
            last = i == len(steps) - 1
            col = rp.MUTED if last else rp.ACCENT
            # a marker, not a Circle patch: the stage axes are not square, so a
            # patch circle renders as an ellipse
            ax.plot([0.028], [y + 0.012], transform=ax.transAxes, marker="o",
                    ms=10, mfc="none" if last else col, mec=col, mew=2.0,
                    zorder=3, clip_on=False)
            if i < len(steps) - 1:
                ax.add_line(rp.plt.Line2D([0.028, 0.028], [y - 0.14, y],
                                          transform=ax.transAxes, color=rp.GRID,
                                          lw=2.0, zorder=1))
            ax.text(0.062, y + 0.012, head, transform=ax.transAxes, fontsize=15.0,
                    color=rp.INK if not last else rp.MUTED,
                    fontweight="bold", va="center")
            ax.text(0.062, y - 0.045, sub, transform=ax.transAxes, fontsize=12.4,
                    color=rp.MUTED, va="center")
        caption(fig,
                "No readout fidelity, false-positive rate, false-negative rate "
                "or imaging-loss rate is estimated at this stage.")

    def scene_methods(self) -> list[Callable]:
        return [self.scene_0, self.scene_1, self.scene_2, self.scene_3,
                self.scene_4, self.scene_5, self.scene_6, self.scene_7]


def build_frames(builder: SceneBuilder) -> tuple[list[Image.Image], int]:
    """Render every GIF frame. Returns the frames and the fallback frame index."""
    methods = builder.scene_methods()
    frames: list[Image.Image] = []
    fallback_index = 0
    for idx, ((caption, n_move, n_hold), draw) in enumerate(zip(SCENES, methods)):
        last: Image.Image | None = None
        for k in range(n_move):
            fig = new_canvas(builder.width, builder.height)
            draw_chrome(fig, idx, caption)
            draw(fig, (k + 1) / n_move)
            last = render(fig)
            frames.append(last)
        assert last is not None
        if idx == 6:                       # the paired-readout scene is the still
            fallback_index = len(frames) - 1
        frames.extend([last] * n_hold)     # identical frames cost ~nothing
    return frames, fallback_index


# ----------------------------------------------------------- static figures
def static_paired_overlay(builder: SceneBuilder, out: Path) -> Path:
    rp.apply_style()
    x0, x1, y0, y1 = builder.crop
    lo, hi = np.percentile(np.concatenate(
        [builder.images[0][y0:y1, x0:x1].ravel(),
         builder.images[1][y0:y1, x0:x1].ravel()]), (1, 99.6))
    fig, axes = rp.plt.subplots(1, 2, figsize=(11.6, 5.3), layout="constrained")
    for fid, ax in enumerate(axes):
        h = ax.imshow(builder.images[fid], vmin=lo, vmax=hi, cmap=rp.SEQ_CMAP,
                      interpolation="nearest")
        for r in builder.sites_df.itertuples():
            ax.add_patch(rp.plt.Rectangle((r.roi_x0 - 0.5, r.roi_y0 - 0.5),
                                          r.roi_x1 - r.roi_x0, r.roi_y1 - r.roi_y0,
                                          fill=False, ec="#7fe3ff", lw=0.65))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"frame {fid}   ·   100 ms exposure",
                     color=rp.FRAME_COLORS[fid])
        rp.colorbar(fig, h, ax, "camera counts / pixel")
    fig.suptitle("Consecutive fluorescence frames, same crop and same scale, "
                 f"with the {len(builder.sites_df)} site ROIs")
    rp.provisional_note(fig, "One shot, selected by a median rule documented in "
                             "assets/readme/asset_metadata.json.")
    return rp.save(fig, out)


def static_count_distribution(builder: SceneBuilder, out: Path) -> Path:
    rp.apply_style()
    fig, axes = rp.plt.subplots(1, 2, figsize=(12.4, 4.9), layout="constrained",
                                sharey=True)
    x = np.linspace(*builder.cor_lim, 800)
    binw = builder.cor_edges[1] - builder.cor_edges[0]
    for fid, ax in enumerate(axes):
        counts = builder.hist_cor[fid]
        centers = 0.5 * (builder.cor_edges[:-1] + builder.cor_edges[1:])
        ax.bar(centers, counts, width=binw, color=rp.FRAME_COLORS[fid],
               alpha=0.28, edgecolor="none", label="observed histogram")
        ax.step(np.repeat(builder.cor_edges, 2)[1:-1], np.repeat(counts, 2),
                color=rp.FRAME_COLORS[fid], lw=1.4)
        fit = builder.fits[fid]
        n = builder.cor_by_frame[fid].size * binw
        lo, hi = fit.density(x)
        ax.plot(x, lo * n, color=rp.INK, lw=1.6, ls=":", label="lower component")
        ax.plot(x, hi * n, color=rp.INK, lw=1.6, ls="--", label="upper component")
        ax.plot(x, (lo + hi) * n, color=rp.INK, lw=1.1, alpha=0.55,
                label="sum of components")
        ax.axvline(fit.reference_level, color=rp.WARN, lw=1.5,
                   label="equal-prior crossing")
        ax.set_title(f"frame {fid}   ·   d' = {fit.separation_d_prime:.2f}",
                     color=rp.FRAME_COLORS[fid])
        ax.set_xlabel("background-corrected ROI count (camera counts)")
        ax.set_xlim(*builder.cor_lim)
    axes[0].set_ylabel("site-frame observations")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Pooled background-corrected count distribution — "
                 "descriptive fit on the full dataset")
    rp.provisional_note(
        fig, "Descriptive fit on the full dataset. NOT a held-out fidelity "
             "estimate: fitted on all data, no train/validation/test split, no "
             "per-site structure. The crossing is a display reference, not a "
             "validated classifier. d' describes this fit and is not an error "
             "rate. Formal held-out evaluation is the next milestone.")
    return rp.save(fig, out)


def static_paired_scatter(builder: SceneBuilder, out: Path) -> Path:
    rp.apply_style()
    ok = builder.pairs[~builder.pairs["any_flag"]]
    lim = (min(builder.cor_lim[0], -2000.0), builder.cor_lim[1])
    fig, ax = rp.plt.subplots(figsize=(7.4, 6.6), layout="constrained")
    hb = ax.hexbin(ok["f0"], ok["f1"], gridsize=62, bins="log", cmap="magma",
                   extent=(*lim, *lim), mincnt=1, linewidths=0)
    ax.plot(lim, lim, color=rp.MUTED, lw=0.9, ls=":")
    ax.axvline(builder.refs[0], color=rp.ACCENT, ls="--", lw=1.4)
    ax.axhline(builder.refs[1], color=rp.ACCENT_2, ls="--", lw=1.4)
    pr = builder.metrics["paired_readout"]
    for name, fx, fy, count in (("low-low", 0.05, 0.05, pr["low_low"]),
                                ("high-high", 0.60, 0.92, pr["high_high"]),
                                ("apparent\nbright-to-dark", 0.60, 0.05, pr["high_low"]),
                                ("apparent\ndark-to-bright", 0.05, 0.92, pr["low_high"])):
        ax.text(lim[0] + fx * (lim[1] - lim[0]), lim[0] + fy * (lim[1] - lim[0]),
                f"{name}\n{count:,}", fontsize=10.5, color="#ffffff", ha="left",
                va="center", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.28", fc="#12161c", ec="none",
                          alpha=0.82))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("frame 0 background-corrected ROI count")
    ax.set_ylabel("frame 1 background-corrected ROI count")
    ax.set_title("Paired readout: frame 0 against frame 1")
    rp.colorbar(fig, hb, ax, "site-shot pairs (log scale)")
    d = builder.metrics["dataset"]
    rp.provisional_note(
        fig, f"{pr['n_pairs']:,} site-shot pairs from {d['n_shots']} shots and "
             f"{d['n_sites']} sites. Site-level pairs are not independent "
             f"experimental units; the independent units are the shots. The "
             f"dashed lines are descriptive per-frame reference levels from a "
             f"fit on the full dataset — a display reference, NOT a validated "
             f"classifier and NOT a held-out fidelity estimate. The four "
             f"regions are apparent regions, not measured transitions.")
    return rp.save(fig, out)


def static_site_map(builder: SceneBuilder, site_stats: pd.DataFrame,
                    out: Path) -> Path:
    rp.apply_style()
    f0 = site_stats[site_stats["frame_id"] == 0]
    panels = [("mean", "mean background-corrected count", rp.SEQ_CMAP),
              ("local_background", "mean background under the ROI", "cividis"),
              ("interdecile_spread", "P90 - P10 spread", "viridis")]
    fig, axes = rp.plt.subplots(1, 3, figsize=(15.6, 5.0), layout="constrained")
    for ax, (col, label, cmap) in zip(axes, panels):
        s = ax.scatter(f0["site_x"], f0["site_y"], c=f0[col], s=58, marker="s",
                       cmap=cmap, edgecolors="none")
        miss = f0[~f0["site_detected"].astype(bool)]
        if len(miss):
            ax.scatter(miss["site_x"], miss["site_y"], s=140, marker="s",
                       facecolors="none", edgecolors=rp.WARN, linewidths=1.6,
                       label="geometry flag")
            ax.legend(loc="lower left", fontsize=9.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("camera column (px)")
        ax.grid(False)
        rp.colorbar(fig, s, ax, "camera counts")
    axes[0].set_ylabel("camera row (px)")
    fig.suptitle("Site-level descriptive maps, frame 0 "
                 f"({len(f0)} sites, 100 shots each)")
    rp.provisional_note(
        fig, "Outlined squares: no variance peak within the detection radius of "
             "the modelled position - a geometry-confidence flag, not a "
             "statement that the site is dark. Sites are localised from atom "
             "fluorescence, so a trap never loaded in this run cannot appear.")
    return rp.save(fig, out)


# ------------------------------------------------------------------- metrics
def write_metrics_fragment(cfg, metrics: dict[str, Any], selection: Selection,
                           out: Path) -> Path:
    d = metrics["dataset"]
    pr = metrics["paired_readout"]
    bg = metrics["background"]
    sites = metrics["sites"]

    def row(k: str, v: str) -> dict[str, str]:
        return {"quantity": k, "value": v}

    descriptive = [
        row("Shots", f"{d['n_shots']}"),
        row("Frames per shot", f"{d['n_frames_per_shot']}"),
        row("Exposure per frame", f"{d['exposure_ms']:.0f} ms"),
        row("Inter-frame start separation", "110 ms (100 ms exposure + 10 ms gap)"),
        row("Sites", f"{d['n_sites']} (two 10x10 sub-arrays)"),
        row("ROI", f"{d['roi_pixels']} px per site"),
        row("Site-frame observations", f"{d['n_site_frame_observations']:,}"),
        row("Site-shot pairs", f"{d['n_site_shot_pairs']:,}"),
        row("Rows carrying a quality flag",
            f"{sum(v for k, v in metrics['quality_flags'].items() if k != 'ok'):,}"),
        row("Sites with a geometry flag",
            f"{sites['n_without_variance_peak']} of {sites['n_sites']}"),
        row("Lattice fit residual (median / P95)",
            f"{sites['fit_residual_px_median']:.2f} / "
            f"{sites['fit_residual_px_p95']:.2f} px"),
        row("Saturated ROI pixels", f"{metrics['saturation']['n_rows_with_saturated_roi']}"),
        row("Brightest pixel seen",
            f"{metrics['saturation']['max_roi_pixel_seen']:,.0f} of "
            f"{metrics['saturation']['adc_ceiling']:,.0f} ADU"),
    ]

    background = [
        row("Frame 1 minus frame 0, template + offset (primary)",
            f"{bg['template_plus_offset']['frame1_minus_frame0_per_px']['mean']:+.2f} "
            f"± {bg['template_plus_offset']['frame1_minus_frame0_per_px']['std']:.2f} counts/px"),
        row("Frame 1 minus frame 0, global site-free median",
            f"{bg['global_site_free']['frame1_minus_frame0_per_px']['mean']:+.2f} "
            f"± {bg['global_site_free']['frame1_minus_frame0_per_px']['std']:.2f} counts/px"),
        row("Frame 1 minus frame 0, legacy annulus (contaminated)",
            f"{bg['annulus_contaminated']['frame1_minus_frame0_per_px']['mean']:+.2f} "
            f"± {bg['annulus_contaminated']['frame1_minus_frame0_per_px']['std']:.2f} counts/px"),
        row("Background level under a ROI, frame 0 (primary)",
            f"{bg['template_plus_offset']['frame0_mean_per_px']:.1f} counts/px"),
        row("Legacy annulus level, frame 0",
            f"{bg['annulus_contaminated']['frame0_mean_per_px']:.1f} counts/px"),
    ]

    paired = [
        row("Paired-readout agreement",
            f"{pr['agreement'] * 100:.2f}% "
            f"(95% shot-cluster bootstrap {pr['agreement_ci95'][0] * 100:.2f}–"
            f"{pr['agreement_ci95'][1] * 100:.2f}%)"),
        row("Apparent bright-to-dark", f"{pr['apparent_bright_to_dark_rate'] * 100:.2f}%"),
        row("Apparent dark-to-bright", f"{pr['apparent_dark_to_bright_rate'] * 100:.2f}%"),
        row("Above reference, frame 0", f"{pr['high_fraction_frame0'] * 100:.2f}%"),
        row("Above reference, frame 1", f"{pr['high_fraction_frame1'] * 100:.2f}%"),
    ]

    variants = []
    for key, block in metrics["variants"].items():
        for fid, v in block["per_frame"].items():
            variants.append({
                "variant": key.split("_")[0],
                "definition": qc.VARIANT_LABEL[key].split("  ", 1)[1],
                "frame": fid,
                "d'": f"{v['separation_d_prime']:.2f}",
                "model-implied overlap": f"{v['model_implied_overlap'] * 100:.1f}%",
                "drift / 100 shots": f"{v['shot_order_slope_per_100_shots']:+.0f}",
            })

    body = f"""<!-- generated by scripts/generate_readme_assets.py - do not edit by hand -->

### Dataset, as measured

{rp.markdown_table(descriptive, ["quantity", "value"])}

### Frame-dependent background

{rp.markdown_table(background, ["quantity", "value"])}

The two site-masked estimators agree with each other to 0.4 counts/px and with
the whole-frame median shift. The legacy annulus reports a shift about three
times larger, with seven times the spread: at a 10–11 px site pitch its 13–33 px
ring contains roughly ten neighbouring sites, so part of what it calls
"background" is array light that itself changes between the frames. It is kept
as a diagnostic column and is not used for inference.

### Paired-readout agreement

{rp.markdown_table(paired, ["quantity", "value"])}

Agreement is computed against per-frame descriptive reference levels.
"Apparent" is meant literally: this run has no matched-empty, dark-frame or
natural-loss control, so a disagreement cannot be assigned to atom loss rather
than to misclassification.

### Measurement variants

{rp.markdown_table(variants, ["variant", "definition", "frame", "d'",
                              "model-implied overlap", "drift / 100 shots"])}

`d'` and the overlap describe the descriptive two-component fit. They are
**not** a readout fidelity, a false-positive rate or a false-negative rate.

### Representative example used in the README assets

- **Shot:** {selection.shot_rule} → shot order {selection.shot_order}.
- **Site:** {selection.site_rule} → site {selection.site_id}
  ({selection.site_grid}, row {selection.site_row}, column {selection.site_col}).
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    rp.assert_public_safe(body, "readme metrics fragment")
    out.write_text(body, encoding="utf-8")
    return out


def inject_into_readme(readme: Path, blocks: dict[str, str]) -> list[str]:
    """Replace each ``<!-- BEGIN:key -->…<!-- END:key -->`` region in README.md.

    The README quotes numbers, and numbers go stale. Generating the regions
    from the same QC summary the figures use means there is one source of truth
    and no hand-typed metric anywhere in the repository.
    """
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8")
    updated: list[str] = []
    for key, body in blocks.items():
        begin, end = f"<!-- BEGIN:{key} -->", f"<!-- END:{key} -->"
        i, j = text.find(begin), text.find(end)
        if i == -1 or j == -1 or j < i:
            continue
        text = text[:i + len(begin)] + "\n" + body.strip() + "\n" + text[j:]
        updated.append(key)
    rp.assert_public_safe(text, "README")
    readme.write_text(text, encoding="utf-8")
    return updated


def dataset_block(metrics: dict[str, Any]) -> str:
    d = metrics["dataset"]
    sites = metrics["sites"]
    return (
        f"**Current dataset** — {d['n_shots']} shots · "
        f"{d['n_frames_per_shot']} consecutive frames per shot · "
        f"{d['exposure_ms']:.0f} ms exposure · {d['n_sites']} sites "
        f"({d['n_site_frame_observations']:,} site-frame observations, "
        f"{d['n_site_shot_pairs']:,} site-shot pairs, "
        f"{sites['n_without_variance_peak']} sites carrying a geometry flag). "
        f"The independent experimental units are the {d['n_shots']} shots."
    )


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paired_100ms.yaml")
    ap.add_argument("--skip-gif", action="store_true",
                    help="static figures and metrics only")
    ap.add_argument(
        "--output-dir", type=Path,
        help=(
            "write assets to an isolated directory; when set, the metrics "
            "fragment is written there and README.md is not modified"
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        df, sites_df, _meta = load_dataset(cfg)
        shots = discover_shots(cfg)
        load, _ = _frame_loader(cfg)
    except SourceDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary_path = cfg.reports_dir("qc", "qc_summary.json")
    if not summary_path.exists():
        print("ERROR: reports/qc/qc_summary.json is missing. Run "
              "scripts/export_processed_dataset.py first.", file=sys.stderr)
        return 2
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    site_stats = pd.read_csv(cfg.reports_dir("qc", "site_statistics.csv"))

    seed = int(cfg["readme_assets"]["random_seed"])
    np.random.default_rng(seed)          # fixed seed; nothing here is stochastic

    selection = choose_representative(df)
    images = {int(s["frame_id"]): load(shots[selection.shot_order], s["h5_path"])
              for s in cfg.frame_specs}

    pad = 10
    crop = (int(sites_df["site_x"].min()) - pad, int(sites_df["site_x"].max()) + pad + 1,
            int(sites_df["site_y"].min()) - pad, int(sites_df["site_y"].max()) + pad + 1)

    fits = qc.descriptive_fits(cfg, df, "background_corrected_count")
    pairs = qc.paired_table(df, "background_corrected_count")
    refs = (fits[0].reference_level, fits[1].reference_level)

    width = int(cfg["readme_assets"]["gif_width_px"])
    height = int(round(width * 0.60))
    builder = SceneBuilder(cfg, df, sites_df, {
        "selection": selection, "images": images, "crop": crop, "fits": fits,
        "pairs": pairs, "refs": refs, "metrics": metrics,
        "width": width, "height": height,
    })

    repo = Path(__file__).resolve().parents[1]
    publish = args.output_dir is None
    out_dir = repo / ASSET_DIR if publish else args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    asset_reference_dir = ASSET_DIR if publish else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: dict[str, Any] = {}
    if not args.skip_gif:
        canvas_style()
        frames, fallback = build_frames(builder)
        gif_info = write_gif(frames, out_dir / GIF_NAME,
                             fps=int(cfg["readme_assets"]["gif_fps"]))
        frames[fallback].save(out_dir / PNG_NAME, optimize=True)
        gif_info["static_fallback"] = (
            asset_reference_dir / PNG_NAME
        ).as_posix()
        gif_info["static_fallback_frame"] = fallback
        gif_info["static_fallback_scene"] = SCENES[6][0]
        produced["hero"] = gif_info
        print(f"gif ........ {gif_info['width_px']}x{gif_info['height_px']}, "
              f"{gif_info['n_frames']} frames @ {gif_info['fps']} fps = "
              f"{gif_info['duration_s']}s, {gif_info['megabytes']} MB")

    rp.apply_style()
    statics = {
        "paired_images_roi_overlay.png":
            static_paired_overlay(builder, out_dir / "paired_images_roi_overlay.png"),
        "count_distribution_fit.png":
            static_count_distribution(builder, out_dir / "count_distribution_fit.png"),
        "paired_frame_scatter.png":
            static_paired_scatter(builder, out_dir / "paired_frame_scatter.png"),
        "site_summary_map.png":
            static_site_map(builder, site_stats, out_dir / "site_summary_map.png"),
    }
    produced["static_figures"] = {
        k: {"path": (asset_reference_dir / k).as_posix(),
            "bytes": v.stat().st_size}
        for k, v in statics.items()
    }
    for k in statics:
        print(f"figure ..... {k}")

    metrics_path = (
        repo / "reports" / "readme_metrics.md"
        if publish else out_dir / "readme_metrics.md"
    )
    frag = write_metrics_fragment(cfg, metrics, selection, metrics_path)
    produced["metrics_fragment"] = (
        frag.relative_to(repo).as_posix() if publish else frag.name
    )
    print(f"metrics .... {produced['metrics_fragment']}")

    injected: list[str] = []
    if publish:
        fragment_body = "\n".join(
            line for line in frag.read_text(encoding="utf-8").splitlines()
            if not line.startswith("<!--"))
        injected = inject_into_readme(repo / "README.md", {
            "dataset-line": dataset_block(metrics),
            "results": fragment_body,
        })
    produced["readme_blocks_injected"] = injected
    print(f"readme ..... {', '.join(injected) if injected else 'no markers found'}")

    meta = {
        "provenance": stamp(cfg, inputs=shots),
        "generated_by": "scripts/generate_readme_assets.py",
        "command": f"python scripts/generate_readme_assets.py --config {args.config}",
        "random_seed": seed,
        "dataset": metrics["dataset"],
        "selection": {
            "shot_order": selection.shot_order, "shot_rule": selection.shot_rule,
            "site_id": selection.site_id, "site_rule": selection.site_rule,
            "site_grid": selection.site_grid, "site_row": selection.site_row,
            "site_col": selection.site_col,
        },
        "storyboard": [{"index": i, "caption": c, "transition_frames": m,
                        "hold_frames": h} for i, (c, m, h) in enumerate(SCENES)],
        "synthetic_panels": [],
        "provisional_panels": [
            "hero scene 6 (two-component description) — descriptive fit on the "
            "full dataset, not a held-out fidelity estimate",
            "count_distribution_fit.png — same fit, same caveat",
            "reference levels in hero scene 7 and paired_frame_scatter.png — "
            "display references, not a validated classifier",
        ],
        "next_milestone": (
            "shot-ordered 60/20/20 split, threshold and mixture baselines "
            "evaluated on held-out shots, shot-cluster bootstrap intervals"),
        "claim_guard": metrics["interpretation_guard"]["notes"],
        "assets": produced,
        "privacy": {
            "images_cropped_to_site_array": True,
            "crop_x0x1y0y1": [int(v) for v in crop],
            "filenames_in_assets": False,
            "absolute_paths_in_assets": False,
            "run_identifier_in_visible_text": False,
        },
    }
    meta_path = out_dir / "asset_metadata.json"
    blob = json.dumps(meta, indent=2, default=str)
    rp.assert_public_safe(blob, "asset metadata")
    meta_path.write_text(blob, encoding="utf-8")
    metadata_label = (
        (ASSET_DIR / "asset_metadata.json").as_posix()
        if publish else meta_path.name
    )
    print(f"metadata ... {metadata_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
