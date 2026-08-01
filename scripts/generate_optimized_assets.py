"""Generate deterministic public figures for the optimized 2026-07-31 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.reporting import (  # noqa: E402
    ACCENT,
    ACCENT_2,
    GRID,
    INK,
    MUTED,
    OK,
    WARN,
    apply_style,
    assert_public_safe,
    save,
)


DEFAULT_RESULT = ROOT / "reports" / "optimized_lifetime_results_20260731.json"
DEFAULT_OUTPUT = ROOT / "assets" / "readme"
DEFAULT_METRICS = ROOT / "reports" / "optimized_readme_metrics_20260731.md"
DEFAULT_STORY_TABLE = (
    ROOT / "data" / "processed" / "imaging_5frame_100ms_20260731_0114.parquet"
)
DEFAULT_STORY_RAW_DIR = ROOT / "Experiment-Data" / "2026-07-31" / "0114"
STORY_FRAME_PATH = "images/hamamatsu/fluor/atoms"
STORY_WIDTH_PX = 1000
STORY_HEIGHT_PX = 600
STORY_FPS = 10
STORY_SCENE_HOLD_MS = 1000
STORY_TRANSITION_FRAMES = 5
ASSET_NAMES = (
    "optimized_occupancy_inference.gif",
    "optimized_occupancy_inference.png",
    "optimized_sequence_design.png",
    "optimized_lifetime_overview.png",
    "exposure_segmentation_result.png",
    "optimized_readout_tradeoff.png",
)
EXPOSURE_COLORS = {"50ms": ACCENT, "100ms": OK, "200ms": ACCENT_2}
HIST_FILL = "#66727f"
HIST_EDGE = "#3d4650"
HIST_ALPHA = 0.82
CORRECTED_HIST_EDGE = "#1f4d70"


class AssetInputError(ValueError):
    """The reviewed result does not support a public asset."""


@dataclass(frozen=True)
class OccupancyStoryData:
    """Frozen inputs for the image-to-apparent-occupancy visual story."""

    raw_image: np.ndarray
    sites: pd.DataFrame
    representative_rows: pd.DataFrame
    heldout_raw_counts: np.ndarray
    heldout_corrected_counts: np.ndarray
    posterior_by_frame: np.ndarray
    selected_shot_id: int
    selected_site_id: int
    shot_rule: str
    site_rule: str


def _closest_to_median(values: pd.Series) -> int:
    target = float(values.median())
    return int(min(values.index, key=lambda key: (abs(float(values.loc[key]) - target), int(key))))


def load_occupancy_story_data(
    result: Mapping[str, Any],
    *,
    table_path: Path = DEFAULT_STORY_TABLE,
    raw_dir: Path = DEFAULT_STORY_RAW_DIR,
) -> OccupancyStoryData:
    """Load one deterministic held-out example without fitting any model."""
    if not table_path.is_file():
        raise AssetInputError(f"frozen 100 ms processed table is missing: {table_path.name}")
    table = pd.read_parquet(table_path)
    test = table.loc[table["split"].eq("test")].copy()
    if test.empty or int(test["shot_id"].nunique()) != 20:
        raise AssetInputError("frozen 100 ms test split must contain 20 shots")

    shot_summary = test.groupby("shot_id", observed=True)[
        "background_corrected_count"
    ].median()
    shot_id = _closest_to_median(shot_summary)
    shot = test.loc[test["shot_id"].eq(shot_id)].copy()
    site_summary = shot.groupby("site_id", observed=True)[
        "background_corrected_count"
    ].median()
    site_id = _closest_to_median(site_summary)
    representative_rows = (
        shot.loc[shot["site_id"].eq(site_id)].sort_values("frame_id").reset_index(drop=True)
    )
    if representative_rows["frame_id"].tolist() != [0, 1, 2, 3, 4]:
        raise AssetInputError("representative site does not contain all five frozen frames")

    raw_name = Path(str(representative_rows.iloc[0]["raw_image_path"])).name
    raw_path = raw_dir / raw_name
    if not raw_path.is_file():
        raise AssetInputError(f"representative raw shot is missing: {raw_name}")
    with h5py.File(raw_path, "r") as handle:
        raw_image = np.asarray(handle[STORY_FRAME_PATH][...], dtype=float)

    sites = (
        table[["site_id", "site_x", "site_y", "site_row", "site_col"]]
        .drop_duplicates("site_id")
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    block = result["repeated_imaging"]["emission_models_by_exposure"]["100ms"]
    posterior_rows = sorted(
        block["selected_test_apparent_occupancy"], key=lambda row: row["frame_index"]
    )
    posterior = np.asarray(
        [row["mean_posterior_occupancy"] for row in posterior_rows], dtype=float
    )
    if posterior.size != 5:
        raise AssetInputError("reviewed 100 ms posterior summary must have five frames")
    return OccupancyStoryData(
        raw_image=raw_image,
        sites=sites,
        representative_rows=representative_rows,
        heldout_raw_counts=test["roi_sum_raw"].to_numpy(dtype=float),
        heldout_corrected_counts=test["background_corrected_count"].to_numpy(dtype=float),
        posterior_by_frame=posterior,
        selected_shot_id=shot_id,
        selected_site_id=site_id,
        shot_rule=(
            "held-out shot whose median background-corrected count across all "
            "sites and five frames is closest to the held-out shot median"
        ),
        site_rule=(
            "site in that shot whose median background-corrected count across "
            "five frames is closest to the shot's site median"
        ),
    )


def _estimate(value: Any) -> float:
    """Return a point estimate from either scalar or interval serialization."""
    if isinstance(value, Mapping):
        value = value["estimate"]
    return float(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_reviewed(result: Mapping[str, Any]) -> None:
    provenance = result.get("provenance", {})
    evidence = result.get("prerequisite_evidence", {})
    if result.get("bootstrap_replicates") != 1000:
        raise AssetInputError("public assets require the reviewed 1000-bootstrap result")
    if provenance.get("publishable_clean_provenance") is not True:
        raise AssetInputError("public assets require clean result provenance")
    if evidence.get("timing_and_compiled_commands_verified") is not True:
        raise AssetInputError("timing/compiled-command gate is not public-ready")
    if not all(evidence.get("geometry_validated", {}).values()):
        raise AssetInputError("one or more geometry gates failed")
    if not all(evidence.get("background_frozen", {}).values()):
        raise AssetInputError("one or more background gates failed")


def _selected_emission_metrics(block: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = block["selected_model"]
    for row in block["test_metrics_once_per_candidate"]:
        if row["candidate"] == selected:
            return row
    raise AssetInputError(f"selected emission test metrics are missing for {selected}")


def _interval(rows: Sequence[Mapping[str, Any]], parameter: str) -> Mapping[str, Any]:
    for row in rows:
        if row["parameter"] == parameter:
            return row
    raise AssetInputError(f"bootstrap interval is missing {parameter}")


def _draw_occupancy(result: Mapping[str, Any], path: Path) -> None:
    apply_style()
    emissions = result["repeated_imaging"]["emission_models_by_exposure"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for ax, exposure in zip(axes, ("50ms", "100ms", "200ms")):
        block = emissions[exposure]
        distribution = block["public_heldout_distribution"]
        metrics = _selected_emission_metrics(block)
        x = np.asarray(distribution["bin_centers"], dtype=float)
        ax.fill_between(
            x,
            distribution["heldout_density"],
            color=HIST_FILL,
            alpha=0.45,
            step="mid",
            label="held-out counts",
        )
        ax.step(
            x,
            distribution["heldout_density"],
            where="mid",
            color=HIST_EDGE,
            linewidth=1.0,
        )
        ax.plot(x, distribution["model_empty_density"], color=ACCENT, label="empty component")
        ax.plot(
            x,
            distribution["model_occupied_density"],
            color=ACCENT_2,
            label="occupied component",
        )
        ax.set_title(f"{exposure[:-2]} ms exposure")
        ax.set_xlabel("background-adjusted count")
        ax.text(
            0.03,
            0.96,
            (
                f"d′ = {metrics['mean_separation_d_prime']:.2f}\n"
                f"overlap = {100 * metrics['mean_model_implied_overlap']:.2f}%"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color=INK,
        )
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Held-out fluorescence count inference", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Two latent Gaussian components; overlap is model-implied, not empirical fidelity.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    save(fig, path, dpi=170)


STORY_STEPS = (
    "raw frame", "frozen geometry", "ROI + background", "raw counts",
    "corrected counts", "held-out model", "apparent occupancy", "survival analysis",
)


def _story_crop(story: OccupancyStoryData, pad: int = 18) -> tuple[int, int, int, int]:
    return (
        max(0, int(np.floor(story.sites["site_x"].min())) - pad),
        min(story.raw_image.shape[1], int(np.ceil(story.sites["site_x"].max())) + pad),
        max(0, int(np.floor(story.sites["site_y"].min())) - pad),
        min(story.raw_image.shape[0], int(np.ceil(story.sites["site_y"].max())) + pad),
    )


def _story_base(index: int, title: str) -> plt.Figure:
    apply_style()
    plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 13.0})
    fig = plt.figure(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.035, 0.945, "Image → apparent occupancy",
             fontsize=20, fontweight="bold", color=INK, va="center")
    fig.text(0.965, 0.945, "100 ms repeated imaging · frozen outputs",
             fontsize=10.5, color=MUTED, va="center", ha="right")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.905, 0.905], color=GRID, lw=1.0))
    fig.text(0.035, 0.865, f"{index + 1}. {title}", fontsize=14.5,
             color=ACCENT, fontweight="bold", va="center")
    width = 0.93 / len(STORY_STEPS)
    for step_index, label in enumerate(STORY_STEPS):
        x = 0.035 + width * (step_index + 0.5)
        active, completed = step_index == index, step_index < index
        color = ACCENT if active else (MUTED if completed else GRID)
        fig.add_artist(plt.Line2D(
            [0.035 + width * step_index + 0.004,
             0.035 + width * (step_index + 1) - 0.004],
            [0.055, 0.055], color=color, lw=3.6 if active else 2.0,
            solid_capstyle="round",
        ))
        fig.text(x, 0.027, label, ha="center", va="center", fontsize=8.0,
                 color=INK if active else MUTED,
                 fontweight="bold" if active else "normal")
    return fig


def _story_caption(fig: plt.Figure, text: str) -> None:
    assert_public_safe(text, "optimized occupancy animation caption")
    fig.text(0.5, 0.105, text, ha="center", va="center", color=MUTED, fontsize=10.6)


def _render_story_scene(
    result: Mapping[str, Any], story: OccupancyStoryData, index: int
) -> Image.Image:
    titles = (
        "Representative raw fluorescence frame",
        "Training-fitted geometry and ROI overlay",
        "Representative ROI signal and site-free background",
        "Raw held-out count distribution",
        "Background-corrected held-out count distribution",
        "Held-out emission model",
        "Posterior apparent occupancy over five frames",
        "Apparent occupancy feeds survival and lifetime analysis",
    )
    fig = _story_base(index, titles[index])
    x0, x1, y0, y1 = _story_crop(story)
    vmin, vmax = (float(value) for value in np.percentile(story.raw_image, (5.0, 99.8)))
    if index == 0:
        ax = fig.add_axes([0.16, 0.18, 0.68, 0.64])
        ax.imshow(story.raw_image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec=ACCENT_2, lw=2.0))
        ax.set_axis_off()
        _story_caption(fig, "Held-out shot selected by a median-count rule; the lattice region is outlined, not hand-picked.")
    elif index == 1:
        ax = fig.add_axes([0.12, 0.18, 0.76, 0.64])
        ax.imshow(story.raw_image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.scatter(story.sites["site_x"], story.sites["site_y"], s=48,
                   facecolors="none", edgecolors="#7fe3ff", linewidths=0.9)
        selected = story.sites.loc[story.sites["site_id"].eq(story.selected_site_id)].iloc[0]
        ax.scatter([selected["site_x"]], [selected["site_y"]], s=130,
                   facecolors="none", edgecolors="#ffd166", linewidths=2.2)
        ax.set_xticks([]); ax.set_yticks([])
        _story_caption(fig, "All 100 ROI centers come from training-fitted geometry; the median-rule site is highlighted.")
    elif index == 2:
        row = story.representative_rows.iloc[0]
        cx, cy = float(row["site_x"]), float(row["site_y"])
        ax = fig.add_axes([0.07, 0.20, 0.43, 0.60])
        ax.imshow(story.raw_image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_xlim(cx - 18, cx + 18); ax.set_ylim(cy + 18, cy - 18)
        ax.add_patch(plt.Rectangle((cx - 2.5, cy - 2.5), 5, 5, fill=False,
                                   ec="#7fe3ff", lw=2.2))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("5×5 px signal ROI")
        bar = fig.add_axes([0.58, 0.27, 0.34, 0.45])
        values = [float(row[key]) for key in
                  ("roi_sum_raw", "background_fixed_offset", "background_corrected_count")]
        bars = bar.bar(
            ["raw ROI", "site-free\nbackground", "corrected\nsignal"],
            values,
            color=[HIST_FILL, ACCENT, ACCENT_2],
            edgecolor=[HIST_EDGE, INK, INK],
            alpha=HIST_ALPHA,
            linewidth=0.7,
        )
        for patch, value in zip(bars, values):
            bar.text(patch.get_x() + patch.get_width() / 2, value, f"{value:,.0f}",
                     ha="center", va="bottom", fontsize=9.5)
        bar.set_ylabel("camera counts"); bar.grid(axis="x", visible=False)
        _story_caption(fig, "Frozen site-free template plus frame offset: raw ROI − background = corrected count.")
    elif index in (3, 4):
        values = story.heldout_raw_counts if index == 3 else story.heldout_corrected_counts
        ax = fig.add_axes([0.09, 0.20, 0.82, 0.60])
        limits = np.percentile(values, (0.2, 99.8))
        ax.hist(
            values,
            bins=np.linspace(*limits, 70),
            color=HIST_FILL if index == 3 else ACCENT,
            edgecolor=HIST_EDGE if index == 3 else CORRECTED_HIST_EDGE,
            alpha=HIST_ALPHA if index == 3 else 0.80,
            linewidth=0.35,
        )
        ax.set_xlabel("raw ROI count" if index == 3 else "background-corrected count")
        ax.set_ylabel("held-out site-frame observations")
        _story_caption(fig, "Same 20 held-out shots before correction." if index == 3 else
                       "Frozen background correction places held-out observations on the emission-model count axis.")
    elif index == 5:
        block = result["repeated_imaging"]["emission_models_by_exposure"]["100ms"]
        distribution, metrics = block["public_heldout_distribution"], _selected_emission_metrics(block)
        x = np.asarray(distribution["bin_centers"], dtype=float)
        ax = fig.add_axes([0.09, 0.20, 0.82, 0.60])
        ax.fill_between(
            x,
            distribution["heldout_density"],
            step="mid",
            color=HIST_FILL,
            alpha=0.45,
            label="held-out counts",
        )
        ax.step(
            x,
            distribution["heldout_density"],
            where="mid",
            color=HIST_EDGE,
            linewidth=1.0,
        )
        ax.plot(x, distribution["model_empty_density"], color=ACCENT, lw=2, label="empty component")
        ax.plot(x, distribution["model_occupied_density"], color=ACCENT_2, lw=2, label="occupied component")
        ax.set_xlabel("background-corrected count"); ax.set_ylabel("density"); ax.legend(loc="upper right")
        ax.text(0.025, 0.94, f"model-implied overlap = {100 * metrics['mean_model_implied_overlap']:.2f}%",
                transform=ax.transAxes, va="top", color=INK, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
        _story_caption(fig, "Held-out emission model; separation is model-implied, not empirical fidelity.")
    elif index == 6:
        ax = fig.add_axes([0.12, 0.22, 0.76, 0.56]); frames = np.arange(1, 6)
        ax.plot(frames, story.posterior_by_frame, marker="o", ms=8, lw=2.2, color=ACCENT_2)
        for frame, value in zip(frames, story.posterior_by_frame):
            ax.text(frame, value + 0.0012, f"{value:.3f}", ha="center", fontsize=9.5)
        ax.set_xticks(frames); ax.set_xlabel("frame"); ax.set_ylabel("mean posterior apparent occupancy")
        ax.set_ylim(min(story.posterior_by_frame) - 0.012, max(story.posterior_by_frame) + 0.012)
        _story_caption(fig, "Posterior apparent occupancy is aggregated over complete held-out shots; occupancy labels are latent.")
    else:
        ax = fig.add_axes([0.04, 0.19, 0.92, 0.62]); ax.set_axis_off()
        box = dict(boxstyle="round,pad=0.6", fc="white", ec=GRID, lw=1.5)
        ax.text(0.13, 0.60, "held-out counts\n↓\napparent occupancy", ha="center", va="center", fontsize=14, color=INK, bbox=box)
        ax.text(0.50, 0.60, "frozen survival model\nbright + dark hazards", ha="center", va="center", fontsize=14, color=INK, bbox=box)
        ax.text(0.87, 0.60, "five-frame survival\n100 ms: 97.46%", ha="center", va="center", fontsize=14, color=INK, bbox=box)
        for start, end in ((0.26, 0.39), (0.63, 0.76)):
            ax.annotate("", xy=(end, 0.60), xytext=(start, 0.60),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color=ACCENT))
        ax.text(0.50, 0.25, "dark lifetime 35.73 s   ·   bright lifetime 20.30 s",
                ha="center", va="center", fontsize=13, color=ACCENT_2, fontweight="bold")
        _story_caption(fig, "Apparent occupancy feeds operational lifetime and survival inference, not an exact physical-loss timestamp.")
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(pixels, mode="RGB")


def _draw_occupancy_animation(
    result: Mapping[str, Any], story: OccupancyStoryData, path: Path
) -> dict[str, Any]:
    scenes = [_render_story_scene(result, story, index) for index in range(8)]
    frames, durations = [], []
    for index, scene in enumerate(scenes):
        frames.append(scene); durations.append(STORY_SCENE_HOLD_MS)
        if index < len(scenes) - 1:
            for step in range(1, STORY_TRANSITION_FRAMES + 1):
                frames.append(Image.blend(scene, scenes[index + 1], step / (STORY_TRANSITION_FRAMES + 1)))
                durations.append(round(1000 / STORY_FPS))
    sample = frames[:: max(1, len(frames) // 10)]
    montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for index, frame in enumerate(sample):
        montage.paste(frame, (0, index * frame.height))
    palette = montage.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, optimize=True, disposal=1)
    size, duration_s = path.stat().st_size, round(sum(durations) / 1000, 2)
    if size >= 8_000_000 or not 10.0 <= duration_s <= 15.0:
        raise AssetInputError("optimized occupancy animation misses size/duration target")
    return {"width_px": STORY_WIDTH_PX, "height_px": STORY_HEIGHT_PX,
            "fps": STORY_FPS, "n_encoded_frames": len(frames),
            "duration_s": duration_s, "bytes": size}


def _draw_occupancy_summary(
    result: Mapping[str, Any], story: OccupancyStoryData, path: Path
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.87,
                        hspace=0.50, wspace=0.28)
    x0, x1, y0, y1 = _story_crop(story)
    vmin, vmax = (float(value) for value in np.percentile(story.raw_image, (5.0, 99.8)))
    axes[0, 0].imshow(story.raw_image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    axes[0, 0].set_xlim(x0, x1); axes[0, 0].set_ylim(y1, y0)
    axes[0, 0].scatter(story.sites["site_x"], story.sites["site_y"], s=19,
                       facecolors="none", edgecolors="#7fe3ff", linewidths=0.6)
    axes[0, 0].set_title("raw frame + frozen ROIs"); axes[0, 0].set_axis_off()
    limits = np.percentile(story.heldout_corrected_counts, (0.2, 99.8))
    axes[0, 1].hist(
        story.heldout_corrected_counts,
        bins=np.linspace(*limits, 55),
        color=ACCENT,
        edgecolor=CORRECTED_HIST_EDGE,
        alpha=0.80,
        linewidth=0.35,
    )
    axes[0, 1].set_title("background-corrected held-out counts")
    axes[0, 1].set_xlabel("corrected count"); axes[0, 1].set_ylabel("observations")
    block = result["repeated_imaging"]["emission_models_by_exposure"]["100ms"]
    distribution = block["public_heldout_distribution"]; x = np.asarray(distribution["bin_centers"], dtype=float)
    axes[1, 0].fill_between(
        x,
        distribution["heldout_density"],
        step="mid",
        color=HIST_FILL,
        alpha=0.45,
        label="held-out counts",
    )
    axes[1, 0].step(
        x,
        distribution["heldout_density"],
        where="mid",
        color=HIST_EDGE,
        linewidth=1.0,
    )
    axes[1, 0].plot(x, distribution["model_empty_density"], color=ACCENT, lw=1.8, label="empty")
    axes[1, 0].plot(x, distribution["model_occupied_density"], color=ACCENT_2, lw=1.8, label="occupied")
    axes[1, 0].set_title("held-out emission model"); axes[1, 0].set_xlabel("corrected count")
    axes[1, 0].set_ylabel("density"); axes[1, 0].legend(fontsize=8)
    frames = np.arange(1, 6)
    axes[1, 1].plot(frames, story.posterior_by_frame, marker="o", color=ACCENT_2, lw=2)
    axes[1, 1].set_xticks(frames); axes[1, 1].set_title("posterior apparent occupancy")
    axes[1, 1].set_xlabel("frame"); axes[1, 1].set_ylabel("mean posterior")
    axes[1, 1].set_ylim(min(story.posterior_by_frame) - 0.012, max(story.posterior_by_frame) + 0.012)
    fig.suptitle("Image to apparent occupancy · 100 ms held-out example", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.025, "Model-implied separation; not empirical fidelity.",
             ha="center", color=MUTED, fontsize=9)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), mode="RGB")
    plt.close(fig); path.parent.mkdir(parents=True, exist_ok=True); image.save(path, optimize=True)


def _pulse(ax: plt.Axes, start: float, width: float, *, color: str = ACCENT) -> None:
    ax.add_patch(plt.Rectangle((start, 0.23), width, 0.54, color=color, ec="none"))


def _draw_sequence(path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 5.6), constrained_layout=True)
    labels = (
        "Dark lifetime",
        "Bright / imaging lifetime",
        "Repeated imaging",
    )
    for ax, label in zip(axes, labels):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=18)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.grid(False)
    for index in range(5):
        _pulse(axes[0], index * 1.45, 0.36)
        if index < 4:
            axes[0].annotate(
                "variable fully-dark hold",
                xy=(index * 1.45 + 0.9, 0.5),
                ha="center",
                va="center",
                fontsize=8.2,
                color=MUTED,
            )
    axes[0].set_xlim(-0.15, 6.3)
    axes[0].set_xticks([])

    axes[1].plot([0.0, 3.4], [0.5, 0.5], color=ACCENT_2, lw=5)
    axes[1].text(1.7, 0.68, "variable continuous bright wait", ha="center", color=ACCENT_2)
    _pulse(axes[1], 3.65, 0.38)
    _pulse(axes[1], 4.25, 0.38)
    axes[1].text(4.15, 0.05, "two 50 ms frames", ha="center", color=MUTED, fontsize=9)
    axes[1].set_xlim(-0.15, 4.9)
    axes[1].set_xticks([])

    for index in range(5):
        _pulse(axes[2], index * 1.05, 0.82, color=EXPOSURE_COLORS["100ms"])
        if index < 4:
            axes[2].text(index * 1.05 + 0.93, 0.50, "10 ms", ha="center", fontsize=8, color=MUTED)
    axes[2].text(2.5, 0.05, "five frames at 50, 100, or 200 ms", ha="center", color=MUTED, fontsize=9)
    axes[2].set_xlim(-0.15, 5.15)
    axes[2].set_xticks([])
    fig.suptitle("Optimized measurement designs", fontsize=14, fontweight="bold")
    fig.text(
        0.99,
        0.01,
        "Schematic; pulse widths are not to scale.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    save(fig, path, dpi=170)


def _curve_by_interval(rows: Sequence[Mapping[str, Any]], interval: str):
    return [row for row in rows if row.get("interval") == interval]


def _draw_lifetime_overview(result: Mapping[str, Any], path: Path) -> None:
    apply_style()
    dark = result["optimized_dark_bright"]["dark_hold"]["operational_model"]
    bright = result["optimized_dark_bright"]["bright_wait"]["effective_model"]
    budget = result["loss_budget"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    dark_interval = "1->2"
    band = _curve_by_interval(dark["model_band"], dark_interval)
    curve = _curve_by_interval(dark["clustered_apparent_retention_curve"], dark_interval)
    x = np.asarray([row["hold_s"] for row in band])
    axes[0].fill_between(x, [row["lower"] for row in band], [row["upper"] for row in band], color=ACCENT, alpha=0.16)
    axes[0].plot(x, [row["prediction"] for row in band], color=ACCENT)
    axes[0].errorbar(
        [row["sweep_value_s"] for row in curve],
        [row["estimate"] for row in curve],
        yerr=[
            [row["estimate"] - row["cluster_lower"] for row in curve],
            [row["cluster_upper"] - row["estimate"] for row in curve],
        ],
        fmt="o",
        ms=3.6,
        color=INK,
        capsize=2,
    )
    axes[0].set_title(
        f"Dark lifetime {_estimate(dark['tau_switch_off']):.1f} s"
    )
    axes[0].set_xlabel("fully-dark hold (s)")
    axes[0].set_ylabel("apparent retention")

    bband = bright["model_band"]
    bcurve = bright["clustered_apparent_occupancy_curve"]
    x = np.asarray([row["wait_s"] for row in bband])
    axes[1].fill_between(x, [row["lower"] for row in bband], [row["upper"] for row in bband], color=ACCENT_2, alpha=0.16)
    axes[1].plot(x, [row["prediction"] for row in bband], color=ACCENT_2)
    axes[1].errorbar(
        [row["sweep_value_s"] for row in bcurve],
        [row["estimate"] for row in bcurve],
        yerr=[
            [row["estimate"] - row["cluster_lower"] for row in bcurve],
            [row["cluster_upper"] - row["estimate"] for row in bcurve],
        ],
        fmt="o",
        ms=3.6,
        color=INK,
        capsize=2,
    )
    axes[1].set_title(
        f"Bright lifetime {_estimate(bright['tau_bright_effective']):.1f} s"
    )
    axes[1].set_xlabel("continuous bright wait (s)")
    axes[1].set_ylabel("apparent occupancy")

    keys = ("50ms", "100ms", "200ms")
    xloc = np.arange(3)
    bottom = np.zeros(3)
    for component, label, color in (
        ("bright_exposure", "bright exposure", ACCENT_2),
        ("dark_gap", "dark gaps", ACCENT),
        ("residual_readout_associated", "residual", WARN),
    ):
        values = np.asarray(
            [budget["point_estimates"][key]["component_loss"][component] for key in keys]
        )
        axes[2].bar(xloc, 100 * values, bottom=100 * bottom, color=color, label=label)
        bottom += values
    axes[2].set_xticks(xloc, ["50 ms", "100 ms", "200 ms"])
    axes[2].set_ylabel("five-frame expected loss (%)")
    axes[2].set_title("Rate-based loss budget")
    axes[2].legend(fontsize=8.5)
    fig.suptitle("Optimized lifetime and loss results", fontsize=14, fontweight="bold")
    save(fig, path, dpi=170)


def _draw_segmentation(result: Mapping[str, Any], path: Path) -> None:
    apply_style()
    rows = result["repeated_imaging"]["heldout_matched_prefix_contrasts"]
    fig, ax = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    positions = []
    labels = []
    estimates = []
    lower = []
    upper = []
    colors = []
    position = 0
    for group in rows:
        for design in group["designs"][:-1]:
            positions.append(position)
            labels.append(
                f"{design['pulse_count']}×{1000 * design['exposure_s']:.0f} ms\nvs fewest pulses"
            )
            estimates.append(100 * design["difference_vs_fewest_pulses"])
            lower.append(100 * design["difference_cluster_lower"])
            upper.append(100 * design["difference_cluster_upper"])
            colors.append(EXPOSURE_COLORS[f"{int(1000 * design['exposure_s'])}ms"])
            position += 1
        position += 0.55
    estimate_array = np.asarray(estimates)
    ax.axhline(0, color=MUTED, lw=1)
    ax.errorbar(
        positions,
        estimate_array,
        yerr=[estimate_array - np.asarray(lower), np.asarray(upper) - estimate_array],
        fmt="none",
        ecolor=INK,
        capsize=4,
        lw=1.2,
    )
    ax.scatter(positions, estimates, c=colors, s=54, zorder=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("normalized apparent-occupancy difference (percentage points)")
    ax.set_title("Matched total illumination: held-out prefix contrasts")
    ax.text(
        0.01,
        0.02,
        "95% intervals resample complete held-out shots; cross-run intercepts are development-fit.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    save(fig, path, dpi=170)


def _draw_readout_tradeoff(result: Mapping[str, Any], path: Path) -> None:
    """Show the frozen readout-separation versus survival trade-off."""
    apply_style()
    exposures = np.asarray([50, 100, 200], dtype=float)
    keys = ("50ms", "100ms", "200ms")
    overlaps = np.asarray([
        100 * _selected_emission_metrics(
            result["repeated_imaging"]["emission_models_by_exposure"][key]
        )["mean_model_implied_overlap"]
        for key in keys
    ])
    survivals = np.asarray([
        100 * result["loss_budget"]["point_estimates"][key]["final_survival"]
        for key in keys
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    axes[0].plot(exposures, overlaps, color=ACCENT, marker="o", ms=8, lw=2.2)
    axes[0].set_title("A. Held-out component overlap")
    axes[0].set_xlabel("exposure duration (ms)")
    axes[0].set_ylabel("model-implied overlap (%)")
    axes[0].set_xticks(exposures)
    axes[0].set_ylim(0, max(overlaps) * 1.35)
    for x, value in zip(exposures, overlaps):
        axes[0].text(x, value + 0.035, f"{value:.2f}%", ha="center", fontsize=10,
                     fontweight="bold")
    axes[0].annotate("better count separation", xy=(190, overlaps[-1]),
                     xytext=(85, overlaps[0] * 1.22), color=ACCENT,
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.5))

    axes[1].plot(exposures, survivals, color=ACCENT_2, marker="s", ms=8, lw=2.2)
    axes[1].set_title("B. Selected-model five-frame survival")
    axes[1].set_xlabel("exposure duration per frame (ms)")
    axes[1].set_ylabel("cumulative survival after frame 5 (%)")
    axes[1].set_xticks(exposures)
    axes[1].set_ylim(min(survivals) - 1.4, 100.0)
    for x, value in zip(exposures, survivals):
        axes[1].text(x, value + 0.28, f"{value:.2f}%", ha="center", fontsize=10,
                     fontweight="bold")
    axes[1].annotate("greater destructive cost", xy=(190, survivals[-1]),
                     xytext=(70, 96.0), color=ACCENT_2,
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT_2, lw=1.5))
    fig.suptitle("Readout quality versus survival cost", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Frozen 2026-07-31 values. Overlap is model-implied, not empirical fidelity.",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    save(fig, path, dpi=170)


def _metrics_markdown(result: Mapping[str, Any]) -> str:
    dark = result["optimized_dark_bright"]["dark_hold"]["operational_model"]
    bright = result["optimized_dark_bright"]["bright_wait"]["effective_model"]
    dark_ci = dark["tau_switch_off"]
    bright_ci = bright["tau_bright_effective"]
    if not isinstance(dark_ci, Mapping):
        dark_ci = _interval(
            dark["bootstrap"]["intervals"], "tau_switch_off__shared"
        )
    if not isinstance(bright_ci, Mapping):
        bright_ci = _interval(
            bright["bootstrap"]["intervals"], "tau_bright_effective"
        )
    rows = []
    for exposure in ("50ms", "100ms", "200ms"):
        metrics = _selected_emission_metrics(
            result["repeated_imaging"]["emission_models_by_exposure"][exposure]
        )
        rows.append(
            f"| {exposure[:-2]} ms | {metrics['mean_count_nll']:.3f} | "
            f"{metrics['mean_separation_d_prime']:.2f} | "
            f"{100 * metrics['mean_model_implied_overlap']:.2f}% | "
            f"{metrics['mean_posterior_entropy']:.3f} |"
        )
    return "\n".join(
        [
            "<!-- generated by scripts/generate_optimized_assets.py; do not edit -->",
            "# Optimized 2026-07-31 README metrics",
            "",
            f"- Dark lifetime: **{_estimate(dark['tau_switch_off']):.2f} s** "
            f"(95% complete-shot CI {dark_ci['lower']:.2f}–{dark_ci['upper']:.2f} s).",
            f"- Bright/imaging lifetime: **{_estimate(bright['tau_bright_effective']):.2f} s** "
            f"(95% complete-shot CI {bright_ci['lower']:.2f}–{bright_ci['upper']:.2f} s).",
            f"- Repeated-imaging structure selected on validation: "
            f"**{result['repeated_imaging']['selected_model']}**.",
            f"- Pulse-associated public claim allowed: "
            f"**{str(result['repeated_imaging']['allowed_pulse_claim']).lower()}**.",
            "",
            "| Exposure | Held-out mean NLL | d-prime | Model-implied overlap | Posterior entropy |",
            "|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Overlap and posterior quantities are model-implied; no externally labelled occupancy truth is available.",
            "",
        ]
    )


def generate_assets(
    result_path: Path,
    output_dir: Path,
    *,
    metrics_path: Path | None = None,
    story_data: OccupancyStoryData | None = None,
    story_table_path: Path = DEFAULT_STORY_TABLE,
    story_raw_dir: Path = DEFAULT_STORY_RAW_DIR,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require_reviewed(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    story = story_data or load_occupancy_story_data(
        result, table_path=story_table_path, raw_dir=story_raw_dir
    )
    gif_path = output_dir / "optimized_occupancy_inference.gif"
    animation = _draw_occupancy_animation(result, story, gif_path)
    _draw_occupancy_summary(
        result, story, output_dir / "optimized_occupancy_inference.png"
    )
    _draw_sequence(output_dir / "optimized_sequence_design.png")
    _draw_lifetime_overview(result, output_dir / "optimized_lifetime_overview.png")
    _draw_segmentation(result, output_dir / "exposure_segmentation_result.png")
    _draw_readout_tradeoff(result, output_dir / "optimized_readout_tradeoff.png")
    paths = [output_dir / name for name in ASSET_NAMES]
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        text = _metrics_markdown(result)
        assert_public_safe(text, "optimized README metrics")
        metrics_path.write_text(text, encoding="utf-8")
    metadata = {
        "generated_by": "scripts/generate_optimized_assets.py",
        "result_sha256": _sha256(result_path),
        "analysis_code_commit": result["provenance"]["analysis_code_commit"],
        "animation": animation,
        "representative_selection": {
            "dataset_id": "imaging_5frame_100ms_20260731_0114",
            "split": "test",
            "shot_id": story.selected_shot_id,
            "site_id": story.selected_site_id,
            "shot_rule": story.shot_rule,
            "site_rule": story.site_rule,
            "uses_frozen_geometry_background_and_emissions": True,
        },
        "assets": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in paths
        },
    }
    metadata_path = output_dir / "optimized_asset_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--story-table", type=Path, default=DEFAULT_STORY_TABLE)
    parser.add_argument("--story-raw-dir", type=Path, default=DEFAULT_STORY_RAW_DIR)
    args = parser.parse_args()
    result = args.results if args.results.is_absolute() else ROOT / args.results
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    metrics = args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    story_table = args.story_table if args.story_table.is_absolute() else ROOT / args.story_table
    story_raw_dir = args.story_raw_dir if args.story_raw_dir.is_absolute() else ROOT / args.story_raw_dir
    metadata = generate_assets(
        result,
        output,
        metrics_path=metrics,
        story_table_path=story_table,
        story_raw_dir=story_raw_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
