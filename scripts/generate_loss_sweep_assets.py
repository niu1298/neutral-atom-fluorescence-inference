"""Generate deterministic public loss-sweep figures from one result JSON.

The renderer never opens processed tables or raw shots.  Its sole scientific
input is ``reports/loss_sweep_results.json`` (or ``--results``), which keeps
the public images coupled to the reviewed machine-readable analysis.

With no ``--output-dir``, the command publishes the figures and reviewed
Markdown fragments into the existing README marker regions and appends or
refreshes the marked V1 section of ``reports/readme_metrics.md``.  Supplying
``--output-dir`` is an isolated preview: figures and fragments are written
there without changing either tracked Markdown file.

Examples
--------
python scripts/generate_loss_sweep_assets.py
python scripts/generate_loss_sweep_assets.py \
    --results reports/loss_sweep_results.json \
    --output-dir reports/local_loss_sweep_preview
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from fluorescence_inference import reporting as rp  # noqa: E402


DEFAULT_RESULTS = Path("reports/loss_sweep_results.json")
DEFAULT_V0_QC = Path("reports/qc/qc_summary.json")
DEFAULT_OUTPUT_DIR = Path("assets/readme")
DEFAULT_README = Path("README.md")
DEFAULT_README_METRICS = Path("reports/readme_metrics.md")
METADATA_NAME = "loss_sweep_asset_metadata.json"
PUBLICATION_METADATA_NAME = "loss_sweep_publication_metadata.json"
README_MARKERS = (
    "experiment-matrix",
    "sweep-validation",
    "loss-sweep-results",
)
FRAGMENT_FILES = {
    "experiment-matrix": "experiment_matrix.md",
    "sweep-validation": "sweep_validation.md",
    "loss-sweep-results": "loss_sweep_results.md",
    "readme-metrics-v1": "readme_metrics_v1.md",
}
METRICS_V1_MARKER = "loss-sweeps-v1"
CORE_ASSETS = (
    "loss_sweep_overview.png",
    "dark_hold_retention.png",
    "bright_wait_decay.png",
    "background_drift_sweeps.png",
)
OPTIONAL_ASSETS = (
    "per_site_retention_map.png",
    "latent_state_example.png",
)

INTERVAL_COLORS = (rp.ACCENT_2, rp.ACCENT, rp.OK, "#7b6aa8", rp.WARN)
INTERVAL_MARKERS = ("s", "o", "^", "D", "v")
RETENTION_MAP_LIMITS = (0.0, 1.0)
INTERVAL_WIDTH_MAP_LIMITS = (0.0, 0.5)
FIGURE_METADATA = {
    "Software": "fluorescence-inference",
    "Description": "Generated from reviewed loss-sweep result JSON",
}


class AssetInputError(ValueError):
    """The result JSON lacks data required for an honest public figure."""


def _dig(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _records(
    value: Any,
    *,
    name: str,
    required: Sequence[str] = (),
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(value, list):
        raise AssetInputError(f"{name} must be a list of records")
    frame = pd.DataFrame(value)
    if frame.empty and not allow_empty:
        raise AssetInputError(f"{name} has no records")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssetInputError(f"{name} is missing columns: {missing}")
    return frame


def _finite(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise AssetInputError(f"{name}.{column} contains non-finite values")


def _probability_interval(
    frame: pd.DataFrame,
    *,
    estimate: str,
    lower: str,
    upper: str,
    name: str,
) -> None:
    _finite(frame, (estimate, lower, upper), name=name)
    point = frame[estimate].to_numpy(float)
    lo = frame[lower].to_numpy(float)
    hi = frame[upper].to_numpy(float)
    valid = (
        (lo >= -1e-12)
        & (lo <= point + 1e-12)
        & (point <= hi + 1e-12)
        & (hi <= 1.0 + 1e-12)
    )
    if not valid.all():
        raise AssetInputError(
            f"{name} must satisfy 0 <= lower <= estimate <= upper <= 1"
        )


def _unique_points(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    if frame.duplicated(list(columns)).any():
        raise AssetInputError(f"{name} contains duplicate points for {list(columns)}")


def _interval_sort_key(value: Any) -> tuple[int, int, str]:
    match = re.fullmatch(r"\s*(\d+)\s*->\s*(\d+)\s*", str(value))
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    return (10**9, 10**9, str(value))


def _style() -> None:
    rp.apply_style()
    rp.plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "lines.linewidth": 1.7,
        }
    )


def _safe_text(text: str, where: str) -> str:
    rp.assert_public_safe(text, where)
    return text


def _save(fig, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        facecolor=rp.PANEL,
        metadata=FIGURE_METADATA,
    )
    rp.plt.close(fig)
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": path.name,
        "width_px": int(width),
        "height_px": int(height),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _interval_label(value: Any) -> str:
    text = str(value)
    match = re.fullmatch(r"\s*(\d+)\s*->\s*(\d+)\s*", text)
    if match:
        label = (
            f"{int(match.group(1)) + 1}\N{RIGHTWARDS ARROW}"
            f"{int(match.group(2)) + 1}"
        )
    else:
        label = text.replace("->", "\N{RIGHTWARDS ARROW}")
    return _safe_text(label, "interval label")


def _shot_note(frame: pd.DataFrame) -> str:
    if "n_independent_shots" not in frame:
        return ""
    counts = sorted(
        {
            int(value)
            for value in frame["n_independent_shots"].dropna().tolist()
            if int(value) > 0
        }
    )
    if not counts:
        return ""
    if len(counts) == 1:
        return f"{counts[0]} independent shots per point"
    return f"{counts[0]}\N{EN DASH}{counts[-1]} independent shots per point"


def _axis_note(ax, text: str) -> None:
    if not text:
        return
    _safe_text(text, "axis note")
    ax.text(
        0.99,
        0.025,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=rp.MUTED,
    )


def _dark_inputs(results: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = _dig(results, "dark_hold", "operational_model")
    if not isinstance(model, Mapping):
        raise AssetInputError("dark_hold.operational_model is missing")
    curve = _records(
        model.get("clustered_apparent_retention_curve"),
        name="dark clustered retention curve",
        required=(
            "sweep_value_s",
            "interval",
            "estimate",
            "cluster_lower",
            "cluster_upper",
            "n_independent_shots",
        ),
    )
    band = _records(
        model.get("model_band"),
        name="dark model band",
        required=("hold_s", "interval", "prediction", "lower", "upper"),
    )
    fixed = _records(
        model.get("fixed_interreadout_survival"),
        name="fixed inter-readout survival",
        required=("interval", "estimate", "lower", "upper"),
    )
    _finite(
        curve,
        ("sweep_value_s",),
        name="dark curve",
    )
    _probability_interval(
        curve,
        estimate="estimate",
        lower="cluster_lower",
        upper="cluster_upper",
        name="dark curve",
    )
    _finite(band, ("hold_s",), name="dark band")
    _probability_interval(
        band,
        estimate="prediction",
        lower="lower",
        upper="upper",
        name="dark band",
    )
    _probability_interval(
        fixed,
        estimate="estimate",
        lower="lower",
        upper="upper",
        name="fixed survival",
    )
    _unique_points(
        curve, ("interval", "sweep_value_s"), name="dark retention curve"
    )
    _unique_points(band, ("interval", "hold_s"), name="dark model band")
    _unique_points(fixed, ("interval",), name="fixed survival")
    curve_intervals = set(map(str, curve["interval"]))
    if set(map(str, band["interval"])) != curve_intervals or set(
        map(str, fixed["interval"])
    ) != curve_intervals:
        raise AssetInputError(
            "dark curve, band, and fixed-survival intervals do not match"
        )
    sweep_sets = [
        tuple(sorted(group["sweep_value_s"].to_numpy(float)))
        for _, group in curve.groupby("interval", observed=True)
    ]
    if len(set(sweep_sets)) != 1:
        raise AssetInputError("dark intervals do not share the same sweep points")
    expected = _dig(results, "dataset_audit", "dark_hold", "sweep_values_s")
    if isinstance(expected, list):
        observed_values = np.asarray(sweep_sets[0], dtype=float)
        expected_values = np.sort(np.asarray(expected, dtype=float))
        if observed_values.shape != expected_values.shape or not np.allclose(
            observed_values, expected_values
        ):
            raise AssetInputError(
                "dark curve does not cover the audited sweep points"
            )
    return curve, band, fixed


def _bright_inputs(
    results: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = _dig(results, "bright_wait", "effective_model")
    if not isinstance(model, Mapping):
        raise AssetInputError("bright_wait.effective_model is missing")
    curve = _records(
        model.get("clustered_apparent_occupancy_curve"),
        name="bright apparent-occupancy curve",
        required=(
            "sweep_value_s",
            "estimate",
            "cluster_lower",
            "cluster_upper",
            "n_independent_shots",
        ),
    )
    band = _records(
        model.get("model_band"),
        name="bright model band",
        required=("wait_s", "prediction", "lower", "upper"),
    )
    control = model.get("post_wait_control")
    if not isinstance(control, Mapping):
        raise AssetInputError("bright post_wait_control is missing")
    control_curve = _records(
        control.get("clustered_curve"),
        name="post-wait retention curve",
        required=(
            "sweep_value_s",
            "estimate",
            "cluster_lower",
            "cluster_upper",
            "n_independent_shots",
        ),
    )
    control_band = _records(
        control.get("model_band"),
        name="post-wait retention band",
        required=("wait_s", "prediction", "lower", "upper"),
    )
    _finite(
        curve,
        ("sweep_value_s",),
        name="bright curve",
    )
    _probability_interval(
        curve,
        estimate="estimate",
        lower="cluster_lower",
        upper="cluster_upper",
        name="bright curve",
    )
    _finite(band, ("wait_s",), name="bright band")
    _probability_interval(
        band,
        estimate="prediction",
        lower="lower",
        upper="upper",
        name="bright band",
    )
    _finite(
        control_curve,
        ("sweep_value_s",),
        name="control curve",
    )
    _probability_interval(
        control_curve,
        estimate="estimate",
        lower="cluster_lower",
        upper="cluster_upper",
        name="control curve",
    )
    _finite(control_band, ("wait_s",), name="control band")
    _probability_interval(
        control_band,
        estimate="prediction",
        lower="lower",
        upper="upper",
        name="control band",
    )
    _unique_points(curve, ("sweep_value_s",), name="bright curve")
    _unique_points(band, ("wait_s",), name="bright band")
    _unique_points(control_curve, ("sweep_value_s",), name="control curve")
    _unique_points(control_band, ("wait_s",), name="control band")
    bright_points = np.sort(curve["sweep_value_s"].to_numpy(float))
    control_points = np.sort(control_curve["sweep_value_s"].to_numpy(float))
    if bright_points.shape != control_points.shape or not np.allclose(
        bright_points, control_points
    ):
        raise AssetInputError(
            "bright occupancy and post-wait control use different sweep points"
        )
    expected = _dig(results, "dataset_audit", "bright_wait", "sweep_values_s")
    if isinstance(expected, list):
        expected_values = np.sort(np.asarray(expected, dtype=float))
        if bright_points.shape != expected_values.shape or not np.allclose(
            bright_points, expected_values
        ):
            raise AssetInputError(
                "bright curves do not cover the audited sweep points"
            )
    return curve, band, control_curve, control_band


def _interval_styles(levels: Sequence[Any]) -> dict[Any, tuple[str, str]]:
    return {
        level: (
            INTERVAL_COLORS[index % len(INTERVAL_COLORS)],
            INTERVAL_MARKERS[index % len(INTERVAL_MARKERS)],
        )
        for index, level in enumerate(levels)
    }


def _draw_dark_retention(
    ax,
    curve: pd.DataFrame,
    band: pd.DataFrame,
    *,
    title: str,
    legend: bool = True,
) -> None:
    levels = sorted(curve["interval"].drop_duplicates(), key=_interval_sort_key)
    styles = _interval_styles(levels)
    for interval in levels:
        color, marker = styles[interval]
        points = curve.loc[curve["interval"] == interval].sort_values(
            "sweep_value_s", kind="stable"
        )
        fitted = band.loc[band["interval"].astype(str) == str(interval)].sort_values(
            "hold_s", kind="stable"
        )
        if not fitted.empty:
            ax.fill_between(
                fitted["hold_s"].to_numpy(float),
                fitted["lower"].to_numpy(float),
                fitted["upper"].to_numpy(float),
                color=color,
                alpha=0.11,
                linewidth=0,
            )
            ax.plot(
                fitted["hold_s"],
                fitted["prediction"],
                color=color,
                alpha=0.9,
            )
        y = points["estimate"].to_numpy(float)
        lower = points["cluster_lower"].to_numpy(float)
        upper = points["cluster_upper"].to_numpy(float)
        ax.errorbar(
            points["sweep_value_s"],
            y,
            yerr=np.vstack([y - lower, upper - y]),
            fmt=marker,
            ms=4.6,
            color=color,
            mec=color,
            mfc=rp.PANEL,
            capsize=2.3,
            elinewidth=1.0,
            label=_interval_label(interval),
            zorder=4,
        )
    ax.set_title(_safe_text(title, "dark title"), loc="left")
    ax.set_xlabel("switch-off hold (s)")
    ax.set_ylabel("apparent retention")
    ax.set_ylim(0.0, 1.025)
    if legend:
        ax.legend(title="readout interval", ncol=2, loc="lower left")
    _axis_note(ax, _shot_note(curve))


def _draw_fixed_survival(
    ax, fixed: pd.DataFrame, *, title: str, shot_note: str = ""
) -> None:
    fixed = fixed.copy()
    fixed["_sort_key"] = fixed["interval"].map(_interval_sort_key)
    fixed = fixed.sort_values("_sort_key", kind="stable").reset_index(drop=True)
    fixed["_order"] = range(len(fixed))
    first_interval = min(fixed["interval"], key=_interval_sort_key)
    x = fixed["_order"].to_numpy(float)
    estimate = fixed["estimate"].to_numpy(float)
    lower = fixed["lower"].to_numpy(float)
    upper = fixed["upper"].to_numpy(float)
    for index, row in fixed.iterrows():
        first = row["interval"] == first_interval
        color = rp.ACCENT_2 if first else rp.ACCENT
        marker = "s" if first else "o"
        ax.errorbar(
            row["_order"],
            row["estimate"],
            yerr=np.array(
                [[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]]
            ),
            fmt=marker,
            color=color,
            mfc=rp.PANEL,
            mec=color,
            ms=7,
            capsize=3,
            elinewidth=1.3,
        )
    ax.set_xticks(x, [_interval_label(v) for v in fixed["interval"]])
    ax.set_xlim(-0.55, max(len(fixed) - 0.45, 0.55))
    pad = max(0.025, 0.15 * float(np.ptp(np.r_[lower, upper])))
    ax.set_ylim(max(0.0, float(lower.min() - pad)), min(1.01, float(upper.max() + pad)))
    ax.set_ylabel("fixed retention factor")
    ax.set_xlabel("readout interval")
    ax.set_title(_safe_text(title, "fixed survival title"), loc="left")
    ax.text(
        0.02,
        0.04,
        "first interval highlighted",
        transform=ax.transAxes,
        fontsize=8.2,
        color=rp.MUTED,
        ha="left",
    )
    _axis_note(ax, shot_note)


def _draw_single_curve(
    ax,
    curve: pd.DataFrame,
    band: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    points = curve.sort_values("sweep_value_s", kind="stable")
    fitted = band.sort_values("wait_s", kind="stable")
    ax.fill_between(
        fitted["wait_s"].to_numpy(float),
        fitted["lower"].to_numpy(float),
        fitted["upper"].to_numpy(float),
        color=color,
        alpha=0.16,
        linewidth=0,
        label="95% clustered fit band",
    )
    ax.plot(fitted["wait_s"], fitted["prediction"], color=color)
    y = points["estimate"].to_numpy(float)
    lower = points["cluster_lower"].to_numpy(float)
    upper = points["cluster_upper"].to_numpy(float)
    ax.errorbar(
        points["sweep_value_s"],
        y,
        yerr=np.vstack([y - lower, upper - y]),
        fmt="o",
        ms=5.0,
        color=color,
        mfc=rp.PANEL,
        mec=color,
        capsize=2.4,
        elinewidth=1.1,
        label="apparent estimate (95% cluster CI)",
        zorder=4,
    )
    ax.set_title(_safe_text(title, "curve title"), loc="left")
    ax.set_xlabel(_safe_text(xlabel, "x label"))
    ax.set_ylabel(_safe_text(ylabel, "y label"))
    ax.set_ylim(0.0, 1.025)
    ax.legend(loc="lower left")
    _axis_note(ax, _shot_note(curve))


def _gate_note(results: Mapping[str, Any]) -> str:
    dark = bool(
        _dig(
            results,
            "dark_hold",
            "operational_model",
            "public_rate_claim_gate_passed",
            default=False,
        )
    )
    bright = bool(
        _dig(
            results,
            "bright_wait",
            "effective_model",
            "public_rate_claim_gate_passed",
            default=False,
        )
    )
    if dark and bright:
        return (
            "Intervals are shot-clustered. Rates are operational for the stated "
            "switch-off and illuminated-wait configurations."
        )
    return (
        "Operational apparent-occupancy estimates; at least one timing, geometry, "
        "background, or rate-resolution claim gate is incomplete."
    )


def _dark_development_shot_note(results: Mapping[str, Any]) -> str:
    counts = _dig(
        results,
        "dark_hold",
        "emission_baselines",
        "split_counts",
        default={},
    )
    if not isinstance(counts, Mapping):
        return ""
    train = counts.get("train")
    validation = counts.get("validation")
    if train is None or validation is None:
        return ""
    total = int(train) + int(validation)
    return f"{total} independent development shots"


def make_overview(results: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], list[str]]:
    dark_curve, dark_band, fixed = _dark_inputs(results)
    bright_curve, bright_band, control_curve, control_band = _bright_inputs(results)
    fig, axes = rp.plt.subplots(2, 2, figsize=(12.4, 8.2), layout="constrained")
    _draw_dark_retention(
        axes[0, 0], dark_curve, dark_band, title="A  Switch-off retention"
    )
    _draw_fixed_survival(
        axes[0, 1],
        fixed,
        title="B  Fixed interval retention",
        shot_note=_dark_development_shot_note(results),
    )
    _draw_single_curve(
        axes[1, 0],
        bright_curve,
        bright_band,
        title="C  Bright-wait occupancy",
        xlabel="prior bright wait (s)",
        ylabel="apparent occupancy",
        color=rp.ACCENT,
    )
    _draw_single_curve(
        axes[1, 1],
        control_curve,
        control_band,
        title="D  Post-wait retention",
        xlabel="prior bright wait (s)",
        ylabel="apparent retention",
        color=rp.ACCENT_2,
    )
    title = _safe_text("Loss-sweep overview", "overview title")
    fig.suptitle(title, fontsize=16, fontweight="bold", color=rp.INK)
    note = _safe_text(_gate_note(results), "overview footnote")
    fig.text(0.005, -0.018, note, ha="left", va="top", fontsize=8.2, color=rp.MUTED)
    visible = [
        title,
        note,
        "A  Switch-off retention",
        "B  Fixed interval retention",
        "C  Bright-wait occupancy",
        "D  Post-wait retention",
        "switch-off hold (s)",
        "prior bright wait (s)",
        "apparent retention",
        "apparent occupancy",
        "fixed retention factor",
        "readout interval",
        "first interval highlighted",
        "95% clustered fit band",
        "apparent estimate (95% cluster CI)",
        _shot_note(dark_curve),
        _shot_note(bright_curve),
        _shot_note(control_curve),
        _dark_development_shot_note(results),
        *[_interval_label(value) for value in fixed["interval"]],
    ]
    return _save(fig, output), [text for text in visible if text]


def make_dark_figure(
    results: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[str]]:
    curve, band, _fixed = _dark_inputs(results)
    fig, ax = rp.plt.subplots(figsize=(9.4, 5.8), layout="constrained")
    title = "Operational switch-off retention"
    _draw_dark_retention(ax, curve, band, title=title)
    note = (
        "Hold is defined by the commanded switch-off configuration; this is not "
        "an intrinsic fully dark lifetime."
    )
    fig.text(
        0.005,
        -0.018,
        _safe_text(note, "dark footnote"),
        ha="left",
        va="top",
        fontsize=8.2,
        color=rp.MUTED,
    )
    visible = [
        title,
        note,
        "switch-off hold (s)",
        "apparent retention",
        "readout interval",
        _shot_note(curve),
        *[_interval_label(value) for value in curve["interval"].drop_duplicates()],
    ]
    return _save(fig, output), [text for text in visible if text]


def make_bright_figure(
    results: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[str]]:
    curve, band, control_curve, control_band = _bright_inputs(results)
    fig, axes = rp.plt.subplots(1, 2, figsize=(12.0, 5.2), layout="constrained")
    _draw_single_curve(
        axes[0],
        curve,
        band,
        title="Bright-wait occupancy",
        xlabel="prior bright wait (s)",
        ylabel="apparent occupancy",
        color=rp.ACCENT,
    )
    _draw_single_curve(
        axes[1],
        control_curve,
        control_band,
        title="Post-wait retention",
        xlabel="prior bright wait (s)",
        ylabel="apparent retention",
        color=rp.ACCENT_2,
    )
    note = (
        "Effective illuminated-wait decay and the post-wait retention control; "
        "neither panel establishes an intrinsic bright-state lifetime or a "
        "heating mechanism."
    )
    fig.text(
        0.005,
        -0.018,
        _safe_text(note, "bright footnote"),
        ha="left",
        va="top",
        fontsize=8.2,
        color=rp.MUTED,
    )
    return _save(fig, output), [
        "Bright-wait occupancy",
        "Post-wait retention",
        "prior bright wait (s)",
        "apparent occupancy",
        "apparent retention",
        "95% clustered fit band",
        "apparent estimate (95% cluster CI)",
        _shot_note(curve),
        _shot_note(control_curve),
        note,
    ]


def _optional_background_curve(results: Mapping[str, Any]) -> pd.DataFrame | None:
    nested: list[pd.DataFrame] = []
    nested_seen = False
    for dataset_key, model_key in (
        ("dark_hold", "operational_model"),
        ("bright_wait", "effective_model"),
    ):
        block = _dig(
            results,
            dataset_key,
            model_key,
            "site_free_background_curve",
        )
        if block is None:
            continue
        nested_seen = True
        if not isinstance(block, Mapping) or not bool(block.get("available")):
            raise AssetInputError(
                f"{dataset_key} site-free background curve is unavailable"
            )
        curve = _records(
            block.get("curve"),
            name=f"{dataset_key} site-free background curve",
            required=(
                "condition_id",
                "sweep_value_s",
                "frame_index",
                "estimate",
                "cluster_lower",
                "cluster_upper",
                "n_independent_shots",
            ),
        ).rename(
            columns={
                "cluster_lower": "lower",
                "cluster_upper": "upper",
            }
        )
        selected_method = _dig(
            results,
            "background_evidence",
            dataset_key,
            "selection",
            "selected_method",
        )
        allowed_columns = {
            "global": {"background_global"},
            "spatial": {"background_spatial"},
            "template": {
                "background_template_offset",
                "background_fixed_offset",
            },
        }
        background_column = str(block.get("background_column", ""))
        if (
            selected_method in allowed_columns
            and background_column not in allowed_columns[str(selected_method)]
        ):
            raise AssetInputError(
                f"{dataset_key} background curve column {background_column!r} "
                f"does not match selected method {selected_method!r}"
            )
        curve["dataset"] = dataset_key
        curve["background_method"] = str(
            _dig(
                results,
                "background_evidence",
                dataset_key,
                "selection",
                "selected_method",
                default=block.get("background_column", "selected"),
            )
        )
        curve["background_column"] = background_column
        curve["units"] = str(block.get("units", "ROI counts"))
        nested.append(curve)
    if nested_seen:
        if len(nested) != 2:
            raise AssetInputError(
                "selected-method background curves must exist for both sweeps"
            )
        frame = pd.concat(nested, ignore_index=True)
        _finite(
            frame,
            (
                "frame_index",
                "sweep_value_s",
                "estimate",
                "lower",
                "upper",
                "n_independent_shots",
            ),
            name="site-free background curve",
        )
        if np.any(frame["lower"].to_numpy(float) > frame["estimate"].to_numpy(float)) or np.any(
            frame["estimate"].to_numpy(float) > frame["upper"].to_numpy(float)
        ):
            raise AssetInputError(
                "background curve bounds must enclose the estimate"
            )
        _unique_points(
            frame,
            ("dataset", "frame_index", "sweep_value_s"),
            name="site-free background curve",
        )
        return frame

    for path in (
        ("public_plot_data", "background_drift_sweeps"),
        ("background_drift_sweeps",),
        ("background_drift",),
    ):
        value = _dig(results, *path)
        if isinstance(value, list) and value:
            frame = _records(
                value,
                name="background drift",
                required=(
                    "dataset",
                    "frame_index",
                    "sweep_value_s",
                    "estimate",
                    "lower",
                    "upper",
                    "n_independent_shots",
                    "background_method",
                ),
            )
            _finite(
                frame,
                ("frame_index", "sweep_value_s", "estimate", "lower", "upper"),
                name="background drift",
            )
            return frame
    return None


def _background_endpoint_rows(
    results: Mapping[str, Any], dataset_key: str
) -> pd.DataFrame:
    model_key = "operational_model" if dataset_key == "dark_hold" else "effective_model"
    coupling = _dig(
        results,
        dataset_key,
        model_key,
        "site_free_background_occupancy_coupling",
        default={},
    )
    rows = []
    selected_method = _dig(
        results,
        "background_evidence",
        dataset_key,
        "selection",
        "selected_method",
    )
    method_columns = {
        "global": ("background_global",),
        "spatial": ("background_spatial",),
        "template": (
            "background_template_offset",
            "background_fixed_offset",
        ),
    }
    if selected_method is not None and str(selected_method) not in method_columns:
        raise AssetInputError(
            f"background evidence selects ineligible method {selected_method!r}"
        )
    preferred = method_columns.get(str(selected_method), ())
    if isinstance(coupling, Mapping):
        for frame, methods in coupling.items():
            if not isinstance(methods, Mapping):
                continue
            selected_name = next(
                (
                    name
                    for name in (
                        *preferred,
                        "background_global",
                        "background_template_offset",
                        "background_fixed_offset",
                        "background_spatial",
                    )
                    if name in methods
                ),
                None,
            )
            if selected_name is None:
                continue
            block = methods[selected_name]
            if not isinstance(block, Mapping):
                continue
            value = block.get("background_change_shortest_to_longest")
            if value is None or not np.isfinite(float(value)):
                continue
            rows.append(
                {
                    "frame_index": int(frame),
                    "estimate": float(value),
                    "method": selected_name,
                }
            )
    return pd.DataFrame(rows)


def _pedestal_note(results: Mapping[str, Any], dataset_key: str) -> str:
    pedestal = _dig(
        results,
        "background_evidence",
        dataset_key,
        "site_free_frame_pedestal",
        default={},
    )
    if not isinstance(pedestal, Mapping) or not pedestal.get("available"):
        return ""
    spread = pedestal.get("median_spread_max_minus_min_roi_counts")
    frames = pedestal.get("frames", {})
    counts = sorted(
        {
            int(block["n_shots"])
            for block in frames.values()
            if isinstance(block, Mapping) and block.get("n_shots") is not None
        }
    )
    pieces = []
    if spread is not None and np.isfinite(float(spread)):
        pieces.append(f"frame pedestal spread {float(spread):.0f} ROI counts")
    if counts:
        pieces.append(f"{counts[0]} shots")
    return " \N{MIDDLE DOT} ".join(pieces)


def make_background_figure(
    results: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[str]]:
    richer = _optional_background_curve(results)
    fig, axes = rp.plt.subplots(
        1, 2, figsize=(11.8, 4.9), sharey=True, layout="constrained"
    )
    visible = []
    datasets = (("dark_hold", "Switch-off sweep"), ("bright_wait", "Bright-wait sweep"))
    if richer is not None:
        for ax, (key, title) in zip(axes, datasets):
            aliases = {key, key.replace("_", " "), "dark" if key == "dark_hold" else "bright"}
            part = richer.loc[richer["dataset"].astype(str).str.lower().isin(aliases)]
            if part.empty:
                raise AssetInputError(f"background drift has no rows for {key}")
            selected_method = _dig(
                results,
                "background_evidence",
                key,
                "selection",
                "selected_method",
            )
            if selected_method is not None and set(
                part["background_method"].astype(str)
            ) != {str(selected_method)}:
                raise AssetInputError(
                    f"background curve for {key} does not use the selected method"
                )
            for frame, block in part.groupby("frame_index", observed=True, sort=True):
                block = block.sort_values("sweep_value_s", kind="stable")
                y = block["estimate"].to_numpy(float)
                lower = block["lower"].to_numpy(float)
                upper = block["upper"].to_numpy(float)
                color = INTERVAL_COLORS[int(frame) % len(INTERVAL_COLORS)]
                marker = INTERVAL_MARKERS[int(frame) % len(INTERVAL_MARKERS)]
                line_style = ("-", "--", "-.", ":")[int(frame) % 4]
                ax.errorbar(
                    block["sweep_value_s"],
                    y,
                    yerr=np.vstack([y - lower, upper - y]),
                    fmt=marker,
                    ls=line_style,
                    ms=4,
                    capsize=2,
                    color=color,
                    label=f"frame {int(frame) + 1}",
                )
            ax.set_title(_safe_text(title, "background title"), loc="left")
            ax.set_xlabel("sweep value (s)")
            ax.set_ylabel("site-free background (ROI counts)")
            ax.legend(ncol=2)
            _axis_note(ax, _shot_note(part))
            visible.extend(
                [
                    title,
                    "sweep value (s)",
                    "site-free background (ROI counts)",
                    _shot_note(part),
                    *[
                        f"frame {int(frame) + 1}"
                        for frame in sorted(part["frame_index"].unique())
                    ],
                ]
            )
        note = "Points show 95% complete-shot cluster intervals."
    else:
        for ax, (key, title) in zip(axes, datasets):
            part = _background_endpoint_rows(results, key)
            if part.empty:
                raise AssetInputError(
                    "result JSON contains neither background drift curves nor "
                    f"endpoint diagnostics for {key}"
                )
            x = np.arange(len(part))
            colors = [
                INTERVAL_COLORS[int(frame) % len(INTERVAL_COLORS)]
                for frame in part["frame_index"]
            ]
            ax.axhline(0.0, color=rp.MUTED, lw=1.0, ls=":")
            ax.vlines(x, 0.0, part["estimate"], color=colors, lw=2.0)
            ax.scatter(
                x,
                part["estimate"],
                c=colors,
                s=44,
                edgecolors=rp.PANEL,
                linewidths=0.8,
                zorder=3,
            )
            ax.set_xticks(
                x, [f"frame {int(frame) + 1}" for frame in part["frame_index"]]
            )
            ax.set_title(_safe_text(title, "background title"), loc="left")
            ax.set_xlabel("fluorescence frame")
            ax.set_ylabel("longest \N{MINUS SIGN} shortest (ROI counts)")
            _axis_note(ax, _pedestal_note(results, key))
            visible.extend(
                [
                    title,
                    "fluorescence frame",
                    "longest \N{MINUS SIGN} shortest (ROI counts)",
                    _pedestal_note(results, key),
                    *[
                        f"frame {int(frame) + 1}"
                        for frame in part["frame_index"]
                    ],
                ]
            )
        note = (
            "Endpoint diagnostic from site-free background: longest minus shortest "
            "sweep point. The result JSON does not provide clustered intervals."
        )
    fig.suptitle(
        _safe_text("Background drift across loss sweeps", "background suptitle"),
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.005,
        -0.018,
        _safe_text(note, "background footnote"),
        ha="left",
        va="top",
        fontsize=8.2,
        color=rp.MUTED,
    )
    return _save(fig, output), [
        "Background drift across loss sweeps",
        *visible,
        note,
    ]


def _optional_per_site_records(results: Mapping[str, Any]) -> pd.DataFrame | None:
    candidates = (
        (_dig(results, "public_plot_data", "per_site_retention_map"), True),
        (
            _dig(
                results,
                "dark_hold",
                "operational_model",
                "heldout_per_site_apparent_retention",
            ),
            True,
        ),
        (
            _dig(
                results, "dark_hold", "operational_model", "per_site_retention_map"
            ),
            False,
        ),
        (
            _dig(results, "dark_hold", "operational_model", "per_site_retention"),
            False,
        ),
        (_dig(results, "per_site_retention_map"), False),
    )
    for value, strict in candidates:
        if not isinstance(value, list) or not value:
            continue
        frame = pd.DataFrame(value)
        estimate = next(
            (
                name
                for name in ("estimate", "apparent_retention", "retention")
                if name in frame
            ),
            None,
        )
        if estimate is None:
            raise AssetInputError("per-site retention rows lack an estimate column")
        frame = frame.rename(columns={estimate: "estimate"})
        if strict:
            required = {
                "cluster_lower",
                "cluster_upper",
                "n_independent_shots",
            }
            missing = sorted(required - set(frame.columns))
            if missing:
                raise AssetInputError(
                    f"held-out per-site rows are missing columns: {missing}"
                )
        if {"site_col", "site_row"}.issubset(frame.columns):
            frame["_x"] = frame["site_col"]
            frame["_y"] = frame["site_row"]
        elif {"site_x", "site_y"}.issubset(frame.columns):
            frame["_x"] = frame["site_x"]
            frame["_y"] = frame["site_y"]
        else:
            raise AssetInputError(
                "per-site retention rows need site row/column or x/y coordinates"
            )
        if frame.duplicated(["_x", "_y"]).any():
            raise AssetInputError(
                "per-site retention rows must contain one predeclared value per site"
            )
        _finite(frame, ("_x", "_y", "estimate"), name="per-site retention")
        if strict:
            _probability_interval(
                frame,
                estimate="estimate",
                lower="cluster_lower",
                upper="cluster_upper",
                name="per-site retention",
            )
        return frame
    return None


def make_per_site_figure(
    data: pd.DataFrame, output: Path
) -> tuple[dict[str, Any], list[str]]:
    aggregation = {"estimate": "first"}
    lower_col = next(
        (name for name in ("cluster_lower", "lower") if name in data), None
    )
    upper_col = next(
        (name for name in ("cluster_upper", "upper") if name in data), None
    )
    if lower_col and upper_col:
        aggregation.update({lower_col: "first", upper_col: "first"})
    grouped = data.groupby(["_x", "_y"], observed=True, as_index=False).agg(
        aggregation
    )
    has_intervals = bool(lower_col and upper_col)
    n_panels = 2 if has_intervals else 1
    fig, axes = rp.plt.subplots(
        1,
        n_panels,
        figsize=(11.2 if has_intervals else 6.8, 5.5),
        squeeze=False,
        layout="constrained",
    )
    ax = axes[0, 0]
    image = ax.scatter(
        grouped["_x"], grouped["_y"], c=grouped["estimate"], cmap=rp.SEQ_CMAP,
        vmin=RETENTION_MAP_LIMITS[0], vmax=RETENTION_MAP_LIMITS[1],
        marker="s", s=150, edgecolors=rp.PANEL,
        linewidths=0.7,
    )
    for axis in axes.ravel():
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.grid(False)
        axis.set_xlabel("site column")
    ax.set_ylabel("site row")
    title = _safe_text("Per-site apparent retention", "per-site title")
    ax.set_title(title, loc="left")
    rp.colorbar(fig, image, ax, "apparent retention")
    if has_intervals:
        width = (
            grouped[upper_col].to_numpy(float)
            - grouped[lower_col].to_numpy(float)
        )
        uncertainty = axes[0, 1].scatter(
            grouped["_x"],
            grouped["_y"],
            c=width,
            cmap="cividis",
            vmin=INTERVAL_WIDTH_MAP_LIMITS[0],
            vmax=INTERVAL_WIDTH_MAP_LIMITS[1],
            marker="s",
            s=150,
            edgecolors=rp.PANEL,
            linewidths=0.7,
        )
        axes[0, 1].set_title("95% cluster interval width", loc="left")
        rp.colorbar(
            fig, uncertainty, axes[0, 1], "interval width", format="%.2f"
        )
        shot_counts = sorted(
            {
                int(value)
                for value in data.get(
                    "n_independent_shots", pd.Series(dtype=float)
                ).dropna()
            }
        )
        if shot_counts:
            site_shot_note = (
                f"{shot_counts[0]} independent test shots per site"
                if len(shot_counts) == 1
                else (
                    f"{shot_counts[0]}\N{EN DASH}{shot_counts[-1]} independent "
                    "test shots per site"
                )
            )
        else:
            site_shot_note = ""
        note = (
            "Held-out apparent retention and complete-shot 95% interval width; "
            "fixed color scales are 0 to 1 and 0 to 0.5. "
            + (site_shot_note + ". " if site_shot_note else "")
            + "Sites describe heterogeneity, not independent experimental repeats."
        )
    else:
        note = (
            "One value per site; repeated interval rows, if supplied, are averaged "
            "without treating sites as independent shots."
        )
    fig.text(
        0.005,
        -0.018,
        _safe_text(note, "per-site footnote"),
        ha="left",
        va="top",
        fontsize=8.2,
        color=rp.MUTED,
    )
    labels = [title, note]
    if has_intervals:
        labels.append("95% cluster interval width")
    return _save(fig, output), labels


def _latent_public_gate_passed(latent: Mapping[str, Any]) -> bool:
    """Require clustered transition uncertainty, not a bare gate flag."""
    uncertainty = latent.get("uncertainty_scope")
    if not isinstance(uncertainty, Mapping):
        return False
    return bool(
        latent.get("gate_passed") is True
        and uncertainty.get("complete_shot_cluster_bootstrap_available") is True
        and uncertainty.get("public_gate_requires_complete_shot_clustering")
        is True
    )


def _latent_representative(results: Mapping[str, Any]) -> Mapping[str, Any] | None:
    latent = results.get("latent_state", {})
    if not isinstance(latent, Mapping) or not _latent_public_gate_passed(latent):
        return None
    representative = latent.get("representative_example")
    if not isinstance(representative, Mapping):
        return None
    required = (
        "observed_corrected_counts",
        "threshold_baseline_calls",
        "posterior_occupied_probability",
        "selection_rule",
    )
    if any(key not in representative for key in required):
        return None
    lengths = [
        len(representative[key])
        for key in required[:3]
        if isinstance(representative[key], list)
    ]
    if len(lengths) != 3 or len(set(lengths)) != 1 or lengths[0] < 2:
        raise AssetInputError("latent representative arrays must have equal length")
    return representative


def make_latent_figure(
    representative: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[str]]:
    counts = np.asarray(representative["observed_corrected_counts"], dtype=float)
    calls = np.asarray(representative["threshold_baseline_calls"], dtype=int)
    posterior = np.asarray(
        representative["posterior_occupied_probability"], dtype=float
    )
    if (
        not np.isfinite(counts).all()
        or not np.isfinite(posterior).all()
        or not np.isin(calls, [0, 1]).all()
    ):
        raise AssetInputError("latent representative contains invalid values")
    frame = np.arange(1, len(counts) + 1)
    fig, axes = rp.plt.subplots(
        2,
        1,
        figsize=(9.4, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
        layout="constrained",
    )
    axes[0].plot(frame, counts, color=rp.INK, marker="o", label="corrected count")
    for state in (0, 1):
        selected = calls == state
        if not selected.any():
            continue
        axes[0].scatter(
            frame[selected],
            counts[selected],
            marker="s",
            s=62,
            facecolors=rp.ACCENT if state else rp.PANEL,
            edgecolors=rp.ACCENT,
            linewidths=1.4,
            label=f"threshold call {state}",
            zorder=4,
        )
    axes[0].set_ylabel("corrected ROI count")
    axes[0].legend(ncol=3, loc="best")
    axes[1].plot(
        frame,
        posterior,
        color=rp.ACCENT_2,
        marker="o",
        mfc=rp.PANEL,
        label="posterior occupied probability",
    )
    axes[1].axhline(0.5, color=rp.MUTED, lw=1.0, ls=":")
    axes[1].fill_between(frame, 0.0, posterior, color=rp.ACCENT_2, alpha=0.10)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("posterior occupancy")
    axes[1].set_xlabel("fluorescence frame")
    axes[1].set_xticks(frame)
    title = _safe_text("Representative five-frame trajectory", "latent title")
    axes[0].set_title(title, loc="left")
    hold = representative.get("hold_s")
    note = "Held-out trajectory selected by the predeclared median-entropy rule"
    if hold is not None and np.isfinite(float(hold)):
        note += f" \N{MIDDLE DOT} switch-off hold {float(hold):.2g} s"
    note += ". Posterior occupancy is conditional on the accepted latent model."
    fig.text(
        0.005,
        -0.018,
        _safe_text(note, "latent footnote"),
        ha="left",
        va="top",
        fontsize=8.2,
        color=rp.MUTED,
    )
    return _save(fig, output), [title, note]


def _load_json_object(path: Path, *, name: str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.exists():
        raise AssetInputError(f"{name} does not exist: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AssetInputError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AssetInputError(f"{name} root must be an object")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetInputError(f"{name} must be an object")
    return value


def _number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssetInputError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise AssetInputError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str) -> int:
    number = _number(value, name=name)
    rounded = int(round(number))
    if not np.isclose(number, rounded):
        raise AssetInputError(f"{name} must be an integer")
    return rounded


def _single_number(value: Any, *, name: str) -> float:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AssetInputError(f"{name} must be a non-empty sequence")
    numbers = [_number(item, name=name) for item in value]
    if not numbers:
        raise AssetInputError(f"{name} must be a non-empty sequence")
    if not np.allclose(numbers, numbers[0]):
        raise AssetInputError(f"{name} must contain one unique value")
    return numbers[0]


def _fmt(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    number = _number(value, name="display value")
    text = f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"
    return text


def _fmt_percent(value: Any, digits: int = 1) -> str:
    return f"{100.0 * _number(value, name='probability'):.{digits}f}%"


def _fmt_ci(
    value: Any,
    *,
    digits: int = 3,
    scale: float = 1.0,
    suffix: str = "",
    percent: bool = False,
) -> str:
    record = _mapping(value, name="confidence interval")
    estimate = _number(record.get("estimate"), name="interval estimate")
    lower = _number(record.get("lower"), name="interval lower")
    upper = _number(record.get("upper"), name="interval upper")
    if lower > estimate or estimate > upper:
        raise AssetInputError("confidence interval must satisfy lower <= estimate <= upper")
    if percent:
        return (
            f"{100.0 * estimate:.{digits}f}% "
            f"({100.0 * lower:.{digits}f}\N{EN DASH}"
            f"{100.0 * upper:.{digits}f}%, 95% cluster CI)"
        )
    return (
        f"{scale * estimate:.{digits}f}{suffix} "
        f"({scale * lower:.{digits}f}\N{EN DASH}"
        f"{scale * upper:.{digits}f}{suffix}, 95% cluster CI)"
    )


def _md_cell(value: Any, *, name: str) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|")
    return _safe_text(text, name)


def _fragment(lines: Sequence[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines).strip() + "\n"
    return _safe_text(text, "generated README fragment")


def _friendly_background(value: Any) -> str:
    names = {
        "raw": "raw counts",
        "global": "global site-free median",
        "spatial": "robust spatial surface",
        "template": "template + frame offset",
        "annulus": "local annulus (diagnostic)",
    }
    key = str(value)
    return names.get(key, key.replace("_", " "))


def _friendly_model(value: Any) -> str:
    names = {
        "shared": "shared slope",
        "shared_late": "first interval separate; later slopes shared",
        "first_separate": "first interval separate; later slopes shared",
        "interval_specific": "interval-specific slopes",
        "no_floor": "no-floor exponential",
        "floor": "exponential with floor",
        "flat": "flat retention",
        "monotone": "constrained monotone retention",
    }
    key = str(value)
    return names.get(key, key.replace("_", " "))


def _experiment_row(
    label: str,
    *,
    shots: Any,
    frames: Any,
    exposure_ms: Any,
    sweep: str,
) -> str:
    exposure = _number(exposure_ms, name=f"{label} exposure")
    exposure_text = (
        f"{int(round(exposure))} ms"
        if np.isclose(exposure, round(exposure))
        else f"{exposure:g} ms"
    )
    return (
        f"| {_md_cell(label, name='dataset label')} "
        f"| {_integer(shots, name=f'{label} shots'):,} "
        f"| {_integer(frames, name=f'{label} frames'):,} "
        f"| {exposure_text} | {_md_cell(sweep, name=f'{label} sweep')} |"
    )


def _sweep_range(audit: Mapping[str, Any], *, name: str) -> str:
    values = audit.get("sweep_values_s")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AssetInputError(f"{name}.sweep_values_s must be a sequence")
    numeric = sorted({_number(value, name=f"{name} sweep value") for value in values})
    if not numeric:
        raise AssetInputError(f"{name}.sweep_values_s is empty")
    if len(numeric) == 1:
        return f"{numeric[0]:g} s"
    return f"{numeric[0]:g}\N{EN DASH}{numeric[-1]:g} s"


def render_experiment_matrix(
    results: Mapping[str, Any], v0_qc: Mapping[str, Any]
) -> str:
    """Create the README experiment matrix only from generated metadata."""
    paired = _mapping(_dig(v0_qc, "dataset"), name="V0 QC dataset")
    audits = _mapping(_dig(results, "dataset_audit"), name="dataset_audit")
    dark = _mapping(audits.get("dark_hold"), name="dataset_audit.dark_hold")
    bright = _mapping(audits.get("bright_wait"), name="dataset_audit.bright_wait")

    lines = [
        "| dataset | shots | frames | exposure | swept variable |",
        "|---|---:|---:|---:|---|",
        _experiment_row(
            "paired readout",
            shots=paired.get("n_shots"),
            frames=paired.get("n_frames_per_shot"),
            exposure_ms=paired.get("exposure_ms"),
            sweep="none",
        ),
        _experiment_row(
            "switch-off hold",
            shots=dark.get("n_complete_shots"),
            frames=dark.get("n_frames"),
            exposure_ms=1000.0
            * _single_number(
                dark.get("commanded_exposure_s"),
                name="dark_hold.commanded_exposure_s",
            ),
            sweep=_sweep_range(dark, name="dark_hold"),
        ),
        _experiment_row(
            "bright wait",
            shots=bright.get("n_complete_shots"),
            frames=bright.get("n_frames"),
            exposure_ms=1000.0
            * _single_number(
                bright.get("commanded_exposure_s"),
                name="bright_wait.commanded_exposure_s",
            ),
            sweep=_sweep_range(bright, name="bright_wait"),
        ),
    ]
    return _fragment(lines)


def _geometry_pitch(results: Mapping[str, Any], dataset: str) -> float:
    train = _mapping(
        _dig(results, "geometry_evidence", dataset, "subset_fits", "train"),
        name=f"geometry_evidence.{dataset}.subset_fits.train",
    )
    grids = train.get("grids")
    if not isinstance(grids, Sequence) or isinstance(grids, (str, bytes)) or not grids:
        raise AssetInputError(f"{dataset} training geometry has no grids")
    if len(grids) != 1:
        raise AssetInputError(f"{dataset} sweep geometry must contain one grid")
    grid = _mapping(grids[0], name=f"{dataset} training grid")
    pitches = [
        _number(grid.get(key), name=f"{dataset} {key}")
        for key in ("row_pitch_px", "col_pitch_px")
    ]
    return float(np.mean(pitches))


def _comparison_matching(
    results: Mapping[str, Any], dataset: str, comparison: str
) -> Mapping[str, Any]:
    record = _mapping(
        _dig(results, "geometry_evidence", dataset, "comparisons", comparison),
        name=f"{dataset}.{comparison}",
    )
    source = str(record.get("gate_source", "registered"))
    selected = record.get(source)
    if not isinstance(selected, Mapping):
        selected = record.get("registered")
    selected = _mapping(selected, name=f"{dataset}.{comparison}.{source}")
    return _mapping(
        selected.get("matching"), name=f"{dataset}.{comparison}.{source}.matching"
    )


def _validation_value(
    results: Mapping[str, Any], dataset: str, field: str
) -> Any:
    if field == "pitch":
        return f"{_geometry_pitch(results, dataset):.3f} px"
    if field == "early_late":
        matching = _comparison_matching(results, dataset, "early_vs_late")
        return (
            f"{_number(matching.get('median_px'), name='median shift'):.3f} / "
            f"{_number(matching.get('max_px'), name='maximum shift'):.3f} px"
        )
    if field == "short_long":
        matching = _comparison_matching(results, dataset, "shortest_vs_longest")
        return f"{_number(matching.get('median_px'), name='endpoint shift'):.3f} px"
    if field == "background":
        return _friendly_background(
            _dig(results, "background_evidence", dataset, "selection", "selected_method")
        )
    if field == "residual":
        value = _dig(
            results,
            "background_evidence",
            dataset,
            "selection",
            "selected_residual_structure",
        )
        return f"{_number(value, name='selected residual structure'):.2f} counts"
    if field == "annulus":
        evidence = _mapping(
            _dig(results, "background_evidence", dataset, "annulus_contamination"),
            name=f"{dataset} annulus evidence",
        )
        return (
            f"{_integer(evidence.get('sites_with_any_overlap'), name='overlap sites')} "
            f"/ {_integer(evidence.get('n_sites'), name='annulus sites')}"
        )
    if field == "pedestal":
        value = _dig(
            results,
            "background_evidence",
            dataset,
            "site_free_frame_pedestal",
            "first_minus_later_median_roi_counts",
        )
        return f"{_number(value, name='first frame pedestal'):+.0f} ROI counts"
    if field == "geometry_gate":
        passed = _dig(results, "geometry_evidence", dataset, "gate", "passed")
        return "passed" if passed is True else "failed"
    raise AssetInputError(f"unknown validation field: {field}")


def render_sweep_validation(results: Mapping[str, Any]) -> str:
    """Create the two-run validation table and its design caveats."""
    rows = (
        ("Geometry gate", "geometry_gate"),
        ("Training-shot lattice pitch", "pitch"),
        ("Early/late registered shift, median / max", "early_late"),
        ("Shortest/longest rigid-shift sensitivity", "short_long"),
        ("Selected background", "background"),
        ("Site-free residual structure, selected", "residual"),
        ("Annuli intersecting neighbouring-site masks", "annulus"),
        ("Site-free first-frame pedestal", "pedestal"),
    )
    lines = [
        "| Gate | Switch-off hold | Bright wait |",
        "|---|---:|---:|",
    ]
    for label, field in rows:
        lines.append(
            f"| {label} "
            f"| {_md_cell(_validation_value(results, 'dark_hold', field), name=label)} "
            f"| {_md_cell(_validation_value(results, 'bright_wait', field), name=label)} |"
        )

    cross = _mapping(
        _dig(results, "geometry_evidence", "cross_run"),
        name="geometry_evidence.cross_run",
    )
    shift = _number(
        _dig(cross, "wide_radius_matching", "bulk_shift", "magnitude_px"),
        name="cross-run coordinate shift",
    )
    interpretation = _md_cell(
        cross.get("interpretation"), name="cross-run geometry interpretation"
    )
    order_caveat = _md_cell(
        results.get("acquisition_order_caveat"), name="acquisition-order caveat"
    )
    comparison = _mapping(
        results.get("cross_dataset_comparison"), name="cross_dataset_comparison"
    )
    pooling = _md_cell(
        comparison.get("pooling_reason"), name="cross-run pooling reason"
    )
    lines.extend(
        [
            "",
            f"The fitted grids differ by {shift:.2f} px. {interpretation} "
            f"The sequences are not pooled: {pooling}",
            "",
            f"**Acquisition-order limitation.** {order_caveat}",
        ]
    )
    return _fragment(lines)


def _condition_split_text(
    results: Mapping[str, Any], dataset: str, split_counts: Mapping[str, Any]
) -> str:
    audit = _mapping(
        _dig(results, "dataset_audit", dataset), name=f"dataset_audit.{dataset}"
    )
    records = audit.get("condition_split_counts")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise AssetInputError(f"{dataset}.condition_split_counts must be records")
    by_split: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    for row_value in records:
        row = _mapping(row_value, name=f"{dataset} condition split row")
        split = str(row.get("split"))
        if split in by_split:
            by_split[split].add(
                _integer(row.get("n_independent_shots"), name=f"{dataset} {split} shots")
            )
    per_condition: list[str] = []
    totals: list[str] = []
    for split in ("train", "validation", "test"):
        values = sorted(by_split[split])
        if len(values) != 1:
            raise AssetInputError(
                f"{dataset} must have one {split} shot count per condition"
            )
        per_condition.append(str(values[0]))
        totals.append(
            str(_integer(split_counts.get(split), name=f"{dataset} total {split} shots"))
        )
    return (
        f"{'/'.join(per_condition)} per condition; "
        f"{'/'.join(totals)} total"
    )


def _selected_test_metrics(emissions: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = str(emissions.get("selected_model"))
    candidates = emissions.get("test_metrics_once_per_candidate")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise AssetInputError("test_metrics_once_per_candidate must be records")
    matches = [
        _mapping(row, name="held-out candidate metrics")
        for row in candidates
        if isinstance(row, Mapping)
        and str(row.get("candidate", row.get("model"))) == selected
    ]
    if len(matches) != 1:
        raise AssetInputError(
            f"expected one held-out metric row for selected model {selected!r}"
        )
    return matches[0]


def _emission_result_row(results: Mapping[str, Any], dataset: str, label: str) -> str:
    emissions = _mapping(
        _dig(results, dataset, "emission_baselines"),
        name=f"{dataset}.emission_baselines",
    )
    metrics = _selected_test_metrics(emissions)
    split_counts = _mapping(emissions.get("split_counts"), name=f"{dataset} split counts")
    split_text = _condition_split_text(results, dataset, split_counts)
    return (
        f"| {label} | {_md_cell(split_text, name='split description')} "
        f"| `{_md_cell(emissions.get('selected_model'), name='emission model')}` "
        f"| {_fmt(metrics.get('mean_count_nll'), 3)} "
        f"| {_fmt_percent(metrics.get('mean_model_implied_overlap'), 1)} "
        f"| {_fmt(metrics.get('mean_separation_d_prime'), 2)} "
        f"| {_fmt(metrics.get('mean_posterior_entropy'), 3)} |"
    )


def _interval_label_markdown(value: Any) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*->\s*(\d+)\s*", str(value))
    if not match:
        return _md_cell(value, name="interval")
    return (
        f"{int(match.group(1)) + 1}\N{RIGHTWARDS ARROW}"
        f"{int(match.group(2)) + 1}"
    )


def _model_rate_lines(
    model: Mapping[str, Any],
    *,
    rate_key: str,
    tau_key: str,
    rate_label: str,
    tau_label: str,
) -> list[str]:
    gate = model.get("public_rate_claim_gate_passed")
    resolved = model.get("rate_resolved")
    if gate is not True or resolved is not True:
        return [
            f"- **{rate_label}:** not reported; the public rate gate did not pass."
        ]
    return [
        f"- **{rate_label}:** "
        f"{_fmt_ci(model.get(rate_key), digits=4, suffix=' s^-1')}.",
        f"- **{tau_label}:** "
        f"{_fmt_ci(model.get(tau_key), digits=2, suffix=' s')}.",
    ]


def render_loss_sweep_results(results: Mapping[str, Any]) -> str:
    """Create the reader-facing V1 results with claim gates enforced."""
    lines = [
        "**Held-out count baselines.** Model choice used validation shots; the "
        "test metrics below were scored once.",
        "",
        "| dataset | train/validation/test shots | selected emission model "
        "| test mean NLL | model-implied overlap | d-prime | posterior entropy |",
        "|---|---|---|---:|---:|---:|---:|",
        _emission_result_row(results, "dark_hold", "switch-off hold"),
        _emission_result_row(results, "bright_wait", "bright wait"),
        "",
        "These count-model quantities are predictive and model-implied; they are "
        "**not** empirical fidelity, FPR, FNR, or labelled physical loss.",
        "",
    ]

    dark = _mapping(
        _dig(results, "dark_hold", "operational_model"),
        name="dark_hold.operational_model",
    )
    lines.extend(
        [
            "**Operational switch-off-hold model.** "
            f"Validation selected {_friendly_model(dark.get('selected_model'))}.",
            "",
            *_model_rate_lines(
                dark,
                rate_key="lambda_switch_off",
                tau_key="tau_switch_off",
                rate_label="`lambda_switch_off`",
                tau_label="`tau_switch_off`",
            ),
        ]
    )
    if dark.get("public_rate_claim_gate_passed") is True:
        fixed = dark.get("fixed_interreadout_survival")
        if not isinstance(fixed, Sequence) or isinstance(fixed, (str, bytes)):
            raise AssetInputError("fixed_interreadout_survival must be records")
        lines.extend(
            [
                "",
                "| interval | fixed inter-readout survival q_j |",
                "|---|---:|",
            ]
        )
        for row_value in sorted(
            fixed,
            key=lambda row: _interval_sort_key(
                row.get("interval") if isinstance(row, Mapping) else ""
            ),
        ):
            row = _mapping(row_value, name="fixed inter-readout survival")
            lines.append(
                f"| {_interval_label_markdown(row.get('interval'))} "
                f"| {_fmt_ci(row, digits=3)} |"
            )
    lines.extend(
        [
            "",
            "This is an operational decay under the switch-off command. DDS "
            "settings remain configured and cooling was not optimized, so it is "
            "not an intrinsic dark lifetime.",
            "",
        ]
    )

    bright = _mapping(
        _dig(results, "bright_wait", "effective_model"),
        name="bright_wait.effective_model",
    )
    lines.extend(
        [
            "**Effective bright-wait model.** "
            f"Validation selected {_friendly_model(bright.get('selected_model'))}.",
            "",
            *_model_rate_lines(
                bright,
                rate_key="lambda_bright_effective",
                tau_key="tau_bright_effective",
                rate_label="`lambda_bright_effective`",
                tau_label="`tau_bright_effective`",
            ),
        ]
    )
    control = _mapping(
        bright.get("post_wait_control"), name="bright post-wait control"
    )
    control_model = _friendly_model(control.get("selected_model"))
    if control.get("trend_resolved") is True:
        control_statement = (
            f"validation selected {control_model}, with a resolved "
            "wait-dependent post-wait retention trend"
        )
    else:
        control_statement = (
            f"validation selected {control_model}; no post-wait retention "
            "trend was resolved"
        )
    lines.extend(
        [
            f"- **Image-1 to image-2 control:** {control_statement}.",
            "",
        ]
    )

    cross = _mapping(
        results.get("cross_dataset_comparison"), name="cross_dataset_comparison"
    )
    if cross.get("public_comparison_gate_passed") is True:
        lines.extend(
            [
                "**Cross-dataset comparison.** The sequences remain separate; "
                "the comparison propagates complete-shot bootstrap uncertainty.",
                "",
                f"- **Bright/switch-off rate ratio:** "
                f"{_fmt_ci(cross.get('lambda_bright_over_lambda_switch_off'), digits=2)}.",
                f"- **Bright-model prediction over 50 ms:** "
                f"{_fmt_ci(cross.get('bright_model_predicted_loss_over_50ms'), digits=2, percent=True)} loss.",
                f"- **Later-interval apparent fixed loss:** "
                f"{_fmt_ci(cross.get('apparent_later_interval_fixed_loss'), digits=2, percent=True)}.",
                f"- **Observed minus predicted:** "
                f"{_fmt_ci(cross.get('observed_minus_predicted_loss'), digits=2, percent=True)}.",
                "",
                f"{_md_cell(cross.get('allowed_conclusion'), name='allowed comparison conclusion')} "
                "This does not prove a fixed per-pulse cost.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**Cross-dataset comparison.** The public comparison gate did "
                "not pass, so no rate ratio or predicted-versus-observed claim "
                "is reported.",
                "",
            ]
        )

    latent = _mapping(results.get("latent_state"), name="latent_state")
    if _latent_public_gate_passed(latent):
        comparison = _mapping(
            latent.get("held_out_comparison"), name="latent held-out comparison"
        )
        lines.extend(
            [
                "**Latent-state gate.** Passed for the "
                f"`{_md_cell(latent.get('selected_structure'), name='latent structure')}` "
                "structure. Held-out count log likelihood improved by "
                f"{_fmt(comparison.get('delta_per_trajectory'), 3)} per trajectory "
                f"across {_integer(comparison.get('n_trajectories'), name='latent trajectories'):,} "
                "test trajectories.",
                "",
                f"{_md_cell(latent.get('interpretation'), name='latent interpretation')}",
            ]
        )
    else:
        failures = latent.get("gate_failures")
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
            reason = "; ".join(str(item) for item in failures) or "acceptance gates failed"
        else:
            reason = "acceptance gates failed"
        lines.extend(
            [
                "**Latent-state gate.** Not accepted: "
                f"{_md_cell(reason, name='latent gate failure')}.",
            ]
        )
    return _fragment(lines)


def build_publication_fragments(
    results: Mapping[str, Any], v0_qc: Mapping[str, Any]
) -> dict[str, str]:
    """Build all generated Markdown bodies from reviewed machine-readable inputs."""
    fragments = {
        "experiment-matrix": render_experiment_matrix(results, v0_qc),
        "sweep-validation": render_sweep_validation(results),
        "loss-sweep-results": render_loss_sweep_results(results),
    }
    fragments["readme-metrics-v1"] = _fragment(
        [
            "### V1 held-out loss-sweep inference",
            "",
            fragments["loss-sweep-results"].rstrip(),
        ]
    )
    for name, text in fragments.items():
        _safe_text(text, f"{name} Markdown fragment")
    return fragments


def _newline_style(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _normalize_newlines(text: str, newline: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def _replace_marker(text: str, marker: str, body: str) -> str:
    begin = f"<!-- BEGIN:{marker} -->"
    end = f"<!-- END:{marker} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise AssetInputError(
            f"expected exactly one README marker pair for {marker!r}"
        )
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    newline = _newline_style(text)
    normalized = _normalize_newlines(body.strip("\r\n"), newline)
    replacement = f"{begin}{newline}{normalized}{newline}{end}"
    updated, count = pattern.subn(lambda _: replacement, text, count=1)
    if count != 1:
        raise AssetInputError(f"could not replace README marker {marker!r}")
    return updated


def inject_readme_fragments(text: str, fragments: Mapping[str, str]) -> str:
    """Replace only the three pre-existing generated README marker bodies."""
    if set(fragments) != set(README_MARKERS):
        raise AssetInputError(
            f"README fragments must be exactly {list(README_MARKERS)}"
        )
    updated = text
    for marker in README_MARKERS:
        updated = _replace_marker(updated, marker, fragments[marker])
    return updated


def update_v1_metrics_text(text: str, body: str) -> str:
    """Replace or append the marked V1 block while preserving V0 byte-for-byte."""
    begin = f"<!-- BEGIN:{METRICS_V1_MARKER} -->"
    end = f"<!-- END:{METRICS_V1_MARKER} -->"
    begin_count = text.count(begin)
    end_count = text.count(end)
    if begin_count == 1 and end_count == 1:
        return _replace_marker(text, METRICS_V1_MARKER, body)
    if begin_count or end_count:
        raise AssetInputError("V1 metrics marker is incomplete or duplicated")
    newline = _newline_style(text)
    normalized = _normalize_newlines(body.strip("\r\n"), newline)
    separator = "" if text.endswith((newline + newline)) else (
        newline if text.endswith(newline) else newline + newline
    )
    return (
        text
        + separator
        + f"{begin}{newline}{normalized}{newline}{end}{newline}"
    )


def _read_text_exact(path: Path) -> str:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise AssetInputError(f"cannot read {Path(path).name}: {exc}") from exc


def _write_text_exact(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def generate_publication(
    results_path: Path,
    v0_qc_path: Path,
    output_dir: Path,
    *,
    publish: bool,
    readme_path: Path | None = None,
    readme_metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Render figures/fragments and optionally inject the tracked Markdown."""
    results_path = Path(results_path)
    v0_qc_path = Path(v0_qc_path)
    output_dir = Path(output_dir)
    asset_metadata = generate_assets(results_path, output_dir)
    results = _load_json_object(results_path, name="loss-sweep result JSON")
    v0_qc = _load_json_object(v0_qc_path, name="V0 QC JSON")
    fragments = build_publication_fragments(results, v0_qc)

    fragment_metadata: dict[str, Any] = {}
    for name, filename in FRAGMENT_FILES.items():
        text = fragments[name]
        path = output_dir / filename
        _write_text_exact(path, text)
        fragment_metadata[name] = {
            "path": filename,
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    injection: dict[str, Any] = {
        "mode": "published" if publish else "isolated_preview",
        "readme_updated": False,
        "readme_metrics_updated": False,
    }
    if publish:
        if readme_path is None or readme_metrics_path is None:
            raise AssetInputError(
                "publish mode requires README and readme-metrics paths"
            )
        readme_path = Path(readme_path)
        readme_metrics_path = Path(readme_metrics_path)
        readme_before = _read_text_exact(readme_path)
        metrics_before = _read_text_exact(readme_metrics_path)
        readme_after = inject_readme_fragments(
            readme_before,
            {marker: fragments[marker] for marker in README_MARKERS},
        )
        metrics_after = update_v1_metrics_text(
            metrics_before, fragments["readme-metrics-v1"]
        )
        if readme_after != readme_before:
            _write_text_exact(readme_path, readme_after)
            injection["readme_updated"] = True
        if metrics_after != metrics_before:
            _write_text_exact(readme_metrics_path, metrics_after)
            injection["readme_metrics_updated"] = True
        injection["readme_sha256"] = hashlib.sha256(
            readme_after.encode("utf-8")
        ).hexdigest()
        injection["readme_metrics_sha256"] = hashlib.sha256(
            metrics_after.encode("utf-8")
        ).hexdigest()

    metadata = {
        **asset_metadata,
        "v0_qc_source_file": v0_qc_path.name,
        "v0_qc_source_sha256": hashlib.sha256(v0_qc_path.read_bytes()).hexdigest(),
        "fragments": fragment_metadata,
        "injection": injection,
    }
    blob = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    _safe_text(blob, "loss-sweep publication metadata")
    _write_text_exact(output_dir / PUBLICATION_METADATA_NAME, blob)
    return metadata


def generate_assets(results_path: Path, output_dir: Path) -> dict[str, Any]:
    """Render every supported public asset and return deterministic metadata."""
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    if not results_path.exists():
        raise AssetInputError(f"result JSON does not exist: {results_path.name}")
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AssetInputError(f"cannot read result JSON: {exc}") from exc
    if not isinstance(results, Mapping):
        raise AssetInputError("result JSON root must be an object")

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}
    visible_text: dict[str, list[str]] = {}

    builders = (
        ("loss_sweep_overview.png", make_overview),
        ("dark_hold_retention.png", make_dark_figure),
        ("bright_wait_decay.png", make_bright_figure),
        ("background_drift_sweeps.png", make_background_figure),
    )
    for filename, builder in builders:
        info, labels = builder(results, output_dir / filename)
        assets[filename] = info
        visible_text[filename] = labels

    skipped: dict[str, str] = {}
    per_site = _optional_per_site_records(results)
    if per_site is not None:
        info, labels = make_per_site_figure(
            per_site, output_dir / "per_site_retention_map.png"
        )
        assets["per_site_retention_map.png"] = info
        visible_text["per_site_retention_map.png"] = labels
    else:
        stale = output_dir / "per_site_retention_map.png"
        if stale.exists():
            stale.unlink()
        skipped["per_site_retention_map.png"] = (
            "no machine-readable per-site retention rows in result JSON"
        )

    representative = _latent_representative(results)
    if representative is not None:
        info, labels = make_latent_figure(
            representative, output_dir / "latent_state_example.png"
        )
        assets["latent_state_example.png"] = info
        visible_text["latent_state_example.png"] = labels
    else:
        stale = output_dir / "latent_state_example.png"
        if stale.exists():
            stale.unlink()
        skipped["latent_state_example.png"] = (
            "latent gate did not pass or representative data are absent"
        )

    for filename, labels in visible_text.items():
        for label in labels:
            _safe_text(label, f"visible text in {filename}")
    source_bytes = results_path.read_bytes()
    metadata = {
        "generated_by": "scripts/generate_loss_sweep_assets.py",
        "source_file": results_path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "analysis_version": results.get("analysis_version"),
        "independent_experimental_unit": results.get(
            "independent_experimental_unit", "shot"
        ),
        "assets": assets,
        "optional_assets_skipped": skipped,
        "visible_text": visible_text,
        "selection_rules": {
            "latent_representative": (
                representative.get("selection_rule")
                if representative is not None
                else None
            ),
            "per_site": (
                "all supplied sites; one predeclared held-out value per site"
                if per_site is not None
                else None
            ),
        },
        "privacy": {
            "absolute_paths_stored": False,
            "raw_filenames_stored": False,
            "internal_sequence_identifiers_visible": False,
            "user_or_machine_identifiers_visible": False,
        },
    }
    blob = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    _safe_text(blob, "loss-sweep asset metadata")
    (output_dir / METADATA_NAME).write_text(blob, encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--v0-qc", type=Path, default=DEFAULT_V0_QC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "isolated preview directory; when omitted, publish assets and "
            "generated fragments into the tracked README surfaces"
        ),
    )
    args = parser.parse_args()
    results = args.results if args.results.is_absolute() else ROOT / args.results
    v0_qc = args.v0_qc if args.v0_qc.is_absolute() else ROOT / args.v0_qc
    publish = args.output_dir is None
    output_arg = DEFAULT_OUTPUT_DIR if publish else args.output_dir
    assert output_arg is not None
    output = output_arg if output_arg.is_absolute() else ROOT / output_arg
    try:
        metadata = generate_publication(
            results,
            v0_qc,
            output,
            publish=publish,
            readme_path=ROOT / DEFAULT_README if publish else None,
            readme_metrics_path=(
                ROOT / DEFAULT_README_METRICS if publish else None
            ),
        )
    except AssetInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for filename, info in metadata["assets"].items():
        print(
            f"figure ..... {filename} "
            f"({info['width_px']}x{info['height_px']}, {info['bytes']} bytes)"
        )
    for filename, reason in metadata["optional_assets_skipped"].items():
        print(f"skipped .... {filename}: {reason}")
    print(f"metadata ... {METADATA_NAME}")
    for name, info in metadata["fragments"].items():
        print(f"fragment ... {name}: {info['path']}")
    print(f"publication  {metadata['injection']['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
