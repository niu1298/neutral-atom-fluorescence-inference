"""Generate deterministic public figures for the optimized 2026-07-31 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.reporting import (  # noqa: E402
    ACCENT,
    ACCENT_2,
    GRID,
    INK,
    MUTED,
    OK,
    SOFT,
    WARN,
    apply_style,
    assert_public_safe,
    save,
)


DEFAULT_RESULT = ROOT / "reports" / "optimized_lifetime_results_20260731.json"
DEFAULT_OUTPUT = ROOT / "assets" / "readme"
DEFAULT_METRICS = ROOT / "reports" / "optimized_readme_metrics_20260731.md"
ASSET_NAMES = (
    "optimized_occupancy_inference.png",
    "optimized_sequence_design.png",
    "optimized_lifetime_overview.png",
    "exposure_segmentation_result.png",
)
EXPOSURE_COLORS = {"50ms": ACCENT, "100ms": OK, "200ms": ACCENT_2}


class AssetInputError(ValueError):
    """The reviewed result does not support a public asset."""


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
            color=SOFT,
            step="mid",
            label="held-out counts",
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
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require_reviewed(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    drawers = (_draw_occupancy, _draw_sequence, _draw_lifetime_overview, _draw_segmentation)
    paths = []
    for name, drawer in zip(ASSET_NAMES, drawers):
        path = output_dir / name
        if drawer is _draw_sequence:
            drawer(path)
        else:
            drawer(result, path)
        paths.append(path)
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        text = _metrics_markdown(result)
        assert_public_safe(text, "optimized README metrics")
        metrics_path.write_text(text, encoding="utf-8")
    metadata = {
        "generated_by": "scripts/generate_optimized_assets.py",
        "result_sha256": _sha256(result_path),
        "analysis_code_commit": result["provenance"]["analysis_code_commit"],
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
    args = parser.parse_args()
    result = args.results if args.results.is_absolute() else ROOT / args.results
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    metrics = args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    metadata = generate_assets(result, output, metrics_path=metrics)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
