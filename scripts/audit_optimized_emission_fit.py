"""Focused held-out calibration audit for optimized repeated-imaging emissions.

This script reads the existing processed tables, reconstructs the declared
training/validation-frozen emission candidates, and scores test rows without
refitting on them.  Its outputs live under the ignored validation tree and do
not modify the reviewed scientific result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.optimized_analysis import (  # noqa: E402
    _public_emission_distribution,
)
from fluorescence_inference.loss_sweep_analysis import (  # noqa: E402
    fit_emission_analysis,
)
from fluorescence_inference.reporting import (  # noqa: E402
    ACCENT,
    ACCENT_2,
    GRID,
    INK,
    MUTED,
    apply_style,
)


EXPECTED_RESULT_SHA256 = (
    "3a56b8f47f844d5962c7b5a8a9c4371676d4457778379e6cbfb12087120fad15"
)
RESULT_PATH = ROOT / "reports" / "optimized_lifetime_results_20260731.json"
OUTPUT_DIR = ROOT / "reports" / "validation" / "optimized_emission_fit_audit"
DISPLAY_QUANTILES = (0.0025, 0.9975)
N_BINS = 60
N_BOOT = 400
RANDOM_SEED = 20260801
HIST_FILL = "#66727f"
HIST_EDGE = "#3d4650"
TOTAL_COLOR = "#111827"

DATASETS = {
    "50ms": ROOT
    / "data"
    / "processed"
    / "imaging_5frame_50ms_20260731_0113.parquet",
    "100ms": ROOT
    / "data"
    / "processed"
    / "imaging_5frame_100ms_20260731_0114.parquet",
    "200ms": ROOT
    / "data"
    / "processed"
    / "imaging_5frame_200ms_20260731_0115.parquet",
}


@dataclass(frozen=True)
class DensityBands:
    """Complete-shot empirical density bands for one or more row scopes."""

    lower: Mapping[str, np.ndarray]
    median: Mapping[str, np.ndarray]
    upper: Mapping[str, np.ndarray]
    draw_indices: np.ndarray
    shot_keys: tuple[tuple[str, Any], ...]
    rows_per_shot: Mapping[str, tuple[int, ...]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    widths = np.diff(edges)
    included = int(counts.sum())
    if included == 0:
        raise ValueError("no observations fall inside the diagnostic range")
    return counts / (included * widths)


def complete_shot_density_bands(
    table: pd.DataFrame,
    edges: np.ndarray,
    *,
    scopes: Mapping[str, np.ndarray],
    n_boot: int,
    seed: int,
    value_col: str = "count_adjusted_for_emission",
    shot_cols: Sequence[str] = ("run_id", "shot_id"),
) -> DensityBands:
    """Resample complete shots and preserve every selected row within a shot."""
    if not 1 <= n_boot <= 500:
        raise ValueError("diagnostic histogram bootstrap must use 1 to 500 draws")
    required = {value_col, *shot_cols}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"complete-shot resampling is missing {missing}")
    scope_masks = {
        name: np.asarray(mask, dtype=bool) for name, mask in scopes.items()
    }
    if any(mask.shape != (len(table),) for mask in scope_masks.values()):
        raise ValueError("every density scope must be a row mask")

    key_frame = table[list(shot_cols)].astype({shot_cols[0]: str})
    shot_keys = tuple(
        tuple(row)
        for row in key_frame.drop_duplicates()
        .sort_values(list(shot_cols), kind="stable")
        .itertuples(index=False, name=None)
    )
    if len(shot_keys) < 2:
        raise ValueError("complete-shot bands require at least two shots")
    widths = np.diff(edges)
    histograms: dict[str, np.ndarray] = {}
    rows_per_shot: dict[str, tuple[int, ...]] = {}
    for name, scope in scope_masks.items():
        rows = []
        counts_by_shot = []
        for key in shot_keys:
            selected = scope.copy()
            for column, value in zip(shot_cols, key):
                selected &= table[column].astype(str).to_numpy() == str(value)
            values = table.loc[selected, value_col].to_numpy(float)
            counts, _ = np.histogram(values, bins=edges)
            rows.append(int(len(values)))
            counts_by_shot.append(counts)
        rows_per_shot[name] = tuple(rows)
        histograms[name] = np.asarray(counts_by_shot, dtype=float)

    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(shot_keys), size=(n_boot, len(shot_keys)), endpoint=False
    )
    lower: dict[str, np.ndarray] = {}
    median: dict[str, np.ndarray] = {}
    upper: dict[str, np.ndarray] = {}
    for name, shot_histograms in histograms.items():
        sampled = shot_histograms[draw_indices].sum(axis=1)
        totals = sampled.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError(f"scope {name!r} has an empty complete-shot draw")
        densities = sampled / (totals[:, None] * widths[None, :])
        lower[name], median[name], upper[name] = np.quantile(
            densities, [0.025, 0.5, 0.975], axis=0
        )
    return DensityBands(
        lower=lower,
        median=median,
        upper=upper,
        draw_indices=draw_indices,
        shot_keys=shot_keys,
        rows_per_shot=rows_per_shot,
    )


def _normal_density(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def model_densities(model: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    comp = model.components
    empty = comp.weight_empty * _normal_density(x, comp.mean_empty, comp.sigma_empty)
    occupied = comp.weight_occupied * _normal_density(
        x, comp.mean_occupied, comp.sigma_occupied
    )
    return {"empty": empty, "occupied": occupied, "total": empty + occupied}


def _mixture_cdf(model: Any, value: float) -> float:
    comp = model.components
    return float(
        comp.weight_empty * ndtr((value - comp.mean_empty) / comp.sigma_empty)
        + comp.weight_occupied
        * ndtr((value - comp.mean_occupied) / comp.sigma_occupied)
    )


def _mixture_quantile(model: Any, probability: float) -> float:
    comp = model.components
    low = min(comp.mean_empty - 12 * comp.sigma_empty, comp.mean_occupied - 12 * comp.sigma_occupied)
    high = max(comp.mean_empty + 12 * comp.sigma_empty, comp.mean_occupied + 12 * comp.sigma_occupied)
    return float(brentq(lambda value: _mixture_cdf(model, value) - probability, low, high))


def _distribution_distance(empirical: np.ndarray, model: np.ndarray, widths: np.ndarray) -> float:
    p = empirical * widths
    q = model * widths
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def divergence(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / reference[mask])))

    return 0.5 * divergence(p, midpoint) + 0.5 * divergence(q, midpoint)


def _component_diagnostics(
    block: pd.DataFrame,
    edges: np.ndarray,
    model: Any,
) -> dict[str, float]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    posterior = block["posterior_occupied"].to_numpy(float)
    values = block["count_adjusted_for_emission"].to_numpy(float)
    comp = model.components
    out: dict[str, float] = {}
    for label, selected, mean, sigma in (
        ("empty", posterior < 0.5, comp.mean_empty, comp.sigma_empty),
        ("occupied", posterior >= 0.5, comp.mean_occupied, comp.sigma_occupied),
    ):
        component_values = values[selected]
        component_density = _density(component_values, edges)
        empirical_mode = float(centers[int(np.argmax(component_density))])
        empirical_sigma = float(np.std(component_values, ddof=0))
        out[f"{label}_empirical_mode"] = empirical_mode
        out[f"{label}_frozen_mean"] = float(mean)
        out[f"{label}_mode_minus_frozen_mean"] = empirical_mode - float(mean)
        out[f"{label}_empirical_sigma"] = empirical_sigma
        out[f"{label}_frozen_sigma"] = float(sigma)
        out[f"{label}_sigma_minus_frozen_sigma"] = empirical_sigma - float(sigma)
        out[f"{label}_sigma_ratio"] = empirical_sigma / float(sigma)
        out[f"{label}_posterior_partition_rows"] = int(selected.sum())
    return out


def distribution_metrics(
    block: pd.DataFrame,
    edges: np.ndarray,
    empirical: np.ndarray,
    bands: tuple[np.ndarray, np.ndarray],
    model: Any,
) -> dict[str, Any]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    densities = model_densities(model, centers)
    total = densities["total"]
    p = empirical * widths
    q = total * widths
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    empirical_mean = float(np.sum(p_norm * centers))
    model_mean = float(np.sum(q_norm * centers))
    lower, upper = bands
    stable = total >= 1e-3 * float(total.max())
    within = stable & (total >= lower) & (total <= upper)
    stable_model_mass = float(np.sum(q[stable]))

    comp = model.components
    valley_low = float(comp.mean_empty + (comp.mean_occupied - comp.mean_empty) / 3.0)
    valley_high = float(comp.mean_empty + 2.0 * (comp.mean_occupied - comp.mean_empty) / 3.0)
    valley = (centers >= valley_low) & (centers <= valley_high)
    central_low = _mixture_quantile(model, DISPLAY_QUANTILES[0])
    central_high = _mixture_quantile(model, DISPLAY_QUANTILES[1])
    values = block["count_adjusted_for_emission"].to_numpy(float)

    full_width = float(widths[0])
    full_edges = np.arange(values.min(), values.max() + 1.01 * full_width, full_width)
    full_density = _density(values, full_edges)
    full_centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    full_mode = float(full_centers[int(np.argmax(full_density))])
    displayed_mode = float(centers[int(np.argmax(empirical))])

    metrics: dict[str, Any] = {
        "n_rows": int(len(block)),
        "n_independent_shots": int(len(block[["run_id", "shot_id"]].drop_duplicates())),
        "empirical_integral_displayed": float(np.sum(p)),
        "empty_component_midpoint_integral": float(np.sum(densities["empty"] * widths)),
        "occupied_component_midpoint_integral": float(np.sum(densities["occupied"] * widths)),
        "total_model_midpoint_integral": float(np.sum(q)),
        "empty_component_exact_range_integral": float(
            comp.weight_empty
            * (
                ndtr((edges[-1] - comp.mean_empty) / comp.sigma_empty)
                - ndtr((edges[0] - comp.mean_empty) / comp.sigma_empty)
            )
        ),
        "occupied_component_exact_range_integral": float(
            comp.weight_occupied
            * (
                ndtr((edges[-1] - comp.mean_occupied) / comp.sigma_occupied)
                - ndtr((edges[0] - comp.mean_occupied) / comp.sigma_occupied)
            )
        ),
        "component_sum_max_abs_error": float(
            np.max(np.abs(total - densities["empty"] - densities["occupied"]))
        ),
        "integrated_absolute_density_error": float(np.sum(np.abs(empirical - total) * widths)),
        "jensen_shannon_divergence_nats": _distribution_distance(empirical, total, widths),
        "empirical_mode": displayed_mode,
        "unclipped_empirical_mode": full_mode,
        "clip_mode_shift": displayed_mode - full_mode,
        "clip_mode_shift_in_bins": (displayed_mode - full_mode) / full_width,
        "model_mode": float(centers[int(np.argmax(total))]),
        "empirical_variance_displayed": float(np.sum(p_norm * (centers - empirical_mean) ** 2)),
        "model_variance_displayed": float(np.sum(q_norm * (centers - model_mean) ** 2)),
        "valley_interval": [valley_low, valley_high],
        "empirical_valley_mass": float(np.sum(p[valley])),
        "model_valley_mass": float(np.sum(q[valley])),
        "frozen_model_central_predictive_range": [central_low, central_high],
        "empirical_tail_mass_outside_frozen_central_range": float(
            np.mean((values < central_low) | (values > central_high))
        ),
        "model_tail_mass_outside_central_range": float(
            _mixture_cdf(model, central_low) + 1.0 - _mixture_cdf(model, central_high)
        ),
        "model_within_complete_shot_band_bin_fraction": float(np.mean(within[stable])),
        "model_within_complete_shot_band_mass_fraction": float(
            np.sum(q[within]) / stable_model_mass
        ),
        "ratio_stable_bin_fraction": float(np.mean(stable)),
    }
    metrics.update(_component_diagnostics(block, edges, model))
    return metrics


def _scope_masks(test: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {"all": np.ones(len(test), dtype=bool)}
    frames = test["frame_index"].to_numpy(int)
    for frame in range(5):
        masks[f"frame_{frame}"] = frames == frame
    return masks


def _plot_exposure(
    label: str,
    test: pd.DataFrame,
    edges: np.ndarray,
    bands: DensityBands,
    model: Any,
    path: Path,
) -> None:
    apply_style()
    centers = 0.5 * (edges[:-1] + edges[1:])
    densities = model_densities(model, centers)
    total = densities["total"]
    stable = total >= 1e-3 * float(total.max())
    scopes = _scope_masks(test)
    fig, axes = plt.subplots(6, 3, figsize=(15, 20), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.055, top=0.93, hspace=0.34, wspace=0.25)
    for row, (scope, mask) in enumerate(scopes.items()):
        empirical = _density(test.loc[mask, "count_adjusted_for_emission"].to_numpy(float), edges)
        lower, upper = bands.lower[scope], bands.upper[scope]
        density_ax, residual_ax, ratio_ax = axes[row]
        density_ax.fill_between(centers, lower, upper, color=HIST_FILL, alpha=0.20, step="mid", label="95% complete-shot band")
        density_ax.step(centers, empirical, where="mid", color=HIST_EDGE, lw=1.1, label="held-out empirical")
        density_ax.plot(centers, total, color=TOTAL_COLOR, lw=1.5, label="total frozen model")
        if row == 0:
            density_ax.plot(centers, densities["empty"], color=ACCENT, lw=1.0, label="empty component")
            density_ax.plot(centers, densities["occupied"], color=ACCENT_2, lw=1.0, label="occupied component")
            density_ax.legend(fontsize=7.4, ncol=2, loc="upper right")
        residual = empirical - total
        residual_ax.axhline(0.0, color=GRID, lw=1.0)
        residual_ax.step(centers, residual, where="mid", color=HIST_EDGE, lw=1.0)
        ratio = np.full_like(total, np.nan)
        ratio[stable] = empirical[stable] / total[stable]
        ratio_ax.axhline(1.0, color=GRID, lw=1.0)
        ratio_ax.step(centers, ratio, where="mid", color=HIST_EDGE, lw=1.0)
        row_name = "all frames" if scope == "all" else f"frame {row - 1}"
        density_ax.set_ylabel(f"{row_name}\ndensity")
        residual_ax.set_ylabel("empirical − model")
        ratio_ax.set_ylabel("empirical / model")
        if row == 0:
            density_ax.set_title("held-out density and clustered band")
            residual_ax.set_title("density residual")
            ratio_ax.set_title("stable-bin density ratio")
    for ax in axes[-1]:
        ax.set_xlabel("emission-adjusted count")
    fig.suptitle(f"Held-out frozen-emission calibration audit · {label}", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.955,
        f"{len(bands.shot_keys)} test shots · {len(test):,} site-frame rows · {len(bands.draw_indices)} complete-shot bootstrap draws · no test-data refit",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.017,
        "Display range is the held-out 0.25th–99.75th percentile; ratios shown where frozen density is at least 0.1% of its peak.",
        ha="center",
        color=MUTED,
        fontsize=8.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, metadata={"Software": "neutral-atom-fluorescence-inference"})
    plt.close(fig)


def _segment_rows(test: pd.DataFrame, edges: np.ndarray, model: Any, label: str) -> list[dict[str, Any]]:
    rows = []
    shot_order = (
        test[["run_id", "shot_id", "shot_order"]]
        .drop_duplicates()
        .sort_values(["shot_order", "run_id", "shot_id"], kind="stable")
    )
    midpoint = len(shot_order) // 2
    half_by_shot = {
        (str(row.run_id), row.shot_id): "early" if index < midpoint else "late"
        for index, row in enumerate(shot_order.itertuples(index=False))
    }
    half = np.asarray(
        [half_by_shot[(str(run), shot)] for run, shot in test[["run_id", "shot_id"]].itertuples(index=False, name=None)]
    )
    site_ids = sorted(test["site_id"].drop_duplicates().tolist())
    site_group = {
        site: f"Q{min(3, 4 * index // len(site_ids)) + 1}"
        for index, site in enumerate(site_ids)
    }
    segments = {"shot_half": half, "site_group": test["site_id"].map(site_group).to_numpy()}
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    total = model_densities(model, centers)["total"]
    for dimension, values in segments.items():
        for segment in sorted(pd.unique(values)):
            mask = values == segment
            empirical = _density(test.loc[mask, "count_adjusted_for_emission"].to_numpy(float), edges)
            rows.append(
                {
                    "exposure": label,
                    "dimension": dimension,
                    "segment": str(segment),
                    "n_rows": int(mask.sum()),
                    "integrated_absolute_density_error": float(np.sum(np.abs(empirical - total) * widths)),
                    "jensen_shannon_divergence_nats": _distribution_distance(empirical, total, widths),
                    "mean_posterior_occupancy": float(test.loc[mask, "posterior_occupied"].mean()),
                }
            )
    return rows


def audit_exposure(
    label: str,
    table: pd.DataFrame,
    stored: Mapping[str, Any],
    *,
    n_boot: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    count_column = str(stored["selected_count_column"])
    report, context = fit_emission_analysis(table, primary_count_col=count_column)
    model = context["selected_model"]
    scored = context["scored"]
    test = scored.loc[scored["split"].astype(str) == "test"].copy()
    if model.name != stored["selected_model"]:
        raise RuntimeError(f"{label}: reconstructed selection {model.name!r} does not match protected result")

    expected_adjusted = table[count_column].to_numpy(float).copy()
    if model.frame_offsets:
        expected_adjusted -= table[model.frame_col].map(model.frame_offsets).to_numpy(float)
    if model.site_offsets:
        expected_adjusted -= table[model.site_col].map(model.site_offsets).fillna(0.0).to_numpy(float)
    adjusted_error = float(np.max(np.abs(expected_adjusted - scored["count_adjusted_for_emission"].to_numpy(float))))

    public = _public_emission_distribution(context)
    stored_public = stored["public_heldout_distribution"]
    public_errors = {}
    for key in ("bin_centers", "heldout_density", "model_empty_density", "model_occupied_density"):
        public_errors[key] = float(np.max(np.abs(np.asarray(public[key], float) - np.asarray(stored_public[key], float))))

    values = test["count_adjusted_for_emission"].to_numpy(float)
    low, high = np.quantile(values, DISPLAY_QUANTILES)
    edges = np.linspace(float(low), float(high), N_BINS + 1)
    scopes = _scope_masks(test)
    bands = complete_shot_density_bands(test, edges, scopes=scopes, n_boot=n_boot, seed=seed)
    metrics = []
    for scope, mask in scopes.items():
        block = test.loc[mask]
        empirical = _density(block["count_adjusted_for_emission"].to_numpy(float), edges)
        row = {
            "exposure": label,
            "scope": scope,
            **distribution_metrics(block, edges, empirical, (bands.lower[scope], bands.upper[scope]), model),
        }
        metrics.append(row)

    components = model.components
    direct_total = (
        components.weight_empty
        * _normal_density(values, components.mean_empty, components.sigma_empty)
        + components.weight_occupied
        * _normal_density(values, components.mean_occupied, components.sigma_occupied)
    )
    scored_density = np.exp(test["log_predictive_density"].to_numpy(float))

    ranking = pd.DataFrame(report["validation_ranking"])
    stored_ranking = pd.DataFrame(stored["validation_ranking"])
    joined = ranking.merge(stored_ranking, on="candidate", suffixes=("_audit", "_stored"), validate="one_to_one")
    ranking_error = float(np.max(np.abs(joined["mean_count_nll_audit"] - joined["mean_count_nll_stored"])))
    candidate_rows = []
    for candidate, candidate_model in context["models"].items():
        candidate_scored = candidate_model.score(table.loc[table["split"].astype(str) == "test"])
        rank = ranking.loc[ranking["candidate"] == candidate].iloc[0]
        candidate_rows.append(
            {
                "candidate": candidate,
                "kind": candidate_model.kind,
                "eligible_for_primary": bool(rank["eligible_for_primary"]),
                "selected_on_validation": bool(rank["selected_on_validation"]),
                "shrinkage_strength": candidate_model.shrinkage_strength,
                "fallback_site_count": int(len(candidate_model.fallback_sites)),
                "validation_mean_count_nll": float(rank["mean_count_nll"]),
                "validation_delta_nll_per_row": float(rank["delta_validation_nll_per_row"]),
                "test_mean_count_nll": float(
                    context["test_evaluations"][candidate].mean_count_nll
                ),
                "test_mean_posterior_occupancy": float(candidate_scored["posterior_occupied"].mean()),
                "test_apparent_occupancy": float(candidate_scored["apparent_occupied"].mean()),
            }
        )

    figure = output_dir / f"emission_fit_audit_{label}.png"
    _plot_exposure(label, test, edges, bands, model, figure)
    return {
        "label": label,
        "selected_model": model.name,
        "selected_count_column": count_column,
        "n_rows_by_split": table.groupby("split").size().astype(int).to_dict(),
        "n_shots_by_split": {
            split: int(len(block[["run_id", "shot_id"]].drop_duplicates()))
            for split, block in table.groupby("split", observed=True)
        },
        "coordinate_audit": {
            "empirical_column": "count_adjusted_for_emission",
            "base_count_column": count_column,
            "frame_offsets_applied": bool(model.frame_offsets),
            "site_offsets_applied": bool(model.site_offsets),
            "max_abs_adjustment_reconstruction_error": adjusted_error,
            "mixture_weight_sum": float(components.weight_empty + components.weight_occupied),
            "max_abs_predictive_density_error": float(np.max(np.abs(direct_total - scored_density))),
            "public_distribution_max_abs_errors_vs_protected_result": public_errors,
            "validation_ranking_max_abs_nll_error_vs_protected_result": ranking_error,
        },
        "display": {
            "quantiles": list(DISPLAY_QUANTILES),
            "low": float(low),
            "high": float(high),
            "fraction_outside_display_range": float(np.mean((values < low) | (values > high))),
            "n_bins": N_BINS,
        },
        "complete_shot_bootstrap": {
            "n_replicates": n_boot,
            "seed": seed,
            "n_independent_shots": len(bands.shot_keys),
            "resampling_unit": "complete test shot; all sites and frames retained",
            "rows_per_shot_by_scope": bands.rows_per_shot,
            "draw_indices_sha256": hashlib.sha256(bands.draw_indices.tobytes()).hexdigest(),
        },
        "metrics": metrics,
        "segment_diagnostics": _segment_rows(test, edges, model, label),
        "candidate_diagnostics": candidate_rows,
        "figure": figure.relative_to(ROOT).as_posix(),
    }


def run_audit(
    *,
    result_path: Path = RESULT_PATH,
    output_dir: Path = OUTPUT_DIR,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    result_hash = _sha256(result_path)
    if result_hash != EXPECTED_RESULT_SHA256:
        raise RuntimeError(f"protected result hash mismatch: {result_hash}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stored_exposures = result["repeated_imaging"]["emission_models_by_exposure"]
    output_dir.mkdir(parents=True, exist_ok=True)
    exposure_results = []
    for index, label in enumerate(("50ms", "100ms", "200ms")):
        table = pd.read_parquet(DATASETS[label])
        exposure_results.append(
            audit_exposure(
                label,
                table,
                stored_exposures[label],
                n_boot=n_boot,
                seed=seed + index,
                output_dir=output_dir,
            )
        )

    metric_rows = [row for exposure in exposure_results for row in exposure["metrics"]]
    segment_rows = [row for exposure in exposure_results for row in exposure["segment_diagnostics"]]
    candidate_rows = [
        {"exposure": exposure["label"], **row}
        for exposure in exposure_results
        for row in exposure["candidate_diagnostics"]
    ]
    payload = {
        "schema_version": "optimized-emission-fit-audit-v1",
        "protected_result": {
            "path": result_path.relative_to(ROOT).as_posix(),
            "sha256": result_hash,
        },
        "inputs": {
            label: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for label, path in DATASETS.items()
        },
        "method": {
            "model_freeze": "candidates fitted on training shots and selected on validation; test rows scored once",
            "test_refit": False,
            "display_quantiles": list(DISPLAY_QUANTILES),
            "histogram_bins": N_BINS,
            "histogram_bootstrap_replicates": n_boot,
            "histogram_bootstrap_unit": "complete test shot",
            "valley_interval_rule": "middle third of the interval between frozen component means",
            "central_predictive_range": "frozen mixture 0.25th to 99.75th percentiles",
            "component_empirical_diagnostic": "test rows partitioned by frozen posterior at 0.5; not labelled truth",
        },
        "exposures": exposure_results,
    }
    audit_json = output_dir / "emission_fit_audit.json"
    _write_json(audit_json, payload)
    pd.DataFrame(metric_rows).to_csv(output_dir / "emission_fit_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(segment_rows).to_csv(output_dir / "emission_fit_segments.csv", index=False, lineterminator="\n")
    pd.DataFrame(candidate_rows).to_csv(output_dir / "emission_fit_candidates.csv", index=False, lineterminator="\n")
    artifact_paths = [
        audit_json,
        output_dir / "emission_fit_metrics.csv",
        output_dir / "emission_fit_segments.csv",
        output_dir / "emission_fit_candidates.csv",
        *[ROOT / exposure["figure"] for exposure in exposure_results],
    ]
    manifest = {
        "schema_version": "optimized-emission-fit-audit-manifest-v1",
        "protected_result_sha256": result_hash,
        "artifacts": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = args.result if args.result.is_absolute() else ROOT / args.result
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_audit(result_path=result, output_dir=output, n_boot=args.bootstrap, seed=args.seed)
    summary = {
        exposure["label"]: {
            "selected_model": exposure["selected_model"],
            "aggregate": exposure["metrics"][0],
        }
        for exposure in payload["exposures"]
    }
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
