"""Held-out analysis assembly for the optimized 2026-07-31 measurements."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .loss_sweep_analysis import (
    BACKGROUND_VALUE_COLUMNS,
    fit_emission_analysis,
    jsonable,
    run_loss_sweep_analysis,
)
from .optimized_models import (
    bootstrap_repeated_imaging,
    bootstrap_repeated_imaging_blocks,
    compare_repeated_models,
    fit_repeated_imaging,
    heldout_matched_prefix_contrasts,
    loss_budget,
    loss_budget_uncertainty,
    matched_total_time_contrasts,
    prefix_curve,
    repeated_shot_order_sensitivity,
)


OPTIMIZED_ANALYSIS_VERSION = "optimized-lifetimes-v1.0"


def _public_emission_distribution(
    context: Mapping[str, Any], *, n_bins: int = 60
) -> dict[str, Any]:
    """Compact held-out histogram and fitted component densities for figures."""
    if n_bins < 10:
        raise ValueError("public emission distribution needs at least 10 bins")
    scored = context["scored"]
    model = context["selected_model"]
    test = scored.loc[scored["split"].astype(str) == "test"]
    values = test["count_adjusted_for_emission"].to_numpy(float)
    low, high = np.quantile(values, [0.0025, 0.9975])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("held-out emission counts do not define a finite range")
    edges = np.linspace(float(low), float(high), n_bins + 1)
    density, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    components = model.components

    def normal_density(mean: float, sigma: float) -> np.ndarray:
        z = (centers - mean) / sigma
        return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))

    return {
        "count_scale": "selected-emission adjusted fluorescence count",
        "bin_centers": centers,
        "heldout_density": density,
        "model_empty_density": (
            components.weight_empty
            * normal_density(components.mean_empty, components.sigma_empty)
        ),
        "model_occupied_density": (
            components.weight_occupied
            * normal_density(
                components.mean_occupied, components.sigma_occupied
            )
        ),
        "n_site_frame_rows": int(len(test)),
        "n_independent_test_shots": int(
            len(test[["run_id", "shot_id"]].drop_duplicates())
        ),
        "interpretation": (
            "Held-out count histogram with training-fit Gaussian components; "
            "the components are latent/model-implied, not externally labelled truth."
        ),
    }


def emission_residual_diagnostic(
    context: Mapping[str, Any],
    *,
    ambiguous_probability_bounds: tuple[float, float] = (0.10, 0.90),
    material_ambiguous_fraction: float = 0.05,
    extreme_residual_z: float = 4.0,
    material_extreme_fraction: float = 0.02,
) -> dict[str, Any]:
    """Validation-only gate for heavier tails or an intermediate count state."""
    scored = context["scored"]
    model = context["selected_model"]
    validation = scored.loc[scored["split"].astype(str) == "validation"].copy()
    if validation.empty:
        raise ValueError("emission residual diagnostic needs validation rows")
    lower, upper = ambiguous_probability_bounds
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("ambiguous posterior bounds must lie strictly inside (0, 1)")

    nearest_z = np.empty(len(validation), dtype=float)
    if model.kind == "frame_pooled":
        groups = validation[model.frame_col].to_numpy()
        components = model.frame_components
    elif model.kind == "prototype_per_site":
        groups = validation[model.site_col].to_numpy()
        components = model.site_components
    else:
        groups = np.zeros(len(validation), dtype=int)
        components = {0: model.components}
    values = validation["count_adjusted_for_emission"].to_numpy(float)
    for group in pd.unique(groups):
        mask = groups == group
        comp = components.get(group, model.components)
        empty_z = np.abs((values[mask] - comp.mean_empty) / comp.sigma_empty)
        occupied_z = np.abs(
            (values[mask] - comp.mean_occupied) / comp.sigma_occupied
        )
        nearest_z[mask] = np.minimum(empty_z, occupied_z)
    validation["nearest_component_abs_z"] = nearest_z
    probability = validation["posterior_occupied"].to_numpy(float)
    validation["ambiguous_component_posterior"] = (
        (probability > lower) & (probability < upper)
    )
    validation["extreme_nearest_component_residual"] = (
        nearest_z > extreme_residual_z
    )

    rows = []
    for frame, block in validation.groupby("frame_index", observed=True, sort=True):
        frame_probability = np.clip(
            block["posterior_occupied"].to_numpy(float), 1e-12, 1.0 - 1e-12
        )
        rows.append(
            {
                "frame_index": int(frame),
                "n_rows": int(len(block)),
                "ambiguous_posterior_fraction": float(
                    block["ambiguous_component_posterior"].mean()
                ),
                "extreme_nearest_component_residual_fraction": float(
                    block["extreme_nearest_component_residual"].mean()
                ),
                "mean_posterior_entropy": float(
                    np.mean(
                        -frame_probability * np.log(frame_probability)
                        - (1.0 - frame_probability)
                        * np.log1p(-frame_probability)
                    )
                ),
            }
        )
    max_ambiguous = max(row["ambiguous_posterior_fraction"] for row in rows)
    max_extreme = max(
        row["extreme_nearest_component_residual_fraction"] for row in rows
    )
    partial_required = bool(max_ambiguous >= material_ambiguous_fraction)
    heavy_tail_required = bool(max_extreme >= material_extreme_fraction)
    return {
        "scope": "validation shots only; no test metric used for this gate",
        "selected_two_component_model": model.name,
        "frame_diagnostics": rows,
        "thresholds": {
            "ambiguous_probability_bounds": [lower, upper],
            "material_ambiguous_fraction": material_ambiguous_fraction,
            "extreme_residual_z": extreme_residual_z,
            "material_extreme_fraction": material_extreme_fraction,
        },
        "partial_exposure_state_required": partial_required,
        "heavy_tail_alternative_required": heavy_tail_required,
        "third_state_attempted": False,
        "heavy_tail_alternative_attempted": False,
        "gate_passed_without_additional_state_or_tail_model": bool(
            not partial_required and not heavy_tail_required
        ),
        "interpretation": (
            "An additional emission structure is attempted only when a material "
            "validation-only residual gate fires; low overlap alone is not called "
            "empirical fidelity or labelled loss-during-exposure truth."
        ),
    }


def _validate_repeated_table(data: pd.DataFrame, *, exposure_s: float) -> None:
    required = {
        "run_id",
        "shot_id",
        "shot_order",
        "condition_id",
        "split",
        "frame_index",
        "site_id",
        "exposure_s",
        "dark_hold_s",
        "interframe_gap_s",
        "cumulative_bright_s",
        "cumulative_dark_s",
        "pulse_count",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"repeated table is missing {missing}")
    if sorted(data["frame_index"].astype(int).unique()) != list(range(5)):
        raise ValueError("repeated table must contain exactly five frame prefixes")
    if not np.allclose(data["exposure_s"].astype(float), exposure_s):
        raise ValueError(f"repeated table does not contain {exposure_s:g} s exposure")
    timing = data[
        [
            "run_id",
            "shot_id",
            "frame_index",
            "exposure_s",
            "dark_hold_s",
            "interframe_gap_s",
            "cumulative_bright_s",
            "cumulative_dark_s",
            "pulse_count",
        ]
    ].drop_duplicates()
    later = timing.loc[timing["frame_index"].astype(int) > 0]
    if not np.allclose(later["dark_hold_s"].astype(float), 0.01, atol=1e-12):
        raise ValueError("repeated table does not preserve the declared 10 ms dark gap")
    if not np.allclose(
        later["interframe_gap_s"].astype(float), 0.010002, atol=1e-9
    ):
        raise ValueError("compiled repeated frame gap is not 10.002 ms")
    expected_bright = timing["pulse_count"].astype(int) * exposure_s
    expected_dark = (timing["pulse_count"].astype(int) - 1) * 0.01
    if not np.allclose(timing["cumulative_bright_s"].astype(float), expected_bright):
        raise ValueError("repeated cumulative bright time is inconsistent")
    if not np.allclose(timing["cumulative_dark_s"].astype(float), expected_dark):
        raise ValueError("repeated cumulative dark time is inconsistent")
    split_per_shot = data.groupby(
        ["run_id", "shot_id"], observed=True
    )["split"].nunique()
    if not split_per_shot.eq(1).all():
        raise ValueError("one or more repeated shots cross frozen splits")


def _selected_dark_rate(context: Mapping[str, Any]) -> tuple[float, str]:
    fit = context["fit"]
    if fit.model == "shared":
        return float(fit.lambda_groups["shared"]), "shared"
    if fit.model == "first_separate":
        return float(fit.lambda_groups["later_intervals"]), "later_intervals"
    raise RuntimeError(
        "the validation-selected dark model has no single reusable dark rate; "
        "a repeated-imaging dark correction is not identifiable"
    )


def _interval_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return jsonable(frame)


def _weak_site_sensitivity(
    scored_by_run: Mapping[str, pd.DataFrame],
    weak_by_run: Mapping[str, Sequence[int]],
    *,
    selected_model: str,
    lambda_bright: float,
    lambda_dark: float,
) -> dict[str, Any]:
    pieces = []
    excluded = {}
    for run, scored in scored_by_run.items():
        weak = sorted(int(value) for value in weak_by_run.get(run, ()))
        excluded[run] = weak
        pieces.append(scored.loc[~scored["site_id"].isin(weak)])
    strong = pd.concat(pieces, ignore_index=True)
    development = strong.loc[
        strong["split"].astype(str).isin(["train", "validation"])
    ]
    fit = fit_repeated_imaging(
        development,
        model=selected_model,
        lambda_bright=lambda_bright,
        lambda_dark=lambda_dark,
        split_col=None,
    )
    return {
        "excluded_training_only_weak_sites_by_run": excluded,
        "n_excluded": int(sum(len(values) for values in excluded.values())),
        "parameters": fit.parameter_dict(),
        "interpretation": (
            "Predeclared training-only weak-site exclusion sensitivity; the "
            "all-site fit remains primary."
        ),
    }


def _fit_repeated_emissions(
    tables: Mapping[float, pd.DataFrame],
    count_columns: Mapping[float, str],
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    reports = {}
    contexts = {}
    for exposure in sorted(tables):
        report, context = fit_emission_analysis(
            tables[exposure], primary_count_col=count_columns[exposure]
        )
        reports[_exposure_key(exposure)] = report
        contexts[exposure] = context
    return reports, contexts


def _repeated_structure_summary(
    contexts: Mapping[float, Mapping[str, Any]],
    *,
    lambda_bright: float,
    lambda_dark: float,
    seed: int,
) -> dict[str, Any]:
    scored = pd.concat(
        [contexts[exposure]["scored"] for exposure in sorted(contexts)],
        ignore_index=True,
    )
    train = scored.loc[scored["split"].astype(str) == "train"]
    validation = scored.loc[scored["split"].astype(str) == "validation"]
    test = scored.loc[scored["split"].astype(str) == "test"]
    selected, selection, _fits = compare_repeated_models(
        train,
        validation,
        lambda_bright=lambda_bright,
        lambda_dark=lambda_dark,
        seed=seed,
    )
    development = scored.loc[
        scored["split"].astype(str).isin(["train", "validation"])
    ]
    fit = fit_repeated_imaging(
        development,
        model=selected,
        lambda_bright=lambda_bright,
        lambda_dark=lambda_dark,
        split_col=None,
    )
    common = fit_repeated_imaging(
        development,
        model="common_additional_readout",
        lambda_bright=lambda_bright,
        lambda_dark=lambda_dark,
        split_col=None,
    )
    return {
        "selected_model": selected,
        "model_selection": selection,
        "selected_parameters": fit.parameter_dict(),
        "selected_test_mean_nll": float(fit.nll(test) / len(test)),
        "common_factor_parameters": common.parameter_dict(),
        "n_independent_test_shots": int(
            len(test[["run_id", "shot_id"]].drop_duplicates())
        ),
    }


def _background_sensitivity(
    repeated_data: Mapping[float, pd.DataFrame],
    primary_contexts: Mapping[float, Mapping[str, Any]],
    primary_columns: Mapping[float, str],
    *,
    lambda_bright: float,
    lambda_dark: float,
    seed: int,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    out = {}
    for method in ("raw", "global", "spatial", "template"):
        column = BACKGROUND_VALUE_COLUMNS[method]
        contexts = {}
        emissions = {}
        for exposure in sorted(repeated_data):
            if primary_columns[exposure] == column:
                context = primary_contexts[exposure]
                report = None
            else:
                progress(
                    "fit repeated background sensitivity "
                    f"{method}/{_exposure_key(exposure)}"
                )
                report, context = fit_emission_analysis(
                    repeated_data[exposure], primary_count_col=column
                )
            contexts[exposure] = context
            emissions[_exposure_key(exposure)] = {
                "selected_model": context["selected_model"].name,
                "reused_primary_fit": report is None,
            }
        out[method] = {
            **_repeated_structure_summary(
                contexts,
                lambda_bright=lambda_bright,
                lambda_dark=lambda_dark,
                seed=seed + list(("raw", "global", "spatial", "template")).index(method),
            ),
            "emission_models_by_exposure": emissions,
        }
    return {
        "methods": out,
        "primary_selection_rule_unchanged": True,
        "interpretation": (
            "Predeclared count-column sensitivity. The primary background was "
            "frozen from site-free validation diagnostics, not these outcomes."
        ),
    }


def run_optimized_analysis(
    dark_data: pd.DataFrame,
    bright_data: pd.DataFrame,
    repeated_data: Mapping[float, pd.DataFrame],
    *,
    dark_count_column: str,
    bright_count_column: str,
    repeated_count_columns: Mapping[float, str],
    repeated_alternate_split_data: Mapping[float, pd.DataFrame] | None = None,
    n_boot: int,
    random_seed: int,
    timing_verified: bool,
    geometry_validated: Mapping[str, bool],
    background_frozen: Mapping[str, bool],
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run held-out emissions, lifetimes, prefix models, and loss attribution."""
    if n_boot < 50:
        raise ValueError("development analysis requires at least 50 bootstraps")
    announce = progress or (lambda _message: None)
    for exposure, data in repeated_data.items():
        _validate_repeated_table(data, exposure_s=float(exposure))

    announce("fit optimized dark/bright held-out lifetime analyses")
    lifetime, lifetime_context = run_loss_sweep_analysis(
        dark_data,
        bright_data,
        dark_count_column=dark_count_column,
        bright_count_column=bright_count_column,
        n_boot=n_boot,
        random_seed=random_seed,
        timing_verified=timing_verified,
        dark_geometry_validated=bool(geometry_validated.get("dark")),
        bright_geometry_validated=bool(geometry_validated.get("bright")),
        dark_background_frozen=bool(background_frozen.get("dark")),
        bright_background_frozen=bool(background_frozen.get("bright")),
        run_latent_gate=False,
        progress=announce,
    )
    dark_rate, dark_rate_group = _selected_dark_rate(lifetime_context["dark"])
    bright_rate = float(
        lifetime_context["bright"]["fit"].lambda_bright_effective
    )

    announce("fit exposure-specific repeated-imaging emission models")
    emission_reports = {}
    emission_contexts = {}
    scored_by_run = {}
    weak_by_run = {}
    for exposure in sorted(repeated_data):
        data = repeated_data[exposure]
        report, context = fit_emission_analysis(
            data, primary_count_col=repeated_count_columns[exposure]
        )
        run = str(data["run_id"].iloc[0])
        emission_reports[_exposure_key(exposure)] = report
        emission_contexts[exposure] = context
        scored_by_run[run] = context["scored"]
        weak_by_run[run] = context["weak_sites"]
        emission_reports[_exposure_key(exposure)][
            "validation_residual_diagnostic"
        ] = emission_residual_diagnostic(context)
        emission_reports[_exposure_key(exposure)][
            "public_heldout_distribution"
        ] = _public_emission_distribution(context)
    scored = pd.concat(scored_by_run.values(), ignore_index=True)

    train = scored.loc[scored["split"].astype(str) == "train"].copy()
    validation = scored.loc[scored["split"].astype(str) == "validation"].copy()
    test = scored.loc[scored["split"].astype(str) == "test"].copy()
    announce("select continuous versus pulse-associated repeated model")
    selected, selection_table, training_fits = compare_repeated_models(
        train,
        validation,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
        seed=random_seed + 700,
    )
    development = scored.loc[
        scored["split"].astype(str).isin(["train", "validation"])
    ].copy()
    final_fit = fit_repeated_imaging(
        development,
        model=selected,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
        split_col=None,
    )
    test_nll = final_fit.nll(test)
    announce("bootstrap repeated model by complete shots and contiguous blocks")
    bootstrap = bootstrap_repeated_imaging(
        development,
        model=selected,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
        n_boot=n_boot,
        seed=random_seed + 710,
    )
    block_bootstrap = bootstrap_repeated_imaging_blocks(
        development,
        model=selected,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
        n_boot=n_boot,
        seed=random_seed + 711,
        block_size=20,
    )
    common_sensitivity_fit = (
        final_fit
        if selected == "common_additional_readout"
        else fit_repeated_imaging(
            development,
            model="common_additional_readout",
            lambda_bright=bright_rate,
            lambda_dark=dark_rate,
            split_col=None,
        )
    )
    common_sensitivity_bootstrap = (
        bootstrap
        if selected == "common_additional_readout"
        else bootstrap_repeated_imaging(
            development,
            model="common_additional_readout",
            lambda_bright=bright_rate,
            lambda_dark=dark_rate,
            n_boot=n_boot,
            seed=random_seed + 712,
        )
    )
    common_sensitivity_blocks = (
        block_bootstrap
        if selected == "common_additional_readout"
        else bootstrap_repeated_imaging_blocks(
            development,
            model="common_additional_readout",
            lambda_bright=bright_rate,
            lambda_dark=dark_rate,
            n_boot=n_boot,
            seed=random_seed + 713,
            block_size=20,
        )
    )
    curve = prefix_curve(scored, n_boot=n_boot, seed=random_seed + 720)
    matched = matched_total_time_contrasts(final_fit)
    heldout_matched = heldout_matched_prefix_contrasts(
        test,
        final_fit,
        n_boot=n_boot,
        seed=random_seed + 721,
    )
    shot_order_sensitivity = repeated_shot_order_sensitivity(
        development,
        selected_model=selected,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
    )
    weak_sensitivity = _weak_site_sensitivity(
        scored_by_run,
        weak_by_run,
        selected_model=selected,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
    )
    alternate_split = None
    if repeated_alternate_split_data is not None:
        announce("refit repeated emissions under alternate contiguous-block split")
        alternate_emissions, alternate_contexts = _fit_repeated_emissions(
            repeated_alternate_split_data, repeated_count_columns
        )
        alternate_split = {
            **_repeated_structure_summary(
                alternate_contexts,
                lambda_bright=bright_rate,
                lambda_dark=dark_rate,
                seed=random_seed + 740,
            ),
            "emission_models_by_exposure": {
                exposure: report["selected_model"]
                for exposure, report in alternate_emissions.items()
            },
            "split_rule": "seeded permutation of five contiguous 20-shot blocks",
        }
    announce("evaluate repeated background-method sensitivities")
    background_sensitivity = _background_sensitivity(
        repeated_data,
        emission_contexts,
        repeated_count_columns,
        lambda_bright=bright_rate,
        lambda_dark=dark_rate,
        seed=random_seed + 750,
        progress=announce,
    )

    dark_bootstrap = lifetime_context["dark"]["bootstrap"]
    dark_column = f"lambda_switch_off__{dark_rate_group}"
    bright_bootstrap = lifetime_context["bright"]["bootstrap"]
    budgets = {
        _exposure_key(exposure): loss_budget(final_fit, exposure_s=exposure)
        for exposure in sorted(repeated_data)
    }
    propagated = loss_budget_uncertainty(
        final_fit,
        bootstrap.draws,
        bright_bootstrap.draws["lambda_bright_effective"].to_numpy(float),
        dark_bootstrap.draws[dark_column].to_numpy(float),
        seed=random_seed + 730,
    )
    common_sensitivity_budgets = {
        _exposure_key(exposure): loss_budget(
            common_sensitivity_fit, exposure_s=exposure
        )
        for exposure in sorted(repeated_data)
    }
    common_sensitivity_propagated = loss_budget_uncertainty(
        common_sensitivity_fit,
        common_sensitivity_bootstrap.draws,
        bright_bootstrap.draws["lambda_bright_effective"].to_numpy(float),
        dark_bootstrap.draws[dark_column].to_numpy(float),
        seed=random_seed + 731,
    )

    q_intervals = _interval_table(bootstrap.intervals())
    q_parameter_names = [
        name
        for name in bootstrap.estimate.index
        if name.startswith("additional_readout_retention__")
    ]
    q_resolved_below_one = bool(
        q_parameter_names
        and all(
            next(
                row["upper"] for row in q_intervals if row["parameter"] == name
            )
            < 1.0
            for name in q_parameter_names
        )
    )
    repeated_claim_gate = bool(
        timing_verified
        and all(geometry_validated.get(key, False) for key in ("50ms", "100ms", "200ms"))
        and all(background_frozen.get(key, False) for key in ("50ms", "100ms", "200ms"))
    )
    repeated_report = {
        "emission_models_by_exposure": emission_reports,
        "lambda_bright_constraint": bright_rate,
        "lambda_dark_constraint": dark_rate,
        "dark_rate_group": dark_rate_group,
        "condition_comparability": (
            "Main imaging-stage commands match, but the bright-lifetime run has "
            "different pre-heat commands; the 0090 rate is an operational "
            "constraint with condition-specific sensitivity, not a universal rate."
        ),
        "selected_model": selected,
        "model_selection": selection_table,
        "training_candidate_parameters": {
            name: fit.parameter_dict() for name, fit in training_fits.items()
        },
        "final_fit_scope": "training_plus_validation_after_structure_freeze",
        "final_parameters": final_fit.parameter_dict(),
        "test_scored_once": {
            "nll": float(test_nll),
            "mean_nll": float(test_nll / len(test)),
            "n_site_frame_observations": int(len(test)),
            "n_independent_shots": int(
                len(test[["run_id", "shot_id"]].drop_duplicates())
            ),
        },
        "clustered_prefix_apparent_occupancy": curve,
        "matched_total_bright_time": matched,
        "heldout_matched_prefix_contrasts": heldout_matched,
        "complete_shot_bootstrap": {
            "unit": "complete shot, stratified by exposure run",
            "n_requested": bootstrap.n_requested,
            "n_successful": bootstrap.n_successful,
            "n_failed": bootstrap.n_failed,
            "seed": bootstrap.seed,
            "intervals": bootstrap.intervals(),
            "failures": bootstrap.failures,
        },
        "contiguous_block_bootstrap_sensitivity": {
            "unit": "contiguous 20-shot acquisition-order block within run",
            "n_requested": block_bootstrap.n_requested,
            "n_successful": block_bootstrap.n_successful,
            "n_failed": block_bootstrap.n_failed,
            "seed": block_bootstrap.seed,
            "intervals": block_bootstrap.intervals(),
            "failures": block_bootstrap.failures,
        },
        "common_additional_readout_sensitivity": {
            "selected_on_validation": selected == "common_additional_readout",
            "parameters": common_sensitivity_fit.parameter_dict(),
            "complete_shot_intervals": common_sensitivity_bootstrap.intervals(),
            "contiguous_block_intervals": common_sensitivity_blocks.intervals(),
            "interpretation": (
                "Reported even when M0 is selected. A fitted q below one does "
                "not override the predeclared validation-prediction gate."
            ),
        },
        "weak_site_sensitivity": weak_sensitivity,
        "alternate_contiguous_block_split": alternate_split,
        "shot_order_sensitivity": shot_order_sensitivity,
        "background_method_sensitivity": background_sensitivity,
        "additional_readout_interval_excludes_one": q_resolved_below_one,
        "public_claim_gate_passed": repeated_claim_gate,
        "allowed_pulse_claim": bool(
            repeated_claim_gate
            and selected != "continuous_only"
            and q_resolved_below_one
        ),
        "interpretation": (
            "A selected factor below one is an apparent pulse/readout-associated "
            "residual beyond cumulative bright and dark time. It does not identify "
            "a unique switching mechanism or an exact physical loss time."
        ),
    }
    report = {
        "analysis_version": OPTIMIZED_ANALYSIS_VERSION,
        "random_seed": int(random_seed),
        "bootstrap_replicates": int(n_boot),
        "independent_experimental_unit": "shot",
        "optimized_dark_bright": lifetime,
        "repeated_imaging": repeated_report,
        "loss_budget": {
            "point_estimates": budgets,
            "uncertainty": propagated,
            "unselected_common_factor_sensitivity": {
                "point_estimates": common_sensitivity_budgets,
                "uncertainty": common_sensitivity_propagated,
                "selected_on_validation": (
                    selected == "common_additional_readout"
                ),
            },
            "loss_location_status": "rate_based_prior_attribution_only",
            "count_conditioned_posterior_attempted": False,
            "exact_loss_timestamp_claimed": False,
        },
    }
    return jsonable(report), {
        "lifetime": lifetime_context,
        "repeated_emissions": emission_contexts,
        "repeated_scored": scored,
        "repeated_fit": final_fit,
        "repeated_bootstrap": bootstrap,
        "repeated_block_bootstrap": block_bootstrap,
    }


def _exposure_key(exposure_s: float) -> str:
    return f"{int(round(float(exposure_s) * 1000.0))}ms"
