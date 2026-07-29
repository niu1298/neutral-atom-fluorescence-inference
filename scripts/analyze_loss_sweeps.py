"""Run held-out, shot-clustered inference for the 2026-07-28 sweeps.

The script consumes standardized schema-V3 tables plus independently generated
raw-command, geometry, and background validation reports.  It never fits a
public claim when a prerequisite report is absent or failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import (  # noqa: E402
    load_dataset,
    load_run_metadata,
)
from fluorescence_inference.loss_sweep_analysis import (  # noqa: E402
    BACKGROUND_VALUE_COLUMNS,
    jsonable,
    run_loss_sweep_analysis,
)
from fluorescence_inference.provenance import git_describe, hash_file  # noqa: E402


DEFAULT_DARK_CONFIG = "configs/dark_hold_50ms_20260728_0044.yaml"
DEFAULT_BRIGHT_CONFIG = "configs/bright_wait_50ms_20260728_0050.yaml"
DEFAULT_DARK_COMMAND_AUDIT = (
    "reports/audit/dark_hold_20260728_0044_source_audit.json"
)
DEFAULT_BRIGHT_COMMAND_AUDIT = (
    "reports/audit/bright_wait_20260728_0050_source_audit.json"
)
DEFAULT_RESULT = "reports/loss_sweep_results.json"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} is missing ({path.name}); run its validation script first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} did not contain a JSON object")
    return value


def _dependency_commit(cfg) -> str | None:
    """Read only the dependency revision; never serialize its local path."""
    checkout = cfg.paths.tweezer_analysis_src.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _condition_split_counts(data: pd.DataFrame) -> list[dict[str, Any]]:
    table = (
        data[
            [
                "condition_id",
                "sweep_value_s",
                "split",
                "run_id",
                "shot_id",
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["condition_id", "sweep_value_s", "split"], observed=True
        )["shot_id"]
        .nunique()
        .rename("n_independent_shots")
        .reset_index()
        .sort_values(["sweep_value_s", "split"], kind="stable")
    )
    return jsonable(table)


def _dataset_summary(data: pd.DataFrame, run_meta: dict[str, Any]) -> dict[str, Any]:
    shot = data[
        [
            "shot_id",
            "shot_order",
            "condition_id",
            "sweep_value_s",
            "cycle_index",
            "split",
        ]
    ].drop_duplicates("shot_id")
    order = shot.sort_values("shot_order")
    within_cycle = (
        order.groupby("cycle_index", observed=True)["sweep_value_s"]
        .apply(lambda values: [float(v) for v in values])
        .tolist()
    )
    expected_order = sorted(float(v) for v in shot["sweep_value_s"].unique())
    fixed_order = bool(
        within_cycle
        and all(np.allclose(values, expected_order) for values in within_cycle)
    )
    frame_table = (
        data[["frame_index", "frame_name", "exposure_s"]]
        .drop_duplicates()
        .sort_values("frame_index")
    )
    return {
        "dataset_id": str(data["dataset_id"].iloc[0]),
        "sequence_type": str(data["sequence_type"].iloc[0]),
        "n_complete_shots": int(shot["shot_id"].nunique()),
        "n_incomplete_shots": 0,
        "n_frames": int(data["frame_index"].nunique()),
        "frame_names": frame_table["frame_name"].astype(str).tolist(),
        "commanded_exposure_s": sorted(
            float(v) for v in frame_table["exposure_s"].dropna().unique()
        ),
        "sweep_values_s": expected_order,
        "n_cycles": int(shot["cycle_index"].nunique()),
        "fixed_ascending_order_within_cycle": fixed_order,
        "condition_is_confounded_with_within_cycle_position": fixed_order,
        "condition_split_counts": _condition_split_counts(data),
        "split_definition": run_meta.get("split"),
        "timing_source": run_meta.get("timing", {}).get("source"),
        "frame_starts_are_hardware_timestamps": run_meta.get("timing", {}).get(
            "frame_starts_are_hardware_timestamps"
        ),
    }


def _compiled_command_audit_verified(
    payload: dict[str, Any],
    *,
    cfg,
    dataset_meta: dict[str, Any],
    n_shots: int,
) -> tuple[bool, dict[str, Any]]:
    """Require raw-HDF5 command evidence tied to this config/input manifest."""

    command = payload.get("command_audit", {})
    provenance = payload.get("provenance", {})
    audit_config = provenance.get("config", {})
    audit_inputs = provenance.get("inputs")
    expected_inputs = dataset_meta.get("provenance", {}).get("inputs")
    checks = {
        "command_audit_required": command.get("required") is True,
        "all_shots_passed": command.get("all_shots_passed") is True,
        "trigger_edges_verified_every_shot": (
            command.get("trigger_edges_verified_every_shot") is True
        ),
        "switch_states_verified_every_shot": (
            command.get("switch_states_verified_every_shot") is True
        ),
        "dds_constancy_verified_every_shot": (
            command.get("dds_constancy_verified_every_shot") is True
        ),
        "dds_state_constant_across_shots": all(
            value is True
            for value in command.get(
                "dds_state_constant_across_shots", {}
            ).values()
        )
        and bool(command.get("dds_state_constant_across_shots")),
        "shot_count_matches_processed_table": (
            int(command.get("n_shots_audited", -1)) == int(n_shots)
        ),
        "dataset_id_matches": (
            audit_config.get("dataset_id") == cfg.dataset_id
        ),
        "config_hash_matches": (
            audit_config.get("config_sha256") == cfg.config_sha256
        ),
        "input_manifest_matches_processed_table": (
            isinstance(audit_inputs, dict)
            and isinstance(expected_inputs, dict)
            and audit_inputs == expected_inputs
        ),
    }
    return bool(all(checks.values())), {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "scope": command.get("scope"),
        "max_abs_trigger_edge_error_s": command.get(
            "max_abs_trigger_edge_error_s"
        ),
        "dds_state_variants": command.get("dds_state_variants"),
        "failure_messages": command.get("failure_messages", []),
    }


def _command_timing_verified(
    dark: pd.DataFrame,
    bright: pd.DataFrame,
    dark_meta: dict[str, Any],
    bright_meta: dict[str, Any],
    dark_audit: dict[str, Any],
    bright_audit: dict[str, Any],
    dark_cfg,
    bright_cfg,
) -> tuple[bool, dict[str, Any]]:
    """Verify commanded timing/state without implying an optical readback."""
    dark_exposure = np.allclose(
        dark["exposure_s"].dropna().astype(float).unique(), [0.05]
    )
    bright_exposure = np.allclose(
        bright["exposure_s"].dropna().astype(float).unique(), [0.05]
    )
    dark_later = dark.loc[dark["frame_index"].astype(int) > 0]
    dark_gap = np.allclose(
        dark_later["interframe_gap_s"].astype(float),
        dark_later["sweep_value_s"].astype(float),
        rtol=0,
        atol=1e-9,
    )
    bright_later = bright.loc[bright["frame_index"].astype(int) == 1]
    bright_gap = np.allclose(
        bright_later["interframe_gap_s"].astype(float), 0.01, rtol=0, atol=1e-9
    )
    metadata_checks = bool(
        dark_meta.get("shot_metadata", {}).get("exposure_matches_config")
        and bright_meta.get("shot_metadata", {}).get("exposure_matches_config")
        and dark_meta.get("shot_metadata", {}).get("shot_ids_unique")
        and bright_meta.get("shot_metadata", {}).get("shot_ids_unique")
    )
    dark_raw_passed, dark_raw_evidence = _compiled_command_audit_verified(
        dark_audit,
        cfg=dark_cfg,
        dataset_meta=dark_meta,
        n_shots=int(dark[["run_id", "shot_id"]].drop_duplicates().shape[0]),
    )
    bright_raw_passed, bright_raw_evidence = _compiled_command_audit_verified(
        bright_audit,
        cfg=bright_cfg,
        dataset_meta=bright_meta,
        n_shots=int(bright[["run_id", "shot_id"]].drop_duplicates().shape[0]),
    )
    passed = bool(
        dark_exposure
        and bright_exposure
        and dark_gap
        and bright_gap
        and metadata_checks
        and dark_raw_passed
        and bright_raw_passed
    )
    return passed, {
        "passed": passed,
        "scope": (
            "commanded EXPOSURES timing and compiled switch/DDS commands; "
            "not measured light power or per-frame hardware timestamps"
        ),
        "dark_exposure_50ms": bool(dark_exposure),
        "bright_exposure_50ms": bool(bright_exposure),
        "dark_interframe_gaps_match_sweep": bool(dark_gap),
        "bright_interframe_gap_is_10ms": bool(bright_gap),
        "metadata_consistency": metadata_checks,
        "compiled_command_state_consistent": bool(
            dark_raw_passed and bright_raw_passed
        ),
        "raw_hdf5_command_audits": {
            "dark_hold": dark_raw_evidence,
            "bright_wait": bright_raw_evidence,
        },
        "optical_power_readback_available": False,
    }


def _background_evidence(
    payload: dict[str, Any], dataset_id: str
) -> tuple[bool, str | None, dict[str, Any]]:
    block = payload.get("datasets", {}).get(dataset_id, {})
    selection = block.get("selection", {})
    selected = selection.get("selected_method")
    frozen = bool(
        selection.get("passed")
        and selection.get("configured_matches_evidence")
        and selected in {"global", "spatial", "template"}
    )
    return frozen, selected, block


def _count_column(method: str | None) -> str:
    if method is None:
        return "count_corrected_template"
    requested = BACKGROUND_VALUE_COLUMNS[method]
    return requested


def _config_provenance(cfg, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_file": cfg.config_path.name,
        "config_sha256": cfg.config_sha256,
        "input_manifest": meta.get("provenance", {}).get("inputs"),
        "schema_version": meta.get("schema_version"),
        "geometry_version": meta.get("geometry_version"),
        "geometry_fit_scope": meta.get("fit_scope"),
        "background_method_configured": meta.get("background", {}).get(
            "primary_method"
        ),
        "split_definition": meta.get("split"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark-config", default=DEFAULT_DARK_CONFIG)
    parser.add_argument("--bright-config", default=DEFAULT_BRIGHT_CONFIG)
    parser.add_argument(
        "--geometry-report",
        default="reports/validation/loss_sweeps_geometry/geometry_validation.json",
    )
    parser.add_argument(
        "--background-report",
        default=(
            "reports/validation/loss_sweeps_background/"
            "background_method_comparison.json"
        ),
    )
    parser.add_argument(
        "--dark-command-audit",
        default=DEFAULT_DARK_COMMAND_AUDIT,
        help=(
            "raw-HDF5 command audit from audit_source_data.py for the "
            "switch-off run"
        ),
    )
    parser.add_argument(
        "--bright-command-audit",
        default=DEFAULT_BRIGHT_COMMAND_AUDIT,
        help=(
            "raw-HDF5 command audit from audit_source_data.py for the "
            "bright-wait run"
        ),
    )
    parser.add_argument("--output", default=DEFAULT_RESULT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--skip-latent-gate",
        action="store_true",
        help="record the latent gate as not attempted",
    )
    args = parser.parse_args()

    dark_cfg = load_config(args.dark_config)
    bright_cfg = load_config(args.bright_config)
    dark, _dark_sites, dark_meta = load_dataset(dark_cfg)
    bright, _bright_sites, bright_meta = load_dataset(bright_cfg)
    dark_run = load_run_metadata(dark_cfg)
    bright_run = load_run_metadata(bright_cfg)

    geometry_path = Path(args.geometry_report)
    background_path = Path(args.background_report)
    dark_command_audit_path = Path(args.dark_command_audit)
    bright_command_audit_path = Path(args.bright_command_audit)
    if not geometry_path.is_absolute():
        geometry_path = ROOT / geometry_path
    if not background_path.is_absolute():
        background_path = ROOT / background_path
    if not dark_command_audit_path.is_absolute():
        dark_command_audit_path = ROOT / dark_command_audit_path
    if not bright_command_audit_path.is_absolute():
        bright_command_audit_path = ROOT / bright_command_audit_path
    geometry = _read_json(geometry_path, "geometry validation report")
    background = _read_json(background_path, "background validation report")
    dark_command_audit = _read_json(
        dark_command_audit_path, "dark-hold raw command audit"
    )
    bright_command_audit = _read_json(
        bright_command_audit_path, "bright-wait raw command audit"
    )

    dark_geometry = bool(
        geometry.get("reports", {})
        .get(dark_cfg.dataset_id, {})
        .get("gate", {})
        .get("passed")
    )
    bright_geometry = bool(
        geometry.get("reports", {})
        .get(bright_cfg.dataset_id, {})
        .get("gate", {})
        .get("passed")
    )
    dark_background, dark_method, dark_background_block = _background_evidence(
        background, dark_cfg.dataset_id
    )
    bright_background, bright_method, bright_background_block = (
        _background_evidence(background, bright_cfg.dataset_id)
    )
    timing_verified, timing_evidence = _command_timing_verified(
        dark,
        bright,
        dark_meta,
        bright_meta,
        dark_command_audit,
        bright_command_audit,
        dark_cfg,
        bright_cfg,
    )

    analysis, _contexts = run_loss_sweep_analysis(
        dark,
        bright,
        dark_count_column=_count_column(dark_method),
        bright_count_column=_count_column(bright_method),
        n_boot=args.bootstrap,
        random_seed=args.seed,
        timing_verified=timing_verified,
        dark_geometry_validated=dark_geometry,
        bright_geometry_validated=bright_geometry,
        dark_background_frozen=dark_background,
        bright_background_frozen=bright_background,
        run_latent_gate=not args.skip_latent_gate,
        progress=lambda message: print(f"[analysis] {message}", flush=True),
    )
    analysis["dataset_audit"] = {
        "dark_hold": _dataset_summary(dark, dark_run),
        "bright_wait": _dataset_summary(bright, bright_run),
    }
    analysis["timing_evidence"] = timing_evidence
    analysis["geometry_evidence"] = {
        "all_required_gates_passed": geometry.get(
            "all_required_geometry_gates_passed"
        ),
        "dark_hold": geometry.get("reports", {}).get(dark_cfg.dataset_id),
        "bright_wait": geometry.get("reports", {}).get(bright_cfg.dataset_id),
        "cross_run": geometry.get("cross_run"),
    }
    analysis["background_evidence"] = {
        "all_configured_backgrounds_supported": background.get(
            "all_configured_backgrounds_supported"
        ),
        "dark_hold": dark_background_block,
        "bright_wait": bright_background_block,
        "selection_rule": background.get("selection_rule"),
    }
    analysis["provenance"] = {
        "inference_repository": git_describe(ROOT),
        "lab_analysis_repository_commit": _dependency_commit(dark_cfg),
        "dark_hold": _config_provenance(dark_cfg, dark_meta),
        "bright_wait": _config_provenance(bright_cfg, bright_meta),
        "geometry_report_sha256": hash_file(geometry_path),
        "background_report_sha256": hash_file(background_path),
        "dark_command_audit_sha256": hash_file(dark_command_audit_path),
        "bright_command_audit_sha256": hash_file(
            bright_command_audit_path
        ),
        "random_seed": int(args.seed),
        "model_version": analysis["analysis_version"],
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(analysis), indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")

    manifest_dir = dark_cfg.paths.reports_root / "validation" / "loss_sweeps_analysis"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "result_file": output.name,
        "result_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bootstrap_replicates": int(args.bootstrap),
        "random_seed": int(args.seed),
        "latent_gate_passed": bool(
            analysis.get("latent_state", {}).get("gate_passed")
        ),
    }
    (manifest_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
