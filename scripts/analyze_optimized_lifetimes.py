"""Analyze the five optimized 2026-07-31 lifetime datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import (  # noqa: E402
    load_dataset,
    load_run_metadata,
    load_split_manifest,
)
from fluorescence_inference.loss_sweep_analysis import (  # noqa: E402
    BACKGROUND_VALUE_COLUMNS,
)
from fluorescence_inference.optimized_analysis import (  # noqa: E402
    run_optimized_analysis,
)
from fluorescence_inference.provenance import git_describe, hash_file  # noqa: E402


DEFAULT_CONFIGS = {
    "bright": "configs/bright_lifetime_20260731_0090.yaml",
    "dark": "configs/dark_lifetime_20260731_0094.yaml",
    "50ms": "configs/imaging_5frame_50ms_20260731_0113.yaml",
    "100ms": "configs/imaging_5frame_100ms_20260731_0114.yaml",
    "200ms": "configs/imaging_5frame_200ms_20260731_0115.yaml",
}
DEFAULT_GEOMETRY = (
    "reports/validation/optimized_lifetimes_geometry/geometry_validation.json"
)
DEFAULT_BACKGROUND = (
    "reports/validation/optimized_lifetimes_background/"
    "background_method_comparison.json"
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _public_background_name(configured: str) -> str:
    return "template" if configured == "fixed_offset" else configured


def _audit_path(cfg) -> Path:
    return cfg.paths.reports_root / "audit" / f"{cfg.dataset_id}_source_audit.json"


def _timing_verified(tables: dict[str, Any], audits: dict[str, Any]) -> bool:
    bright = tables["bright"]
    dark = tables["dark"]
    bright_later = bright.loc[bright["frame_index"].astype(int) == 1]
    dark_later = dark.loc[dark["frame_index"].astype(int) > 0]
    checks = [
        np.allclose(bright["exposure_s"].astype(float), 0.05),
        np.allclose(dark["exposure_s"].astype(float), 0.05),
        np.allclose(
            bright["bright_wait_s"].astype(float),
            bright["sweep_value_s"].astype(float),
        ),
        np.allclose(bright_later["dark_hold_s"].astype(float), 0.01),
        np.allclose(bright_later["interframe_gap_s"].astype(float), 0.010002),
        np.allclose(
            dark_later["dark_hold_s"].astype(float),
            dark_later["sweep_value_s"].astype(float),
        ),
        np.allclose(
            dark_later["interframe_gap_s"].astype(float),
            dark_later["sweep_value_s"].astype(float) + 0.000002,
        ),
        all(
            audit.get("command_audit", {}).get("all_shots_passed") is True
            for audit in audits.values()
        ),
    ]
    return bool(all(checks))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, path in DEFAULT_CONFIGS.items():
        parser.add_argument(f"--{key}-config", default=path)
    parser.add_argument("--geometry-report", default=DEFAULT_GEOMETRY)
    parser.add_argument("--background-report", default=DEFAULT_BACKGROUND)
    parser.add_argument(
        "--output", default="reports/optimized_lifetime_results_20260731.json"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--require-clean-provenance", action="store_true")
    parser.add_argument(
        "--allow-stale-development-inputs",
        action="store_true",
        help=(
            "allow recovered dirty-tree exports during model development; "
            "the result is explicitly non-publishable"
        ),
    )
    args = parser.parse_args()
    repository = git_describe(ROOT)
    if args.require_clean_provenance and repository.get("dirty") is not False:
        parser.error("clean provenance requires a clean worktree")
    if args.require_clean_provenance and args.allow_stale_development_inputs:
        parser.error("clean provenance cannot allow stale development inputs")

    configs = {
        key: load_config(getattr(args, f"{key}_config"))
        for key in DEFAULT_CONFIGS
    }
    loaded = {key: load_dataset(cfg) for key, cfg in configs.items()}
    tables = {key: value[0] for key, value in loaded.items()}
    metadata = {key: value[2] for key, value in loaded.items()}
    run_metadata = {key: load_run_metadata(cfg) for key, cfg in configs.items()}
    split_manifests = {
        key: load_split_manifest(configs[key])
        for key in ("50ms", "100ms", "200ms")
    }
    audits = {key: _read_json(_audit_path(cfg), f"{key} source audit") for key, cfg in configs.items()}

    geometry_path = (ROOT / args.geometry_report).resolve()
    background_path = (ROOT / args.background_report).resolve()
    geometry = _read_json(geometry_path, "optimized geometry report")
    background = _read_json(background_path, "optimized background report")
    stale = []
    geometry_validated = {}
    background_frozen = {}
    selected_columns = {}
    key_by_dataset = {cfg.dataset_id: key for key, cfg in configs.items()}
    for dataset_id, key in key_by_dataset.items():
        cfg = configs[key]
        meta = metadata[key]
        audit = audits[key]
        processed_hash = (
            meta.get("provenance", {}).get("config", {}).get("config_sha256")
        )
        audit_hash = (
            audit.get("provenance", {}).get("config", {}).get("config_sha256")
        )
        input_hash = meta.get("provenance", {}).get("inputs")
        audit_inputs = audit.get("provenance", {}).get("inputs")
        if processed_hash != cfg.config_sha256:
            stale.append(f"{key}: processed config hash")
        if audit_hash != cfg.config_sha256:
            stale.append(f"{key}: source-audit config hash")
        if input_hash != audit_inputs:
            raise ValueError(f"{key}: processed and source-audit input manifests differ")

        geometry_block = geometry.get("reports", {}).get(dataset_id, {})
        geometry_validated[key] = bool(
            geometry_block.get("gate", {}).get("passed")
        )
        background_block = background.get("datasets", {}).get(dataset_id, {})
        selection = background_block.get("selection", {})
        selected = selection.get("selected_method")
        configured = _public_background_name(
            str(cfg["background"]["primary_method"])
        )
        background_frozen[key] = bool(
            selection.get("passed")
            and selected == configured
            and selected in BACKGROUND_VALUE_COLUMNS
        )
        if selected not in BACKGROUND_VALUE_COLUMNS:
            raise ValueError(f"{key}: no valid selected background method")
        selected_columns[key] = BACKGROUND_VALUE_COLUMNS[selected]

    if stale and not args.allow_stale_development_inputs:
        parser.error(
            "stale prerequisite bindings: " + "; ".join(stale)
        )
    if not all(geometry_validated.values()):
        parser.error("one or more optimized geometry gates failed")
    if not all(background_frozen.values()):
        parser.error("one or more optimized background selections are not frozen")
    timing = _timing_verified(tables, audits)
    if not timing:
        parser.error("optimized timing/compiled-command gate failed")

    result, _context = run_optimized_analysis(
        tables["dark"],
        tables["bright"],
        {0.05: tables["50ms"], 0.10: tables["100ms"], 0.20: tables["200ms"]},
        dark_count_column=selected_columns["dark"],
        bright_count_column=selected_columns["bright"],
        repeated_count_columns={
            0.05: selected_columns["50ms"],
            0.10: selected_columns["100ms"],
            0.20: selected_columns["200ms"],
        },
        repeated_alternate_split_data={
            exposure: tables[key]
            .drop(columns=["split"])
            .merge(
                split_manifests[key][["shot_id", "sensitivity_split"]].rename(
                    columns={"sensitivity_split": "split"}
                ),
                on="shot_id",
                how="left",
                validate="many_to_one",
            )
            for exposure, key in (
                (0.05, "50ms"),
                (0.10, "100ms"),
                (0.20, "200ms"),
            )
        },
        n_boot=args.bootstrap,
        random_seed=args.seed,
        timing_verified=timing,
        geometry_validated={
            "dark": geometry_validated["dark"],
            "bright": geometry_validated["bright"],
            "50ms": geometry_validated["50ms"],
            "100ms": geometry_validated["100ms"],
            "200ms": geometry_validated["200ms"],
        },
        background_frozen={
            "dark": background_frozen["dark"],
            "bright": background_frozen["bright"],
            "50ms": background_frozen["50ms"],
            "100ms": background_frozen["100ms"],
            "200ms": background_frozen["200ms"],
        },
        progress=lambda message: print(f"[optimized] {message}", flush=True),
    )
    result["dataset_audit"] = {
        key: {
            "dataset_id": cfg.dataset_id,
            "n_shots": int(tables[key]["shot_id"].nunique()),
            "n_frames": int(tables[key]["frame_index"].nunique()),
            "exposure_s": sorted(
                float(value) for value in tables[key]["exposure_s"].unique()
            ),
            "split": run_metadata[key].get("split"),
            "background_method": _public_background_name(
                str(cfg["background"]["primary_method"])
            ),
        }
        for key, cfg in configs.items()
    }
    result["prerequisite_evidence"] = {
        "timing_and_compiled_commands_verified": timing,
        "geometry_validated": geometry_validated,
        "background_frozen": background_frozen,
    }
    result["provenance"] = {
        "analysis_code_commit": repository.get("commit"),
        "repository": repository,
        "bootstrap_replicates": int(args.bootstrap),
        "random_seed": int(args.seed),
        "geometry_report_sha256": hash_file(geometry_path),
        "background_report_sha256": hash_file(background_path),
        "configs": {
            key: {
                "file": cfg.config_path.name,
                "config_sha256": cfg.config_sha256,
                "processed_input_manifest": metadata[key].get(
                    "provenance", {}
                ).get("inputs"),
                "source_audit_sha256": hash_file(_audit_path(cfg)),
            }
            for key, cfg in configs.items()
        },
        "development_stale_bindings": stale,
        "publishable_clean_provenance": bool(
            repository.get("dirty") is False and not stale
        ),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output.name,
                "bootstrap": args.bootstrap,
                "selected_repeated_model": result["repeated_imaging"][
                    "selected_model"
                ],
                "publishable_clean_provenance": result["provenance"][
                    "publishable_clean_provenance"
                ],
                "development_stale_bindings": stale,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
