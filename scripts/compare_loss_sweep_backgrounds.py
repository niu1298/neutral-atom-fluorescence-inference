"""Compare physically motivated background methods on the two loss sweeps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import load_dataset  # noqa: E402
from fluorescence_inference.sweep_validation import (  # noqa: E402
    annulus_neighbour_overlap,
    frame_pedestal_summary,
    select_background_method,
    summarize_background_methods,
)


def _processed_binding(cfg, meta):
    """Bind this validation report to the exact processed-input contract."""
    provenance = meta.get("provenance", {})
    processed_config = provenance.get("config", {})
    processed_inputs = provenance.get("inputs", {})
    config_hash = processed_config.get("config_sha256")
    input_hash = processed_inputs.get("manifest_sha256")
    if config_hash != cfg.config_sha256 or not input_hash:
        raise ValueError(
            f"{cfg.dataset_id}: processed metadata is stale or lacks a "
            "config/input-manifest binding; regenerate the export"
        )
    return {
        "dataset_id": cfg.dataset_id,
        "config_sha256": cfg.config_sha256,
        "input_manifest_sha256": input_hash,
        "schema_version": meta.get("schema_version"),
        "geometry_version": meta.get("geometry_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    reports = {}
    cfgs = [load_config(p) for p in args.config]
    output_dir = cfgs[0].paths.reports_root / "validation" / "loss_sweeps_background"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pass = True

    for cfg in cfgs:
        df, sites, meta = load_dataset(cfg)
        if "split" not in df:
            raise ValueError(f"{cfg.dataset_id}: background selection requires a split")
        validation = df.loc[df["split"].astype(str) == "validation"].copy()
        validation_orders = set(validation["shot_order"].astype(int).unique())
        validation_frame_diagnostics = [
            row
            for row in meta.get("frame_background_diagnostics", [])
            if int(row["shot_order"]) in validation_orders
        ]
        methods = summarize_background_methods(
            validation,
            frame_diagnostics=validation_frame_diagnostics,
        )
        selected = select_background_method(methods)
        configured = str(cfg["background"]["primary_method"])
        configured_public = "template" if configured == "fixed_offset" else configured
        matches = selected.get("selected_method") == configured_public
        all_pass &= bool(selected["passed"] and matches)
        ann = annulus_neighbour_overlap(
            sites,
            tuple(meta["frame_shape"]),
            inner_half_width=int(cfg["legacy_annulus"]["inner_half_width"]),
            outer_half_width=int(cfg["legacy_annulus"]["outer_half_width"]),
            neighbour_mask_radius_px=float(cfg["background"]["mask_radius_px"]),
        )
        reports[cfg.dataset_id] = {
            "methods": methods,
            "selection": {
                **selected,
                "selection_split": "validation",
                "n_independent_validation_shots": int(
                    validation["shot_id"].nunique()
                ),
                "configured_primary_method": configured_public,
                "configured_matches_evidence": matches,
            },
            "provenance_binding": _processed_binding(cfg, meta),
            "annulus_contamination": ann,
            # Pedestal estimation is descriptive and does not select a method.
            # It is therefore reported for all shots, with its scope explicit.
            "site_free_frame_pedestal": frame_pedestal_summary(df),
            "occupancy_dependence_gate": (
                "Selection requires the validation-shot corrected-count versus "
                "site-free-background coupling reported above to remain below "
                "the predeclared method gate. Frozen train-only apparent-occupancy "
                "coupling is evaluated downstream as a separate sensitivity."
            ),
        }

    payload = {
        "datasets": reports,
        "all_configured_backgrounds_supported": all_pass,
        "selection_rule": (
            "validation-shot site-free residual spatial structure with "
            "background coupling as a gate/tie-break; model separation and "
            "test shots are excluded"
        ),
    }
    output = (
        Path(args.output)
        if args.output
        else output_dir / "background_method_comparison.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.name,
                "datasets": list(reports),
                "passed": all_pass,
            },
            indent=2,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
