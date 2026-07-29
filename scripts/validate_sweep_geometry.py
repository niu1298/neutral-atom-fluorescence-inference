"""Refit and validate the 2026-07-28 sweep geometries.

The primary geometry is fitted on training shots only.  Independent early,
late, shortest-condition and longest-condition fits are evidence for the
freeze; the all-data fit is retained only as an explicitly labelled
unsupervised sensitivity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import load_dataset  # noqa: E402
from fluorescence_inference.reporting import apply_style, draw_roi_boxes, save, show_image  # noqa: E402
from fluorescence_inference.sweep_validation import (  # noqa: E402
    fit_geometry_subsets,
    geometry_match_report,
    lattice_step_explanation,
)


def _shot_manifest(df):
    wanted = [
        "shot_id",
        "shot_order",
        "split",
        "condition_id",
        "sweep_value_s",
        "cycle_index",
        "repetition_index",
    ]
    cols = [c for c in wanted if c in df.columns]
    out = df[cols].drop_duplicates("shot_id").sort_values("shot_order")
    return out.reset_index(drop=True)


def _overlay(cfg, ctx: dict[str, Any], out: Path) -> Path:
    site_map = ctx["fits"]["train"]
    rows = ctx["subset_rows"]["all_unsupervised"].sort_values("shot_order")
    quantiles = np.linspace(0, len(rows) - 1, 4).round().astype(int)
    chosen = rows.iloc[quantiles]
    frame_path = str(cfg.frame_specs[0]["h5_path"])
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.1), constrained_layout=True)
    y = site_map.centers_yx[:, 0]
    x = site_map.centers_yx[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    pad = 12
    for ax, (_, shot) in zip(axes, chosen.iterrows()):
        path = ctx["paths_by_shot_id"][int(shot["shot_id"])]
        img = ctx["load"](path, frame_path)
        show_image(ax, img, percentile=(2, 99.7))
        draw_roi_boxes(ax, site_map.boxes, color="#57d0ff", lw=0.45)
        ax.set_xlim(max(0, x0 - pad), min(img.shape[1], x1 + pad))
        ax.set_ylim(min(img.shape[0], y1 + pad), max(0, y0 - pad))
        ax.set_title(
            f"shot {int(shot['shot_order'])} · {float(shot['sweep_value_s']):.1f} s",
            fontsize=9,
        )
        ax.set_xlabel("camera x (px)")
    axes[0].set_ylabel("camera y (px)")
    fig.suptitle("Training-fit ROI overlay across the run", fontsize=12)
    return save(fig, out, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Repeat for each sweep YAML; two configs enable cross-run registration.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON destination; defaults to the configured ignored validation directory.",
    )
    args = parser.parse_args()

    reports: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    configs = [load_config(p) for p in args.config]
    default_dir = configs[0].paths.reports_root / "validation" / "loss_sweeps_geometry"
    default_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        df, _sites, meta = load_dataset(cfg)
        provenance = meta.get("provenance", {})
        processed_config = provenance.get("config", {})
        processed_inputs = provenance.get("inputs", {})
        if (
            processed_config.get("config_sha256") != cfg.config_sha256
            or not processed_inputs.get("manifest_sha256")
        ):
            raise ValueError(
                f"{cfg.dataset_id}: processed metadata is stale or lacks a "
                "config/input-manifest binding; regenerate the export"
            )
        report, ctx = fit_geometry_subsets(cfg, _shot_manifest(df))
        report["provenance_binding"] = {
            "dataset_id": cfg.dataset_id,
            "config_sha256": cfg.config_sha256,
            "input_manifest_sha256": processed_inputs["manifest_sha256"],
            "schema_version": meta.get("schema_version"),
            "geometry_version": meta.get("geometry_version"),
            "fit_scope": meta.get("fit_scope"),
        }
        overlay = default_dir / f"{cfg.dataset_id}_roi_overlays.png"
        _overlay(cfg, ctx, overlay)
        report["overlay_file"] = overlay.name
        reports[cfg.dataset_id] = report
        contexts[cfg.dataset_id] = ctx

    cross_run = None
    if len(configs) == 2:
        a, b = configs
        map_a = contexts[a.dataset_id]["fits"].get("train")
        map_b = contexts[b.dataset_id]["fits"].get("train")
        if map_a is not None and map_b is not None:
            centroid_shift = np.median(map_b.centers_yx, axis=0) - np.median(
                map_a.centers_yx, axis=0
            )
            basis = [
                map_a.grids[0].row_vector_yx,
                map_a.grids[0].col_vector_yx,
            ]
            cross_run = {
                "datasets": [a.dataset_id, b.dataset_id],
                "camera_config_recorded_identical": True,
                "wide_radius_matching": geometry_match_report(
                    map_a.centers_yx, map_b.centers_yx, match_radius_px=15.0
                ),
                "centroid_shift_yx_px": [float(v) for v in centroid_shift],
                "lattice_step_test": lattice_step_explanation(centroid_shift, basis),
                "interpretation": (
                    "Recorded crop/offset metadata is identical. A one-basis-step "
                    "match is consistent with an index alias or a real one-pitch "
                    "translation; image data alone do not identify which."
                ),
            }

    payload = {
        "reports": reports,
        "cross_run": cross_run,
        "all_required_geometry_gates_passed": all(
            r["gate"]["passed"] for r in reports.values()
        ),
    }
    output = Path(args.output) if args.output else default_dir / "geometry_validation.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.name,
                "datasets": list(reports),
                "passed": payload["all_required_geometry_gates_passed"],
            },
            indent=2,
        )
    )
    return 0 if payload["all_required_geometry_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
