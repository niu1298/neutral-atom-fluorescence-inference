"""Read-only audit of the raw shot files.

Answers, from the files themselves rather than from assumption: how many shots
exist, whether each has exactly the configured frames, what the frames actually
are, what the exposures and inter-frame timing were, what metadata exists, and
whether any control data (reference, empty, dark) is present.  For the schema-3
loss sweeps it also decodes every compiled camera-trigger/science-switch word
and both compiled DDS raw programs.

The backwards-compatible paired-readout default is
``reports/audit/source_audit.json``.  Sweep reports use
``reports/audit/<dataset_id>_source_audit.json`` so the two runs cannot
overwrite each other.  Every HDF5 file is opened read-only.

    python scripts/audit_source_data.py --config configs/paired_100ms.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.command_audit import (  # noqa: E402
    SUPPORTED_SEQUENCE_TYPES,
    CommandAuditError,
    audit_compiled_commands,
    spec_from_config,
    summarize_command_audits,
)
from fluorescence_inference.dataset import discover_shots, read_shot_meta  # noqa: E402
from fluorescence_inference.provenance import stamp  # noqa: E402

#: dataset paths that would indicate a control acquisition rather than atoms
CONTROL_HINTS = ("reference", "ref", "empty", "dark", "background", "bg",
                 "probe", "noatoms", "no_atoms")


def walk_datasets(f: h5py.File, root: str = "images") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if root not in f:
        return out

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out[f"{root}/{name}"] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

    f[root].visititems(visit)
    return out


def audit(cfg) -> dict[str, Any]:
    shots = discover_shots(cfg)
    frames_cfg = {int(s["frame_id"]): s["h5_path"] for s in cfg.frame_specs}
    saturation = float(cfg["source"]["saturation_adu"])
    command_audit_required = cfg.sequence_type in SUPPORTED_SEQUENCE_TYPES
    command_spec = (
        spec_from_config(cfg) if command_audit_required else None
    )

    per_shot: list[dict[str, Any]] = []
    command_records: list[dict[str, Any]] = []
    image_inventory: dict[str, int] = {}
    dataset_shapes: dict[str, set[str]] = {}
    control_candidates: dict[str, int] = {}

    for i, p in enumerate(shots):
        meta = read_shot_meta(p, cfg, order=i)
        rec: dict[str, Any] = {
            "file": p.name,
            "bytes": p.stat().st_size,
            "shot_id": meta.shot_id,
            "shot_order": i,
            "timestamp": meta.timestamp,
            "sequence_index": meta.sequence_index,
            "script_basename": meta.script_basename,
            "n_runs_attr": meta.n_runs,
            "exposure_ms": {str(k): v for k, v in meta.exposure_ms.items()},
            "frame_elapsed_s": {str(k): v for k, v in meta.frame_elapsed_s.items()},
            "exposure_names": {str(k): v for k, v in meta.exposure_names.items()},
            "frames_present": {},
            "frame_stats": {},
        }
        with h5py.File(p, "r") as f:
            if command_spec is not None:
                try:
                    command_record = audit_compiled_commands(f, command_spec)
                except CommandAuditError as exc:
                    command_record = {
                        "passed": False,
                        "schema_version": "1.0",
                        "failures": [str(exc)],
                    }
                rec["command_program"] = command_record
                command_records.append(command_record)
            inv = walk_datasets(f)
            for k, v in inv.items():
                image_inventory[k] = image_inventory.get(k, 0) + 1
                dataset_shapes.setdefault(k, set()).add(f"{v['shape']} {v['dtype']}")
                if any(h in k.lower() for h in CONTROL_HINTS):
                    control_candidates[k] = control_candidates.get(k, 0) + 1
            for fid, h5_path in frames_cfg.items():
                present = h5_path in f
                rec["frames_present"][str(fid)] = present
                if not present:
                    continue
                arr = np.asarray(f[h5_path][:], dtype=np.float64)
                rec["frame_stats"][str(fid)] = {
                    "shape": list(arr.shape),
                    "dtype": str(f[h5_path].dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "median": float(np.median(arr)),
                    "mean": float(arr.mean()),
                    "n_saturated_px": int(np.count_nonzero(arr >= saturation)),
                    "n_nonfinite_px": int(np.count_nonzero(~np.isfinite(arr))),
                }
        per_shot.append(rec)

    return {
        "provenance": stamp(cfg, inputs=shots),
        "shot_directory": cfg["source"]["shot_subdir"],
        "n_shot_files": len(shots),
        "expected_n_shots": cfg["source"].get("expected_n_shots"),
        "image_dataset_inventory": {
            k: {"present_in_n_shots": v, "shapes_seen": sorted(dataset_shapes[k])}
            for k, v in sorted(image_inventory.items())
        },
        "control_dataset_candidates": control_candidates,
        "per_shot": per_shot,
        "aggregate": aggregate(per_shot, cfg),
        "command_audit": summarize_command_audits(
            command_records,
            required=command_audit_required,
        ),
    }


def aggregate(per_shot: list[dict[str, Any]], cfg) -> dict[str, Any]:
    frame_ids = [str(int(s["frame_id"])) for s in cfg.frame_specs]
    complete = [r for r in per_shot if all(r["frames_present"].get(f) for f in frame_ids)]
    exposures = sorted({v for r in per_shot for v in r["exposure_ms"].values()})
    deltas = sorted({round(r["frame_elapsed_s"]["1"] - r["frame_elapsed_s"]["0"], 9)
                     for r in per_shot
                     if "0" in r["frame_elapsed_s"] and "1" in r["frame_elapsed_s"]})
    shapes = sorted({tuple(st["shape"]) for r in per_shot for st in r["frame_stats"].values()})
    dtypes = sorted({st["dtype"] for r in per_shot for st in r["frame_stats"].values()})
    sat = sum(st["n_saturated_px"] for r in per_shot for st in r["frame_stats"].values())
    nonfin = sum(st["n_nonfinite_px"] for r in per_shot for st in r["frame_stats"].values())

    by_frame: dict[str, Any] = {}
    for fid in frame_ids:
        med = [r["frame_stats"][fid]["median"] for r in per_shot if fid in r["frame_stats"]]
        mx = [r["frame_stats"][fid]["max"] for r in per_shot if fid in r["frame_stats"]]
        if med:
            by_frame[fid] = {
                "median_counts": {"mean": float(np.mean(med)), "std": float(np.std(med, ddof=1)),
                                  "min": float(np.min(med)), "max": float(np.max(med))},
                "max_pixel": {"min": float(np.min(mx)), "max": float(np.max(mx))},
            }
    shift = None
    if {"0", "1"} <= set(by_frame):
        m0 = np.array([r["frame_stats"]["0"]["median"] for r in per_shot])
        m1 = np.array([r["frame_stats"]["1"]["median"] for r in per_shot])
        d = m1 - m0
        shift = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                 "min": float(d.min()), "max": float(d.max()),
                 "n_shots_negative": int((d < 0).sum())}

    return {
        "n_shots_with_all_frames": len(complete),
        "n_shots_missing_a_frame": len(per_shot) - len(complete),
        "shot_ids_unique": len({r["shot_id"] for r in per_shot}) == len(per_shot),
        "shot_id_range": [min(r["shot_id"] for r in per_shot),
                          max(r["shot_id"] for r in per_shot)] if per_shot else None,
        "timestamp_first": per_shot[0]["timestamp"] if per_shot else None,
        "timestamp_last": per_shot[-1]["timestamp"] if per_shot else None,
        "exposure_ms_values": exposures,
        "interframe_start_delta_s_values": deltas,
        "frame_shapes": [list(s) for s in shapes],
        "frame_dtypes": dtypes,
        "total_saturated_px": sat,
        "total_nonfinite_px": nonfin,
        "per_frame": by_frame,
        "frame1_minus_frame0_median_shift": shift,
        "script_basenames": sorted({r["script_basename"] for r in per_shot
                                    if r["script_basename"]}),
        "sequence_indices": sorted({r["sequence_index"] for r in per_shot
                                    if r["sequence_index"] is not None}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paired_100ms.yaml")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = audit(cfg)

    default_name = (
        f"{cfg.dataset_id}_source_audit.json"
        if cfg.sequence_type in SUPPORTED_SEQUENCE_TYPES
        else "source_audit.json"
    )
    out = (
        Path(args.out)
        if args.out
        else cfg.reports_dir("audit", default_name)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    agg = report["aggregate"]
    print(f"shots ................ {report['n_shot_files']}")
    print(f"complete frame sets .. {agg['n_shots_with_all_frames']}")
    print(f"exposure (ms) ........ {agg['exposure_ms_values']}")
    print(f"inter-frame delta (s)  {agg['interframe_start_delta_s_values']}")
    print(f"frame shapes ......... {agg['frame_shapes']} {agg['frame_dtypes']}")
    print(f"saturated pixels ..... {agg['total_saturated_px']}")
    if agg["frame1_minus_frame0_median_shift"]:
        s = agg["frame1_minus_frame0_median_shift"]
        print(f"frame1-frame0 median . {s['mean']:+.2f} +/- {s['std']:.2f} counts/px")
    print(f"control candidates ... {report['control_dataset_candidates'] or 'none'}")
    command = report["command_audit"]
    if command["required"]:
        print(
            "compiled commands .... "
            f"{command['n_shots_passed']}/{command['n_shots_audited']} "
            "shots passed"
        )
        edge_error = command["max_abs_trigger_edge_error_s"]
        print(
            "trigger edge error ... "
            + (
                "unavailable"
                if edge_error is None
                else f"{edge_error:.3g} s maximum"
            )
        )
    print(f"written .............. {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out.name}")
    if command["required"] and not command["all_shots_passed"]:
        for message in command["failure_messages"]:
            print(f"command audit failure  {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
