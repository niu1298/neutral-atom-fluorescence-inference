"""Build and validate the standardized frame-site table.

    python scripts/export_processed_dataset.py --config configs/paired_100ms.yaml

Writes ``data/processed/<dataset_id>.parquet`` (plus a site table and a JSON
metadata sidecar) and, unless ``--no-qc`` is given, the quality-control report
under ``reports/qc/``.

Fails loudly when the raw shots are missing, when the shot count contradicts
the configuration, or when the resulting table fails schema validation. It
never substitutes synthetic data for missing measurements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluorescence_inference import quality_control as qc  # noqa: E402
from fluorescence_inference import schema  # noqa: E402
from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import (  # noqa: E402
    SourceDataError, build_frame_site_table, write_dataset,
)
from fluorescence_inference.provenance import stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paired_100ms.yaml")
    ap.add_argument("--no-qc", action="store_true", help="skip the QC report")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        df, sites_df, ctx = build_frame_site_table(cfg, progress=not args.quiet)
    except SourceDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = schema.validate(
        df,
        expected_frames=cfg.n_frames,
        expected_shots=cfg["source"].get("expected_n_shots"),
        expected_sites=cfg.n_sites_expected,
        version=cfg.schema_version,
    )
    meta = dict(ctx["meta"])
    meta["provenance"] = stamp(cfg, inputs=ctx["shots"])
    meta["validation"] = {
        "ok": report.ok, "errors": report.errors,
        "warnings": report.warnings, "stats": report.stats,
    }
    meta["schema"] = {
        "primary_key": list(schema.primary_key_for_version(cfg.schema_version)),
        "columns": {c: {"dtype": t, "nullable": n, "description": d}
                    for c, (t, n, d) in
                    schema.columns_for_version(cfg.schema_version).items()},
    }

    paths = write_dataset(
        cfg, df, sites_df, meta,
        run_metadata=ctx.get("run_metadata"),
        split_manifest=ctx.get("split_manifest"),
    )

    if not args.quiet:
        s = report.stats
        print(f"rows ................. {len(df)}")
        print(f"shots x frames x sites {s['n_shots']} x {len(s['frame_ids'])} x {s['n_sites']}")
        print(f"duplicate keys ....... {s['duplicate_primary_keys']}")
        print(f"flagged rows ......... {s['n_rows_flagged']} "
              f"({', '.join(v for v in s['quality_flag_values'] if v != 'ok') or 'none'})")
        print(f"sites detected ....... {int(sites_df['site_detected'].sum())}"
              f"/{len(sites_df)}")
        for name, path in paths.items():
            print(f"{name:<20} {path.name}")

    if not report.ok:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for e in report.errors:
            print(f"  - {e}", file=sys.stderr)
        return 3
    for w in report.warnings:
        print(f"warning: {w}")

    if not args.no_qc:
        qc_out = qc.run_quality_control(cfg, df, sites_df, ctx)
        if not args.quiet:
            print(f"\nqc summary ........... {Path(qc_out['summary_path']).name}")
            print(f"qc figures ........... {len(qc_out['figures'])}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
