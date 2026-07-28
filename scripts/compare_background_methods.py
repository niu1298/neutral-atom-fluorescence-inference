"""Compare the four background methods and choose a primary on the evidence.

    python scripts/compare_background_methods.py --config configs/paired_100ms.yaml

Writes ``reports/validation/background_method_comparison.json`` and
``assets/readme/background_method_comparison.png``.

The recommendation is deliberately **not** "whichever maximises separation".
Maximising separation rewards any estimator that happens to subtract less, and
would pick "no correction" on almost any dataset. The ranking here is:

1. residual spatial structure left on site-free pixels — a background model
   that leaves the fringe pattern behind is wrong regardless of what it does
   to a histogram;
2. stability against acquisition order — a residual drift is a background the
   model failed to track;
3. how strongly the corrected counts still track the frame's own background —
   a valid correction should decorrelate them;
4. only then, the frame-0 to frame-1 offset and the descriptive separation,
   reported but not used to rank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fluorescence_inference import quality_control as qc  # noqa: E402
from fluorescence_inference import reporting as rp  # noqa: E402
from fluorescence_inference import schema  # noqa: E402
from fluorescence_inference.config import load_config  # noqa: E402
from fluorescence_inference.dataset import SourceDataError, load_dataset  # noqa: E402
from fluorescence_inference.descriptive_fit import fit_descriptive, load_em_fit  # noqa: E402
from fluorescence_inference.provenance import stamp  # noqa: E402

#: methods in report order, with the human-readable definition
METHODS = {
    "raw": "A — no background correction",
    "global": "B — global site-free median of the same frame",
    "spatial": "C — robust smooth spatial surface, refitted per frame",
    "fixed_offset": "D — fixed spatial template + per-frame common-mode offset",
    "annulus_contaminated": "legacy — local 13–33 px annulus (contaminated)",
}
RANKED = ["global", "spatial", "fixed_offset"]   # candidates for primary


def per_method_stats(df: pd.DataFrame, meta: dict[str, Any], em) -> dict[str, Any]:
    clean = df[df["quality_flag"].astype(str) == "ok"]
    n_px = int(df["roi_n_pixels"].mode().iat[0])
    frame_diag = pd.DataFrame(meta.get("frame_background_diagnostics", []))

    structure_key = {"global": "structure_global", "spatial": "structure_spatial",
                     "fixed_offset": "structure_fixed"}
    out: dict[str, Any] = {}
    for method, label in METHODS.items():
        bg_col, ct_col = schema.METHOD_COLUMNS[method]
        if ct_col not in clean.columns:
            continue
        block: dict[str, Any] = {"definition": label,
                                 "background_column": bg_col,
                                 "corrected_column": ct_col,
                                 "per_frame": {}}
        for fid, g in clean.groupby("frame_id", observed=True):
            counts = g[ct_col].to_numpy(float)
            per_shot = g.groupby("shot_order", observed=True)[ct_col].mean()
            slope = float(np.polyfit(per_shot.index.to_numpy(float),
                                     per_shot.to_numpy(float), 1)[0] * 100.0)
            fit = fit_descriptive(counts, em)
            entry = {
                "n": int(counts.size),
                "mean": float(counts.mean()),
                "std": float(counts.std(ddof=1)),
                "background_mean_per_px": (float(g[bg_col].mean() / n_px)
                                           if bg_col else 0.0),
                "background_std_per_px": (float(g[bg_col].std(ddof=1) / n_px)
                                          if bg_col else 0.0),
                "corrected_drift_per_100_shots": slope,
                "separation_d_prime": fit.separation_d_prime,
                "model_implied_overlap": fit.model_implied_overlap,
                "reference_level": fit.reference_level,
            }
            # does the corrected count still follow the frame's own background?
            if bg_col:
                shot_bg = g.groupby("shot_order", observed=True)[bg_col].mean()
                common = per_shot.index.intersection(shot_bg.index)
                entry["corr_corrected_vs_background_across_shots"] = float(
                    np.corrcoef(per_shot.loc[common], shot_bg.loc[common])[0, 1])
            else:
                entry["corr_corrected_vs_background_across_shots"] = float("nan")
            block["per_frame"][str(int(fid))] = entry

        # frame-1 minus frame-0 shift of the background itself
        if bg_col:
            w = clean.pivot_table(index=["shot_id", "site_id"], columns="frame_id",
                                  values=bg_col, observed=True)
            if {0, 1} <= set(w.columns):
                d = (w[1] - w[0]) / n_px
                block["background_frame1_minus_frame0_per_px"] = {
                    "mean": float(d.mean()), "std": float(d.std(ddof=1))}
        # frame-1 minus frame-0 shift of the *corrected* count
        wc = clean.pivot_table(index=["shot_id", "site_id"], columns="frame_id",
                               values=ct_col, observed=True)
        if {0, 1} <= set(wc.columns):
            dc = wc[1] - wc[0]
            block["corrected_frame1_minus_frame0"] = {
                "mean": float(dc.mean()), "median": float(dc.median()),
                "std": float(dc.std(ddof=1))}

        # residual structure left on site-free pixels
        key = structure_key.get(method)
        if key and not frame_diag.empty and key in frame_diag:
            st = pd.json_normalize(frame_diag[key])
            block["site_free_residual"] = {
                "block_median_std_mean": float(st["block_median_std"].mean()),
                "pixel_residual_rms_mean": float(st["pixel_residual_rms"].mean()),
                "block_px": int(st["block_px"].iloc[0]),
            }
        out[method] = block
    return out


def recommend(stats: dict[str, Any]) -> dict[str, Any]:
    """Rank the candidates on physical validity and stability."""
    rows = []
    for m in RANKED:
        if m not in stats:
            continue
        s = stats[m]
        resid = s.get("site_free_residual", {})
        drift = np.mean([abs(v["corrected_drift_per_100_shots"])
                         for v in s["per_frame"].values()])
        cvals = [abs(v["corr_corrected_vs_background_across_shots"])
                 for v in s["per_frame"].values()]
        cvals = [v for v in cvals if np.isfinite(v)]
        corr = float(np.mean(cvals)) if cvals else float("nan")
        rows.append({
            "method": m,
            "residual_block_median_std": resid.get("block_median_std_mean", float("nan")),
            "mean_abs_drift_per_100_shots": float(drift),
            "mean_abs_corr_with_background": float(corr),
            "mean_d_prime": float(np.mean([v["separation_d_prime"]
                                           for v in s["per_frame"].values()])),
        })
    if not rows:
        return {"selected": None, "reason": "no candidate methods present"}

    def rank(key: str) -> dict[str, int]:
        order = sorted(rows, key=lambda r: r[key])
        return {r["method"]: i for i, r in enumerate(order)}

    r1, r2, r3 = (rank("residual_block_median_std"),
                  rank("mean_abs_drift_per_100_shots"),
                  rank("mean_abs_corr_with_background"))
    for r in rows:
        r["rank_residual_structure"] = r1[r["method"]]
        r["rank_drift"] = r2[r["method"]]
        r["rank_background_coupling"] = r3[r["method"]]
        # residual structure counts double: it is the one criterion that speaks
        # to whether the model is describing the background at all
        r["score"] = 2 * r1[r["method"]] + r2[r["method"]] + r3[r["method"]]
    best = min(rows, key=lambda r: (r["score"], r["method"]))
    return {
        "candidates": rows,
        "selected": best["method"],
        "criteria": [
            "residual spatial structure on site-free pixels (weight 2)",
            "absolute drift of the corrected count with acquisition order",
            "absolute correlation between corrected count and the frame background",
        ],
        "explicitly_not_used": "the descriptive separation d'",
        "why_separation_is_not_a_criterion": (
            "d' rewards subtracting less. 'No correction' scores well on it "
            "while leaving the entire frame-dependent background in the "
            "signal, so ranking on d' would select the wrong model."),
    }


def figure(stats: dict[str, Any], rec: dict[str, Any], df: pd.DataFrame,
           out: Path) -> Path:
    rp.apply_style()
    present = [m for m in METHODS if m in stats]
    labels = [METHODS[m].split("—")[0].strip() for m in present]
    x = np.arange(len(present))

    fig, axes = rp.plt.subplots(1, 4, figsize=(19.0, 4.6), layout="constrained")

    ax = axes[0]
    vals = [stats[m].get("site_free_residual", {}).get("block_median_std_mean", np.nan)
            for m in present]
    bars = ax.bar(x, vals, color=[rp.OK if m == rec["selected"] else rp.MUTED
                                  for m in present])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("counts / pixel")
    ax.set_title("Residual spatial structure\n(lower is better)")
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9)

    ax = axes[1]
    for i, fid in enumerate((0, 1)):
        vals = [stats[m]["per_frame"][str(fid)]["corrected_drift_per_100_shots"]
                for m in present]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=rp.FRAME_COLORS[fid],
               label=f"frame {fid}")
    ax.axhline(0, color=rp.MUTED, lw=0.9, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("counts / 100 shots")
    ax.set_title("Drift of the corrected count\nwith acquisition order")
    ax.legend()

    ax = axes[2]
    for i, fid in enumerate((0, 1)):
        vals = [abs(stats[m]["per_frame"][str(fid)]
                    ["corr_corrected_vs_background_across_shots"]) for m in present]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=rp.FRAME_COLORS[fid],
               label=f"frame {fid}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("|correlation|")
    ax.set_title("Corrected count still tracking\nits own background")
    ax.legend()

    ax = axes[3]
    for i, fid in enumerate((0, 1)):
        vals = [stats[m]["per_frame"][str(fid)]["separation_d_prime"] for m in present]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=rp.FRAME_COLORS[fid],
               label=f"frame {fid}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("descriptive d'")
    ax.set_title("Two-component separation\n(reported, NOT a selection criterion)")
    ax.legend()

    fig.suptitle(f"Background methods compared — selected: "
                 f"{METHODS[rec['selected']]}")
    rp.provisional_note(
        fig, "Selection is on residual structure, drift and background coupling. "
             "d' is shown for completeness and is deliberately not used to rank: "
             "it rewards subtracting less. The legacy annulus is contaminated by "
             "neighbouring sites and is not a candidate.")
    return rp.save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paired_100ms.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        df, _sites, meta = load_dataset(cfg)
    except SourceDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    em = load_em_fit(cfg)
    stats = per_method_stats(df, meta, em)
    rec = recommend(stats)

    repo = Path(__file__).resolve().parents[1]
    fig_path = figure(stats, rec, df, repo / "assets" / "readme"
                      / "background_method_comparison.png")

    report = {
        "provenance": stamp(cfg),
        "schema_version": meta.get("schema_version"),
        "geometry_version": meta.get("geometry_version"),
        "mask": meta.get("background", {}).get("mask"),
        "surface_degree": meta.get("background", {}).get("surface_degree"),
        "methods": stats,
        "recommendation": rec,
        "configured_primary_method": str(cfg["background"]["primary_method"]),
        "configured_matches_recommendation":
            str(cfg["background"]["primary_method"]) == rec["selected"],
        "figure": (Path("assets/readme") / fig_path.name).as_posix(),
        "caveat": (
            "Methods C and D interpolate the background underneath the array "
            "from surrounding site-free pixels, because a dense array leaves "
            "no unmasked pixel at a site. The interpolation is smooth by "
            "construction and cannot represent structure finer than the mask "
            "spacing."),
    }
    out = cfg.reports_dir("validation") / "background_method_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"{'method':<22}{'resid struct':>13}{'|drift|':>10}{'|corr bg|':>11}{'d0':>7}{'d1':>7}")
    for m in METHODS:
        if m not in stats:
            continue
        s = stats[m]
        r = s.get("site_free_residual", {}).get("block_median_std_mean", float("nan"))
        dr = np.mean([abs(v["corrected_drift_per_100_shots"]) for v in s["per_frame"].values()])
        vals = [abs(v["corr_corrected_vs_background_across_shots"])
                for v in s["per_frame"].values()]
        vals = [v for v in vals if np.isfinite(v)]
        co = float(np.mean(vals)) if vals else float("nan")
        d0 = s["per_frame"]["0"]["separation_d_prime"]
        d1 = s["per_frame"]["1"]["separation_d_prime"]
        print(f"{m:<22}{r:13.3f}{dr:10.1f}{co:11.3f}{d0:7.2f}{d1:7.2f}")
    print(f"\nrecommended primary: {rec['selected']}")
    print(f"configured primary : {cfg['background']['primary_method']}"
          f"  ({'match' if report['configured_matches_recommendation'] else 'MISMATCH'})")
    print(f"written            : {out.name}, {fig_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
