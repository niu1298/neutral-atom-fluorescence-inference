"""Image-level and table-level quality control.

Everything here is descriptive. No occupancy is claimed, no fidelity is
estimated, and a disagreement between the two frames is reported as a
disagreement, never as observed atom loss.

Outputs
-------
``reports/qc/qc_summary.json``   machine-readable summary
``reports/qc/qc_*.png``          figures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import reporting as rp
from .config import Config
from .descriptive_fit import DescriptiveFit, fit_descriptive, load_em_fit
from .provenance import stamp

#: the three measurement variants carried through the table
VARIANTS = {
    "A_raw_roi_sum": "roi_sum",
    "B_local_background_corrected": "background_corrected_count",
    "C_common_mode_corrected": "common_mode_corrected_count",
}
VARIANT_LABEL = {
    "A_raw_roi_sum": "A  raw ROI sum",
    "B_local_background_corrected": "B  ROI sum - local background",
    "C_common_mode_corrected": "C  ROI sum - site-free common mode",
}


# --------------------------------------------------------------- statistics
def bootstrap_by_shot(df: pd.DataFrame, statistic, *, n_boot: int, seed: int
                      ) -> tuple[float, float, float]:
    """Cluster bootstrap over shots: ``(point, lo95, hi95)``.

    Resampling shots, not rows: the 100 shots are the independent experimental
    units, the site-level rows within a shot are not.
    """
    point = float(statistic(df))
    shots = df["shot_id"].unique()
    rng = np.random.default_rng(seed)
    by_shot = {s: g for s, g in df.groupby("shot_id", observed=True)}
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(shots, size=shots.size, replace=True)
        draws[b] = statistic(pd.concat([by_shot[s] for s in pick]))
    lo, hi = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """One row per (shot, site) with the two frame values side by side."""
    wide = df.pivot_table(index=["shot_id", "site_id"], columns="frame_id",
                          values=value_col, observed=True)
    wide.columns = [f"f{int(c)}" for c in wide.columns]
    flags = (df.assign(_bad=df["quality_flag"].astype(str) != "ok")
               .groupby(["shot_id", "site_id"], observed=True)["_bad"].any())
    out = wide.join(flags.rename("any_flag")).reset_index()
    return out.dropna(subset=["f0", "f1"])


def _agreement_stats(pairs: pd.DataFrame, ref0: float, ref1: float) -> dict[str, float]:
    """Counts of the four apparent regions, relative to descriptive references.

    Naming is deliberate. ``apparent_bright_to_dark`` is a statement about two
    measurements, not about an atom: with a single imaging condition and no
    matched-empty or natural-loss control, this run cannot separate a lost atom
    from a misclassified frame.
    """
    hi0 = pairs["f0"].to_numpy() > ref0
    hi1 = pairs["f1"].to_numpy() > ref1
    n = int(hi0.size)
    return {
        "n_pairs": n,
        "low_low": int(np.sum(~hi0 & ~hi1)),
        "high_high": int(np.sum(hi0 & hi1)),
        "high_low": int(np.sum(hi0 & ~hi1)),
        "low_high": int(np.sum(~hi0 & hi1)),
        "agreement": float(np.mean(hi0 == hi1)) if n else float("nan"),
        "apparent_bright_to_dark_rate": (float(np.sum(hi0 & ~hi1) / hi0.sum())
                                         if hi0.sum() else float("nan")),
        "apparent_dark_to_bright_rate": (float(np.sum(~hi0 & hi1) / (~hi0).sum())
                                         if (~hi0).sum() else float("nan")),
        "high_fraction_frame0": float(np.mean(hi0)) if n else float("nan"),
        "high_fraction_frame1": float(np.mean(hi1)) if n else float("nan"),
    }


def descriptive_fits(cfg: Config, df: pd.DataFrame, value_col: str
                     ) -> dict[int, DescriptiveFit]:
    """Per-frame pooled descriptive two-component summary. Display use only."""
    em = load_em_fit(cfg)
    clean = df[df["quality_flag"].astype(str) == "ok"]
    return {int(fid): fit_descriptive(g[value_col].to_numpy(float), em)
            for fid, g in clean.groupby("frame_id", observed=True)}


def site_table(df: pd.DataFrame, sites_df: pd.DataFrame, value_col: str
               ) -> pd.DataFrame:
    """Per-site, per-frame descriptive statistics joined onto site geometry."""
    g = df.groupby(["site_id", "frame_id"], observed=True)
    stats = g.agg(
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        p10=(value_col, lambda s: float(np.percentile(s, 10))),
        p90=(value_col, lambda s: float(np.percentile(s, 90))),
        local_background=("local_background", "mean"),
        n_obs=(value_col, "size"),
        n_flagged=("quality_flag", lambda s: int((s.astype(str) != "ok").sum())),
    ).reset_index()
    stats["interdecile_spread"] = stats["p90"] - stats["p10"]
    return stats.merge(sites_df, on="site_id", how="left")


# ------------------------------------------------------------------ figures
def _crop(site_map, pad: int = 10) -> tuple[int, int, int, int]:
    """Tight crop around the sites themselves, as ``(x0, x1, y0, y1)``.

    Cropping to the sites rather than to the whole sensor is also the privacy
    measure: the published frames show the array and nothing else.
    """
    c = site_map.centers_yx
    return (int(np.floor(c[:, 1].min())) - pad, int(np.ceil(c[:, 1].max())) + pad + 1,
            int(np.floor(c[:, 0].min())) - pad, int(np.ceil(c[:, 0].max())) + pad + 1)


def fig_roi_overlay(cfg: Config, ctx: dict[str, Any], shot_index: int,
                    out: Path) -> Path:
    """Both frames of one shot, identical crop and scale, ROI boxes overlaid."""
    from .dataset import _frame_loader

    load, _ = _frame_loader(cfg)
    metas, site_map = ctx["metas"], ctx["site_map"]
    m = metas[shot_index]
    x0, x1, y0, y1 = _crop(site_map)
    exposure = cfg["source"]["expected_exposure_s"] * 1e3

    imgs = [load(m.path, s["h5_path"])[y0:y1, x0:x1] for s in cfg.frame_specs]
    lo, hi = np.percentile(np.concatenate([i.ravel() for i in imgs]), (1, 99.6))

    fig, axes = rp.plt.subplots(1, 2, figsize=(11.8, 5.4), layout="constrained")
    for ax, img, spec in zip(axes, imgs, cfg.frame_specs):
        fid = int(spec["frame_id"])
        h = rp.show_image(ax, img, vlim=(lo, hi),
                          extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        boxes = [(b[0], b[1], b[2], b[3]) for b in site_map.boxes]
        rp.draw_roi_boxes(ax, boxes, color="#7fe3ff", lw=0.6)
        ax.set_title(f"frame {fid} - {exposure:.0f} ms exposure", color=rp.INK)
        ax.set_xlabel("camera column (px)")
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y1 - 0.5, y0 - 0.5)
        ax.grid(False)
        if fid == 0:
            ax.set_ylabel("camera row (px)")
        rp.colorbar(fig, h, ax, "camera counts / pixel")
    fig.suptitle(
        f"Consecutive fluorescence frames with site ROIs "
        f"({site_map.n_sites} sites, {2 * cfg['roi']['trap_half_width'] + 1}"
        f"x{2 * cfg['roi']['trap_half_width'] + 1} px each)", fontsize=12.5)
    rp.provisional_note(fig, "Single shot, single 100 ms imaging condition. "
                             "Same crop and same intensity scale in both panels.")
    return rp.save(fig, out)


def fig_count_histograms(cfg: Config, df: pd.DataFrame,
                         fits: dict[str, dict[int, DescriptiveFit]],
                         out: Path) -> Path:
    """The three measurement variants, per frame."""
    bins = int(cfg["qc"]["hist_bins"])
    clean = df[df["quality_flag"].astype(str) == "ok"]
    fig, axes = rp.plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
    for ax, (key, col) in zip(axes, VARIANTS.items()):
        lo, hi = np.percentile(clean[col], (0.2, 99.8))
        edges = np.linspace(lo, hi, bins + 1)
        for fid, g in clean.groupby("frame_id", observed=True):
            ax.hist(g[col], bins=edges, histtype="step", linewidth=1.7,
                    color=rp.FRAME_COLORS[int(fid)], label=f"frame {int(fid)}")
        f = fits.get(key)
        if f:
            for fid, fit in f.items():
                ax.axvline(fit.reference_level, color=rp.FRAME_COLORS[int(fid)],
                           ls="--", lw=1.1, alpha=0.8)
        ax.set_title(VARIANT_LABEL[key])
        ax.set_xlabel("counts")
        ax.set_yscale("log")
        if key == "A_raw_roi_sum":
            ax.set_ylabel("site-frame observations")
            ax.legend()
    fig.suptitle("Measurement variants: pooled site-frame count distributions")
    rp.provisional_note(
        fig, "Dashed lines: descriptive two-component reference level, display only "
             "- not a fitted decision rule and not a fidelity estimate.")
    return rp.save(fig, out)


def fig_paired_scatter(cfg: Config, pairs: pd.DataFrame, refs: tuple[float, float],
                       n_shots: int, out: Path) -> Path:
    ok = pairs[~pairs["any_flag"]]
    fig, ax = rp.plt.subplots(figsize=(6.8, 6.1), layout="constrained")
    lo = float(min(ok["f0"].min(), ok["f1"].min()))
    hi = float(max(np.percentile(ok["f0"], 99.9), np.percentile(ok["f1"], 99.9)))
    hb = ax.hexbin(ok["f0"], ok["f1"], gridsize=60, bins="log",
                   cmap="magma", extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], color=rp.MUTED, lw=0.9, ls=":")
    ax.axvline(refs[0], color=rp.ACCENT, ls="--", lw=1.1)
    ax.axhline(refs[1], color=rp.ACCENT_2, ls="--", lw=1.1)
    ax.set_xlabel("frame 0 background-corrected ROI count")
    ax.set_ylabel("frame 1 background-corrected ROI count")
    ax.set_title("Paired readout: frame 0 vs frame 1")
    rp.colorbar(fig, hb, ax, "site-shot pairs (log)")
    rp.provisional_note(
        fig, f"{len(ok):,} site-shot pairs from {n_shots} shots. Site-level pairs "
             f"are not independent experimental units. Dashed lines are "
             f"descriptive reference levels, not fitted thresholds.")
    return rp.save(fig, out)


def fig_background_drift(cfg: Config, df: pd.DataFrame, ctx: dict[str, Any],
                         out: Path) -> Path:
    per = (df.groupby(["shot_order", "frame_id"], observed=True)
             .agg(global_bg=("global_background", "mean"),
                  local_bg=("local_background", "mean"),
                  roi_sum=("roi_sum", "mean")).reset_index())
    n_px = int(df["roi_n_pixels"].mode().iat[0])

    fig, axes = rp.plt.subplots(1, 3, figsize=(15.5, 4.4), layout="constrained")
    for fid, g in per.groupby("frame_id", observed=True):
        c = rp.FRAME_COLORS[int(fid)]
        axes[0].plot(g["shot_order"], g["global_bg"] / n_px, ".-", ms=3.5, lw=0.9,
                     color=c, label=f"frame {int(fid)}")
        axes[1].plot(g["shot_order"], g["local_bg"] / n_px, ".-", ms=3.5, lw=0.9,
                     color=c, label=f"frame {int(fid)}")
    axes[0].set_title("Site-free common-mode level")
    axes[1].set_title("Mean local (annulus) background")
    for ax in axes[:2]:
        ax.set_xlabel("shot order")
        ax.set_ylabel("camera counts / pixel")
        ax.legend()

    w = per.pivot(index="shot_order", columns="frame_id", values="global_bg")
    if {0, 1} <= set(w.columns):
        d = (w[1] - w[0]) / n_px
        axes[2].plot(d.index, d.to_numpy(), ".-", ms=3.5, lw=0.9, color=rp.WARN)
        axes[2].axhline(0.0, color=rp.MUTED, lw=0.9, ls=":")
        axes[2].axhline(float(d.mean()), color=rp.INK, lw=1.0, ls="--",
                        label=f"mean {d.mean():+.2f}")
        axes[2].legend()
    axes[2].set_title("Frame 1 minus frame 0 common mode")
    axes[2].set_xlabel("shot order")
    axes[2].set_ylabel("counts / pixel")
    fig.suptitle("Background level against acquisition order")
    rp.provisional_note(fig, "Common mode measured on site-free pixels of the same "
                             "frame; never a median over the sites themselves.")
    return rp.save(fig, out)


def fig_local_background_by_frame(df: pd.DataFrame, out: Path) -> Path:
    n_px = int(df["roi_n_pixels"].mode().iat[0])
    fig, axes = rp.plt.subplots(1, 2, figsize=(11.4, 4.4), layout="constrained")
    for fid, g in df.groupby("frame_id", observed=True):
        axes[0].hist(g["local_background"] / n_px, bins=90, histtype="step",
                     lw=1.7, color=rp.FRAME_COLORS[int(fid)], label=f"frame {int(fid)}")
    axes[0].set_xlabel("local background (counts / pixel)")
    axes[0].set_ylabel("site-frame observations")
    axes[0].set_title("Local background by frame")
    axes[0].legend()

    w = df.pivot_table(index=["shot_id", "site_id"], columns="frame_id",
                       values="local_background", observed=True)
    if {0, 1} <= set(w.columns):
        d = (w[1] - w[0]) / n_px
        axes[1].hist(d, bins=90, histtype="stepfilled", color=rp.WARN, alpha=0.35)
        axes[1].hist(d, bins=90, histtype="step", lw=1.6, color=rp.WARN)
        axes[1].axvline(0.0, color=rp.MUTED, ls=":", lw=1.0)
        axes[1].axvline(float(d.mean()), color=rp.INK, ls="--", lw=1.1,
                        label=f"mean {d.mean():+.2f} counts/px")
        axes[1].legend()
    axes[1].set_xlabel("frame 1 minus frame 0 (counts / pixel)")
    axes[1].set_ylabel("site-shot pairs")
    axes[1].set_title("Per-site frame-to-frame background shift")
    rp.provisional_note(fig, "A systematic offset here is a frame-dependent "
                             "background, not evidence about atoms.")
    return rp.save(fig, out)


def fig_site_maps(sites_stats: pd.DataFrame, out: Path) -> Path:
    f0 = sites_stats[sites_stats["frame_id"] == 0]
    panels = [
        ("mean", "mean background-corrected count", rp.SEQ_CMAP),
        ("std", "shot-to-shot std (counts)", "viridis"),
        ("local_background", "mean local background (counts)", "cividis"),
        ("interdecile_spread", "P90 - P10 (counts)", "magma"),
    ]
    fig, axes = rp.plt.subplots(1, 4, figsize=(21.0, 5.0), layout="constrained")
    for ax, (col, label, cmap) in zip(axes, panels):
        s = ax.scatter(f0["site_x"], f0["site_y"], c=f0[col], s=42, marker="s",
                       cmap=cmap, edgecolors="none")
        miss = f0[~f0["site_detected"].astype(bool)]
        if len(miss):
            ax.scatter(miss["site_x"], miss["site_y"], s=90, marker="s",
                       facecolors="none", edgecolors=rp.WARN, linewidths=1.4,
                       label="no variance peak")
            ax.legend(loc="lower left", fontsize=8.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("camera column (px)")
        ax.grid(False)
        rp.colorbar(fig, s, ax, "camera counts")
    axes[0].set_ylabel("camera row (px)")
    fig.suptitle("Site-level descriptive maps, frame 0")
    rp.provisional_note(fig, "Sites are localised from the shot-to-shot variance "
                             "map; a trap never loaded in these 100 shots cannot "
                             "appear here.")
    return rp.save(fig, out)


def fig_variant_comparison(df: pd.DataFrame, summary: dict[str, Any],
                           out: Path) -> Path:
    fig, axes = rp.plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    keys = list(VARIANTS)
    x = np.arange(len(keys))
    for i, fid in enumerate((0, 1)):
        vals = [summary["variants"][k]["per_frame"][str(fid)]["separation_d_prime"]
                for k in keys]
        axes[0].bar(x + (i - 0.5) * 0.36, vals, width=0.34,
                    color=rp.FRAME_COLORS[fid], label=f"frame {fid}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([VARIANT_LABEL[k].split("  ")[0] for k in keys])
    axes[0].set_ylabel("descriptive separation d'")
    axes[0].set_title("Two-component separation by variant")
    axes[0].legend()

    for i, fid in enumerate((0, 1)):
        vals = [summary["variants"][k]["per_frame"][str(fid)]["shot_order_slope_per_100_shots"]
                for k in keys]
        axes[1].bar(x + (i - 0.5) * 0.36, vals, width=0.34,
                    color=rp.FRAME_COLORS[fid], label=f"frame {fid}")
    axes[1].axhline(0.0, color=rp.MUTED, lw=0.9, ls=":")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([VARIANT_LABEL[k].split("  ")[0] for k in keys])
    axes[1].set_ylabel("drift over the run (counts / 100 shots)")
    axes[1].set_title("Residual drift with acquisition order")
    axes[1].legend()
    fig.suptitle("Which background treatment is more stable")
    rp.provisional_note(fig, "d' describes how far apart the two fitted components "
                             "sit; it is not a fidelity and implies no error rate.")
    return rp.save(fig, out)


def fig_outliers(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = rp.plt.subplots(1, 2, figsize=(11.6, 4.4), layout="constrained")
    for fid, g in df.groupby("frame_id", observed=True):
        axes[0].hist(g["roi_max_pixel"], bins=90, histtype="step", lw=1.7,
                     color=rp.FRAME_COLORS[int(fid)], label=f"frame {int(fid)}")
    axes[0].set_xlabel("brightest pixel inside an ROI (counts)")
    axes[0].set_ylabel("site-frame observations")
    axes[0].set_title("Headroom to the 16-bit ceiling")
    axes[0].legend()

    per_shot = (df.groupby(["shot_order", "frame_id"], observed=True)["roi_sum"]
                  .mean().reset_index())
    for fid, g in per_shot.groupby("frame_id", observed=True):
        axes[1].plot(g["shot_order"], g["roi_sum"], ".-", ms=3.5, lw=0.9,
                     color=rp.FRAME_COLORS[int(fid)], label=f"frame {int(fid)}")
    axes[1].set_xlabel("shot order")
    axes[1].set_ylabel("mean raw ROI sum (counts)")
    axes[1].set_title("Whole-array brightness against acquisition order")
    axes[1].legend()
    rp.provisional_note(fig, "Array brightness mixes loading fraction with imaging "
                             "conditions; it is a monitor, not a measurement.")
    return rp.save(fig, out)


# ---------------------------------------------------------------- the driver
def run_quality_control(cfg: Config, df: pd.DataFrame, sites_df: pd.DataFrame,
                        ctx: dict[str, Any]) -> dict[str, Any]:
    """Produce every QC figure and the machine-readable summary."""
    rp.apply_style()
    out_dir = cfg.reports_dir("qc")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["qc"]["random_seed"])

    fits: dict[str, dict[int, DescriptiveFit]] = {
        key: descriptive_fits(cfg, df, col) for key, col in VARIANTS.items()
    }
    primary = "B_local_background_corrected"
    refs = (fits[primary][0].reference_level, fits[primary][1].reference_level)

    pairs = paired_table(df, VARIANTS[primary])
    clean_pairs = pairs[~pairs["any_flag"]]
    agreement = _agreement_stats(clean_pairs, *refs)

    def _agree(sub: pd.DataFrame) -> float:
        return float(np.mean((sub["f0"] > refs[0]) == (sub["f1"] > refs[1])))

    a_pt, a_lo, a_hi = bootstrap_by_shot(clean_pairs, _agree, n_boot=2000, seed=seed)
    agreement["agreement_ci95"] = [a_lo, a_hi]
    agreement["agreement_bootstrap_point"] = a_pt

    stats_tbl = site_table(df, sites_df, VARIANTS[primary])

    summary: dict[str, Any] = {
        "provenance": stamp(cfg),
        "dataset": {
            "run_id": cfg.run_id,
            "n_shots": int(df["shot_id"].nunique()),
            "n_frames_per_shot": int(df["frame_id"].nunique()),
            "n_sites": int(df["site_id"].nunique()),
            "n_site_frame_observations": int(len(df)),
            "n_site_shot_pairs": int(len(pairs)),
            "exposure_ms": float(pd.to_numeric(df["exposure_ms"]).dropna().unique()[0]),
            "roi_pixels": int(df["roi_n_pixels"].mode().iat[0]),
        },
        "quality_flags": _flag_counts(df),
        "saturation": {
            "n_rows_with_saturated_roi": int(df["quality_flag"].astype(str)
                                             .str.contains("roi_saturated").sum()),
            "max_roi_pixel_seen": float(df["roi_max_pixel"].max()),
            "adc_ceiling": float(cfg["source"]["saturation_adu"]),
        },
        "sites": {
            "n_sites": int(len(sites_df)),
            "n_detected": int(sites_df["site_detected"].sum()),
            "n_without_variance_peak": int((~sites_df["site_detected"].astype(bool)).sum()),
            "fit_residual_px_median": float(sites_df["fit_residual_px"].median()),
            "fit_residual_px_p95": float(sites_df["fit_residual_px"].quantile(0.95)),
            "geometry": ctx["meta"]["sites"],
        },
        "variants": {key: _variant_block(df, col, fits[key])
                     for key, col in VARIANTS.items()},
        "paired_readout": agreement,
        "background": _background_block(df),
        "site_heterogeneity": _heterogeneity_block(stats_tbl),
        "interpretation_guard": {
            "descriptive_only": True,
            "notes": [
                "Reference levels come from a pooled two-component description "
                "fitted on all data. They are display references, not a "
                "validated decision rule.",
                "Frame-to-frame disagreement is reported as disagreement. This "
                "run has no matched-empty, dark-frame or natural-loss control, "
                "so it cannot separate atom loss from misclassification.",
                "No false-positive rate, false-negative rate, readout fidelity "
                "or imaging-loss rate is estimated anywhere in this report.",
            ],
        },
    }

    figures: list[str] = []
    rng = np.random.default_rng(seed)
    n_ex = int(cfg["qc"]["n_example_shots"])
    picks = sorted(rng.choice(len(ctx["metas"]), size=min(n_ex, len(ctx["metas"])),
                              replace=False).tolist())
    for k in picks:
        figures.append(str(fig_roi_overlay(
            cfg, ctx, k, out_dir / f"qc_01_roi_overlay_shot{k:03d}.png").name))
    figures.append(fig_count_histograms(
        cfg, df, fits, out_dir / "qc_02_count_histograms.png").name)
    figures.append(fig_paired_scatter(
        cfg, pairs, refs, summary["dataset"]["n_shots"],
        out_dir / "qc_03_paired_scatter.png").name)
    figures.append(fig_background_drift(
        cfg, df, ctx, out_dir / "qc_04_background_drift.png").name)
    figures.append(fig_local_background_by_frame(
        df, out_dir / "qc_05_local_background_by_frame.png").name)
    figures.append(fig_site_maps(stats_tbl, out_dir / "qc_06_site_maps.png").name)
    figures.append(fig_variant_comparison(
        df, summary, out_dir / "qc_07_variant_comparison.png").name)
    figures.append(fig_outliers(df, out_dir / "qc_08_outliers.png").name)

    summary["figures"] = figures
    summary["example_shot_orders"] = picks
    summary_path = out_dir / "qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable),
                            encoding="utf-8")
    stats_tbl.to_csv(out_dir / "site_statistics.csv", index=False)
    return {"summary": summary, "summary_path": str(summary_path), "figures": figures,
            "site_statistics": stats_tbl, "fits": fits, "pairs": pairs,
            "reference_levels": refs}


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, DescriptiveFit):
        return o.to_dict()
    return str(o)


def _flag_counts(df: pd.DataFrame) -> dict[str, int]:
    exploded = df["quality_flag"].astype(str).str.split("|").explode()
    return {k: int(v) for k, v in exploded.value_counts().items()}


def _variant_block(df: pd.DataFrame, col: str,
                   fits: dict[int, DescriptiveFit]) -> dict[str, Any]:
    clean = df[df["quality_flag"].astype(str) == "ok"]
    per_frame: dict[str, Any] = {}
    for fid, g in clean.groupby("frame_id", observed=True):
        fit = fits[int(fid)]
        per_shot = g.groupby("shot_order", observed=True)[col].mean()
        slope = float(np.polyfit(per_shot.index.to_numpy(float),
                                 per_shot.to_numpy(float), 1)[0] * 100.0)
        per_frame[str(int(fid))] = {
            "n": int(len(g)),
            "mean": float(g[col].mean()),
            "median": float(g[col].median()),
            "std": float(g[col].std(ddof=1)),
            "separation_d_prime": fit.separation_d_prime,
            "reference_level": fit.reference_level,
            "component_mean_low": fit.mean_low,
            "component_mean_high": fit.mean_high,
            "component_sigma_low": fit.sigma_low,
            "component_sigma_high": fit.sigma_high,
            "model_implied_overlap": fit.model_implied_overlap,
            "shot_order_slope_per_100_shots": slope,
        }
    return {"column": col, "per_frame": per_frame}


def _background_block(df: pd.DataFrame) -> dict[str, Any]:
    n_px = int(df["roi_n_pixels"].mode().iat[0])
    out: dict[str, Any] = {"roi_pixels": n_px}
    for name, col in (("local", "local_background"), ("common_mode", "global_background")):
        w = df.pivot_table(index=["shot_id", "site_id"], columns="frame_id",
                           values=col, observed=True)
        block = {f"frame{int(c)}_mean_per_px": float(w[c].mean() / n_px)
                 for c in w.columns}
        if {0, 1} <= set(w.columns):
            d = (w[1] - w[0]) / n_px
            block["frame1_minus_frame0_per_px"] = {
                "mean": float(d.mean()), "std": float(d.std(ddof=1)),
                "median": float(d.median()),
            }
        out[name] = block
    return out


def _heterogeneity_block(stats_tbl: pd.DataFrame) -> dict[str, Any]:
    f0 = stats_tbl[stats_tbl["frame_id"] == 0]
    out: dict[str, Any] = {}
    for col in ("mean", "std", "local_background", "interdecile_spread"):
        v = f0[col].to_numpy(float)
        out[col] = {
            "min": float(np.nanmin(v)), "p25": float(np.nanpercentile(v, 25)),
            "median": float(np.nanmedian(v)), "p75": float(np.nanpercentile(v, 75)),
            "max": float(np.nanmax(v)),
            "coefficient_of_variation": float(np.nanstd(v) / abs(np.nanmean(v))),
        }
    by_grid = (f0.groupby("grid", observed=True)["mean"]
                 .agg(["count", "mean", "std"]).reset_index())
    out["by_grid"] = by_grid.to_dict(orient="records")
    return out
