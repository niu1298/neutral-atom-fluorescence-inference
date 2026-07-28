"""QC statistics: the parts that must be right independently of the data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.descriptive_fit import (
    _equal_prior_crossing, fit_descriptive,
)
from fluorescence_inference.quality_control import (
    _agreement_stats, bootstrap_by_shot, paired_table,
)


def _em(x, mu0, mu1, sigma0, sigma1, pi0, pi1, sigma_floor, max_iter=600,
        tol=1e-7):
    """Standalone two-Gaussian EM with the same contract as the lab routine.

    `test_reuse_matches_lab.py` pins the real pipeline to the lab version; this
    keeps the statistics tests runnable without that checkout.
    """
    x = np.asarray(x, dtype=float)
    prev = -np.inf
    for _ in range(max_iter):
        def lp(pi, mu, s):
            z = (x - mu) / s
            return np.log(pi + 1e-15) - 0.5 * z * z - np.log(s * np.sqrt(2 * np.pi))
        l0, l1 = lp(pi0, mu0, sigma0), lp(pi1, mu1, sigma1)
        m = np.maximum(l0, l1)
        den = m + np.log(np.exp(l0 - m) + np.exp(l1 - m))
        r0, r1 = np.exp(l0 - den), np.exp(l1 - den)
        n0, n1 = r0.sum(), r1.sum()
        pi0 = float(np.clip(n0 / x.size, 1e-4, 1 - 1e-4))
        pi1 = 1.0 - pi0
        mu0 = float((r0 * x).sum() / max(n0, 1e-12))
        mu1 = float((r1 * x).sum() / max(n1, 1e-12))
        sigma0 = max(float(np.sqrt((r0 * (x - mu0) ** 2).sum() / max(n0, 1e-12))),
                     sigma_floor)
        sigma1 = max(float(np.sqrt((r1 * (x - mu1) ** 2).sum() / max(n1, 1e-12))),
                     sigma_floor)
        ll = float(den.sum())
        if abs(ll - prev) < tol * (1 + abs(prev)):
            break
        prev = ll
    if mu0 > mu1:
        mu0, mu1, sigma0, sigma1, pi0, pi1 = mu1, mu0, sigma1, sigma0, pi1, pi0
    return {"pi0": pi0, "pi1": pi1, "mu0": mu0, "mu1": mu1, "sigma0": sigma0,
            "sigma1": sigma1, "log_likelihood": prev}


# ------------------------------------------------------------------ pairing
def _long(n_shots=5, n_sites=3):
    rows = []
    for s in range(n_shots):
        for f in (0, 1):
            for k in range(n_sites):
                rows.append({"shot_id": s, "frame_id": f, "site_id": k,
                             "v": 100 * s + 10 * f + k,
                             "quality_flag": "ok" if k else "site_not_detected"})
    return pd.DataFrame(rows)


def test_paired_table_aligns_the_two_frames():
    p = paired_table(_long(), "v")
    assert len(p) == 15
    row = p[(p["shot_id"] == 3) & (p["site_id"] == 2)].iloc[0]
    assert row["f0"] == 302 and row["f1"] == 312


def test_paired_table_propagates_flags_to_the_pair():
    p = paired_table(_long(), "v")
    assert p[p["site_id"] == 0]["any_flag"].all()
    assert not p[p["site_id"] == 1]["any_flag"].any()


# --------------------------------------------------------------- agreement
def test_agreement_quadrants_are_exhaustive_and_correctly_named():
    p = pd.DataFrame({"f0": [10, 10, -10, -10, 10], "f1": [10, -10, 10, -10, 10]})
    s = _agreement_stats(p, 0.0, 0.0)
    assert s["high_high"] == 2 and s["low_low"] == 1
    assert s["high_low"] == 1 and s["low_high"] == 1
    assert s["low_low"] + s["high_high"] + s["high_low"] + s["low_high"] == s["n_pairs"]
    assert s["agreement"] == pytest.approx(3 / 5)
    assert s["apparent_bright_to_dark_rate"] == pytest.approx(1 / 3)
    assert s["apparent_dark_to_bright_rate"] == pytest.approx(1 / 2)


def test_agreement_uses_a_per_frame_reference():
    """A frame-dependent offset must not masquerade as disagreement."""
    p = pd.DataFrame({"f0": [100.0, 100.0], "f1": [90.0, 90.0]})
    shared = _agreement_stats(p, 95.0, 95.0)
    per_frame = _agreement_stats(p, 95.0, 85.0)
    assert shared["agreement"] == 0.0
    assert per_frame["agreement"] == 1.0


# --------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_shots_not_rows():
    """With a between-shot effect, a row bootstrap would understate the width."""
    rng = np.random.default_rng(0)
    frames = []
    for s in range(40):
        level = rng.normal(0.0, 1.0)
        frames.append(pd.DataFrame({"shot_id": s,
                                    "x": level + rng.normal(0, 0.05, 50)}))
    df = pd.concat(frames, ignore_index=True)
    stat = lambda d: float(d["x"].mean())  # noqa: E731
    pt, lo, hi = bootstrap_by_shot(df, stat, n_boot=400, seed=1)
    assert pt == pytest.approx(df["x"].mean())
    assert lo < pt < hi
    naive = df["x"].std(ddof=1) / np.sqrt(len(df))
    assert (hi - lo) > 6 * naive, "cluster CI must be far wider than a row CI"


def test_bootstrap_is_reproducible():
    df = pd.DataFrame({"shot_id": np.repeat(np.arange(20), 10),
                       "x": np.arange(200, dtype=float)})
    stat = lambda d: float(d["x"].mean())  # noqa: E731
    a = bootstrap_by_shot(df, stat, n_boot=200, seed=7)
    b = bootstrap_by_shot(df, stat, n_boot=200, seed=7)
    assert a == b


# --------------------------------------------------------- descriptive fit
def test_descriptive_fit_recovers_known_components():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0.0, 300.0, 6000),
                        rng.normal(3000.0, 500.0, 6000)])
    fit = fit_descriptive(x, _em)
    assert fit.mean_low == pytest.approx(0.0, abs=120)
    assert fit.mean_high == pytest.approx(3000.0, abs=150)
    assert 0.0 < fit.reference_level < 3000.0
    assert fit.separation_d_prime > 6.0
    assert fit.model_implied_overlap < 0.01


def test_descriptive_fit_is_deterministic():
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(0, 400, 3000), rng.normal(2500, 600, 3000)])
    assert fit_descriptive(x, _em).to_dict() == fit_descriptive(x, _em).to_dict()


def test_descriptive_fit_refuses_a_tiny_sample():
    with pytest.raises(ValueError, match="at least 50"):
        fit_descriptive(np.arange(10.0), _em)


def test_equal_prior_crossing_sits_between_the_means():
    assert _equal_prior_crossing(0, 1, 10, 1) == pytest.approx(5.0)
    t = _equal_prior_crossing(0.0, 1.0, 10.0, 3.0)
    assert 0.0 < t < 10.0


def test_descriptive_output_carries_its_own_disclaimer():
    rng = np.random.default_rng(8)
    x = np.concatenate([rng.normal(0, 300, 3000), rng.normal(2800, 500, 3000)])
    note = fit_descriptive(x, _em).to_dict()["_interpretation"]
    for phrase in ("not a measured false-positive", "readout fidelity",
                   "atom-loss probability", "no held-out split"):
        assert phrase in note
