"""Descriptive two-component summary of a count distribution.

**Scope, stated once and meant literally.** Everything in this module is a
*descriptive summary of an observed histogram*, produced so that the quality
report and the README figures can draw a reference level and show where the two
populations sit. It is deliberately **not** the modelling baseline:

* it is fitted on all data, with no train / validation / test split;
* it is pooled over sites and fitted separately per frame, with no per-site or
  shrinkage structure;
* it is never evaluated on held-out data.

Therefore none of its outputs may be reported as readout fidelity, as a false
positive or false negative rate, or as an atom-loss probability. The quantity
:attr:`DescriptiveFit.model_implied_overlap` is exactly what its name says: the
overlap *implied by this two-Gaussian description*, which is a property of the
description, not a measured error rate. Establishing real error rates needs the
identification controls listed in ``docs/DATA_AUDIT.md``.

The expectation-maximisation step is the lab package's own implementation, so
the numbers agree with the existing analysis rather than forking it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

#: deterministic initialisations, same ladder the lab code uses
INIT_QUANTILE_PAIRS = ((5, 95), (10, 90), (15, 85), (20, 80), (25, 75), (30, 70))
MIN_SIGMA_ABS = 60.0
MIN_SIGMA_FRAC_OF_IQR = 0.1


@dataclass(frozen=True)
class DescriptiveFit:
    """Two-Gaussian description of one pooled count distribution."""

    n: int
    weight_low: float
    weight_high: float
    mean_low: float
    mean_high: float
    sigma_low: float
    sigma_high: float
    log_likelihood: float
    separation_d_prime: float
    reference_level: float
    model_implied_overlap: float
    high_fraction_above_reference: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_interpretation"] = (
            "Descriptive pooled two-component summary. Fitted on all data with "
            "no held-out split. 'model_implied_overlap' is implied by this "
            "description and is not a measured false-positive or "
            "false-negative rate, not a readout fidelity, and not an atom-loss "
            "probability."
        )
        return d

    def density(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted component densities, for overlaying on a histogram."""
        return (self.weight_low * _normal_pdf(x, self.mean_low, self.sigma_low),
                self.weight_high * _normal_pdf(x, self.mean_high, self.sigma_high))


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def _sigma_floor(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return float(max(MIN_SIGMA_ABS, MIN_SIGMA_FRAC_OF_IQR * (q3 - q1)))


def _equal_prior_crossing(mu0: float, s0: float, mu1: float, s1: float) -> float:
    """Where two equally weighted Gaussians cross, between their means.

    Equal priors on purpose: the crossing is then a property of the two
    *shapes* alone and does not move when the occupied fraction changes, which
    keeps the reference level comparable between frames.
    """
    if abs(s0 - s1) < 1e-9:
        return 0.5 * (mu0 + mu1)
    a = 1.0 / (2 * s0 * s0) - 1.0 / (2 * s1 * s1)
    b = mu1 / (s1 * s1) - mu0 / (s0 * s0)
    c = (mu0 * mu0) / (2 * s0 * s0) - (mu1 * mu1) / (2 * s1 * s1) + np.log(s1 / s0)
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.5 * (mu0 + mu1)
    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    lo, hi = min(mu0, mu1), max(mu0, mu1)
    inside = [r for r in roots if lo <= r <= hi]
    return float(inside[0]) if inside else float(0.5 * (mu0 + mu1))


def _normal_sf(x: float, mu: float, sigma: float) -> float:
    from math import erfc, sqrt

    return 0.5 * erfc((x - mu) / (sigma * sqrt(2.0)))


def fit_descriptive(values: np.ndarray, em_fit) -> DescriptiveFit:
    """Fit the pooled two-component description.

    ``em_fit`` is the lab package's ``fit_two_gaussian_em_one_init``, injected
    so this module has no import-time dependency on that checkout.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        raise ValueError(f"need at least 50 finite values, got {x.size}")

    floor = _sigma_floor(x)
    sigma_init = max(float(np.std(x, ddof=1)), floor)
    best: dict[str, Any] | None = None
    for qlo, qhi in INIT_QUANTILE_PAIRS:
        m0, m1 = np.percentile(x, [qlo, qhi])
        if m0 == m1:
            continue
        p0 = float(np.clip(np.mean(x <= 0.5 * (m0 + m1)), 0.1, 0.9))
        fit = em_fit(x, mu0=float(m0), mu1=float(m1), sigma0=sigma_init,
                     sigma1=sigma_init, pi0=p0, pi1=1.0 - p0, sigma_floor=floor)
        if best is None or fit["log_likelihood"] > best["log_likelihood"]:
            best = fit
    if best is None:
        raise RuntimeError("no two-component initialisation converged")

    mu0, mu1 = best["mu0"], best["mu1"]
    s0, s1 = best["sigma0"], best["sigma1"]
    ref = _equal_prior_crossing(mu0, s0, mu1, s1)
    d_prime = abs(mu1 - mu0) / np.sqrt(0.5 * (s0 * s0 + s1 * s1))
    overlap = 0.5 * (_normal_sf(ref, mu0, s0) + (1.0 - _normal_sf(ref, mu1, s1)))

    return DescriptiveFit(
        n=int(x.size),
        weight_low=float(best["pi0"]), weight_high=float(best["pi1"]),
        mean_low=float(mu0), mean_high=float(mu1),
        sigma_low=float(s0), sigma_high=float(s1),
        log_likelihood=float(best["log_likelihood"]),
        separation_d_prime=float(d_prime),
        reference_level=float(ref),
        model_implied_overlap=float(overlap),
        high_fraction_above_reference=float(np.mean(x > ref)),
        converged=np.isfinite(best["log_likelihood"]),
    )


def load_em_fit(cfg) -> Any:
    """Fetch the lab package's EM routine."""
    from .config import ensure_rydlab_importable

    ensure_rydlab_importable(cfg)
    from rydlab.atoms.histograms import fit_two_gaussian_em_one_init

    return fit_two_gaussian_em_one_init
