"""Statistical inference on standardized neutral-atom fluorescence data.

Scope boundary
--------------
Image loading, ROI sums, local-background estimation and grid geometry are the
responsibility of the general lab analysis package (`rydlab`, from the
`mit-tweezer-array-analysis` checkout). This package consumes that machinery,
standardizes its output into a long frame-site table, and does statistics on
top. It contains no laboratory control code and no absolute data paths.

V0 preserves the paired-readout audit and descriptive validation. V1 adds
held-out sweep baselines, shot-cluster uncertainty, operational decay models
and a gated latent-state model. No result is labelled empirical readout
fidelity or intrinsic atom lifetime without the identifying controls.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .schema import (  # noqa: F401
    FRAME_SITE_COLUMNS,
    PRIMARY_KEY,
    V3_ADDITIONAL_COLUMNS,
    V3_PRIMARY_KEY,
    V3_SCHEMA_VERSION,
)

__all__ = [
    "Config",
    "load_config",
    "FRAME_SITE_COLUMNS",
    "PRIMARY_KEY",
    "V3_ADDITIONAL_COLUMNS",
    "V3_PRIMARY_KEY",
    "V3_SCHEMA_VERSION",
    "__version__",
]
