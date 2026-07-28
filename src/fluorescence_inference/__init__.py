"""Statistical inference on standardized paired-readout fluorescence data.

Scope boundary
--------------
Image loading, ROI sums, local-background estimation and grid geometry are the
responsibility of the general lab analysis package (`rydlab`, from the
`mit-tweezer-array-analysis` checkout). This package consumes that machinery,
standardizes its output into a long frame-site table, and does statistics on
top. It contains no laboratory control code and no absolute data paths.

V0 stage: audit, standardized data layer, and quality control only. No
thresholding, mixture, per-site classifier or hidden-Markov model is fitted
here, and no readout-fidelity or atom-loss quantity is estimated.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .schema import FRAME_SITE_COLUMNS, PRIMARY_KEY  # noqa: F401

__all__ = ["Config", "load_config", "FRAME_SITE_COLUMNS", "PRIMARY_KEY", "__version__"]
