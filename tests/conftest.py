"""Shared fixtures.

Synthetic fixtures run everywhere. Tests that need the real shots are marked
``real_data`` and skip cleanly when they are not present, so a fresh clone is
green without the private data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_data: needs the private raw shots and the lab package")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def scratch(request) -> Path:
    """A writable temp directory inside the repo.

    pytest's own tmp root is not writable in every environment this runs in,
    and a test that skips because of that is a test that never runs.
    """
    import shutil

    d = ROOT / "_scratch" / "tests" / request.node.name.replace("/", "_")
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def cfg():
    from fluorescence_inference.config import load_config

    return load_config("configs/paired_100ms.yaml", root=ROOT)


@pytest.fixture(scope="session")
def real_dataset(cfg):
    """The exported table, or a skip."""
    from fluorescence_inference.dataset import SourceDataError, load_dataset

    try:
        return load_dataset(cfg)
    except SourceDataError as exc:
        pytest.skip(f"processed dataset unavailable: {exc}")


@pytest.fixture(scope="session")
def rydlab_available(cfg) -> bool:
    from fluorescence_inference.config import ConfigError, ensure_rydlab_importable

    try:
        ensure_rydlab_importable(cfg)
        return True
    except (ConfigError, ImportError):
        return False


# ------------------------------------------------------------ synthetic data
@pytest.fixture
def synthetic_array():
    """A tiny two-grid synthetic run with known ground truth.

    Deliberately built like the real thing: a static fringe background that
    dominates the mean image, plus stochastic site occupancy that only shows up
    in the variance. A site finder that looks at the mean fails this fixture;
    the variance-based one passes it.
    """
    rng = np.random.default_rng(4242)
    h, w = 160, 180
    n_shots, ny, nx = 60, 4, 5
    pitch, psf = 9.0, 1.1
    truth = []
    for gi, (oy, ox) in enumerate(((40.0, 35.0), (95.0, 105.0))):
        for i in range(ny):
            for j in range(nx):
                truth.append((oy + i * pitch + 0.15 * j,
                              ox + j * pitch - 0.15 * i, gi))
    truth = np.array([(y, x) for y, x, _ in truth])

    yy, xx = np.mgrid[0:h, 0:w]
    fringe = 500.0 + 60.0 * np.sin((yy * 0.9 + xx * 1.3) / 7.0)

    frames = np.empty((n_shots, 2, h, w), dtype=np.float64)
    occupancy = rng.random((n_shots, 2, len(truth))) < 0.55
    for s in range(n_shots):
        for f in range(2):
            img = fringe - 12.0 * f + rng.normal(0.0, 8.0, (h, w))
            for k, (cy, cx) in enumerate(truth):
                if occupancy[s, f, k]:
                    img += 2400.0 * np.exp(
                        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * psf ** 2)
                    ) / (2 * np.pi * psf ** 2)
            frames[s, f] = img
    return {"frames": frames, "truth_yx": truth, "occupancy": occupancy,
            "shape": (h, w), "ny": ny, "nx": nx, "n_grids": 2, "pitch": pitch}
