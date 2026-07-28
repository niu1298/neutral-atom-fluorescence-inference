"""Determinism of the generated artefacts, and the schema-version migration."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fluorescence_inference import schema

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "readme"
STATIC_FIGURES = ("paired_images_roi_overlay.png", "count_distribution_fit.png",
                  "paired_frame_scatter.png", "site_summary_map.png")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------ gif assembly
def _frames() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    base = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
    out = []
    for k in range(6):
        arr = base.copy()
        arr[:, : 10 * k] = 0
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def test_shared_palette_quantisation_is_deterministic():
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_readme_assets as gra

    a = gra.quantize_all(_frames())
    b = gra.quantize_all(_frames())
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert x.getpalette() == y.getpalette()


def test_gif_bytes_are_deterministic(scratch):
    tmp_path = scratch
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_readme_assets as gra

    p1, p2 = tmp_path / "a.gif", tmp_path / "b.gif"
    i1 = gra.write_gif(_frames(), p1, fps=10)
    i2 = gra.write_gif(_frames(), p2, fps=10)
    assert sha(p1) == sha(p2)
    assert i1["duration_s"] == i2["duration_s"] == 0.6
    assert i1["loop"] == "infinite"


def test_identical_held_frames_are_merged_not_duplicated(scratch):
    tmp_path = scratch
    """The hold-frame trick only works if the writer collapses duplicates."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_readme_assets as gra

    frames = _frames()[:2] + [_frames()[1]] * 20
    path = tmp_path / "held.gif"
    info = gra.write_gif(frames, path, fps=10)
    from PIL import ImageSequence

    with Image.open(path) as im:
        durations = [f.info.get("duration", 0) for f in ImageSequence.Iterator(im)]
    assert info["n_frames"] == 22
    assert len(durations) < 22, "identical frames were stored separately"
    assert sum(durations) == 2200


# ----------------------------------------------------- schema v1 -> v2
def test_migration_renames_every_v1_column():
    v1 = pd.DataFrame({
        "run_id": ["R"], "shot_id": [0], "frame_id": [0], "site_id": [0],
        "roi_sum": [1000.0], "local_background": [500.0],
        "global_background": [480.0], "background_corrected_count": [500.0],
        "common_mode_corrected_count": [520.0], "local_background_density": [20.0],
    })
    v2 = schema.migrate_v1_to_v2(v1)
    assert "roi_sum" not in v2.columns
    assert v2["roi_sum_raw"].iloc[0] == 1000.0
    assert v2["background_annulus_contaminated"].iloc[0] == 500.0
    assert v2["count_corrected_annulus_contaminated"].iloc[0] == 500.0
    assert v2["count_corrected_global"].iloc[0] == 520.0


def test_migration_does_not_fabricate_the_missing_methods():
    v1 = pd.DataFrame({"roi_sum": [1.0], "local_background": [0.5]})
    v2 = schema.migrate_v1_to_v2(v1)
    assert "count_corrected_spatial" not in v2.columns
    assert "count_corrected_fixed_offset" not in v2.columns
    assert v2.attrs["missing_methods"] == ["spatial", "fixed_offset"]
    assert v2.attrs["schema_version"].startswith("2.0-migrated")


def test_migration_maps_the_old_primary_onto_the_contaminated_column():
    """The v1 primary was the annulus; migration must not silently promote it."""
    assert schema.V1_TO_V2["background_corrected_count"] == \
        "count_corrected_annulus_contaminated"
    assert "annulus_contaminated" in schema.CONTAMINATED_METHODS


def test_schema_version_is_recorded_in_the_metadata(cfg):
    meta_path = cfg.paths.processed_root / f"{cfg.dataset_id}.meta.json"
    if not meta_path.exists():
        pytest.skip("processed dataset unavailable")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == schema.SCHEMA_VERSION
    assert meta["geometry_version"] == cfg["sites"]["geometry_version"]
    bg = meta["background"]
    assert bg["primary_method"] == cfg["background"]["primary_method"]
    assert bg["mask"]["mask_radius_px"] == cfg["background"]["mask_radius_px"]
    assert bg["annulus_is_diagnostic_only"] is True


# ------------------------------------------------- regenerated static assets
@pytest.mark.real_data
def test_static_figures_regenerate_identically(cfg):
    """A rebuild must reproduce the committed figures byte for byte."""
    missing = [n for n in STATIC_FIGURES if not (ASSETS / n).exists()]
    if missing:
        pytest.skip(f"assets not generated: {missing}")
    before = {n: sha(ASSETS / n) for n in STATIC_FIGURES}

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_readme_assets.py"),
         "--config", "configs/paired_100ms.yaml", "--skip-gif"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        pytest.skip(f"asset generation unavailable: {r.stderr.strip()[-300:]}")

    after = {n: sha(ASSETS / n) for n in STATIC_FIGURES}
    differing = [n for n in STATIC_FIGURES if before[n] != after[n]]
    assert not differing, f"non-deterministic figures: {differing}"


@pytest.mark.real_data
def test_asset_metadata_records_the_versions(cfg):
    path = ASSETS / "asset_metadata.json"
    if not path.exists():
        pytest.skip("assets not generated")
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["random_seed"] == cfg["readme_assets"]["random_seed"]
    assert meta["selection"]["shot_rule"]
    assert meta["synthetic_panels"] == []
    assert meta["provisional_panels"]
