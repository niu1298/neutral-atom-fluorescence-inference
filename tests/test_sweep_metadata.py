"""Raw HDF5 timing, frame, condition, and leakage guards for 0044/0050."""
from __future__ import annotations

import numpy as np
import pytest

from fluorescence_inference.config import load_config
from fluorescence_inference.dataset import (
    SourceDataError,
    _shot_design,
    _v3_row_fields,
    _validate_shot_metadata,
    _varying_globals,
    discover_shots,
    read_shot_meta,
    select_fit_metas,
)

pytestmark = pytest.mark.real_data


def _load(repo_root, config_name):
    cfg = load_config(config_name, root=repo_root)
    try:
        shots = discover_shots(cfg)
    except SourceDataError as exc:
        pytest.skip(f"raw sweep shots unavailable: {exc}")
    metas = [read_shot_meta(path, cfg, order) for order, path in enumerate(shots)]
    _validate_shot_metadata(cfg, metas)
    return cfg, metas


@pytest.mark.parametrize(
    ("config_name", "n_shots", "frame_names", "sweep_global"),
    [
        (
            "configs/dark_hold_50ms_20260728_0044.yaml",
            110,
            ["fluor", "fluor2", "fluor3", "fluor4", "fluor5"],
            "SECOND_HAMAMATSU_FLUOR_DELAY",
        ),
        (
            "configs/bright_wait_50ms_20260728_0050.yaml",
            100,
            ["fluor", "fluor2"],
            "WAIT_BEFORE_FIRST_FLUOR",
        ),
    ],
)
def test_exact_frame_and_condition_mapping(
    repo_root, config_name, n_shots, frame_names, sweep_global
):
    cfg, metas = _load(repo_root, config_name)
    assert len(metas) == n_shots
    assert all(
        [meta.exposure_names[i] for i in range(len(frame_names))] == frame_names
        for meta in metas
    )
    assert all(set(meta.exposure_ms.values()) == {50.0} for meta in metas)
    assert _varying_globals(metas) == {sweep_global: len(cfg["sweep"]["values_s"])}

    manifest, split_meta, _ = _shot_design(cfg, metas)
    assert manifest is not None
    assert split_meta["cycle_sets"]["train"] == [0, 1, 2, 3, 4, 5]
    per = manifest.groupby(["split", "condition_id"], observed=True).size()
    assert set(per.loc["train"]) == {6}
    assert set(per.loc["validation"]) == {2}
    assert set(per.loc["test"]) == {2}


def test_0044_commanded_frame_timing_is_exact(repo_root):
    cfg, metas = _load(repo_root, "configs/dark_hold_50ms_20260728_0044.yaml")
    base = 0.6842116770321879
    for meta in metas:
        hold = float(meta.globals_hash_input["SECOND_HAMAMATSU_FLUOR_DELAY"])
        expected = [base + frame * (0.05 + hold) for frame in range(5)]
        np.testing.assert_allclose(
            [meta.frame_elapsed_s[frame] for frame in range(5)],
            expected, rtol=0, atol=5e-15)


def test_0050_commanded_frame_timing_is_exact(repo_root):
    cfg, metas = _load(repo_root, "configs/bright_wait_50ms_20260728_0050.yaml")
    base = 0.6841116770321879
    for meta in metas:
        wait = float(meta.globals_hash_input["WAIT_BEFORE_FIRST_FLUOR"])
        assert meta.frame_elapsed_s[0] == pytest.approx(base + wait, abs=5e-15)
        assert meta.frame_elapsed_s[1] - meta.frame_elapsed_s[0] == pytest.approx(
            0.06, abs=5e-15)


@pytest.mark.parametrize(
    ("config_name", "n_train"),
    [
        ("configs/dark_hold_50ms_20260728_0044.yaml", 66),
        ("configs/bright_wait_50ms_20260728_0050.yaml", 60),
    ],
)
def test_geometry_and_template_fit_selection_contains_only_training_shots(
    repo_root, config_name, n_train
):
    cfg, metas = _load(repo_root, config_name)
    manifest, _, _ = _shot_design(cfg, metas)
    selected = select_fit_metas(metas, manifest, fit_scope="train")
    selected_ids = {meta.shot_id for meta in selected}
    train_ids = set(manifest.loc[manifest["split"] == "train", "shot_id"])
    held_out_ids = set(manifest.loc[manifest["split"] != "train", "shot_id"])
    assert len(selected) == n_train
    assert selected_ids == train_ids
    assert selected_ids.isdisjoint(held_out_ids)


def test_v3_row_timing_keeps_unmeasured_hardware_fields_null(repo_root):
    cfg, metas = _load(repo_root, "configs/dark_hold_50ms_20260728_0044.yaml")
    manifest, _, _ = _shot_design(cfg, metas)
    design = manifest.set_index("shot_id").loc[metas[0].shot_id].to_dict()
    values = {
        "background_fixed_offset": 100.0,
        "count_corrected_fixed_offset": 200.0,
    }
    row = _v3_row_fields(
        cfg, metas[0], 1, cfg.frame_specs[1], design,
        grid_id="grid_A", values=values)
    assert row["frame_name"] == "fluor2"
    assert row["interframe_gap_s"] == pytest.approx(0.1)
    assert row["dark_hold_s"] == pytest.approx(0.1)
    assert row["actual_light_on_s"] is None
    assert row["dds_frequency"] is None
    assert row["dds_amplitude"] is None
    assert row["background_template_offset"] == 100.0
    assert row["count_corrected_template"] == 200.0
