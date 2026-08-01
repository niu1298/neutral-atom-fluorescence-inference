"""Machine-independent contracts for the optimized 2026-07-31 datasets."""
from __future__ import annotations

from fluorescence_inference.config import load_config


CONFIG_EXPECTATIONS = {
    "bright_lifetime_20260731_0090.yaml": {
        "dataset_id": "bright_lifetime_20260731_0090",
        "sequence_type": "bright_lifetime",
        "n_shots": 200,
        "n_frames": 2,
        "exposure_s": 0.05,
        "primary_background": "fixed_offset",
    },
    "dark_lifetime_20260731_0094.yaml": {
        "dataset_id": "dark_lifetime_20260731_0094",
        "sequence_type": "dark_lifetime",
        "n_shots": 220,
        "n_frames": 5,
        "exposure_s": 0.05,
        "primary_background": "fixed_offset",
    },
    "imaging_5frame_50ms_20260731_0113.yaml": {
        "dataset_id": "imaging_5frame_50ms_20260731_0113",
        "sequence_type": "repeated_imaging",
        "n_shots": 100,
        "n_frames": 5,
        "exposure_s": 0.05,
        "primary_background": "spatial",
    },
    "imaging_5frame_100ms_20260731_0114.yaml": {
        "dataset_id": "imaging_5frame_100ms_20260731_0114",
        "sequence_type": "repeated_imaging",
        "n_shots": 100,
        "n_frames": 5,
        "exposure_s": 0.10,
        "primary_background": "fixed_offset",
    },
    "imaging_5frame_200ms_20260731_0115.yaml": {
        "dataset_id": "imaging_5frame_200ms_20260731_0115",
        "sequence_type": "repeated_imaging",
        "n_shots": 100,
        "n_frames": 5,
        "exposure_s": 0.20,
        "primary_background": "fixed_offset",
    },
}


def test_optimized_configs_are_schema_v4_and_machine_independent(repo_root):
    for filename, expected in CONFIG_EXPECTATIONS.items():
        cfg = load_config(repo_root / "configs" / filename, root=repo_root)

        assert cfg.schema_version == "4.0"
        assert cfg.dataset_id == expected["dataset_id"]
        assert cfg.sequence_type == expected["sequence_type"]
        assert cfg["source"]["expected_n_shots"] == expected["n_shots"]
        assert cfg.n_frames == expected["n_frames"]
        assert cfg["source"]["expected_exposure_s"] == expected["exposure_s"]
        assert cfg["extraction"]["fit_scope"] == "train"
        assert cfg["background"]["primary_method"] == expected[
            "primary_background"
        ]
        assert not cfg["source"]["shot_subdir"].startswith(("/", "\\"))


def test_optimized_sweeps_and_fixed_condition_splits_are_explicit(repo_root):
    bright = load_config(
        repo_root / "configs" / "bright_lifetime_20260731_0090.yaml",
        root=repo_root,
    )
    dark = load_config(
        repo_root / "configs" / "dark_lifetime_20260731_0094.yaml",
        root=repo_root,
    )
    assert len(bright["sweep"]["values_s"]) == 10
    assert len(dark["sweep"]["values_s"]) == 11
    assert bright["split"]["strategy"] == "chronological_cycle_60_20_20"
    assert dark["split"]["strategy"] == "chronological_cycle_60_20_20"

    for filename in CONFIG_EXPECTATIONS:
        if not filename.startswith("imaging_"):
            continue
        cfg = load_config(repo_root / "configs" / filename, root=repo_root)
        assert cfg["design"]["kind"] == "fixed_condition"
        assert cfg["split"]["strategy"] == "chronological_shot_60_20_20"
        assert cfg["timing"].get(
            "dark_gap_global", "SECOND_HAMAMATSU_FLUOR_DELAY"
        ) == "SECOND_HAMAMATSU_FLUOR_DELAY"
