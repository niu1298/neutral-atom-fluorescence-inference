"""Additive schema-V4 timing and fixed-condition dataset invariants."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fluorescence_inference import schema
from fluorescence_inference.config import Config, Paths
from fluorescence_inference.dataset import ShotMeta, _versioned_row_fields


def _v4_config(scratch: Path) -> Config:
    raw = {
        "schema_version": "4.0",
        "dataset_id": "synthetic_repeated_50ms",
        "run_id": "synthetic_run",
        "date": "2026-07-31",
        "sequence_id": "synthetic_run",
        "sequence_type": "repeated_imaging",
        "timing": {
            "bright_wait_global": "WAIT_BEFORE_FIRST_FLUOR",
            "dark_gap_global": "SECOND_HAMAMATSU_FLUOR_DELAY",
        },
        "hardware": {
            "commanded_switch_state_during_exposure": (
                "science_imaging_switches_on"
            ),
        },
    }
    paths = Paths(
        experiment_data_root=scratch,
        tweezer_analysis_src=scratch,
        processed_root=scratch,
        reports_root=scratch,
    )
    return Config(
        raw=raw,
        paths=paths,
        config_path=scratch / "synthetic.yaml",
        config_sha256="synthetic-config-sha256",
    )


def _v4_rows(scratch: Path) -> pd.DataFrame:
    cfg = _v4_config(scratch)
    meta = ShotMeta(
        path=scratch / "shot_000.h5",
        shot_id=0,
        shot_order=0,
        timestamp="20260731T000000.0",
        sequence_index=113,
        sequence_date="2026-07-31",
        script_basename="synthetic_repeated_imaging",
        frame_elapsed_s={frame: frame * 0.060002 for frame in range(5)},
        exposure_ms={frame: 50.0 for frame in range(5)},
        exposure_names={frame: f"fluor{frame + 1}" for frame in range(5)},
        n_runs=1,
        globals_hash_input={
            "WAIT_BEFORE_FIRST_FLUOR": 0.0,
            "SECOND_HAMAMATSU_FLUOR_DELAY": 0.01,
        },
    )
    design = {
        "repetition_index": 0,
        "cycle_index": 0,
        "condition_id": "exposure_50ms",
        "split": "train",
        "sweep_value_s": float("nan"),
    }
    rows = []
    for frame in range(5):
        versioned = _versioned_row_fields(
            cfg,
            meta,
            frame,
            {"exposure_name": f"fluor{frame + 1}"},
            design,
            grid_id="grid_A",
            values={
                "background_fixed_offset": 100.0,
                "count_corrected_fixed_offset": 200.0,
            },
        )
        rows.append({
            **{name: pd.NA for name in schema.V4_ALL_COLUMNS},
            **versioned,
            "run_id": "synthetic_run",
            "shot_id": 0,
            "shot_order": 0,
            "frame_id": frame,
            "site_id": 0,
            "timestamp": "20260731T000000.0",
            "exposure_ms": 50.0,
            "frame_elapsed_s": frame * 0.060002,
            "site_x": 10.0,
            "site_y": 20.0,
            "roi_sum_raw": 300.0,
            "background_global": 100.0,
            "count_corrected_global": 200.0,
            "background_spatial": 100.0,
            "count_corrected_spatial": 200.0,
            "background_fixed_offset": 100.0,
            "count_corrected_fixed_offset": 200.0,
            "background_corrected_count": 200.0,
            "background_annulus_contaminated": 100.0,
            "count_corrected_annulus_contaminated": 200.0,
            "background_annulus_density_contaminated": 4.0,
            "raw_image_path": "shot_000.h5",
            "quality_flag": "ok",
            "grid": "grid_A",
            "site_row": 0,
            "site_col": 0,
            "roi_n_pixels": 25,
            "roi_max_pixel": 500.0,
            "site_detected": True,
        })
    return schema.coerce(pd.DataFrame(rows), version="4.0")


def test_v4_row_fields_keep_declared_dark_hold_distinct_from_compiled_gap(
    scratch: Path,
) -> None:
    table = _v4_rows(scratch)

    assert table["interframe_gap_s"].iloc[1:].to_numpy() == pytest.approx(
        [0.010002] * 4
    )
    assert table["dark_hold_s"].iloc[1:].to_numpy() == pytest.approx([0.01] * 4)
    assert table["cumulative_bright_s"].to_numpy() == pytest.approx(
        [0.05, 0.10, 0.15, 0.20, 0.25]
    )
    assert table["cumulative_dark_s"].to_numpy() == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04]
    )
    assert table["pulse_count"].tolist() == [1, 2, 3, 4, 5]


def test_v4_table_validates_and_rejects_inconsistent_prefix_timing(
    scratch: Path,
) -> None:
    table = _v4_rows(scratch)
    report = schema.validate(
        table, version="4.0", expected_shots=1, expected_frames=5, expected_sites=1
    )
    assert report.ok, report.errors

    broken = table.copy()
    broken.loc[broken["frame_index"] == 4, "cumulative_dark_s"] = 0.05
    failed = schema.validate(broken, version="4.0")
    assert not failed.ok
    assert "cumulative_dark_s does not equal prefix declared dark holds" in failed.errors


def test_v4_contract_is_additive_to_v3() -> None:
    assert list(schema.V4_ALL_COLUMNS)[: len(schema.V3_ALL_COLUMNS)] == list(
        schema.V3_ALL_COLUMNS
    )
    assert set(schema.V4_ADDITIONAL_COLUMNS).isdisjoint(schema.V3_ALL_COLUMNS)
