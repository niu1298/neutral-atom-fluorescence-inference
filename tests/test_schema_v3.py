"""Additive schema-V3 migration and validation."""
from __future__ import annotations

import pandas as pd
import pytest

from fluorescence_inference import schema


def _v2_table() -> pd.DataFrame:
    rows = []
    for shot in range(2):
        for frame in range(2):
            rows.append({
                "run_id": "legacy_run",
                "shot_id": shot,
                "shot_order": shot,
                "frame_id": frame,
                "site_id": 0,
                "timestamp": "20260727T000000.0",
                "exposure_ms": 100.0,
                "frame_elapsed_s": 0.7 + 0.11 * frame,
                "site_x": 10.0,
                "site_y": 20.0,
                "roi_sum_raw": 1000.0,
                "background_global": 480.0,
                "count_corrected_global": 520.0,
                "background_spatial": 490.0,
                "count_corrected_spatial": 510.0,
                "background_fixed_offset": 495.0,
                "count_corrected_fixed_offset": 505.0,
                "background_corrected_count": 505.0,
                "background_annulus_contaminated": 500.0,
                "count_corrected_annulus_contaminated": 500.0,
                "background_annulus_density_contaminated": 20.0,
                "raw_image_path": f"shot_{shot:02d}.h5",
                "quality_flag": "ok",
                "grid": "grid_A",
                "site_row": 0,
                "site_col": 0,
                "roi_n_pixels": 25,
                "roi_max_pixel": 900.0,
                "site_detected": True,
            })
    return schema.coerce(pd.DataFrame(rows))


def _v3_run(
    *,
    dataset_id: str,
    run_id: str,
    n_shots: int,
    n_frames: int,
    n_sites: int,
) -> pd.DataFrame:
    template = schema.migrate_v2_to_v3(
        _v2_table().iloc[[0]],
        dataset_id=dataset_id,
        date="2026-07-28",
        sequence_id=run_id,
        sequence_type="test_sweep",
    ).iloc[0].to_dict()
    rows = []
    for shot in range(n_shots):
        for frame in range(n_frames):
            for site in range(n_sites):
                row = dict(template)
                row.update({
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "sequence_id": run_id,
                    "shot_id": shot,
                    "shot_order": shot,
                    "repetition_index": shot,
                    "cycle_index": shot,
                    "condition_id": "single_condition",
                    "split": "train",
                    "frame_id": frame,
                    "frame_index": frame,
                    "frame_name": f"fluor{frame + 1}",
                    "frame_elapsed_s": 0.7 + 0.11 * frame,
                    "frame_start_s": 0.7 + 0.11 * frame,
                    "interframe_gap_s": None if frame == 0 else 0.01,
                    "site_id": site,
                    "site_row": site,
                    "site_col": 0,
                    "site_x": 10.0 + site,
                    "site_y": 20.0,
                    "raw_image_path": f"{run_id}_{shot:03d}.h5",
                })
                rows.append(row)
    return schema.coerce(pd.DataFrame(rows), version="3.0")


def test_v0_schema_constant_and_default_contract_are_unchanged():
    assert schema.SCHEMA_VERSION == "2.0"
    assert schema.PRIMARY_KEY == ("run_id", "shot_id", "frame_id", "site_id")
    assert list(schema.empty_frame().columns) == list(schema.ALL_COLUMNS)
    assert "dataset_id" not in schema.ALL_COLUMNS


def test_v2_to_v3_migration_is_additive_and_valid():
    old = _v2_table()
    migrated = schema.migrate_v2_to_v3(
        old,
        dataset_id="paired_100ms_v0",
        date="2026-07-27",
        sequence_id="20260727_0211",
        sequence_type="paired_readout",
        frame_names={0: "fluor", 1: "fluor2"},
    )
    pd.testing.assert_frame_equal(
        migrated[list(old.columns)].reset_index(drop=True),
        old.reset_index(drop=True),
        check_dtype=False,
    )
    report = schema.validate(
        migrated, version="3.0", expected_shots=2,
        expected_frames=2, expected_sites=1)
    assert report.ok, report.errors
    assert migrated.attrs["schema_version"] == "3.0-migrated-from-2.0"
    assert migrated["actual_light_on_s"].isna().all()
    assert "actual_light_on_s" in migrated.attrs["unrecoverable_fields"]
    assert migrated.loc[migrated["frame_id"] == 1, "interframe_gap_s"].iloc[0] == (
        pytest.approx(0.01))


def test_v3_alias_mismatch_is_rejected():
    migrated = schema.migrate_v2_to_v3(
        _v2_table(),
        dataset_id="paired_100ms_v0",
        date="2026-07-27",
        sequence_id="20260727_0211",
        sequence_type="paired_readout",
    )
    migrated.loc[0, "count_corrected_template"] += 1.0
    report = schema.validate(migrated, version="3.0")
    assert not report.ok
    assert any("backwards-compatible alias" in error for error in report.errors)


def test_v3_migration_requires_public_identifiers():
    with pytest.raises(ValueError, match="dataset_id"):
        schema.migrate_v2_to_v3(
            _v2_table(),
            dataset_id="",
            date="2026-07-27",
            sequence_id="20260727_0211",
            sequence_type="paired_readout",
        )


def test_v3_validation_supports_concatenated_runs_with_distinct_shapes():
    two_frame_one_site = _v3_run(
        dataset_id="two_frame",
        run_id="run_2f",
        n_shots=2,
        n_frames=2,
        n_sites=1,
    )
    three_frame_two_site = _v3_run(
        dataset_id="three_frame",
        run_id="run_3f",
        n_shots=3,
        n_frames=3,
        n_sites=2,
    )
    combined = pd.concat(
        [two_frame_one_site, three_frame_two_site], ignore_index=True)

    report = schema.validate(combined, version="3.0", expected_shots=5)

    assert report.ok, report.errors
    assert report.stats["frame_ids_by_run"] == {
        "three_frame/run_3f": [0, 1, 2],
        "two_frame/run_2f": [0, 1],
    }
    assert report.stats["shots_with_incomplete_block"] == 0

    incomplete = combined.drop(
        three_frame_two_site.index[-1] + len(two_frame_one_site))
    failed = schema.validate(incomplete, version="3.0")
    assert not failed.ok
    assert failed.stats["shots_with_incomplete_block"] == 1
