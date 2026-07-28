"""Schema validation: it must catch the failures it claims to catch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference import schema


def make_table(n_shots: int = 3, n_frames: int = 2, n_sites: int = 4
               ) -> pd.DataFrame:
    rows = []
    for s in range(n_shots):
        for f in range(n_frames):
            for k in range(n_sites):
                rows.append({
                    "run_id": "R", "shot_id": s, "shot_order": s, "frame_id": f,
                    "site_id": k, "timestamp": "20260727T000000.0",
                    "exposure_ms": 100.0, "frame_elapsed_s": 0.7 + 0.11 * f,
                    "site_x": 10.0 + k, "site_y": 20.0 + k,
                    "roi_sum": 1000.0 + k, "local_background": 500.0,
                    "global_background": 480.0,
                    "background_corrected_count": 500.0 + k,
                    "raw_image_path": f"shot_{s:02d}.h5", "quality_flag": "ok",
                    "grid": "grid_A", "site_row": k // 2, "site_col": k % 2,
                    "roi_n_pixels": 25, "local_background_density": 20.0,
                    "roi_max_pixel": 900.0,
                    "common_mode_corrected_count": 520.0 + k,
                    "site_detected": True,
                })
    return schema.coerce(pd.DataFrame(rows))


def test_clean_table_validates():
    r = schema.validate(make_table(), expected_frames=2, expected_shots=3,
                        expected_sites=4)
    assert r.ok, r.errors
    assert r.stats["duplicate_primary_keys"] == 0
    assert r.stats["n_rows"] == 24
    assert r.stats["n_rows_flagged"] == 0


def test_duplicate_primary_key_is_an_error():
    df = make_table()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    r = schema.validate(df)
    assert not r.ok
    assert any("duplicate" in e for e in r.errors)


def test_incomplete_shot_block_is_an_error():
    df = make_table().drop(index=0).reset_index(drop=True)
    r = schema.validate(df, expected_frames=2, expected_sites=4)
    assert not r.ok
    assert any("do not have exactly" in e for e in r.errors)


def test_missing_frame_id_is_an_error():
    df = make_table()
    df = df[df["frame_id"] == 0].reset_index(drop=True)
    r = schema.validate(df, expected_frames=2)
    assert not r.ok
    assert any("frame ids" in e for e in r.errors)


def test_site_count_mismatch_is_an_error():
    r = schema.validate(make_table(n_sites=4), expected_sites=200)
    assert not r.ok
    assert any("expected 200 sites" in e for e in r.errors)


def test_unflagged_nonfinite_value_is_an_error():
    df = make_table()
    df.loc[0, "background_corrected_count"] = np.nan
    r = schema.validate(df)
    assert not r.ok
    assert any("non-finite" in e for e in r.errors)


def test_flagged_nonfinite_value_is_accepted():
    df = make_table()
    df.loc[0, "background_corrected_count"] = np.nan
    df.loc[0, "quality_flag"] = "nonfinite_count"
    r = schema.validate(df, expected_frames=2, expected_shots=3, expected_sites=4)
    assert r.ok, r.errors
    assert r.stats["n_rows_flagged"] == 1


def test_unknown_quality_flag_is_an_error():
    df = make_table()
    df.loc[0, "quality_flag"] = "made_up_flag"
    r = schema.validate(df)
    assert not r.ok
    assert any("unknown quality flags" in e for e in r.errors)


def test_absolute_path_in_raw_image_path_is_an_error():
    df = make_table()
    df.loc[0, "raw_image_path"] = r"C:\somewhere\shot_00.h5"
    r = schema.validate(df)
    assert not r.ok
    assert any("bare file name" in e for e in r.errors)


def test_moving_site_coordinates_are_an_error():
    df = make_table()
    df.loc[0, "site_x"] = 999.0
    r = schema.validate(df)
    assert not r.ok
    assert any("non-constant coordinates" in e for e in r.errors)


def test_missing_required_column_short_circuits():
    df = make_table().drop(columns=["local_background"])
    r = schema.validate(df)
    assert not r.ok
    assert "missing required columns" in r.errors[0]


def test_raise_if_failed():
    with pytest.raises(ValueError, match="failed validation"):
        schema.validate(make_table().drop(columns=["roi_sum"])).raise_if_failed()


def test_empty_frame_has_declared_columns():
    assert list(schema.empty_frame().columns) == list(schema.ALL_COLUMNS)
