"""The standardized dataset: structure, pairing, determinism, flags.

These run against the real exported table and skip when it is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference import schema
from fluorescence_inference.dataset import SourceDataError, discover_shots

pytestmark = pytest.mark.real_data


def test_table_passes_schema_validation(cfg, real_dataset):
    df, _, _ = real_dataset
    report = schema.validate(df, expected_frames=cfg.n_frames,
                             expected_shots=cfg["source"]["expected_n_shots"],
                             expected_sites=cfg.n_sites_expected)
    assert report.ok, report.errors


def test_expected_two_frame_structure(cfg, real_dataset):
    df, _, _ = real_dataset
    assert sorted(df["frame_id"].unique()) == [0, 1]
    per = df.groupby(["shot_id", "site_id"], observed=True)["frame_id"].nunique()
    assert (per == 2).all(), "every (shot, site) must appear in both frames"


def test_pairing_is_by_shot_and_site(real_dataset):
    from fluorescence_inference.quality_control import paired_table

    df, _, _ = real_dataset
    pairs = paired_table(df, "background_corrected_count")
    assert len(pairs) == df["shot_id"].nunique() * df["site_id"].nunique()
    assert pairs.duplicated(subset=["shot_id", "site_id"]).sum() == 0

    probe = pairs.iloc[123]
    sub = df[(df["shot_id"] == probe["shot_id"]) & (df["site_id"] == probe["site_id"])]
    got = sub.set_index("frame_id")["background_corrected_count"]
    assert got.loc[0] == pytest.approx(probe["f0"])
    assert got.loc[1] == pytest.approx(probe["f1"])


def test_primary_key_is_unique(real_dataset):
    df, _, _ = real_dataset
    assert df.duplicated(subset=list(schema.PRIMARY_KEY)).sum() == 0


def test_background_columns_are_self_consistent(real_dataset):
    df, _, _ = real_dataset
    np.testing.assert_allclose(
        df["background_corrected_count"], df["roi_sum"] - df["local_background"],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        df["common_mode_corrected_count"], df["roi_sum"] - df["global_background"],
        rtol=0, atol=1e-9)


def test_flagging_behaviour(real_dataset):
    df, sites_df, _ = real_dataset
    flagged = df[df["quality_flag"].astype(str).str.contains("site_not_detected")]
    undetected = set(sites_df.loc[~sites_df["site_detected"].astype(bool), "site_id"])
    assert set(flagged["site_id"]) == undetected
    # a flag never removes a row
    assert len(flagged) == len(undetected) * df["shot_id"].nunique() * 2
    # and rows are only flagged for a declared reason
    seen = set(df["quality_flag"].astype(str).str.split("|").explode())
    assert seen <= set(schema.QUALITY_FLAGS)


def test_no_absolute_paths_in_the_table(real_dataset):
    df, _, _ = real_dataset
    paths = df["raw_image_path"].dropna().astype(str)
    assert not paths.str.contains(r"[/\\]").any()
    assert not paths.str.contains(r"[A-Za-z]:").any()


def test_metadata_matches_the_configured_acquisition(cfg, real_dataset):
    _, _, meta = real_dataset
    sm = meta["shot_metadata"]
    assert sm["exposure_matches_config"]
    assert sm["interframe_delta_matches_config"]
    assert sm["shot_ids_unique"]
    assert len(sm["script_basenames"]) == 1
    assert len(sm["sequence_indices"]) == 1


def test_only_a_repetition_counter_varies(real_dataset):
    """A physics parameter varying inside a 'single-condition' run is a bug."""
    _, _, meta = real_dataset
    varying = meta["shot_metadata"]["varying_globals"]
    assert set(varying) == {"REPEAT"}, varying


def test_site_geometry_is_stable_across_the_run(real_dataset):
    df, sites_df, _ = real_dataset
    per_site = df.groupby("site_id", observed=True)[["site_x", "site_y"]].nunique()
    assert (per_site == 1).all().all()
    assert len(sites_df) == df["site_id"].nunique()


def test_dataset_generation_is_deterministic(cfg, real_dataset):
    """Rebuild from the raw shots and require an identical table."""
    from fluorescence_inference.dataset import build_frame_site_table

    df, sites_df, _ = real_dataset
    try:
        again, sites_again, _ = build_frame_site_table(cfg, progress=False)
    except SourceDataError as exc:
        pytest.skip(f"raw shots unavailable: {exc}")

    pd.testing.assert_frame_equal(
        df.sort_values(list(schema.PRIMARY_KEY)).reset_index(drop=True),
        again.sort_values(list(schema.PRIMARY_KEY)).reset_index(drop=True),
        check_dtype=False)
    pd.testing.assert_frame_equal(
        sites_df.sort_values("site_id").reset_index(drop=True),
        sites_again.sort_values("site_id").reset_index(drop=True),
        check_dtype=False)


def test_exporter_refuses_a_wrong_shot_count(cfg):
    """The guard that stops a partial download becoming a silent dataset."""
    from copy import deepcopy

    from fluorescence_inference.config import Config
    from fluorescence_inference.dataset import build_frame_site_table

    try:
        discover_shots(cfg)
    except SourceDataError as exc:
        pytest.skip(f"raw shots unavailable: {exc}")

    raw = deepcopy(cfg.raw)
    raw["source"]["expected_n_shots"] = 99
    bad = Config(raw=raw, paths=cfg.paths, config_path=cfg.config_path,
                 config_sha256=cfg.config_sha256)
    with pytest.raises(SourceDataError, match="expected 99 shots"):
        build_frame_site_table(bad, progress=False)


def test_load_dataset_fails_with_an_actionable_message(cfg):
    from copy import deepcopy

    from fluorescence_inference.config import Config, Paths
    from fluorescence_inference.dataset import load_dataset

    paths = Paths(experiment_data_root=cfg.paths.experiment_data_root,
                  tweezer_analysis_src=cfg.paths.tweezer_analysis_src,
                  processed_root=cfg.paths.processed_root / "does_not_exist",
                  reports_root=cfg.paths.reports_root)
    bad = Config(raw=deepcopy(cfg.raw), paths=paths, config_path=cfg.config_path,
                 config_sha256=cfg.config_sha256)
    with pytest.raises(SourceDataError, match="export_processed_dataset"):
        load_dataset(bad)
