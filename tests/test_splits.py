"""Shot/cycle split invariants for the loss sweeps."""
from __future__ import annotations

import pandas as pd
import pytest

from fluorescence_inference.splits import (
    CHRONOLOGICAL_STRATEGY,
    SEEDED_STRATEGY,
    attach_split,
    build_cycle_split_manifest,
)


def _design(n_condition: int = 11, n_cycle: int = 10) -> pd.DataFrame:
    rows = []
    for cycle in range(n_cycle):
        for condition in range(n_condition):
            shot = cycle * n_condition + condition
            rows.append({
                "shot_id": shot,
                "shot_order": shot,
                "condition_id": f"condition_{condition:02d}",
                "sweep_value_s": 0.1 + 0.2 * condition,
                "repetition_index": cycle,
                "cycle_index": cycle,
            })
    return pd.DataFrame(rows)


def test_chronological_cycle_split_is_exact_60_20_20():
    manifest, meta = build_cycle_split_manifest(_design())
    assert meta["strategy"] == CHRONOLOGICAL_STRATEGY
    assert meta["cycle_sets"] == {
        "train": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7],
        "test": [8, 9],
    }
    assert manifest["split"].value_counts().to_dict() == {
        "train": 66, "validation": 22, "test": 22,
    }
    per = manifest.groupby(["split", "condition_id"], observed=True).size()
    assert set(per.loc["train"]) == {6}
    assert set(per.loc["validation"]) == {2}
    assert set(per.loc["test"]) == {2}


def test_split_is_deterministic_and_hash_pinned():
    first, meta_first = build_cycle_split_manifest(_design())
    second, meta_second = build_cycle_split_manifest(_design())
    pd.testing.assert_frame_equal(first, second)
    assert meta_first["manifest_sha256"] == meta_second["manifest_sha256"]


def test_seeded_interleaved_cycle_sensitivity_is_deterministic():
    first, meta_first = build_cycle_split_manifest(
        _design(), strategy=SEEDED_STRATEGY, seed=20260728)
    second, meta_second = build_cycle_split_manifest(
        _design(), strategy=SEEDED_STRATEGY, seed=20260728)
    pd.testing.assert_frame_equal(first, second)
    assert meta_first == meta_second
    assert meta_first["cycle_sets"] != {
        "train": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7],
        "test": [8, 9],
    }


def test_all_sites_and_frames_from_a_shot_keep_one_split():
    manifest, _ = build_cycle_split_manifest(_design(n_condition=3))
    rows = pd.DataFrame([
        {"shot_id": shot, "frame_id": frame, "site_id": site}
        for shot in manifest["shot_id"]
        for frame in range(5)
        for site in range(4)
    ])
    out = attach_split(rows, manifest)
    assert (out.groupby("shot_id", observed=True)["split"].nunique() == 1).all()


def test_incomplete_or_duplicate_cycles_are_rejected():
    incomplete = _design().iloc[:-1]
    with pytest.raises(ValueError, match="cycles are not complete"):
        build_cycle_split_manifest(incomplete)
    duplicate = pd.concat([_design(), _design().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        build_cycle_split_manifest(duplicate)
