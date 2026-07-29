"""Complete-shot clustered-bootstrap contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fluorescence_inference.cluster_bootstrap import (
    condition_stratified_cluster_bootstrap,
    resample_cycle_blocks,
    resample_shot_clusters,
    whole_cycle_cluster_bootstrap,
)


def _long_sweep(n_conditions: int = 4, n_repetitions: int = 10) -> pd.DataFrame:
    rows = []
    shot_id = 0
    for repetition in range(n_repetitions):
        for condition in range(n_conditions):
            for frame in range(3):
                for site in range(5):
                    rows.append(
                        {
                            "run_id": "R",
                            "shot_id": shot_id,
                            "condition_id": condition,
                            "cycle": repetition,
                            "frame_index": frame,
                            "site_id": site,
                            "value": 100 * shot_id + 10 * frame + site,
                        }
                    )
            shot_id += 1
    return pd.DataFrame(rows)


def test_cluster_resample_preserves_every_site_and_frame_in_each_draw():
    data = _long_sweep(n_conditions=3, n_repetitions=4)
    sample = resample_shot_clusters(data, rng=np.random.default_rng(19))
    original_rows_per_shot = 3 * 5
    assert (
        sample.groupby("_bootstrap_cluster_id").size() == original_rows_per_shot
    ).all()
    for _, block in sample.groupby("_bootstrap_cluster_id"):
        assert len(block[["frame_index", "site_id"]].drop_duplicates()) == 15
        assert block["shot_id"].nunique() == 1
        assert block["condition_id"].nunique() == 1
    assert sample["_bootstrap_cluster_id"].nunique() == 12
    assert (
        sample[["condition_id", "_bootstrap_cluster_id"]]
        .drop_duplicates()
        .groupby("condition_id")
        .size()
        .eq(4)
        .all()
    )


def test_cluster_bootstrap_is_deterministic_and_reports_requested_interval():
    data = _long_sweep(n_conditions=3, n_repetitions=6)

    def statistic(table):
        shot_mean = table.groupby(
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in table
            else "shot_id"
        )["value"].mean()
        return {"shot_mean": float(shot_mean.mean())}

    a = condition_stratified_cluster_bootstrap(
        data, statistic, n_boot=40, seed=55
    )
    b = condition_stratified_cluster_bootstrap(
        data, statistic, n_boot=40, seed=55
    )
    pd.testing.assert_frame_equal(a.draws, b.draws)
    interval = a.intervals().iloc[0]
    assert interval["lower"] < interval["estimate"] < interval["upper"]
    assert interval["n_bootstrap"] == 40


def test_cycle_resample_preserves_complete_cycles_conditions_and_shots():
    data = _long_sweep(n_conditions=3, n_repetitions=5)
    sample = resample_cycle_blocks(
        data,
        rng=np.random.default_rng(812),
        cycle_cols=("run_id", "cycle"),
    )
    assert sample["_bootstrap_cycle_id"].nunique() == 5
    for _, cycle in sample.groupby("_bootstrap_cycle_id"):
        assert cycle["condition_id"].nunique() == 3
        assert cycle["_bootstrap_source_cycle"].nunique() == 1
        assert cycle["_bootstrap_cluster_id"].nunique() == 3
        for _, shot in cycle.groupby("_bootstrap_cluster_id"):
            assert shot["shot_id"].nunique() == 1
            assert len(shot[["frame_index", "site_id"]].drop_duplicates()) == 15


def test_cycle_bootstrap_is_deterministic_and_records_failed_refits():
    data = _long_sweep(n_conditions=3, n_repetitions=6)

    def deterministic_statistic(table):
        return {"mean": float(table["value"].mean())}

    first = whole_cycle_cluster_bootstrap(
        data,
        deterministic_statistic,
        n_boot=20,
        seed=91,
        cycle_cols=("run_id", "cycle"),
    )
    second = whole_cycle_cluster_bootstrap(
        data,
        deterministic_statistic,
        n_boot=20,
        seed=91,
        cycle_cols=("run_id", "cycle"),
    )
    pd.testing.assert_frame_equal(first.draws, second.draws)

    calls = 0

    def intermittently_failing(table):
        nonlocal calls
        calls += 1
        if calls > 1 and calls % 3 == 0:
            raise RuntimeError("declared synthetic refit failure")
        return {"mean": float(table["value"].mean())}

    result = whole_cycle_cluster_bootstrap(
        data,
        intermittently_failing,
        n_boot=9,
        seed=19,
        cycle_cols=("run_id", "cycle"),
        max_fail_fraction=0.5,
    )
    assert result.n_requested == 9
    assert result.n_failed == 3
    assert result.n_successful == 6
    assert [failure["replicate"] for failure in result.failures] == [1, 4, 7]
    assert all(
        failure["exception"] == "RuntimeError" for failure in result.failures
    )


def test_cycle_resample_rejects_an_incomplete_condition_cycle():
    data = _long_sweep(n_conditions=3, n_repetitions=4)
    incomplete = data.loc[
        ~((data["cycle"] == 2) & (data["condition_id"] == 1))
    ]
    with np.testing.assert_raises_regex(ValueError, "every condition"):
        resample_cycle_blocks(
            incomplete,
            rng=np.random.default_rng(4),
            cycle_cols=("run_id", "cycle"),
        )
