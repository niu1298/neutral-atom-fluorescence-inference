"""Deterministic shot-level splits for cyclic sweep acquisitions.

The independent unit is a complete shot.  A cycle contains one shot from
every sweep condition, so assigning whole cycles keeps all frames and sites
together while representing every condition in every subset.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")
CHRONOLOGICAL_STRATEGY = "chronological_cycle_60_20_20"
SEEDED_STRATEGY = "seeded_cycle_60_20_20"
SUPPORTED_STRATEGIES = (CHRONOLOGICAL_STRATEGY, SEEDED_STRATEGY)


def build_cycle_split_manifest(
    shots: pd.DataFrame,
    *,
    strategy: str = CHRONOLOGICAL_STRATEGY,
    seed: int = 0,
    require_complete_cycles: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign one split to each shot using whole acquisition cycles.

    Parameters
    ----------
    shots:
        One row per shot with ``shot_id``, ``shot_order``, ``condition_id`` and
        ``cycle_index``.  ``repetition_index`` and ``sweep_value_s`` are
        retained in the returned manifest when present.
    strategy:
        Chronological uses earliest cycles for training, the next cycles for
        validation, and the latest cycles for the one-shot test.  The seeded
        sensitivity split permutes cycles with a recorded seed.
    """
    required = {"shot_id", "shot_order", "condition_id", "cycle_index"}
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"shot design is missing columns: {missing}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"unsupported split strategy {strategy!r}; expected {SUPPORTED_STRATEGIES}")

    manifest = shots.copy()
    if manifest["shot_id"].duplicated().any():
        raise ValueError("shot design must have exactly one row per shot_id")
    manifest = manifest.sort_values("shot_order", kind="stable").reset_index(drop=True)

    cycles = sorted(int(v) for v in manifest["cycle_index"].unique())
    if len(cycles) < 5:
        raise ValueError("at least five complete cycles are required for a 60/20/20 split")

    conditions = set(manifest["condition_id"].astype(str))
    cycle_conditions = {
        int(c): set(g["condition_id"].astype(str))
        for c, g in manifest.groupby("cycle_index", observed=True)
    }
    incomplete = {
        c: sorted(conditions - got)
        for c, got in cycle_conditions.items() if got != conditions
    }
    duplicate_in_cycle = (
        manifest.duplicated(subset=["cycle_index", "condition_id"]).any())
    if require_complete_cycles and (incomplete or duplicate_in_cycle):
        detail = f"incomplete={incomplete}" if incomplete else "duplicate condition in cycle"
        raise ValueError(f"cycles are not complete one-shot-per-condition blocks: {detail}")

    ordered = np.asarray(cycles, dtype=int)
    if strategy == SEEDED_STRATEGY:
        ordered = np.random.default_rng(seed).permutation(ordered)

    n_cycle = len(ordered)
    n_train = int(np.floor(0.60 * n_cycle))
    n_validation = int(np.floor(0.20 * n_cycle))
    n_test = n_cycle - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("split allocation produced an empty subset")

    cycle_sets = {
        "train": sorted(int(v) for v in ordered[:n_train]),
        "validation": sorted(
            int(v) for v in ordered[n_train:n_train + n_validation]),
        "test": sorted(int(v) for v in ordered[n_train + n_validation:]),
    }
    cycle_to_split = {
        cycle: split for split, selected in cycle_sets.items() for cycle in selected
    }
    manifest["split"] = manifest["cycle_index"].map(cycle_to_split).astype("string")

    representation = (
        manifest.groupby(["split", "condition_id"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(SPLIT_NAMES, fill_value=0)
    )
    if (representation == 0).any().any():
        raise ValueError("not every sweep condition is represented in every split")

    public_columns = [
        c for c in (
            "shot_id", "shot_order", "condition_id", "sweep_value_s",
            "repetition_index", "cycle_index", "split",
        ) if c in manifest.columns
    ]
    manifest = manifest[public_columns]
    records = manifest.to_dict(orient="records")
    digest = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()
    metadata: dict[str, Any] = {
        "strategy": strategy,
        "seed": int(seed) if strategy == SEEDED_STRATEGY else None,
        "unit": "complete_acquisition_cycle",
        "fractions_target": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "cycle_sets": cycle_sets,
        "n_cycles": n_cycle,
        "n_shots": int(len(manifest)),
        "n_conditions": int(len(conditions)),
        "shots_per_split": {
            k: int(v) for k, v in manifest["split"].value_counts().items()
        },
        "shots_per_condition_per_split": {
            split: {
                str(condition): int(n)
                for condition, n in representation.loc[split].items()
            }
            for split in SPLIT_NAMES
        },
        "manifest_sha256": digest,
    }
    return manifest, metadata


def attach_split(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach a manifest and fail if any shot is missing or crosses subsets."""
    if "shot_id" not in rows or "shot_id" not in manifest:
        raise ValueError("both rows and manifest must contain shot_id")
    scope_cols = ("dataset_id", "run_id")
    join_keys = [
        *[
            col for col in scope_cols
            if col in rows.columns and col in manifest.columns
        ],
        "shot_id",
    ]
    # Falling back to shot_id is backwards compatible for a single V0 run.
    # It is not safe when a missing scope key distinguishes multiple runs.
    for col in scope_cols:
        if col in join_keys:
            continue
        for label, table in (("rows", rows), ("manifest", manifest)):
            if col not in table.columns:
                continue
            scoped_keys = table[[*join_keys, col]].drop_duplicates()
            if scoped_keys.duplicated(subset=join_keys).any():
                raise ValueError(
                    f"cannot attach split: {label} require {col} to distinguish "
                    f"shot keys but the other table does not provide that key")

    shot_split = manifest[[*join_keys, "split"]].drop_duplicates()
    if shot_split.duplicated(subset=join_keys).any():
        raise ValueError("manifest assigns more than one split to a shot")
    out = rows.drop(columns=["split"], errors="ignore").merge(
        shot_split, on=join_keys, how="left", validate="many_to_one")
    if out["split"].isna().any():
        missing = (
            out.loc[out["split"].isna(), join_keys]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"rows contain shots absent from split manifest: {missing}")
    return out
