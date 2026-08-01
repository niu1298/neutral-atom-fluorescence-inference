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
CHRONOLOGICAL_SHOT_STRATEGY = "chronological_shot_60_20_20"
SEEDED_BLOCK_STRATEGY = "seeded_contiguous_block_60_20_20"


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


def build_shot_split_manifest(
    shots: pd.DataFrame,
    *,
    strategy: str = CHRONOLOGICAL_SHOT_STRATEGY,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign complete fixed-condition shots to deterministic 60/20/20 splits.

    The primary strategy is chronological.  The sensitivity strategy first
    partitions acquisition order into five contiguous blocks and then applies
    a seeded permutation to those blocks.  In both cases a shot remains the
    indivisible unit and every one of its frames/sites receives one split.
    """
    required = {"shot_id", "shot_order", "condition_id"}
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"shot design is missing columns: {missing}")
    if strategy not in (CHRONOLOGICAL_SHOT_STRATEGY, SEEDED_BLOCK_STRATEGY):
        raise ValueError(
            f"unsupported fixed-condition strategy {strategy!r}; expected "
            f"{(CHRONOLOGICAL_SHOT_STRATEGY, SEEDED_BLOCK_STRATEGY)}"
        )
    manifest = shots.sort_values("shot_order", kind="stable").reset_index(drop=True).copy()
    if manifest["shot_id"].duplicated().any():
        raise ValueError("shot design must have exactly one row per shot_id")
    n_shot = len(manifest)
    if n_shot < 5:
        raise ValueError("at least five complete shots are required for a 60/20/20 split")

    # Five contiguous blocks make the alternate split drift-aware while
    # retaining deterministic 60/20/20 sizes for the 100-shot optimized runs.
    block_index = np.floor(np.arange(n_shot) * 5 / n_shot).astype(int)
    block_order = np.arange(5, dtype=int)
    if strategy == SEEDED_BLOCK_STRATEGY:
        block_order = np.random.default_rng(seed).permutation(block_order)
    assignment = {
        int(block_order[0]): "train",
        int(block_order[1]): "train",
        int(block_order[2]): "train",
        int(block_order[3]): "validation",
        int(block_order[4]): "test",
    }
    manifest["split_block_index"] = block_index
    manifest["split"] = pd.Series(block_index).map(assignment).astype("string")
    if manifest["split"].isna().any():
        raise RuntimeError("fixed-condition split left a shot unassigned")

    public_columns = [
        c for c in (
            "shot_id", "shot_order", "condition_id", "sweep_value_s",
            "repetition_index", "cycle_index", "split_block_index", "split",
        ) if c in manifest.columns
    ]
    manifest = manifest[public_columns]
    records = manifest.to_dict(orient="records")
    digest = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()
    counts = manifest["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    metadata: dict[str, Any] = {
        "strategy": strategy,
        "seed": int(seed) if strategy == SEEDED_BLOCK_STRATEGY else None,
        "unit": "complete_shot",
        "sensitivity_blocks": (
            "five contiguous acquisition-order blocks"
            if strategy == SEEDED_BLOCK_STRATEGY else None
        ),
        "fractions_target": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "n_shots": n_shot,
        "shots_per_split": {name: int(counts[name]) for name in SPLIT_NAMES},
        "manifest_sha256": digest,
    }
    return manifest, metadata
