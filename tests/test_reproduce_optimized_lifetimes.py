"""Frozen orchestration tests for the optimized 2026-07-31 reproduction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce_optimized_lifetimes as reproduce  # noqa: E402


def test_optimized_reproduction_has_one_frozen_raw_to_result_order():
    steps = reproduce.build_steps(
        bootstrap=1000,
        seed=20260731,
        output=reproduce.RESULT_PATH,
    )
    assert len(steps) == 13
    assert [step.name.split()[0] for step in steps[:5]] == ["audit"] * 5
    assert [step.name.split()[0] for step in steps[5:10]] == ["export"] * 5
    assert steps[-3].name == "validate five-run geometry"
    assert steps[-2].name == "freeze five-run background methods"
    assert steps[-1].arguments[-1] == "--require-clean-provenance"
    assert steps[-1].arguments.count("--bootstrap") == 1


def test_reviewed_reproduction_rejects_development_bootstrap_count():
    with pytest.raises(ValueError, match="requires 1000"):
        reproduce.build_steps(
            bootstrap=100,
            seed=1,
            output=reproduce.RESULT_PATH,
        )


def test_result_validation_requires_clean_commit_binding(tmp_path):
    commit = "a" * 40
    payload = {
        "bootstrap_replicates": 1000,
        "provenance": {
            "analysis_code_commit": commit,
            "publishable_clean_provenance": True,
        },
        "prerequisite_evidence": {
            "timing_and_compiled_commands_verified": True,
            "geometry_validated": {"dark": True, "bright": True},
            "background_frozen": {"dark": True, "bright": True},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert reproduce.validate_result(path, bootstrap=1000, commit=commit) == payload
    payload["provenance"]["publishable_clean_provenance"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(reproduce.ReproductionError, match="not marked"):
        reproduce.validate_result(path, bootstrap=1000, commit=commit)
