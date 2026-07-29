"""Filesystem-level contracts for reviewed publication verification."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from fluorescence_inference.publication_verification import (
    COMMAND_AUDIT_RUNTIME_PATH_STRINGS,
    PROVENANCE_ONLY_PATH_STRINGS,
    PUBLIC_ASSET_PATHS,
    REQUIRED_SEED_KEYS,
    PublicationVerificationError,
    build_publication_manifest,
    command_audit_scientific_sha256,
    verify_command_audit_files,
    verify_reviewed_artifacts,
    write_publication_manifest,
)


def _load_analysis_script(repo_root: Path):
    path = repo_root / "scripts" / "analyze_loss_sweeps.py"
    spec = importlib.util.spec_from_file_location(
        "_test_publication_analysis_script", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(commit: str = "a" * 40) -> dict:
    return {
        "analysis_version": "test-analysis-v1",
        "cross_dataset_comparison": {
            "observed_minus_predicted_gap_percentage_points": 10.25,
        },
        "provenance": {
            "analysis_code_commit": "1" * 40,
            "inference_repository": {
                "commit": commit,
                "branch": "analysis-worktree",
                "dirty": False,
            },
            "dark_hold": {
                "config_sha256": "2" * 64,
                "input_manifest": {"manifest_sha256": "3" * 64},
            },
            "bright_wait": {
                "config_sha256": "4" * 64,
                "input_manifest": {"manifest_sha256": "5" * 64},
            },
            "geometry_report_sha256": "6" * 64,
            "background_report_sha256": "7" * 64,
            "dark_command_audit_sha256": "8" * 64,
            "bright_command_audit_sha256": "9" * 64,
            "lab_analysis_repository_commit": "b" * 40,
            "model_version": "test-analysis-v1",
            "random_seed": 17,
            "bootstrap_replicates": 100,
            "seeds": {
                name: 17 + index
                for index, name in enumerate(REQUIRED_SEED_KEYS)
            },
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_command_audit(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_audit(commit: str, branch: str) -> dict:
    return {
        "provenance": {
            "git": {
                "commit": commit,
                "dirty": False,
                "branch": branch,
            }
        },
        "aggregate": {
            "n_shots_with_all_frames": 10,
            "exposure_ms_values": [50.0],
        },
        "command_audit": {
            "all_shots_passed": True,
            "switch_states_verified_every_shot": True,
            "dds_amplitude_constant_every_shot": True,
            "max_abs_trigger_edge_error_s": 1e-9,
        },
    }


def _publication_fixture(root: Path):
    reviewed_result = root / "reports" / "loss_sweep_results.json"
    paired_metadata = root / "assets" / "readme" / "asset_metadata.json"
    reviewed = _result()
    reviewed_audits: dict[str, Path] = {}
    for label, field in (
        ("dark_hold", "dark_command_audit_sha256"),
        ("bright_wait", "bright_command_audit_sha256"),
    ):
        path = root / "reviewed_audits" / f"{label}.json"
        reviewed["provenance"][field] = _write_command_audit(
            path,
            _command_audit("a" * 40, "analysis-worktree"),
        )
        reviewed_audits[label] = path
    _write_json(reviewed_result, reviewed)
    _write_json(
        paired_metadata,
        {
            "provenance": {
                "config": {"config_sha256": "c" * 64},
                "inputs": {"manifest_sha256": "d" * 64},
            }
        },
    )
    for index, relative in enumerate(PUBLIC_ASSET_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"reviewed-asset-{index}".encode("ascii"))

    manifest = build_publication_manifest(
        root=root,
        result_path=reviewed_result,
        paired_asset_metadata_path=paired_metadata,
    )
    manifest_path = root / "reports" / "publication_manifest.json"
    write_publication_manifest(manifest_path, manifest)

    candidate_root = root / "candidate"
    candidate_result = candidate_root / "loss_sweep_results.json"
    candidate = _result(commit="e" * 40)
    candidate["provenance"]["inference_repository"]["branch"] = (
        "publication-branch"
    )
    candidate_audits: dict[str, Path] = {}
    for label, field in (
        ("dark_hold", "dark_command_audit_sha256"),
        ("bright_wait", "bright_command_audit_sha256"),
    ):
        path = candidate_root / "audits" / f"{label}.json"
        candidate["provenance"][field] = _write_command_audit(
            path,
            _command_audit("e" * 40, "publication-branch"),
        )
        candidate_audits[label] = path
    _write_json(candidate_result, candidate)
    candidate_assets: dict[str, Path] = {}
    for relative in PUBLIC_ASSET_PATHS:
        candidate_path = candidate_root / Path(relative).name
        shutil.copyfile(root / relative, candidate_path)
        candidate_assets[relative] = candidate_path
    return (
        reviewed_result,
        candidate_result,
        manifest_path,
        candidate_assets,
        reviewed_audits,
        candidate_audits,
    )


def test_reviewed_verification_detects_scientific_payload_mutation(scratch):
    (
        reviewed_result,
        candidate_result,
        manifest_path,
        candidate_assets,
        _,
        candidate_audits,
    ) = _publication_fixture(scratch)

    summary = verify_reviewed_artifacts(
        reviewed_root=scratch,
        reviewed_result_path=reviewed_result,
        candidate_result_path=candidate_result,
        manifest_path=manifest_path,
        candidate_assets=candidate_assets,
        candidate_command_audits=candidate_audits,
    )
    assert summary["n_assets_verified"] == len(PUBLIC_ASSET_PATHS)
    assert summary["provenance_only_exclusions"] == list(
        PROVENANCE_ONLY_PATH_STRINGS
    )

    mutated = json.loads(candidate_result.read_text(encoding="utf-8"))
    mutated["cross_dataset_comparison"][
        "observed_minus_predicted_gap_percentage_points"
    ] += 0.01
    _write_json(candidate_result, mutated)
    with pytest.raises(PublicationVerificationError) as caught:
        verify_reviewed_artifacts(
            reviewed_root=scratch,
            reviewed_result_path=reviewed_result,
            candidate_result_path=candidate_result,
            manifest_path=manifest_path,
            candidate_assets=candidate_assets,
            candidate_command_audits=candidate_audits,
        )
    message = str(caught.value)
    assert "scientific payload drift" in message
    assert (
        "cross_dataset_comparison."
        "observed_minus_predicted_gap_percentage_points"
    ) in message
    assert "reviewed=" in message
    assert "candidate=" in message


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("provenance", "inference_repository", "dirty"),
            True,
        ),
        (
            ("provenance", "dark_hold", "config_sha256"),
            "f" * 64,
        ),
        (
            ("provenance", "analysis_code_commit"),
            "0" * 40,
        ),
    ],
)
def test_only_declared_runtime_git_fields_are_excluded(
    scratch,
    path,
    replacement,
):
    (
        reviewed_result,
        candidate_result,
        manifest_path,
        candidate_assets,
        _,
        candidate_audits,
    ) = _publication_fixture(scratch)
    candidate = json.loads(candidate_result.read_text(encoding="utf-8"))
    target = candidate
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement
    _write_json(candidate_result, candidate)

    with pytest.raises(PublicationVerificationError):
        verify_reviewed_artifacts(
            reviewed_root=scratch,
            reviewed_result_path=reviewed_result,
            candidate_result_path=candidate_result,
            manifest_path=manifest_path,
            candidate_assets=candidate_assets,
            candidate_command_audits=candidate_audits,
        )


def test_manifest_contains_no_machine_path_or_self_hash(scratch):
    reviewed_result, _, manifest_path, _, _, _ = _publication_fixture(scratch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = manifest_path.read_text(encoding="utf-8")
    assert str(scratch) not in serialized
    assert "manifest_sha256" not in manifest
    assert manifest["result"]["path"] == (
        reviewed_result.relative_to(scratch).as_posix()
    )
    assert manifest["verification"]["scientific_payload_excluded_paths"] == list(
        PROVENANCE_ONLY_PATH_STRINGS
    )


def test_command_audit_runtime_git_identity_is_canonicalized(scratch):
    reviewed = _command_audit("a" * 40, "HEAD")
    candidate = _command_audit("b" * 40, "publication-branch")
    assert command_audit_scientific_sha256(
        reviewed
    ) == command_audit_scientific_sha256(candidate)

    reviewed_path = scratch / "reviewed.json"
    candidate_path = scratch / "candidate.json"
    _write_command_audit(reviewed_path, reviewed)
    _write_command_audit(candidate_path, candidate)
    summary = verify_command_audit_files(
        reviewed_path=reviewed_path,
        candidate_path=candidate_path,
        label="dark_hold",
    )
    assert summary["reviewed_canonical_sha256"] == (
        summary["candidate_canonical_sha256"]
    )
    assert summary["runtime_provenance_exclusions"] == list(
        COMMAND_AUDIT_RUNTIME_PATH_STRINGS
    )


def test_scientific_command_audit_mutation_fails_verification(scratch):
    (
        reviewed_result,
        candidate_result,
        manifest_path,
        candidate_assets,
        _,
        candidate_audits,
    ) = _publication_fixture(scratch)
    candidate = json.loads(candidate_result.read_text(encoding="utf-8"))
    dark_path = candidate_audits["dark_hold"]
    dark = json.loads(dark_path.read_text(encoding="utf-8"))
    dark["command_audit"]["dds_amplitude_constant_every_shot"] = False
    candidate["provenance"]["dark_command_audit_sha256"] = (
        _write_command_audit(dark_path, dark)
    )
    _write_json(candidate_result, candidate)

    with pytest.raises(
        PublicationVerificationError,
        match="reviewed reconstruction drift",
    ):
        verify_reviewed_artifacts(
            reviewed_root=scratch,
            reviewed_result_path=reviewed_result,
            candidate_result_path=candidate_result,
            manifest_path=manifest_path,
            candidate_assets=candidate_assets,
            candidate_command_audits=candidate_audits,
        )


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_missing_or_malformed_command_audit_fails(scratch, failure):
    (
        reviewed_result,
        candidate_result,
        manifest_path,
        candidate_assets,
        _,
        candidate_audits,
    ) = _publication_fixture(scratch)
    dark_path = candidate_audits["dark_hold"]
    if failure == "missing":
        dark_path.unlink()
    else:
        dark_path.write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(PublicationVerificationError):
        verify_reviewed_artifacts(
            reviewed_root=scratch,
            reviewed_result_path=reviewed_result,
            candidate_result_path=candidate_result,
            manifest_path=manifest_path,
            candidate_assets=candidate_assets,
            candidate_command_audits=candidate_audits,
        )


def test_provenance_seeds_are_read_from_serialized_analysis_blocks(repo_root):
    analysis_script = _load_analysis_script(repo_root)
    analysis = {
        "dark_hold": {
            "operational_model": {
                "bootstrap": {"seed": 101},
                "whole_cycle_bootstrap_sensitivity": {"seed": 102},
                "bootstrap_multistart_diagnostic": {"seed": 103},
            }
        },
        "bright_wait": {
            "effective_model": {
                "bootstrap": {"seed": 201},
                "whole_cycle_bootstrap_sensitivity": {"seed": 202},
                "bootstrap_multistart_diagnostic": {"seed": 203},
                "post_wait_control": {
                    "alternative_model_sensitivity": {
                        "bootstrap": {"seed": 204}
                    }
                },
            }
        },
    }
    assert analysis_script._provenance_seeds(
        analysis, primary_seed=99
    ) == {
        "primary_analysis": 99,
        "dark_condition_bootstrap": 101,
        "bright_condition_bootstrap": 201,
        "dark_cycle_bootstrap": 102,
        "bright_cycle_bootstrap": 202,
        "dark_multistart_diagnostic": 103,
        "bright_multistart_diagnostic": 203,
        "bright_monotone_control_sensitivity": 204,
    }
