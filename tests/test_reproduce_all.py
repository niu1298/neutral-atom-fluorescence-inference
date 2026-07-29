"""Focused tests for the deterministic reproduction orchestrator."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce_all as reproduce  # noqa: E402


def _custom_paths(root: Path) -> reproduce.OutputPaths:
    reports = root / "configured reports"
    return reproduce.OutputPaths(
        geometry_report=reports / "geometry.json",
        background_report=reports / "background.json",
        dark_command_audit=reports / "dark audit.json",
        bright_command_audit=reports / "bright audit.json",
        v0_qc=reports / "v0 qc.json",
    )


def test_generation_order_preserves_the_standalone_workflow():
    steps = reproduce.build_generation_steps(
        bootstrap=1000,
        seed=20260728,
    )
    assert [step.name for step in steps] == [
        "audit paired readout",
        "export paired readout",
        "validate paired-readout geometry",
        "compare paired-readout backgrounds",
        "audit switch-off hold sweep",
        "audit bright-wait sweep",
        "export switch-off hold sweep",
        "export bright-wait sweep",
        "validate loss-sweep geometry",
        "compare loss-sweep backgrounds",
        "analyze loss sweeps",
        "generate paired-readout assets",
        "generate loss-sweep assets",
    ]
    assert steps[-3].arguments[-4:] == (
        "--bootstrap",
        "1000",
        "--seed",
        "20260728",
    )


def test_configured_report_paths_are_forwarded_across_stages(scratch):
    paths = _custom_paths(scratch)
    steps = reproduce.build_generation_steps(
        bootstrap=100,
        seed=7,
        paths=paths,
    )
    analysis = steps[-3].arguments
    assets = steps[-1].arguments
    for flag, expected in (
        ("--geometry-report", paths.geometry_report),
        ("--background-report", paths.background_report),
        ("--dark-command-audit", paths.dark_command_audit),
        ("--bright-command-audit", paths.bright_command_audit),
    ):
        assert analysis[analysis.index(flag) + 1] == reproduce._argument_path(
            expected
        )
    assert analysis[analysis.index("--output") + 1] == reproduce._argument_path(
        reproduce.PUBLIC_RESULT
    )
    assert assets[assets.index("--results") + 1] == reproduce._argument_path(
        reproduce.PUBLIC_RESULT
    )
    assert assets[assets.index("--v0-qc") + 1] == reproduce._argument_path(
        paths.v0_qc
    )


def test_workflow_contains_no_cleanup_or_git_commands():
    steps = reproduce.build_generation_steps(bootstrap=100, seed=7)
    programs = [step.arguments[0].lower() for step in steps]
    assert not any(
        token in program
        for program in programs
        for token in ("remove", "delete", "clean", "reset", "git")
    )


def test_code_guard_excludes_only_the_publication_orchestrator():
    exclusions = [
        path
        for path in reproduce.CODE_GUARD_PATHS
        if path.startswith(":(exclude)")
    ]
    assert exclusions == [":(exclude)scripts/reproduce_all.py"]
    assert {"src", "scripts", "configs", "pyproject.toml"} <= set(
        reproduce.CODE_GUARD_PATHS
    )


def test_execute_steps_is_ordered_and_fail_fast():
    steps = (
        reproduce.Step("first", ("scripts/first.py",)),
        reproduce.Step("second", ("scripts/second.py",)),
        reproduce.Step("must not run", ("scripts/third.py",)),
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=9 if len(calls) == 2 else 0)

    with pytest.raises(reproduce.ReproductionError, match="second.py"):
        reproduce.execute_steps(
            steps,
            environment={"PYTHONHASHSEED": "0"},
            runner=fake_runner,
        )

    assert [call[0][1] for call in calls] == [
        "scripts/first.py",
        "scripts/second.py",
    ]
    assert all(call[1]["cwd"] == ROOT for call in calls)
    assert all(call[1]["check"] is False for call in calls)


def test_dry_run_uses_resolved_paths_and_writes_nothing(
    monkeypatch,
    capsys,
    scratch,
):
    paths = _custom_paths(scratch)
    monkeypatch.setattr(reproduce, "resolve_output_paths", lambda: paths)

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("dry-run invoked a subprocess")

    monkeypatch.setattr(reproduce, "execute_steps", unexpected_execution)
    assert reproduce.main(["--dry-run", "--skip-tests"]) == 0
    output = capsys.readouterr().out
    assert "python scripts/audit_source_data.py" in output
    assert "reports/loss_sweep_results.json" in output
    assert "python -m pytest" not in output


def test_main_executes_generation_then_tests(monkeypatch):
    paths = reproduce.default_output_paths()
    monkeypatch.setattr(reproduce, "resolve_output_paths", lambda: paths)
    monkeypatch.setattr(reproduce, "require_clean_worktree", lambda: None)
    analysis_commit = "a" * 40
    monkeypatch.setattr(
        reproduce, "resolve_head_commit", lambda: analysis_commit
    )
    calls: list[tuple[str, ...]] = []
    executed_steps: list[tuple[reproduce.Step, ...]] = []
    manifests: list[Path] = []

    def fake_execute(steps, **_kwargs):
        executed_steps.append(tuple(steps))
        calls.append(tuple(step.name for step in steps))

    monkeypatch.setattr(reproduce, "execute_steps", fake_execute)
    monkeypatch.setattr(
        reproduce,
        "write_reviewed_manifest",
        lambda path: manifests.append(path),
    )
    assert reproduce.main([]) == 0
    assert calls[0][0] == "audit paired readout"
    assert calls[0][-1] == "generate loss-sweep assets"
    assert calls[1] == ("run complete test suite",)
    assert manifests == [reproduce.PUBLICATION_MANIFEST]
    analysis = next(
        step for step in executed_steps[0] if step.name == "analyze loss sweeps"
    )
    assert (
        analysis.arguments[
            analysis.arguments.index("--analysis-code-commit") + 1
        ]
        == analysis_commit
    )
    assert "--require-clean-provenance" in analysis.arguments


def test_manifest_is_not_written_when_tests_fail(monkeypatch):
    paths = reproduce.default_output_paths()
    monkeypatch.setattr(reproduce, "resolve_output_paths", lambda: paths)
    monkeypatch.setattr(reproduce, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(reproduce, "resolve_head_commit", lambda: "a" * 40)
    calls = 0

    def fake_execute(_steps, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise reproduce.ReproductionError("test stage failed")

    manifest_called = False

    def unexpected_manifest(_path):
        nonlocal manifest_called
        manifest_called = True

    monkeypatch.setattr(reproduce, "execute_steps", fake_execute)
    monkeypatch.setattr(
        reproduce, "write_reviewed_manifest", unexpected_manifest
    )
    assert reproduce.main([]) == 2
    assert calls == 2
    assert manifest_called is False


def test_verify_reviewed_branches_before_publish_generation(monkeypatch):
    paths = reproduce.default_output_paths()
    monkeypatch.setattr(reproduce, "resolve_output_paths", lambda: paths)
    calls: list[tuple[reproduce.OutputPaths, Path]] = []

    def fake_verify(*, paths, manifest_path, environment):
        assert environment["PYTHONHASHSEED"] == "0"
        calls.append((paths, manifest_path))

    def unexpected_generation(**_kwargs):
        raise AssertionError("verification built publish-mode generation steps")

    monkeypatch.setattr(reproduce, "verify_reviewed", fake_verify)
    monkeypatch.setattr(
        reproduce, "build_generation_steps", unexpected_generation
    )
    assert reproduce.main(["--verify-reviewed"]) == 0
    assert calls == [(paths, reproduce.PUBLICATION_MANIFEST)]


def test_environment_overrides_do_not_mutate_the_input():
    base = {"EXAMPLE": "preserved", "OMP_NUM_THREADS": "99"}
    environment = reproduce.deterministic_environment(base)
    assert base["OMP_NUM_THREADS"] == "99"
    assert environment["EXAMPLE"] == "preserved"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"


def test_bootstrap_minimum_is_enforced():
    with pytest.raises(SystemExit):
        reproduce.parse_args(["--bootstrap", "99", "--dry-run"])


def test_verification_regenerates_prerequisites_and_isolates_candidates(
    scratch,
):
    paths = _custom_paths(scratch)
    frozen = {
        "paired": scratch / "frozen" / "paired.yaml",
        "dark": scratch / "frozen" / "dark.yaml",
        "bright": scratch / "frozen" / "bright.yaml",
    }
    targets = reproduce.PublicationTargets(
        result=scratch / "candidate" / "result.json",
        paired_assets_dir=scratch / "candidate" / "paired",
        sweep_assets_dir=scratch / "candidate" / "sweeps",
    )
    commit = "a" * 40
    steps = reproduce.build_verification_steps(
        bootstrap=100,
        seed=17,
        paths=paths,
        targets=targets,
        analysis_code_commit=commit,
        paired_config=frozen["paired"],
        dark_config=frozen["dark"],
        bright_config=frozen["bright"],
    )
    assert [step.name for step in steps] == [
        "audit paired readout",
        "export paired readout",
        "validate paired-readout geometry",
        "compare paired-readout backgrounds",
        "audit switch-off hold sweep",
        "audit bright-wait sweep",
        "export switch-off hold sweep",
        "export bright-wait sweep",
        "validate loss-sweep geometry",
        "compare loss-sweep backgrounds",
        "generate candidate loss-sweep result",
        "generate candidate paired-readout assets",
        "generate candidate loss-sweep assets",
    ]
    assert steps[:10] == reproduce.build_configured_prerequisite_steps(
        paired_config=frozen["paired"],
        dark_config=frozen["dark"],
        bright_config=frozen["bright"],
    )
    analysis, paired, sweeps = (step.arguments for step in steps[-3:])
    assert steps[0].arguments[-1] == reproduce._argument_path(frozen["paired"])
    assert steps[4].arguments[-1] == reproduce._argument_path(frozen["dark"])
    assert steps[5].arguments[-1] == reproduce._argument_path(frozen["bright"])
    assert analysis[analysis.index("--dark-config") + 1] == (
        reproduce._argument_path(frozen["dark"])
    )
    assert analysis[analysis.index("--bright-config") + 1] == (
        reproduce._argument_path(frozen["bright"])
    )
    assert analysis[analysis.index("--output") + 1] == reproduce._argument_path(
        targets.result
    )
    assert analysis[analysis.index("--analysis-code-commit") + 1] == commit
    assert "--require-clean-provenance" in analysis
    assert paired[paired.index("--output-dir") + 1] == reproduce._argument_path(
        targets.paired_assets_dir
    )
    assert paired[paired.index("--config") + 1] == reproduce._argument_path(
        frozen["paired"]
    )
    assert sweeps[sweeps.index("--output-dir") + 1] == reproduce._argument_path(
        targets.sweep_assets_dir
    )
    assert reproduce._argument_path(reproduce.PUBLIC_RESULT) not in analysis


def test_reviewed_configs_use_checkout_filtered_commit_bytes(scratch):
    calls: list[tuple[str, str]] = []

    def fake_blob_reader(commit, relative):
        calls.append((commit, relative))
        return f"{relative}\n".encode("utf-8")

    commit = "c" * 40
    configs = reproduce.materialize_reviewed_configs(
        analysis_code_commit=commit,
        destination=scratch / "frozen",
        blob_reader=fake_blob_reader,
    )
    assert calls == [
        (commit, reproduce.PAIRED_CONFIG),
        (commit, reproduce.DARK_CONFIG),
        (commit, reproduce.BRIGHT_CONFIG),
    ]
    for name, relative in (
        ("paired", reproduce.PAIRED_CONFIG),
        ("dark", reproduce.DARK_CONFIG),
        ("bright", reproduce.BRIGHT_CONFIG),
    ):
        assert configs[name].name == Path(relative).name
        assert configs[name].read_bytes() == f"{relative}\n".encode("utf-8")


def test_verify_reviewed_rejects_publish_or_test_flags():
    for arguments in (
        ["--verify-reviewed", "--dry-run"],
        ["--verify-reviewed", "--skip-tests"],
    ):
        with pytest.raises(SystemExit):
            reproduce.parse_args(arguments)
