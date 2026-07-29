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
        "generate paired-readout assets",
        "audit switch-off hold sweep",
        "audit bright-wait sweep",
        "export switch-off hold sweep",
        "export bright-wait sweep",
        "validate loss-sweep geometry",
        "compare loss-sweep backgrounds",
        "analyze loss sweeps",
        "generate loss-sweep assets",
    ]
    assert steps[-2].arguments[-4:] == (
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
    analysis = steps[-2].arguments
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
    calls: list[tuple[str, ...]] = []

    def fake_execute(steps, **_kwargs):
        calls.append(tuple(step.name for step in steps))

    monkeypatch.setattr(reproduce, "execute_steps", fake_execute)
    assert reproduce.main([]) == 0
    assert calls[0][0] == "audit paired readout"
    assert calls[0][-1] == "generate loss-sweep assets"
    assert calls[1] == ("run complete test suite",)


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
