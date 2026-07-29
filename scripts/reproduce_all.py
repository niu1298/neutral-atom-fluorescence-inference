"""Run the complete public analysis pipeline in a deterministic order.

This is a thin, fail-fast orchestration layer over the existing standalone
commands. Scientific implementation remains in package modules and the
individual scripts; no cleanup, Git operation, or synthetic-data fallback is
performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import ConfigError, load_config  # noqa: E402


PAIRED_CONFIG = "configs/paired_100ms.yaml"
DARK_CONFIG = "configs/dark_hold_50ms_20260728_0044.yaml"
BRIGHT_CONFIG = "configs/bright_wait_50ms_20260728_0050.yaml"
PUBLIC_RESULT = ROOT / "reports" / "loss_sweep_results.json"

DETERMINISTIC_ENVIRONMENT = {
    "MPLBACKEND": "Agg",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class OutputPaths:
    """Configured reports consumed across otherwise independent stages."""

    geometry_report: Path
    background_report: Path
    dark_command_audit: Path
    bright_command_audit: Path
    v0_qc: Path


@dataclass(frozen=True)
class Step:
    """One independently runnable repository command."""

    name: str
    arguments: tuple[str, ...]

    @property
    def display_command(self) -> str:
        return " ".join(
            _quote_display(part) for part in ("python", *self.arguments)
        )


class ReproductionError(RuntimeError):
    """A configured input or pipeline stage failed."""


def _quote_display(part: str) -> str:
    if part and all(ch.isalnum() or ch in "._/\\:-" for ch in part):
        return part
    return json.dumps(part)


def _argument_path(path: Path) -> str:
    """Prefer repository-relative command arguments without changing targets."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def default_output_paths() -> OutputPaths:
    """Return the paths produced by the machine-independent example config."""
    reports = ROOT / "reports"
    return OutputPaths(
        geometry_report=(
            reports
            / "validation"
            / "loss_sweeps_geometry"
            / "geometry_validation.json"
        ),
        background_report=(
            reports
            / "validation"
            / "loss_sweeps_background"
            / "background_method_comparison.json"
        ),
        dark_command_audit=(
            reports / "audit" / "dark_hold_20260728_0044_source_audit.json"
        ),
        bright_command_audit=(
            reports / "audit" / "bright_wait_20260728_0050_source_audit.json"
        ),
        v0_qc=reports / "qc" / "qc_summary.json",
    )


def resolve_output_paths() -> OutputPaths:
    """Resolve cross-stage paths from all three local configurations."""
    paired = load_config(PAIRED_CONFIG)
    dark = load_config(DARK_CONFIG)
    bright = load_config(BRIGHT_CONFIG)
    dark_reports = dark.paths.reports_root
    bright_reports = bright.paths.reports_root
    paired_reports = paired.paths.reports_root
    return OutputPaths(
        geometry_report=(
            dark_reports
            / "validation"
            / "loss_sweeps_geometry"
            / "geometry_validation.json"
        ),
        background_report=(
            dark_reports
            / "validation"
            / "loss_sweeps_background"
            / "background_method_comparison.json"
        ),
        dark_command_audit=(
            dark_reports / "audit" / f"{dark.dataset_id}_source_audit.json"
        ),
        bright_command_audit=(
            bright_reports / "audit" / f"{bright.dataset_id}_source_audit.json"
        ),
        v0_qc=paired_reports / "qc" / "qc_summary.json",
    )


def build_generation_steps(
    *,
    bootstrap: int,
    seed: int,
    paths: OutputPaths | None = None,
) -> tuple[Step, ...]:
    """Return the frozen generation order with configured cross-stage paths."""
    paths = default_output_paths() if paths is None else paths
    return (
        Step(
            "audit paired readout",
            ("scripts/audit_source_data.py", "--config", PAIRED_CONFIG),
        ),
        Step(
            "export paired readout",
            ("scripts/export_processed_dataset.py", "--config", PAIRED_CONFIG),
        ),
        Step(
            "validate paired-readout geometry",
            ("scripts/validate_site_geometry.py", "--config", PAIRED_CONFIG),
        ),
        Step(
            "compare paired-readout backgrounds",
            ("scripts/compare_background_methods.py", "--config", PAIRED_CONFIG),
        ),
        Step(
            "generate paired-readout assets",
            ("scripts/generate_readme_assets.py", "--config", PAIRED_CONFIG),
        ),
        Step(
            "audit switch-off hold sweep",
            ("scripts/audit_source_data.py", "--config", DARK_CONFIG),
        ),
        Step(
            "audit bright-wait sweep",
            ("scripts/audit_source_data.py", "--config", BRIGHT_CONFIG),
        ),
        Step(
            "export switch-off hold sweep",
            (
                "scripts/export_processed_dataset.py",
                "--config",
                DARK_CONFIG,
                "--no-qc",
            ),
        ),
        Step(
            "export bright-wait sweep",
            (
                "scripts/export_processed_dataset.py",
                "--config",
                BRIGHT_CONFIG,
                "--no-qc",
            ),
        ),
        Step(
            "validate loss-sweep geometry",
            (
                "scripts/validate_sweep_geometry.py",
                "--config",
                DARK_CONFIG,
                "--config",
                BRIGHT_CONFIG,
            ),
        ),
        Step(
            "compare loss-sweep backgrounds",
            (
                "scripts/compare_loss_sweep_backgrounds.py",
                "--config",
                DARK_CONFIG,
                "--config",
                BRIGHT_CONFIG,
            ),
        ),
        Step(
            "analyze loss sweeps",
            (
                "scripts/analyze_loss_sweeps.py",
                "--geometry-report",
                _argument_path(paths.geometry_report),
                "--background-report",
                _argument_path(paths.background_report),
                "--dark-command-audit",
                _argument_path(paths.dark_command_audit),
                "--bright-command-audit",
                _argument_path(paths.bright_command_audit),
                "--output",
                _argument_path(PUBLIC_RESULT),
                "--bootstrap",
                str(bootstrap),
                "--seed",
                str(seed),
            ),
        ),
        Step(
            "generate loss-sweep assets",
            (
                "scripts/generate_loss_sweep_assets.py",
                "--results",
                _argument_path(PUBLIC_RESULT),
                "--v0-qc",
                _argument_path(paths.v0_qc),
            ),
        ),
    )


def test_step() -> Step:
    return Step("run complete test suite", ("-m", "pytest"))


def deterministic_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment with deterministic numeric settings."""
    environment = dict(os.environ if base is None else base)
    environment.update(DETERMINISTIC_ENVIRONMENT)
    return environment


Runner = Callable[..., subprocess.CompletedProcess]


def execute_steps(
    steps: Sequence[Step],
    *,
    environment: Mapping[str, str],
    runner: Runner = subprocess.run,
) -> None:
    """Run steps in order and stop at the first nonzero return code."""
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f"[{index:02d}/{total:02d}] {step.display_command}", flush=True)
        completed = runner(
            [sys.executable, *step.arguments],
            cwd=ROOT,
            env=dict(environment),
            check=False,
        )
        if completed.returncode != 0:
            raise ReproductionError(
                f"stage failed with exit code {completed.returncode}: "
                f"{step.display_command}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="shot-cluster bootstrap replicates for loss-sweep analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="random seed forwarded to loss-sweep analysis",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip the final complete pytest invocation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ordered configured commands without running them",
    )
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = resolve_output_paths()
        generation_steps = build_generation_steps(
            bootstrap=args.bootstrap,
            seed=args.seed,
            paths=paths,
        )
        if args.dry_run:
            for step in generation_steps:
                print(step.display_command)
            if not args.skip_tests:
                print(test_step().display_command)
            return 0

        environment = deterministic_environment()
        execute_steps(generation_steps, environment=environment)
        if not args.skip_tests:
            execute_steps((test_step(),), environment=environment)
    except (ConfigError, ReproductionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
