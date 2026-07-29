"""Run the complete public analysis pipeline in a deterministic order.

This is a thin, fail-fast orchestration layer over the existing standalone
commands. Scientific implementation remains in package modules and the
individual scripts; no cleanup, mutating Git operation, or synthetic-data
fallback is performed here. Read-only Git checks protect reviewed provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluorescence_inference.config import ConfigError, load_config  # noqa: E402
from fluorescence_inference.publication_verification import (  # noqa: E402
    PUBLIC_ASSET_PATHS,
    PublicationVerificationError,
    build_publication_manifest,
    verify_reviewed_artifacts,
    write_publication_manifest,
)


PAIRED_CONFIG = "configs/paired_100ms.yaml"
DARK_CONFIG = "configs/dark_hold_50ms_20260728_0044.yaml"
BRIGHT_CONFIG = "configs/bright_wait_50ms_20260728_0050.yaml"
PUBLIC_RESULT = ROOT / "reports" / "loss_sweep_results.json"
PUBLICATION_MANIFEST = ROOT / "reports" / "publication_manifest.json"
PAIRED_ASSET_METADATA = ROOT / "assets" / "readme" / "asset_metadata.json"
CODE_GUARD_PATHS = (
    "src",
    "scripts",
    "configs",
    "pyproject.toml",
    # Publication Commit B may repair this non-scientific verifier without
    # changing the analysis implementation or any generator it launches.
    ":(exclude)scripts/reproduce_all.py",
)

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
class PublicationTargets:
    """Scientific result and asset destinations for publish or preview mode."""

    result: Path
    paired_assets_dir: Path | None = None
    sweep_assets_dir: Path | None = None


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


def default_publication_targets() -> PublicationTargets:
    """Return tracked reviewed outputs used by the public landing page."""
    return PublicationTargets(result=PUBLIC_RESULT)


def _paired_asset_arguments(
    targets: PublicationTargets,
    *,
    paired_config: str | Path = PAIRED_CONFIG,
) -> tuple[str, ...]:
    arguments = [
        "scripts/generate_readme_assets.py",
        "--config",
        _argument_path(Path(paired_config)),
    ]
    if targets.paired_assets_dir is not None:
        arguments.extend(
            ("--output-dir", _argument_path(targets.paired_assets_dir))
        )
    return tuple(arguments)


def _analysis_arguments(
    *,
    bootstrap: int,
    seed: int,
    paths: OutputPaths,
    targets: PublicationTargets,
    analysis_code_commit: str | None,
    require_clean_provenance: bool,
    dark_config: str | Path = DARK_CONFIG,
    bright_config: str | Path = BRIGHT_CONFIG,
) -> tuple[str, ...]:
    arguments = [
        "scripts/analyze_loss_sweeps.py",
        "--dark-config",
        _argument_path(Path(dark_config)),
        "--bright-config",
        _argument_path(Path(bright_config)),
        "--geometry-report",
        _argument_path(paths.geometry_report),
        "--background-report",
        _argument_path(paths.background_report),
        "--dark-command-audit",
        _argument_path(paths.dark_command_audit),
        "--bright-command-audit",
        _argument_path(paths.bright_command_audit),
        "--output",
        _argument_path(targets.result),
    ]
    if analysis_code_commit is not None:
        arguments.extend(("--analysis-code-commit", analysis_code_commit))
    if require_clean_provenance:
        arguments.append("--require-clean-provenance")
    arguments.extend(("--bootstrap", str(bootstrap), "--seed", str(seed)))
    return tuple(arguments)


def _sweep_asset_arguments(
    *,
    paths: OutputPaths,
    targets: PublicationTargets,
) -> tuple[str, ...]:
    arguments = [
        "scripts/generate_loss_sweep_assets.py",
        "--results",
        _argument_path(targets.result),
        "--v0-qc",
        _argument_path(paths.v0_qc),
    ]
    if targets.sweep_assets_dir is not None:
        arguments.extend(
            ("--output-dir", _argument_path(targets.sweep_assets_dir))
        )
    return tuple(arguments)


def build_generation_steps(
    *,
    bootstrap: int,
    seed: int,
    paths: OutputPaths | None = None,
    targets: PublicationTargets | None = None,
    analysis_code_commit: str | None = None,
    require_clean_provenance: bool = False,
) -> tuple[Step, ...]:
    """Return the frozen generation order with configured cross-stage paths."""
    paths = default_output_paths() if paths is None else paths
    targets = default_publication_targets() if targets is None else targets
    return (
        *build_prerequisite_steps(),
        Step(
            "analyze loss sweeps",
            _analysis_arguments(
                bootstrap=bootstrap,
                seed=seed,
                paths=paths,
                targets=targets,
                analysis_code_commit=analysis_code_commit,
                require_clean_provenance=require_clean_provenance,
            ),
        ),
        Step(
            "generate paired-readout assets",
            _paired_asset_arguments(targets),
        ),
        Step(
            "generate loss-sweep assets",
            _sweep_asset_arguments(
                paths=paths,
                targets=targets,
            ),
        ),
    )


def build_prerequisite_steps() -> tuple[Step, ...]:
    """Regenerate every ignored input consumed by public result generation."""
    return build_configured_prerequisite_steps(
        paired_config=PAIRED_CONFIG,
        dark_config=DARK_CONFIG,
        bright_config=BRIGHT_CONFIG,
    )


def build_configured_prerequisite_steps(
    *,
    paired_config: str | Path,
    dark_config: str | Path,
    bright_config: str | Path,
) -> tuple[Step, ...]:
    """Build preprocessing stages for explicit frozen configuration files."""
    paired = _argument_path(Path(paired_config))
    dark = _argument_path(Path(dark_config))
    bright = _argument_path(Path(bright_config))
    return (
        Step(
            "audit paired readout",
            ("scripts/audit_source_data.py", "--config", paired),
        ),
        Step(
            "export paired readout",
            ("scripts/export_processed_dataset.py", "--config", paired),
        ),
        Step(
            "validate paired-readout geometry",
            ("scripts/validate_site_geometry.py", "--config", paired),
        ),
        Step(
            "compare paired-readout backgrounds",
            ("scripts/compare_background_methods.py", "--config", paired),
        ),
        Step(
            "audit switch-off hold sweep",
            ("scripts/audit_source_data.py", "--config", dark),
        ),
        Step(
            "audit bright-wait sweep",
            ("scripts/audit_source_data.py", "--config", bright),
        ),
        Step(
            "export switch-off hold sweep",
            (
                "scripts/export_processed_dataset.py",
                "--config",
                dark,
                "--no-qc",
            ),
        ),
        Step(
            "export bright-wait sweep",
            (
                "scripts/export_processed_dataset.py",
                "--config",
                bright,
                "--no-qc",
            ),
        ),
        Step(
            "validate loss-sweep geometry",
            (
                "scripts/validate_sweep_geometry.py",
                "--config",
                dark,
                "--config",
                bright,
            ),
        ),
        Step(
            "compare loss-sweep backgrounds",
            (
                "scripts/compare_loss_sweep_backgrounds.py",
                "--config",
                dark,
                "--config",
                bright,
            ),
        ),
    )


def build_verification_steps(
    *,
    bootstrap: int,
    seed: int,
    paths: OutputPaths,
    targets: PublicationTargets,
    analysis_code_commit: str,
    paired_config: str | Path = PAIRED_CONFIG,
    dark_config: str | Path = DARK_CONFIG,
    bright_config: str | Path = BRIGHT_CONFIG,
) -> tuple[Step, ...]:
    """Regenerate ignored prerequisites, then isolated reviewed candidates."""
    return (
        *build_configured_prerequisite_steps(
            paired_config=paired_config,
            dark_config=dark_config,
            bright_config=bright_config,
        ),
        Step(
            "generate candidate loss-sweep result",
            _analysis_arguments(
                bootstrap=bootstrap,
                seed=seed,
                paths=paths,
                targets=targets,
                analysis_code_commit=analysis_code_commit,
                require_clean_provenance=True,
                dark_config=dark_config,
                bright_config=bright_config,
            ),
        ),
        Step(
            "generate candidate paired-readout assets",
            _paired_asset_arguments(
                targets,
                paired_config=paired_config,
            ),
        ),
        Step(
            "generate candidate loss-sweep assets",
            _sweep_asset_arguments(paths=paths, targets=targets),
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


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReproductionError(f"Git preflight failed: {exc}") from exc


def _read_git_checkout_bytes(commit: str, relative_path: str) -> bytes:
    """Read Commit-A content with this environment's checkout filters."""
    try:
        result = subprocess.run(
            [
                "git",
                "cat-file",
                "--filters",
                f"--path={relative_path}",
                f"{commit}:{relative_path}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=False,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReproductionError(
            f"cannot read reviewed config blob {relative_path}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReproductionError(
            f"cannot read reviewed config blob {relative_path}: {detail}"
        )
    return bytes(result.stdout)


BlobReader = Callable[[str, str], bytes]


def materialize_reviewed_configs(
    *,
    analysis_code_commit: str,
    destination: Path,
    blob_reader: BlobReader | None = None,
) -> dict[str, Path]:
    """Write exact Commit-A config blobs for provenance-stable verification."""
    reader = _read_git_checkout_bytes if blob_reader is None else blob_reader
    destination.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Path] = {}
    for name, relative in (
        ("paired", PAIRED_CONFIG),
        ("dark", DARK_CONFIG),
        ("bright", BRIGHT_CONFIG),
    ):
        target = destination / Path(relative).name
        target.write_bytes(reader(analysis_code_commit, relative))
        materialized[name] = target
    return materialized


def require_clean_worktree() -> None:
    """Fail before publication work when tracked or untracked state is dirty."""
    result = _run_git(("status", "--porcelain"))
    if result.returncode != 0:
        raise ReproductionError(
            f"cannot inspect Git status: {result.stderr.strip()}"
        )
    if result.stdout.strip():
        changed = [line for line in result.stdout.splitlines() if line.strip()]
        raise ReproductionError(
            "publication reproduction requires a clean worktree; "
            f"changed paths={changed}"
        )


def resolve_head_commit() -> str:
    """Return the full current revision used as the analysis-code commit."""
    result = _run_git(("rev-parse", "--verify", "HEAD^{commit}"))
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise ReproductionError(
            f"cannot resolve the current Git commit: {result.stderr.strip()}"
        )
    return commit


def require_analysis_code_unchanged(analysis_code_commit: str) -> None:
    """Require final-branch runtime analysis and config code to match Commit A."""
    commit = _run_git(
        ("rev-parse", "--verify", f"{analysis_code_commit}^{{commit}}")
    )
    resolved = commit.stdout.strip()
    if commit.returncode != 0 or len(resolved) != 40:
        raise ReproductionError(
            "publication manifest analysis_code_commit is not a Git commit"
        )
    comparison = _run_git(
        (
            "diff",
            "--quiet",
            resolved,
            "HEAD",
            "--",
            *CODE_GUARD_PATHS,
        )
    )
    if comparison.returncode == 1:
        names = _run_git(
            (
                "diff",
                "--name-only",
                resolved,
                "HEAD",
                "--",
                *CODE_GUARD_PATHS,
            )
        )
        paths = [line for line in names.stdout.splitlines() if line]
        raise ReproductionError(
            "analysis code differs from the reviewed analysis commit; "
            f"changed paths={paths}"
        )
    if comparison.returncode != 0:
        raise ReproductionError(
            "cannot compare final analysis code to analysis_code_commit: "
            f"{comparison.stderr.strip()}"
        )


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


def write_reviewed_manifest(
    manifest_path: Path = PUBLICATION_MANIFEST,
) -> dict[str, object]:
    """Build the external reviewed manifest after a certified clean run."""
    manifest = build_publication_manifest(
        root=ROOT,
        result_path=PUBLIC_RESULT,
        paired_asset_metadata_path=PAIRED_ASSET_METADATA,
    )
    write_publication_manifest(manifest_path, manifest)
    print(
        "manifest ... "
        f"{_argument_path(manifest_path)} "
        f"(analysis_code_commit={manifest['analysis_code_commit']})",
        flush=True,
    )
    return manifest


def _load_manifest_settings(
    manifest_path: Path,
) -> tuple[str, int, int]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionError(
            f"cannot read publication manifest ({Path(manifest_path).name}): {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ReproductionError("publication manifest must be a JSON object")
    analysis_commit = manifest.get("analysis_code_commit")
    bootstrap = manifest.get("bootstrap_replicates")
    seeds = manifest.get("seeds", {})
    seed = seeds.get("primary_analysis") if isinstance(seeds, dict) else None
    if not isinstance(analysis_commit, str) or len(analysis_commit) != 40:
        raise ReproductionError(
            "publication manifest has no full analysis_code_commit"
        )
    if not isinstance(bootstrap, int) or bootstrap < 100:
        raise ReproductionError(
            "publication manifest has no valid bootstrap_replicates"
        )
    if not isinstance(seed, int):
        raise ReproductionError(
            "publication manifest has no integer primary-analysis seed"
        )
    return analysis_commit, bootstrap, seed


def verify_reviewed(
    *,
    paths: OutputPaths,
    manifest_path: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Regenerate temporary candidates and compare them to reviewed outputs."""
    analysis_commit, bootstrap, seed = _load_manifest_settings(manifest_path)
    require_clean_worktree()
    require_analysis_code_unchanged(analysis_commit)

    temporary_parent = ROOT / "reports" / "validation"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="verify_reviewed_",
        dir=temporary_parent,
    ) as temporary:
        candidate_root = Path(temporary)
        frozen_configs = materialize_reviewed_configs(
            analysis_code_commit=analysis_commit,
            destination=candidate_root / "frozen_configs",
        )
        paired_assets = candidate_root / "paired_assets"
        sweep_assets = candidate_root / "sweep_assets"
        candidate_result = candidate_root / "loss_sweep_results.json"
        targets = PublicationTargets(
            result=candidate_result,
            paired_assets_dir=paired_assets,
            sweep_assets_dir=sweep_assets,
        )
        steps = build_verification_steps(
            bootstrap=bootstrap,
            seed=seed,
            paths=paths,
            targets=targets,
            analysis_code_commit=analysis_commit,
            paired_config=frozen_configs["paired"],
            dark_config=frozen_configs["dark"],
            bright_config=frozen_configs["bright"],
        )
        execute_steps(steps, environment=environment)
        candidate_assets = {
            PUBLIC_ASSET_PATHS[0]: paired_assets
            / "fluorescence_inference_overview.gif",
            PUBLIC_ASSET_PATHS[1]: sweep_assets / "sequence_design.png",
            PUBLIC_ASSET_PATHS[2]: sweep_assets / "loss_sweep_overview.png",
            PUBLIC_ASSET_PATHS[3]: sweep_assets
            / "background_drift_sweeps.png",
        }
        summary = verify_reviewed_artifacts(
            reviewed_root=ROOT,
            reviewed_result_path=PUBLIC_RESULT,
            candidate_result_path=candidate_result,
            manifest_path=manifest_path,
            candidate_assets=candidate_assets,
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


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
    parser.add_argument(
        "--verify-reviewed",
        action="store_true",
        help=(
            "generate temporary candidates and compare them to the reviewed "
            "result and manifest without overwriting either"
        ),
    )
    parser.add_argument(
        "--publication-manifest",
        type=Path,
        default=PUBLICATION_MANIFEST,
        help="reviewed publication manifest used by --verify-reviewed",
    )
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.verify_reviewed and args.dry_run:
        parser.error("--verify-reviewed cannot be combined with --dry-run")
    if args.verify_reviewed and args.skip_tests:
        parser.error("--verify-reviewed does not accept --skip-tests")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = resolve_output_paths()
        manifest_path = args.publication_manifest
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        environment = deterministic_environment()
        if args.verify_reviewed:
            verify_reviewed(
                paths=paths,
                manifest_path=manifest_path,
                environment=environment,
            )
            return 0

        if not args.dry_run:
            require_clean_worktree()
        analysis_code_commit = resolve_head_commit()
        generation_steps = build_generation_steps(
            bootstrap=args.bootstrap,
            seed=args.seed,
            paths=paths,
            analysis_code_commit=analysis_code_commit,
            require_clean_provenance=True,
        )
        if args.dry_run:
            for step in generation_steps:
                print(step.display_command)
            if not args.skip_tests:
                print(test_step().display_command)
            return 0

        execute_steps(generation_steps, environment=environment)
        if not args.skip_tests:
            execute_steps((test_step(),), environment=environment)
            write_reviewed_manifest(manifest_path)
    except (
        ConfigError,
        PublicationVerificationError,
        ReproductionError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
