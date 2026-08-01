"""Run the optimized 2026-07-31 pipeline once from raw data to reviewed JSON.

This orchestration is intentionally separate from ``reproduce_all.py`` so the
reviewed 2026-07-28 benchmark remains unchanged and reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (ROOT / ".venv" / "Scripts" / "python.exe").resolve()
RESULT_PATH = ROOT / "reports" / "optimized_lifetime_results_20260731.json"
GEOMETRY_PATH = (
    ROOT
    / "reports"
    / "validation"
    / "optimized_lifetimes_geometry"
    / "geometry_validation.json"
)
BACKGROUND_PATH = (
    ROOT
    / "reports"
    / "validation"
    / "optimized_lifetimes_background"
    / "background_method_comparison.json"
)
EXECUTION_MANIFEST = (
    ROOT / "_scratch" / "optimized_final_reproduction" / "execution_manifest.json"
)
CONFIGS = (
    "configs/bright_lifetime_20260731_0090.yaml",
    "configs/dark_lifetime_20260731_0094.yaml",
    "configs/imaging_5frame_50ms_20260731_0113.yaml",
    "configs/imaging_5frame_100ms_20260731_0114.yaml",
    "configs/imaging_5frame_200ms_20260731_0115.yaml",
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
class Step:
    name: str
    arguments: tuple[str, ...]


class ReproductionError(RuntimeError):
    """The clean optimized reproduction failed a declared gate."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def build_steps(*, bootstrap: int, seed: int, output: Path) -> tuple[Step, ...]:
    """Return the single frozen raw-to-result execution order."""
    if bootstrap != 1000:
        raise ValueError("the reviewed optimized reproduction requires 1000 bootstraps")
    audits = tuple(
        Step(
            f"audit {Path(config).stem}",
            ("scripts/audit_source_data.py", "--config", config),
        )
        for config in CONFIGS
    )
    exports = tuple(
        Step(
            f"export {Path(config).stem}",
            (
                "scripts/export_processed_dataset.py",
                "--config",
                config,
                "--no-qc",
                "--quiet",
            ),
        )
        for config in CONFIGS
    )
    geometry_arguments: list[str] = ["scripts/validate_sweep_geometry.py"]
    background_arguments: list[str] = [
        "scripts/compare_loss_sweep_backgrounds.py"
    ]
    for config in CONFIGS:
        geometry_arguments.extend(("--config", config))
        background_arguments.extend(("--config", config))
    geometry_arguments.extend(("--output", _relative(GEOMETRY_PATH)))
    background_arguments.extend(("--output", _relative(BACKGROUND_PATH)))
    analysis_arguments = (
        "scripts/analyze_optimized_lifetimes.py",
        "--geometry-report",
        _relative(GEOMETRY_PATH),
        "--background-report",
        _relative(BACKGROUND_PATH),
        "--output",
        _relative(output),
        "--bootstrap",
        str(bootstrap),
        "--seed",
        str(seed),
        "--require-clean-provenance",
    )
    return (
        *audits,
        *exports,
        Step("validate five-run geometry", tuple(geometry_arguments)),
        Step("freeze five-run background methods", tuple(background_arguments)),
        Step("analyze optimized lifetimes", analysis_arguments),
    )


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def require_exact_repository_python() -> None:
    observed = Path(sys.executable).resolve()
    if observed != EXPECTED_PYTHON:
        raise ReproductionError(
            f"use the repository interpreter {EXPECTED_PYTHON}; got {observed}"
        )


def require_clean_worktree() -> str:
    status = _git("status", "--porcelain")
    if status.returncode != 0:
        raise ReproductionError(f"cannot inspect Git status: {status.stderr.strip()}")
    if status.stdout.strip():
        raise ReproductionError(
            "final optimized reproduction requires a clean worktree: "
            + status.stdout.strip().replace("\n", "; ")
        )
    head = _git("rev-parse", "HEAD")
    commit = head.stdout.strip()
    if head.returncode != 0 or len(commit) != 40:
        raise ReproductionError("cannot resolve the analysis-code commit")
    return commit


def validate_result(path: Path, *, bootstrap: int, commit: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    if payload.get("bootstrap_replicates") != bootstrap:
        raise ReproductionError("result bootstrap count does not match the command")
    if provenance.get("analysis_code_commit") != commit:
        raise ReproductionError("result does not bind to the clean analysis commit")
    if provenance.get("publishable_clean_provenance") is not True:
        raise ReproductionError("result is not marked as clean and publishable")
    evidence = payload.get("prerequisite_evidence", {})
    if evidence.get("timing_and_compiled_commands_verified") is not True:
        raise ReproductionError("timing/compiled-command result gate failed")
    if not all(evidence.get("geometry_validated", {}).values()):
        raise ReproductionError("one or more geometry result gates failed")
    if not all(evidence.get("background_frozen", {}).values()):
        raise ReproductionError("one or more background result gates failed")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    require_exact_repository_python()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    commit = require_clean_worktree()
    steps = build_steps(bootstrap=args.bootstrap, seed=args.seed, output=output)
    environment = dict(os.environ)
    environment.update(DETERMINISTIC_ENVIRONMENT)
    records = []
    started = time.time()
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}", flush=True)
        step_started = time.time()
        completed = subprocess.run(
            [str(EXPECTED_PYTHON), *step.arguments],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        records.append(
            {
                "name": step.name,
                "arguments": list(step.arguments),
                "returncode": int(completed.returncode),
                "elapsed_s": float(time.time() - step_started),
            }
        )
        if completed.returncode != 0:
            raise ReproductionError(
                f"step {index}/{len(steps)} failed: {step.name}"
            )
    validate_result(output, bootstrap=args.bootstrap, commit=commit)
    manifest = {
        "analysis_code_commit": commit,
        "python_executable": ".venv/Scripts/python.exe",
        "bootstrap_replicates": int(args.bootstrap),
        "random_seed": int(args.seed),
        "elapsed_s": float(time.time() - started),
        "steps": records,
        "result": {
            "path": _relative(output),
            "sha256": _sha256(output),
        },
    }
    EXECUTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXECUTION_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["result"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
