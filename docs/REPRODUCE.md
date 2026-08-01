# Reproduction

This repository preserves two independently reproducible analysis generations:
the optimized 2026-07-31 lifetime/repeated-imaging result and the earlier
2026-07-28 pre-optimization benchmark.

## Environment

Python 3.10 or newer is required. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item configs\local.example.toml configs\local.toml
```

Edit only `configs/local.toml` to identify the raw-data root, read-only
reference-analysis checkout, and local processed/report roots. Local paths are
redacted from tracked outputs. Raw shots and full processed tables remain
ignored and are never modified by the pipeline.

## Optimized one-command reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_optimized_lifetimes.py
```

The default command uses seed `20260731` and 1,000 bootstrap replicates. It
requires a clean scientific-code state and runs these fail-fast stages:

1. audit all five raw acquisitions and compiled sequence commands;
2. export schema-4.0 standardized tables and sidecars;
3. fit and validate independent training-only geometry for every run;
4. compare and freeze training-only background correction for every run;
5. fit held-out emissions, lifetimes, repeated-imaging models, and loss budgets;
6. write the reviewed optimized result and ignored execution manifest.

The command does not edit the README, build publication prose, commit, push, or
merge. Development controls are available without changing the final contract:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_optimized_lifetimes.py --dry-run
.\.venv\Scripts\python.exe scripts\analyze_optimized_lifetimes.py --bootstrap 100 --seed 20260731
```

The 100-replicate command is for development only. It must not replace the
reviewed 1,000-replicate result.

## Optimized output contract

| Stage | Principal output |
|---|---|
| source audit | ignored JSON under `reports/audit/` |
| standardized export | ignored Parquet tables and schema/provenance sidecars |
| geometry validation | ignored reports under `reports/validation/optimized_lifetimes_geometry/` |
| background validation | ignored reports under `reports/validation/optimized_lifetimes_background/` |
| final inference | `reports/optimized_lifetime_results_20260731.json` |
| execution record | ignored `_scratch/optimized_final_reproduction/execution_manifest.json` |

The reviewed result records the analysis commit, clean-worktree gate, config
hashes, raw input-manifest hashes, geometry/background hashes, model version,
seed, and bootstrap count.

## Publication-only generation

After reviewing a clean optimized result, public figures and the publication
manifest are built without rerunning the scientific bootstrap:

```powershell
.\.venv\Scripts\python.exe scripts\generate_optimized_assets.py
.\.venv\Scripts\python.exe scripts\build_optimized_publication_manifest.py
```

The asset generator reads only the reviewed result and creates the four
optimized landing-page figures, generated metrics, and exact asset hashes. The
manifest verifies that those hashes target the reviewed result and binds the
public documents and environment. Formatting or prose changes require a
manifest rebuild, not another scientific reproduction.

## Verification

Focused gates can be run while editing:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_optimized_assets.py tests\test_readme_narrative.py tests\test_privacy.py
```

At the final code gate, run the complete suite once:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Tests do not replace review of the four public figures or the scientific claim
table. Exact asset byte equality is guaranteed only in the recorded
same-environment reproduction because rendering libraries can differ across
platforms.

## Pre-optimization benchmark

The earlier V0/V1 workflow remains runnable and is not silently rewritten with
optimized estimates:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_all.py
```

Its reviewed result is `reports/loss_sweep_results.json`, its publication
manifest is `reports/publication_manifest.json`, and its interpretation is
`docs/LOSS_SWEEP_INTERPRETATION.md`. Those artifacts document the 2026-07-28
paired-readout and loss-sweep measurements as a historical benchmark.

## Reproducibility boundaries

- Missing raw data cause a hard failure; synthetic measurements are never
  substituted.
- Geometry, fixed backgrounds, and emissions remain training-only.
- No shot crosses a train/validation/test boundary.
- Random seeds, numeric thread limits, and the noninteractive plotting backend
  are fixed by the orchestrator.
- Reproduction overwrites only declared generated outputs. It never edits raw
  data or performs Git cleanup, branch switching, commits, pushes, or merges.
