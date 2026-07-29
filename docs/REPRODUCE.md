# Complete reproduction

This document preserves the independently runnable commands for all three
datasets. `scripts/reproduce_all.py` is a thin, fail-fast orchestration layer;
the scientific implementation remains in the package modules and standalone
scripts.

The workflow:

- reads raw HDF5 shots without modifying, renaming, or moving them;
- overwrites generated processed/report/publication outputs in place;
- never substitutes synthetic measurements when raw data are missing;
- performs no Git command, cleanup, reset, merge, or push;
- stops at the first failed audit, validation, analysis, asset, or test stage.

The publication generator may remove a stale **optional** per-site or latent
figure when its documented data/acceptance gate is not met. Raw data and
required supporting assets are not deleted.

## 1. Environment and local configuration

Python 3.10 or newer is required. Create and activate an environment, then
install the package with test dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Copy the untracked local configuration:

```bash
cp configs/local.example.toml configs/local.toml
```

Edit only `configs/local.toml`. It identifies the raw-data root, the read-only
general laboratory analysis checkout, and the local processed/report roots.
Relative and absolute local roots are supported; they are never written into
tracked public outputs.

Before a full run, confirm that the configured data root contains:

| dataset | complete shots | frames per shot |
|:--|--:|--:|
| paired 100 ms readout | 100 | 2 |
| switch-off hold | 110 | 5 |
| bright wait | 100 | 2 |

The audits verify the exact source contract and fail rather than repair or
replace a missing measurement.

## 2. One-command reproduction

From the repository root:

```bash
python scripts/reproduce_all.py
```

The default run uses 1,000 complete-shot bootstrap replicates, seed `20260728`,
and finishes with the complete test suite. The orchestrator resolves the
configured report roots for all three datasets and explicitly forwards the
geometry, background, command-audit, and V0-QC paths between stages. The
reviewed compact loss-sweep result is always regenerated at the tracked
repository path used by the public assets.

Useful controls:

```bash
python scripts/reproduce_all.py --dry-run
python scripts/reproduce_all.py --bootstrap 1000 --seed 20260728
python scripts/reproduce_all.py --skip-tests
```

`--dry-run` prints the configured commands without executing a stage.
`--skip-tests` is diagnostic only and does not certify a complete
reproduction.

Every subprocess uses the active interpreter and inherits these fixed
settings:

| variable | value |
|:--|:--|
| `PYTHONHASHSEED` | `0` |
| `MPLBACKEND` | `Agg` |
| `OMP_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `NUMEXPR_NUM_THREADS` | `1` |

The fixed order, seed, backend, and numeric thread counts remove common
within-environment nondeterminism. Byte equality across different Matplotlib,
FreeType, operating-system, or dependency versions is not promised.

## 3. Standalone commands

For the standard example output roots, the individual commands remain:

```bash
python scripts/audit_source_data.py --config configs/paired_100ms.yaml
python scripts/export_processed_dataset.py --config configs/paired_100ms.yaml
python scripts/validate_site_geometry.py --config configs/paired_100ms.yaml
python scripts/compare_background_methods.py --config configs/paired_100ms.yaml
python scripts/generate_readme_assets.py --config configs/paired_100ms.yaml

python scripts/audit_source_data.py --config configs/dark_hold_50ms_20260728_0044.yaml
python scripts/audit_source_data.py --config configs/bright_wait_50ms_20260728_0050.yaml
python scripts/export_processed_dataset.py --config configs/dark_hold_50ms_20260728_0044.yaml --no-qc
python scripts/export_processed_dataset.py --config configs/bright_wait_50ms_20260728_0050.yaml --no-qc
python scripts/validate_sweep_geometry.py --config configs/dark_hold_50ms_20260728_0044.yaml --config configs/bright_wait_50ms_20260728_0050.yaml
python scripts/compare_loss_sweep_backgrounds.py --config configs/dark_hold_50ms_20260728_0044.yaml --config configs/bright_wait_50ms_20260728_0050.yaml
python scripts/analyze_loss_sweeps.py --bootstrap 1000 --seed 20260728
python scripts/generate_loss_sweep_assets.py
python -m pytest
```

When report roots differ from the example, use
`python scripts/reproduce_all.py --dry-run` to see the explicit cross-stage
paths supplied to the final analysis and publication commands. Configured
roots hold ignored audits, QC, and detailed validation products; the reviewed
compact `reports/loss_sweep_results.json` remains the tracked public source of
truth.

## 4. Stage contract

| stage | principal output |
|:--|:--|
| source audits | ignored JSON under the configured report root |
| standardized exports | ignored Parquet tables and JSON sidecars |
| geometry/background validation | ignored validation reports |
| paired asset generation | preserved paired-readout assets and V0 metrics |
| held-out sweep analysis | reviewed `loss_sweep_results.json` plus ignored execution metadata |
| sweep asset generation | public figures, compact README markers, detailed generated metrics |
| tests | pass/fail result; no scientific fitting by the test runner |

The V0 exporter retains its QC stage. The sweep exports use `--no-qc` because
their dedicated geometry, background, held-out-model, and clustering gates run
immediately afterward.

Processed tables, raw-data audits, QC products, and detailed validation reports
remain ignored local outputs. Reviewed compact results, generated metrics, and
public assets are tracked. Raw shots and full processed experimental tables
must not be committed.

## 5. Determinism and review

The public sweep renderer reads only the reviewed loss-sweep result JSON.
Running it repeatedly in an unchanged environment must leave the README,
metrics fragments, asset metadata, and PNG hashes unchanged. The paired asset
tests apply the same byte-determinism rule to the preserved V0 figures and hero
animation.

After a complete run:

```bash
git status --short
git diff --check
python -m pytest
```

Review every tracked change. A changed hash can reflect a legitimate input,
dependency, rendering, or source-state difference; it must be explained rather
than hidden. A reproducible computation does not convert apparent occupancy
into ground truth, model overlap into empirical fidelity, or an operational
time constant into an intrinsic lifetime.

## 6. Failure behavior

Exit code `0` means every requested stage passed. Exit code `2` means local
configuration could not be resolved or a stage failed.

Common recovery checks:

1. Missing shots: verify `configs/local.toml` and the expected date/sequence
   directories.
2. Audit failure: inspect the matching source-audit report before continuing.
3. Geometry/background failure: preserve the failed report and diagnose the
   run; do not select a method for a preferred rate.
4. Stale provenance: rerun the affected export, then its downstream
   validation and analysis.
5. Test failure: rerun the named test, then the complete suite.

Existing outputs remain available for inspection after a failure. No broad
repository cleanup is required.
