"""Build the non-self-referential optimized publication manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "reports" / "optimized_lifetime_results_20260731.json"
DEFAULT_OUTPUT = ROOT / "reports" / "optimized_publication_manifest_20260731.json"
PUBLIC_ASSETS = (
    "assets/readme/fluorescence_inference_overview.gif",
    "assets/readme/optimized_occupancy_inference.gif",
    "assets/readme/optimized_occupancy_inference.png",
    "assets/readme/optimized_sequence_design.png",
    "assets/readme/optimized_lifetime_overview.png",
    "assets/readme/exposure_segmentation_result.png",
    "assets/readme/optimized_readout_tradeoff.png",
)
PUBLIC_DOCUMENTS = (
    "README.md",
    "docs/DATA_AUDIT_2026-07-31_OPTIMIZED_LIFETIMES.md",
    "docs/OPTIMIZED_LIFETIME_RESULTS.md",
    "docs/STATISTICAL_METHODS.md",
    "docs/LOSS_DECOMPOSITION.md",
    "docs/REPRODUCE.md",
    "docs/LOSS_SWEEP_INTERPRETATION.md",
    "reports/optimized_readme_metrics_20260731.md",
)
ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "h5py",
    "pyarrow",
)


class ManifestError(RuntimeError):
    """A reviewed result or publication file is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_records(root: Path, paths: Sequence[str]) -> dict[str, Any]:
    records = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise ManifestError(f"publication file is missing: {relative}")
        records[relative] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def build_manifest(
    root: Path,
    *,
    result_path: Path,
    asset_paths: Sequence[str] = PUBLIC_ASSETS,
    document_paths: Sequence[str] = PUBLIC_DOCUMENTS,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    provenance = result.get("provenance", {})
    if result.get("bootstrap_replicates") != 1000:
        raise ManifestError("optimized manifest requires 1000 bootstrap replicates")
    if provenance.get("publishable_clean_provenance") is not True:
        raise ManifestError("optimized result provenance is not clean")
    if provenance.get("development_stale_bindings"):
        raise ManifestError("optimized result retains stale development bindings")
    assets = _file_records(root, asset_paths)
    documents = _file_records(root, document_paths)
    metadata_path = root / "assets" / "readme" / "optimized_asset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("result_sha256") != sha256_file(result_path):
        raise ManifestError("optimized asset metadata targets a different result")
    for name, record in metadata.get("assets", {}).items():
        relative = f"assets/readme/{name}"
        if relative not in assets or assets[relative]["sha256"] != record["sha256"]:
            raise ManifestError(f"optimized asset metadata hash mismatch: {name}")
    return {
        "schema_version": "optimized-publication-manifest-v1",
        "analysis_code_commit": provenance["analysis_code_commit"],
        "bootstrap_replicates": int(result["bootstrap_replicates"]),
        "random_seed": int(result["random_seed"]),
        "model_version": result["analysis_version"],
        "clean_analysis_worktree": True,
        "result": {
            "path": result_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(result_path),
            "scientific_payload_sha256": scientific_payload_sha256(result),
        },
        "configs": provenance["configs"],
        "validation_inputs": {
            "geometry_report_sha256": provenance["geometry_report_sha256"],
            "background_report_sha256": provenance["background_report_sha256"],
        },
        "public_assets": assets,
        "public_documents": documents,
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "packages": {
                package: importlib.metadata.version(package)
                for package in ENVIRONMENT_PACKAGES
            },
        },
        "reproduction_command": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\reproduce_optimized_lifetimes.py"
        ),
        "verification": {
            "asset_hash_mode": "exact-same-environment",
            "independent_experimental_unit": "complete shot",
            "privacy_and_authorship_guard": "tests/test_privacy.py",
            "landing_page_guard": "tests/test_readme_narrative.py",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = args.results if args.results.is_absolute() else ROOT / args.results
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_manifest(ROOT, result_path=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": output.name, "result": manifest["result"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
