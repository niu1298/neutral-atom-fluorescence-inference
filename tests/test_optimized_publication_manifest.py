"""Publication-manifest bindings for the optimized reviewed result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_optimized_publication_manifest as publication  # noqa: E402


def test_manifest_binds_result_assets_and_documents(tmp_path):
    result = {
        "analysis_version": "optimized-lifetimes-v1.0",
        "bootstrap_replicates": 1000,
        "random_seed": 20260731,
        "provenance": {
            "analysis_code_commit": "a" * 40,
            "publishable_clean_provenance": True,
            "development_stale_bindings": [],
            "configs": {"50ms": {"config_sha256": "b" * 64}},
            "geometry_report_sha256": "c" * 64,
            "background_report_sha256": "d" * 64,
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    asset = tmp_path / "assets" / "readme" / "figure.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"figure")
    document = tmp_path / "README.md"
    document.write_text("public", encoding="utf-8")
    metadata = {
        "result_sha256": publication.sha256_file(result_path),
        "assets": {
            "figure.png": {
                "sha256": publication.sha256_file(asset),
                "bytes": asset.stat().st_size,
            }
        },
    }
    (asset.parent / "optimized_asset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    manifest = publication.build_manifest(
        tmp_path,
        result_path=result_path,
        asset_paths=("assets/readme/figure.png",),
        document_paths=("README.md",),
    )
    assert manifest["analysis_code_commit"] == "a" * 40
    assert manifest["public_assets"]["assets/readme/figure.png"]["bytes"] == 6
    assert manifest["result"]["scientific_payload_sha256"]


def test_manifest_rejects_stale_development_result(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "bootstrap_replicates": 1000,
                "provenance": {
                    "publishable_clean_provenance": True,
                    "development_stale_bindings": ["stale"],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(publication.ManifestError, match="stale"):
        publication.build_manifest(
            tmp_path,
            result_path=result_path,
            asset_paths=(),
            document_paths=(),
        )
