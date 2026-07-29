"""Build and verify non-self-referential reviewed publication manifests.

The reviewed scientific result records the commit containing the analysis
code, not the later commit that publishes the result.  A final-branch
verification run may therefore differ in its runtime Git commit and branch,
but no scientific input, configuration, seed, clean-state flag, or model
output is normalized away.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "publication-manifest-v1"
PROVENANCE_ONLY_PATHS: tuple[tuple[str, ...], ...] = (
    ("provenance", "inference_repository", "commit"),
    ("provenance", "inference_repository", "branch"),
)
PROVENANCE_ONLY_PATH_STRINGS = tuple(
    ".".join(path) for path in PROVENANCE_ONLY_PATHS
)
PUBLIC_ASSET_PATHS = (
    "assets/readme/fluorescence_inference_overview.gif",
    "assets/readme/sequence_design.png",
    "assets/readme/loss_sweep_overview.png",
    "assets/readme/background_drift_sweeps.png",
)
ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "h5py",
    "pyarrow",
)
REQUIRED_SEED_KEYS = (
    "primary_analysis",
    "dark_condition_bootstrap",
    "bright_condition_bootstrap",
    "dark_cycle_bootstrap",
    "bright_cycle_bootstrap",
    "dark_multistart_diagnostic",
    "bright_multistart_diagnostic",
    "bright_monotone_control_sensitivity",
)


class PublicationVerificationError(RuntimeError):
    """A reviewed result, manifest, or candidate output did not agree."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationVerificationError(
            f"cannot read {label} ({Path(path).name}): {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PublicationVerificationError(f"{label} must contain a JSON object")
    return value


def _drop_path(value: dict[str, Any], path: Sequence[str]) -> None:
    parent: Any = value
    for component in path[:-1]:
        if not isinstance(parent, dict) or component not in parent:
            return
        parent = parent[component]
    if isinstance(parent, dict):
        parent.pop(path[-1], None)


def canonical_scientific_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a result and remove only explicitly declared runtime Git fields."""
    canonical = copy.deepcopy(dict(payload))
    for path in PROVENANCE_ONLY_PATHS:
        _drop_path(canonical, path)
    return canonical


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def scientific_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the scientific payload after the two declared exclusions."""
    return hashlib.sha256(
        _canonical_json_bytes(canonical_scientific_payload(payload))
    ).hexdigest()


def json_difference_paths(
    expected: Any,
    actual: Any,
    *,
    prefix: str = "",
    limit: int = 50,
) -> list[str]:
    """Return bounded, deterministic paths whose JSON values differ."""
    differences: list[str] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            differences.append(path or "<root>")
            return
        if isinstance(left, dict):
            keys = sorted(set(left) | set(right))
            for key in keys:
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    differences.append(child)
                else:
                    walk(left[key], right[key], child)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left, list):
            if len(left) != len(right):
                differences.append(f"{path}.length" if path else "<root>.length")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                child = f"{path}[{index}]" if path else f"[{index}]"
                walk(left_item, right_item, child)
                if len(differences) >= limit:
                    return
            return
        if left != right:
            differences.append(path or "<root>")

    walk(expected, actual, prefix)
    return differences


def _nested(
    value: Mapping[str, Any],
    *path: str,
    default: Any = None,
) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return default
        current = current[component]
    return current


def _environment_summary() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "<missing>"
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "platform": sys.platform,
        "packages": packages,
    }


def _require_hex(value: Any, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise PublicationVerificationError(
            f"reviewed result has no valid {label}"
        )
    return value


def build_publication_manifest(
    *,
    root: Path,
    result_path: Path,
    paired_asset_metadata_path: Path,
    public_asset_paths: Sequence[str] = PUBLIC_ASSET_PATHS,
    reproduction_command: str = (
        r".\.venv\Scripts\python.exe scripts\reproduce_all.py"
    ),
) -> dict[str, Any]:
    """Build a reviewed manifest without including the manifest's own hash."""
    root = Path(root)
    result_path = Path(result_path)
    result = _read_json_object(result_path, "reviewed result")
    paired = _read_json_object(
        paired_asset_metadata_path, "paired-readout asset metadata"
    )
    provenance = result.get("provenance", {})
    analysis_code_commit = provenance.get("analysis_code_commit")
    dirty = _nested(
        provenance, "inference_repository", "dirty", default=None
    )
    analysis_code_commit = _require_hex(
        analysis_code_commit,
        length=40,
        label="provenance.analysis_code_commit",
    )
    if dirty is not False:
        raise PublicationVerificationError(
            "reviewed result was not generated from a clean worktree"
        )

    assets: dict[str, dict[str, Any]] = {}
    for relative in public_asset_paths:
        path = root / relative
        if not path.is_file():
            raise PublicationVerificationError(
                f"reviewed public asset is missing: {relative}"
            )
        assets[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    paired_provenance = paired.get("provenance", {})
    dark = provenance.get("dark_hold", {})
    bright = provenance.get("bright_wait", {})
    config_hashes = {
        "paired_readout": _nested(
            paired_provenance, "config", "config_sha256"
        ),
        "switch_off_hold": dark.get("config_sha256"),
        "bright_wait": bright.get("config_sha256"),
    }
    input_hashes = {
        "paired_readout": _nested(
            paired_provenance,
            "inputs",
            "manifest_sha256",
        ),
        "switch_off_hold": _nested(
            dark, "input_manifest", "manifest_sha256"
        ),
        "bright_wait": _nested(
            bright, "input_manifest", "manifest_sha256"
        ),
    }
    validation_hashes = {
        key: provenance.get(key)
        for key in (
            "geometry_report_sha256",
            "background_report_sha256",
            "dark_command_audit_sha256",
            "bright_command_audit_sha256",
        )
    }
    for label, value in {
        **{f"config_sha256.{key}": value for key, value in config_hashes.items()},
        **{
            f"input_manifest_sha256.{key}": value
            for key, value in input_hashes.items()
        },
        **validation_hashes,
    }.items():
        _require_hex(value, length=64, label=label)
    dependency_commit = _require_hex(
        provenance.get("lab_analysis_repository_commit"),
        length=40,
        label="dependency commit",
    )
    model_version = provenance.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise PublicationVerificationError(
            "reviewed result has no model version"
        )
    bootstrap_replicates = provenance.get("bootstrap_replicates")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 100:
        raise PublicationVerificationError(
            "reviewed result has no valid bootstrap replicate count"
        )
    seeds = provenance.get("seeds")
    if not isinstance(seeds, Mapping):
        raise PublicationVerificationError(
            "reviewed result has no declared provenance seeds"
        )
    missing_seeds = [
        key for key in REQUIRED_SEED_KEYS if not isinstance(seeds.get(key), int)
    ]
    if missing_seeds:
        raise PublicationVerificationError(
            f"reviewed result is missing integer provenance seeds: {missing_seeds}"
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "analysis_code_commit": analysis_code_commit,
        "clean_worktree": True,
        "result": {
            "path": result_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(result_path),
            "scientific_payload_sha256": scientific_payload_sha256(result),
        },
        "public_assets": assets,
        "config_sha256": config_hashes,
        "input_manifest_sha256": input_hashes,
        "validation_inputs": validation_hashes,
        "dependency_commit": dependency_commit,
        "model_version": model_version,
        "bootstrap_replicates": bootstrap_replicates,
        "seeds": dict(seeds),
        "reproduction_command": reproduction_command,
        "environment": _environment_summary(),
        "verification": {
            "scientific_payload_excluded_paths": list(
                PROVENANCE_ONLY_PATH_STRINGS
            ),
            "asset_hash_mode": "exact-same-environment",
        },
    }
    return manifest


def write_publication_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a deterministic manifest; it deliberately has no self-hash."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_reviewed_artifacts(
    *,
    reviewed_root: Path,
    reviewed_result_path: Path,
    candidate_result_path: Path,
    manifest_path: Path,
    candidate_assets: Mapping[str, Path],
) -> dict[str, Any]:
    """Verify a temporary candidate without modifying reviewed artifacts."""
    reviewed_root = Path(reviewed_root)
    reviewed_result_path = Path(reviewed_result_path)
    candidate_result_path = Path(candidate_result_path)
    manifest = _read_json_object(manifest_path, "publication manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PublicationVerificationError(
            "unsupported publication manifest schema"
        )
    declared = _nested(
        manifest,
        "verification",
        "scientific_payload_excluded_paths",
        default=[],
    )
    if tuple(declared) != PROVENANCE_ONLY_PATH_STRINGS:
        raise PublicationVerificationError(
            "manifest provenance exclusions differ from the declared code contract"
        )

    reviewed = _read_json_object(reviewed_result_path, "reviewed result")
    candidate = _read_json_object(candidate_result_path, "candidate result")
    errors: list[str] = []

    expected_reviewed_hash = _nested(manifest, "result", "sha256")
    reviewed_hash = sha256_file(reviewed_result_path)
    if reviewed_hash != expected_reviewed_hash:
        errors.append(
            "reviewed result hash drift: "
            f"expected={expected_reviewed_hash} actual={reviewed_hash} "
            f"path={reviewed_result_path}"
        )

    expected_analysis_commit = manifest.get("analysis_code_commit")
    candidate_analysis_commit = _nested(
        candidate, "provenance", "analysis_code_commit"
    )
    if candidate_analysis_commit != expected_analysis_commit:
        errors.append(
            "candidate analysis-code commit mismatch: "
            f"expected={expected_analysis_commit} "
            f"actual={candidate_analysis_commit}"
        )
    candidate_dirty = _nested(
        candidate,
        "provenance",
        "inference_repository",
        "dirty",
        default=None,
    )
    if candidate_dirty is not False:
        errors.append(
            "candidate result does not record a clean generation worktree"
        )

    reviewed_scientific = canonical_scientific_payload(reviewed)
    candidate_scientific = canonical_scientific_payload(candidate)
    reviewed_scientific_hash = hashlib.sha256(
        _canonical_json_bytes(reviewed_scientific)
    ).hexdigest()
    candidate_scientific_hash = hashlib.sha256(
        _canonical_json_bytes(candidate_scientific)
    ).hexdigest()
    if reviewed_scientific_hash != candidate_scientific_hash:
        differences = json_difference_paths(
            reviewed_scientific, candidate_scientific
        )
        errors.append(
            "scientific payload drift: "
            f"reviewed={reviewed_scientific_hash} "
            f"candidate={candidate_scientific_hash}; "
            f"differing_paths={differences}"
        )

    manifest_assets = manifest.get("public_assets", {})
    if not isinstance(manifest_assets, Mapping):
        errors.append("manifest public_assets is not an object")
        manifest_assets = {}
    for relative, expected in sorted(manifest_assets.items()):
        expected_hash = (
            expected.get("sha256") if isinstance(expected, Mapping) else None
        )
        reviewed_asset = reviewed_root / relative
        candidate_asset = candidate_assets.get(relative)
        if not reviewed_asset.is_file():
            errors.append(f"reviewed public asset missing: {relative}")
            continue
        reviewed_asset_hash = sha256_file(reviewed_asset)
        if reviewed_asset_hash != expected_hash:
            errors.append(
                "reviewed asset hash drift: "
                f"path={relative} expected={expected_hash} "
                f"actual={reviewed_asset_hash}"
            )
        if candidate_asset is None or not Path(candidate_asset).is_file():
            errors.append(f"candidate public asset missing: {relative}")
            continue
        candidate_hash = sha256_file(Path(candidate_asset))
        if candidate_hash != expected_hash:
            errors.append(
                "candidate asset hash drift: "
                f"path={relative} expected={expected_hash} "
                f"actual={candidate_hash}"
            )

    if errors:
        raise PublicationVerificationError("\n".join(errors))
    return {
        "reviewed_result_sha256": reviewed_hash,
        "reviewed_scientific_payload_sha256": reviewed_scientific_hash,
        "candidate_scientific_payload_sha256": candidate_scientific_hash,
        "n_assets_verified": len(manifest_assets),
        "provenance_only_exclusions": list(PROVENANCE_ONLY_PATH_STRINGS),
    }
