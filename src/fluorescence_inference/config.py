"""Configuration loading.

Two layers, deliberately separated:

* ``configs/paired_100ms.yaml`` - tracked, machine independent, hashed into
  every generated artefact.
* ``configs/local.toml``        - untracked, holds the absolute paths of this
  machine. Only its *resolved* existence is recorded downstream, never its
  contents.

Anything that would leak a username, a workstation name or a lab directory
layout belongs in the local layer.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

DEFAULT_CONFIG = "configs/paired_100ms.yaml"
DEFAULT_LOCAL = "configs/local.toml"
EXAMPLE_LOCAL = "configs/local.example.toml"


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class Paths:
    experiment_data_root: Path
    tweezer_analysis_src: Path
    processed_root: Path
    reports_root: Path

    def as_public_dict(self) -> dict[str, str]:
        """Path *shape* without machine-identifying prefixes.

        Only ever emit this into generated metadata: it records that a path
        was configured and, for read-only inputs, whether it exists, never
        where it points.  Output-directory existence is deliberately omitted:
        a clean first build creates those directories, so recording that state
        would make otherwise identical rebuild metadata differ.
        """
        return {
            "experiment_data_root": _redact(self.experiment_data_root),
            "tweezer_analysis_src": _redact(self.tweezer_analysis_src),
            "processed_root": "<configured>",
            "reports_root": "<configured>",
        }


def _redact(p: Path) -> str:
    """``<configured:present>`` / ``<configured:missing>`` - never the path."""
    return f"<configured:{'present' if p.exists() else 'missing'}>"


@dataclass(frozen=True)
class Config:
    """Resolved configuration plus the hash of the tracked YAML layer."""

    raw: dict[str, Any]
    paths: Paths
    config_path: Path
    config_sha256: str
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------- accessors
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset_id"])

    @property
    def run_id(self) -> str:
        return str(self.raw["run_id"])

    @property
    def schema_version(self) -> str:
        """Explicit table schema; absent means the backwards-compatible V2."""
        return str(self.raw.get("schema_version", "2.0"))

    @property
    def sequence_type(self) -> str:
        return str(self.raw.get("sequence_type", "paired_readout"))

    @property
    def shot_dir(self) -> Path:
        return self.paths.experiment_data_root / self.raw["source"]["shot_subdir"]

    @property
    def frame_specs(self) -> list[dict[str, Any]]:
        return list(self.raw["source"]["frames"])

    @property
    def n_frames(self) -> int:
        return len(self.frame_specs)

    @property
    def n_sites_expected(self) -> int:
        s = self.raw["sites"]
        return int(s["n_grids"]) * int(s["ny"]) * int(s["nx"])

    def processed_path(self, suffix: str = ".parquet") -> Path:
        return self.paths.processed_root / f"{self.dataset_id}{suffix}"

    def reports_dir(self, *parts: str) -> Path:
        return self.paths.reports_root.joinpath(*parts)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p)


def load_local_paths(root: Path | None = None, local_path: str | Path | None = None) -> Paths:
    """Read ``configs/local.toml``, falling back to the committed example.

    The fallback makes a fresh clone work out of the box on a machine whose
    layout matches the example; it never invents data, because every consumer
    checks that the resolved directories actually exist.
    """
    root = root or repo_root()
    candidates = [Path(local_path)] if local_path else [root / DEFAULT_LOCAL, root / EXAMPLE_LOCAL]
    for cand in candidates:
        cand = cand if cand.is_absolute() else (root / cand)
        if cand.exists():
            with open(cand, "rb") as fh:
                data = tomllib.load(fh)
            paths = data.get("paths", {})
            missing = {"experiment_data_root", "tweezer_analysis_src",
                       "processed_root", "reports_root"} - set(paths)
            if missing:
                raise ConfigError(
                    f"{cand.name} is missing [paths] keys: {sorted(missing)}"
                )
            return Paths(
                experiment_data_root=_resolve(root, paths["experiment_data_root"]),
                tweezer_analysis_src=_resolve(root, paths["tweezer_analysis_src"]),
                processed_root=_resolve(root, paths["processed_root"]),
                reports_root=_resolve(root, paths["reports_root"]),
            )
    raise ConfigError(
        f"no local path configuration found. Copy {EXAMPLE_LOCAL} to "
        f"{DEFAULT_LOCAL} and edit it for this machine."
    )


def load_config(config_path: str | Path = DEFAULT_CONFIG,
                local_path: str | Path | None = None,
                root: Path | None = None) -> Config:
    """Load the tracked YAML config and the local path layer."""
    root = root or repo_root()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path.name}")

    text = cfg_path.read_bytes()
    raw = yaml.safe_load(text.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path.name} did not parse to a mapping")

    return Config(
        raw=raw,
        paths=load_local_paths(root=root, local_path=local_path),
        config_path=cfg_path,
        config_sha256=hashlib.sha256(text).hexdigest(),
    )


def ensure_rydlab_importable(cfg: Config) -> str:
    """Put the lab analysis package on ``sys.path`` and return its version.

    Deliberately a path injection rather than an install: the general analysis
    checkout is treated as read-only, and an editable install would write build
    metadata into it.
    """
    src = cfg.paths.tweezer_analysis_src
    if not (src / "rydlab").is_dir():
        raise ConfigError(
            "tweezer_analysis_src does not contain a `rydlab` package. "
            "Point it at the `src` directory of the general analysis checkout."
        )
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    import rydlab  # noqa: PLC0415  (import is the point of this function)

    return getattr(rydlab, "__version__", "unknown")


def config_fingerprint(cfg: Config) -> dict[str, Any]:
    """Machine-independent provenance block for generated artefacts."""
    return {
        "config_file": cfg.config_path.name,
        "config_sha256": cfg.config_sha256,
        "dataset_id": cfg.dataset_id,
        "run_id": cfg.run_id,
        "paths": cfg.paths.as_public_dict(),
    }


def hash_mapping(obj: Any) -> str:
    """Stable sha256 of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def env_summary() -> dict[str, str]:
    """Interpreter facts only. No user, host or path information."""
    return {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "platform": sys.platform,
        "cpu_count": str(os.cpu_count() or 0),
    }
