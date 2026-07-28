"""Provenance stamps for generated artefacts.

Every generated table, report and figure set records enough to answer "which
code, which configuration, which input files produced this?" without recording
anything that identifies the machine or the person who ran it.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import Config, config_fingerprint, env_summary, repo_root


def git_describe(root: Path | None = None) -> dict[str, Any]:
    """Commit hash and dirty flag of this repository, if it is a Git repo."""
    root = root or repo_root()
    out: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=15)
        if rev.returncode != 0:
            return out
        out["commit"] = rev.stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, timeout=30)
        out["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                            capture_output=True, text=True, timeout=15)
        if br.returncode == 0:
            out["branch"] = br.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # git absent or unusable
        pass
    return out


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_input_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    """One hash over (relative name, size) of every input file.

    Content hashing 100 multi-megabyte HDF5 files on every run is wasteful and
    would still not detect a same-size, same-name substitution any better than
    the audit does. Names and sizes pin the *set* of inputs; the audit report
    pins their contents.
    """
    items = sorted((p.name, p.stat().st_size) for p in paths)
    blob = "\n".join(f"{n}:{s}" for n, s in items).encode("utf-8")
    return {
        "n_files": len(items),
        "total_bytes": sum(s for _, s in items),
        "manifest_sha256": hashlib.sha256(blob).hexdigest(),
    }


def stamp(cfg: Config, *, inputs: Iterable[Path] | None = None,
          extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Standard provenance block embedded in every generated artefact."""
    block: dict[str, Any] = {
        "git": git_describe(),
        "config": config_fingerprint(cfg),
        "environment": env_summary(),
    }
    if inputs is not None:
        block["inputs"] = hash_input_manifest(inputs)
    if extra:
        block.update(extra)
    return block
