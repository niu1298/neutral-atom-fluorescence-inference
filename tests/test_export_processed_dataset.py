"""Export command safety checks."""
from __future__ import annotations

import sys

import pandas as pd

from fluorescence_inference import schema


class _InvalidExportConfig:
    schema_version = "3.0"
    n_frames = 2
    n_sites_expected = 1

    def __getitem__(self, key):
        if key == "source":
            return {"expected_n_shots": 1}
        raise KeyError(key)


def test_invalid_table_is_never_written(monkeypatch, capsys):
    from scripts import export_processed_dataset as command

    invalid = schema.ValidationReport(
        ok=False,
        n_rows=1,
        errors=["synthetic invalid contract"],
        warnings=[],
        stats={},
    )
    write_calls = []
    monkeypatch.setattr(command, "load_config", lambda _path: _InvalidExportConfig())
    monkeypatch.setattr(
        command,
        "build_frame_site_table",
        lambda _cfg, progress: (
            pd.DataFrame(),
            pd.DataFrame(),
            {"meta": {}, "shots": []},
        ),
    )
    monkeypatch.setattr(command.schema, "validate", lambda *_args, **_kwargs: invalid)
    monkeypatch.setattr(command, "stamp", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        command,
        "write_dataset",
        lambda *_args, **_kwargs: write_calls.append(True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_processed_dataset.py", "--quiet", "--no-qc"],
    )

    assert command.main() == 3
    assert write_calls == []
    assert "synthetic invalid contract" in capsys.readouterr().err
