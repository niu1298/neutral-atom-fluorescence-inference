from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fluorescence_inference.command_audit import (
    CommandAuditError,
    CommandAuditSpec,
    audit_compiled_commands,
)
from fluorescence_inference.config import Config, Paths


def _command_module(repo_root: Path):
    path = repo_root / "scripts" / "analyze_loss_sweeps.py"
    spec = importlib.util.spec_from_file_location(
        "_test_analyze_loss_sweeps", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_script(repo_root: Path):
    path = repo_root / "scripts" / "audit_source_data.py"
    spec = importlib.util.spec_from_file_location("_test_audit_source_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _word(switch_1: int, switch_2: int, trigger: int) -> int:
    return (switch_1 << 1) | (switch_2 << 2) | (trigger << 3)


def _write_synthetic_shot(
    path: Path,
    *,
    sequence_type: str,
    sweep_value_s: float = 0.1,
    corrupt_gap_switch: bool = False,
    corrupt_dds: bool = False,
    trigger_time_offset_s: float = 0.0,
) -> CommandAuditSpec:
    if sequence_type == "switch_off_hold":
        frame_names = ("fluor", "fluor2", "fluor3", "fluor4", "fluor5")
        sweep_global = "SECOND_HAMAMATSU_FLUOR_DELAY"
        segments = [(0.01, _word(0, 0, 0))]
        exposure_starts: list[float] = []
        elapsed = 0.01
        for index in range(5):
            exposure_starts.append(elapsed)
            segments.append((0.05, _word(0, 0, 1)))
            elapsed += 0.05
            if index < 4:
                gap_word = (
                    _word(0, 0, 0)
                    if corrupt_gap_switch and index == 1
                    else _word(1, 1, 0)
                )
                segments.append((sweep_value_s, gap_word))
                elapsed += sweep_value_s
        dds_pattern = [0.05, sweep_value_s] * 4 + [0.05]
    elif sequence_type == "bright_wait":
        frame_names = ("fluor", "fluor2")
        sweep_global = "WAIT_BEFORE_FIRST_FLUOR"
        gap_word = (
            _word(0, 0, 0)
            if corrupt_gap_switch
            else _word(1, 1, 0)
        )
        segments = [
            (sweep_value_s, _word(0, 0, 0)),
            (0.05, _word(0, 0, 1)),
            (0.01, gap_word),
            (0.05, _word(0, 0, 1)),
        ]
        exposure_starts = [sweep_value_s, sweep_value_s + 0.06]
        dds_pattern = [sweep_value_s, 0.05, 0.01, 0.05]
    else:
        raise ValueError(sequence_type)

    with h5py.File(path, "w") as h5:
        h5.attrs["run number"] = 0
        h5.attrs["run time"] = "2026-07-28 00:00:00"
        h5.attrs["n_runs"] = 1
        h5.attrs["sequence_index"] = 44
        h5.attrs["sequence_date"] = "2026-07-28"
        h5.attrs["script_basename"] = "synthetic_loss_sweep"

        globals_group = h5.create_group("globals")
        globals_group.attrs[sweep_global] = sweep_value_s

        connection_dtype = np.dtype(
            [
                ("name", "S64"),
                ("class", "S32"),
                ("parent", "S64"),
                ("parent port", "S16"),
                ("properties", "S128"),
            ]
        )
        connection = np.zeros(3, dtype=connection_dtype)
        properties = b'Content-Type: application/json {"inverted": false}'
        for index, (name, cls, port) in enumerate(
            (
                ("do_sci_img_light", "DigitalOut", "do1"),
                ("do_sci_img_light2", "DigitalOut", "do2"),
                ("hamamatsu_trigger", "Trigger", "do3"),
            )
        ):
            connection[index] = (
                name.encode(),
                cls.encode(),
                b"prawn_do_0__pod",
                port.encode(),
                properties,
            )
        h5.create_dataset("connection table", data=connection)

        digital = h5.create_group("devices/prawn_do_0")
        digital.attrs["clock_frequency"] = 1_000_000.0
        program_dtype = np.dtype([("bit_sets", "<u2"), ("reps", "<u4")])
        program = np.asarray(
            [
                (word, int(round(duration * 1_000_000)))
                for duration, word in segments
            ]
            + [(0, 0)],
            dtype=program_dtype,
        )
        digital.create_dataset("pulse_program", data=program)

        exposure_dtype = np.dtype(
            [
                ("t", "<f8"),
                ("name", "S16"),
                ("frametype", "S16"),
                ("trigger_duration", "<f8"),
            ]
        )
        exposure_rows = np.asarray(
            [
                (
                    start + trigger_time_offset_s,
                    name.encode(),
                    b"atoms",
                    0.05,
                )
                for name, start in zip(frame_names, exposure_starts)
            ],
            dtype=exposure_dtype,
        )
        hamamatsu = h5.create_group("devices/hamamatsu")
        hamamatsu.create_dataset("EXPOSURES", data=exposure_rows)

        string_dtype = h5py.string_dtype("utf-8")
        for device_index, device_name in enumerate(("dds_0", "dds_1")):
            commands = ["debug off", "setchannels 2"]
            for segment_index, duration in enumerate(dds_pattern):
                for channel in range(2):
                    amplitude = 0.5 - 0.1 * channel
                    if (
                        corrupt_dds
                        and device_index == 0
                        and segment_index == 2
                        and channel == 1
                    ):
                        amplitude += 0.05
                    commands.append(
                        "set "
                        f"{channel} {segment_index} "
                        f"{80_000_000 - channel * 10_000_000} "
                        f"{amplitude} 0 {duration}"
                    )
            commands.extend([f"set 2 {len(dds_pattern)}", "hwstart"])
            group = h5.create_group(f"devices/{device_name}")
            group.create_dataset(
                "RAW_PROGRAMS",
                data=np.asarray(commands, dtype=object),
                dtype=string_dtype,
            )

        source = """
def sequence():
    do_sci_img_light.go_low(t)
    do_sci_img_light2.go_low(t)
    hamamatsu.expose(t)
    do_sci_img_light.go_high(t)
    do_sci_img_light2.go_high(t)
"""
        h5.create_dataset("script", data=source, dtype=string_dtype)

    return CommandAuditSpec(
        sequence_type=sequence_type,
        frame_names=frame_names,
        sweep_global=sweep_global,
        expected_exposure_s=0.05,
    )


@pytest.mark.parametrize("sequence_type", ["switch_off_hold", "bright_wait"])
def test_compiled_command_audit_recovers_timing_switch_and_dds(
    scratch: Path, sequence_type: str
) -> None:
    path = scratch / f"{sequence_type}.h5"
    spec = _write_synthetic_shot(path, sequence_type=sequence_type)
    with h5py.File(path, "r") as h5:
        result = audit_compiled_commands(h5, spec)

    assert result["passed"]
    assert result["trigger_edges_verified"]
    assert result["max_abs_trigger_edge_error_s"] < 1e-12
    assert result["science_switch_levels"]["during_exposure"] == [0, 0]
    assert result["science_switch_levels"]["during_interframe_gap"] == [1, 1]
    if sequence_type == "bright_wait":
        assert result["science_switch_levels"]["during_bright_wait"] == [0, 0]
        assert result["science_switch_levels"]["pre_first_exposure_low_s"] == (
            pytest.approx(0.1)
        )
    for block in result["dds"].values():
        assert block["frequency_amplitude_constant"]
        assert len(block["state"]) == 2


@pytest.mark.parametrize("sequence_type", ["switch_off_hold", "bright_wait"])
def test_compiled_command_audit_rejects_wrong_switch_level(
    scratch: Path, sequence_type: str
) -> None:
    path = scratch / f"{sequence_type}_switch_bad.h5"
    spec = _write_synthetic_shot(
        path,
        sequence_type=sequence_type,
        corrupt_gap_switch=True,
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(CommandAuditError, match="inter-frame gap"):
            audit_compiled_commands(h5, spec)


def test_compiled_command_audit_rejects_dds_change(scratch: Path) -> None:
    path = scratch / "dds_bad.h5"
    spec = _write_synthetic_shot(
        path,
        sequence_type="bright_wait",
        corrupt_dds=True,
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(CommandAuditError, match="frequency/amplitude"):
            audit_compiled_commands(h5, spec)


def test_compiled_command_audit_rejects_trigger_edge_disagreement(
    scratch: Path,
) -> None:
    path = scratch / "trigger_bad.h5"
    spec = _write_synthetic_shot(
        path,
        sequence_type="bright_wait",
        trigger_time_offset_s=0.001,
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(CommandAuditError, match="trigger edges disagree"):
            audit_compiled_commands(h5, spec)


def test_source_audit_emits_config_and_input_provenance(
    scratch: Path, repo_root: Path
) -> None:
    shot_dir = scratch / "shots"
    shot_dir.mkdir()
    path = shot_dir / "shot_000.h5"
    _write_synthetic_shot(path, sequence_type="bright_wait")
    raw = {
        "schema_version": "3.0",
        "dataset_id": "synthetic_bright_wait",
        "run_id": "synthetic_run",
        "sequence_type": "bright_wait",
        "source": {
            "shot_subdir": "shots",
            "file_glob": "*.h5",
            "saturation_adu": 65535,
            "expected_exposure_s": 0.05,
            "h5": {
                "run_number_attr": "run number",
                "run_time_attr": "run time",
                "n_runs_attr": "n_runs",
                "sequence_index_attr": "sequence_index",
                "sequence_date_attr": "sequence_date",
                "script_basename_attr": "script_basename",
                "globals_group": "globals",
                "exposures_dataset": "devices/hamamatsu/EXPOSURES",
            },
            "frames": [
                {
                    "frame_id": 0,
                    "exposure_name": "fluor",
                    "h5_path": "images/hamamatsu/fluor/atoms",
                },
                {
                    "frame_id": 1,
                    "exposure_name": "fluor2",
                    "h5_path": "images/hamamatsu/fluor2/atoms",
                },
            ],
        },
        "sweep": {
            "global": "WAIT_BEFORE_FIRST_FLUOR",
            "axis": "bright_wait_s",
            "values_s": [0.1, 0.2],
        },
    }
    paths = Paths(
        experiment_data_root=scratch,
        tweezer_analysis_src=scratch,
        processed_root=scratch / "processed",
        reports_root=scratch / "reports",
    )
    cfg = Config(
        raw=raw,
        paths=paths,
        config_path=scratch / "synthetic.yaml",
        config_sha256="synthetic-config-sha256",
    )
    report = _audit_script(repo_root).audit(cfg)

    assert report["provenance"]["config"]["config_sha256"] == (
        "synthetic-config-sha256"
    )
    assert report["provenance"]["inputs"]["n_files"] == 1
    assert report["command_audit"]["all_shots_passed"]
    assert report["command_audit"]["n_shots_audited"] == 1


def test_analysis_gate_requires_matching_raw_command_audit(
    repo_root: Path,
) -> None:
    module = _command_module(repo_root)
    cfg = SimpleNamespace(
        dataset_id="synthetic_bright_wait",
        config_sha256="config-hash",
    )
    dataset_meta = {
        "provenance": {
            "inputs": {
                "n_files": 2,
                "total_bytes": 123,
                "manifest_sha256": "input-hash",
            }
        }
    }
    payload = {
        "provenance": {
            "config": {
                "dataset_id": "synthetic_bright_wait",
                "config_sha256": "config-hash",
            },
            "inputs": dataset_meta["provenance"]["inputs"],
        },
        "command_audit": {
            "required": True,
            "all_shots_passed": True,
            "trigger_edges_verified_every_shot": True,
            "switch_states_verified_every_shot": True,
            "dds_constancy_verified_every_shot": True,
            "dds_state_constant_across_shots": {
                "dds_0": True,
                "dds_1": True,
            },
            "n_shots_audited": 2,
        },
    }
    passed, evidence = module._compiled_command_audit_verified(
        payload,
        cfg=cfg,
        dataset_meta=dataset_meta,
        n_shots=2,
    )
    assert passed
    assert all(evidence["checks"].values())

    payload["command_audit"]["switch_states_verified_every_shot"] = False
    passed, evidence = module._compiled_command_audit_verified(
        payload,
        cfg=cfg,
        dataset_meta=dataset_meta,
        n_shots=2,
    )
    assert not passed
    assert not evidence["checks"]["switch_states_verified_every_shot"]
