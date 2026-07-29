"""Read-only audit of compiled timing, switch, and DDS command programs.

The HDF5 files contain two independent representations of the commanded
fluorescence timing:

* ``devices/hamamatsu/EXPOSURES`` records the requested camera exposures;
* ``devices/prawn_do_0/pulse_program`` records the compiled digital word and
  its duration at the 100 MHz output clock.

They also contain the compiled ``RAW_PROGRAMS`` sent to both DDS sweepers.
This module compares those records directly.  It deliberately does not claim
that a compiled command was realised by the camera, optical switch, DDS, or
light field: no independent hardware timestamp or optical readback is present.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np


COMMAND_AUDIT_SCHEMA_VERSION = "1.0"
SUPPORTED_SEQUENCE_TYPES = frozenset({"switch_off_hold", "bright_wait"})
SCIENCE_SWITCH_NAMES = ("do_sci_img_light", "do_sci_img_light2")
CAMERA_TRIGGER_NAME = "hamamatsu_trigger"
DIGITAL_DEVICE_NAME = "prawn_do_0"
DDS_DEVICE_NAMES = ("dds_0", "dds_1")


class CommandAuditError(RuntimeError):
    """A compiled command program is missing, malformed, or contradictory."""


@dataclass(frozen=True)
class CommandAuditSpec:
    """Machine-independent names needed to interpret one sweep shot."""

    sequence_type: str
    frame_names: tuple[str, ...]
    sweep_global: str
    exposures_dataset: str = "devices/hamamatsu/EXPOSURES"
    globals_group: str = "globals"
    expected_exposure_s: float | None = None


@dataclass(frozen=True)
class _DigitalSegment:
    start_s: float
    end_s: float
    bit_word: int


@dataclass(frozen=True)
class _DdsSegment:
    index: int
    duration_s: float
    state: tuple[tuple[int, float, float], ...]


def spec_from_config(cfg: Any) -> CommandAuditSpec:
    """Build the audit spec from names in a tracked dataset configuration."""

    source = cfg["source"]
    sweep = cfg.get("sweep") or {}
    h5 = source["h5"]
    return CommandAuditSpec(
        sequence_type=str(cfg.sequence_type),
        frame_names=tuple(
            str(frame["exposure_name"]) for frame in cfg.frame_specs
        ),
        sweep_global=str(sweep.get("global", "")),
        exposures_dataset=str(h5["exposures_dataset"]),
        globals_group=str(h5.get("globals_group", "globals")),
        expected_exposure_s=(
            None
            if source.get("expected_exposure_s") is None
            else float(source["expected_exposure_s"])
        ),
    )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _normalised_hash(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_property(value: Any) -> dict[str, Any]:
    text = _text(value)
    prefix = "Content-Type: application/json "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CommandAuditError(
            "connection-table properties are not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise CommandAuditError("connection-table properties are not a mapping")
    return parsed


def _connection_mapping(h5: Any) -> dict[str, dict[str, Any]]:
    if "connection table" not in h5:
        raise CommandAuditError("missing HDF5 connection table")
    table = h5["connection table"]
    required = {"name", "class", "parent", "parent port", "properties"}
    fields = set(table.dtype.names or ())
    missing = sorted(required - fields)
    if missing:
        raise CommandAuditError(
            f"connection table is missing fields: {missing}"
        )
    mapping: dict[str, dict[str, Any]] = {}
    for row in table[:]:
        name = _text(row["name"])
        if name in mapping:
            raise CommandAuditError(f"duplicate connection-table name {name!r}")
        mapping[name] = {
            "class": _text(row["class"]),
            "parent": _text(row["parent"]),
            "parent_port": _text(row["parent port"]),
            "properties": _json_property(row["properties"]),
        }
    return mapping


def _digital_bit(
    mapping: Mapping[str, Mapping[str, Any]], name: str
) -> tuple[int, dict[str, Any]]:
    if name not in mapping:
        raise CommandAuditError(f"connection table has no {name!r} output")
    row = dict(mapping[name])
    match = re.fullmatch(r"do(\d+)", str(row["parent_port"]))
    if match is None:
        raise CommandAuditError(
            f"{name!r} parent port is not a PrawnDO bit: "
            f"{row['parent_port']!r}"
        )
    properties = row["properties"]
    if bool(properties.get("inverted", False)):
        raise CommandAuditError(
            f"{name!r} is inverted; this audit requires explicit "
            "non-inverted compiled line levels"
        )
    return int(match.group(1)), row


def _digital_segments(
    h5: Any, *, device_name: str = DIGITAL_DEVICE_NAME
) -> tuple[list[_DigitalSegment], float, str]:
    path = f"devices/{device_name}/pulse_program"
    if path not in h5:
        raise CommandAuditError(f"missing compiled digital program {path!r}")
    group = h5[f"devices/{device_name}"]
    try:
        clock_hz = float(group.attrs["clock_frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CommandAuditError(
            f"{device_name!r} has no valid clock_frequency"
        ) from exc
    if not np.isfinite(clock_hz) or clock_hz <= 0.0:
        raise CommandAuditError("digital clock frequency must be positive")
    program = group["pulse_program"][:]
    fields = set(program.dtype.names or ())
    if not {"bit_sets", "reps"} <= fields:
        raise CommandAuditError(
            "compiled digital program lacks bit_sets/reps fields"
        )
    segments: list[_DigitalSegment] = []
    elapsed = 0.0
    saw_zero_duration = False
    for row in program:
        reps = int(row["reps"])
        if reps < 0:
            raise CommandAuditError("digital program contains negative reps")
        if reps == 0:
            saw_zero_duration = True
            continue
        if saw_zero_duration:
            raise CommandAuditError(
                "positive-duration digital instruction follows a sentinel"
            )
        duration = reps / clock_hz
        segments.append(
            _DigitalSegment(
                start_s=elapsed,
                end_s=elapsed + duration,
                bit_word=int(row["bit_sets"]),
            )
        )
        elapsed += duration
    if not segments:
        raise CommandAuditError("compiled digital program has no duration")
    program_hash = hashlib.sha256(np.asarray(program).tobytes()).hexdigest()
    return segments, clock_hz, program_hash


def _bit_level(segment: _DigitalSegment, bit: int) -> int:
    return int((segment.bit_word >> bit) & 1)


def _level_windows(
    segments: Sequence[_DigitalSegment], bit: int, level: int
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for segment in segments:
        if _bit_level(segment, bit) != level:
            continue
        if windows and np.isclose(
            windows[-1][1], segment.start_s, rtol=0.0, atol=1e-15
        ):
            windows[-1] = (windows[-1][0], segment.end_s)
        else:
            windows.append((segment.start_s, segment.end_s))
    return windows


def _levels_in_interval(
    segments: Sequence[_DigitalSegment],
    bits: Sequence[int],
    start_s: float,
    end_s: float,
    *,
    tolerance_s: float,
) -> set[tuple[int, ...]]:
    if end_s <= start_s:
        raise CommandAuditError("command interval must have positive duration")
    overlap = [
        segment
        for segment in segments
        if segment.end_s > start_s + tolerance_s
        and segment.start_s < end_s - tolerance_s
    ]
    if not overlap:
        raise CommandAuditError("compiled digital program does not cover interval")
    covered_start = min(segment.start_s for segment in overlap)
    covered_end = max(segment.end_s for segment in overlap)
    if covered_start > start_s + tolerance_s:
        raise CommandAuditError("compiled digital interval begins too late")
    if covered_end < end_s - tolerance_s:
        raise CommandAuditError("compiled digital interval ends too early")
    return {
        tuple(_bit_level(segment, bit) for bit in bits)
        for segment in overlap
    }


def _exposure_schedule(
    h5: Any, spec: CommandAuditSpec
) -> list[dict[str, Any]]:
    if spec.exposures_dataset not in h5:
        raise CommandAuditError(
            f"missing exposure table {spec.exposures_dataset!r}"
        )
    rows = h5[spec.exposures_dataset][:]
    fields = set(rows.dtype.names or ())
    if not {"t", "name", "trigger_duration"} <= fields:
        raise CommandAuditError(
            "exposure table lacks t/name/trigger_duration fields"
        )
    by_name: dict[str, Any] = {}
    for row in rows:
        name = _text(row["name"])
        if name in by_name:
            raise CommandAuditError(f"duplicate exposure name {name!r}")
        by_name[name] = row
    missing = [name for name in spec.frame_names if name not in by_name]
    if missing:
        raise CommandAuditError(f"exposure table is missing frames: {missing}")
    selected = []
    for name in spec.frame_names:
        row = by_name[name]
        start = float(row["t"])
        duration = float(row["trigger_duration"])
        if (
            not np.isfinite(start)
            or not np.isfinite(duration)
            or start < 0.0
            or duration <= 0.0
        ):
            raise CommandAuditError(f"exposure {name!r} has invalid timing")
        if (
            spec.expected_exposure_s is not None
            and not np.isclose(
                duration,
                spec.expected_exposure_s,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise CommandAuditError(
                f"exposure {name!r} duration {duration!r} does not match "
                f"{spec.expected_exposure_s!r}"
            )
        selected.append(
            {
                "name": name,
                "start_s": start,
                "duration_s": duration,
                "end_s": start + duration,
            }
        )
    if any(
        later["start_s"] <= earlier["start_s"]
        for earlier, later in zip(selected, selected[1:])
    ):
        raise CommandAuditError("configured exposure order is not chronological")
    return selected


def _sweep_value(h5: Any, spec: CommandAuditSpec) -> float:
    if not spec.sweep_global:
        raise CommandAuditError("sweep global name is empty")
    if spec.globals_group not in h5:
        raise CommandAuditError(
            f"missing globals group {spec.globals_group!r}"
        )
    attrs = h5[spec.globals_group].attrs
    if spec.sweep_global not in attrs:
        raise CommandAuditError(
            f"missing swept global {spec.sweep_global!r}"
        )
    try:
        value = float(attrs[spec.sweep_global])
    except (TypeError, ValueError) as exc:
        raise CommandAuditError("sweep value is not numeric") from exc
    if not np.isfinite(value) or value < 0.0:
        raise CommandAuditError("sweep value must be finite and non-negative")
    return value


def _embedded_source_evidence(h5: Any) -> dict[str, Any]:
    if "script" not in h5:
        raise CommandAuditError("embedded sequence source is missing")
    source = h5["script"][()]
    source_bytes = (
        bytes(source)
        if isinstance(source, (bytes, np.bytes_))
        else str(source).encode("utf-8")
    )
    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise CommandAuditError(
            "embedded sequence source cannot be parsed"
        ) from exc
    call_counts = {
        name: {"go_low": 0, "go_high": 0} for name in SCIENCE_SWITCH_NAMES
    }
    camera_expose_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in call_counts:
            if node.func.attr in call_counts[owner.id]:
                call_counts[owner.id][node.func.attr] += 1
        if (
            isinstance(owner, ast.Name)
            and owner.id == "hamamatsu"
            and node.func.attr == "expose"
        ):
            camera_expose_calls += 1
    if camera_expose_calls < 1:
        raise CommandAuditError(
            "embedded source has no hamamatsu.expose command"
        )
    for name, counts in call_counts.items():
        if counts["go_low"] < 1 or counts["go_high"] < 1:
            raise CommandAuditError(
                f"embedded source lacks low/high commands for {name!r}"
            )
    return {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "science_switch_call_counts": call_counts,
        "hamamatsu_expose_call_count": camera_expose_calls,
    }


def _parse_dds_program(h5: Any, device_name: str) -> tuple[
    list[_DdsSegment], str
]:
    path = f"devices/{device_name}/RAW_PROGRAMS"
    if path not in h5:
        raise CommandAuditError(f"missing compiled DDS program {path!r}")
    commands = [_text(value) for value in h5[path][:]]
    channel_count: int | None = None
    segments: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for command in commands:
        fields = command.split()
        if not fields:
            continue
        if fields[0] == "setchannels":
            if len(fields) != 2:
                raise CommandAuditError(
                    f"{device_name} has malformed setchannels command"
                )
            channel_count = int(fields[1])
            continue
        if fields[0] != "set" or len(fields) != 7:
            # The hardware program also contains mode/debug/start and a short
            # terminal "set <channels> <index>" marker.  Only the seven-field
            # waveform rows carry frequency/amplitude/duration state.
            continue
        try:
            channel = int(fields[1])
            segment_index = int(fields[2])
            frequency_hz = float(fields[3])
            amplitude = float(fields[4])
            phase = float(fields[5])
            duration_s = float(fields[6])
        except ValueError as exc:
            raise CommandAuditError(
                f"{device_name} has a malformed set command"
            ) from exc
        if (
            not np.isfinite(
                [frequency_hz, amplitude, phase, duration_s]
            ).all()
            or duration_s <= 0.0
        ):
            raise CommandAuditError(
                f"{device_name} has non-finite or non-positive set values"
            )
        block = segments.setdefault(segment_index, {})
        if channel in block:
            raise CommandAuditError(
                f"{device_name} segment {segment_index} repeats channel {channel}"
            )
        block[channel] = (frequency_hz, amplitude, phase, duration_s)
    if channel_count is None or channel_count <= 0:
        raise CommandAuditError(
            f"{device_name} has no valid setchannels command"
        )
    if not segments:
        raise CommandAuditError(f"{device_name} has no waveform segments")
    indices = sorted(segments)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise CommandAuditError(
            f"{device_name} waveform segment indices are not contiguous"
        )
    expected_channels = set(range(channel_count))
    parsed: list[_DdsSegment] = []
    for index in indices:
        block = segments[index]
        if set(block) != expected_channels:
            raise CommandAuditError(
                f"{device_name} segment {index} does not define every channel"
            )
        durations = {block[channel][3] for channel in expected_channels}
        if len(durations) != 1:
            raise CommandAuditError(
                f"{device_name} segment {index} has inconsistent durations"
            )
        state = tuple(
            (
                channel,
                float(block[channel][0]),
                float(block[channel][1]),
            )
            for channel in sorted(expected_channels)
        )
        parsed.append(
            _DdsSegment(
                index=index,
                duration_s=float(next(iter(durations))),
                state=state,
            )
        )
    program_hash = hashlib.sha256(
        "\n".join(commands).encode("utf-8")
    ).hexdigest()
    return parsed, program_hash


def _duration_pattern(
    sequence_type: str,
    exposures: Sequence[Mapping[str, Any]],
    sweep_value_s: float,
) -> list[float]:
    durations = [float(exposure["duration_s"]) for exposure in exposures]
    gaps = [
        float(later["start_s"]) - float(earlier["end_s"])
        for earlier, later in zip(exposures, exposures[1:])
    ]
    if sequence_type == "switch_off_hold":
        if len(exposures) != 5:
            raise CommandAuditError(
                "switch-off command audit requires five exposures"
            )
        if not np.allclose(gaps, sweep_value_s, rtol=0.0, atol=1e-9):
            raise CommandAuditError(
                "exposure schedule gaps do not match the raw hold global"
            )
        pattern: list[float] = []
        for index, duration in enumerate(durations):
            pattern.append(duration)
            if index < len(gaps):
                pattern.append(gaps[index])
        return pattern
    if sequence_type == "bright_wait":
        if len(exposures) != 2:
            raise CommandAuditError(
                "bright-wait command audit requires two exposures"
            )
        return [sweep_value_s, durations[0], gaps[0], durations[1]]
    raise CommandAuditError(
        f"unsupported command-audit sequence type {sequence_type!r}"
    )


def _match_dds_pattern(
    segments: Sequence[_DdsSegment],
    expected_durations_s: Sequence[float],
    *,
    device_name: str,
) -> dict[str, Any]:
    n = len(expected_durations_s)
    hits: list[int] = []
    durations = np.asarray(
        [segment.duration_s for segment in segments], dtype=float
    )
    expected = np.asarray(expected_durations_s, dtype=float)
    for start in range(len(segments) - n + 1):
        if np.allclose(
            durations[start : start + n], expected, rtol=0.0, atol=1e-9
        ):
            hits.append(start)
    if len(hits) != 1:
        raise CommandAuditError(
            f"{device_name} science-duration pattern matched "
            f"{len(hits)} locations, expected exactly one"
        )
    selected = list(segments[hits[0] : hits[0] + n])
    reference = selected[0].state
    if any(segment.state != reference for segment in selected[1:]):
        raise CommandAuditError(
            f"{device_name} frequency/amplitude commands change in the "
            "science wait/exposure block"
        )
    state_rows = [
        {
            "channel": int(channel),
            "frequency_hz": float(frequency),
            "amplitude_command": float(amplitude),
        }
        for channel, frequency, amplitude in reference
    ]
    return {
        "frequency_amplitude_constant": True,
        "matched_segment_indices": [
            int(segment.index) for segment in selected
        ],
        "matched_duration_pattern_s": [
            float(segment.duration_s) for segment in selected
        ],
        "state": state_rows,
        "state_sha256": _normalised_hash(state_rows),
    }


def audit_compiled_commands(h5: Any, spec: CommandAuditSpec) -> dict[str, Any]:
    """Audit one open HDF5 shot against its compiled command programs.

    Passing means that the two commanded camera-timing representations agree,
    both compiled switch outputs have the expected logical level during each
    wait/exposure interval, and each DDS raw program contains exactly one
    duration-matched science block with unchanged frequency/amplitude commands.
    It does not mean that the hardware or optical field followed those commands.
    """

    if spec.sequence_type not in SUPPORTED_SEQUENCE_TYPES:
        raise CommandAuditError(
            f"unsupported sequence type {spec.sequence_type!r}"
        )
    exposures = _exposure_schedule(h5, spec)
    sweep_value_s = _sweep_value(h5, spec)
    mapping = _connection_mapping(h5)
    trigger_bit, trigger_row = _digital_bit(mapping, CAMERA_TRIGGER_NAME)
    switch_info = [
        _digital_bit(mapping, name) for name in SCIENCE_SWITCH_NAMES
    ]
    switch_bits = tuple(info[0] for info in switch_info)
    if len(set(switch_bits)) != len(switch_bits):
        raise CommandAuditError("science switches map to the same digital bit")
    if trigger_bit in switch_bits:
        raise CommandAuditError("camera trigger shares a science-switch bit")
    parents = {
        trigger_row["parent"],
        *(info[1]["parent"] for info in switch_info),
    }
    if len(parents) != 1:
        raise CommandAuditError(
            "camera trigger and science switches are not on one digital pod"
        )

    segments, clock_hz, digital_hash = _digital_segments(h5)
    tolerance_s = max(5.0 / clock_hz, 1e-7)
    trigger_windows = _level_windows(segments, trigger_bit, 1)
    if len(trigger_windows) != len(exposures):
        raise CommandAuditError(
            f"compiled camera trigger has {len(trigger_windows)} high pulses; "
            f"expected {len(exposures)}"
        )
    trigger_rows = []
    for exposure, (start_s, end_s) in zip(exposures, trigger_windows):
        start_error = start_s - float(exposure["start_s"])
        duration_error = (
            (end_s - start_s) - float(exposure["duration_s"])
        )
        end_error = end_s - float(exposure["end_s"])
        if max(
            abs(start_error), abs(duration_error), abs(end_error)
        ) > tolerance_s:
            raise CommandAuditError(
                f"compiled trigger edges disagree with exposure "
                f"{exposure['name']!r} beyond {tolerance_s:g} s"
            )
        levels = _levels_in_interval(
            segments,
            switch_bits,
            start_s,
            end_s,
            tolerance_s=tolerance_s,
        )
        if levels != {(0, 0)}:
            raise CommandAuditError(
                f"science switches are not both low during "
                f"{exposure['name']!r}: {sorted(levels)}"
            )
        trigger_rows.append(
            {
                "name": str(exposure["name"]),
                "exposure_table_start_s": float(exposure["start_s"]),
                "compiled_start_s": float(start_s),
                "compiled_duration_s": float(end_s - start_s),
                "start_error_s": float(start_error),
                "duration_error_s": float(duration_error),
                "end_error_s": float(end_error),
            }
        )

    compiled_gaps = [
        later[0] - earlier[1]
        for earlier, later in zip(trigger_windows, trigger_windows[1:])
    ]
    gap_levels: list[list[int]] = []
    for earlier, later in zip(trigger_windows, trigger_windows[1:]):
        levels = _levels_in_interval(
            segments,
            switch_bits,
            earlier[1],
            later[0],
            tolerance_s=tolerance_s,
        )
        if levels != {(1, 1)}:
            raise CommandAuditError(
                "science switches are not both high during an inter-frame gap: "
                f"{sorted(levels)}"
            )
        gap_levels.append([1, 1])

    first_trigger_start = trigger_windows[0][0]
    switch_low_windows = [
        window
        for window in _level_windows(segments, switch_bits[0], 0)
        if window[0] <= first_trigger_start + tolerance_s
        and window[1] >= first_trigger_start - tolerance_s
        and _levels_in_interval(
            segments,
            switch_bits,
            window[0],
            window[1],
            tolerance_s=tolerance_s,
        )
        == {(0, 0)}
    ]
    if len(switch_low_windows) != 1:
        raise CommandAuditError(
            "could not identify one joint switch-low interval at first exposure"
        )
    pre_first_low_s = first_trigger_start - switch_low_windows[0][0]
    if spec.sequence_type == "bright_wait":
        wait_start = first_trigger_start - sweep_value_s
        levels = _levels_in_interval(
            segments,
            switch_bits,
            wait_start,
            first_trigger_start,
            tolerance_s=tolerance_s,
        )
        if levels != {(0, 0)}:
            raise CommandAuditError(
                "science switches are not both low throughout the bright wait"
            )
        if not np.isclose(
            pre_first_low_s, sweep_value_s, rtol=0.0, atol=tolerance_s
        ):
            raise CommandAuditError(
                "compiled switch-low lead does not match the raw bright-wait "
                "global"
            )

    expected_dds_pattern = _duration_pattern(
        spec.sequence_type, exposures, sweep_value_s
    )
    dds: dict[str, Any] = {}
    for device_name in DDS_DEVICE_NAMES:
        dds_segments, program_hash = _parse_dds_program(h5, device_name)
        matched = _match_dds_pattern(
            dds_segments,
            expected_dds_pattern,
            device_name=device_name,
        )
        matched["raw_program_sha256"] = program_hash
        dds[device_name] = matched

    source = _embedded_source_evidence(h5)
    connection_bits = {
        CAMERA_TRIGGER_NAME: int(trigger_bit),
        **{
            name: int(bit)
            for name, (bit, _row) in zip(
                SCIENCE_SWITCH_NAMES, switch_info
            )
        },
    }
    evidence = {
        "schema_version": COMMAND_AUDIT_SCHEMA_VERSION,
        "passed": True,
        "sequence_type": spec.sequence_type,
        "sweep_global": spec.sweep_global,
        "sweep_value_s": float(sweep_value_s),
        "scope": (
            "compiled command programs only; no camera timestamp, switch "
            "readback, DDS readback, optical-power readback, or extinction "
            "measurement"
        ),
        "connection_bits": connection_bits,
        "digital_clock_hz": float(clock_hz),
        "edge_tolerance_s": float(tolerance_s),
        "digital_program_sha256": digital_hash,
        "trigger_edges_verified": True,
        "trigger_edges": trigger_rows,
        "max_abs_trigger_edge_error_s": float(
            max(
                abs(row[key])
                for row in trigger_rows
                for key in ("start_error_s", "duration_error_s", "end_error_s")
            )
        ),
        "science_switch_levels": {
            "during_exposure": [0, 0],
            "during_interframe_gap": [1, 1],
            "during_bright_wait": (
                [0, 0] if spec.sequence_type == "bright_wait" else None
            ),
            "interpretation": (
                "The compiled outputs are non-inverted. Low coincides with "
                "the embedded sequence's imaging commands and is termed "
                "commanded on; high is termed commanded off."
            ),
            "pre_first_exposure_low_s": float(pre_first_low_s),
            "compiled_interframe_gap_s": [
                float(value) for value in compiled_gaps
            ],
            "gap_levels": gap_levels,
        },
        "switch_states_verified": True,
        "dds_frequency_amplitude_constant": True,
        "dds": dds,
        "embedded_sequence": source,
    }
    evidence["evidence_sha256"] = _normalised_hash(evidence)
    return evidence


def summarize_command_audits(
    records: Sequence[Mapping[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    """Aggregate per-shot evidence without exposing file-system paths."""

    if not required:
        return {
            "required": False,
            "schema_version": COMMAND_AUDIT_SCHEMA_VERSION,
            "all_shots_passed": None,
            "scope": "not required for this sequence type",
        }
    failures = [
        record for record in records if not bool(record.get("passed"))
    ]
    passes = [record for record in records if bool(record.get("passed"))]
    dds_state_variants: dict[str, list[dict[str, Any]]] = {}
    dds_program_hashes: dict[str, list[str]] = {}
    for device in DDS_DEVICE_NAMES:
        by_hash: dict[str, dict[str, Any]] = {}
        program_hashes = set()
        for record in passes:
            block = record.get("dds", {}).get(device, {})
            state_hash = str(block.get("state_sha256", ""))
            if state_hash:
                by_hash[state_hash] = {
                    "state_sha256": state_hash,
                    "state": block.get("state"),
                }
            raw_hash = str(block.get("raw_program_sha256", ""))
            if raw_hash:
                program_hashes.add(raw_hash)
        dds_state_variants[device] = [
            by_hash[key] for key in sorted(by_hash)
        ]
        dds_program_hashes[device] = sorted(program_hashes)
    edge_errors = [
        float(record["max_abs_trigger_edge_error_s"])
        for record in passes
        if record.get("max_abs_trigger_edge_error_s") is not None
    ]
    return {
        "required": True,
        "schema_version": COMMAND_AUDIT_SCHEMA_VERSION,
        "scope": (
            "compiled camera-trigger, science-switch, and DDS command "
            "programs; not independent hardware or optical readback"
        ),
        "n_shots_audited": int(len(records)),
        "n_shots_passed": int(len(passes)),
        "n_shots_failed": int(len(failures)),
        "all_shots_passed": bool(records and not failures),
        "trigger_edges_verified_every_shot": bool(
            passes
            and len(passes) == len(records)
            and all(record.get("trigger_edges_verified") for record in passes)
        ),
        "switch_states_verified_every_shot": bool(
            passes
            and len(passes) == len(records)
            and all(record.get("switch_states_verified") for record in passes)
        ),
        "dds_constancy_verified_every_shot": bool(
            passes
            and len(passes) == len(records)
            and all(
                record.get("dds_frequency_amplitude_constant")
                for record in passes
            )
        ),
        "max_abs_trigger_edge_error_s": (
            None if not edge_errors else float(max(edge_errors))
        ),
        "dds_state_variants": dds_state_variants,
        "dds_state_constant_across_shots": {
            device: len(dds_state_variants[device]) == 1
            for device in DDS_DEVICE_NAMES
        },
        "dds_raw_program_sha256": dds_program_hashes,
        "failure_messages": sorted(
            {
                str(message)
                for record in failures
                for message in record.get("failures", [])
            }
        ),
    }


__all__ = [
    "COMMAND_AUDIT_SCHEMA_VERSION",
    "CommandAuditError",
    "CommandAuditSpec",
    "SUPPORTED_SEQUENCE_TYPES",
    "audit_compiled_commands",
    "spec_from_config",
    "summarize_command_audits",
]
