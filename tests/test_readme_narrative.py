"""Narrative and figure-budget guards for the public landing page."""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _displayed_assets(text: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for pattern in (
        r"!\[[^\]]*\]\(([^)]+)\)",
        r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']",
    ):
        matches.extend(
            (match.start(), match.group(1))
            for match in re.finditer(pattern, text)
        )
    return [value for _offset, value in sorted(matches)]


def test_readme_has_compact_data_first_narrative():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 230 <= len(lines) <= 290
    headings = [
        "## From fluorescence images to measurements",
        "## Data and experiments",
        "## Statistical design",
        "## Operational loss-sweep results",
        "## Background and robustness",
        "## Scientific interpretation",
        "## Reproduce",
        "## Documentation",
        "## Limitations",
        "## License",
    ]
    assert [line for line in lines if line.startswith("## ")] == headings
    assert "## Layout" not in lines
    assert "## Status" not in lines
    assert lines.index(headings[0]) < 30
    assert (
        "From raw neutral-atom fluorescence images to held-out occupancy "
        "inference and operational imaging-loss diagnostics."
    ) in text
    assert "cycles 0–5: training, six shots per condition" in text
    assert "cycles 6–7: validation, two shots per condition" in text
    assert "cycles 8–9: final test, two shots per condition" in text
    assert "Points summarize all 10 shots per condition" in text
    assert re.search(r"6\.57–10\.26\s+percentage-point range", text)
    assert "it is **not** a confidence interval or total uncertainty" in text
    assert "10.26 percentage points\n(8.81–11.61 percentage points)" in text
    assert "Apparent retention and occupancy on held-out shots" not in text


def test_readme_displays_only_the_four_primary_assets():
    text = README.read_text(encoding="utf-8")
    displayed = _displayed_assets(text)
    assert displayed == [
        "assets/readme/fluorescence_inference_overview.gif",
        "assets/readme/sequence_design.png",
        "assets/readme/loss_sweep_overview.png",
        "assets/readme/background_drift_sweeps.png",
    ]
    for relative in displayed:
        assert (ROOT / relative).is_file()

    hidden_from_landing_page = {
        "dark_hold_retention.png",
        "bright_wait_decay.png",
        "per_site_retention_map.png",
        "count_distribution_fit.png",
        "paired_frame_scatter.png",
        "site_summary_map.png",
        "background_method_comparison.png",
    }
    assert not hidden_from_landing_page.intersection(
        {Path(value).name for value in displayed}
    )


def test_generated_markers_are_exact_and_in_their_owned_sections():
    text = README.read_text(encoding="utf-8")
    sections = {
        "experiment-matrix": (
            "## Data and experiments",
            "## Statistical design",
            ROOT / "assets" / "readme" / "experiment_matrix.md",
        ),
        "loss-sweep-results": (
            "## Operational loss-sweep results",
            "## Background and robustness",
            ROOT / "assets" / "readme" / "loss_sweep_results.md",
        ),
        "sweep-validation": (
            "## Background and robustness",
            "## Scientific interpretation",
            ROOT / "assets" / "readme" / "sweep_validation.md",
        ),
    }
    for marker, (owner, next_section, fragment_path) in sections.items():
        begin = f"<!-- BEGIN:{marker} -->"
        end = f"<!-- END:{marker} -->"
        assert text.count(begin) == 1
        assert text.count(end) == 1
        assert text.index(owner) < text.index(begin) < text.index(end)
        assert text.index(end) < text.index(next_section)
        body = text.split(begin, 1)[1].split(end, 1)[0].strip()
        expected = fragment_path.read_text(encoding="utf-8").strip()
        assert body == expected


def test_headline_table_and_claim_boundary_are_exactly_scoped():
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- BEGIN:loss-sweep-results -->(.*?)"
        r"<!-- END:loss-sweep-results -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    fragment = match.group(1)
    table_rows = [
        line
        for line in fragment.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    data_rows = table_rows[1:]
    assert len(data_rows) == 3
    assert "`tau_switch_off`" in data_rows[0]
    assert "`tau_bright_effective`" in data_rows[1]
    assert "observed − predicted 50 ms apparent-loss gap" in data_rows[2]
    assert "6.57–10.26 percentage points" in data_rows[2]
    assert (
        "The sign remains positive, but the selected-model sampling interval "
        "is not total model uncertainty."
    ) in fragment
    assert (
        "The simple constant-rate bright-wait model does not explain the full "
        "apparent inter-readout loss. This does not establish a fixed per-pulse "
        "mechanism."
    ) in fragment


def test_removed_figures_remain_linked_from_detailed_documents():
    validation = (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
    interpretation = (
        ROOT / "docs" / "LOSS_SWEEP_INTERPRETATION.md"
    ).read_text(encoding="utf-8")
    for filename in (
        "count_distribution_fit.png",
        "paired_frame_scatter.png",
        "site_summary_map.png",
        "background_method_comparison.png",
    ):
        assert filename in validation
    for filename in (
        "dark_hold_retention.png",
        "bright_wait_decay.png",
        "per_site_retention_map.png",
    ):
        assert filename in interpretation
