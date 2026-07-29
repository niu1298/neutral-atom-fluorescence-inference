"""Narrative and figure-budget guards for the public landing page."""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _displayed_assets(text: str) -> list[str]:
    markdown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    html = re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", text)
    return markdown + html


def test_readme_has_compact_analysis_first_narrative():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 180 <= len(lines) <= 250
    assert lines.index("## Main result") < lines.index("## Experiments")
    assert lines.index("## Experiments") < lines.index("## Method and data pipeline")
    assert lines.index("## Method and data pipeline") < lines.index(
        "## Background and robustness"
    )
    assert lines.index("## Background and robustness") < lines.index(
        "## Scientific interpretation"
    )
    assert "## Layout" not in lines
    assert "## Status" not in lines
    assert lines.index("## Main result") < 30
    assert "Points summarize all 10 shots per condition" in text
    assert "Apparent retention and occupancy on held-out shots" not in text


def test_readme_displays_only_the_four_primary_assets():
    text = README.read_text(encoding="utf-8")
    displayed = _displayed_assets(text)
    assert displayed == [
        "assets/readme/loss_sweep_overview.png",
        "assets/readme/sequence_design.png",
        "assets/readme/background_drift_sweeps.png",
        "assets/readme/fluorescence_inference_overview.gif",
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
