"""Presentation-story and scientific-claim guards for the public landing page."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _displayed_assets(text: str) -> list[str]:
    patterns = (
        r'<source[^>]+srcset=["\'](assets/readme/[^"\']+)["\']',
        r"!\[[^\]]*\]\((assets/readme/[^)]+)\)",
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        matches.extend(
            (match.start(), match.group(1)) for match in re.finditer(pattern, text)
        )
    return [value for _offset, value in sorted(matches)]


def test_readme_has_requested_compact_section_order():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 180 <= len(lines) <= 240
    headings = [
        "## From images to apparent occupancy",
        "## What was measured",
        "## Main results",
        "## Readout quality versus survival cost",
        "## Repeated imaging and pulse segmentation",
        "## Statistical contribution",
        "## Reproduce",
        "## Limitations",
    ]
    assert [line for line in lines if line.startswith("## ")] == headings
    assert lines.index(headings[0]) < 20
    assert lines.index(headings[1]) < lines.index(headings[2])
    assert lines.index(headings[3]) < lines.index(headings[4])
    opening = text.split(headings[1], 1)[0]
    assert "camera frame" in opening
    assert "apparent occupancy" in opening
    assert "held-out emission model" in opening
    assert "not empirical fidelity" in opening


def test_readme_displays_exactly_four_nonduplicated_primary_assets():
    text = README.read_text(encoding="utf-8")
    displayed = _displayed_assets(text)
    assert displayed == [
        "assets/readme/optimized_occupancy_inference.gif",
        "assets/readme/optimized_sequence_design.png",
        "assets/readme/optimized_lifetime_overview.png",
        "assets/readme/optimized_readout_tradeoff.png",
    ]
    assert len(displayed) == len(set(displayed)) == 4
    for relative in displayed:
        assert (ROOT / relative).is_file()
    assert "assets/readme/optimized_occupancy_inference.png" in text
    assert "assets/readme/exposure_segmentation_result.png" in text
    assert "exposure_segmentation_result.png" not in {
        Path(value).name for value in displayed
    }


def test_main_table_has_only_requested_results():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Main results", 1)[1].split(
        "## Readout quality versus survival cost", 1
    )[0]
    for phrase in (
        "dark operational lifetime",
        "bright operational lifetime",
        "model-implied overlap",
        "selected-model five-frame survival",
        "pulse segmentation",
    ):
        assert phrase in section
    assert "Mean NLL" not in section
    assert "Posterior entropy" not in section


def test_landing_page_moves_detailed_statistics_to_docs():
    text = README.read_text(encoding="utf-8")
    forbidden = (
        "Mean NLL",
        "mean NLL",
        "posterior entropy",
        "q₀ =",
        "2 µs",
        "robust spatial",
        "fixed template plus",
        "M2 exposure-specific",
        "M3 diagnostic",
        "Difference (95% interval)",
    )
    for phrase in forbidden:
        assert phrase not in text
    assert "public pulse-loss claim gate remains closed" in text
    assert "zero by selected-model\nstructure" in text
    assert "not a precisely measured physical zero" in text


def test_statistical_contribution_is_exactly_four_bullets():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Statistical contribution", 1)[1].split(
        "## Reproduce", 1
    )[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) == 4
    for phrase in (
        "Latent-class inference without external labels",
        "Frozen train/validation/test model selection",
        "Complete-shot and block-bootstrap uncertainty",
        "Materiality-aware model selection",
    ):
        assert phrase in section


def test_readme_links_all_detailed_claim_documents():
    text = README.read_text(encoding="utf-8")
    documents = (
        "docs/DATA_AUDIT_2026-07-31_OPTIMIZED_LIFETIMES.md",
        "docs/OPTIMIZED_LIFETIME_RESULTS.md",
        "docs/STATISTICAL_METHODS.md",
        "docs/LOSS_DECOMPOSITION.md",
        "docs/REPRODUCE.md",
        "docs/LOSS_SWEEP_INTERPRETATION.md",
    )
    for relative in documents:
        assert relative in text
        assert (ROOT / relative).is_file()


def test_old_benchmark_values_are_not_mixed_into_headline():
    text = README.read_text(encoding="utf-8")
    main = text.split("## Main results", 1)[1].split(
        "## Readout quality versus survival cost", 1
    )[0]
    assert "22.29 s" not in main
    assert "1.64 s" not in main
    assert "10.26 percentage points" not in main
