"""Landing-page scope and scientific-claim guards."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _displayed_assets(text: str) -> list[str]:
    patterns = (
        r"!\[[^\]]*\]\((assets/readme/[^)]+)\)",
        r'<img[^>]+src=["\'](assets/readme/[^"\']+)["\']',
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        matches.extend(
            (match.start(), match.group(1))
            for match in re.finditer(pattern, text)
        )
    return [value for _offset, value in sorted(matches)]


def test_readme_is_compact_and_result_first():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 200 <= len(lines) <= 280
    headings = [
        "## From images to apparent occupancy",
        "## Main results",
        "## Experiments and timing",
        "## Repeated imaging and loss attribution",
        "## Statistical design",
        "## Reproduce",
        "## Documentation",
        "## Limitations",
        "## License",
    ]
    assert [line for line in lines if line.startswith("## ")] == headings
    assert lines.index(headings[0]) < 15
    assert lines.index(headings[1]) < lines.index(headings[2])
    assert (
        "From raw fluorescence images to held-out latent-occupancy inference, "
        "operational lifetimes, and repeated-imaging loss attribution."
    ) in text
    assert "35.73 s" in text
    assert "20.30 s" in text
    assert "continuous-only" in text
    assert "pre-optimization benchmark" in text


def test_readme_displays_exactly_four_primary_assets():
    text = README.read_text(encoding="utf-8")
    displayed = _displayed_assets(text)
    assert displayed == [
        "assets/readme/optimized_occupancy_inference.png",
        "assets/readme/optimized_lifetime_overview.png",
        "assets/readme/optimized_sequence_design.png",
        "assets/readme/exposure_segmentation_result.png",
    ]
    for relative in displayed:
        assert (ROOT / relative).is_file()
    assert "fluorescence_inference_overview.gif" not in displayed
    assert "loss_sweep_overview.png" not in displayed


def test_readme_claim_boundaries_are_explicit():
    text = README.read_text(encoding="utf-8")
    assert "model-implied component overlap" in text
    assert "empirical readout\nfidelity" in text
    assert "posterior transition as an observed loss timestamp" in text
    assert "0% by structure" in text
    assert "unselected sensitivity" in text
    assert "public pulse-loss claim gate remains closed" in text
    assert "rate-based prior attributions" in text
    assert "count-conditioned posterior\nlocation" in text


def test_readme_methods_are_four_compact_bullets():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Statistical design", 1)[1].split(
        "## Reproduce", 1
    )[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) == 4
    assert "training data only" in section
    assert "1,000" in section
    assert "block bootstrap" in section
    assert "matched-prefix" in section


def test_readme_links_detailed_claim_documents():
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
        "## Experiments and timing", 1
    )[0]
    assert "22.29 s" not in main
    assert "1.64 s" not in main
    assert "10.26 percentage points" not in main
