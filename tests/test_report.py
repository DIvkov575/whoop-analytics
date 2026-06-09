from datetime import date

import pytest

from whoop_analytics.analysis.discovery import CausalLink
from whoop_analytics.analysis.estimation import EffectResult
from whoop_analytics.report.generator import ReportGenerator, AnalysisReport


@pytest.fixture
def sample_links():
    return [
        CausalLink(source="hrv_rmssd", target="brain_fog", lag=1, strength=-0.45, p_value=0.001),
        CausalLink(source="sws_minutes", target="brain_fog", lag=1, strength=-0.30, p_value=0.01),
    ]


@pytest.fixture
def sample_effects():
    return [
        EffectResult(
            source="hrv_rmssd", target="brain_fog", ate=-0.42,
            confidence_interval=(-0.55, -0.29),
            refutation_passed=True, refutation_p_value=0.82,
            method="backdoor.linear_regression",
        ),
        EffectResult(
            source="sws_minutes", target="brain_fog", ate=-0.015,
            confidence_interval=(-0.025, -0.005),
            refutation_passed=True, refutation_p_value=0.67,
            method="backdoor.linear_regression",
        ),
    ]


def test_report_generates_markdown(sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "# Causal Analysis Report" in md
    assert "hrv_rmssd" in md
    assert "brain_fog" in md
    assert "158" in md
    assert "2026-06-08" in md


def test_report_includes_effect_interpretation(sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "refutation" in md.lower() or "robust" in md.lower()


def test_report_handles_empty_links():
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=50,
        causal_links=[],
        effects=[],
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "no significant causal" in md.lower() or "no causal links" in md.lower()


def test_report_saves_to_file(tmp_path, sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    output_path = tmp_path / "report.md"
    generator.save(report, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "# Causal Analysis Report" in content
