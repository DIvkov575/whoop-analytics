from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from whoop_analytics.analysis.discovery import CausalLink
from whoop_analytics.analysis.estimation import EffectResult


@dataclass
class AnalysisReport:
    generated_date: date
    data_start: date
    data_end: date
    n_observations: int
    causal_links: list[CausalLink]
    effects: list[EffectResult]


@dataclass
class ReportGenerator:
    def render(self, report: AnalysisReport) -> str:
        sections = [
            self._header(report),
            self._data_summary(report),
            self._causal_links_section(report),
            self._effects_section(report),
            self._interpretation(report),
        ]
        return "\n\n".join(sections)

    def save(self, report: AnalysisReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(report))

    def _header(self, report: AnalysisReport) -> str:
        return f"# Causal Analysis Report\n\n**Generated:** {report.generated_date}"

    def _data_summary(self, report: AnalysisReport) -> str:
        return (
            f"## Data Summary\n\n"
            f"- **Period:** {report.data_start} to {report.data_end}\n"
            f"- **Observations:** {report.n_observations} days"
        )

    def _causal_links_section(self, report: AnalysisReport) -> str:
        if not report.causal_links:
            return "## Causal Discovery\n\nNo causal links found at the specified significance level."

        lines = ["## Causal Discovery\n"]
        lines.append("| Source | → Target | Lag (days) | Strength | p-value |")
        lines.append("|--------|----------|-----------|----------|---------|")

        for link in report.causal_links:
            lines.append(
                f"| {link.source} | {link.target} | {link.lag} | "
                f"{link.strength:.3f} | {link.p_value:.4f} |"
            )

        return "\n".join(lines)

    def _effects_section(self, report: AnalysisReport) -> str:
        if not report.effects:
            return ""

        lines = ["## Effect Estimation\n"]
        for effect in report.effects:
            status = "Robust" if effect.refutation_passed else "Failed refutation"
            ci_str = ""
            if effect.confidence_interval:
                ci_str = f" (95% CI: [{effect.confidence_interval[0]:.3f}, {effect.confidence_interval[1]:.3f}])"

            lines.append(
                f"### {effect.source} → {effect.target}\n\n"
                f"- **Average Treatment Effect:** {effect.ate:.4f}{ci_str}\n"
                f"- **Refutation:** {status} (p={effect.refutation_p_value:.3f})\n"
                f"- **Method:** {effect.method}"
            )

        return "\n\n".join(lines)

    def _interpretation(self, report: AnalysisReport) -> str:
        if not report.causal_links:
            return (
                "## Interpretation\n\n"
                "No significant causal relationships were found between the measured "
                "variables and brain fog. This could indicate insufficient data, "
                "missing confounders, or genuinely no causal effect."
            )

        robust_effects = [e for e in report.effects if e.refutation_passed]
        if not robust_effects:
            return (
                "## Interpretation\n\n"
                "Causal links were discovered but none passed refutation tests. "
                "Results should be treated with caution."
            )

        lines = ["## Interpretation\n"]
        lines.append("The following causal relationships passed robustness checks:\n")
        for effect in robust_effects:
            direction = "increases" if effect.ate > 0 else "decreases"
            lines.append(
                f"- **{effect.source}** {direction} **{effect.target}** "
                f"(ATE: {effect.ate:.4f}, refutation p={effect.refutation_p_value:.3f})"
            )

        return "\n".join(lines)
