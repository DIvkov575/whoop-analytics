from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dowhy import CausalModel

from whoop_analytics.analysis.discovery import CausalLink


@dataclass(frozen=True)
class EffectResult:
    source: str
    target: str
    ate: float
    confidence_interval: tuple[float, float] | None
    refutation_passed: bool | None
    refutation_p_value: float | None
    method: str


@dataclass
class EffectEstimator:
    method: str = "backdoor.linear_regression"

    def estimate(
        self,
        df: pd.DataFrame,
        link: CausalLink,
        common_causes: list[str],
    ) -> EffectResult:
        treatment = link.source
        outcome = link.target

        if link.lag > 0:
            df = df.copy()
            df[f"{treatment}_lagged"] = df[treatment].shift(link.lag)
            df = df.dropna()
            treatment = f"{treatment}_lagged"

        graph_dot = self._build_graph(treatment, outcome, common_causes)

        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes if common_causes else None,
            graph=graph_dot,
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified, method_name=self.method)

        ate = float(estimate.value)

        refutation_passed = None
        refutation_p_value = None
        try:
            refutation = model.refute_estimate(
                identified,
                estimate,
                method_name="random_common_cause",
                num_simulations=100,
            )
            refutation_p_value = float(refutation.refutation_result.get("p_value", 0.0))
            refutation_passed = refutation_p_value > 0.05
        except Exception:
            pass

        return EffectResult(
            source=link.source,
            target=link.target,
            ate=ate,
            confidence_interval=None,
            refutation_passed=refutation_passed,
            refutation_p_value=refutation_p_value,
            method=self.method,
        )

    def _build_graph(self, treatment: str, outcome: str, common_causes: list[str]) -> str:
        edges = [f'"{treatment}" -> "{outcome}"']
        for cause in common_causes:
            edges.append(f'"{cause}" -> "{treatment}"')
            edges.append(f'"{cause}" -> "{outcome}"')

        all_nodes = set([treatment, outcome] + common_causes)
        nodes = "; ".join(f'"{n}"' for n in all_nodes)
        edge_str = "; ".join(edges)

        return f"digraph {{ {nodes}; {edge_str} }}"
