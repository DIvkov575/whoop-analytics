from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from whoop_analytics.analysis.discovery import CausalDiscovery, DiscoveryResult
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult
from whoop_analytics.pipeline.transform import build_daily_dataset
from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features


@dataclass
class AnalysisState:
    daily_df: pd.DataFrame
    discovery_result: DiscoveryResult | None
    effects: list[EffectResult]


def load_daily_data(data_dir: Path) -> pd.DataFrame:
    return build_daily_dataset(data_dir)


def run_analysis(
    data_dir: Path,
    target: str = "brain_fog",
    max_lag: int = 3,
    alpha: float = 0.05,
) -> AnalysisState:
    df = build_daily_dataset(data_dir)

    if df.empty:
        return AnalysisState(daily_df=df, discovery_result=None, effects=[])

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    if len(df) < 30:
        return AnalysisState(daily_df=df, discovery_result=None, effects=[])

    discovery = CausalDiscovery(max_lag=max_lag, significance_level=alpha)
    disc_result = discovery.run(df, target=target)

    effects = []
    if disc_result.links:
        estimator = EffectEstimator()
        for link in disc_result.links:
            if link.source == target:
                continue
            effect = estimator.estimate(df=df, link=link, common_causes=[])
            effects.append(effect)

    return AnalysisState(
        daily_df=df,
        discovery_result=disc_result,
        effects=effects,
    )
