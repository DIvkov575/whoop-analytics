from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag)
    return result


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        for window in windows:
            result[f"{col}_roll{window}_mean"] = result[col].rolling(window).mean()
            result[f"{col}_roll{window}_std"] = result[col].rolling(window).std()
    return result
