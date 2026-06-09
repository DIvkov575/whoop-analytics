import pandas as pd
import numpy as np
import pytest

from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features


@pytest.fixture
def daily_df():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "hrv_rmssd": np.linspace(50, 70, 10),
        "total_sleep_minutes": np.linspace(400, 480, 10),
        "brain_fog": [2, 3, 1, 4, 2, 3, 1, 2, 4, 3],
    }, index=dates)


def test_add_lag_features_creates_columns(daily_df):
    result = add_lag_features(daily_df, columns=["hrv_rmssd"], lags=[1, 2])

    assert "hrv_rmssd_lag1" in result.columns
    assert "hrv_rmssd_lag2" in result.columns
    assert pd.isna(result["hrv_rmssd_lag1"].iloc[0])
    assert result["hrv_rmssd_lag1"].iloc[1] == daily_df["hrv_rmssd"].iloc[0]


def test_add_rolling_features_creates_columns(daily_df):
    result = add_rolling_features(daily_df, columns=["hrv_rmssd"], windows=[3])

    assert "hrv_rmssd_roll3_mean" in result.columns
    assert "hrv_rmssd_roll3_std" in result.columns
    assert pd.isna(result["hrv_rmssd_roll3_mean"].iloc[0])
    assert pd.isna(result["hrv_rmssd_roll3_mean"].iloc[1])


def test_add_lag_features_handles_empty_dataframe():
    df = pd.DataFrame(columns=["hrv_rmssd", "brain_fog"])
    result = add_lag_features(df, columns=["hrv_rmssd"], lags=[1])
    assert "hrv_rmssd_lag1" in result.columns
    assert len(result) == 0
