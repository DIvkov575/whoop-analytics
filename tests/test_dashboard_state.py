import pandas as pd
import numpy as np
from pathlib import Path

import pytest

from whoop_analytics.dashboard.state import load_daily_data, run_analysis, AnalysisState


@pytest.fixture
def data_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sleep_df = pd.DataFrame([
        {"id": 1, "start": "2026-01-14T22:00:00", "end": "2026-01-15T06:00:00", "nap": False,
         "total_sleep_minutes": 450.0, "sws_minutes": 120.0, "rem_minutes": 90.0,
         "light_minutes": 240.0, "disturbance_count": 3, "respiratory_rate": 15.0,
         "sleep_efficiency": 94.0, "sleep_debt_minutes": 30.0},
    ])
    sleep_df.to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_df = pd.DataFrame([
        {"cycle_id": 100, "created_at": "2026-01-15T07:00:00", "recovery_score": 72.0,
         "hrv_rmssd": 65.0, "resting_hr": 52.0, "spo2": 97.5, "skin_temp": 33.0},
    ])
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame([
        {"id": 500, "created_at": "2026-01-15T20:00:00", "Brain Fog": 2, "Caffeine": 1},
    ])
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    return tmp_path


def test_load_daily_data_returns_dataframe(data_dir):
    df = load_daily_data(data_dir)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "hrv_rmssd" in df.columns
    assert "brain_fog" in df.columns


def test_load_daily_data_returns_empty_when_no_data(tmp_path):
    df = load_daily_data(tmp_path)
    assert df.empty


def test_run_analysis_returns_state(data_dir):
    np.random.seed(42)
    n = 60
    raw_dir = data_dir / "raw"

    sleep_df = pd.DataFrame({
        "id": range(n),
        "start": pd.date_range("2026-01-01 22:00", periods=n, freq="D").astype(str),
        "end": pd.date_range("2026-01-02 06:00", periods=n, freq="D").astype(str),
        "nap": [False] * n,
        "total_sleep_minutes": np.random.uniform(380, 480, n),
        "sws_minutes": np.random.uniform(80, 140, n),
        "rem_minutes": np.random.uniform(60, 100, n),
        "light_minutes": np.random.uniform(200, 260, n),
        "disturbance_count": np.random.randint(0, 8, n),
        "respiratory_rate": np.random.uniform(13, 17, n),
        "sleep_efficiency": np.random.uniform(85, 98, n),
        "sleep_debt_minutes": np.random.uniform(0, 90, n),
    })
    sleep_df.to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_df = pd.DataFrame({
        "cycle_id": range(n),
        "created_at": pd.date_range("2026-01-02 07:00", periods=n, freq="D").astype(str),
        "recovery_score": np.random.uniform(40, 90, n),
        "hrv_rmssd": np.random.uniform(30, 80, n),
        "resting_hr": np.random.uniform(45, 65, n),
        "spo2": np.random.uniform(95, 99, n),
        "skin_temp": np.random.uniform(32, 34, n),
    })
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame({
        "id": range(n),
        "created_at": pd.date_range("2026-01-02 20:00", periods=n, freq="D").astype(str),
        "Brain Fog": np.random.randint(1, 5, n),
        "Caffeine": np.random.randint(0, 3, n),
    })
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    state = run_analysis(data_dir, target="brain_fog", max_lag=2, alpha=0.05)

    assert isinstance(state, AnalysisState)
    assert state.daily_df is not None
    assert not state.daily_df.empty
    assert state.discovery_result is not None
    assert isinstance(state.effects, list)
