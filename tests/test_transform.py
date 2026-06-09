import pandas as pd
import numpy as np
from pathlib import Path

import pytest

from whoop_analytics.pipeline.transform import build_daily_dataset


@pytest.fixture
def raw_data(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sleep_df = pd.DataFrame([
        {"id": 1, "start": "2026-01-14T22:00:00", "end": "2026-01-15T06:00:00", "nap": False,
         "total_sleep_minutes": 450.0, "sws_minutes": 120.0, "rem_minutes": 90.0,
         "light_minutes": 240.0, "disturbance_count": 3, "respiratory_rate": 15.0,
         "sleep_efficiency": 94.0, "sleep_debt_minutes": 30.0},
        {"id": 2, "start": "2026-01-15T22:00:00", "end": "2026-01-16T06:00:00", "nap": False,
         "total_sleep_minutes": 420.0, "sws_minutes": 100.0, "rem_minutes": 80.0,
         "light_minutes": 240.0, "disturbance_count": 5, "respiratory_rate": 16.0,
         "sleep_efficiency": 88.0, "sleep_debt_minutes": 60.0},
    ])
    sleep_df.to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_df = pd.DataFrame([
        {"cycle_id": 100, "created_at": "2026-01-15T07:00:00", "recovery_score": 72.0,
         "hrv_rmssd": 65.0, "resting_hr": 52.0, "spo2": 97.5, "skin_temp": 33.0},
        {"cycle_id": 101, "created_at": "2026-01-16T07:00:00", "recovery_score": 55.0,
         "hrv_rmssd": 45.0, "resting_hr": 58.0, "spo2": 96.0, "skin_temp": 33.5},
    ])
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame([
        {"id": 500, "created_at": "2026-01-15T20:00:00", "Brain Fog": 2, "Caffeine": 1},
        {"id": 501, "created_at": "2026-01-16T20:00:00", "Brain Fog": 4, "Caffeine": 2},
    ])
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    return tmp_path


def test_build_daily_dataset_merges_all_sources(raw_data):
    df = build_daily_dataset(raw_data)

    assert len(df) == 2
    assert "hrv_rmssd" in df.columns
    assert "total_sleep_minutes" in df.columns
    assert "brain_fog" in df.columns


def test_build_daily_dataset_aligns_sleep_to_wake_date(raw_data):
    df = build_daily_dataset(raw_data)

    assert df.index[0].strftime("%Y-%m-%d") == "2026-01-15"
    assert df.index[1].strftime("%Y-%m-%d") == "2026-01-16"


def test_build_daily_dataset_excludes_naps(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sleep_df = pd.DataFrame([
        {"id": 1, "start": "2026-01-14T22:00:00", "end": "2026-01-15T06:00:00", "nap": False,
         "total_sleep_minutes": 450.0, "sws_minutes": 120.0, "rem_minutes": 90.0,
         "light_minutes": 240.0, "disturbance_count": 3, "respiratory_rate": 15.0,
         "sleep_efficiency": 94.0, "sleep_debt_minutes": 30.0},
        {"id": 2, "start": "2026-01-15T13:00:00", "end": "2026-01-15T13:30:00", "nap": True,
         "total_sleep_minutes": 30.0, "sws_minutes": 10.0, "rem_minutes": 5.0,
         "light_minutes": 15.0, "disturbance_count": 0, "respiratory_rate": 14.0,
         "sleep_efficiency": 100.0, "sleep_debt_minutes": 0.0},
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

    df = build_daily_dataset(tmp_path)

    assert len(df) == 1
