import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from whoop_analytics.api.models import SleepRecord, RecoveryRecord, JournalEntry
from whoop_analytics.pipeline.ingest import IngestPipeline


@pytest.fixture
def data_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    return tmp_path


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_sleep_records.return_value = [
        SleepRecord(
            id=1,
            start=datetime(2026, 1, 14, 22, 0),
            end=datetime(2026, 1, 15, 6, 0),
            nap=False,
            total_sleep_minutes=450.0,
            sws_minutes=120.0,
            rem_minutes=90.0,
            light_minutes=240.0,
            disturbance_count=3,
            respiratory_rate=15.0,
            sleep_efficiency=94.0,
            sleep_debt_minutes=30.0,
        ),
    ]
    client.get_recovery_records.return_value = [
        RecoveryRecord(
            cycle_id=100,
            created_at=datetime(2026, 1, 15, 7, 0),
            recovery_score=72.0,
            hrv_rmssd=65.0,
            resting_hr=52.0,
            spo2=97.5,
            skin_temp=33.0,
        ),
    ]
    client.get_journal_entries.return_value = [
        JournalEntry(
            id=500,
            created_at=datetime(2026, 1, 15, 20, 0),
            answers={"Brain Fog": 3, "Caffeine": 1},
        ),
    ]
    return client


def test_ingest_saves_sleep_parquet(mock_client, data_dir):
    pipeline = IngestPipeline(client=mock_client, data_dir=data_dir)
    pipeline.ingest(start_date="2026-01-14", end_date="2026-01-15")

    sleep_file = data_dir / "raw" / "sleep.parquet"
    assert sleep_file.exists()
    df = pd.read_parquet(sleep_file)
    assert len(df) == 1
    assert df.iloc[0]["total_sleep_minutes"] == 450.0


def test_ingest_saves_recovery_parquet(mock_client, data_dir):
    pipeline = IngestPipeline(client=mock_client, data_dir=data_dir)
    pipeline.ingest(start_date="2026-01-14", end_date="2026-01-15")

    recovery_file = data_dir / "raw" / "recovery.parquet"
    assert recovery_file.exists()
    df = pd.read_parquet(recovery_file)
    assert len(df) == 1
    assert df.iloc[0]["hrv_rmssd"] == 65.0


def test_ingest_saves_journal_parquet(mock_client, data_dir):
    pipeline = IngestPipeline(client=mock_client, data_dir=data_dir)
    pipeline.ingest(start_date="2026-01-14", end_date="2026-01-15")

    journal_file = data_dir / "raw" / "journal.parquet"
    assert journal_file.exists()
    df = pd.read_parquet(journal_file)
    assert len(df) == 1
    assert df.iloc[0]["Brain Fog"] == 3


def test_ingest_handles_empty_responses(data_dir):
    client = MagicMock()
    client.get_sleep_records.return_value = []
    client.get_recovery_records.return_value = []
    client.get_journal_entries.return_value = []

    pipeline = IngestPipeline(client=client, data_dir=data_dir)
    pipeline.ingest(start_date="2026-01-14", end_date="2026-01-15")

    sleep_file = data_dir / "raw" / "sleep.parquet"
    assert not sleep_file.exists()
