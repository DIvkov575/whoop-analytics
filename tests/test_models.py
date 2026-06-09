from datetime import datetime, timezone
from whoop_analytics.api.models import SleepRecord, RecoveryRecord, JournalEntry, CycleRecord


def test_sleep_record_from_api_response():
    raw = {
        "id": 12345,
        "user_id": 99,
        "created_at": "2026-01-15T07:30:00.000Z",
        "updated_at": "2026-01-15T07:30:00.000Z",
        "start": "2026-01-14T22:15:00.000Z",
        "end": "2026-01-15T06:45:00.000Z",
        "timezone_offset": "-05:00",
        "nap": False,
        "score": {
            "stage_summary": {
                "total_in_bed_time_milli": 30600000,
                "total_awake_time_milli": 1800000,
                "total_no_data_time_milli": 0,
                "total_light_sleep_time_milli": 14400000,
                "total_slow_wave_sleep_time_milli": 7200000,
                "total_rem_sleep_time_milli": 5400000,
                "sleep_cycle_count": 4,
                "disturbance_count": 3,
            },
            "sleep_needed": {"baseline_milli": 28800000, "need_from_sleep_debt_milli": 3600000},
            "respiratory_rate": 15.2,
            "sleep_performance_percentage": 87.5,
            "sleep_consistency_percentage": 82.0,
            "sleep_efficiency_percentage": 94.1,
        },
    }

    record = SleepRecord.from_api(raw)

    assert record.id == 12345
    assert record.nap is False
    assert record.total_sleep_minutes == 450.0
    assert record.sws_minutes == 120.0
    assert record.rem_minutes == 90.0
    assert record.light_minutes == 240.0
    assert record.disturbance_count == 3
    assert record.respiratory_rate == 15.2
    assert record.sleep_efficiency == 94.1
    assert record.sleep_debt_minutes == 60.0


def test_recovery_record_from_api_response():
    raw = {
        "cycle_id": 100,
        "sleep_id": 12345,
        "user_id": 99,
        "created_at": "2026-01-15T07:30:00.000Z",
        "updated_at": "2026-01-15T07:30:00.000Z",
        "score": {
            "user_calibrating": False,
            "recovery_score": 72.0,
            "resting_heart_rate": 52.0,
            "hrv_rmssd_milli": 65.3,
            "spo2_percentage": 97.5,
            "skin_temp_celsius": 33.2,
        },
    }

    record = RecoveryRecord.from_api(raw)

    assert record.cycle_id == 100
    assert record.created_at is not None
    assert record.recovery_score == 72.0
    assert record.hrv_rmssd == 65.3
    assert record.resting_hr == 52.0
    assert record.spo2 == 97.5
    assert record.skin_temp == 33.2


def test_journal_entry_from_api_response():
    raw = {
        "id": 500,
        "user_id": 99,
        "created_at": "2026-01-15T20:00:00.000Z",
        "updated_at": "2026-01-15T20:00:00.000Z",
        "answers": [
            {"id": 1, "question_id": 100, "text": "Brain Fog", "value": 3},
            {"id": 2, "question_id": 101, "text": "Caffeine", "value": 1},
            {"id": 3, "question_id": 102, "text": "Alcohol", "value": 0},
        ],
    }

    entry = JournalEntry.from_api(raw)

    assert entry.id == 500
    assert entry.answers["Brain Fog"] == 3
    assert entry.answers["Caffeine"] == 1
    assert entry.answers["Alcohol"] == 0


def test_sleep_record_handles_missing_score():
    raw = {
        "id": 999,
        "user_id": 99,
        "created_at": "2026-01-15T07:30:00.000Z",
        "updated_at": "2026-01-15T07:30:00.000Z",
        "start": "2026-01-14T22:15:00.000Z",
        "end": "2026-01-15T06:45:00.000Z",
        "timezone_offset": "-05:00",
        "nap": False,
        "score": None,
    }

    record = SleepRecord.from_api(raw)

    assert record.id == 999
    assert record.total_sleep_minutes is None
    assert record.sws_minutes is None
    assert record.respiratory_rate is None
