import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from whoop_analytics.api.client import WhoopClient
from whoop_analytics.api.auth import TokenManager


@pytest.fixture
def token_file(tmp_path):
    token_path = tmp_path / "tokens.json"
    token_path.write_text(json.dumps({
        "access_token": "test_token",
        "refresh_token": "test_refresh",
        "expires_at": time.time() + 3600,
    }))
    return token_path


@pytest.fixture
def client(token_file):
    token_manager = TokenManager(
        client_id="id",
        client_secret="secret",
        redirect_uri="http://localhost:8080/callback",
        token_file=token_file,
    )
    return WhoopClient(token_manager=token_manager)


@respx.mock
def test_client_fetches_sleep_records(client):
    respx.get("https://api.prod.whoop.com/developer/v1/activity/sleep").mock(
        return_value=httpx.Response(200, json={
            "records": [
                {
                    "id": 1,
                    "user_id": 99,
                    "created_at": "2026-01-15T07:30:00.000Z",
                    "updated_at": "2026-01-15T07:30:00.000Z",
                    "start": "2026-01-14T22:00:00.000Z",
                    "end": "2026-01-15T06:00:00.000Z",
                    "timezone_offset": "-05:00",
                    "nap": False,
                    "score": {
                        "stage_summary": {
                            "total_in_bed_time_milli": 28800000,
                            "total_awake_time_milli": 1800000,
                            "total_no_data_time_milli": 0,
                            "total_light_sleep_time_milli": 13000000,
                            "total_slow_wave_sleep_time_milli": 7000000,
                            "total_rem_sleep_time_milli": 5000000,
                            "sleep_cycle_count": 4,
                            "disturbance_count": 2,
                        },
                        "sleep_needed": {"baseline_milli": 28800000, "need_from_sleep_debt_milli": 0},
                        "respiratory_rate": 14.5,
                        "sleep_performance_percentage": 90.0,
                        "sleep_consistency_percentage": 85.0,
                        "sleep_efficiency_percentage": 93.75,
                    },
                }
            ],
            "next_token": None,
        })
    )

    records = client.get_sleep_records(start_date="2026-01-14", end_date="2026-01-15")

    assert len(records) == 1
    assert records[0].id == 1
    assert records[0].total_sleep_minutes is not None


@respx.mock
def test_client_paginates(client):
    respx.get("https://api.prod.whoop.com/developer/v1/activity/sleep").mock(
        side_effect=[
            httpx.Response(200, json={
                "records": [{"id": 1, "user_id": 99, "created_at": "2026-01-15T07:30:00.000Z", "updated_at": "2026-01-15T07:30:00.000Z", "start": "2026-01-14T22:00:00.000Z", "end": "2026-01-15T06:00:00.000Z", "timezone_offset": "-05:00", "nap": False, "score": None}],
                "next_token": "page2",
            }),
            httpx.Response(200, json={
                "records": [{"id": 2, "user_id": 99, "created_at": "2026-01-16T07:30:00.000Z", "updated_at": "2026-01-16T07:30:00.000Z", "start": "2026-01-15T22:00:00.000Z", "end": "2026-01-16T06:00:00.000Z", "timezone_offset": "-05:00", "nap": False, "score": None}],
                "next_token": None,
            }),
        ]
    )

    records = client.get_sleep_records(start_date="2026-01-14", end_date="2026-01-16")

    assert len(records) == 2
    assert records[0].id == 1
    assert records[1].id == 2


@respx.mock
def test_client_respects_rate_limit(client):
    call_count = 0

    def rate_limited(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"records": [], "next_token": None})

    respx.get("https://api.prod.whoop.com/developer/v1/activity/sleep").mock(side_effect=rate_limited)

    records = client.get_sleep_records(start_date="2026-01-14", end_date="2026-01-15")

    assert records == []
    assert call_count == 2
