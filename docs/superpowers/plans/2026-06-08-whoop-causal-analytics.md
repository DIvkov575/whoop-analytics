# Whoop Causal Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python application that fetches sleep/HRV/journal data from the Whoop API and performs causal inference to identify what drives brain fog.

**Architecture:** A CLI-driven pipeline with three stages: (1) data ingestion from Whoop's OAuth 2.0 API into local Parquet files, (2) causal discovery via Tigramite's PCMCI algorithm to learn time-series causal structure, (3) effect estimation via DoWhy with refutation tests, producing a Markdown report. Each stage is independently runnable and testable.

**Tech Stack:** Python 3.9+, httpx (async HTTP), Tigramite (PCMCI causal discovery), DoWhy (effect estimation), pandas, numpy, pytest, python-dotenv

---

## File Structure

```
whoop-analytics/
├── pyproject.toml              # Project metadata + dependencies
├── .env.example                # Template for OAuth credentials
├── .gitignore                  # Python/venv/env patterns
├── src/
│   └── whoop_analytics/
│       ├── __init__.py
│       ├── cli.py              # CLI entrypoint (argparse)
│       ├── config.py           # Settings from env vars
│       ├── api/
│       │   ├── __init__.py
│       │   ├── auth.py         # OAuth 2.0 token management
│       │   ├── client.py       # Whoop API client (rate-limited)
│       │   └── models.py       # Response dataclasses
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── ingest.py       # Raw API → Parquet
│       │   ├── transform.py    # Parquet → analysis-ready DataFrame
│       │   └── features.py     # Feature engineering (lags, rolling)
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── discovery.py    # Tigramite PCMCI causal discovery
│       │   └── estimation.py   # DoWhy effect estimation + refutation
│       └── report/
│           ├── __init__.py
│           └── generator.py    # Markdown report generation
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_config.py
│   ├── test_auth.py
│   ├── test_client.py
│   ├── test_models.py
│   ├── test_ingest.py
│   ├── test_transform.py
│   ├── test_features.py
│   ├── test_discovery.py
│   ├── test_estimation.py
│   └── test_report.py
└── data/                       # Local data dir (gitignored)
    ├── raw/                    # Raw API responses as Parquet
    └── processed/              # Analysis-ready datasets
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/whoop_analytics/__init__.py`
- Create: `src/whoop_analytics/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/divkov/workplace/whoop-analytics
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "whoop-analytics"
version = "0.1.0"
description = "Causal inference on Whoop sleep/HRV data to explain brain fog"
requires-python = ">=3.9"
dependencies = [
    "httpx>=0.27",
    "pandas>=2.0",
    "numpy>=1.24",
    "python-dotenv>=1.0",
    "tigramite>=5.2",
    "dowhy>=0.11",
    "networkx>=3.0",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]

[project.scripts]
whoop-analytics = "whoop_analytics.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
.env
data/raw/
data/processed/
*.parquet
.pytest_cache/
```

- [ ] **Step 4: Create .env.example**

```
WHOOP_CLIENT_ID=your_client_id_here
WHOOP_CLIENT_SECRET=your_client_secret_here
WHOOP_REDIRECT_URI=http://localhost:8080/callback
WHOOP_ACCESS_TOKEN=
WHOOP_REFRESH_TOKEN=
```

- [ ] **Step 5: Create src/whoop_analytics/__init__.py**

```python
"""Whoop Analytics - Causal inference on sleep/HRV data."""
```

- [ ] **Step 6: Write the failing test for config**

Create `tests/test_config.py`:

```python
import os
from whoop_analytics.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("WHOOP_CLIENT_ID", "test_id")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("WHOOP_REDIRECT_URI", "http://localhost:8080/callback")

    settings = Settings.from_env()

    assert settings.client_id == "test_id"
    assert settings.client_secret == "test_secret"
    assert settings.redirect_uri == "http://localhost:8080/callback"


def test_settings_raises_when_missing(monkeypatch):
    monkeypatch.delenv("WHOOP_CLIENT_ID", raising=False)
    monkeypatch.delenv("WHOOP_CLIENT_SECRET", raising=False)

    try:
        Settings.from_env()
        assert False, "Should have raised"
    except ValueError as e:
        assert "WHOOP_CLIENT_ID" in str(e)


def test_settings_data_dir_defaults(monkeypatch):
    monkeypatch.setenv("WHOOP_CLIENT_ID", "x")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "x")
    settings = Settings.from_env()
    assert settings.data_dir.name == "data"
```

- [ ] **Step 7: Run test to verify it fails**

```bash
cd /Users/divkov/workplace/whoop-analytics
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'whoop_analytics.config'`

- [ ] **Step 8: Implement config.py**

Create `src/whoop_analytics/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    client_id: str
    client_secret: str
    redirect_uri: str
    access_token: str
    refresh_token: str
    data_dir: Path
    api_base_url: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        client_id = os.environ.get("WHOOP_CLIENT_ID", "")
        client_secret = os.environ.get("WHOOP_CLIENT_SECRET", "")

        missing = []
        if not client_id:
            missing.append("WHOOP_CLIENT_ID")
        if not client_secret:
            missing.append("WHOOP_CLIENT_SECRET")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=os.environ.get("WHOOP_REDIRECT_URI", "http://localhost:8080/callback"),
            access_token=os.environ.get("WHOOP_ACCESS_TOKEN", ""),
            refresh_token=os.environ.get("WHOOP_REFRESH_TOKEN", ""),
            data_dir=Path(os.environ.get("WHOOP_DATA_DIR", "data")),
            api_base_url=os.environ.get("WHOOP_API_BASE_URL", "https://api.prod.whoop.com/developer"),
        )
```

- [ ] **Step 9: Create tests/conftest.py**

```python
import os
import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests don't leak env vars."""
    for key in list(os.environ):
        if key.startswith("WHOOP_"):
            monkeypatch.delenv(key, raising=False)
```

- [ ] **Step 10: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 tests PASS

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/ tests/
git commit -m "feat: project scaffolding with config management"
```

---

### Task 2: Whoop API Response Models

**Files:**
- Create: `src/whoop_analytics/api/__init__.py`
- Create: `src/whoop_analytics/api/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test for models**

Create `tests/test_models.py`:

```python
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
    assert record.total_sleep_minutes == 450.0  # (30600000 - 1800000) / 60000
    assert record.sws_minutes == 120.0
    assert record.rem_minutes == 90.0
    assert record.light_minutes == 240.0
    assert record.disturbance_count == 3
    assert record.respiratory_rate == 15.2
    assert record.sleep_efficiency == 94.1
    assert record.sleep_debt_minutes == 60.0  # 3600000 / 60000


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement models**

Create `src/whoop_analytics/api/__init__.py`:

```python
"""Whoop API client package."""
```

Create `src/whoop_analytics/api/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _ms_to_min(ms: int | None) -> float | None:
    if ms is None:
        return None
    return ms / 60_000


@dataclass(frozen=True)
class SleepRecord:
    id: int
    start: datetime
    end: datetime
    nap: bool
    total_sleep_minutes: float | None
    sws_minutes: float | None
    rem_minutes: float | None
    light_minutes: float | None
    disturbance_count: int | None
    respiratory_rate: float | None
    sleep_efficiency: float | None
    sleep_debt_minutes: float | None

    @classmethod
    def from_api(cls, data: dict) -> SleepRecord:
        score = data.get("score")
        if score is None:
            return cls(
                id=data["id"],
                start=_parse_dt(data["start"]),
                end=_parse_dt(data["end"]),
                nap=data.get("nap", False),
                total_sleep_minutes=None,
                sws_minutes=None,
                rem_minutes=None,
                light_minutes=None,
                disturbance_count=None,
                respiratory_rate=None,
                sleep_efficiency=None,
                sleep_debt_minutes=None,
            )

        stages = score["stage_summary"]
        total_in_bed = stages["total_in_bed_time_milli"]
        total_awake = stages["total_awake_time_milli"]

        sleep_needed = score.get("sleep_needed", {})
        debt_ms = sleep_needed.get("need_from_sleep_debt_milli")

        return cls(
            id=data["id"],
            start=_parse_dt(data["start"]),
            end=_parse_dt(data["end"]),
            nap=data.get("nap", False),
            total_sleep_minutes=_ms_to_min(total_in_bed - total_awake),
            sws_minutes=_ms_to_min(stages["total_slow_wave_sleep_time_milli"]),
            rem_minutes=_ms_to_min(stages["total_rem_sleep_time_milli"]),
            light_minutes=_ms_to_min(stages["total_light_sleep_time_milli"]),
            disturbance_count=stages.get("disturbance_count"),
            respiratory_rate=score.get("respiratory_rate"),
            sleep_efficiency=score.get("sleep_efficiency_percentage"),
            sleep_debt_minutes=_ms_to_min(debt_ms),
        )


@dataclass(frozen=True)
class RecoveryRecord:
    cycle_id: int
    created_at: datetime
    recovery_score: float
    hrv_rmssd: float
    resting_hr: float
    spo2: float | None
    skin_temp: float | None

    @classmethod
    def from_api(cls, data: dict) -> RecoveryRecord:
        score = data["score"]
        return cls(
            cycle_id=data["cycle_id"],
            created_at=_parse_dt(data["created_at"]),
            recovery_score=score["recovery_score"],
            hrv_rmssd=score["hrv_rmssd_milli"],
            resting_hr=score["resting_heart_rate"],
            spo2=score.get("spo2_percentage"),
            skin_temp=score.get("skin_temp_celsius"),
        )


@dataclass(frozen=True)
class JournalEntry:
    id: int
    created_at: datetime
    answers: dict[str, int]

    @classmethod
    def from_api(cls, data: dict) -> JournalEntry:
        answers = {a["text"]: a["value"] for a in data.get("answers", [])}
        return cls(
            id=data["id"],
            created_at=_parse_dt(data["created_at"]),
            answers=answers,
        )


@dataclass(frozen=True)
class CycleRecord:
    id: int
    start: datetime
    end: datetime | None
    strain: float | None

    @classmethod
    def from_api(cls, data: dict) -> CycleRecord:
        score = data.get("score")
        return cls(
            id=data["id"],
            start=_parse_dt(data["start"]),
            end=_parse_dt(data["end"]) if data.get("end") else None,
            strain=score.get("strain") if score else None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/api/ tests/test_models.py
git commit -m "feat: Whoop API response models with parsing"
```

---

### Task 3: OAuth 2.0 Token Management

**Files:**
- Create: `src/whoop_analytics/api/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write the failing test for auth**

Create `tests/test_auth.py`:

```python
import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from whoop_analytics.api.auth import TokenManager


@pytest.fixture
def token_file(tmp_path):
    return tmp_path / "tokens.json"


@pytest.fixture
def token_manager(token_file):
    return TokenManager(
        client_id="test_id",
        client_secret="test_secret",
        redirect_uri="http://localhost:8080/callback",
        token_file=token_file,
    )


def test_token_manager_stores_tokens(token_manager, token_file):
    token_manager.save_tokens(access_token="abc", refresh_token="xyz", expires_in=3600)

    assert token_file.exists()
    data = json.loads(token_file.read_text())
    assert data["access_token"] == "abc"
    assert data["refresh_token"] == "xyz"
    assert "expires_at" in data


def test_token_manager_loads_valid_token(token_manager, token_file):
    token_file.write_text(json.dumps({
        "access_token": "valid",
        "refresh_token": "refresh",
        "expires_at": time.time() + 3600,
    }))

    assert token_manager.get_access_token() == "valid"


def test_token_manager_detects_expired_token(token_manager, token_file):
    token_file.write_text(json.dumps({
        "access_token": "expired",
        "refresh_token": "refresh",
        "expires_at": time.time() - 100,
    }))

    assert token_manager.is_expired()


@respx.mock
def test_token_manager_refreshes_expired_token(token_manager, token_file):
    token_file.write_text(json.dumps({
        "access_token": "old",
        "refresh_token": "old_refresh",
        "expires_at": time.time() - 100,
    }))

    respx.post("https://api.prod.whoop.com/oauth/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        })
    )

    token = token_manager.get_valid_token()

    assert token == "new_token"
    data = json.loads(token_file.read_text())
    assert data["access_token"] == "new_token"
    assert data["refresh_token"] == "new_refresh"


def test_token_manager_returns_none_when_no_file(token_manager):
    assert token_manager.get_access_token() is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_auth.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement auth.py**

Create `src/whoop_analytics/api/auth.py`:

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"


@dataclass
class TokenManager:
    client_id: str
    client_secret: str
    redirect_uri: str
    token_file: Path

    def save_tokens(self, access_token: str, refresh_token: str, expires_in: int) -> None:
        data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + expires_in,
        }
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps(data))

    def _load_tokens(self) -> dict | None:
        if not self.token_file.exists():
            return None
        return json.loads(self.token_file.read_text())

    def get_access_token(self) -> str | None:
        data = self._load_tokens()
        if data is None:
            return None
        return data["access_token"]

    def is_expired(self) -> bool:
        data = self._load_tokens()
        if data is None:
            return True
        return time.time() >= data["expires_at"]

    def get_valid_token(self) -> str:
        if not self.is_expired():
            return self.get_access_token()
        return self._refresh()

    def _refresh(self) -> str:
        data = self._load_tokens()
        if data is None:
            raise ValueError("No tokens stored. Run initial OAuth flow first.")

        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": data["refresh_token"],
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        response.raise_for_status()
        body = response.json()

        self.save_tokens(
            access_token=body["access_token"],
            refresh_token=body["refresh_token"],
            expires_in=body["expires_in"],
        )
        return body["access_token"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_auth.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/api/auth.py tests/test_auth.py
git commit -m "feat: OAuth 2.0 token management with refresh"
```

---

### Task 4: Rate-Limited Whoop API Client

**Files:**
- Create: `src/whoop_analytics/api/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write the failing test for client**

Create `tests/test_client.py`:

```python
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
    assert records[0].total_sleep_minutes == 450.0


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
    assert call_count == 2  # retried after 429
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_client.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement client.py**

Create `src/whoop_analytics/api/client.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from whoop_analytics.api.auth import TokenManager
from whoop_analytics.api.models import (
    CycleRecord,
    JournalEntry,
    RecoveryRecord,
    SleepRecord,
)

API_BASE = "https://api.prod.whoop.com/developer"
MAX_RETRIES = 3


@dataclass
class WhoopClient:
    token_manager: TokenManager
    _client: httpx.Client = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.Client(base_url=API_BASE, timeout=30.0)

    def _headers(self) -> dict[str, str]:
        token = self.token_manager.get_valid_token()
        return {"Authorization": f"Bearer {token}"}

    def _get_paginated(self, path: str, params: dict) -> list[dict]:
        all_records = []
        next_token = None

        while True:
            if next_token:
                params["nextToken"] = next_token

            response = self._request_with_retry("GET", path, params=params)
            body = response.json()

            all_records.extend(body.get("records", []))
            next_token = body.get("next_token")

            if not next_token:
                break

        return all_records

    def _request_with_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(MAX_RETRIES):
            response = self._client.request(method, path, headers=self._headers(), **kwargs)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "1"))
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response

        raise httpx.HTTPStatusError(
            "Rate limit exceeded after retries",
            request=response.request,
            response=response,
        )

    def get_sleep_records(self, start_date: str, end_date: str) -> list[SleepRecord]:
        raw = self._get_paginated("/v1/activity/sleep", {"start": start_date, "end": end_date})
        return [SleepRecord.from_api(r) for r in raw]

    def get_recovery_records(self, start_date: str, end_date: str) -> list[RecoveryRecord]:
        raw = self._get_paginated("/v1/recovery", {"start": start_date, "end": end_date})
        return [RecoveryRecord.from_api(r) for r in raw]

    def get_journal_entries(self, start_date: str, end_date: str) -> list[JournalEntry]:
        raw = self._get_paginated("/v1/journal", {"start": start_date, "end": end_date})
        return [JournalEntry.from_api(r) for r in raw]

    def get_cycles(self, start_date: str, end_date: str) -> list[CycleRecord]:
        raw = self._get_paginated("/v1/cycle", {"start": start_date, "end": end_date})
        return [CycleRecord.from_api(r) for r in raw]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_client.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/api/client.py tests/test_client.py
git commit -m "feat: rate-limited Whoop API client with pagination"
```

---

### Task 5: Data Ingestion Pipeline

**Files:**
- Create: `src/whoop_analytics/pipeline/__init__.py`
- Create: `src/whoop_analytics/pipeline/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test for ingest**

Create `tests/test_ingest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingest.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ingest.py**

Create `src/whoop_analytics/pipeline/__init__.py`:

```python
"""Data pipeline package."""
```

Create `src/whoop_analytics/pipeline/ingest.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from whoop_analytics.api.client import WhoopClient


@dataclass
class IngestPipeline:
    client: WhoopClient
    data_dir: Path

    def ingest(self, start_date: str, end_date: str) -> None:
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        self._ingest_sleep(start_date, end_date, raw_dir)
        self._ingest_recovery(start_date, end_date, raw_dir)
        self._ingest_journal(start_date, end_date, raw_dir)

    def _ingest_sleep(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        records = self.client.get_sleep_records(start_date, end_date)
        if not records:
            return
        rows = [asdict(r) for r in records]
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "sleep.parquet", index=False)

    def _ingest_recovery(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        records = self.client.get_recovery_records(start_date, end_date)
        if not records:
            return
        rows = [asdict(r) for r in records]
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "recovery.parquet", index=False)

    def _ingest_journal(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        entries = self.client.get_journal_entries(start_date, end_date)
        if not entries:
            return
        rows = []
        for entry in entries:
            row = {"id": entry.id, "created_at": entry.created_at}
            row.update(entry.answers)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "journal.parquet", index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/pipeline/ tests/test_ingest.py
git commit -m "feat: data ingestion pipeline (API → Parquet)"
```

---

### Task 6: Data Transformation & Feature Engineering

**Files:**
- Create: `src/whoop_analytics/pipeline/transform.py`
- Create: `src/whoop_analytics/pipeline/features.py`
- Create: `tests/test_transform.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write the failing test for transform**

Create `tests/test_transform.py`:

```python
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
        {"cycle_id": 100, "recovery_score": 72.0, "hrv_rmssd": 65.0, "resting_hr": 52.0,
         "spo2": 97.5, "skin_temp": 33.0},
    ])
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame([
        {"id": 500, "created_at": "2026-01-15T20:00:00", "Brain Fog": 2, "Caffeine": 1},
    ])
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    df = build_daily_dataset(tmp_path)

    assert len(df) == 1
```

- [ ] **Step 2: Write the failing test for features**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_transform.py tests/test_features.py -v
```

Expected: FAIL

- [ ] **Step 4: Implement transform.py**

Create `src/whoop_analytics/pipeline/transform.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_daily_dataset(data_dir: Path) -> pd.DataFrame:
    raw_dir = data_dir / "raw"

    sleep_df = _load_sleep(raw_dir)
    recovery_df = _load_recovery(raw_dir)
    journal_df = _load_journal(raw_dir)

    daily = sleep_df.copy()

    if recovery_df is not None and not recovery_df.empty:
        daily = daily.join(recovery_df, how="left")

    if journal_df is not None and not journal_df.empty:
        daily = daily.join(journal_df, how="left")

    return daily


def _load_sleep(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "sleep.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = df[df["nap"] == False].copy()

    df["end"] = pd.to_datetime(df["end"])
    df["date"] = df["end"].dt.normalize()
    df = df.set_index("date")
    df = df.drop(columns=["id", "start", "end", "nap"])

    return df


def _load_recovery(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "recovery.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    df = df.set_index("date")
    df = df.drop(columns=["cycle_id", "created_at", "recovery_score"], errors="ignore")
    return df


def _load_journal(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "journal.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    df = df.set_index("date")
    df = df.drop(columns=["id", "created_at"])

    col_rename = {}
    for col in df.columns:
        col_rename[col] = col.lower().replace(" ", "_")
    df = df.rename(columns=col_rename)

    return df
```

- [ ] **Step 5: Implement features.py**

Create `src/whoop_analytics/pipeline/features.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_transform.py tests/test_features.py -v
```

Expected: 6 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/whoop_analytics/pipeline/transform.py src/whoop_analytics/pipeline/features.py tests/test_transform.py tests/test_features.py
git commit -m "feat: data transformation and feature engineering"
```

---

### Task 7: Causal Discovery with Tigramite PCMCI

**Files:**
- Create: `src/whoop_analytics/analysis/__init__.py`
- Create: `src/whoop_analytics/analysis/discovery.py`
- Create: `tests/test_discovery.py`

- [ ] **Step 1: Write the failing test for discovery**

Create `tests/test_discovery.py`:

```python
import numpy as np
import pandas as pd
import pytest

from whoop_analytics.analysis.discovery import CausalDiscovery, DiscoveryResult


@pytest.fixture
def synthetic_data():
    """Create synthetic data where X causes Y with 1-day lag."""
    np.random.seed(42)
    n = 150
    x = np.random.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * x[t - 1] + 0.3 * np.random.randn()
    z = np.random.randn(n)  # noise variable, no causal role

    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"x": x, "y": y, "z": z}, index=dates)


def test_discovery_finds_causal_link(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    assert isinstance(result, DiscoveryResult)
    assert len(result.links) > 0

    x_causes = [link for link in result.links if link.source == "x"]
    assert len(x_causes) > 0
    assert x_causes[0].lag == 1


def test_discovery_ignores_noise_variable(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    z_causes = [link for link in result.links if link.source == "z"]
    assert len(z_causes) == 0


def test_discovery_result_has_graph(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    assert result.variable_names == ["x", "y", "z"]
    assert result.val_matrix is not None
    assert result.val_matrix.shape[0] == 3


def test_discovery_handles_short_data():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "a": np.random.randn(20),
        "b": np.random.randn(20),
    }, index=dates)

    discovery = CausalDiscovery(max_lag=2, significance_level=0.05)
    result = discovery.run(df, target="b")

    assert isinstance(result, DiscoveryResult)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_discovery.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement discovery.py**

Create `src/whoop_analytics/analysis/__init__.py`:

```python
"""Causal analysis package."""
```

Create `src/whoop_analytics/analysis/discovery.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


@dataclass(frozen=True)
class CausalLink:
    source: str
    target: str
    lag: int
    strength: float
    p_value: float


@dataclass(frozen=True)
class DiscoveryResult:
    links: list[CausalLink]
    variable_names: list[str]
    val_matrix: np.ndarray
    p_matrix: np.ndarray
    graph: np.ndarray


@dataclass
class CausalDiscovery:
    max_lag: int = 3
    significance_level: float = 0.05

    def run(self, df: pd.DataFrame, target: str) -> DiscoveryResult:
        variable_names = list(df.columns)
        target_idx = variable_names.index(target)

        data_array = df.values.astype(np.float64)
        dataframe = pp.DataFrame(data_array, var_names=variable_names)

        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=None)

        val_matrix = results["val_matrix"]
        p_matrix = results["p_matrix"]
        graph = pcmci.get_graph_from_pmatrix(
            p_matrix=p_matrix,
            alpha_level=self.significance_level,
            val_matrix=val_matrix,
        )

        links = self._extract_links(
            val_matrix=val_matrix,
            p_matrix=p_matrix,
            graph=graph,
            variable_names=variable_names,
            target_idx=target_idx,
        )

        return DiscoveryResult(
            links=links,
            variable_names=variable_names,
            val_matrix=val_matrix,
            p_matrix=p_matrix,
            graph=graph,
        )

    def _extract_links(
        self,
        val_matrix: np.ndarray,
        p_matrix: np.ndarray,
        graph: np.ndarray,
        variable_names: list[str],
        target_idx: int,
    ) -> list[CausalLink]:
        links = []
        n_vars = len(variable_names)

        for source_idx in range(n_vars):
            for lag in range(1, self.max_lag + 1):
                if graph[source_idx, target_idx, lag] == "-->":
                    links.append(CausalLink(
                        source=variable_names[source_idx],
                        target=variable_names[target_idx],
                        lag=lag,
                        strength=float(val_matrix[source_idx, target_idx, lag]),
                        p_value=float(p_matrix[source_idx, target_idx, lag]),
                    ))

        links.sort(key=lambda l: abs(l.strength), reverse=True)
        return links
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_discovery.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/analysis/ tests/test_discovery.py
git commit -m "feat: causal discovery with Tigramite PCMCI"
```

---

### Task 8: Causal Effect Estimation with DoWhy

**Files:**
- Create: `src/whoop_analytics/analysis/estimation.py`
- Create: `tests/test_estimation.py`

- [ ] **Step 1: Write the failing test for estimation**

Create `tests/test_estimation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from whoop_analytics.analysis.discovery import CausalLink
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult


@pytest.fixture
def causal_data():
    """X causes Y with known effect size ~0.7."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 0.7 * x + 0.3 * np.random.randn(n)
    z = np.random.randn(n)  # confounder-free noise

    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def causal_link():
    return CausalLink(source="x", target="y", lag=0, strength=0.7, p_value=0.001)


def test_estimate_effect_returns_result(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=["z"],
    )

    assert isinstance(result, EffectResult)
    assert result.ate is not None
    assert abs(result.ate - 0.7) < 0.15  # within tolerance


def test_estimate_runs_refutation(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=["z"],
    )

    assert result.refutation_passed is not None
    assert isinstance(result.refutation_p_value, float)


def test_estimate_handles_no_common_causes(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=[],
    )

    assert isinstance(result, EffectResult)
    assert result.ate is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_estimation.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement estimation.py**

Create `src/whoop_analytics/analysis/estimation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import dowhy
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_estimation.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/analysis/estimation.py tests/test_estimation.py
git commit -m "feat: causal effect estimation with DoWhy + refutation"
```

---

### Task 9: Report Generator

**Files:**
- Create: `src/whoop_analytics/report/__init__.py`
- Create: `src/whoop_analytics/report/generator.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write the failing test for report**

Create `tests/test_report.py`:

```python
from datetime import date

import pytest

from whoop_analytics.analysis.discovery import CausalLink
from whoop_analytics.analysis.estimation import EffectResult
from whoop_analytics.report.generator import ReportGenerator, AnalysisReport


@pytest.fixture
def sample_links():
    return [
        CausalLink(source="hrv_rmssd", target="brain_fog", lag=1, strength=-0.45, p_value=0.001),
        CausalLink(source="sws_minutes", target="brain_fog", lag=1, strength=-0.30, p_value=0.01),
    ]


@pytest.fixture
def sample_effects():
    return [
        EffectResult(
            source="hrv_rmssd", target="brain_fog", ate=-0.42,
            confidence_interval=(-0.55, -0.29),
            refutation_passed=True, refutation_p_value=0.82,
            method="backdoor.linear_regression",
        ),
        EffectResult(
            source="sws_minutes", target="brain_fog", ate=-0.015,
            confidence_interval=(-0.025, -0.005),
            refutation_passed=True, refutation_p_value=0.67,
            method="backdoor.linear_regression",
        ),
    ]


def test_report_generates_markdown(sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "# Causal Analysis Report" in md
    assert "hrv_rmssd" in md
    assert "brain_fog" in md
    assert "158" in md
    assert "2026-06-08" in md


def test_report_includes_effect_interpretation(sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "refutation" in md.lower() or "robust" in md.lower()


def test_report_handles_empty_links():
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=50,
        causal_links=[],
        effects=[],
    )

    generator = ReportGenerator()
    md = generator.render(report)

    assert "no significant causal" in md.lower() or "no causal links" in md.lower()


def test_report_saves_to_file(tmp_path, sample_links, sample_effects):
    report = AnalysisReport(
        generated_date=date(2026, 6, 8),
        data_start=date(2026, 1, 1),
        data_end=date(2026, 6, 7),
        n_observations=158,
        causal_links=sample_links,
        effects=sample_effects,
    )

    generator = ReportGenerator()
    output_path = tmp_path / "report.md"
    generator.save(report, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "# Causal Analysis Report" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_report.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement report generator**

Create `src/whoop_analytics/report/__init__.py`:

```python
"""Report generation package."""
```

Create `src/whoop_analytics/report/generator.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_report.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/report/ tests/test_report.py
git commit -m "feat: Markdown report generator for causal analysis"
```

---

### Task 10: CLI Entrypoint

**Files:**
- Create: `src/whoop_analytics/cli.py`
- Modify: `tests/conftest.py` (add CLI fixtures)

- [ ] **Step 1: Write the CLI module**

Create `src/whoop_analytics/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from whoop_analytics.config import Settings
from whoop_analytics.api.auth import TokenManager
from whoop_analytics.api.client import WhoopClient
from whoop_analytics.pipeline.ingest import IngestPipeline
from whoop_analytics.pipeline.transform import build_daily_dataset
from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features
from whoop_analytics.analysis.discovery import CausalDiscovery
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult
from whoop_analytics.report.generator import AnalysisReport, ReportGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whoop Causal Analytics")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Fetch data from Whoop API")
    ingest_parser.add_argument("--days", type=int, default=180, help="Days of history to fetch")

    analyze_parser = subparsers.add_parser("analyze", help="Run causal analysis")
    analyze_parser.add_argument("--target", default="brain_fog", help="Target variable")
    analyze_parser.add_argument("--max-lag", type=int, default=3, help="Maximum lag in days")
    analyze_parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")

    report_parser = subparsers.add_parser("report", help="Generate report from latest analysis")

    full_parser = subparsers.add_parser("run", help="Full pipeline: ingest → analyze → report")
    full_parser.add_argument("--days", type=int, default=180)
    full_parser.add_argument("--target", default="brain_fog")
    full_parser.add_argument("--max-lag", type=int, default=3)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    settings = Settings.from_env()

    if args.command == "ingest":
        return _cmd_ingest(settings, args.days)
    elif args.command == "analyze":
        return _cmd_analyze(settings, args.target, args.max_lag, args.alpha)
    elif args.command == "report":
        return _cmd_report(settings)
    elif args.command == "run":
        return _cmd_run(settings, args.days, args.target, args.max_lag)

    return 0


def _cmd_ingest(settings: Settings, days: int) -> int:
    token_manager = TokenManager(
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        redirect_uri=settings.redirect_uri,
        token_file=settings.data_dir / ".tokens.json",
    )
    client = WhoopClient(token_manager=token_manager)
    pipeline = IngestPipeline(client=client, data_dir=settings.data_dir)

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    print(f"Ingesting {days} days of data ({start_date} to {end_date})...")
    pipeline.ingest(start_date=start_date, end_date=end_date)
    print("Done.")
    return 0


def _cmd_analyze(settings: Settings, target: str, max_lag: int, alpha: float) -> int:
    print("Building daily dataset...")
    df = build_daily_dataset(settings.data_dir)

    if df.empty:
        print("No data found. Run 'ingest' first.", file=sys.stderr)
        return 1

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    print(f"Running causal discovery (target={target}, max_lag={max_lag})...")
    discovery = CausalDiscovery(max_lag=max_lag, significance_level=alpha)
    result = discovery.run(df, target=target)

    print(f"Found {len(result.links)} causal links.")
    for link in result.links:
        print(f"  {link.source} → {link.target} (lag={link.lag}, strength={link.strength:.3f})")

    return 0


def _cmd_report(settings: Settings) -> int:
    print("Generating report...")
    # This would load cached analysis results in a full implementation
    print("Report generation requires running 'run' command for full pipeline.")
    return 0


def _cmd_run(settings: Settings, days: int, target: str, max_lag: int) -> int:
    ret = _cmd_ingest(settings, days)
    if ret != 0:
        return ret

    print("\nBuilding daily dataset...")
    df = build_daily_dataset(settings.data_dir)

    if df.empty:
        print("No data after ingestion.", file=sys.stderr)
        return 1

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    if len(df) < 30:
        print(f"Only {len(df)} observations after feature engineering. Need at least 30.", file=sys.stderr)
        return 1

    print(f"\nRunning causal discovery (n={len(df)}, target={target}, max_lag={max_lag})...")
    discovery = CausalDiscovery(max_lag=max_lag, significance_level=0.05)
    disc_result = discovery.run(df, target=target)

    print(f"Found {len(disc_result.links)} causal links.")

    effects = []
    if disc_result.links:
        print("\nEstimating causal effects...")
        estimator = EffectEstimator()
        for link in disc_result.links:
            other_sources = [l.source for l in disc_result.links if l.source != link.source]
            effect = estimator.estimate(df=df, link=link, common_causes=other_sources)
            effects.append(effect)

    report = AnalysisReport(
        generated_date=date.today(),
        data_start=df.index[0].date(),
        data_end=df.index[-1].date(),
        n_observations=len(df),
        causal_links=disc_result.links,
        effects=effects,
    )

    generator = ReportGenerator()
    report_path = settings.data_dir / "reports" / f"report-{date.today().isoformat()}.md"
    generator.save(report, report_path)
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Test CLI argument parsing**

```bash
python -m whoop_analytics.cli --help
python -m whoop_analytics.cli ingest --help
python -m whoop_analytics.cli analyze --help
```

Expected: help text displayed without errors

- [ ] **Step 3: Commit**

```bash
git add src/whoop_analytics/cli.py
git commit -m "feat: CLI entrypoint with ingest/analyze/report/run commands"
```

---

## Self-Review Notes

**Spec coverage:** All components from BRAINSTORM.md are covered — Whoop API (OAuth + client), data pipeline (ingest + transform + features), causal discovery (Tigramite PCMCI), effect estimation (DoWhy), report generation (Markdown), and CLI orchestration.

**Type consistency:** `CausalLink` and `EffectResult` dataclass fields are consistent across discovery.py, estimation.py, and generator.py. `SleepRecord.from_api()` returns `float | None` for metrics and the transform handles this correctly.

**Known limitations:**
- DoWhy's refutation test API may differ slightly between versions — the `refutation_result` dict access in estimation.py may need adjustment based on installed version
- The recovery data alignment in transform.py uses positional index matching, which assumes 1:1 correspondence between sleep and recovery records (valid for Whoop's API which pairs them by cycle)
