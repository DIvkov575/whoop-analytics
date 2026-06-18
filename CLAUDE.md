# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
python -m pytest tests/test_client.py::test_client_fetches_sleep_records -v
```

## Usage

```bash
# Set credentials in .env (WHOOP_CLIENT_ID, WHOOP_CLIENT_SECRET, WHOOP_REFRESH_TOKEN)
# Then:
whoop-analytics ingest --days 90
whoop-analytics analyze --target hrv_rmssd
whoop-analytics run --days 90  # ingest + analyze in one shot
```

## Architecture

CLI tool that fetches Whoop wearable data and runs causal inference.

**Pipeline:** OAuth token refresh → paginated v2 API fetch → parquet files → daily dataset → feature engineering (lags, rolling windows) → Tigramite PCMCI causal discovery → DoWhy effect estimation

### Key modules

- **`api/client.py`** — `WhoopClient` with paginated v2 API calls + 429 retry
- **`api/auth.py`** — `TokenManager` for file-based token persistence and refresh
- **`api/models.py`** — Dataclasses for Sleep, Recovery, Journal, Cycle records
- **`pipeline/ingest.py`** — `IngestPipeline` orchestrates fetching all endpoints
- **`pipeline/transform.py`** — `build_daily_dataset()` merges parquet files into date-indexed DataFrame
- **`pipeline/features.py`** — Lag and rolling window feature engineering
- **`analysis/discovery.py`** — Wraps Tigramite PCMCI for causal link detection
- **`analysis/estimation.py`** — Wraps DoWhy for ATE estimation + refutation

### Whoop API

All endpoints are v2 at `https://api.prod.whoop.com/developer/v2/`:
- `/v2/cycle` — strain, HR, kilojoules (WORKS)
- `/v2/activity/sleep` — sleep stages, efficiency (WORKS)
- `/v2/recovery` — HRV, recovery score (WORKS)
- `/v2/activity/workout` — workout strain, distance (WORKS)
- `/v2/user/profile/basic` — user info (WORKS)
- `/v2/user/measurement/body` — body measurements (WORKS)
- `/v2/journal` — lifestyle factors (REQUIRES `read:journal` scope, NOT registered in dev portal yet)

Rate limiting is aggressive — 0.3-0.5s delays between calls, retry on 429 with Retry-After header.

### OAuth

Token refresh uses `https://api.prod.whoop.com/oauth/oauth2/token` with `grant_type=refresh_token`. The `WHOOP_REFRESH_TOKEN` env var bootstraps auth without browser flow.
