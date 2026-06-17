# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Install (editable, from repo root)
pip install -e ".[dev]"

# Run all tests (44 tests)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_token_cache.py::test_warm_token_cache_loads_valid -v

# Local dev server (needs WHOOP_CLIENT_ID, WHOOP_CLIENT_SECRET, REDIRECT_URI env vars)
uvicorn whoop_analytics.web.app:app --reload --port 7860

# Generate static preview of results dashboard with mock data
python preview_dashboard.py && open preview_results.html
```

The venv is at `.venv/` — use `.venv/bin/python` if not activated.

## Architecture

FastAPI web app deployed to HF Spaces (Docker SDK) that does causal inference on Whoop wearable data.

**Data flow:** OAuth login → fetch Whoop v2 API → parquet files → daily dataset → feature engineering → PCMCI causal discovery → DoWhy effect estimation → Plotly dashboard

### Key modules

- **`web/app.py`** — The main application. Handles OAuth, data fetching, analysis orchestration, token caching, and serves templates. This is ~650 lines and is the primary file you'll modify.
- **`api/client.py`** — `WhoopClient` class with paginated v2 API calls + retry logic. Used by CLI path (not the web app, which does its own fetching inline in `app.py`).
- **`api/auth.py`** — `TokenManager` for file-based token persistence and refresh. Also not used by the web app directly (web app has its own `_save_cached_tokens`/`_refresh_cached_tokens`).
- **`pipeline/transform.py`** — `build_daily_dataset()` merges sleep, recovery, and journal parquet files into a date-indexed DataFrame.
- **`pipeline/features.py`** — Adds lag and rolling window features for causal discovery.
- **`analysis/discovery.py`** — Wraps Tigramite PCMCI for causal link detection.
- **`analysis/estimation.py`** — Wraps DoWhy for average treatment effect estimation + refutation.
- **`web/templates/results.html`** — Main dashboard template (Jinja2 + Plotly.js). Tabbed chart viewer, sleep quality drivers table, causal relationship cards.

### Whoop API

All endpoints are v2 at `https://api.prod.whoop.com/developer/v2/`. The web app fetches: cycle, activity/sleep, recovery, activity/workout, user/profile/basic, user/measurement/body. Journal endpoint (`/v2/journal`) exists but requires `read:journal` scope which is NOT yet registered in the dev portal.

Rate limiting is aggressive — the app adds 0.5s delays between endpoint calls and retries on 429.

### Token caching

Tokens persist to `/data/tokens.json` on HF Spaces (survives container restarts). On startup, `_warm_token_cache()` loads or refreshes. On 401 during requests, `_try_refresh_session()` auto-refreshes. Falls back to `WHOOP_REFRESH_TOKEN` env var.

### Deployment

- **GitHub:** `origin` at `github.com/DIvkov575/whoop-analytics`
- **HF Spaces:** Auto-synced via GitHub Actions (`.github/workflows/sync-to-hf.yml`). Every push to `master` triggers `git push hf master:main`.
- **Always push to both** — the GH Action handles HF sync automatically.
- The `README.md` frontmatter (`sdk: docker`, `app_port: 7860`) is required by HF Spaces. Don't remove it.

### OAuth quirk

The callback route is `/oath/callback` (not `/oauth/callback`) — this matches what's registered in the Whoop developer portal. Don't "fix" this typo.
