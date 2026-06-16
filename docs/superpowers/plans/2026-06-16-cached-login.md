# Cached Login Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Whoop OAuth tokens across redeploys so the app auto-authenticates on cold start without requiring browser login.

**Architecture:** On OAuth callback, save the refresh_token to an env-backed file path (defaults to `/data/tokens.json` on HF Spaces, which persists across restarts). On app startup and on each request, if no in-memory session exists, attempt to load tokens from file and refresh if expired. The `WHOOP_REFRESH_TOKEN` env var serves as a bootstrap fallback if the file doesn't exist yet.

**Tech Stack:** Python, httpx, FastAPI, HF Spaces persistent storage (`/data/`)

---

## File Structure

| File | Role |
|------|------|
| `src/whoop_analytics/web/app.py` | Modify: add token persistence on callback, auto-restore on startup/request |
| `tests/test_token_cache.py` | Create: unit tests for cache read/write/refresh cycle |

---

### Task 1: Token File Cache — Write on OAuth Callback

**Files:**
- Modify: `src/whoop_analytics/web/app.py:156-183` (oauth_callback function)
- Test: `tests/test_token_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_cache.py
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def token_dir(tmp_path):
    return tmp_path


def test_save_tokens_writes_file(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens

    _save_cached_tokens(
        access_token="acc_123",
        refresh_token="ref_456",
        expires_in=3600,
        token_file=token_dir / "tokens.json",
    )

    data = json.loads((token_dir / "tokens.json").read_text())
    assert data["access_token"] == "acc_123"
    assert data["refresh_token"] == "ref_456"
    assert data["expires_at"] > time.time()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py::test_save_tokens_writes_file -v`
Expected: FAIL with "cannot import name '_save_cached_tokens'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/whoop_analytics/web/app.py` after the `_sessions` dict:

```python
TOKEN_CACHE_PATH = Path(os.environ.get("TOKEN_CACHE_PATH", "/data/tokens.json"))


def _save_cached_tokens(access_token: str, refresh_token: str, expires_in: int, token_file: Path = None) -> None:
    path = token_file or TOKEN_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
    }
    path.write_text(json.dumps(data))
```

Add `import json` to the imports at the top (it's already used inline for body.json — move to top-level).

Then update `oauth_callback` to call it after storing in session:

```python
    tokens = response.json()
    store = _get_store(request)
    store["access_token"] = tokens["access_token"]
    store["refresh_token"] = tokens.get("refresh_token", "")
    _save_cached_tokens(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        expires_in=tokens.get("expires_in", 3600),
    )
    return RedirectResponse("/")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py::test_save_tokens_writes_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/web/app.py tests/test_token_cache.py
git commit -m "feat: persist OAuth tokens to file on callback"
```

---

### Task 2: Token File Cache — Load and Refresh on Request

**Files:**
- Modify: `src/whoop_analytics/web/app.py`
- Test: `tests/test_token_cache.py`

- [ ] **Step 1: Write the failing test for loading valid tokens**

```python
def test_load_cached_tokens_returns_valid(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens, _load_cached_tokens

    _save_cached_tokens("acc_123", "ref_456", 3600, token_file=token_dir / "tokens.json")
    result = _load_cached_tokens(token_file=token_dir / "tokens.json")

    assert result["access_token"] == "acc_123"
    assert result["refresh_token"] == "ref_456"


def test_load_cached_tokens_returns_none_when_missing(token_dir):
    from whoop_analytics.web.app import _load_cached_tokens

    result = _load_cached_tokens(token_file=token_dir / "nonexistent.json")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py::test_load_cached_tokens_returns_valid -v`
Expected: FAIL with "cannot import name '_load_cached_tokens'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/whoop_analytics/web/app.py`:

```python
def _load_cached_tokens(token_file: Path = None) -> dict | None:
    path = token_file or TOKEN_CACHE_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py -v`
Expected: All PASS

- [ ] **Step 5: Write failing test for refresh**

```python
import respx
import httpx


@respx.mock
def test_refresh_cached_tokens_updates_file(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens, _refresh_cached_tokens

    # Save expired tokens
    token_file = token_dir / "tokens.json"
    _save_cached_tokens("old_acc", "ref_456", -100, token_file=token_file)

    # Mock the token refresh endpoint
    respx.post("https://api.prod.whoop.com/oauth/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "new_acc",
            "refresh_token": "new_ref",
            "expires_in": 3600,
        })
    )

    result = _refresh_cached_tokens(token_file=token_file)

    assert result == "new_acc"
    data = json.loads(token_file.read_text())
    assert data["access_token"] == "new_acc"
    assert data["refresh_token"] == "new_ref"
```

- [ ] **Step 6: Implement refresh function**

```python
def _refresh_cached_tokens(token_file: Path = None) -> str | None:
    path = token_file or TOKEN_CACHE_PATH
    data = _load_cached_tokens(token_file=path)
    if not data:
        return None

    refresh_token = data.get("refresh_token") or os.environ.get("WHOOP_REFRESH_TOKEN", "")
    if not refresh_token:
        return None

    try:
        response = httpx.post(WHOOP_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
        }, timeout=10)
        if response.status_code != 200:
            return None
        body = response.json()
        _save_cached_tokens(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", refresh_token),
            expires_in=body.get("expires_in", 3600),
            token_file=path,
        )
        return body["access_token"]
    except Exception:
        return None
```

- [ ] **Step 7: Run all tests**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/whoop_analytics/web/app.py tests/test_token_cache.py
git commit -m "feat: add token cache load and refresh functions"
```

---

### Task 3: Auto-Restore Session from Cache on Request

**Files:**
- Modify: `src/whoop_analytics/web/app.py:98-105` (`_get_store` function)
- Test: `tests/test_token_cache.py`

- [ ] **Step 1: Write failing test**

```python
def test_get_store_restores_from_cache(token_dir):
    from unittest.mock import MagicMock, patch
    from whoop_analytics.web.app import _get_store, _save_cached_tokens, _sessions

    # Save valid cached tokens
    token_file = token_dir / "tokens.json"
    _save_cached_tokens("cached_acc", "cached_ref", 3600, token_file=token_file)

    # Create a mock request with no existing session
    request = MagicMock()
    request.session = {}

    with patch("whoop_analytics.web.app.TOKEN_CACHE_PATH", token_file):
        store = _get_store(request)

    assert store.get("access_token") == "cached_acc"
    assert store.get("refresh_token") == "cached_ref"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py::test_get_store_restores_from_cache -v`
Expected: FAIL — `access_token` not in store

- [ ] **Step 3: Modify `_get_store` to restore from cache**

Replace the `_get_store` function:

```python
def _get_store(request: Request) -> dict:
    sid = request.session.get("sid")
    if not sid or sid not in _sessions:
        sid = secrets.token_urlsafe(16)
        request.session["sid"] = sid
        _sessions[sid] = {}
        # Try to restore from cached tokens
        cached = _load_cached_tokens()
        if cached:
            if cached.get("expires_at", 0) > time.time():
                _sessions[sid]["access_token"] = cached["access_token"]
                _sessions[sid]["refresh_token"] = cached.get("refresh_token", "")
            elif cached.get("refresh_token"):
                new_token = _refresh_cached_tokens()
                if new_token:
                    _sessions[sid]["access_token"] = new_token
                    refreshed = _load_cached_tokens()
                    _sessions[sid]["refresh_token"] = refreshed.get("refresh_token", "")
    return _sessions[sid]
```

- [ ] **Step 4: Run all tests**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py tests/test_client.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/web/app.py tests/test_token_cache.py
git commit -m "feat: auto-restore session from cached tokens on cold start"
```

---

### Task 4: Bootstrap from WHOOP_REFRESH_TOKEN env var

**Files:**
- Modify: `src/whoop_analytics/web/app.py`
- Test: `tests/test_token_cache.py`

- [ ] **Step 1: Write failing test**

```python
@respx.mock
def test_bootstrap_from_env_refresh_token(token_dir):
    from unittest.mock import MagicMock, patch
    from whoop_analytics.web.app import _get_store, _sessions

    token_file = token_dir / "tokens.json"
    # No cached file exists, but env var is set

    respx.post("https://api.prod.whoop.com/oauth/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "bootstrapped_acc",
            "refresh_token": "bootstrapped_ref",
            "expires_in": 3600,
        })
    )

    request = MagicMock()
    request.session = {}

    with patch("whoop_analytics.web.app.TOKEN_CACHE_PATH", token_file), \
         patch.dict(os.environ, {"WHOOP_REFRESH_TOKEN": "env_refresh_token"}):
        store = _get_store(request)

    assert store.get("access_token") == "bootstrapped_acc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py::test_bootstrap_from_env_refresh_token -v`
Expected: FAIL — no access_token in store (env fallback not wired)

- [ ] **Step 3: Update `_get_store` to handle env-var bootstrap**

Add to `_get_store`, after the cache restore block, before `return`:

```python
        # If still no token, try bootstrapping from env var
        if "access_token" not in _sessions[sid]:
            env_refresh = os.environ.get("WHOOP_REFRESH_TOKEN", "")
            if env_refresh:
                try:
                    response = httpx.post(WHOOP_TOKEN_URL, data={
                        "grant_type": "refresh_token",
                        "refresh_token": env_refresh,
                        "client_id": _client_id(),
                        "client_secret": _client_secret(),
                    }, timeout=10)
                    if response.status_code == 200:
                        body = response.json()
                        _sessions[sid]["access_token"] = body["access_token"]
                        _sessions[sid]["refresh_token"] = body.get("refresh_token", env_refresh)
                        _save_cached_tokens(
                            access_token=body["access_token"],
                            refresh_token=body.get("refresh_token", env_refresh),
                            expires_in=body.get("expires_in", 3600),
                        )
                except Exception:
                    pass
```

- [ ] **Step 4: Run all tests**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/test_token_cache.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/ -q`
Expected: 36+ passed

- [ ] **Step 6: Commit**

```bash
git add src/whoop_analytics/web/app.py tests/test_token_cache.py
git commit -m "feat: bootstrap auth from WHOOP_REFRESH_TOKEN env var on cold start"
```

---

### Task 5: Integration — verify full flow and push

**Files:**
- None created/modified (verification only)

- [ ] **Step 1: Run full test suite**

Run: `/Users/divkov/workplace/whoop-analytics/.venv/bin/python -m pytest tests/ -v`
Expected: All pass (36 existing + 4-5 new)

- [ ] **Step 2: Verify the token cache path is sensible for HF Spaces**

Check that `/data/` is the HF Spaces persistent volume. The `TOKEN_CACHE_PATH` env var can override for local dev (e.g., `TOKEN_CACHE_PATH=./tokens.json`).

- [ ] **Step 3: Push to master**

```bash
git push origin worktree-fix-endpoints:master
```

- [ ] **Step 4: Set WHOOP_REFRESH_TOKEN in HF Spaces secrets**

After first successful login on the deployed app, copy the refresh_token from `/debug/api` or from the token cache file, and add it as `WHOOP_REFRESH_TOKEN` in HF Spaces Settings > Secrets. This provides the bootstrap path for future redeploys.

---

## Summary of Behavior After Implementation

1. **First-ever deploy:** User logs in via browser. Tokens saved to `/data/tokens.json`.
2. **Subsequent redeploys:** App starts, in-memory sessions are empty. First request triggers `_get_store` which loads `/data/tokens.json`, refreshes if expired, and populates the session. No browser login needed.
3. **If file is lost** (e.g., HF Space rebuilt from scratch): Falls back to `WHOOP_REFRESH_TOKEN` env var to bootstrap.
4. **If both are gone:** Falls through to normal login flow (no regression).
