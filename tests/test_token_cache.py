import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx


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


@respx.mock
def test_refresh_cached_tokens_updates_file(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens, _refresh_cached_tokens

    token_file = token_dir / "tokens.json"
    _save_cached_tokens("old_acc", "ref_456", -100, token_file=token_file)

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


def test_get_store_restores_from_warmed_token(token_dir):
    from whoop_analytics.web.app import _get_store, _save_cached_tokens, _sessions
    import whoop_analytics.web.app as app_module

    token_file = token_dir / "tokens.json"
    _save_cached_tokens("cached_acc", "cached_ref", 3600, token_file=token_file)

    request = MagicMock()
    request.session = {}

    with patch.object(app_module, "_cached_access_token", "cached_acc"), \
         patch.object(app_module, "TOKEN_CACHE_PATH", token_file):
        store = _get_store(request)

    assert store.get("access_token") == "cached_acc"
    assert store.get("refresh_token") == "cached_ref"

    # Cleanup
    sid = request.session.get("sid")
    if sid and sid in _sessions:
        del _sessions[sid]


def test_warm_token_cache_loads_valid(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens, _warm_token_cache
    import whoop_analytics.web.app as app_module

    token_file = token_dir / "tokens.json"
    _save_cached_tokens("warm_acc", "warm_ref", 3600, token_file=token_file)
    app_module._cached_access_token = None

    with patch.object(app_module, "TOKEN_CACHE_PATH", token_file):
        _warm_token_cache()

    assert app_module._cached_access_token == "warm_acc"


@respx.mock
def test_warm_token_cache_refreshes_expired(token_dir):
    from whoop_analytics.web.app import _save_cached_tokens, _warm_token_cache
    import whoop_analytics.web.app as app_module

    token_file = token_dir / "tokens.json"
    _save_cached_tokens("old_acc", "ref_tok", -100, token_file=token_file)
    app_module._cached_access_token = None

    respx.post("https://api.prod.whoop.com/oauth/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "refreshed_acc",
            "refresh_token": "refreshed_ref",
            "expires_in": 3600,
        })
    )

    with patch.object(app_module, "TOKEN_CACHE_PATH", token_file):
        _warm_token_cache()

    assert app_module._cached_access_token == "refreshed_acc"


@respx.mock
def test_warm_token_cache_bootstraps_from_env(token_dir):
    from whoop_analytics.web.app import _warm_token_cache
    import whoop_analytics.web.app as app_module

    token_file = token_dir / "tokens.json"
    app_module._cached_access_token = None

    respx.post("https://api.prod.whoop.com/oauth/oauth2/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "env_acc",
            "refresh_token": "env_ref",
            "expires_in": 3600,
        })
    )

    with patch.object(app_module, "TOKEN_CACHE_PATH", token_file), \
         patch.dict(os.environ, {"WHOOP_REFRESH_TOKEN": "env_refresh_token"}):
        _warm_token_cache()

    assert app_module._cached_access_token == "env_acc"
