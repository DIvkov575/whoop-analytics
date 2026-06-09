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
