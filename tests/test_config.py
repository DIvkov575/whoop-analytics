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
