import os
import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests don't leak env vars."""
    for key in list(os.environ):
        if key.startswith("WHOOP_"):
            monkeypatch.delenv(key, raising=False)
