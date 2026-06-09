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
        import os

        data = self._load_tokens()
        refresh_token = None
        if data:
            refresh_token = data["refresh_token"]
        else:
            refresh_token = os.environ.get("WHOOP_REFRESH_TOKEN", "")

        if not refresh_token:
            raise ValueError("No tokens stored and WHOOP_REFRESH_TOKEN env var not set.")

        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
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
