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
