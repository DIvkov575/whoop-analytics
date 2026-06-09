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
