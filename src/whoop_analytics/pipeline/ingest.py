from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from whoop_analytics.api.client import WhoopClient


@dataclass
class IngestPipeline:
    client: WhoopClient
    data_dir: Path

    def ingest(self, start_date: str, end_date: str) -> None:
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        self._ingest_sleep(start_date, end_date, raw_dir)
        self._ingest_recovery(start_date, end_date, raw_dir)
        self._ingest_journal(start_date, end_date, raw_dir)

    def _ingest_sleep(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        records = self.client.get_sleep_records(start_date, end_date)
        if not records:
            return
        rows = [asdict(r) for r in records]
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "sleep.parquet", index=False)

    def _ingest_recovery(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        records = self.client.get_recovery_records(start_date, end_date)
        if not records:
            return
        rows = [asdict(r) for r in records]
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "recovery.parquet", index=False)

    def _ingest_journal(self, start_date: str, end_date: str, raw_dir: Path) -> None:
        entries = self.client.get_journal_entries(start_date, end_date)
        if not entries:
            return
        rows = []
        for entry in entries:
            row = {"id": entry.id, "created_at": entry.created_at}
            row.update(entry.answers)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "journal.parquet", index=False)
