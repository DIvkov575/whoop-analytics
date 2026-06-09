from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_daily_dataset(data_dir: Path) -> pd.DataFrame:
    raw_dir = data_dir / "raw"

    sleep_df = _load_sleep(raw_dir)
    recovery_df = _load_recovery(raw_dir)
    journal_df = _load_journal(raw_dir)

    daily = sleep_df.copy()

    if recovery_df is not None and not recovery_df.empty:
        daily = daily.join(recovery_df, how="left")

    if journal_df is not None and not journal_df.empty:
        daily = daily.join(journal_df, how="left")

    return daily


def _load_sleep(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "sleep.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = df[df["nap"] == False].copy()

    df["end"] = pd.to_datetime(df["end"])
    df["date"] = df["end"].dt.normalize()
    df = df.drop(columns=["id", "start", "end", "nap"])
    df = df.sort_values("total_sleep_minutes", ascending=False)
    df = df.set_index("date")
    df = df[~df.index.duplicated(keep="first")]

    return df


def _load_recovery(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "recovery.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    df = df.set_index("date")
    df = df.drop(columns=["cycle_id", "created_at", "recovery_score"], errors="ignore")
    return df


def _load_journal(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "journal.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    df = df.set_index("date")
    df = df.drop(columns=["id", "created_at"])

    col_rename = {}
    for col in df.columns:
        col_rename[col] = col.lower().replace(" ", "_")
    df = df.rename(columns=col_rename)

    return df
