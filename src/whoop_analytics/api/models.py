from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _ms_to_min(ms: int | None) -> float | None:
    if ms is None:
        return None
    return ms / 60_000


@dataclass(frozen=True)
class SleepRecord:
    id: int
    start: datetime
    end: datetime
    nap: bool
    total_sleep_minutes: float | None
    sws_minutes: float | None
    rem_minutes: float | None
    light_minutes: float | None
    disturbance_count: int | None
    respiratory_rate: float | None
    sleep_efficiency: float | None
    sleep_debt_minutes: float | None

    @classmethod
    def from_api(cls, data: dict) -> SleepRecord:
        score = data.get("score")
        if score is None:
            return cls(
                id=data["id"],
                start=_parse_dt(data["start"]),
                end=_parse_dt(data["end"]),
                nap=data.get("nap", False),
                total_sleep_minutes=None,
                sws_minutes=None,
                rem_minutes=None,
                light_minutes=None,
                disturbance_count=None,
                respiratory_rate=None,
                sleep_efficiency=None,
                sleep_debt_minutes=None,
            )

        stages = score["stage_summary"]
        total_sleep_ms = (
            stages["total_light_sleep_time_milli"]
            + stages["total_slow_wave_sleep_time_milli"]
            + stages["total_rem_sleep_time_milli"]
        )

        sleep_needed = score.get("sleep_needed", {})
        debt_ms = sleep_needed.get("need_from_sleep_debt_milli")

        return cls(
            id=data["id"],
            start=_parse_dt(data["start"]),
            end=_parse_dt(data["end"]),
            nap=data.get("nap", False),
            total_sleep_minutes=_ms_to_min(total_sleep_ms),
            sws_minutes=_ms_to_min(stages["total_slow_wave_sleep_time_milli"]),
            rem_minutes=_ms_to_min(stages["total_rem_sleep_time_milli"]),
            light_minutes=_ms_to_min(stages["total_light_sleep_time_milli"]),
            disturbance_count=stages.get("disturbance_count"),
            respiratory_rate=score.get("respiratory_rate"),
            sleep_efficiency=score.get("sleep_efficiency_percentage"),
            sleep_debt_minutes=_ms_to_min(debt_ms),
        )


@dataclass(frozen=True)
class RecoveryRecord:
    cycle_id: int
    created_at: datetime
    recovery_score: float
    hrv_rmssd: float
    resting_hr: float
    spo2: float | None
    skin_temp: float | None

    @classmethod
    def from_api(cls, data: dict) -> RecoveryRecord:
        score = data["score"]
        return cls(
            cycle_id=data["cycle_id"],
            created_at=_parse_dt(data["created_at"]),
            recovery_score=score["recovery_score"],
            hrv_rmssd=score["hrv_rmssd_milli"],
            resting_hr=score["resting_heart_rate"],
            spo2=score.get("spo2_percentage"),
            skin_temp=score.get("skin_temp_celsius"),
        )


@dataclass(frozen=True)
class JournalEntry:
    id: int
    created_at: datetime
    answers: dict[str, int]

    @classmethod
    def from_api(cls, data: dict) -> JournalEntry:
        answers = {a["text"]: a["value"] for a in data.get("answers", [])}
        return cls(
            id=data["id"],
            created_at=_parse_dt(data["created_at"]),
            answers=answers,
        )


@dataclass(frozen=True)
class CycleRecord:
    id: int
    start: datetime
    end: datetime | None
    strain: float | None

    @classmethod
    def from_api(cls, data: dict) -> CycleRecord:
        score = data.get("score")
        return cls(
            id=data["id"],
            start=_parse_dt(data["start"]),
            end=_parse_dt(data["end"]) if data.get("end") else None,
            strain=score.get("strain") if score else None,
        )
