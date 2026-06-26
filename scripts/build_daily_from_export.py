import sys, json
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path("data/raw")

# ---------- Load physiological cycles (already daily, sleep+recovery+strain joined) ----------
phys = pd.read_csv(RAW/"physiological_cycles.csv")
phys["date"] = pd.to_datetime(phys["Cycle start time"], errors="coerce").dt.normalize()
phys = phys.dropna(subset=["date"])

# Map export columns -> tidy names (sleep-quality outcome + physiological context)
colmap = {
    "Asleep duration (min)": "total_sleep_minutes",
    "Deep (SWS) duration (min)": "sws_minutes",
    "REM duration (min)": "rem_minutes",
    "Light sleep duration (min)": "light_minutes",
    "Awake duration (min)": "awake_minutes",
    "Sleep efficiency %": "sleep_efficiency",
    "Sleep performance %": "sleep_performance",
    "Respiratory rate (rpm)": "respiratory_rate",
    "Sleep debt (min)": "sleep_debt_minutes",
    "Recovery score %": "recovery_score",
    "Resting heart rate (bpm)": "resting_hr",
    "Heart rate variability (ms)": "hrv_rmssd",
    "Skin temp (celsius)": "skin_temp",
    "Blood oxygen %": "spo2",
    "Day Strain": "strain",
    "Average HR (bpm)": "average_heart_rate",
    "Max HR (bpm)": "max_heart_rate",
    "Energy burned (cal)": "energy_cal",
}
keep = {k:v for k,v in colmap.items() if k in phys.columns}
daily_phys = phys.groupby("date")[list(keep)].first().rename(columns=keep)

# ---------- Load journal -> wide do_* controllable columns ----------
jour = pd.read_csv(RAW/"journal_entries.csv")
jour = jour.dropna(subset=["Cycle start time"]).copy()
jour["date"] = pd.to_datetime(jour["Cycle start time"], errors="coerce").dt.normalize()
jour = jour.dropna(subset=["date"])
def to01(v):
    s=str(v).strip().lower()
    return 1.0 if s in ("true","yes","y","1","1.0") else (0.0 if s in ("false","no","n","0","0.0") else np.nan)
jour["val"] = jour["Answered yes"].map(to01)
def slug(q): return "do_" + "".join(ch if ch.isalnum() else "_" for ch in str(q).lower()).strip("_")
jour["col"] = jour["Question text"].map(slug)
wide = jour.pivot_table(index="date", columns="col", values="val", aggfunc="max")

# Only keep behaviors logged on >=60 days with real variance (not all-yes/all-no)
good = []
for c in wide.columns:
    s = wide[c].dropna()
    if len(s) >= 60 and 0.05 <= s.mean() <= 0.95:
        good.append(c)
wide = wide[good]
CONTROLLABLE = list(wide.columns)

# ---------- Merge ----------
daily = daily_phys.join(wide, how="left")
# behavior not logged on a day = didn't do it
daily[CONTROLLABLE] = daily[CONTROLLABLE].fillna(0)
daily = daily.sort_index()
# Drop physiological rows with no sleep data
daily = daily.dropna(subset=["total_sleep_minutes"])
# Drop all-null cols, then fill residual physio NaNs with column mean
daily = daily.dropna(axis=1, how="all")
CONTROLLABLE = [c for c in CONTROLLABLE if c in daily.columns]
phys_cols = [c for c in daily.columns if c not in CONTROLLABLE]
daily[phys_cols] = daily[phys_cols].fillna(daily[phys_cols].mean())

print(f"Daily dataset: {daily.shape[0]} days, {daily.shape[1]} features")
print(f"Date range: {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"Controllable factors ({len(CONTROLLABLE)}): {CONTROLLABLE}")

daily.to_parquet("data/raw/daily_export.parquet")
print("saved daily_export.parquet")
