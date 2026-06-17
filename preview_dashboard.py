"""Generate a static HTML preview of the results dashboard with mock data."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass

@dataclass
class MockLink:
    source: str
    target: str
    lag: int
    strength: float
    p_value: float

@dataclass
class MockEffect:
    source: str
    target: str
    ate: float
    refutation_passed: bool
    refutation_p_value: float
    method: str

np.random.seed(42)
dates = pd.date_range("2026-03-15", periods=60, freq="D")

df = pd.DataFrame({
    "recovery_score": np.random.uniform(25, 98, 60),
    "strain": np.random.uniform(4, 19, 60),
    "hrv_rmssd": np.random.uniform(25, 110, 60),
    "total_sleep_minutes": np.random.uniform(300, 510, 60),
    "sws_minutes": np.random.uniform(40, 130, 60),
    "rem_minutes": np.random.uniform(60, 130, 60),
    "light_minutes": np.random.uniform(140, 260, 60),
    "resting_hr": np.random.uniform(44, 62, 60),
    "average_heart_rate": np.random.uniform(55, 85, 60),
    "max_heart_rate": np.random.uniform(120, 190, 60),
    "kilojoule": np.random.uniform(6000, 14000, 60),
    "sleep_efficiency": np.random.uniform(80, 98, 60),
    "respiratory_rate": np.random.uniform(13, 17, 60),
    "spo2": np.random.uniform(94, 99, 60),
    "caffeine": np.random.randint(0, 4, 60),
    "alcohol": np.random.randint(0, 3, 60),
    "stress": np.random.randint(0, 4, 60),
}, index=dates)

links = [
    MockLink("total_sleep_minutes", "hrv_rmssd", 1, 0.42, 0.003),
    MockLink("strain", "recovery_score", 1, -0.38, 0.008),
    MockLink("caffeine", "total_sleep_minutes", 1, -0.29, 0.021),
    MockLink("alcohol", "hrv_rmssd", 1, -0.35, 0.011),
    MockLink("stress", "recovery_score", 2, -0.25, 0.034),
]

effects = [
    MockEffect("total_sleep_minutes", "hrv_rmssd", 0.042, True, 0.82, "linear_regression"),
    MockEffect("strain", "recovery_score", -1.85, True, 0.71, "linear_regression"),
    MockEffect("caffeine", "total_sleep_minutes", -18.5, False, 0.12, "linear_regression"),
    MockEffect("alcohol", "hrv_rmssd", -4.2, True, 0.65, "linear_regression"),
]

from whoop_analytics.web.app import _compute_insights

insights = _compute_insights(df)

env = Environment(loader=FileSystemLoader("src/whoop_analytics/web/templates"))
template = env.get_template("results.html")

html = template.render(
    df=df,
    links=links,
    effects=effects,
    target="hrv_rmssd",
    n_obs=len(df),
    date_start=df.index[0].strftime("%Y-%m-%d"),
    date_end=df.index[-1].strftime("%Y-%m-%d"),
    debug_info=["profile check: 200", "cycles: 60 records", "sleep: 60 records", "recovery: 60 records", "journal: 60 records"],
    insights=insights,
)

out = Path("preview_results.html")
out.write_text(html)
print(f"Preview written to {out.resolve()}")
print(f"Open: file://{out.resolve()}")
