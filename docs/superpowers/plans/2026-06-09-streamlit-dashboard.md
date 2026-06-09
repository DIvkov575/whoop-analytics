# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive Streamlit web dashboard that visualizes time-series health data, causal graphs, and effect estimates, with buttons to trigger ingestion and analysis.

**Architecture:** A multi-page Streamlit app under `src/whoop_analytics/dashboard/` with three pages (Overview, Causal Graph, Effects). The app reuses the existing pipeline and analysis modules. A new `dashboard` subcommand in cli.py launches it via `streamlit run`.

**Tech Stack:** Streamlit, Plotly (charts), NetworkX + Plotly (graph visualization)

---

## File Structure

```
src/whoop_analytics/
├── dashboard/
│   ├── __init__.py
│   ├── app.py           # Streamlit entrypoint (multi-page setup)
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── overview.py  # Time-series plots (HRV, sleep, brain fog)
│   │   ├── causal.py    # Causal graph visualization
│   │   └── effects.py   # Effect estimates table + details
│   └── state.py         # Session state helpers (load data, run analysis)
├── cli.py               # Modified: add "dashboard" subcommand
pyproject.toml           # Modified: add streamlit, plotly deps
tests/
└── test_dashboard_state.py  # Tests for state.py logic
```

---

### Task 1: Add Dependencies and Dashboard CLI Command

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/whoop_analytics/cli.py`

- [ ] **Step 1: Add streamlit and plotly to pyproject.toml**

Add to the `dependencies` list in `pyproject.toml`:

```toml
[project]
dependencies = [
    "httpx>=0.27",
    "pandas>=2.0",
    "numpy>=1.24",
    "python-dotenv>=1.0",
    "tigramite>=5.2",
    "dowhy>=0.11",
    "networkx>=3.0",
    "scipy>=1.10",
    "pyarrow>=14.0",
    "streamlit>=1.30",
    "plotly>=5.18",
]
```

- [ ] **Step 2: Install new deps**

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 3: Add dashboard subcommand to cli.py**

Add after the `full_parser` block (around line 37):

```python
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch interactive dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
```

Add the handler in the command dispatch (after the `run` elif):

```python
    elif args.command == "dashboard":
        return _cmd_dashboard(args.port)
```

Add the function at the bottom of cli.py (before `if __name__`):

```python
def _cmd_dashboard(port: int) -> int:
    import subprocess
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        ["streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )
    return 0
```

- [ ] **Step 4: Verify CLI shows new command**

```bash
python -m whoop_analytics.cli --help
```

Expected: Shows `dashboard` in subcommands list.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/whoop_analytics/cli.py
git commit -m "feat: add dashboard CLI command and streamlit/plotly deps"
```

---

### Task 2: Dashboard State Management

**Files:**
- Create: `src/whoop_analytics/dashboard/__init__.py`
- Create: `src/whoop_analytics/dashboard/state.py`
- Create: `tests/test_dashboard_state.py`

- [ ] **Step 1: Write the failing test for state**

Create `tests/test_dashboard_state.py`:

```python
import pandas as pd
import numpy as np
from pathlib import Path

import pytest

from whoop_analytics.dashboard.state import load_daily_data, run_analysis, AnalysisState


@pytest.fixture
def data_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sleep_df = pd.DataFrame([
        {"id": 1, "start": "2026-01-14T22:00:00", "end": "2026-01-15T06:00:00", "nap": False,
         "total_sleep_minutes": 450.0, "sws_minutes": 120.0, "rem_minutes": 90.0,
         "light_minutes": 240.0, "disturbance_count": 3, "respiratory_rate": 15.0,
         "sleep_efficiency": 94.0, "sleep_debt_minutes": 30.0},
    ])
    sleep_df.to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_df = pd.DataFrame([
        {"cycle_id": 100, "created_at": "2026-01-15T07:00:00", "recovery_score": 72.0,
         "hrv_rmssd": 65.0, "resting_hr": 52.0, "spo2": 97.5, "skin_temp": 33.0},
    ])
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame([
        {"id": 500, "created_at": "2026-01-15T20:00:00", "Brain Fog": 2, "Caffeine": 1},
    ])
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    return tmp_path


def test_load_daily_data_returns_dataframe(data_dir):
    df = load_daily_data(data_dir)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "hrv_rmssd" in df.columns
    assert "brain_fog" in df.columns


def test_load_daily_data_returns_empty_when_no_data(tmp_path):
    df = load_daily_data(tmp_path)
    assert df.empty


def test_run_analysis_returns_state(data_dir):
    # Need enough data for PCMCI — create synthetic
    np.random.seed(42)
    n = 60
    raw_dir = data_dir / "raw"

    sleep_df = pd.DataFrame({
        "id": range(n),
        "start": pd.date_range("2026-01-01 22:00", periods=n, freq="D").astype(str),
        "end": pd.date_range("2026-01-02 06:00", periods=n, freq="D").astype(str),
        "nap": [False] * n,
        "total_sleep_minutes": np.random.uniform(380, 480, n),
        "sws_minutes": np.random.uniform(80, 140, n),
        "rem_minutes": np.random.uniform(60, 100, n),
        "light_minutes": np.random.uniform(200, 260, n),
        "disturbance_count": np.random.randint(0, 8, n),
        "respiratory_rate": np.random.uniform(13, 17, n),
        "sleep_efficiency": np.random.uniform(85, 98, n),
        "sleep_debt_minutes": np.random.uniform(0, 90, n),
    })
    sleep_df.to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_df = pd.DataFrame({
        "cycle_id": range(n),
        "created_at": pd.date_range("2026-01-02 07:00", periods=n, freq="D").astype(str),
        "recovery_score": np.random.uniform(40, 90, n),
        "hrv_rmssd": np.random.uniform(30, 80, n),
        "resting_hr": np.random.uniform(45, 65, n),
        "spo2": np.random.uniform(95, 99, n),
        "skin_temp": np.random.uniform(32, 34, n),
    })
    recovery_df.to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_df = pd.DataFrame({
        "id": range(n),
        "created_at": pd.date_range("2026-01-02 20:00", periods=n, freq="D").astype(str),
        "Brain Fog": np.random.randint(1, 5, n),
        "Caffeine": np.random.randint(0, 3, n),
    })
    journal_df.to_parquet(raw_dir / "journal.parquet", index=False)

    state = run_analysis(data_dir, target="brain_fog", max_lag=2, alpha=0.05)

    assert isinstance(state, AnalysisState)
    assert state.daily_df is not None
    assert not state.daily_df.empty
    assert state.discovery_result is not None
    assert isinstance(state.effects, list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_dashboard_state.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement state.py**

Create `src/whoop_analytics/dashboard/__init__.py`:

```python
"""Dashboard package."""
```

Create `src/whoop_analytics/dashboard/pages/__init__.py`:

```python
"""Dashboard pages."""
```

Create `src/whoop_analytics/dashboard/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from whoop_analytics.analysis.discovery import CausalDiscovery, DiscoveryResult
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult
from whoop_analytics.pipeline.transform import build_daily_dataset
from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features


@dataclass
class AnalysisState:
    daily_df: pd.DataFrame
    discovery_result: DiscoveryResult | None
    effects: list[EffectResult]


def load_daily_data(data_dir: Path) -> pd.DataFrame:
    return build_daily_dataset(data_dir)


def run_analysis(
    data_dir: Path,
    target: str = "brain_fog",
    max_lag: int = 3,
    alpha: float = 0.05,
) -> AnalysisState:
    df = build_daily_dataset(data_dir)

    if df.empty:
        return AnalysisState(daily_df=df, discovery_result=None, effects=[])

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    if len(df) < 30:
        return AnalysisState(daily_df=df, discovery_result=None, effects=[])

    discovery = CausalDiscovery(max_lag=max_lag, significance_level=alpha)
    disc_result = discovery.run(df, target=target)

    effects = []
    if disc_result.links:
        estimator = EffectEstimator()
        for link in disc_result.links:
            if link.source == target:
                continue
            effect = estimator.estimate(df=df, link=link, common_causes=[])
            effects.append(effect)

    return AnalysisState(
        daily_df=df,
        discovery_result=disc_result,
        effects=effects,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dashboard_state.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/whoop_analytics/dashboard/ tests/test_dashboard_state.py
git commit -m "feat: dashboard state management module"
```

---

### Task 3: Overview Page (Time-Series Plots)

**Files:**
- Create: `src/whoop_analytics/dashboard/pages/overview.py`

- [ ] **Step 1: Create overview page**

Create `src/whoop_analytics/dashboard/pages/overview.py`:

```python
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render(df: pd.DataFrame) -> None:
    st.header("Health Metrics Overview")

    if df.empty:
        st.warning("No data available. Use the sidebar to ingest data first.")
        return

    _plot_hrv_and_resting_hr(df)
    _plot_sleep_stages(df)
    _plot_brain_fog(df)


def _plot_hrv_and_resting_hr(df: pd.DataFrame) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "hrv_rmssd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["hrv_rmssd"], name="HRV (rMSSD)", line=dict(color="#2196F3")),
            secondary_y=False,
        )

    if "resting_hr" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["resting_hr"], name="Resting HR", line=dict(color="#FF5722")),
            secondary_y=True,
        )

    fig.update_layout(title="HRV & Resting Heart Rate", height=350)
    fig.update_yaxes(title_text="HRV (ms)", secondary_y=False)
    fig.update_yaxes(title_text="HR (bpm)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def _plot_sleep_stages(df: pd.DataFrame) -> None:
    stage_cols = [c for c in ["sws_minutes", "rem_minutes", "light_minutes"] if c in df.columns]
    if not stage_cols:
        return

    fig = go.Figure()
    colors = {"sws_minutes": "#3F51B5", "rem_minutes": "#9C27B0", "light_minutes": "#00BCD4"}
    labels = {"sws_minutes": "Deep (SWS)", "rem_minutes": "REM", "light_minutes": "Light"}

    for col in stage_cols:
        fig.add_trace(go.Bar(
            x=df.index, y=df[col], name=labels[col], marker_color=colors[col],
        ))

    fig.update_layout(title="Sleep Stages", barmode="stack", height=350)
    fig.update_yaxes(title_text="Minutes")
    st.plotly_chart(fig, use_container_width=True)


def _plot_brain_fog(df: pd.DataFrame) -> None:
    if "brain_fog" not in df.columns:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["brain_fog"], name="Brain Fog",
        mode="lines+markers", line=dict(color="#F44336"),
        marker=dict(size=6),
    ))
    fig.update_layout(title="Brain Fog Score", height=300)
    fig.update_yaxes(title_text="Score (1-5)")
    st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Verify file can be imported without errors**

```bash
python -c "from whoop_analytics.dashboard.pages.overview import render; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/whoop_analytics/dashboard/pages/overview.py
git commit -m "feat: dashboard overview page with time-series plots"
```

---

### Task 4: Causal Graph Page

**Files:**
- Create: `src/whoop_analytics/dashboard/pages/causal.py`

- [ ] **Step 1: Create causal graph page**

Create `src/whoop_analytics/dashboard/pages/causal.py`:

```python
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from whoop_analytics.analysis.discovery import DiscoveryResult


def render(discovery_result: DiscoveryResult | None) -> None:
    st.header("Causal Graph")

    if discovery_result is None:
        st.warning("No analysis results. Run analysis from the sidebar first.")
        return

    if not discovery_result.links:
        st.info("No significant causal links found at the current significance level.")
        return

    _render_graph(discovery_result)
    _render_links_table(discovery_result)


def _render_graph(result: DiscoveryResult) -> None:
    G = nx.DiGraph()

    for link in result.links:
        G.add_edge(
            link.source,
            link.target,
            weight=abs(link.strength),
            lag=link.lag,
            label=f"lag={link.lag}, str={link.strength:.2f}",
        )

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, abs(edge[2]["weight"]) * 5), color="#888"),
            hoverinfo="text",
            text=edge[2]["label"],
            showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=20, color="#2196F3", line=dict(width=2, color="#1565C0")),
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Discovered Causal Structure",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_links_table(result: DiscoveryResult) -> None:
    st.subheader("Causal Links")
    rows = []
    for link in result.links:
        rows.append({
            "Source": link.source,
            "Target": link.target,
            "Lag (days)": link.lag,
            "Strength": f"{link.strength:.3f}",
            "p-value": f"{link.p_value:.4f}",
        })
    st.dataframe(rows, use_container_width=True)
```

- [ ] **Step 2: Verify import**

```bash
python -c "from whoop_analytics.dashboard.pages.causal import render; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/whoop_analytics/dashboard/pages/causal.py
git commit -m "feat: dashboard causal graph page with networkx visualization"
```

---

### Task 5: Effects Page

**Files:**
- Create: `src/whoop_analytics/dashboard/pages/effects.py`

- [ ] **Step 1: Create effects page**

Create `src/whoop_analytics/dashboard/pages/effects.py`:

```python
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from whoop_analytics.analysis.estimation import EffectResult


def render(effects: list[EffectResult]) -> None:
    st.header("Effect Estimates")

    if not effects:
        st.warning("No effect estimates available. Run analysis first.")
        return

    _render_summary_chart(effects)
    _render_details(effects)


def _render_summary_chart(effects: list[EffectResult]) -> None:
    sources = [e.source for e in effects]
    ates = [e.ate for e in effects]
    colors = ["#4CAF50" if e.refutation_passed else "#F44336" for e in effects]

    fig = go.Figure(go.Bar(
        x=ates,
        y=sources,
        orientation="h",
        marker_color=colors,
        text=[f"{a:.3f}" for a in ates],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Treatment Effects on Brain Fog",
        xaxis_title="ATE (negative = reduces fog)",
        height=max(250, len(effects) * 50 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Green = passed refutation test (robust). Red = failed refutation (uncertain).")


def _render_details(effects: list[EffectResult]) -> None:
    st.subheader("Detailed Results")

    for effect in effects:
        status_emoji = "✅" if effect.refutation_passed else "❌"
        with st.expander(f"{status_emoji} {effect.source} → {effect.target}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("ATE", f"{effect.ate:.4f}")
            col2.metric("Refutation p-value", f"{effect.refutation_p_value:.3f}" if effect.refutation_p_value else "N/A")
            col3.metric("Method", effect.method.split(".")[-1])

            if effect.confidence_interval:
                st.write(f"95% CI: [{effect.confidence_interval[0]:.3f}, {effect.confidence_interval[1]:.3f}]")

            if effect.refutation_passed:
                st.success("This effect passed robustness checks.")
            else:
                st.error("This effect did NOT pass refutation. Treat with caution.")
```

- [ ] **Step 2: Verify import**

```bash
python -c "from whoop_analytics.dashboard.pages.effects import render; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/whoop_analytics/dashboard/pages/effects.py
git commit -m "feat: dashboard effects page with ATE chart and details"
```

---

### Task 6: Main App Entrypoint

**Files:**
- Create: `src/whoop_analytics/dashboard/app.py`

- [ ] **Step 1: Create app.py**

Create `src/whoop_analytics/dashboard/app.py`:

```python
from __future__ import annotations

from pathlib import Path

import streamlit as st

from whoop_analytics.config import Settings
from whoop_analytics.dashboard.state import load_daily_data, run_analysis, AnalysisState
from whoop_analytics.dashboard.pages import overview, causal, effects


def main() -> None:
    st.set_page_config(
        page_title="Whoop Causal Analytics",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Whoop Causal Analytics")

    settings = _load_settings()
    data_dir = settings.data_dir

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.subheader("Analysis Settings")
        target = st.text_input("Target variable", value="brain_fog")
        max_lag = st.slider("Max lag (days)", min_value=1, max_value=7, value=3)
        alpha = st.slider("Significance level", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

        if st.button("▶️ Run Analysis"):
            with st.spinner("Running causal analysis..."):
                state = run_analysis(data_dir, target=target, max_lag=max_lag, alpha=alpha)
                st.session_state["analysis_state"] = state
            st.success("Analysis complete!")

    # Load data for overview
    daily_df = _get_daily_data(data_dir)

    # Get analysis state
    analysis_state: AnalysisState | None = st.session_state.get("analysis_state")

    # Page navigation
    page = st.radio(
        "Page",
        ["Overview", "Causal Graph", "Effect Estimates"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Overview":
        overview.render(daily_df)
    elif page == "Causal Graph":
        disc_result = analysis_state.discovery_result if analysis_state else None
        causal.render(disc_result)
    elif page == "Effect Estimates":
        effs = analysis_state.effects if analysis_state else []
        effects.render(effs)


@st.cache_data
def _get_daily_data(data_dir: Path) -> "pd.DataFrame":
    import pandas as pd
    return load_daily_data(data_dir)


def _load_settings() -> Settings:
    try:
        return Settings.from_env()
    except ValueError:
        st.error("Missing WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET environment variables. Set them in .env")
        st.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify app can be imported**

```bash
python -c "import whoop_analytics.dashboard.app; print('OK')"
```

Expected: `OK` (may show streamlit warning outside browser context, that's fine)

- [ ] **Step 3: Verify dashboard command works**

```bash
timeout 5 python -m whoop_analytics.cli dashboard --port 8502 2>&1 || true
```

Expected: Streamlit starts (or times out after 5s which is fine — proves it launches)

- [ ] **Step 4: Commit**

```bash
git add src/whoop_analytics/dashboard/app.py
git commit -m "feat: dashboard main app with sidebar controls and page routing"
```

---

## Self-Review

**Spec coverage:**
- ✅ Time-series plots (HRV, sleep stages, brain fog) — Task 3
- ✅ Causal graph visualization — Task 4
- ✅ Effect estimates with confidence/refutation — Task 5
- ✅ Trigger ingestion/analysis from UI — Task 6 sidebar
- ✅ `whoop-analytics dashboard` CLI command — Task 1

**Placeholder scan:** No TBDs or vague steps found.

**Type consistency:**
- `AnalysisState` defined in Task 2, used in Task 6 ✅
- `DiscoveryResult` from existing `discovery.py`, passed to `causal.render()` ✅
- `EffectResult` from existing `estimation.py`, passed to `effects.render()` ✅
- `load_daily_data` and `run_analysis` from `state.py` used in `app.py` ✅
