from __future__ import annotations

from pathlib import Path

import streamlit as st

from whoop_analytics.config import Settings
from whoop_analytics.dashboard.state import load_daily_data, run_analysis, AnalysisState
from whoop_analytics.dashboard.pages import overview, causal, effects


def main() -> None:
    st.set_page_config(
        page_title="Whoop Causal Analytics",
        layout="wide",
    )

    st.title("Whoop Causal Analytics")

    settings = _load_settings()
    data_dir = settings.data_dir

    with st.sidebar:
        st.header("Controls")

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.subheader("Analysis Settings")
        target = st.text_input("Target variable", value="brain_fog")
        max_lag = st.slider("Max lag (days)", min_value=1, max_value=7, value=3)
        alpha = st.slider("Significance level", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

        if st.button("Run Analysis"):
            with st.spinner("Running causal analysis..."):
                state = run_analysis(data_dir, target=target, max_lag=max_lag, alpha=alpha)
                st.session_state["analysis_state"] = state
            st.success("Analysis complete!")

    daily_df = _get_daily_data(data_dir)

    analysis_state: AnalysisState | None = st.session_state.get("analysis_state")

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
