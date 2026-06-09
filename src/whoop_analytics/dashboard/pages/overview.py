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
