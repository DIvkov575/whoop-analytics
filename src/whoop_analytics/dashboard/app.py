from __future__ import annotations

import os
from pathlib import Path
from datetime import date, timedelta
from tempfile import mkdtemp

import httpx
import streamlit as st
import pandas as pd

from whoop_analytics.dashboard.state import load_daily_data, run_analysis, AnalysisState
from whoop_analytics.dashboard.pages import overview, causal, effects

WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
SCOPES = "read:recovery read:cycles read:sleep read:workout read:profile read:body_measurement"


def main() -> None:
    st.set_page_config(page_title="Whoop Causal Analytics", layout="wide")
    _load_secrets()

    if not _is_authenticated():
        _render_login()
        return

    _render_app()


def _load_secrets() -> None:
    """Bridge Streamlit Cloud secrets to os.environ."""
    for key in ("WHOOP_CLIENT_ID", "WHOOP_CLIENT_SECRET", "REDIRECT_URI"):
        if key not in os.environ:
            try:
                os.environ[key] = st.secrets[key]
            except (KeyError, FileNotFoundError):
                pass


def _is_authenticated() -> bool:
    return "access_token" in st.session_state and st.session_state["access_token"]


def _get_redirect_uri() -> str:
    return os.environ.get("REDIRECT_URI", "https://whoop-analytics.streamlit.app/")


def _render_login() -> None:
    st.title("Whoop Causal Analytics")
    st.markdown("Connect your Whoop account to analyze what causes your brain fog.")

    query_params = st.query_params
    if "code" in query_params:
        _handle_oauth_callback(query_params["code"])
        return

    client_id = os.environ.get("WHOOP_CLIENT_ID", "")
    if not client_id:
        st.error("App not configured: missing WHOOP_CLIENT_ID in Streamlit secrets.")
        return

    redirect_uri = _get_redirect_uri()
    auth_url = (
        f"{WHOOP_AUTH_URL}?"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope={SCOPES.replace(' ', '%20')}"
    )

    st.markdown(f"""
    <a href="{auth_url}" style="
        display: inline-block;
        padding: 12px 24px;
        background: #2196F3;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 600;
    ">Connect Whoop Account</a>
    """, unsafe_allow_html=True)

    st.caption("This will redirect you to Whoop to authorize access to your data.")


def _handle_oauth_callback(code: str) -> None:
    client_id = os.environ.get("WHOOP_CLIENT_ID", "")
    client_secret = os.environ.get("WHOOP_CLIENT_SECRET", "")
    redirect_uri = _get_redirect_uri()

    with st.spinner("Connecting to Whoop..."):
        response = httpx.post(WHOOP_TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        })

    if response.status_code != 200:
        st.error(f"Authorization failed: {response.text}")
        return

    tokens = response.json()
    st.session_state["access_token"] = tokens["access_token"]
    st.session_state["refresh_token"] = tokens["refresh_token"]
    st.query_params.clear()
    st.rerun()


def _render_app() -> None:
    st.title("Whoop Causal Analytics")

    with st.sidebar:
        st.header("Controls")

        days = st.slider("Days of history", min_value=30, max_value=365, value=180)

        if st.button("Fetch & Analyze", type="primary"):
            _run_full_pipeline(days)

        st.divider()
        st.subheader("Analysis Settings")
        target = st.text_input("Target variable", value="brain_fog")
        max_lag = st.slider("Max lag (days)", min_value=1, max_value=7, value=3)
        alpha = st.slider("Significance level", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

        if st.button("Re-run Analysis"):
            if "data_dir" in st.session_state:
                with st.spinner("Running causal analysis..."):
                    state = run_analysis(
                        Path(st.session_state["data_dir"]),
                        target=target, max_lag=max_lag, alpha=alpha,
                    )
                    st.session_state["analysis_state"] = state
                st.rerun()

        st.divider()
        if st.button("Disconnect"):
            st.session_state.clear()
            st.rerun()

    analysis_state: AnalysisState | None = st.session_state.get("analysis_state")

    if analysis_state is None:
        st.info("Click **Fetch & Analyze** in the sidebar to load your Whoop data.")
        return

    page = st.radio(
        "Page", ["Overview", "Causal Graph", "Effect Estimates"],
        horizontal=True, label_visibility="collapsed",
    )

    if page == "Overview":
        overview.render(analysis_state.daily_df)
    elif page == "Causal Graph":
        disc_result = analysis_state.discovery_result
        causal.render(disc_result)
    elif page == "Effect Estimates":
        effects.render(analysis_state.effects)


def _run_full_pipeline(days: int) -> None:
    data_dir = Path(st.session_state.get("data_dir", mkdtemp(prefix="whoop_")))
    st.session_state["data_dir"] = str(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    access_token = st.session_state["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    with st.spinner("Fetching sleep data..."):
        sleep_records = _fetch_paginated(
            "https://api.prod.whoop.com/developer/v1/activity/sleep",
            {"start": start_date, "end": end_date}, headers,
        )
        if sleep_records:
            _save_sleep_parquet(sleep_records, raw_dir)

    with st.spinner("Fetching recovery data..."):
        recovery_records = _fetch_paginated(
            "https://api.prod.whoop.com/developer/v1/recovery",
            {"start": start_date, "end": end_date}, headers,
        )
        if recovery_records:
            _save_recovery_parquet(recovery_records, raw_dir)

    with st.spinner("Fetching journal entries..."):
        journal_records = _fetch_paginated(
            "https://api.prod.whoop.com/developer/v1/journal",
            {"start": start_date, "end": end_date}, headers,
        )
        if journal_records:
            _save_journal_parquet(journal_records, raw_dir)

    with st.spinner("Running causal analysis..."):
        state = run_analysis(data_dir, target="brain_fog", max_lag=3, alpha=0.05)
        st.session_state["analysis_state"] = state

    st.rerun()


def _fetch_paginated(url: str, params: dict, headers: dict) -> list[dict]:
    all_records = []
    next_token = None

    while True:
        if next_token:
            params["nextToken"] = next_token

        response = httpx.get(url, params=params, headers=headers, timeout=30.0)
        if response.status_code == 401:
            st.session_state.clear()
            st.error("Session expired. Please reconnect.")
            st.stop()
        response.raise_for_status()

        body = response.json()
        all_records.extend(body.get("records", []))
        next_token = body.get("next_token") or body.get("nextToken")

        if not next_token:
            break

    return all_records


def _save_sleep_parquet(records: list[dict], raw_dir: Path) -> None:
    from whoop_analytics.api.models import SleepRecord
    from dataclasses import asdict

    parsed = [SleepRecord.from_api(r) for r in records]
    df = pd.DataFrame([asdict(r) for r in parsed])
    df.to_parquet(raw_dir / "sleep.parquet", index=False)


def _save_recovery_parquet(records: list[dict], raw_dir: Path) -> None:
    from whoop_analytics.api.models import RecoveryRecord
    from dataclasses import asdict

    parsed = [RecoveryRecord.from_api(r) for r in records]
    df = pd.DataFrame([asdict(r) for r in parsed])
    df.to_parquet(raw_dir / "recovery.parquet", index=False)


def _save_journal_parquet(records: list[dict], raw_dir: Path) -> None:
    rows = []
    for entry in records:
        row = {"id": entry["id"], "created_at": entry["created_at"]}
        for answer in entry.get("answers", []):
            row[answer["text"]] = answer["value"]
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(raw_dir / "journal.parquet", index=False)


if __name__ == "__main__":
    main()
