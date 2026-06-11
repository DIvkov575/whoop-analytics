from __future__ import annotations

import os
import secrets
from datetime import date, timedelta
from pathlib import Path
from tempfile import mkdtemp
from dataclasses import asdict

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from whoop_analytics.api.models import SleepRecord, RecoveryRecord
from whoop_analytics.pipeline.transform import build_daily_dataset
from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features
from whoop_analytics.analysis.discovery import CausalDiscovery
from whoop_analytics.analysis.estimation import EffectEstimator

import pandas as pd

WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
WHOOP_API_BASE = "https://api.prod.whoop.com/developer"
SCOPES = "read:recovery read:cycles read:sleep read:workout read:profile read:body_measurement"

# Server-side session store (avoids cookie size limits)
_sessions: dict[str, dict] = {}

app = FastAPI(title="Whoop Causal Analytics")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "dev-secret-change-in-production"),
    max_age=86400,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _get_store(request: Request) -> dict:
    sid = request.session.get("sid")
    if not sid or sid not in _sessions:
        sid = secrets.token_urlsafe(16)
        request.session["sid"] = sid
        _sessions[sid] = {}
    return _sessions[sid]


def _client_id() -> str:
    return os.environ.get("WHOOP_CLIENT_ID", "")


def _client_secret() -> str:
    return os.environ.get("WHOOP_CLIENT_SECRET", "")


def _redirect_uri(request: Request) -> str:
    override = os.environ.get("REDIRECT_URI")
    if override:
        return override
    return str(request.url_for("oauth_callback"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    store = _get_store(request)
    if "access_token" not in store:
        return templates.TemplateResponse(request=request, name="login.html")
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "has_data": store.get("has_analysis", False),
    })


@app.get("/login")
def login(request: Request):
    store = _get_store(request)
    state = secrets.token_urlsafe(32)
    store["oauth_state"] = state
    redirect_uri = _redirect_uri(request)
    auth_url = (
        f"{WHOOP_AUTH_URL}?"
        f"client_id={_client_id()}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"state={state}&"
        f"scope={SCOPES.replace(' ', '%20')}"
    )
    return RedirectResponse(auth_url)


@app.get("/oath/callback")
def oauth_callback(request: Request):
    error = request.query_params.get("error")
    if error:
        desc = request.query_params.get("error_description", "")
        raise HTTPException(400, f"OAuth error: {error} - {desc}")

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(400, f"Missing authorization code. Params: {dict(request.query_params)}")

    redirect_uri = _redirect_uri(request)
    response = httpx.post(WHOOP_TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "redirect_uri": redirect_uri,
    })

    if response.status_code != 200:
        raise HTTPException(502, f"Token exchange failed: {response.text}")

    tokens = response.json()
    store = _get_store(request)
    store["access_token"] = tokens["access_token"]
    store["refresh_token"] = tokens["refresh_token"]
    return RedirectResponse("/")


@app.get("/logout")
def logout(request: Request):
    sid = request.session.get("sid")
    if sid and sid in _sessions:
        del _sessions[sid]
    request.session.clear()
    return RedirectResponse("/")


@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request):
    store = _get_store(request)
    token = store.get("access_token")
    if not token:
        return RedirectResponse("/")

    try:
        return _do_analyze(request, store, token)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "has_data": False, "error": f"Analysis failed: {type(e).__name__}: {e}\n\n{tb}",
        })


def _do_analyze(request: Request, store: dict, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    data_dir = Path(store.get("data_dir", mkdtemp(prefix="whoop_")))
    store["data_dir"] = str(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=90)).isoformat()
    params = {"start": start_date, "end": end_date}

    sleep_records = _fetch_paginated(f"{WHOOP_API_BASE}/v1/activity/sleep", params, headers)
    if sleep_records:
        parsed = [SleepRecord.from_api(r) for r in sleep_records]
        pd.DataFrame([asdict(r) for r in parsed]).to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_records = _fetch_paginated(f"{WHOOP_API_BASE}/v1/recovery", params, headers)
    if recovery_records:
        parsed = [RecoveryRecord.from_api(r) for r in recovery_records]
        pd.DataFrame([asdict(r) for r in parsed]).to_parquet(raw_dir / "recovery.parquet", index=False)

    journal_records = _fetch_paginated(f"{WHOOP_API_BASE}/v1/journal", params, headers)
    if journal_records:
        rows = []
        for entry in journal_records:
            row = {"id": entry["id"], "created_at": entry["created_at"]}
            for answer in entry.get("answers", []):
                row[answer["text"]] = answer["value"]
            rows.append(row)
        pd.DataFrame(rows).to_parquet(raw_dir / "journal.parquet", index=False)

    df = build_daily_dataset(data_dir)
    if df.empty:
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "has_data": False, "error": "No data returned from Whoop API.",
        })

    target = "brain_fog"
    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    links = []
    effects = []
    if len(df) >= 30 and target in df.columns:
        discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
        result = discovery.run(df, target=target)
        links = result.links

        if links:
            estimator = EffectEstimator()
            for link in links:
                if link.source == target:
                    continue
                effect = estimator.estimate(df=df, link=link, common_causes=[])
                effects.append(effect)

    store["has_analysis"] = True
    df = df.fillna(0)

    return templates.TemplateResponse(request=request, name="results.html", context={
        "df": df,
        "links": links,
        "effects": effects,
        "n_obs": len(df),
        "date_start": df.index[0].strftime("%Y-%m-%d") if not df.empty else "",
        "date_end": df.index[-1].strftime("%Y-%m-%d") if not df.empty else "",
    })


@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    store = _get_store(request)
    if not store.get("has_analysis"):
        return RedirectResponse("/")
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"has_data": True})


def _fetch_paginated(url: str, params: dict, headers: dict) -> list[dict]:
    all_records = []
    next_token = None

    while True:
        req_params = {**params}
        if next_token:
            req_params["nextToken"] = next_token

        response = httpx.get(url, params=req_params, headers=headers, timeout=30.0)
        if response.status_code == 401:
            return []
        if response.status_code != 200:
            return all_records
        response.raise_for_status()

        body = response.json()
        all_records.extend(body.get("records", []))
        next_token = body.get("next_token") or body.get("nextToken")

        if not next_token:
            break

    return all_records
