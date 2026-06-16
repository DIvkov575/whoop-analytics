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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    return HTMLResponse(
        content=f"<pre>UNHANDLED ERROR:\n{type(exc).__name__}: {exc}\n\n{tb}\n\nPath: {request.url}\nMethod: {request.method}</pre>",
        status_code=500,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "client_id_set": bool(os.environ.get("WHOOP_CLIENT_ID")),
        "client_secret_set": bool(os.environ.get("WHOOP_CLIENT_SECRET")),
        "redirect_uri": os.environ.get("REDIRECT_URI", "NOT SET"),
        "session_secret_set": bool(os.environ.get("SESSION_SECRET")),
        "active_sessions": len(_sessions),
    }


@app.get("/debug/api")
def debug_api(request: Request):
    """Hit every Whoop endpoint with the user's token and return raw responses."""
    store = _get_store(request)
    token = store.get("access_token")
    if not token:
        return {"error": "not authenticated - go to / and connect first"}

    headers = {"Authorization": f"Bearer {token}"}
    results = {}

    endpoints = [
        "/v2/cycle",
        "/v2/activity/sleep",
        "/v2/recovery",
        "/v2/activity/workout",
        "/v2/user/profile/basic",
        "/v2/user/measurement/body",
    ]

    for ep in endpoints:
        url = f"{WHOOP_API_BASE}{ep}"
        try:
            r = httpx.get(url, headers=headers, timeout=10)
            results[ep] = {
                "status": r.status_code,
                "body": r.text[:500],
            }
        except Exception as e:
            results[ep] = {"status": "error", "body": str(e)}

    return results
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
        raise HTTPException(502, f"Token exchange failed (redirect_uri={redirect_uri}): {response.text}")

    tokens = response.json()
    store = _get_store(request)
    store["access_token"] = tokens["access_token"]
    store["refresh_token"] = tokens.get("refresh_token", "")
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

    end_date = date.today().isoformat() + "T23:59:59.999Z"
    start_date = (date.today() - timedelta(days=90)).isoformat() + "T00:00:00.000Z"
    params = {"start": start_date, "end": end_date}

    debug_info = []

    # Verify token works by hitting profile endpoint
    profile_resp = httpx.get(f"{WHOOP_API_BASE}/v2/user/profile/basic", headers=headers, timeout=10)
    debug_info.append(f"profile check: {profile_resp.status_code}")
    if profile_resp.status_code == 401:
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "has_data": False, "error": "Access token expired. Please reconnect your Whoop account.",
        })

    # Fetch cycles
    cycle_records = _try_fetch(f"{WHOOP_API_BASE}/v2/cycle", {}, headers, debug_info, "cycles")

    if cycle_records:
        rows = []
        for c in cycle_records:
            if not c.get("end"):
                continue
            score = c.get("score") or {}
            rows.append({
                "id": c["id"],
                "start": c["start"],
                "end": c["end"],
                "nap": False,
                "total_sleep_minutes": None,
                "sws_minutes": None,
                "rem_minutes": None,
                "light_minutes": None,
                "disturbance_count": None,
                "respiratory_rate": None,
                "sleep_efficiency": None,
                "sleep_debt_minutes": None,
                "strain": score.get("strain"),
                "average_heart_rate": score.get("average_heart_rate"),
                "max_heart_rate": score.get("max_heart_rate"),
                "kilojoule": score.get("kilojoule"),
            })
        if rows:
            pd.DataFrame(rows).to_parquet(raw_dir / "sleep.parquet", index=False)
            debug_info.append(f"wrote {len(rows)} cycle rows as sleep.parquet")

    # Fetch sleep and recovery (v2 endpoints)
    sleep_records = _try_fetch(f"{WHOOP_API_BASE}/v2/activity/sleep", {}, headers, debug_info, "sleep")
    if sleep_records:
        parsed = [SleepRecord.from_api(r) for r in sleep_records]
        pd.DataFrame([asdict(r) for r in parsed]).to_parquet(raw_dir / "sleep.parquet", index=False)

    recovery_records = _try_fetch(f"{WHOOP_API_BASE}/v2/recovery", {}, headers, debug_info, "recovery")
    if recovery_records:
        parsed = [RecoveryRecord.from_api(r) for r in recovery_records]
        pd.DataFrame([asdict(r) for r in parsed]).to_parquet(raw_dir / "recovery.parquet", index=False)

    # Fetch workouts
    workout_records = _try_fetch(f"{WHOOP_API_BASE}/v2/activity/workout", {}, headers, debug_info, "workouts")
    if workout_records:
        rows = []
        for w in workout_records:
            score = w.get("score") or {}
            rows.append({
                "id": w["id"],
                "start": w.get("start"),
                "end": w.get("end"),
                "sport_id": w.get("sport_id"),
                "strain": score.get("strain"),
                "average_heart_rate": score.get("average_heart_rate"),
                "max_heart_rate": score.get("max_heart_rate"),
                "kilojoule": score.get("kilojoule"),
                "distance_meter": score.get("distance_meter"),
            })
        if rows:
            pd.DataFrame(rows).to_parquet(raw_dir / "workouts.parquet", index=False)
            debug_info.append(f"wrote {len(rows)} workout rows")

    df = build_daily_dataset(data_dir)
    debug_info.append(f"daily_df shape: {df.shape}")
    debug_info.append(f"daily_df columns: {list(df.columns)[:10]}")

    if df.empty:
        all_404 = all("404" in d for d in debug_info if "ERROR" in d)
        if all_404:
            msg = (
                "Your Whoop account has no synced data available.\n\n"
                "This usually means your Whoop band hasn't synced recently. "
                "Charge your band, open the Whoop app to sync, then try again.\n\n"
                "Technical detail:\n" + "\n".join(debug_info)
            )
        else:
            msg = f"No usable data after processing.\n\nDebug:\n" + "\n".join(debug_info)

        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "has_data": False,
            "error": msg,
        })

    # Pick best available target for causal analysis
    target_priority = ["hrv_rmssd", "recovery_score", "strain", "average_heart_rate"]
    target = None
    for t in target_priority:
        if t in df.columns:
            target = t
            break

    links = []
    effects = []
    if target:
        feature_cols = [c for c in df.columns if c != target]
        df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
        df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
        df = df.dropna()
        debug_info.append(f"after feature eng: {df.shape}, target={target}")

        if len(df) >= 10:
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
        "target": target,
        "n_obs": len(df),
        "date_start": df.index[0].strftime("%Y-%m-%d") if not df.empty else "",
        "date_end": df.index[-1].strftime("%Y-%m-%d") if not df.empty else "",
        "debug_info": debug_info,
    })


@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    store = _get_store(request)
    if not store.get("has_analysis"):
        return RedirectResponse("/")
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"has_data": True})


def _try_fetch(url: str, params: dict, headers: dict, debug_info: list, label: str) -> list[dict]:
    try:
        records = _fetch_paginated(url, params, headers)
        debug_info.append(f"{label}: {len(records)} records")
        return records
    except ValueError as e:
        debug_info.append(f"{label}: ERROR - {e}")
        return []


def _fetch_paginated(url: str, params: dict, headers: dict) -> list[dict]:
    all_records = []
    next_token = None

    while True:
        req_params = {**params}
        if next_token:
            req_params["nextToken"] = next_token

        response = httpx.get(url, params=req_params, headers=headers, timeout=30.0)
        if response.status_code == 401:
            raise ValueError(f"Whoop API returned 401 Unauthorized for {url}. Token may be expired.")
        if response.status_code == 404:
            raise ValueError(f"Whoop API returned 404 for {url}: {response.text[:200]}")
        if response.status_code != 200:
            raise ValueError(f"Whoop API returned {response.status_code} for {url}: {response.text[:200]}")

        body = response.json()
        all_records.extend(body.get("records", []))
        next_token = body.get("next_token") or body.get("nextToken")

        if not next_token:
            break

    return all_records
