from __future__ import annotations

import json
import os
import time
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

TOKEN_CACHE_PATH = Path(os.environ.get("TOKEN_CACHE_PATH", "/data/tokens.json"))


def _save_cached_tokens(access_token: str, refresh_token: str, expires_in: int, token_file: Path = None) -> None:
    path = token_file or TOKEN_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
    }
    path.write_text(json.dumps(data))


def _load_cached_tokens(token_file: Path = None) -> dict | None:
    path = token_file or TOKEN_CACHE_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _refresh_cached_tokens(token_file: Path = None) -> str | None:
    path = token_file or TOKEN_CACHE_PATH
    data = _load_cached_tokens(token_file=path)

    refresh_token = (data or {}).get("refresh_token") or os.environ.get("WHOOP_REFRESH_TOKEN", "")
    if not refresh_token:
        return None

    try:
        response = httpx.post(WHOOP_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
        }, timeout=10)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
            response = httpx.post(WHOOP_TOKEN_URL, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
            }, timeout=10)
        if response.status_code != 200:
            return None
        body = response.json()
        _save_cached_tokens(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", refresh_token),
            expires_in=body.get("expires_in", 3600),
            token_file=path,
        )
        return body["access_token"]
    except Exception:
        return None

_cached_access_token: str | None = None

app = FastAPI(title="Whoop Causal Analytics")


@app.on_event("startup")
def _warm_token_cache():
    """Refresh tokens once at startup so _get_store never blocks on HTTP."""
    global _cached_access_token
    cached = _load_cached_tokens()
    if cached and cached.get("expires_at", 0) > time.time():
        _cached_access_token = cached.get("access_token")
        return
    token = _refresh_cached_tokens()
    if token:
        _cached_access_token = token


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

    # Pre-check: if token is stale, refresh before probing
    check = httpx.get(f"{WHOOP_API_BASE}/v2/user/profile/basic", headers=headers, timeout=10)
    if check.status_code == 401:
        if _try_refresh_session(store):
            token = store["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

    results = {}
    endpoints = [
        "/v2/cycle",
        "/v2/activity/sleep",
        "/v2/recovery",
        "/v2/activity/workout",
        "/v2/journal",
        "/v2/user/profile/basic",
        "/v2/user/measurement/body",
    ]

    for ep in endpoints:
        url = f"{WHOOP_API_BASE}{ep}"
        try:
            r = httpx.get(url, headers=headers, timeout=10)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                r = httpx.get(url, headers=headers, timeout=10)
            results[ep] = {
                "status": r.status_code,
                "body": r.text[:500],
            }
        except Exception as e:
            results[ep] = {"status": "error", "body": str(e)}
        time.sleep(0.5)

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
        # Restore from startup-warmed token or fresh cache
        cached = _load_cached_tokens()
        if cached and cached.get("access_token"):
            if cached.get("expires_at", 0) > time.time():
                _sessions[sid]["access_token"] = cached["access_token"]
                _sessions[sid]["refresh_token"] = cached.get("refresh_token", "")
            else:
                # Token expired — refresh inline (fast, single POST)
                _sessions[sid]["refresh_token"] = cached.get("refresh_token", "")
                _try_refresh_session(_sessions[sid])
        elif _cached_access_token:
            _sessions[sid]["access_token"] = _cached_access_token
            _sessions[sid]["refresh_token"] = (cached or {}).get("refresh_token", "")
    return _sessions[sid]


def _try_refresh_session(store: dict) -> bool:
    """Attempt to refresh an expired access_token using the stored refresh_token. Returns True on success."""
    global _cached_access_token
    refresh_token = store.get("refresh_token") or ""
    if not refresh_token:
        cached = _load_cached_tokens()
        refresh_token = (cached or {}).get("refresh_token", "")
    if not refresh_token:
        refresh_token = os.environ.get("WHOOP_REFRESH_TOKEN", "")
    if not refresh_token:
        return False

    try:
        response = httpx.post(WHOOP_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
        }, timeout=10)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
            response = httpx.post(WHOOP_TOKEN_URL, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
            }, timeout=10)
        if response.status_code != 200:
            return False
        body = response.json()
        store["access_token"] = body["access_token"]
        store["refresh_token"] = body.get("refresh_token", refresh_token)
        _cached_access_token = body["access_token"]
        _save_cached_tokens(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", refresh_token),
            expires_in=body.get("expires_in", 3600),
        )
        return True
    except Exception:
        return False


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
    _save_cached_tokens(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        expires_in=tokens.get("expires_in", 3600),
    )
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

    # Verify token works by hitting profile endpoint; auto-refresh if expired
    profile_resp = httpx.get(f"{WHOOP_API_BASE}/v2/user/profile/basic", headers=headers, timeout=10)
    debug_info.append(f"profile check: {profile_resp.status_code}")
    if profile_resp.status_code == 401:
        if _try_refresh_session(store):
            token = store["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            profile_resp = httpx.get(f"{WHOOP_API_BASE}/v2/user/profile/basic", headers=headers, timeout=10)
            debug_info.append(f"profile after refresh: {profile_resp.status_code}")
        if profile_resp.status_code == 401:
            return templates.TemplateResponse(request=request, name="dashboard.html", context={
                "has_data": False, "error": "Access token expired and refresh failed. Please reconnect your Whoop account.",
            })

    # Fetch cycles (delays between endpoints to avoid 429 rate limits)
    cycle_records = _try_fetch(f"{WHOOP_API_BASE}/v2/cycle", {}, headers, debug_info, "cycles")
    time.sleep(0.5)

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
    time.sleep(0.5)

    recovery_records = _try_fetch(f"{WHOOP_API_BASE}/v2/recovery", {}, headers, debug_info, "recovery")
    if recovery_records:
        parsed = [RecoveryRecord.from_api(r) for r in recovery_records]
        pd.DataFrame([asdict(r) for r in parsed]).to_parquet(raw_dir / "recovery.parquet", index=False)
    time.sleep(0.5)

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

    time.sleep(0.5)

    # Fetch journal (lifestyle factors: caffeine, alcohol, stress, etc.)
    journal_records = _try_fetch(f"{WHOOP_API_BASE}/v2/journal", {}, headers, debug_info, "journal")
    if journal_records:
        rows = []
        for j in journal_records:
            row = {"id": j["id"], "created_at": j.get("created_at", j.get("updated_at", ""))}
            for answer in j.get("answers", []):
                if "text" in answer and "value" in answer:
                    row[answer["text"]] = answer["value"]
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_parquet(raw_dir / "journal.parquet", index=False)
            debug_info.append(f"wrote {len(rows)} journal rows")

    time.sleep(0.5)

    # Fetch body measurements
    body_resp = _try_fetch_single(f"{WHOOP_API_BASE}/v2/user/measurement/body", headers, debug_info, "body")
    if body_resp:
        body_file = raw_dir / "body.json"
        body_file.write_text(json.dumps(body_resp))
        debug_info.append(f"wrote body measurements")

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

    # Run causal discovery against multiple targets for richer insights
    causal_targets = [t for t in ["hrv_rmssd", "recovery_score", "strain", "total_sleep_minutes"]
                      if t in df.columns]
    target = causal_targets[0] if causal_targets else None

    links = []
    effects = []
    seen_links = set()

    if causal_targets:
        feature_cols = [c for c in df.columns if c not in causal_targets]
        df = add_lag_features(df, columns=feature_cols + causal_targets, lags=[1, 2])
        df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
        df = df.dropna()
        debug_info.append(f"after feature eng: {df.shape}, targets={causal_targets}")

        if len(df) >= 10:
            discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
            discovery_relaxed = CausalDiscovery(max_lag=3, significance_level=0.20)
            estimator = EffectEstimator()

            for t in causal_targets:
                # Use relaxed threshold for sleep to surface more influences
                disc = discovery_relaxed if t == "total_sleep_minutes" else discovery
                result = disc.run(df, target=t)
                for link in result.links:
                    link_key = (link.source, link.target, link.lag)
                    if link_key in seen_links:
                        continue
                    seen_links.add(link_key)
                    links.append(link)
                    if link.source != t:
                        effect = estimator.estimate(df=df, link=link, common_causes=[])
                        effects.append(effect)

    store["has_analysis"] = True
    df = df.fillna(0)

    # Compute extra insights for richer dashboard
    insights = _compute_insights(df)

    return templates.TemplateResponse(request=request, name="results.html", context={
        "df": df,
        "links": links,
        "effects": effects,
        "target": target,
        "n_obs": len(df),
        "date_start": df.index[0].strftime("%Y-%m-%d") if not df.empty else "",
        "date_end": df.index[-1].strftime("%Y-%m-%d") if not df.empty else "",
        "debug_info": debug_info,
        "insights": insights,
    })


@app.get("/results", response_class=HTMLResponse)
def results(request: Request):
    store = _get_store(request)
    if not store.get("has_analysis"):
        return RedirectResponse("/")
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"has_data": True})


def _compute_insights(df: pd.DataFrame) -> dict:
    """Compute derived insights for the dashboard: correlations, day-of-week patterns, trends."""
    import numpy as np

    insights = {}

    # Correlation matrix of core metrics
    core_cols = [c for c in ["recovery_score", "strain", "hrv_rmssd", "total_sleep_minutes",
                             "resting_hr", "kilojoule", "sleep_efficiency", "caffeine",
                             "alcohol", "stress", "respiratory_rate", "spo2"] if c in df.columns]
    if len(core_cols) >= 3:
        corr = df[core_cols].corr()
        insights["corr_labels"] = [c.replace("_", " ").title() for c in core_cols]
        insights["corr_matrix"] = corr.values.tolist()

    # Day-of-week averages
    if hasattr(df.index, 'dayofweek'):
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_data = {}
        for col in ["recovery_score", "strain", "total_sleep_minutes", "hrv_rmssd"]:
            if col in df.columns:
                avgs = df.groupby(df.index.dayofweek)[col].mean()
                dow_data[col] = [float(avgs.get(i, 0)) for i in range(7)]
        if dow_data:
            insights["dow_names"] = dow_names
            insights["dow_data"] = dow_data

    # Linear trend (slope) for key metrics
    trends = {}
    x = np.arange(len(df))
    for col in ["recovery_score", "hrv_rmssd", "strain", "resting_hr"]:
        if col in df.columns and len(df) >= 7:
            y = df[col].values.astype(float)
            mask = ~np.isnan(y) & (y != 0)
            if mask.sum() >= 7:
                slope = np.polyfit(x[mask], y[mask], 1)[0]
                trends[col] = {"slope_per_week": float(slope * 7), "direction": "up" if slope > 0 else "down"}
    if trends:
        insights["trends"] = trends

    # Lagged correlation: yesterday's strain vs today's recovery
    if "strain" in df.columns and "recovery_score" in df.columns:
        prev_strain = df["strain"].shift(1)
        valid = prev_strain.notna() & df["recovery_score"].notna()
        if valid.sum() >= 10:
            insights["lag_strain_recovery"] = {
                "prev_strain": prev_strain[valid].tolist(),
                "recovery": df["recovery_score"][valid].tolist(),
            }

    # Sleep quality drivers: correlate every raw metric against sleep quality
    sleep_quality_col = None
    for candidate in ["sleep_efficiency", "sws_minutes", "total_sleep_minutes"]:
        if candidate in df.columns:
            sleep_quality_col = candidate
            break
    if sleep_quality_col:
        from scipy import stats
        drivers = []
        for col in df.columns:
            if col == sleep_quality_col:
                continue
            # Skip engineered features — only raw data from Whoop
            if any(col.endswith(s) for s in ("_lag1", "_lag2", "_roll3_mean", "_roll3_std", "_roll7_mean", "_roll7_std")):
                continue
            valid = df[[sleep_quality_col, col]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[col], valid[sleep_quality_col])
            name = col.replace("_", " ").title()
            direction = "improves" if r > 0 else "worsens"
            strength = "strongly" if abs(r) > 0.4 else "moderately" if abs(r) > 0.25 else "slightly"
            drivers.append({"name": name, "r": float(r), "p": float(p), "direction": direction, "strength": strength})
        if drivers:
            drivers.sort(key=lambda d: abs(d["r"]), reverse=True)
            insights["sleep_drivers"] = drivers
            insights["sleep_quality_metric"] = sleep_quality_col.replace("_", " ").title()

    return insights


def _try_fetch(url: str, params: dict, headers: dict, debug_info: list, label: str) -> list[dict]:
    try:
        records = _fetch_paginated(url, params, headers)
        debug_info.append(f"{label}: {len(records)} records")
        return records
    except ValueError as e:
        debug_info.append(f"{label}: ERROR - {e}")
        return []


def _try_fetch_single(url: str, headers: dict, debug_info: list, label: str) -> dict | None:
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
            response = httpx.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            debug_info.append(f"{label}: ERROR - HTTP {response.status_code}")
            return None
        debug_info.append(f"{label}: OK")
        return response.json()
    except Exception as e:
        debug_info.append(f"{label}: ERROR - {e}")
        return None


def _fetch_paginated(url: str, params: dict, headers: dict) -> list[dict]:
    all_records = []
    next_token = None

    while True:
        req_params = {**params}
        if next_token:
            req_params["nextToken"] = next_token

        response = httpx.get(url, params=req_params, headers=headers, timeout=30.0)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
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

        time.sleep(0.3)

    return all_records
