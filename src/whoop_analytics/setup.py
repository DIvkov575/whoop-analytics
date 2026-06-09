from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
SCOPES = "read:recovery read:cycles read:sleep read:workout read:profile read:body_measurement"
REDIRECT_URI = "http://localhost:8080/callback"


def run_setup() -> int:
    print("\n=== Whoop Analytics Setup ===\n")

    client_id, client_secret = _get_credentials()
    print("\n[2/3] Opening browser for Whoop authorization...")
    tokens = _run_oauth_flow(client_id, client_secret)

    if not tokens:
        print("OAuth flow failed.", file=sys.stderr)
        return 1

    print(f"\n  Access token acquired (expires in {tokens['expires_in']}s)")

    _save_env(client_id, client_secret, tokens)
    print("\n  Saved to .env")

    _set_github_secrets(client_id, client_secret, tokens)

    print("\n=== Setup complete ===")
    print("  - Local: run 'whoop-analytics run' to test")
    print("  - Remote: daily cron will auto-generate reports")
    print(f"  - Site: https://<your-username>.github.io/whoop-analytics/")
    return 0


def _get_credentials() -> tuple[str, str]:
    print("[1/3] Whoop Developer Credentials")
    print("    If you don't have these yet, create an app at:")
    print("    https://developer.whoop.com/")
    print(f"    Set redirect URI to: {REDIRECT_URI}\n")

    env_id = os.environ.get("WHOOP_CLIENT_ID", "")
    env_secret = os.environ.get("WHOOP_CLIENT_SECRET", "")

    if env_id and env_secret:
        print(f"  Found in environment: {env_id[:8]}...")
        use = input("  Use these? [Y/n] ").strip().lower()
        if use != "n":
            return env_id, env_secret

    client_id = input("  Client ID: ").strip()
    client_secret = input("  Client Secret: ").strip()

    if not client_id or not client_secret:
        print("Both are required.", file=sys.stderr)
        sys.exit(1)

    return client_id, client_secret


def _run_oauth_flow(client_id: str, client_secret: str) -> dict | None:
    auth_code_holder: dict = {}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            if "code" in params:
                auth_code_holder["code"] = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorized!</h1><p>You can close this tab.</p></body></html>")
            else:
                error = params.get("error", ["unknown"])[0]
                auth_code_holder["error"] = error
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h1>Error: {error}</h1></body></html>".encode())

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("localhost", 8080), CallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    auth_params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
    })
    auth_url = f"{WHOOP_AUTH_URL}?{auth_params}"
    webbrowser.open(auth_url)
    print("  Waiting for authorization (browser should open)...")

    server_thread.join(timeout=120)
    server.server_close()

    if "error" in auth_code_holder:
        print(f"  Authorization error: {auth_code_holder['error']}", file=sys.stderr)
        return None

    if "code" not in auth_code_holder:
        print("  Timed out waiting for authorization.", file=sys.stderr)
        return None

    code = auth_code_holder["code"]
    print("  Authorization code received, exchanging for tokens...")

    response = httpx.post(WHOOP_TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
    })

    if response.status_code != 200:
        print(f"  Token exchange failed: {response.status_code} {response.text}", file=sys.stderr)
        return None

    return response.json()


def _save_env(client_id: str, client_secret: str, tokens: dict) -> None:
    env_path = Path(".env")
    env_path.write_text(
        f"WHOOP_CLIENT_ID={client_id}\n"
        f"WHOOP_CLIENT_SECRET={client_secret}\n"
        f"WHOOP_REDIRECT_URI={REDIRECT_URI}\n"
        f"WHOOP_ACCESS_TOKEN={tokens['access_token']}\n"
        f"WHOOP_REFRESH_TOKEN={tokens['refresh_token']}\n"
    )


def _set_github_secrets(client_id: str, client_secret: str, tokens: dict) -> None:
    print("\n[3/3] Setting GitHub repo secrets...")

    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "name"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Skipped: 'gh' CLI not available or not in a GitHub repo.")
        print("  Set these secrets manually in your repo settings:")
        print(f"    WHOOP_CLIENT_ID = {client_id}")
        print(f"    WHOOP_CLIENT_SECRET = {client_secret[:4]}...")
        print(f"    WHOOP_REFRESH_TOKEN = {tokens['refresh_token'][:8]}...")
        return

    secrets = {
        "WHOOP_CLIENT_ID": client_id,
        "WHOOP_CLIENT_SECRET": client_secret,
        "WHOOP_REFRESH_TOKEN": tokens["refresh_token"],
    }

    for name, value in secrets.items():
        proc = subprocess.run(
            ["gh", "secret", "set", name],
            input=value, capture_output=True, text=True,
        )
        if proc.returncode == 0:
            print(f"  Set {name}")
        else:
            print(f"  Failed to set {name}: {proc.stderr.strip()}")
