# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Whoop API Reference

Base URL: `https://api.prod.whoop.com/developer`

### Auth

Token endpoint: `POST https://api.prod.whoop.com/oauth/oauth2/token`

Refresh:
```
grant_type=refresh_token
refresh_token=<token>
client_id=<id>
client_secret=<secret>
```

OAuth authorize: `GET https://api.prod.whoop.com/oauth/oauth2/auth`
Scopes: `read:recovery read:cycles read:sleep read:workout read:profile read:body_measurement`
Note: `read:journal` scope exists but is NOT registered in the dev portal — requesting it causes `invalid_scope` error.

### Endpoints (all v2, all working)

| Endpoint | Returns |
|----------|---------|
| `GET /v2/cycle` | strain, avg HR, max HR, kilojoule |
| `GET /v2/activity/sleep` | sleep stages (SWS, REM, light), efficiency, respiratory rate, debt |
| `GET /v2/recovery` | HRV rMSSD, recovery score, resting HR, SpO2, skin temp |
| `GET /v2/activity/workout` | workout strain, HR, kilojoule, distance |
| `GET /v2/user/profile/basic` | user profile |
| `GET /v2/user/measurement/body` | body measurements |
| `GET /v2/journal` | lifestyle factors (caffeine, alcohol, stress, etc.) — needs `read:journal` scope |

### Pagination

All collection endpoints return `{"records": [...], "next_token": "..."}`. Pass `?nextToken=<token>` for next page.

### Rate Limiting

Aggressive. Retry on HTTP 429 using `Retry-After` header. Add 0.3-0.5s delay between requests.

### v1 is dead

All `/v1/` endpoints return 404. Always use `/v2/`.
