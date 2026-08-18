# Deployment platform — run the bot anywhere, watch and steer it from your phone

The bot is now a **self-hosted platform**: a web dashboard (mobile-first, no
CDN) that shows the live loop's heartbeat, position, pending limit, equity,
recent trades and log tail, and can start / stop / restart the loop. Package it
with Docker, reach it from any device over Tailscale (or a domain + Caddy), and
it becomes your own private trading control panel.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  one container (or native process):  slytrade dashboard        │
│                                                                │
│   ┌──────────────────────────┐   ┌───────────────────────────┐ │
│   │  web dashboard :8080     │   │  supervised trading loop  │ │
│   │  HTML + JSON APIs        │◄──│  (paper or live, child)   │ │
│   │  /api/control start/stop │   │  → state/live_status.json │ │
│   └──────────────────────────┘   │  → data/live_journal      │ │
│              ▲                   │  → logs/slytrade.jsonl    │ │
└──────────────┼───────────────────┴───────────────────────────┘ │
               │ HTTPS (Tailscale / Caddy)
        ┌──────┴──────┐
        │ phone / web │
        └─────────────┘
```

* The **dashboard** reads the loop's `state/live_status.json` (rewritten every
  heartbeat, atomically), the **live journal** (`data/live_journal/trades.parquet`),
  and the **structured log** (`logs/slytrade.jsonl`).
* The **supervisor** runs the loop as a child process; `/api/control` starts,
  stops and restarts it, and its stdout is captured into the log tail.

## 1. Run it natively (on the box with the MT5 bridge)

```bash
# watch + steer, supervise the paper loop (safe default):
slytrade dashboard

# supervise the LIVE loop (real orders on the connected MT5 account):
SLYTRADE_ALLOW_LIVE=1 SLYTRADE_STAGE=demo SLYTRADE_DASHBOARD_COMMAND=live slytrade dashboard

# protect it before exposing to a network:
SLYTRADE_DASHBOARD_TOKEN=please-change-me slytrade dashboard
```

Open `http://<host>:8080`.

## 2. Run it in Docker

```bash
mkdir -p data logs state           # once (see the ownership note below)
docker compose up -d               # starts the dashboard service
# dashboard on :8080, metrics on :9108
```

Profiles (opt-in):

| Command | What it runs |
|---|---|
| `docker compose up -d` | **dashboard** (default) — platform + supervised loop |
| `docker compose --profile paper up -d` | bare paper loop, metrics only |
| `docker compose --profile research run --rm research learn --bars-file /app/data/...` | RL research (CPU torch) |
| `docker compose --profile https up -d` | Caddy HTTPS edge (needs a domain) |
| `docker compose --profile dev run --rm dev doctor` | dev/CI image |

**Ownership note:** the container runs as your host UID/GID, and Docker creates
missing bind-mount dirs as root. Run `mkdir -p data logs state` once before the
first `docker compose up`, or native runs may hit "Permission denied".

## 3. Reach it from your phone (two options)

### Option A — Tailscale (recommended, zero-config HTTPS)

1. Install Tailscale on the host (`curl -fsSL https://tailscale.com/install.sh | sh` and `tailscale up`).
2. Install the Tailscale app on your phone, log in to the same tailnet.
3. Browse to `https://<machine-name>.<tailnet>.ts.net:8080`.

Tailscale gives you an encrypted point-to-point tunnel with a stable HTTPS
name — no open ports, no domain, no Caddy. **Always set
`SLYTRADE_DASHBOARD_TOKEN`** so only you (with the token) can use the UI.

### Option B — public domain + Caddy (automatic TLS)

1. Point a domain at the host.
2. `SLYTRADE_DOMAIN=bot.example.com docker compose --profile https up -d`.
3. Browse `https://bot.example.com` (Caddy obtains + renews the certificate).

## 4. The dashboard endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | mobile dashboard (self-contained, works on slow links) |
| `GET /healthz` | liveness (container health check) |
| `GET /readyz` | readiness (a fresh loop status file exists) |
| `GET /api/status` | loop snapshot + supervisor state |
| `GET /api/trades` | recent trades from the live journal |
| `GET /api/logs?lines=100` | tail of the structured log |
| `POST /api/control` | `{"action":"start"\|"stop"\|"restart"}` |

## 5. Security checklist (before exposing anything)

- [ ] `SLYTRADE_DASHBOARD_TOKEN` set to a long random value.
- [ ] Reachable only over Tailscale or Caddy HTTPS — never port-forward the raw
      `:8080` to the public internet.
- [ ] Live orders stay disabled (`SLYTRADE_ALLOW_LIVE=0`) until the deployment
      gate is approved; the container entrypoint refuses `ALLOW_LIVE=1` without
      `STAGE=demo`.
- [ ] Data volumes (`data`, `logs`, `state`, `models`) are on the host, backed up.

## 6. Distribution ("your own platform")

* **To another machine / VPS:** `git clone` + `docker compose up -d`, or build
  the image once (`docker build -t slytrade .`) and `docker save | docker load`
  onto any host. Everything is in the image except `configs/`, `data/`, `logs/`,
  `state/` (mounted, portable).
* **Research and live on different machines:** run the `research` profile
  wherever CPU is plentiful; run the dashboard+live where the MT5 bridge lives.
  Both read/write the same directory layout, so you can ship data folders
  between them.
