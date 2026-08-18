# Access & Authorization — how users get in, the whole process

The dashboard is a **self-hosted control plane**. There is no cloud login: the
only gate is a **bearer token** the operator (you) issues. This page explains
the full flow, from generating a token to revoking a user.

## The model in one line

**You are the identity provider.** You generate one secret token per user or
device, list those tokens on the server, and hand each token to its owner over
a private channel. The dashboard compares the presented token against the list;
no token → `401`. To revoke someone, remove their token from the list and
restart the dashboard.

```
   you (operator)                      server                       user's phone
   ─────────────                       ──────                       ─────────────
   slytrade gen-token   ──►  SLYTRADE_DASHBOARD_TOKEN=            opens dashboard
                             "tokA,tokB,tokC"                     enters their token
   (share tokB privately) ──────────────────────────────────────►  (stored in browser)
                                                                   every request sends
                                                                   Authorization: Bearer tokB
                                                                   server checks tokB ∈ list ✔
```

## 1. Generate a token (where auth comes from)

On the machine that runs the bot:

```bash
slytrade gen-token
# New dashboard token: 7f2c... (a 43-char URL-safe secret)
```

It is a cryptographically-random secret (`secrets.token_urlsafe(32)`). No
database, no account, no password reset — the token *is* the credential.

## 2. Register tokens on the server (who is allowed)

One token or a **comma-separated list** (one per user/device):

```bash
# one user
SLYTRADE_DASHBOARD_TOKEN=7f2c... slytrade dashboard

# several users — each with their own token so each can be revoked individually
SLYTRADE_DASHBOARD_TOKEN=aliceTok,bobTok,carolTok slytrade dashboard
```

With Docker, put it in the `.env` file next to `docker-compose.yml` (gitignored):

```
SLYTRADE_DASHBOARD_TOKEN=aliceTok,bobTok
SLYTRADE_DASHBOARD_COMMAND=live
```

Every token in the list grants the **same full access**: view the dashboard,
see trades/equity/logs, and start/stop/restart the loop. (There are no
read-only roles yet — see the caveat at the bottom.)

## 3. The user signs in (once per device)

1. Open the dashboard URL (Tailscale name or IP, see `DEPLOYMENT_PLATFORM.md`).
2. The UI shows a **token box** (it appears whenever a token is configured).
3. Paste the token they were given → the browser stores it in local storage.
4. Every request then sends `Authorization: Bearer <token>` (or `?token=`).

Wrong or missing token → `401 Unauthorized` and the login box stays.

## 4. Revoke / rotate

- **Revoke one user:** remove their token from `SLYTRADE_DASHBOARD_TOKEN`,
  restart the dashboard. Their stored token stops working immediately.
- **Rotate everything:** run `slytrade gen-token`, replace the whole list,
  restart, re-share.

## 5. Where the traffic runs (why this is safe on Tailscale)

- The dashboard runs on your machine (bind `0.0.0.0:8080`), reachable over the
  **Tailscale WireGuard tunnel** — traffic is end-to-end encrypted between
  devices and never touches the public internet.
- With `tailscale serve` you additionally get a real **HTTPS certificate**
  (`https://<machine>.<tailnet>.ts.net`), so the token is never sent over
  plaintext HTTP even inside the tunnel.
- A token is only as secret as the channel you share it through: send it in
  Signal/WhatsApp/Tailscale Send, never in a public chat or a screenshot you
  post anywhere.

## 6. Sharing the platform with someone on a *different* tailnet

Two options:

1. **Invite them into your tailnet** (recommended): Tailscale admin console →
   Users → invite, or share the machine directly (Machines → … → Share). They
   install Tailscale, join, and use `https://<machine>.ts.net:8080` + their
   token. You control both the network and the token.
2. **Expose it publicly** (NOT recommended for a trading control panel):
   `tailscale funnel` or the Caddy profile with a domain. Anyone on the
   internet can then reach the login page — you're relying entirely on the
   token, with no network-level access control. Prefer option 1.

## 7. Security checklist for distribution

- [ ] One token per person/device (auditable revocation).
- [ ] Tokens live in `.env` / env vars — **never** in git (`.env` is gitignored).
- [ ] Shared only over private channels.
- [ ] Dashboard reachable only via Tailscale (or Caddy HTTPS with a domain).
- [ ] Live trading stays `SLYTRADE_ALLOW_LIVE=0` for anyone who should only
      watch; the container entrypoint refuses `ALLOW_LIVE=1` without
      `STAGE=demo`.

## Caveat (honest)

Today every token has the **same privileges** (view + control). If you want
roles — e.g. *read-only* observers vs *operators* who can start/stop — that's a
small addition (a second token list checked on `/api/control` only). Ask and I
will add it.
