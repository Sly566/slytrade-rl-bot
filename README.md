# SlyTrade

MT5 raw data collection foundation (Layers 0+1).

Layers (built one at a time, verified before moving on):
- **Layer 0 — Foundation:** venv, deps, MT5 connector, CLI, doctor.
- **Layer 1 — Collection:** MT5 bars (M1..W1), MT5 ticks, Exness archive backfill, merged ticks.
- **Layer 2+ (planned):** per-TF processing, MTF alignment, ICT/SMC scalper, backtest, live trading.

## Quick start

```bash
# 1. One-time setup (creates .venv and installs deps)
bash scripts/setup.sh

# 2. Activate
source .venv/bin/activate

# 3. Start MT5 bridge (Wine + Xvfb, see start_mt5_bridge.sh)
./start_mt5_bridge.sh &

# 4. Sanity check
slytrade doctor
slytrade mt5-info

# 5. Collect 2y XAUUSD (bars + ticks + Exness archive backfill + merged ticks)
slytrade collect --symbol XAUUSD --lookback 2y --source hybrid
```

Data lands in `data/raw/` partitioned by year/month (and day for MT5 ticks):
```
data/raw/
  mt5_bars/symbol=XAUUSDm/timeframe=M1/year=YYYY/month=MM/part-0.parquet
  mt5_ticks/symbol=XAUUSD/year=YYYY/month=MM/day=DD.parquet
  exness_ticks/symbol=XAUUSD/year=YYYY/month=MM/part-0.parquet
  merged_ticks/symbol=XAUUSD/year=YYYY/month=MM/part-0.parquet
```

Second and subsequent runs skip existing files; `--clean` wipes and re-fetches.

## Key design choices (Layers 0+1)

- **Pinned `mt5linux==1.0.11`** — 1.1.x breaks the Wine/RPyC bridge.
- **`mt5.initialize()` before every operation** with one retry after 5 seconds.
- **Tick early-stop probing** — walks backward from today, stops after 30 consecutive empty days (MT5 demo history cliff).
- **Streaming month-by-month merge** — never loads all ticks in RAM at once.
- **All parquet, no CSVs** for processed outputs — snappy-compressed, typed, fast.
- **Pandas 3 safe** — all datetime casts use `.array.as_unit("ns")` to avoid ns/us merge errors; never `fillna(0.0)` on datetime columns.

## Important notes

- Exness archive speeds from South Africa are ~0.03–0.2 MB/s; a full 2y first-time run takes 3–4 hours. After that, `skip_existing` makes runs complete in minutes.
- MT5 demo tick history only goes back ~6–8 weeks. The Exness backfill covers everything before that.
- Data is not committed to git. Snapshots move between machines via GitHub Releases (below).

## Transferring data between machines

`data/` never enters the `main` branch. Snapshots travel as **orphan
snapshot tags**: a single throwaway commit pushed only as `refs/tags/<tag>`,
so code history stays pristine and the branch list stays `main`-only.
(Git is used instead of Release assets because the receiving environment can
only reach `github.com` — the release-asset CDN is blocked there.)

```bash
# Sender (any clone with push access):
bash scripts/upload_data.sh processed-v1 data/processed
# or several trees at once:
bash scripts/upload_data.sh data-v1 data/raw data/processed

# Receiver (any clone of this repo):
bash scripts/fetch_data.sh processed-v1          # restores under repo root
bash scripts/fetch_data.sh processed-v1 /tmp/x   # ...or into another dir
```

Details:
- Files >90MB are byte-sharded automatically (GitHub rejects single files
  >100MB) and reassembled transparently on fetch.
- Every restored file is verified against `MANIFEST.sha256` in the snapshot;
  re-running a fetch skips files that already checksum-match.
- Delete an old snapshot with: `git push origin :refs/tags/<tag>`

If you already uploaded an archive as a **Release asset** via the GitHub web
UI (e.g. a `.tar.zst` of processed data), convert it into a snapshot tag
with the `import-data-release` workflow (Actions tab → Run workflow → enter
the numeric release id + a tag name), then fetch it as above.
