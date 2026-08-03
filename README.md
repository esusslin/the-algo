# The Algo

NFL betting model + pick engine. Multi-model predictions across sides, totals, derivatives,
and player props, with market-relative edge detection and CLV tracking.

Design docs live outside this repo:
- `nfl_betting_system_architecture.md` — what and why (data sources, models, validation, betting ops)
- `nfl_implementation_architecture.md` — how (two-plane design, schema, AI agents, UI, build calendar)

---

## Two planes

| | Plane A — Research | Plane B — Serving |
|---|---|---|
| Runs | your laptop | Railway, 24/7 |
| Code | `research/` | `server.py`, `src/` |
| Data | DuckDB + parquet, 1999–present | SQLite on a volume |
| Deps | `requirements-research.txt` | `requirements.txt` |
| Output | versioned artifact bundle | picks |

Plane B never trains and never reads historical parquet. It loads an artifact bundle and
does arithmetic. `research/` must never be imported by `server.py`.

---

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in keys
python -m src.db              # run migrations
uvicorn server:app --reload
```

Open http://localhost:8000/health

For the research plane: `pip install -r requirements.txt -r requirements-research.txt`

---

## Data ingestion

```bash
# see what nflverse actually publishes right now (never hardcode asset names)
python -m src.fetchers.nflverse discover
python -m src.fetchers.nflverse discover --tag ftn_charting

# schedules + closing lines back to 1999, and the player ID crosswalk
python -m src.fetchers.nflverse games
python -m src.fetchers.nflverse players

# one-time historical pull (research plane — big, run locally)
python -m src.fetchers.nflverse backfill --start 1999

# in-season refresh (what the scheduler calls)
python -m src.fetchers.nflverse refresh

# injuries — check sources are alive before trusting output
python -m src.fetchers.injuries probe
python -m src.fetchers.injuries refresh
```

### ⚠️ The injury feed is ours to maintain

nflverse's injury source died after the 2024 season — no 2025+ data, no ETA. Availability is
one of the highest-value non-market features, so `src/fetchers/injuries.py` builds the live
feed from Sleeper (primary) and ESPN (secondary). Historical 2009–2024 nflverse injury data is
still fine for training.

Run `probe` after any dependency bump or unexplained gap. These are undocumented endpoints
and they will change without notice.

---

## Key invariants

**1. Bitemporal storage.** Every context row carries `knowledge_time` — when *we* learned it,
not when it happened. The Wednesday injury report and the Friday one are separate rows. Never
upsert an older snapshot away; a model predicting at Wednesday-time must not see Friday's data.

**2. `shared/feature_spec.py` is the single source of truth.** Both planes import it. The
artifact bundle records `spec_hash()`, and the serving loader refuses to serve on mismatch.
Training/serving skew produces confident garbage that no metric will catch.

**3. Never fuzzy-match player names silently.** Odds-feed names vs. nflverse `gsis_id` is the
#1 source of bugs in this kind of system. Unresolved names are quarantined and counted, never
guessed. See `player_aliases`.

**4. Feature flags gate capability, not code.** Everything deferred is wired and switched off:

| Flag | Default | Flip when |
|---|---|---|
| `PUBLISH_MODEL_PICKS` | `false` | models log 2–3 weeks of positive live CLV |
| `ENABLE_PROPS` | `false` | ~Week 4–6 |
| `ENABLE_SIMULATOR` | `false` | ~Week 8 |
| `ENABLE_SMS` | `false` | after a full dry run you've watched |

---

## Deployment (Railway)

1. New project → deploy from this repo
2. Add a volume mounted at `/data`
3. Set env vars from `.env.example` (`DATABASE_PATH=/data/nfl.db`, `DATA_DIR=/data`, `ENV=production`)
4. **Pin to a single replica** — SQLite on a volume corrupts with concurrent writers
5. Push to `main` → Nixpacks builds → auto-deploy

`/health` reports per-job staleness, not just HTTP 200. It returns 200 while the process is
alive (so Railway doesn't restart a container over stale data) and sets `ok: false` when a job
is overdue. Point an external uptime pinger at it and alert on `ok`, not status code.

---

## Status

**Built:** repo scaffold, config, SQLite schema + migrations, nflverse ingestion with runtime
asset discovery, injury feed, FastAPI + APScheduler service, health monitoring.

**Next (per build calendar):** Odds API client with adaptive polling + credit ledger, devig
module, cross-book shopping, market edge engine, pick tiering, mobile UI.
