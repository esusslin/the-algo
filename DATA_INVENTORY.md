# Data inventory and system structure

What exists, where it lives, how much of it there is, and what's missing.
Generated 17 Aug 2026 from the actual stores, not from memory.

Companion docs: `ALGORITHM.md` (how the pipeline works), `README.md` (setup),
`PROP_ENGINE_DESIGN.md` (where this is going).

---

## 1. Two planes

```
  RESEARCH PLANE (local only)              SERVING PLANE (Railway)
  ───────────────────────────              ────────────────────────
  data/raw/nflverse/**  546 MB parquet     data/nfl.db  7.8 MB SQLite
            │                                        │
            ▼                                        ▼
  data/warehouse.duckdb  35 MB              FastAPI + APScheduler
  (views over parquet + 4 derived tables)   Alpine.js single page
            │                                        ▲
            ▼                                        │
  training / validation ──── artifact bundle ────────┘
                             (hash-validated)
```

Nothing from the research plane runs on Railway. The only thing that crosses is
an artifact bundle, and `shared/feature_spec.py` is imported by both sides so
training and serving can't drift apart silently.

---

## 2. Raw data — 546 MB, 13 nflverse sources

| Source | Size | Files | Seasons | What it's for |
|---|---|---|---|---|
| `pbp` | 466 MB | 27 | 1999–2025 | Play-by-play, 372 columns. The foundation. |
| `stats_player` | 36 MB | 108 | 1999–2025 | Weekly player box + advanced |
| `depth_charts` | 15 MB | 26 | 2001–2025 | Role and starter inference |
| `stats_team` | 8.0 MB | 108 | 1999–2025 | Weekly team aggregates |
| `contracts` | 6.5 MB | 1 | historical | Unused so far |
| `players` | 3.3 MB | 1 | current | ID crosswalk, 25k rows |
| `snap_counts` | 2.9 MB | 14 | 2013–2025 | **Layer 1 of the prop model** |
| `pfr_advstats` | 2.8 MB | 36 | 2018–2025 | Yards before/after contact, broken tackles |
| `nextgen_stats` | 2.2 MB | 3 | 2016–2025 | Cushion, separation, YAC over expected, box counts |
| `ftn_charting` | 2.1 MB | 4 | 2022–2025 | Blitzers, play-action, screens, RPO, drops |
| `injuries` | 2.0 MB | 17 | 2009–2025 | **Layer 0 of the prop model** |
| `draft_picks` | 684 KB | 1 | historical | Unused so far |
| `combine` | 368 KB | 1 | historical | Unused so far |

---

## 3. Warehouse — DuckDB views and derived tables

Views read parquet directly; there is no import step and no second copy.

| Object | Rows | Cols | Seasons | Notes |
|---|---|---|---|---|
| `pbp` (view) | 1,279,628 | 372 | 1999–2025 | Raw play-by-play |
| `plays` (view) | 948,121 | 24 | 1999–2025 | Cleaned, typed subset — EPA, success, depth bucket, player IDs |
| `stats_player` (view) | 476,156 | 145 | 1999–2025 | |
| `snaps` (view) | 324,611 | 16 | 2013–2025 | `offense_pct` is the key field |
| `ftn` (view) | 185,215 | 29 | 2022–2025 | Only 4 seasons — small but rich |
| `player_weeks` (table) | 135,065 | 13 | 1999–2025 | targets, carries, air yards, EPA per target/carry, shares |
| `injuries_hist` (view) | 90,752 | 17 | 2009–2025 | report + practice status, `date_modified` for point-in-time |
| `unit_ratings` (table) | 52,140 | 9 | 1999–2025 | Ridge-decomposed off/def by play class, per week, with `confident` flag |
| `defense_depth_weeks` (table) | 42,888 | 8 | 2006–2025 | EPA allowed by target depth bucket |
| `ngs` (view) | 26,723 | 51 | 2016–2025 | Cushion, separation, CPOE, box count |
| `team_weeks` (table) | 14,546 | 18 | 1999–2025 | Pace, EPA splits, pass rate, explosive rate, red zone |

**The four derived tables are the ones that took work.** `unit_ratings` in
particular is computed point-in-time — each week's rating uses only earlier
weeks, which is what makes it usable as a training feature without leakage.

---

## 4. Serving DB — SQLite, 7.8 MB

| Table | Rows | Status |
|---|---|---|
| `games` | 7,548 | 1999–2026. 7,276 with scores, 7,327 with a closing spread, **272 for 2026** |
| `players` | 25,048 | ID crosswalk |
| `odds_current` | 5,678 | 23 books × 3 markets |
| `odds_changes` | 5,678 | ⚠️ see below |
| `fair_prices` | 2,532 | Devigged consensus |
| `teams` | 32 | With stadium lat/lon/elevation |
| `source_freshness` | 16 | All 16 sources reporting healthy as of 4 Aug |
| `odds_credit_ledger` | 6 | 19,988 of 20,000 credits remaining |
| `picks` | 3 | Demo remnants — purge before launch |
| `schema_version` | 6 | 6 migrations applied |

**Empty and expected to be:** `users`, `invites`, `user_bets`, `password_resets`
(this is the local dev copy; Railway has the real ones), `ai_calls`, `job_runs`,
`live_state`, `live_player_stats`, `projections`, `artifact_registry`.

**Empty and worth noting:** `injuries` and `weather_snapshots` are both at zero.
The historical injury data is in parquet, but the live serving table hasn't been
populated — the Wed/Thu/Fri cron hasn't had a real week to run against yet.

---

## 5. The gap that matters most

**`odds_changes` contains a single timestamp: 2026-08-03T14:52:21Z.**

All 5,678 rows share it. That's one poll, written as "changed" because it was the
first. **There is no line-movement history at all.**

Consequences, in order of how much they hurt:

1. The three market-microstructure features in `shared/feature_spec.py` —
   `line_move_spread`, `book_dispersion`, `sharp_soft_delta` — **cannot be
   backfilled.** I said last week that they could be. They can't; there's nothing
   to backfill from. They can only be filled forward from the first September
   polls.
2. The closing-line model in `PROP_ENGINE_DESIGN.md` §5 has zero training data
   today. It becomes buildable around Week 4–6, once several weeks of movement
   have accumulated.
3. Book latency profiling — which book moves first, which follows — likewise
   starts from zero in September.

This makes the September collection weeks more valuable than they looked. The
data being gathered isn't just CLV measurement; it's the *only* training set the
market-microstructure work will ever have.

Also worth flagging: `odds_current` covers `h2h`, `spreads`, `totals` only. No
period markets, no team totals, no props. First-half markets are central to
`PROP_ENGINE_DESIGN.md` §4 and we are not currently polling them — that needs
turning on before there's any 1H data to reason about.

---

## 6. Code structure

```
server.py                 FastAPI app, routes, scheduler wiring
railway.toml              deploy config

shared/feature_spec.py    single source of truth for feature names + order
                          imported by BOTH planes — prevents training/serving skew

src/
  config.py               settings
  db.py                   SQLite layer, WAL, 6 migrations, insert_row/upsert_row
  teams.py                explicit 32-team crosswalk + stadium geo
  markets.py              market taxonomy — describe_market() parses any key
  auth.py                 JWT + bcrypt

  fetchers/
    odds_api.py           adaptive polling tiers + credit ledger + throttle
    nflverse.py           parquet backfill
    injuries.py           Wed/Thu/Fri 17:00 ET
    weather.py            forecast by stadium

  market/
    devig.py              4 methods: multiplicative, additive, power, Shin
    consensus.py          per-book devig, weighted median, sharp anchoring
    shop.py               main_lines(), find_opportunities()

  picks/
    generator.py          tiering, blended_probability() — the single blend point
    kelly.py              fractional Kelly + correlation haircuts
    grading.py            settlement + capture_closing_lines()
    history.py            performance with confidence intervals
    live.py               in-game state

  models/artifacts.py     bundle load + hash validation
  ai/redteam.py           downgrade-only veto agent
  notify/sms.py           Twilio

research/                 local only — never installed on Railway
  warehouse.py            DuckDB views over parquet
  ratings.py              ridge decomposition, point-in-time
  features.py             training matrix + directional sign checks
  train.py                walk-forward vs market baseline
  props.py                distributional projections, hurdle model, PIT
  validate.py             leakage suite, 55% accuracy tripwire
  export.py               artifact bundle builder

tests/                    94 tests — test_grading, test_market, test_safety
templates/app.html        Alpine.js single page
```

---

## 7. What the models have actually seen

Relevant because it's narrower than previously described.

The residual model's feature vector contained: market spread and total, rest and
rest differential, divisional flag, dome flag, kickoff slot, week, opponent-
adjusted off/def ratings (overall, pass, rush), and unit-vs-unit matchup edges.

Declared in `feature_spec.py` but **hardcoded constants** — never seen by any
model:

| Feature | Value it was pinned to |
|---|---|
| `wind_kph`, `wind_gust_kph`, `high_wind` | 0.0, 0.0, False |
| `temp_c`, `precip_mm` | 15.0, 0.0 |
| `market_home_prob` | 0.5 |
| `line_move_spread` | 0.0 |
| `book_dispersion` | 0.0 |
| `sharp_soft_delta` | 0.0 |

Not present in the feature spec at all: injuries, travel distance, timezone
shift, snap counts, any player-level input.

So the 50.0% null result is a real result about *team strength, rest and venue*
against the closing spread. It is not evidence about weather, injuries, or market
microstructure, because none of those were in the model.

---

## 8. Freshness

All 16 sources reported success on 4 Aug. Upstream stamps show most nflverse
files were last published in Feb 2026 (end of the 2025 season) — expected, since
they only update during the season. `depth_charts` and `contracts` refreshed
4 Aug, which is right for the offseason.

The 2026 schedule is loaded: 272 games.
