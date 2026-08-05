# Roadmap — Aug 3 → Sept 8

Five weeks. What's built, what's missing, what actually has to happen.

**Priorities:** P0 = go-live blocker · P1 = ship if possible · P2 = post-launch

---

## Where things stand

**Live on Railway and working:** odds ingest with adaptive polling and a credit
ledger · devig (4 methods) · sharp-anchored consensus · line shopping with
thin-market filters · edge detection · Kelly sizing with correlation haircuts ·
A/B/C tiering · grading and CLV · auth, invites, Twilio · mobile UI (Picks,
Track, Algorithm, Admin) · demo data for tester feedback.

**The honest gap:** there is no football model. Every pick to date is arithmetic
on market prices — finding books that disagree with Pinnacle. Nothing has looked
at a snap of football. `shared/feature_spec.py` has 18 features: 6 market,
7 situational, 5 weather. Zero team or player performance features. No training
code, no `research/` directory, no trained artifact.

That's the largest remaining piece of work and it drives the schedule below.

---

## P0 — go-live blockers

### Operational gaps (week of Aug 4)

- [ ] **Schedule the jobs that exist but never run.** `grade_picks`,
      `grade_user_bets`, `capture_closing_lines`, `live.fetch_live` are all
      written and unregistered. Without `capture_closing_lines` running *before*
      kickoff there is no CLV — and CLV is the only signal that tells you
      whether any of this works before results do. **Half a day.**
- [ ] **Weather ingestion.** Stadium lat/lon/elevation are in `src/teams.py`;
      the Open-Meteo fetcher isn't written. Needed for totals modelling and it's
      a genuine edge source (wind gusts move totals; books are slow to adjust).
      **1 day.**
- [ ] **Verify the injury feed catches real reports.** The Wed/Thu/Fri 5pm ET
      cron is a guess. Official reports land in the afternoon; confirm we're not
      missing the publication window by an hour. **Check in week 1 of preseason.**
- [ ] **Per-market-class tier thresholds.** Currently one threshold set for all
      markets. A 3% edge on a WR4 prop is not the same asset as 3% on a main
      spread — different variance, different limits. Props should need a higher
      bar for the same tier. **Half a day.**
- [ ] **Book accounts.** `BETTABLE_BOOKS` is my guess at a US list. Set it to
      the books you can actually bet at, or every edge number is fiction.

### Validation (week of Aug 11) — do NOT skip

- [ ] **`tests/` directory with the leakage suite.** Shuffle test, future-data
      ablation, feature-availability assertions, and a tripwire that halts on
      any ATS backtest above 55%. **This is the single highest-value item in
      this document.** A subtle leakage bug producing a beautiful backtest is
      the most likely way this project fails, and it fails silently.
      **2 days.**
- [ ] **Grading sign-convention tests in CI.** `grading selftest` passes 11
      hand-verified cases; wire it into a test run so it can't regress.
      **2 hours.**
- [ ] **Walk-forward harness** with market-implied baseline built in. Every model
      is scored against the market, never in isolation. **2 days.**

### Deploy hygiene (week of Sept 1)

- [ ] Rotate `ODDS_API_KEY` — the dev key was pasted into a chat
- [ ] Confirm one Railway replica (SQLite corruption is unrecoverable)
- [ ] Purge demo data before real users rely on the record
- [ ] Disclaimers visible: informational only, no guarantees, 21+,
      responsible-gambling resources (already in the Algorithm tab — confirm
      they survived)
- [ ] External uptime pinger on `/health`, alerting on the `ok` field not the
      status code

---

## P0 — the models

The reason for the schedule. Five weeks is enough for game-level models if the
research plane starts by Aug 11.

### Research plane (Aug 11–17)

- [ ] `research/` — never imported by `server.py`
- [ ] DuckDB warehouse over 1999–present play-by-play (run the backfill locally,
      not on Railway — it would blow out the volume)
- [ ] Transform chain: pbp → drives → games → team-weeks → player-weeks
- [ ] **Point-in-time feature store.** Every feature a pure `(entity, as_of)`
      function. Bitemporal: `event_time` and `knowledge_time` on everything.
- [ ] Leakage suite green before any model is trained

### Features (Aug 18–24)

- [ ] Opponent-adjusted EPA (ridge or hierarchical decomposition into
      offence/defence/situation) — the step that separates a real model from a
      toy one
- [ ] Success rate, explosive-play rate, early-down vs. 3rd-down splits
- [ ] Neutral-script EPA (exclude win prob outside 0.10–0.90 — garbage time
      distorts everything)
- [ ] Regression-to-mean on luck-driven stats (turnover margin, fumble
      recoveries, 3rd-down over/under expectation)
- [ ] QB-conditional team strength — backup QB is worth 4–7 points of spread
- [ ] O-line continuity, snap-weighted roster continuity
- [ ] Pressure rate generated vs. allowed (FTN charting — far more predictive
      than sack rate)
- [ ] Rest, travel, time zones, altitude, surface, kickoff slot
- [ ] Weather interactions (wind gust thresholds, not linear)
- [ ] Market features: line movement, cross-book dispersion, sharp-soft delta

### Models (Aug 25–31)

- [ ] **M3 market-anchored residual model first.** Predict
      `actual_margin − market_spread`, not the margin. The market has done 95%
      of the work; you're modelling only where it's systematically wrong. Low R²
      is the correct and healthy outcome.
- [ ] M2 GBM on the full feature set (margin and total as regression targets,
      then derive binaries from the implied distribution)
- [ ] Calibration (isotonic/Platt) + reliability diagrams. **Calibration matters
      more than accuracy** — a 55% model that says 70% will bankrupt you.
- [ ] Artifact bundle + `feature_spec` hash validation + refuse-to-serve on
      mismatch
- [ ] Blend weights `w` per market type — expect ~0.05–0.20 for full-game
      spreads, higher for softer markets

### Dark launch (Sept 1–8)

- [ ] `PUBLISH_MODEL_PICKS=false` — models run, get graded, get CLV-tracked,
      stay invisible
- [ ] Week 1 cold-start priors: prior-season ratings regressed to mean, roster
      turnover, draft capital, coaching changes. **Early season is when models
      are weakest and markets are softest** — solving cold start is
      disproportionately valuable.
- [ ] Freeze Sept 5. No changes into opening weekend.

---

## P1 — ship if time allows

- [ ] **AI red-team agent (A2).** Downgrade-only veto that catches what features
      can't see: coaches announcing snap counts, teams resting starters, news
      after the data snapshot. Highest-value AI role and bounded-risk by design.
      **2 days.**
- [ ] **Signal extraction (A1).** Beat-reporter and injury-report text → structured
      features via Haiku. Genuinely differentiating — nobody else has this
      vectorised. **2 days.**
- [ ] **Live player prop stats.** Game-level live status works; props show "stat
      unavailable" until box-score parsing lands. **1 day.**
- [ ] **Grounded narratives (A3).** Replace templated pick text with SHAP-grounded
      one-liners. **1 day.**
- [ ] Replace `prompt()` bet logging with a real sheet UI
- [ ] SMS alerts on A-tier picks and Sunday inactives

---

## P2 — after launch

- [ ] **Player props** (`ENABLE_PROPS`) — Week 4–6, and now with a specific
      blocker identified.

      **Status (Aug 2026):** the distributional engine is built — volume ×
      efficiency, lognormal with a participation mixture, walk-forward
      calibration. It was evaluated on a clean holdout (tuned ≤2023, tested
      once on 2024-25) and **fails**:

      | stat | holdout KS | train→holdout drift |
      |---|---|---|
      | receptions | 0.058 | −0.002 (generalises) |
      | rush_yards | 0.073 | +0.025 (overfit) |
      | rec_yards | 0.101 | +0.024 (overfit) |

      Target is KS < 0.03. At 0.06–0.10 the probabilities are off by more than
      the edge being hunted, so a mispricing cannot be distinguished from model
      error. Do not price props on this.

      **The diagnosed cause:** bust probability is modelled as a player's
      historical rate, but it isn't a player constant — it depends on *this
      game*. Injury-report status, snap-share trend, whether the team is a big
      favourite likely to rest him, the game total. Every decile table shows
      the same signature: too much mass in the bottom 10%, meaning quiet games
      are underpredicted and every `P(over)` runs high.

      **What unblocks it:** `p_bust` becomes a model with game-level inputs
      rather than a historical average. The inputs are the injury feed, snap
      counts and market-implied game script — all of which only started
      collecting in Aug 2026. Several weeks of in-season data makes this
      buildable; it is not buildable now.
- [ ] **Monte Carlo simulator** (`ENABLE_SIMULATOR`) — Week ~8. Unlocks alt
      lines, key numbers at 3 and 7, 1H/2H, team totals, and SGP correlation.
      Also fixes sharp-anchor coverage when Pinnacle quotes a different number.
- [ ] M1 Bayesian state-space team ratings
- [ ] Ask tab (natural language over your own data)
- [ ] Games tab (per-game matchup detail)
- [ ] Weekly post-mortem agent (A7)
- [ ] Odds archive roll to parquet (SQLite bloat prevention)

---

## Calendar

| Week | Focus | Exit criteria |
|---|---|---|
| **Aug 4–10** | Operational gaps, tester feedback round 1 | Grading + CLV jobs running; weather ingesting; UI feedback collected |
| **Aug 11–17** | Research plane + **leakage suite** | Suite green; walk-forward harness reports vs. market baseline |
| **Aug 18–24** | Feature engineering | Team-strength features built and leak-free |
| **Aug 25–31** | Train, calibrate, bundle | Artifact loads; refuses on hash mismatch; models dark-launched |
| **Sep 1–8** | Cold-start priors, monitoring, freeze | `/health` green; demo purged; key rotated; freeze Sept 5 |

---

## Decisions needed

| When | Decision |
|---|---|
| Mid-Aug | Odds API tier — measure a real week's burn, then decide 20K vs 100K |
| Late Aug | `MIN_EDGE_PCT` for launch — set from measured CLV, not a guess |
| Late Aug | Which books you actually hold accounts at |
| Sept 5 | Ship game models dark, or ship market-engine only |

---

## If only five things get done

1. **Leakage test suite.** Everything else is worthless if the backtest lies.
2. **`capture_closing_lines` scheduled.** No CLV, no way to know if this works.
3. **Market-anchored residual model.** The highest-value single model.
4. **Calibration.** Miscalibrated probabilities plus Kelly is how bankrolls die.
5. **Dark launch discipline.** Don't publish model picks to paying users until
   live CLV earns it.

---

## Honest expectations for Sept 8

The market engine will be solid — that part is built and working.

Game-level models are achievable but will most likely show **no edge** on
full-game spreads and totals. Those are among the most efficiently priced
markets in the world, and that's the expected result rather than a failure. Real
edge lives in props and derivatives, which are P2.

So opening day is: a working market engine, models running dark and accumulating
CLV, and a product your testers already know how to use. That is a good place to
be, and it's better than shipping a model nobody has validated.
