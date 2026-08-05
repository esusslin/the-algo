# Plan: August build-out and September operations

Two halves. The first is what can genuinely be improved before real games exist.
The second is what to do the moment data starts arriving — which is where the
actual answers come from.

Written 4 Aug 2026. Season opens 8 Sept.

---

## Where things actually stand

**Shipping and working:** odds ingest with adaptive polling and a credit ledger ·
devig across four methods · sharp-anchored consensus · cross-book shopping with
thin-market filters · Kelly sizing with correlation caps · A/B/C tiering ·
grading and CLV capture · weather · injuries · mobile UI (Picks, Open bets,
History, Algo, Admin) · invites and password reset over Twilio · red-team veto
agent · 94 tests in CI.

**Built, validated, and deliberately dark:**

| Component | Status | Why |
|---|---|---|
| Market-anchored residual model | 50.0% accuracy over 22 seasons | No edge on full-game markets. Expected. Weight 0. |
| Prop projection engine | Holdout KS 0.058–0.101 | Miscalibrated by more than the edge being hunted. Not priceable. |
| Artifact bundle | Working, hash-validated | Nothing worth loading into it yet. |

**The honest summary:** the product that ships on 8 Sept is a price-discrepancy
engine with a safety layer. That is a real product — it finds 2.5% edges today —
but nobody should describe it as predicting games.

---

# PART ONE — August (4 Aug → 7 Sept)

Ordered by value. The first section is worth more than everything below it.

## 1. Do nothing to the models

The strongest recommendation in this document.

Both models have been validated and both said no. The temptation over four idle
weeks is to keep tuning until a number turns green. Twenty attempts and one will
look good by chance, and you will have no way to tell which.

**Specifically do not:**
- Re-run the prop holdout after adjusting anything. It stops being a holdout.
- Add features to the residual model hoping Δbrier flips positive.
- Lower `MIN_EDGE_PCT` because the slate looks thin.

Everything genuinely blocking the models needs September data.

## 2. Tester feedback (highest actual value)

The only source of new information available before games start.

- [ ] Collect feedback from all 12 testers, in writing
- [ ] Sort into: confusing, broken, missing, cosmetic
- [ ] Fix "confusing" first — a misread probability is worse than an ugly card
- [ ] Watch specifically whether anyone misreads a **50% win / 6% edge** pick as
      a weak bet. If they do, the card needs rethinking, because that pattern is
      most of what the market engine produces

## 3. Operational hardening

- [ ] **Rotate `ODDS_API_KEY`** — it has been in a chat log for a week
- [ ] Rotate `JWT_SECRET_KEY` if it was ever pasted anywhere
- [ ] Confirm Railway is pinned to one replica
- [ ] External uptime pinger on `/health`, alerting on the `ok` field rather
      than the status code
- [ ] Watch `odds_credit_ledger` for a full week and compare to the 20k budget.
      Decide 20K vs 100K on measured burn, not my estimate
- [ ] Verify the Wed/Thu/Fri 17:00 ET injury cron actually catches publication.
      That timing is a guess

## 4. Worth building (in priority order)

**Live player prop stats.** Game-level live status works; props read "stat
unavailable" because box-score parsing isn't wired. Needed before props launch
regardless. ~1 day.

**Odds archive roll to parquet.** `odds_changes` grows unbounded. A Tuesday job
should roll rows older than three weeks out of SQLite. Not urgent in August;
becomes urgent by October. ~half a day.

**Grounded narratives (A3).** Replace templated pick text with one-liners built
from actual SHAP contributions. Only worth doing once a model contributes
something, so realistically later. ~1 day.

## 5. Explicitly NOT worth building in August

- **A1 signal extraction** — no text source is ingested, so there is nothing to
  extract from. Adding beat-reporter scraping is ToS-sensitive and fragile, and
  the structured Sleeper feed already covers status and practice participation.
- **Monte Carlo simulator** — significant build, and its main payoffs (alt
  lines, key numbers, SGP correlation) all need prop and derivative markets that
  aren't enabled.
- **M1 Bayesian ratings** — would improve mid-season regime handling, which is a
  problem that does not exist until mid-season.

---

# PART TWO — September operations

The plan for when real data starts arriving. This is where the answers are.

## Week 0 — 5 to 8 Sept (freeze and launch)

- [ ] **Freeze code Friday 5 Sept.** No changes into opening weekend.
- [ ] Purge all demo data
- [ ] Confirm `MIN_EDGE_PCT` (2.0 for collection, higher only if CLV justifies)
- [ ] Confirm `PUBLISH_MODEL_PICKS=false`, `ENABLE_PROPS=false`
- [ ] Set `BETTABLE_BOOKS` to accounts you actually hold
- [ ] Sunday 7 Sept: watch `poll_odds` tighten as kickoff approaches
- [ ] **Monday: confirm `capture_closing_lines` captured everything.** If it
      missed games, CLV is lost permanently for that week and the whole
      measurement programme starts a week late

## Weeks 1–3 — collect and verify

The goal is not profit. It is confirming the machinery is honest.

**Every Monday, check:**

| Check | Where | What's wrong if it fails |
|---|---|---|
| Closing lines captured for every pick | `picks.closing_price` | No CLV, no measurement |
| Grading matches reality | spot-check 5 picks vs box scores | Sign convention |
| Credit burn vs budget | `odds_credit_ledger` | Polling schedule |
| Red-team downgrade rate < 15% | Admin tab | Over-aggressive agent |
| Unresolved player names | `source_freshness` injuries row | Crosswalk decay |
| Volume of picks per tier | History tab | Threshold miscalibration |

**Do not draw conclusions about profitability.** Three weeks is 30–60 bets. The
confidence interval on that is enormous and the History tab prints it for a
reason.

## Week 4 — the first real decision point

By now there are ~4 weeks of CLV. This is the first honest evidence the project
has ever had.

- [ ] **Compute CLV by market class.** Moneyline vs spread vs total vs period.
      This is the number that tells you where edge actually lives — and it may
      well contradict everything assumed so far
- [ ] **Compute CLV by tier.** If A-tier isn't beating B-tier, the tier rules
      are decorative and need re-cutting from data
- [ ] **Compute CLV by book.** Which books are consistently slow? That's where
      to concentrate
- [ ] Set `MIN_EDGE_PCT` from measured CLV rather than a guess
- [ ] Decide the Odds API tier from measured burn

**Decision rule for publishing model picks:** only if the dark-launched residual
model shows positive CLV over ≥3 weeks and ≥40 picks. Otherwise it stays dark.
It currently shows nothing, so the default is dark.

## Weeks 4–6 — props, properly this time

Props are blocked on one specific thing, now identified.

**The blocker:** bust probability is currently a player's historical rate. It
isn't a player constant — it depends on *this game*. That's why the holdout
failed with too much mass in the bottom decile: quiet games are underpredicted,
so every `P(over)` runs high.

**What to build, in order:**

1. **Participation model.** Predict P(minimal involvement) from game-level
   inputs: injury-report status, practice participation trend, snap-share trend
   over recent weeks, spread magnitude (blowout risk), implied team total.
   Target: the player's own volume relative to his recent norm.
2. **Re-run the holdout** with the participation model in place. Tune on
   ≤2023, test once on 2024–25. **Same discipline — one run.**
3. **Only if holdout KS < 0.03**, enable that stat. Per stat, not all at once.
   Two working markets beat three where one lies.
4. Upgrade the Odds API plan before enabling — prop polling is ~144 credits a
   sweep.
5. Consistency reconciliation: projected player yards must sum to projected team
   yards. Also flags books whose own numbers don't reconcile.

**If the holdout fails again, props don't launch.** That is an acceptable
outcome.

## Weeks 6–10 — depending on what CLV says

Pick based on evidence, not this document.

- **If period markets (1H/2H) show good CLV** — they're less liquid and
  plausibly softer. Expand coverage there.
- **If a specific book is consistently slow** — poll it harder, concentrate
  volume, and expect to get limited eventually.
- **If props calibrate** — the simulator becomes worthwhile, since it unlocks
  alt ladders, key numbers at 3 and 7, and SGP correlation.
- **If nothing shows CLV** — that is a real answer. The market engine may be a
  small, honest edge and no more. Better to know than to keep building.

---

## Things that will go wrong

Written down now so they're recognised rather than debugged from scratch.

**Player name resolution will drift.** New players, trades, suffix changes.
Watch the unresolved count weekly; a jump means props are silently dropping.

**ESPN endpoints will break.** They're undocumented. Live scores will vanish
without warning. Everything degrades gracefully, but the Open bets tab will look
broken and someone will report it as a bug.

**The injury scrape will miss a week.** Sources change. The `/health` staleness
check catches it if you're watching.

**Books will limit you.** If the market engine works, this is a matter of when.
Track it as data — it's a real constraint on bet sizing and on how many
subscribers a prop product can support.

**You will be tempted to override the system.** A pick will look wrong and you
will skip it, or one will look great and you'll size up. Log those decisions.
Over a season, whether your overrides beat the system is itself measurable — and
usually the answer is no.

---

## The one-paragraph version

August: don't touch the models, collect tester feedback, rotate the keys, watch
credit burn. September: launch the market engine, verify closing lines are being
captured, and spend four weeks measuring CLV rather than counting wins. At week
four, use that CLV to decide where edge actually lives — it is the first real
evidence this project will have had, and it should override every assumption in
this document, including mine.
