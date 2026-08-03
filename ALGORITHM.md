# The algorithm

Exactly what the system does today, in order, with the file that does it. Kept
in sync with the code — if you change a threshold, change it here.

The Algorithm tab in the app is the plain-English version of this document for
subscribers. This one is the technical record.

---

## What it is (and isn't)

It compares the price at every sportsbook against a best estimate of true
probability, and flags cases where a book pays more than the outcome is worth.

**It is not predicting who wins.** No model of football is running yet. Every
pick currently produced is arithmetic on live market prices — finding books that
disagree with the sharpest available price. Statistical models are built but
dark-launched (`PUBLISH_MODEL_PICKS=false`), and won't publish until they've
logged positive CLV on live data.

---

## Pipeline

```
odds_api.poll ──► parse ──► persist ──► consensus.build_fair_prices
                                              │
                                              ▼
                              shop.find_opportunities  ◄── filters
                                              │
                                              ▼
                              generator.generate ──► kelly ──► tiers ──► picks
```

### 1. Ingest — `src/fetchers/odds_api.py`

The Odds API v4, `americanfootball_nfl`, american odds, regions `us` + `eu`
(Pinnacle lives in `eu`).

Polling is adaptive: cost is `markets × regions` per call, so frequency scales
with time-to-kickoff rather than running flat out.

| Time to kickoff | Featured (h2h/spreads/totals) | Period (1H/2H) | Props |
|---|---|---|---|
| < 2h | 5 min | 60 min | 20 min |
| < 12h | 10 min | 180 min | 60 min |
| < 3d | 30 min | 720 min | 360 min |
| < 10d | 60 min | — | — |
| < 30d | 6h | — | — |
| beyond | 12h | — | — |

Each tier records its last poll and skips until its interval elapses. Without
that throttle the 5-minute scheduler tick fires every tier every time — roughly
52,000 credits/month against a 20,000 budget.

Featured markets come from the bulk `/odds` endpoint (6 credits for all games).
Period markets and props return 422 there and must be requested per event —
96 and 144 credits per sweep respectively. That cost difference is why props
stay off until Week 4.

**Storage is three-tier.** `odds_current` upserts (~24k rows, never grows);
`odds_changes` appends only when a price actually moves; older history rolls to
parquet. Naive appends would put 5–10M rows/week into SQLite.

### 2. Devig — `src/market/devig.py`

Posted odds sum to more than 100%. The excess is the book's margin, and removing
it correctly matters most exactly where edges live.

| Method | Behaviour |
|---|---|
| multiplicative | divide by the sum; overprices favourites |
| additive | subtract margin equally; overprices longshots |
| **power** (default) | solve `k` where `Σ pᵢᵏ = 1`; best general choice |
| shin | models insider share; closest to sharp-book behaviour |

At −110/−110 all four agree. On a +2000 longshot they imply fair prices from
+2032 to +2728. `fair_prices_all_methods()` returns all four — **if a bet only
shows edge under one method, it is not a bet.**

### 3. Consensus — `src/market/consensus.py`

Each book is devigged **separately** — devigging a blend removes an average
margin no book actually charges.

Fair value is a weighted median across books (median, not mean, so one stale
book can't drag it). Pinnacle carries 5× weight; Circa 3×; reduced-juice books
2×; everything else 1×.

Also recorded per market: `sharp_prob` (Pinnacle alone), `book_count`, and
`dispersion` (standard deviation of fair probability across books).

**Two-sided pairing is explicit, not inferred.** Totals and props pair on the
same line (Over 48.5 / Under 48.5). Spreads pair on *opposite* lines (home −2.5
with away +2.5). Getting this wrong devigs mismatched outcomes and produces fair
prices that look entirely plausible and are silently wrong.

### 4. Shop and filter — `src/market/shop.py`

Best available price per market across all books, then three filters, all on by
default:

- **`main_line_only`** — evaluate only the number most books quote. A consensus
  built from four books, one of which is the book you'd bet into, isn't a fair
  price. In testing, a 3-book minority line produced a fake +11% edge.
- **`min_books = 8`** — thin markets manufacture edges.
- **`bettable_only`** — an edge at a book you have no account with isn't an edge.
  Set `BETTABLE_BOOKS` to your real accounts.

Fair value anchors on Pinnacle where available, falling back to consensus. This
is deliberate: consensus includes the soft book you're betting into, which drags
fair value toward the mispricing and understates the edge.

### 5. Edge

```
edge = p_fair × (decimal_odds − 1) − (1 − p_fair)
```

At −110 you need 52.38% to break even. A market with no mispricing scores
−4.55%, which is exactly the vig — **a median edge near −4.55% across all
markets is the expected healthy null result**, not a bug.

### 6. Size — `src/picks/kelly.py`

```
f* = (p·b − q) / b          b = decimal − 1
stake = min(f* × FRACTION, CAP) × bankroll
```

`KELLY_FRACTION = 0.125` while unproven, 0.25 maximum ever. Hard cap of 2% of
bankroll per bet regardless of what Kelly says.

The reason is asymmetry: believing 58% when the truth is 55% — a small error for
a sports model — produces a **2.15× overbet, and that ratio is identical at
every fraction.** Fractional Kelly doesn't fix miscalibration; it makes the
consequence survivable. This is why calibration matters more than accuracy.

Then two correlation haircuts: **4% max exposure per game** (over on a total,
over on both QBs' passing yards and over on the WR1 are close to the same bet
four times) and **25% max across the slate**. Crude — a proper simultaneous
Kelly needs the correlation matrix the simulator will produce — but it errs
toward betting less.

### 7. Tier — `src/picks/generator.py`

Four gates. All must clear.

| Tier | Min edge | Min books | Sharp anchor | Max dispersion |
|---|---|---|---|---|
| A | 5.0% | 15 | required | 0.03 |
| B | 3.0% | 10 | optional | 0.05 |
| C | 2.0% | 8 | optional | any |

`MIN_EDGE_PCT` is a global floor beneath all three; tiers subdivide above it.

A-tier is meant to be rare — a handful a week at most. **If you start seeing 30
A-tier picks in a week, something is broken.** That's a deliberate alarm.

### 8. Grade and CLV — `src/picks/grading.py`

Results resolve against final scores, with pushes counted as pushes. Sign
conventions are unit-tested against 11 hand-verified cases (`grading selftest`)
because a flipped spread sign produces a plausible ~50% hit rate while silently
inverting every result.

`capture_closing_lines()` snapshots the final pre-kickoff price **before**
kickoff — odds vanish afterwards, and without a closing price there is no CLV.

CLV is the difference in EV between the price taken and the closing price, at
the closing fair probability. It is reported first everywhere, because over a
few hundred bets win rate is mostly noise, and CLV stabilises far sooner than
profit does.

---

## Built but not enabled

| Capability | Flag | Target |
|---|---|---|
| Model picks published | `PUBLISH_MODEL_PICKS` | after 2–3 weeks positive live CLV |
| Player props | `ENABLE_PROPS` | ~Week 4 (needs the $59 credit plan) |
| Monte Carlo simulator | `ENABLE_SIMULATOR` | ~Week 8 — unlocks alt lines, key numbers, SGP correlation |
| SMS alerts | `ENABLE_SMS` | after a full dry run |

The model portfolio (Bayesian team ratings, GBM, market-anchored residual,
simulator, distributional prop models) is specified in
`nfl_betting_system_architecture.md`. None of it is in the live pick path yet.

---

## Known limits

- **Main spreads and totals are among the most efficiently priced markets in the
  world.** An empty pick list is usually the correct answer, not a failure.
- **Sharp anchor coverage is partial.** When Pinnacle quotes −3.5 and DraftKings
  −3, those are different bets; comparing them properly needs the margin
  distribution the simulator will provide. Moneyline has full coverage (no line
  to mismatch); spreads and totals lean on consensus more often.
- **Injury data is ours to maintain.** nflverse's feed died after 2024; we scrape
  Sleeper. Only ~27% of injured players carry a `gsis_id`, so name matching does
  most of the work and ambiguous names are quarantined rather than guessed.
- **Player prop live stats aren't wired.** Game-level bets show live status; prop
  bets show "stat unavailable" until box-score parsing lands.
- **ESPN endpoints are undocumented** and will break without notice. Everything
  consuming them degrades rather than erroring.
- **Nothing here models the unmodellable** — a coach announcing a snap count on
  Friday is in no dataset. That gap is what the AI red-team agent is designed to
  cover, and it isn't built yet.

---

## Honest expectations

Full-game spreads and totals: most likely **no edge**. That's the expected
result and why props and derivatives are on the roadmap.

Props and derivatives with disciplined line shopping: **2–5% ROI is a good
outcome.** Anyone claiming 15% hasn't accounted for vig, limits and slippage.

Biggest risk by a wide margin is a subtle leakage or labelling bug producing a
beautiful backtest. The validation protocol exists because of that, not despite
it.
