# Prop engine and derivative markets — design

Written 17 Aug 2026. Supersedes the multi-agent proposal; see
`NOTES_ON_MULTIAGENT_PROPOSAL.md` for what was taken from it and what wasn't.

---

## 0. The one idea this rests on

**Don't fight the market on the game. Use its game-level price as an input.**

The full-game spread and total are the most efficiently priced numbers in sports.
That's normally framed as bad news. It isn't — it means the market is handing us
a free, high-quality forecast of *game script*, and game script is the single
biggest driver of player volume.

A 3-point favourite in a 51-point game implies a very different set of RB carries
and WR targets than a 10-point favourite in a 38-point game. The market has
already done that work, accurately, for nothing. We consume it and add value
where the market is thin: the individual player.

This resolves the tension in wanting to be good at both. We don't need to beat
the market on the game in order to beat it on props. We need to be *downstream*
of it.

---

## 1. Where the edge actually is

Ordered by how soft the market is:

| Market | Why it's priced the way it is | Our view |
|---|---|---|
| Full-game spread / total / ML | Highest volume in sports, sharp attention on every game | Effectively unbeatable. Consume, don't predict. |
| First half / second half | **Derived** from the full game by rule of thumb, not independently modelled. Lower volume. | Real opportunity. The derivation is an assumption and assumptions break. |
| Team totals | Derived from spread + total | Same as above, mildly softer |
| Player props | 500+ per game, priced largely off vendor feeds, adjusted manually only where money lands | **The best opportunity.** Nobody is carefully pricing WR3 receptions. |
| Alt lines / SGP | Priced off the main line with a shape assumption; parlay legs often treated as independent when they're correlated | Needs a simulator, which needs props working first |

Note the pattern: **edge lives wherever a price is derived rather than
independently formed.** That is a structural statement about how books operate,
not a claim about football knowledge.

---

## 2. The division of labour

The failure mode in the multi-agent proposal was assigning numbers to LLMs. An
LLM-produced probability has no calibration curve, no backtest, and no way to
tell whether a prompt change improved it or just moved it.

**Models produce numbers. Agents produce inputs and judgements.**

```
   messy world                 agents                    models              output
─────────────────      ──────────────────────    ────────────────────    ───────────
 injury report text
 depth chart churn  →   assemble, normalise,  →   calibrated layered  →   distribution
 beat writer notes      interpret, flag           statistical model       per player-stat
 snap trends            anomalies                                              │
 data source health                                                            ▼
                                                                        price vs book
                                                                              │
                                                                              ▼
                                                                        red team veto
```

Agents are good at: reading "limited participation, Wednesday" and knowing it
means something different from "did not participate, Friday"; noticing a starting
left tackle vanished from the depth chart; catching that the snap-count feed
hasn't updated since Tuesday. Heterogeneous, text-heavy, no ground truth to
regress against. That's genuinely agent-shaped work.

Agents never emit a probability. That rule is what keeps the system testable.

---

## 3. The layered player model

Every prop is the same chain. Build it once, apply per stat.

```
  P(active)  ──►  snap share  ──►  opportunities  ──►  per-opportunity  ──►  full
                                   (targets,           outcome dist         predictive
                                    carries,           (yards, catch)       distribution
                                    routes)
      ▲               ▲                  ▲                    ▲
      │               │                  │                    │
  injury         game script        play-calling          matchup
  status         (from market)      tendency              adjustment
```

### Layer 0 — Availability

Binary: does he play at all. Inputs: `report_status`, `practice_status` across
Wed/Thu/Fri, position, and the club's historical behaviour with that designation.
Point-in-time correct — `injuries_hist.date_modified` is what makes this honest.

*This layer is the identified blocker on the current prop engine.* Bust
probability is currently a player constant. It isn't.

### Layer 1 — Snap share, conditional on active

Not a constant either. Depends on recent trend, on whether the guy ahead of him
is out, and on game script — a blowout pulls starters, a close game doesn't.
Source: `snaps` (324k rows, `offense_pct`), `depth_charts`.

**Game script enters here, from the market.** Spread magnitude → blowout
probability. Implied team total → pace and play volume.

### Layer 2 — Opportunities

Targets, carries, routes run. This is where most of the variance lives, and it's
the layer books model least carefully.

Play-calling tendency matters and is computable: pass rate over expected from
`pbp`, plus FTN charting (2022+) for `is_play_action`, `is_screen_pass`,
`is_rpo`, `is_no_huddle`, `is_motion`. Coach-level tendencies are stickier than
player efficiency and persist across seasons.

Target share must be computed against *attributable* plays only — not all
dropbacks. Counting sacks and throwaways in the denominator is what made shares
sum to 0.762 last time.

### Layer 3 — Per-opportunity outcome, matchup-adjusted

Given a target, what's the distribution of yards. This is the layer the request
was really about — RB vs the front, WR vs the secondary.

What we can actually build from data on hand:

- **Receiving:** NGS gives `avg_cushion`, `avg_separation`,
  `avg_yac_above_expectation`, `percent_share_of_intended_air_yards`. Defensive
  side from `pbp` by target depth and `pass_location` — a defence that concedes
  deep-left is a different matchup from one that concedes short-middle.
- **Rushing:** NGS `percent_attempts_gte_eight_defenders` (box count) and
  `avg_time_to_los`. PFR advstats splits `rushing_yards_before_contact` from
  `after_contact` and gives `broken_tackles` — which separates the runner from
  the offensive line. That distinction is the whole game in RB modelling.
- **Passing:** NGS `completion_percentage_above_expectation`,
  `avg_time_to_throw`; FTN `n_blitzers`, `n_pass_rushers`, `is_catchable_ball`,
  `is_drop` — the last two separate QB skill from receiver hands, which is
  otherwise a large noise source.
- **QB rushing:** designed runs and scrambles are separate processes with
  different distributions. Never pool them.

**Honest data gap:** we have no coverage-assignment data, so "WR vs this specific
CB, shadowed or not" is not directly available. That's PFF/SIS paid territory.
The proxy above — defensive concession by target depth and field location, plus
NGS separation allowed — captures a good part of it. Worth knowing we're
approximating, and worth revisiting whether the CB layer is worth paying for
*after* the free version is calibrated, not before.

### Layer 4 — Compose the distribution

The output must be a **distribution, not a point estimate.** An over/under is a
question about the whole shape, and alt ladders are questions about the tails. A
mean with an assumed variance will misprice systematically.

Compose by simulation over the chain: sample availability, then snap share, then
opportunities, then per-opportunity outcomes. Uncertainty at each layer
propagates naturally, which is the point.

### Layer 5 — Correlation

Same-game legs are not independent. A QB passing over correlates with his WR1
receiving over. Books frequently price these as independent. That is a
structural, exploitable error — but it requires Layers 0–4 to be trustworthy
first, so it's last.

---

## 4. First half specifically

Halves deserve their own treatment rather than being scaled full-game numbers,
because the *market's* 1H number is a scaled full-game number. That's the
opening.

Things that differ systematically between halves and are computable from `pbp`:

- Scripted opening drives — many coaches script the first 15 plays
- Second-half adjustments, which vary hugely by coaching staff
- Clock-management effects concentrated in 2H
- Blowout effects (starters pulled) are almost entirely a 2H phenomenon
- Pace differs: 2H two-minute drill inflates plays

If a book derives 1H total as roughly 47% of the full game with a flat
adjustment, and the real split for a specific team pairing is reliably different,
that's an edge that requires no forecast of the game at all — only a forecast of
*how the game divides*. Much lower-variance target.

---

## 5. The closing-line model

Separate from everything above, and the most under-rated idea in the project.

Instead of predicting football, predict **where the line will close.** Then bet
today's number when it differs.

Why it's a better target:

- Far less noise. Game outcomes are one draw from a wide distribution; line
  movement is a much tighter, more predictable process.
- Vastly more observations. Every line change on every market on every game, not
  one result per game per week.
- Directly actionable and directly measurable — CLV *is* the metric, so there's
  no gap between what we optimise and what we're paid for.
- It doesn't require knowing anything the market doesn't. It requires knowing
  what the market will know soon.

Inputs are exactly the three feature slots that were designed into
`shared/feature_spec.py` and then hardcoded to zero: `line_move_spread`,
`book_dispersion`, `sharp_soft_delta`. Plus book-by-book latency profiles from
`odds_changes` — which book moves first, which follows, and how long the lag is.

This is closer to how sharp shops actually operate than anything else in this
document.

---

## 6. Evaluation — what would prove any of it

Non-negotiable, because this is where the last attempt went wrong.

**Per layer:** each layer gets its own calibration check. Layer 0 is a binary
classifier — reliability curve. Layers 1–3 are distributional — PIT histograms,
not z-scores, because these distributions are skewed and z-scores will lie.

**End to end:** KS against holdout, target < 0.03. Tune on ≤2023, test *once* on
2024–25. One run. The discipline is the point; a holdout re-run after adjustment
is not a holdout.

**Operationally:** CLV by market class, tier, and book. Nothing else settles an
argument. A model can pass every offline check and still fail here, and if it
does, the offline checks were measuring the wrong thing.

**Layer-level diagnostics exist to answer "which layer is wrong"** when the end
number is off. That's the real value of decomposition — not accuracy, but
debuggability. The last prop failure took real effort to trace to bust
probability; layered diagnostics would have pointed straight at Layer 0.

---

## 7. Build order, against the real calendar

Nothing here ships by 8 Sept, and nothing should. The model needs this season's
data, which is exactly what we don't have.

| When | Build | Gate to proceed |
|---|---|---|
| Now → 5 Sept | Nothing to models. Finish the totals-target run once and record it. Backfill the three market-microstructure features. | Code freeze 5 Sept |
| Weeks 1–3 | Collect. Verify closing-line capture, grading, credit burn. | Machinery honest |
| Weeks 4–6 | **Layer 0 (availability) + Layer 1 (snap share).** These two alone fix the diagnosed prop failure. Re-run holdout once. | KS < 0.03 on at least one stat |
| Weeks 6–8 | Layer 2 (opportunities) + play-calling tendency. Enable props **per stat**, not all at once. | CLV positive on enabled stats |
| Weeks 6–10 | First-half markets — independent of the prop chain, can run in parallel | CLV by market class |
| Weeks 8–12 | Layer 3 matchup adjustment | Improvement over Layer 2 baseline |
| Weeks 10+ | Closing-line model. Correlation / SGP once Layers 0–4 hold. | — |

Two working prop markets beat five where one lies.

---

## 8. What would make us stop

Written now, while it's cheap to be honest.

- Layer 0 holdout fails again after the availability model → props don't launch.
  That's an acceptable outcome, not a reason to keep tuning.
- Week 4 CLV is flat across every market class → the market engine is a small
  honest edge and no more, and the right response is to stop building rather
  than to build differently.
- First-half markets show no CLV → the "derived prices are soft" thesis is wrong
  and most of section 1 goes with it.
- Books limit the accounts before any of this matters → the constraint was never
  modelling.

---

## 9. What's carried over from the multi-agent proposal

- **Namespaced state per signal, one change at a time.** Becomes load-bearing the
  moment Layer 0 and Layer 1 both feed one number.
- **Injury as the highest-value signal** — correct, but aimed at props rather
  than spreads, where it's already public and priced.
- **A critic that can veto but not promote** — already exists as
  `src/ai/redteam.py`; the layered model gives it far more to check against.
