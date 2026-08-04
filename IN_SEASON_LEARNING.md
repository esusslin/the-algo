# In-season learning

How current-season data accumulates and improves picks as the season runs.

The original plan handled this implicitly — weekly data refresh, weekly retrain
— which is the intuitive answer and mostly the wrong emphasis. This document
separates the mechanisms by how much they actually matter.

---

## Two clocks

| Mechanism | Cadence | Impact | Effort |
|---|---|---|---|
| **Feature updating** | weekly, automatic | **Large** | already built |
| **Blend-weight adaptation** | weekly, from CLV | **Large** | not built |
| **Cold-start shrinkage** | weeks 1–6 | **Large early** | not built |
| **Tier threshold tuning** | monthly, from CLV | Moderate | not built |
| Model retraining | weekly | **Small** | planned |

Adding 16 games to a 7,000-game training set moves coefficients almost not at
all. Retraining weekly feels productive and is mostly theatre. The three items
above it are where in-season improvement actually comes from.

---

## 1. Feature updating (built)

Rolling team and player features recompute from current-season data on every
scoring run. A team's Week 8 opponent-adjusted EPA is a genuinely different
input than their Week 1 prior, and this is the single biggest source of
in-season improvement.

**Refinement — opponent adjustment must be recomputed, not incrementally
updated.** Raw EPA through six games means different things depending on whom
you played. The offence/defence decomposition has to be re-solved over the full
current season each week, because a Week 2 opponent's strength estimate changes
once you've seen them play six more times. Cheap to do (a ridge solve over one
season is milliseconds); expensive to get wrong.

**Refinement — exponential recency weighting.** A flat 8-game window treats
Week 1 and Week 8 identically, then discards Week 1 entirely at Week 9. Both are
wrong. Weight game `i` by `λ^(weeks_ago)` with `λ ≈ 0.93`, tuned by
walk-forward. Smooth decay beats a cliff.

---

## 2. Blend-weight adaptation (the real learning loop — not built)

This is the most valuable unbuilt thing in the system.

Final probability blends model and market:

```
logit(p_final) = w · logit(p_model) + (1 − w) · logit(p_market)
```

`w` is currently fit once, offline, per market type. It should adapt weekly from
**measured CLV** — the one signal that stabilises fast enough to act on
in-season.

```
For each market type, over a trailing window:
    observed_clv = mean CLV of picks where model disagreed with market
    if observed_clv > 0:   w drifts up   (model is finding something)
    if observed_clv < 0:   w drifts down (model is adding noise)
```

Constraints that keep this from becoming a random walk:

- **Cap the step.** `|Δw| ≤ 0.02/week`. A single lucky week must not double your
  model's influence.
- **Floor the sample.** No adjustment below ~30 picks in that market type.
- **Bound the range.** `w ∈ [0, 0.6]` for full-game markets. Even a good model
  should not outvote the market on a main spread.
- **Asymmetric response.** Move `w` down faster than up. Evidence that you're
  adding noise deserves more weight than evidence you're adding signal.

Why CLV and not results: CLV is measurable on every pick immediately and
stabilises within weeks. Win/loss over 250 bets a season is almost pure noise —
adapting on it would fit randomness.

---

## 3. Cold-start shrinkage (weeks 1–6 — not built)

Week 1 features are pure prior. Week 10 features are pure observation. The
transition should be explicit rather than emergent:

```
strength = (n/(n+k)) · observed + (k/(n+k)) · prior
```

where `n` = current-season games and `k` is fit per feature by walk-forward.

`k` differs a lot by statistic, and this is worth measuring rather than guessing:

| Feature | Stabilises | Rough `k` |
|---|---|---|
| Pass EPA | fast | ~4 games |
| Success rate | fast | ~4 games |
| Rush EPA | slow | ~10 games |
| Turnover margin | mostly luck | ~16+ games |
| Fumble recovery rate | pure luck | never — shrink fully to mean |

**Priors themselves need construction**, not just last season's number:
prior-season rating regressed to mean, plus roster turnover (snap-weighted
continuity), plus draft capital for rookies, plus a variance bump for coaching
or coordinator changes.

Early season is when models are weakest **and markets are softest**. Getting
cold start right is disproportionately valuable, not a nuisance to be tolerated
until the data arrives.

---

## 4. Regime changes within a season

A team's identity can change overnight — starting QB injured, coordinator fired,
a trade. Rolling windows handle this badly: they average across the break.

Three mitigations, increasing in cost:

1. **QB-conditional features** (planned). Team strength is computed *given* the
   projected starter, so a backup start doesn't corrupt the team's rating.
2. **Change-point flags.** New OC/DC/HC, or a QB change, resets or heavily
   down-weights pre-change observations for the affected units.
3. **State-space team ratings (M1).** A Bayesian random walk over weekly team
   strength handles this natively — recent evidence dominates automatically and
   uncertainty widens after a discontinuity. This is the principled answer and
   the strongest argument for building M1 rather than treating it as optional.

---

## 5. Tier thresholds from measured performance (not built)

Tier rules are currently my guesses. After ~6 weeks there's enough data to set
them from evidence:

- If A-tier CLV is not meaningfully better than B-tier, the gates aren't
  separating anything and should be re-cut.
- If a market class (say 1H totals) shows consistently positive CLV, lower its
  threshold. If props show negative CLV, raise theirs.
- If `MIN_EDGE_PCT` at 2% produces picks whose CLV is indistinguishable from
  zero, raise the floor.

**This is the feedback loop that turns the tier system from an assumption into a
measurement.** Review monthly, not weekly — thresholds chasing noise is worse
than thresholds that are slightly wrong.

---

## 6. Retraining (small effect, do it anyway)

Weekly retrain, walk-forward validated, current season included. It matters less
than the above but it's cheap and it keeps the artifact pipeline exercised — a
retrain path that runs weekly is a retrain path that works when you need it.

**Do not let a weekly retrain silently ship.** Every bundle passes the leakage
suite and a calibration check before it can be marked active. A model that
degrades gradually is far more dangerous than one that fails loudly.

---

## Proposed weekly rhythm (in season)

| Day | Action |
|---|---|
| Tue | nflverse refresh; recompute opponent adjustments over the season |
| Tue | Retrain locally, run leakage suite, export bundle |
| Tue | Update blend weights `w` from trailing CLV |
| Wed | Upload bundle, mark active after calibration check |
| Thu | Re-pull data (stat corrections land Mon–Wed) |
| Mon | Post-mortem: CLV by tier, by market class, by anchor type |
| Monthly | Re-cut tier thresholds from measured CLV |

---

## Priority if time is short

1. **Blend-weight adaptation from CLV.** The actual learning loop.
2. **Cold-start shrinkage.** Biggest effect in the weeks that matter most.
3. **Opponent adjustment recomputed weekly.** Cheap, and wrong otherwise.
4. Tier thresholds from measured CLV — after ~6 weeks of data.
5. Weekly retraining — last, despite being the thing that sounds most like
   "learning".
