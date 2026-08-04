# Matchup modelling

How current-season data becomes a projection: unit-level decomposition,
opponent adjustment, and conditioning on who is actually playing.

This is the spec for the feature work in the Aug 18–24 block.

---

## Three principles

**1. Decompose to units, not teams.** "Buffalo's offence vs Miami's defence" is
too coarse to be useful. What predicts is *Buffalo's pass offence on early downs
vs Miami's pass defence on early downs*, and separately their rush units, and
separately red zone. A team can have a top-5 pass defence and a bottom-5 run
defence; a team-level rating averages that away.

**2. Nothing is meaningful without opponent adjustment.** A defence allowing 3.8
yards per carry has told you nothing until you know whether they faced the
league's best backs or its worst. Every unit rating is the residual after
solving for who they played.

**3. Volume before efficiency.** For props especially: how many times a player
touches the ball is far more predictable than what he does with it. Target share
and snap share are stable; yards per target is noisy. Model volume carefully,
then apply an efficiency distribution.

---

## Opponent adjustment — the mechanism

Raw rolling stats are confounded by schedule. The fix is a ridge regression over
every play in the current season:

```
epa_play ≈ β_off[offense_unit] + β_def[defense_unit] + β_situation + ε
```

fit separately for pass and rush, with situation covering down, distance, field
position and score state. The coefficients are the opponent-adjusted ratings.

Three properties that matter:

- **Re-solve weekly over the whole season**, don't update incrementally. A Week 2
  opponent's rating changes once you've seen them six more times, which changes
  what your Week 2 performance meant.
- **Ridge penalty handles small samples.** Early season, coefficients shrink
  toward zero automatically — which is the correct behaviour, not a bug.
- **Exponential recency weight** (`λ ≈ 0.93` per week, tuned by walk-forward) so
  Week 8 counts more than Week 1 without Week 1 falling off a cliff.

Output: per team, per week, adjusted ratings for pass offence, rush offence,
pass defence, rush defence — plus splits by down, field zone, and target depth.

---

## Worked example: WR receiving yards

The chain, in order. Each step is its own small model.

```
1. Team pass volume        = f(pace, PROE, projected game script)
2. Player target share     = f(rolling share, availability of others, alignment)
3. Player targets          = 1 × 2
4. Air yards per target    = f(player aDOT, coverage matchup)
5. Catch rate              = f(aDOT, opponent EPA allowed at that depth, separation)
6. YAC                     = f(player YAC-over-expected, opponent tackling)
7. Receiving yards         = distribution over 3 × (4,5,6), not a point estimate
```

**Where the matchup enters:**

- **Step 1** — projected game script comes from the market itself. Implied team
  total (from spread + total) tells you scoring opportunity; spread magnitude
  tells you whether they'll be throwing to catch up or running to close it out.
  Large underdogs throw more; that inflates receiving volume while making
  anytime-TD less likely.
- **Step 2** — alignment matters. A slot receiver faces the nickel corner, not
  the outside CB1. Slot vs wide rate comes from participation data; the
  opponent's EPA allowed is split the same way.
- **Step 5** — this is the real matchup term. Opponent pass defence EPA allowed
  **by target depth** (0–9 / 10–19 / 20+ yards). A deep threat against a defence
  that's stout underneath but leaky over the top is a very different projection
  than the reverse, and team-level pass defence rating hides that completely.

**Coverage tendency:** man/zone rates come from FTN charting, which updates
in-season. Some receivers are markedly better against one coverage family.

**Output is a distribution, not a number.** A prop is `P(X > line)`, so the
spread of the distribution matters as much as its centre. Two receivers with
identical projected yards and different variance have genuinely different fair
prices.

---

## Worked example: RB rushing yards

```
1. Team rush volume    = f(pace, projected game script, PROE)
2. Backfield share     = f(rolling carry share, availability, game script)
3. Carries             = 1 × 2
4. Yards per carry     = f(O-line run block, opponent run defence, box count)
5. Rushing yards       = negative-binomial volume × gamma-ish YPC distribution
```

**Game script dominates here** more than for receiving. A three-score favourite
runs out the clock; the same back in a three-score deficit gets a handful of
carries and some checkdowns. Spread magnitude is one of the strongest features
in the whole prop model.

**Run defence must be split by direction and gap** where data allows — a defence
strong up the middle and soft on the edge is a different matchup for a zone-run
back than a power back.

**O-line vs front seven** is a unit matchup in its own right: run-block win rate
and adjusted line yards against the opponent's run-stop rate.

---

## Worked example: QB rushing yards

The one people model badly, because it's **two different processes averaged
together**:

- **Designed runs** — scheme-driven, stable, predictable from usage rate
- **Scrambles** — pressure-driven, and therefore a function of the *opponent's
  pass rush* and the QB's scramble tendency under pressure

Modelling them jointly produces a mushy average that fits neither. Split them:

```
designed_rush_yards  = f(designed rush rate, goal-line usage, opponent run D)
scramble_yards       = f(opponent pressure rate, own pass-pro, QB scramble rate)
```

Opponent pressure rate generated vs. the offence's pressure rate allowed comes
from FTN charting — both update in-season. This is also why pressure rate beats
sack rate as a feature: sacks are the noisy outcome, pressure is the stable
process.

---

## Availability — the layer that governs everything

Every projection above is conditional on **who is actually on the field.** This
is where most public models are weakest and where the injury feed earns its
keep.

**Chain:** injury report → practice participation trend → depth chart →
projected snap share → volume features.

Practice participation is more informative than the game-status label. `DNP /
DNP / LP` and `LP / LP / FP` both end at "Questionable" and mean opposite
things.

**Role redistribution is a model, not an afterthought.** When a WR1 is out, his
targets don't vanish — they redistribute, and not evenly. The WR2's share rises
more than the TE's; the slot absorbs more than the outside. Fit the
redistribution empirically from historical games where a starter missed, rather
than scaling everyone proportionally.

**QB conditioning is first-class.** A backup QB is worth roughly 4–7 points of
spread depending on the pair. Team strength must be computed *given the
projected starter*, not as a team average with an injury flag bolted on. That
single feature is worth more than most of the matchup detail above.

**Inactives at 90 minutes pre-kickoff** are the highest-value information moment
of the week. Everything above re-projects at that point, and the markets that
haven't updated yet are where the edge is.

---

## What updates in-season, and what doesn't

A real constraint on how granular the live model can be:

| Source | In-season? | Use |
|---|---|---|
| Play-by-play (EPA, air yards, YAC) | ✅ nightly | Core unit ratings |
| FTN charting (pressure, blitz, play-action, motion) | ✅ 4×/day | Pass rush vs pass pro, coverage tendency |
| Snap counts | ✅ 4×/day | Volume, role |
| Next Gen Stats (separation, cushion, RYOE) | ✅ nightly | Efficiency, matchup |
| PFR advanced (pressures, broken tackles) | ✅ daily | Corroboration |
| **Participation (personnel groupings, coverage type)** | ❌ **postseason only** | Prior-season tendencies only |
| Injuries | ⚠️ our own scraper | Availability |

**The participation gap matters.** Personnel groupings (11 personnel rate,
nickel/dime rate) and per-play coverage type are only published after the
postseason. So:

- **Live:** use FTN charting for current-season coverage and pressure signal
- **Prior-season tendencies** carry forward reasonably well for scheme identity
  — a defence that played 70% zone last year usually still does
- **Never** build a live feature that depends on current-season participation.
  It will silently have no data on Sunday.

---

## How it composes into a pick

```
matchup features ──┐
availability ──────┤
game script ───────┼──► player projection (distribution)
weather ───────────┤              │
market context ────┘              ▼
                        P(over line) vs P(under line)
                                  │
                                  ▼
                     blend with market fair price (w)
                                  │
                                  ▼
                          edge → Kelly → tier
```

And the consistency check from the architecture doc still applies: projected
player receiving yards must sum to projected team passing yards, which must
reconcile with the projected team total. Enforcing that improves accuracy by
borrowing strength from the better-estimated team numbers — and when a *book's*
numbers fail to reconcile, that inconsistency is itself a bet.

---

## Honest limits

- **This is where the edge is, and it's still hard.** Books employ people doing
  exactly this. The edge comes from breadth (400 props they can't all price
  carefully) and speed (inactives), not from out-analysing them on Mahomes
  passing yards.
- **Matchup effects are smaller than they feel.** "Elite WR vs bad secondary"
  is mostly already in the line. The residual edge is in the second-order stuff
  — depth splits, alignment, redistribution after an injury.
- **Sample sizes are brutal.** A WR has ~100 targets a season. Split by
  coverage type and target depth and you have a dozen observations per cell.
  Hierarchical shrinkage is mandatory, not optional.
- **None of this is built yet.** The data is flowing; the feature layer is the
  Aug 18–24 block.
