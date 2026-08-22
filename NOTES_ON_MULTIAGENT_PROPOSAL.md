# Notes on the multi-agent / LangGraph proposal

Written 17 Aug 2026. Season opens 8 Sept; code freeze 5 Sept.

**Verdict: don't build it. Take one piece.**

The proposal is competent architecture. The problem is that it answers a question
we already answered with evidence, and it arrives three weeks before a freeze.
Worth being clear that the other agent couldn't have known this — it didn't have
the validation results.

---

## 1. The core issue: architecture isn't the bottleneck

The proposal assumes that if you decompose prediction into injury, matchup,
market, situational and weather signals and combine them carefully, you get a
prediction worth betting. We tested a version of that assumption:

| What we tested | Result |
|---|---|
| Market-anchored residual model, 22 seasons | 50.0% accuracy, Brier 0.2530 vs market 0.2500 |
| Prop projection engine, honest holdout | KS 0.058–0.101 against a 0.03 target |

The residual model already consumed opponent-adjusted efficiency, rest, travel,
weather and injury availability. That is four of the five proposed subgraphs. It
produced no edge, and every leakage check passed, so the null result is real.

The reason is not that the features were combined badly. It's that **injury
reports, DVOA-style efficiency, rest, travel and weather are all public by
Wednesday and priced by Thursday.** Reorganising public information into
subgraphs doesn't create private information. Nothing in the proposal changes the
information content of the inputs.

## 2. Things it proposes that already exist

- **Critic/Audit subgraph** → `src/ai/redteam.py`. Already has the three
  properties that matter and were learned the hard way: fails open, downgrade-only
  (it can veto, never promote), evidence required. Plus an explicit
  "missing data is not evidence" rule added after it gutted 41% of a slate for
  absent context.
- **Aggregator with per-signal weights** → `blended_probability()` in
  `src/picks/generator.py`. Single blending point, log-odds space, per-market
  class weight from the artifact bundle. Weight is currently 0 because the model
  hasn't earned more.
- **HIGH/MEDIUM/LOW confidence tier** → A/B/C tiering, with per-market-class
  multipliers so a 3% edge on a WR4 prop doesn't outrank 3% on a main spread.
- **Namespaced state so components don't clobber each other** → `prob_components`
  JSON on every pick, which is what powers the "where does this number come
  from?" expander.

## 3. Where LLM agents are the wrong instrument

A subgraph that "computes a per-position weighted impact score" via an LLM
returns a number with no calibration curve, no backtest, and no PIT diagnostic.
We can't walk-forward it, can't measure whether it's over- or under-confident,
and can't tell whether a change to the prompt improved it or just moved it.

Replacing deterministic feature computation with LLM inference makes the system
**less** testable, not more. The current split is the right one: numbers computed
numerically and validated statistically; the LLM used only for judgement tasks
where there's no ground truth to regress against — narration, and vetoing
something that looks structurally wrong.

## 4. The evaluation strategy measures the wrong thing

Per-subgraph LangSmith datasets test whether a subgraph does what it says it
does. That's conformance, not edge. A subgraph can score 100% on its own eval and
contribute nothing — the residual model would pass every conformance test ever
written while being anti-predictive when confident (predicted 63%, actual 45%).

**The only unit of evaluation that matters here is CLV against the closing
line.** It's the leading indicator, it stabilises far faster than P&L, and it
doesn't care how the number was produced.

## 5. The proposal contains its own counterargument

> "17 games/season means that threshold takes years, not weeks"

Correct, and it's the strongest line in the document. But it argues against the
architecture rather than for a workaround. If you can't fit weights on available
data, don't build a five-signal weighted aggregator — you'd be shipping five
hand-set weights and calling the result a model.

## 6. What's actually worth taking

**Line movement, reverse line movement, and steam detection.** This is the one
genuinely new signal in the proposal. Everything else is either already built or
already tested and found empty. It is:

- a *market microstructure* signal, not a game-outcome prediction — so it's not
  competing with public information the market has already priced
- cheap to compute from data we already store in `odds_changes`
- deterministic, needing no LLM and no subgraph
- directly testable against CLV, which is the measurement we're about to start
  collecting anyway

Nearest thing to a real idea in the document. It belongs in the existing market
engine as a feature on the opportunity record, not as an agent.

## 7. Timing

`PLAN_AUG_SEPT.md` item one is "do nothing to the models," and the reasoning
holds: with four idle weeks and no new information, tuning until a number turns
green produces a number that turned green by chance. A framework rewrite three
weeks before freeze is a larger version of that same mistake, with a deploy risk
attached.

## 8. What to say back

Roughly: the architecture is sound and the instinct to isolate signals so one
can't silently corrupt another is right. But we've already run the experiment
these subgraphs are designed to run, on 22 seasons, and it came back flat. The
constraint isn't orchestration — it's that the inputs are public. Send the RLM /
steam detection idea through; hold the rest until Week 4 CLV tells us where edge
actually lives, because that evidence should decide the architecture rather than
the other way round.
