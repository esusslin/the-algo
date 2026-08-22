# Agent layer — deep agent, PTC, and the annotation queue

Written 17 Aug 2026, after reviewing the direction adops-ai-assistant is taking
(deep agents, middleware-first, no subagents, LangChain PTC, annotation queue for
a trained LLM evaluator).

Short version: **three of those four transfer directly and one of them fixes the
standing objection to having any LLM in this system at all.**

---

## 1. Why "ditch subagents" is right here too

`NOTES_ON_MULTIAGENT_PROPOSAL.md` argued against the five-subgraph design on the
grounds that the specialisation wasn't real — each "specialist" was the same
reasoning against a different query. That's the same conclusion adops reached
from the other direction.

One agent that can write code to fetch what it needs beats five agents that each
own a data source, because the boundary between them was never a boundary. It
was a join.

---

## 2. PTC fixes the red team's actual failure mode

The red team's worst behaviour was **reasoning about absence**. It killed picks
because weather data wasn't present, at a 41% downgrade rate. The fix was prompt
rules plus code-level overrides — effective, but fundamentally a patch on a model
that can't tell "I wasn't given this" from "this is bad."

With PTC the agent writes JavaScript that queries the tools directly:

```ts
const wx = await tools.getWeather({ gameId });
const inj = await tools.getInjuries({ gameId });
const hist = await tools.getLineHistory({ gameId, market: "totals" });

// The distinction the model could never reliably make, made structurally:
const evidence = [];
if (wx.rows.length === 0) {
  // absence — explicitly NOT evidence
} else if (wx.wind_kph > 32) {
  evidence.push(`wind ${wx.wind_kph}kph on a totals pick`);
}
```

`rows.length === 0` is a fact. "No forecast available" in a prompt is a
sentence the model has to interpret, and it interpreted it wrong. Moving that
judgement into code removes the failure mode rather than instructing around it.

Second benefit: intermediate results never enter model context. Today, giving the
agent enough context to judge a pick means stuffing the prompt. With PTC it pulls
what it needs, aggregates in the interpreter, and the model sees only the
computed evidence.

### Hard rule: the PTC allowlist is read-only

From the docs: *"PTC calls currently execute through the interpreter bridge and
do not go through the normal tool calling path. As a result, `interrupt_on`
approval workflows are not enforced per PTC-invoked tool call."*

In a betting application that is a serious constraint, not a footnote. **Nothing
that writes a pick, sizes a bet, moves money, or changes config goes in the PTC
allowlist.** Reads only: odds, injuries, weather, line history, player stats,
pick metadata. If a future tool can mutate anything, it stays on the normal tool
path where approval is enforced.

---

## 3. Middleware is where the safety invariants belong

Today the downgrade-only property is enforced *after* the agent responds:
`review_pick()` inspects the verdict and clamps a promotion back to OK. It works,
and there's a test for it (`test_redteam_cannot_promote`).

As middleware it becomes structural — the agent's output boundary doesn't admit a
promotion in the first place. Same for fail-open on `AIUnavailable`, and same for
the unevidenced-verdict demotion.

The difference matters: a post-hoc check is one refactor away from being
bypassed by a new call site. A middleware wrapper is on every path by
construction. Both keep their tests.

---

## 4. The annotation queue is the important one

This resolves the objection running through every previous note in this repo:
**LLM output has no ground truth to regress against, so prompt changes can't be
measured.**

An annotation queue creates the ground truth.

Concretely: every red-team verdict is logged with its inputs, its evidence, and
its outcome. Weekly, a sample gets labelled by hand. That accumulating labelled
set becomes the eval that makes the next prompt change measurable instead of
vibes.

The cost is small enough to actually happen — roughly 14 games a week, 20–40
picks, so labelling a 30-verdict sample is minutes, not hours. Most eval
programmes die on labelling burden. This one won't.

### The trap: do not label on outcome

The obvious label is "was the KILL correct — did the bet lose?" **That is wrong
and would actively make the system worse.**

A red-team veto is a judgement about *process*, not outcome. A pick can be
correctly flagged and still win; a pick can be wrongly cleared and still win. If
labels are outcome-derived, the trained evaluator learns outcome bias, and it
will punish good process on a 45% loser and reward bad process on a lucky
winner. That's worse than no evaluator.

The label must be: **was the stated reason valid and grounded in the data
available at the time?** Four buckets:

| Label | Meaning |
|---|---|
| `valid` | Reason is real, evidence supports it, correct call |
| `unevidenced` | Reason might be true but the evidence cited doesn't support it |
| `absence` | Downgraded for missing data — the original failure mode |
| `missed` | Should have flagged something and didn't |

CLV already measures outcomes objectively. The annotation queue exists to measure
the *reasoning*, which nothing else can.

---

## 5. What stays in Python

PTC runs QuickJS — JavaScript, 5-second default timeout, 64 MB heap, 256 tool
calls per eval. That's right for querying, filtering, and aggregating. It is not
where statistical work goes.

The `PROP_ENGINE_DESIGN.md` split holds unchanged:

- **Agents assemble and judge** — now with PTC doing the assembly in code
- **Python models produce numbers** — layered prop chain, calibration, holdouts

Nothing in the layered model moves into the interpreter. The temptation will be
there because PTC makes data access easy; resist it, because a projection
computed in QuickJS has no walk-forward, no PIT diagnostic, and no way to detect
that it drifted.

---

## 6. Where it runs

The serving plane is deliberately lean and the red team is a **batch job after
pick generation**, not a request-path dependency. So the agent layer does not
have to live inside the FastAPI service.

Preference: run it as a separate scheduled process, so LangGraph and deepagents
never become a dependency of the thing that has to answer `/health` in 200ms.
The existing fail-open behaviour means a dead agent layer degrades to publishing
unreviewed picks, which is the correct failure.

---

## 7. Sequencing

There's a lesson this repo learned the expensive way last week and it applies
again.

`odds_changes` turned out to hold a single timestamp — no line-movement history
at all, and none recoverable retroactively. **The same is true of annotation
labels.** Every week the system runs without a queue is a week of ground truth
that can never be collected.

So the ordering inverts what you'd expect: start collecting before building.

| When | Do |
|---|---|
| Now → 5 Sept | Nothing. Freeze holds. |
| Week 1 | **Stand up the annotation queue.** Log every red-team verdict with inputs and evidence; add a minimal label UI in the Admin tab. No agent changes. |
| Weeks 1–3 | Label weekly while the machinery is being verified anyway. ~30 verdicts. |
| Week 4 | First measurable baseline: what fraction of verdicts are `valid` vs `absence`. This is a number the system has never had. |
| Weeks 4–6 | Rebuild the red team as a deep agent with PTC and middleware-enforced invariants. Measure against accumulated labels — the first prompt change in this project's history that can be evaluated rather than argued about. |
| Weeks 6+ | Extend PTC to the prop assembly layer once Layers 0–1 exist to assemble for. |

The Week 1 item is roughly a day of work and it's the highest-leverage thing on
this list, purely because of the ordering constraint.

---

## 8. What not to take

- **Deep agents in the request path.** The serving plane stays thin.
- **PTC for anything that writes.** See §2 — approval workflows don't apply.
- **Dynamic subagents / `task()` fan-out.** Nothing here has many independent
  units of work. 14 games a week is a loop, not a fan-out.
- **Migrating existing numeric code into the interpreter.** See §5.
