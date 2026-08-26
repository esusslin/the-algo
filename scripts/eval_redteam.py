"""Golden-set evaluation for the red-team agent.

    python scripts/eval_redteam.py              # run every case
    python scripts/eval_redteam.py --case empty_context
    python scripts/eval_redteam.py --repeat 3   # measure non-determinism

**Why this is a script and not a test.** It calls a real model, so it costs money and
does not return the same answer every time. Putting it in CI would make the build flaky
and the natural response to a flaky build is to disable it. Run it deliberately: before a
prompt change, and after.

**What it measures.** Four numbers, in descending order of how much they matter:

1. **False-KILL rate on clean and empty contexts.** The agent may only downgrade, so its
   entire risk is suppressing good bets. A KILL on a game with nothing wrong is the most
   expensive error available to it. Target: zero.
2. **Absence handling.** The single documented production failure was a 41% downgrade
   rate when weather data was merely *missing* — "no value" read as "bad value". Cases
   marked `absence=True` exist only to catch that regressing.
3. **Recall on genuine disqualifiers.** A starting quarterback ruled Out should not
   return OK. Missing these is cheap (you place a bet you shouldn't), so it is third.
4. **Determinism.** `--repeat` runs each case N times. If verdicts vary run to run, the
   downstream numbers are sampling noise and comparing two prompts is meaningless.

**What "expected" means here.** These are my judgements about what a careful analyst
would say, not ground truth. `OK` cases are the confident ones — nothing in the context
disqualifies the bet. `FLAG`/`KILL` cases are directional: the point is that the agent
notices *something*, so both count as a pass where noted.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Run as a file, so the repo root isn't on the path. Same shim as
# scripts/simulate_odds_credits.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Case:
    name: str
    why: str
    ctx: dict[str, Any]
    pick: dict[str, Any] = field(default_factory=dict)
    expect: tuple[str, ...] = ("OK",)
    absence: bool = False
    """True when the case exists to prove missing data is not treated as evidence."""


def _pick(**kw: Any) -> dict[str, Any]:
    base = {"pick_id": 1, "game_id": "G1", "market_type": "spreads", "side": "home",
            "line": -3.0, "tier": "A", "headline": "Home -3.0"}
    return {**base, **kw}


def _ctx(**kw: Any) -> dict[str, Any]:
    base = {
        "matchup": "SF at SEA", "season": 2026, "week": 5, "season_type": "REG",
        "kickoff_utc": "2026-10-11T20:05:00", "roof": "outdoors", "surface": "fieldturf",
        "market_spread": -3.0, "market_total": 44.5,
        "home_rest_days": 7, "away_rest_days": 7, "divisional": True,
        "injuries": [], "inactives": [], "weather": None, "recent_line_moves": [],
    }
    return {**base, **kw}


CASES: list[Case] = [
    # ---- the ones that must return OK -------------------------------------------
    Case(
        "clean_game",
        "Nothing is wrong. Full context present, no injuries of note, benign weather. "
        "A downgrade here is pure cost.",
        _ctx(
            injuries=[{"player_name": "Backup Guard", "team": "SEA",
                       "game_status": "Questionable", "practice_status": "Full",
                       "body_part": "Ankle"}],
            weather={"temp_c": 16.0, "wind_kph": 8.0, "wind_gust_kph": 12.0,
                     "precip_mm": 0.0, "snow_mm": 0.0},
        ),
    ),
    Case(
        "empty_context",
        "**The regression case.** No injuries collected, no forecast, no line history. "
        "That is ignorance, not evidence, and it must not downgrade anything. This is the "
        "exact shape of the 41% false-downgrade incident.",
        _ctx(),
        absence=True,
    ),
    Case(
        "injuries_absent_weather_present",
        "Weather is fine and the injury table is empty. A model reaching for something to "
        "object to will cite the empty injury report; it must not.",
        _ctx(weather={"temp_c": 14.0, "wind_kph": 6.0, "wind_gust_kph": 9.0,
                      "precip_mm": 0.0, "snow_mm": 0.0}),
        absence=True,
    ),
    Case(
        "irrelevant_player_out",
        "A third-string safety is Out. True, stated, and irrelevant to a full-game "
        "spread. Noticing a fact is not the same as it mattering.",
        _ctx(injuries=[{"player_name": "Third String Safety", "team": "SEA",
                        "game_status": "Out", "practice_status": "DNP",
                        "body_part": "Hamstring"}]),
    ),
    Case(
        "line_moved_toward_us",
        "The market moved in our favour. Not a warning.",
        _ctx(recent_line_moves=[
            {"market_type": "spreads", "side": "home", "line": -2.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-09T12:00:00"},
            {"market_type": "spreads", "side": "home", "line": -3.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-11T12:00:00"},
        ]),
    ),

    # ---- the ones that should draw an objection ---------------------------------
    Case(
        "starting_qb_out",
        "The starting quarterback is Out and it is a bet on his team's spread. A "
        "concrete, checkable disqualifier stated plainly in the context.",
        _ctx(injuries=[{"player_name": "Starting Quarterback", "team": "SEA",
                        "game_status": "Out", "practice_status": "DNP",
                        "body_part": "Shoulder"}]),
        expect=("KILL", "FLAG"),
    ),
    Case(
        "high_wind_on_a_total",
        "35 km/h gusts on an outdoor total. Kicking and deep passing degrade "
        "non-linearly above roughly 24 km/h — a real reason to doubt an over.",
        _ctx(weather={"temp_c": 4.0, "wind_kph": 28.0, "wind_gust_kph": 46.0,
                      "precip_mm": 2.0, "snow_mm": 0.0}),
        pick=_pick(market_type="totals", side="over", line=44.5, headline="Over 44.5"),
        expect=("FLAG", "KILL"),
    ),
    Case(
        "week_18_nothing_to_play_for",
        "Week 18, regular season. Teams rest starters when seeding is settled — the "
        "classic situation a feature matrix cannot see.",
        _ctx(week=18, season_type="REG",
             inactives=[{"player_name": "Starting Quarterback", "team": "SEA"},
                        {"player_name": "Starting Running Back", "team": "SEA"}]),
        expect=("KILL", "FLAG"),
    ),
    Case(
        "sharp_line_moved_hard_against_us",
        "Pinnacle moved three points away from our side in two days. Either we are early "
        "or we are wrong, and the agent should say so.",
        _ctx(recent_line_moves=[
            {"market_type": "spreads", "side": "home", "line": -3.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-09T12:00:00"},
            {"market_type": "spreads", "side": "home", "line": 0.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-11T12:00:00"},
        ]),
        # Pinned to FLAG exactly, and both neighbours are failures.
        #
        # OK was tolerated until 2026-08-25, when the eval showed the agent returning
        # it three times out of three while citing missing injury data — it never
        # engaged with the move at all. The cause was the prompt rendering line and
        # price joined by an arrow, no timestamps, newest-first.
        #
        # Once the section was legible the agent swung the other way and returned KILL
        # on six of six samples, which unpublishes the pick. `_movement_only` now caps
        # that at FLAG in code. So OK means the agent stopped seeing movement, and KILL
        # means the cap has broken — the two failures this case exists to separate.
        expect=("FLAG",),
    ),

    # ---- and the opposite failure, which the fix above could easily cause -------
    Case(
        "minor_move_against_us",
        "Half a point, the direction we don't like. Spreads do this constantly. "
        "Teaching the agent to see line movement is only safe if it also learns that "
        "most movement is noise — otherwise the fix trades one blind spot for a "
        "downgrade on every pick.",
        _ctx(recent_line_moves=[
            {"market_type": "spreads", "side": "home", "line": -3.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-09T12:00:00"},
            {"market_type": "spreads", "side": "home", "line": -2.5, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-11T12:00:00"},
        ]),
    ),
    Case(
        "books_disagree_but_nothing_moved",
        "Pinnacle steady at -3.0 while a soft book sits at -6.0. That is a shopping "
        "opportunity, not a warning, and reading it as a three-point move would kill a "
        "pick over an artefact of how the rows happen to be ordered.",
        _ctx(recent_line_moves=[
            {"market_type": "spreads", "side": "home", "line": -3.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-09T12:00:00"},
            {"market_type": "spreads", "side": "home", "line": -6.0, "price": -110,
             "book": "softbook", "observed_at": "2026-10-10T12:00:00"},
            {"market_type": "spreads", "side": "home", "line": -3.0, "price": -110,
             "book": "pinnacle", "observed_at": "2026-10-11T12:00:00"},
        ]),
    ),
]


def run(cases: list[Case], repeat: int = 1, verbose: bool = False) -> int:
    from src.ai.redteam import review_pick

    results: dict[str, list[dict]] = {}
    for case in cases:
        pick = case.pick or _pick()
        results[case.name] = [review_pick(pick, case.ctx) for _ in range(repeat)]

    # ---- report ----
    width = max(len(c.name) for c in cases)
    print(f"\n{'case':<{width}}  {'expected':<12} {'got':<24} pass")
    print("-" * (width + 46))

    passed = 0
    false_kills = 0
    absence_failures = 0
    unstable: list[str] = []
    not_reviewed = 0

    for case in cases:
        # A result whose source isn't "model" is not a verdict. The agent fails
        # open, so a broken call returns OK — and counting that as agreement
        # would let a completely dead AI layer score 5/5 on the OK cases. Judge
        # only the runs where the model actually answered.
        answered = [r for r in results[case.name] if r.get("source") == "model"]
        skipped = len(results[case.name]) - len(answered)
        not_reviewed += skipped

        verdicts = [r["verdict"] for r in answered]
        counts = Counter(verdicts)
        # No answer at all is a failure, not a pass by default.
        ok = bool(verdicts) and all(v in case.expect for v in verdicts)
        passed += ok

        if "OK" in case.expect and any(v == "KILL" for v in verdicts):
            false_kills += 1
        if case.absence and any(v != "OK" for v in verdicts):
            absence_failures += 1
        if len(counts) > 1:
            unstable.append(case.name)

        got = ", ".join(f"{v}x{n}" if n > 1 else v for v, n in counts.items())
        if skipped:
            got = (got + ", " if got else "") + f"no answer x{skipped}"
        print(f"{case.name:<{width}}  {'/'.join(case.expect):<12} {got:<24} "
              f"{'PASS' if ok else 'FAIL'}")

        if not ok or verbose:
            for r in results[case.name][:1]:
                if r.get("reason"):
                    print(f"{'':<{width}}    reason:   {r['reason'][:90]}")
                if r.get("evidence"):
                    print(f"{'':<{width}}    evidence: {r['evidence'][:90]}")
                if r.get("source") != "model":
                    print(f"{'':<{width}}    source:   {r.get('source')}")

    print(f"\n{passed}/{len(cases)} cases as expected")
    print(f"false KILLs on should-be-OK cases : {false_kills}   (target 0)")
    print(f"absence treated as evidence       : {absence_failures}   (target 0)")
    if not_reviewed:
        print(f"calls that never reached the model: {not_reviewed}   (target 0) "
              f"— see the warnings above; these are infrastructure, not judgement")
    if repeat > 1:
        print(f"unstable across {repeat} runs           : "
              f"{len(unstable)}{' — ' + ', '.join(unstable) if unstable else ''}")
        if unstable:
            print("  verdicts that vary run to run make prompt comparisons meaningless")

    # Absence handling and false KILLs are the failures worth a non-zero exit; missing a
    # genuine disqualifier is cheaper and shouldn't block anyone. A call that never
    # reached the model also exits non-zero — not because the agent judged wrongly, but
    # because the run produced no evidence about the agent at all.
    return 1 if (false_kills or absence_failures or not_reviewed) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_redteam")
    parser.add_argument("--case", default=None, help="run one case by name")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each case N times to measure determinism")
    parser.add_argument("--verbose", action="store_true",
                        help="print the model's reason and evidence for every case")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for c in CASES:
            print(f"{c.name:<34} expect {'/'.join(c.expect)}")
            print(f"  {c.why}\n")
        return 0

    cases = [c for c in CASES if c.name == args.case] if args.case else CASES
    if not cases:
        print(f"no case named {args.case!r}", file=sys.stderr)
        return 1

    from src.ai.client import configured
    from src.config import settings

    if not configured():
        print("ANTHROPIC_API_KEY not set — nothing to evaluate", file=sys.stderr)
        return 1
    if not settings.ENABLE_AI_REDTEAM:
        print("ENABLE_AI_REDTEAM is false; review_pick will short-circuit to OK and "
              "every case will look like it passed. Enable it or this proves nothing.",
              file=sys.stderr)
        return 1

    return run(cases, repeat=args.repeat, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
