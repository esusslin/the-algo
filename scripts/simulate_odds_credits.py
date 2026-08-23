"""How many Odds API credits does one real game week cost?

    python scripts/simulate_odds_credits.py
    python scripts/simulate_odds_credits.py --no-props
    python scripts/simulate_odds_credits.py --games 14 --regions-featured 1

Why this exists
---------------
`odds_changes` has no history yet, so the in-season burn has never been observed.
The one time it was, it came to ~52,000 credits against a 20,000 monthly budget — the
cause was a missing throttle, since fixed. Whether the *corrected* schedules fit inside
the plan is still an open question, and the honest way to answer it is arithmetic over
the real tier tables rather than a guess.

`TIERS` is imported from `src.fetchers.odds_api`, not copied. A simulator that drifts
from the code it models is worse than none: it produces a confident wrong number.

Method
------
Walk a clock across one NFL week at the same 5-minute cadence the scheduler ticks at,
and reproduce the poller's actual decisions:

* `soonest` = hours to the nearest un-kicked game, which is what drives every tier's
  interval — so tiers re-tighten for Thursday, then again for Sunday, then Monday.
* a tier fires only when time since its last poll >= its interval for that bucket.
* per-event tiers additionally skip any game more than 72h out.
* cost = markets x regions per call, once per event for per-event tiers.

What it can't tell you
----------------------
Real credit cost comes from the API's `x-requests-last` header, and the client records
that in preference to its own arithmetic. Byes, flexed games and international kickoffs
shift the schedule. Treat the output as a planning estimate, not a bill.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetchers.odds_api import TIERS  # noqa: E402

TICK_MINUTES = 5
"""The scheduler's interval. A tier cannot fire between ticks, so polls quantise here."""


def typical_week(anchor: datetime) -> list[datetime]:
    """One week's kickoffs, in UTC. Thursday night, four Sunday waves, Monday night.

    Times are the usual ET slots converted to UTC during the regular season. The exact
    minutes matter less than the shape: several waves, each pulling `soonest` back down
    and re-tightening every tier.
    """
    monday = anchor - timedelta(days=anchor.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def at(day_offset: int, hour: int, minute: int = 0) -> datetime:
        return monday + timedelta(days=day_offset, hours=hour, minutes=minute)

    return [
        at(3, 0, 15),  # Thu 20:15 ET
        *[at(6, 17, 0)] * 9,  # Sun 13:00 ET — the big wave
        *[at(6, 20, 25)] * 4,  # Sun 16:25 ET
        at(6, 20, 20) + timedelta(hours=4),  # Sun night
        at(7, 0, 15),  # Mon 20:15 ET
    ]


@dataclass
class TierUsage:
    name: str
    calls: int = 0
    credits: int = 0
    fires: int = 0
    per_call: int = 0
    events_touched: int = 0


@dataclass
class Result:
    tiers: dict[str, TierUsage] = field(default_factory=dict)

    @property
    def credits(self) -> int:
        return sum(t.credits for t in self.tiers.values())

    @property
    def calls(self) -> int:
        return sum(t.calls for t in self.tiers.values())


def simulate_week(
    kickoffs: list[datetime],
    *,
    enable_props: bool = True,
    featured_regions: int | None = None,
    start_days_before: int = 7,
) -> Result:
    """Replay one week of polling decisions."""
    result = Result()
    last_poll: dict[str, datetime] = {}

    first, last = min(kickoffs), max(kickoffs)
    clock = first - timedelta(days=start_days_before)
    end = last

    while clock <= end:
        upcoming = [k for k in kickoffs if k > clock]
        if not upcoming:
            break
        soonest = (min(upcoming) - clock).total_seconds() / 3600.0

        for tier in TIERS:
            if tier.name == "props" and not enable_props:
                continue

            interval = tier.interval_minutes(soonest)
            if interval is None:
                continue

            previous = last_poll.get(tier.name)
            if previous is not None and (clock - previous) < timedelta(minutes=interval):
                continue

            usage = result.tiers.setdefault(tier.name, TierUsage(tier.name))
            regions = len(tier.regions)
            if tier.name == "featured" and featured_regions is not None:
                regions = featured_regions
            usage.per_call = len(tier.markets) * regions
            usage.fires += 1

            if tier.per_event:
                # cost scales with event count, and only games inside 72h are polled
                eligible = sum(
                    1 for k in upcoming if (k - clock).total_seconds() / 3600.0 <= 72
                )
                usage.calls += eligible
                usage.events_touched += eligible
                usage.credits += eligible * usage.per_call
            else:
                usage.calls += 1
                usage.credits += usage.per_call

            last_poll[tier.name] = clock

        clock += timedelta(minutes=TICK_MINUTES)

    return result


def report(result: Result, *, budget: int, weeks_per_month: float, label: str) -> None:
    print(f"\n=== {label} ===")
    header = f"{'tier':<10}{'fires':>8}{'calls':>8}{'per call':>10}{'credits':>10}"
    print(header)
    print("-" * len(header))
    for name in ("featured", "period", "props"):
        usage = result.tiers.get(name)
        if usage is None:
            print(f"{name:<10}{'--':>8}{'--':>8}{'--':>10}{'disabled':>10}")
            continue
        print(
            f"{usage.name:<10}{usage.fires:>8}{usage.calls:>8}"
            f"{usage.per_call:>10}{usage.credits:>10,}"
        )

    monthly = result.credits * weeks_per_month
    print("-" * len(header))
    print(f"{'week':<10}{'':>8}{result.calls:>8}{'':>10}{result.credits:>10,}")
    print(f"{'month':<10}{'':>8}{'':>8}{'':>10}{monthly:>10,.0f}   budget {budget:,}")

    pct = monthly / budget * 100 if budget else 0
    verdict = "WITHIN BUDGET" if monthly <= budget else "OVER BUDGET"
    print(f"\n  {verdict} — {pct:.0f}% of plan")
    if monthly > budget:
        over = monthly - budget
        print(f"  short by ~{over:,.0f} credits/month")
        print("  levers: ENABLE_PROPS=false, drop 'eu' from featured regions,")
        print("          trim PROPS_CORE, or widen the props intervals in TIERS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--weeks-per-month", type=float, default=4.3)
    parser.add_argument("--no-props", action="store_true", help="model ENABLE_PROPS=false")
    parser.add_argument(
        "--regions-featured",
        type=int,
        default=None,
        help="override featured region count (2 today: us + eu for Pinnacle)",
    )
    args = parser.parse_args(argv)

    kickoffs = typical_week(datetime(2026, 9, 13, tzinfo=timezone.utc))
    print(f"simulating {len(kickoffs)} games, {TICK_MINUTES}-minute scheduler ticks")
    print("tier schedules imported from src.fetchers.odds_api — not copied")

    baseline = simulate_week(
        kickoffs,
        enable_props=not args.no_props,
        featured_regions=args.regions_featured,
    )
    report(
        baseline,
        budget=args.budget,
        weeks_per_month=args.weeks_per_month,
        label="current configuration",
    )

    if not args.no_props:
        # The obvious lever, shown without needing a second run.
        without = simulate_week(kickoffs, enable_props=False)
        report(
            without,
            budget=args.budget,
            weeks_per_month=args.weeks_per_month,
            label="if ENABLE_PROPS were false",
        )
        saved = (baseline.credits - without.credits) * args.weeks_per_month
        print(f"\n  props cost ~{saved:,.0f} credits/month")
        print("  which is also where the softest prices are — see PROP_ENGINE_DESIGN.md §1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
