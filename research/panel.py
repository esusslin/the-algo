"""The corrected player-week panel — the foundation Layer 0 and props both need.

WHY THIS FILE EXISTS
--------------------
`player_weeks` contains a row only when a player recorded at least one target or
carry. Games where he played and did nothing, and games he missed, are ABSENT
rows — not zero rows.

`research/props.py` builds its history straight from `player_weeks`, so it has
never seen a single zero-involvement game. Measured on 2013-2025, for players
with a real baseline:

    bust rate props.py measures (present rows only) :  6.9%
    bust rate over the true panel, excl. ruled-out  : 19.0%
    understatement                                  : 2.75x

That is almost certainly the root cause of the prop holdout failure. The recorded
symptom was "too much mass in the bottom decile — quiet games are underpredicted,
so every P(over) runs high", which is precisely what understating bust
probability by 2.75x produces.

It is a data-assembly bug, not a modelling limitation, which means it is fixable
without in-season data.

THE TENURE CONSTRAINT
---------------------
Materialising every team-week for every player overcounts badly: a player traded
in week 8 would be scored absent for his old team's weeks 9-18, and a player lost
to IR would be scored absent for the rest of the season. Both are non-events for
prop pricing — no book prices a prop for a player who isn't on the roster.

Restricting the panel to weeks between a player's first and last active week WITH
THAT TEAM removes the artifact. It matters: without it, "absent and not on the
injury report" reads 19.1%; with it, 9.8%.

POINT-IN-TIME
-------------
Every trailing feature is computed from weeks STRICTLY EARLIER than the row's
week. `_trailing()` enforces this; the self-test checks it.
"""
from __future__ import annotations

import logging
from typing import Any

from research.warehouse import connect

log = logging.getLogger(__name__)

# Matches props.BUST_CUT_FRAC. Kept in sync deliberately — if that constant
# moves, the measured bust rate here must move with it or the two disagree
# about what a bust is.
BUST_CUT_FRAC = 0.35

# A player needs some history before "below his norm" means anything. A WR5 with
# a median of 1 target does not have a bust rate, he has noise.
MIN_ACTIVE_WEEKS = 4
MIN_MEDIAN_VOLUME = 3.0

# Trailing window for form features. Four games is a compromise: long enough to
# be stable, short enough to track a role change.
TRAILING_N = 4


PANEL_SQL = """
WITH act AS (
    SELECT player_id, season, team,
           count(*)                   AS n_active,
           median(targets + carries)  AS baseline_vol,
           min(week)                  AS first_wk,
           max(week)                  AS last_wk
    FROM player_weeks
    WHERE season BETWEEN ? AND ?
    GROUP BY 1, 2, 3
    HAVING count(*) >= {min_weeks} AND median(targets + carries) >= {min_vol}
),
team_wk AS (
    SELECT DISTINCT season, week, team FROM team_weeks WHERE season BETWEEN ? AND ?
),
panel AS (
    -- The tenure constraint. See module docstring.
    SELECT a.player_id, a.season, a.team, t.week, a.baseline_vol, a.n_active
    FROM act a
    JOIN team_wk t ON t.season = a.season AND t.team = a.team
    WHERE t.week BETWEEN a.first_wk AND a.last_wk
)
SELECT
    p.player_id, p.season, p.week, p.team, p.baseline_vol,
    -- COALESCE is the whole point of this file: absent means zero, not missing.
    COALESCE(pw.targets, 0)                    AS targets,
    COALESCE(pw.carries, 0)                    AS carries,
    COALESCE(pw.targets, 0) + COALESCE(pw.carries, 0) AS volume,
    COALESCE(pw.target_share, 0.0)             AS target_share,
    CASE WHEN pw.player_id IS NULL THEN 0 ELSE 1 END AS played,
    i.report_status, i.practice_status,
    -- NOT FEATURES. These describe the game that has already been played, so
    -- using them would be flat leakage. Prefixed to make that impossible to
    -- forget; research.availability.FEATURES asserts nothing starting with
    -- `post_` ever reaches a model.
    tw.opponent,
    tw.plays     AS post_team_plays,
    tw.pass_rate AS post_pass_rate
FROM panel p
LEFT JOIN player_weeks pw
       ON pw.player_id = p.player_id AND pw.season = p.season AND pw.week = p.week
LEFT JOIN injuries_hist i
       ON i.gsis_id = p.player_id AND i.season = p.season AND i.week = p.week
LEFT JOIN team_weeks tw
       ON tw.season = p.season AND tw.week = p.week AND tw.team = p.team
ORDER BY p.player_id, p.season, p.week
"""


def build_panel(from_season: int = 2013, to_season: int = 2025) -> list[dict[str, Any]]:
    """Full player-week panel with zero-involvement games materialised.

    2013 is the floor because snap counts start there; earlier seasons can be
    included but Layer 1 features will be null.
    """
    sql = PANEL_SQL.format(min_weeks=MIN_ACTIVE_WEEKS, min_vol=MIN_MEDIAN_VOLUME)
    con = connect(read_only=True)
    try:
        cur = con.execute(sql, [from_season, to_season, from_season, to_season])
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()

    _add_trailing(rows)
    log.info("panel: %d player-weeks, %d seasons", len(rows), to_season - from_season + 1)
    return rows


def _add_trailing(rows: list[dict]) -> None:
    """Attach point-in-time trailing features, in place.

    Rows arrive ordered by (player_id, season, week). For each row, features are
    computed from that player-season's EARLIER weeks only. Never the current week
    — that is the leak this function exists to avoid.
    """
    history: list[float] = []
    key: tuple | None = None

    for r in rows:
        k = (r["player_id"], r["season"])
        if k != key:
            key, history = k, []

        # Computed BEFORE appending the current week. Order is load-bearing.
        if history:
            window = history[-TRAILING_N:]
            r["trail_mean"] = sum(window) / len(window)
            r["trail_max"] = max(window)
            r["trail_zero_rate"] = sum(1 for v in window if v == 0) / len(window)
            r["trail_n"] = len(window)
            r["last_week_vol"] = history[-1]
        else:
            r["trail_mean"] = None
            r["trail_max"] = None
            r["trail_zero_rate"] = None
            r["trail_n"] = 0
            r["last_week_vol"] = None

        r["bust"] = 1 if r["volume"] < BUST_CUT_FRAC * r["baseline_vol"] else 0
        r["ruled_out"] = 1 if (r["report_status"] or "") == "Out" else 0

        history.append(float(r["volume"]))


def summarise(rows: list[dict]) -> dict[str, Any]:
    """The numbers that motivated this file. Recomputed so they can't rot."""
    n = len(rows)
    present = [r for r in rows if r["played"]]
    priceable = [r for r in rows if not r["ruled_out"]]

    def rate(xs: list[dict]) -> float:
        return sum(r["bust"] for r in xs) / len(xs) if xs else 0.0

    return {
        "panel_rows": n,
        "played": len(present),
        "absent": n - len(present),
        "absent_pct": round(100 * (n - len(present)) / max(n, 1), 1),
        "ruled_out": sum(r["ruled_out"] for r in rows),
        "bust_rate_present_only": round(rate(present), 4),
        "bust_rate_full_panel": round(rate(rows), 4),
        "bust_rate_priceable": round(rate(priceable), 4),
        "understatement_factor": round(
            rate(priceable) / rate(present), 2) if rate(present) else None,
    }


def selftest(rows: list[dict] | None = None) -> list[str]:
    """Checks that would each have caught a real bug."""
    rows = rows or build_panel()
    problems: list[str] = []

    # 1. Zero-involvement rows must exist. Their absence is the whole bug.
    zeros = sum(1 for r in rows if r["volume"] == 0)
    if zeros == 0:
        problems.append("no zero-volume rows — the panel is not materialising absences")

    # 2. Point-in-time: a player's first row can have no trailing history.
    firsts = {}
    for r in rows:
        k = (r["player_id"], r["season"])
        if k not in firsts:
            firsts[k] = r
    leaked = [r for r in firsts.values() if r["trail_n"] != 0]
    if leaked:
        problems.append(f"{len(leaked)} first-of-season rows carry trailing history — "
                        f"point-in-time violation")

    # 3. Trailing window can never exceed its cap.
    over = [r for r in rows if r["trail_n"] > TRAILING_N]
    if over:
        problems.append(f"{len(over)} rows have trail_n > {TRAILING_N}")

    # 4. The bust rate over present rows must be materially LOWER than over the
    #    full panel. If they converge, the join has silently stopped working.
    s = summarise(rows)
    if s["bust_rate_present_only"] >= s["bust_rate_priceable"]:
        problems.append(
            f"bust rate present-only ({s['bust_rate_present_only']}) is not below "
            f"priceable ({s['bust_rate_priceable']}) — absences are not being counted")

    # 5. Sanity: absence share should sit in a plausible band. Way outside it
    #    means the tenure constraint or the team join has broken.
    if not 5.0 <= s["absent_pct"] <= 30.0:
        problems.append(f"absent_pct {s['absent_pct']}% outside plausible 5-30% band — "
                        f"check the tenure constraint")

    return problems


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="corrected player-week panel")
    p.add_argument("command", choices=["build", "selftest"], nargs="?", default="build")
    p.add_argument("--from-season", type=int, default=2013)
    p.add_argument("--to-season", type=int, default=2025)
    args = p.parse_args()

    rows = build_panel(args.from_season, args.to_season)
    s = summarise(rows)

    print("\n=== corrected player-week panel ===\n")
    print(f"  rows                      : {s['panel_rows']:,}")
    print(f"  played                    : {s['played']:,}")
    print(f"  absent (zero involvement) : {s['absent']:,}  ({s['absent_pct']}%)")
    print(f"  of which ruled Out        : {s['ruled_out']:,}")
    print()
    print("  --- bust rate, BUST_CUT_FRAC = {:.2f} ---".format(BUST_CUT_FRAC))
    print(f"  present rows only (what props.py sees) : {s['bust_rate_present_only']:.1%}")
    print(f"  full panel                             : {s['bust_rate_full_panel']:.1%}")
    print(f"  priceable (excl. ruled Out)            : {s['bust_rate_priceable']:.1%}")
    print(f"  understatement factor                  : {s['understatement_factor']}x")

    print("\n=== selftest ===")
    problems = selftest(rows)
    if problems:
        for pr in problems:
            print(f"  FAIL  {pr}")
    else:
        print("  all checks passed")
