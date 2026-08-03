"""History and performance aggregation.

Two views over the same data:

  overall   Every pick the app has ever generated, graded honestly — including
            the losers and the weeks you'd rather forget. This is the system's
            record and it's also your marketing asset, so it must be complete.

  mine      Only the bets a given user opted into, at the price THEY got. This
            differs from overall: you'll miss picks, get worse numbers, and skip
            ones you didn't like. Overall tells you if the model works; yours
            tells you if you're executing it.

Both are filterable by bet type through src.markets.

On reading the numbers: with a few hundred bets, win rate and ROI carry huge
error bars — a 54% shooter and a 48% shooter look identical over 200 bets. That
is why every ROI here ships with a confidence interval and why CLV is reported
first. CLV stabilises far faster than profit.
"""
from __future__ import annotations

import math
from typing import Any

from src.db import query
from src.markets import FILTER_GROUPS, describe_market, matches_filter, side_label
from src.picks.grading import payout_for

SETTLED = ("win", "loss", "push")


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sanely at small n, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _summarise(rows: list[dict]) -> dict[str, Any]:
    settled = [r for r in rows if r.get("result") in SETTLED]
    decided = [r for r in settled if r["result"] != "push"]
    wins = sum(1 for r in decided if r["result"] == "win")

    staked = sum(r.get("stake") or 0 for r in settled)
    profit = sum(r.get("payout") or 0 for r in settled)

    clvs = [r["clv_pct"] for r in rows if r.get("clv_pct") is not None]
    beat_close = sum(1 for c in clvs if c > 0)

    lo, hi = _wilson(wins, len(decided))
    return {
        "picks": len(rows),
        "settled": len(settled),
        "pending": len(rows) - len(settled),
        "wins": wins,
        "losses": len(decided) - wins,
        "pushes": len(settled) - len(decided),
        "win_rate": round(wins / len(decided) * 100, 2) if decided else None,
        "win_rate_ci": [round(lo * 100, 1), round(hi * 100, 1)] if decided else None,
        "units_staked": round(staked, 3),
        "units_profit": round(profit, 3),
        "roi_pct": round(profit / staked * 100, 2) if staked else None,
        # CLV first: it stabilises long before profit does
        "avg_clv_pct": round(sum(clvs) / len(clvs), 3) if clvs else None,
        "beat_close_pct": round(beat_close / len(clvs) * 100, 1) if clvs else None,
        "clv_sample": len(clvs),
    }


def _rows_for_filter(rows: list[dict], filt: str) -> list[dict]:
    if filt in (None, "all"):
        return rows
    return [r for r in rows if matches_filter(r["market_type"], filt)]


def _decorate(rows: list[dict]) -> list[dict]:
    """Attach human labels so the UI doesn't need the taxonomy."""
    out = []
    for r in rows:
        info = describe_market(r["market_type"])
        out.append({
            **r,
            "market_label": info.label,
            "bet_class": info.bet_class,
            "period": info.period,
            "family": info.family,
            "description": side_label(
                r["market_type"], r["side"], r["line"] or 0,
                r.get("home_team", ""), r.get("away_team", ""),
                r.get("player_name", "") or "",
            ),
        })
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def overall_history(filt: str = "all", season: int | None = None,
                    week: int | None = None, limit: int = 500,
                    include_demo: bool = False) -> dict[str, Any]:
    """Every pick the app generated, with filterable aggregates.

    Demo rows are excluded by default and must be opted into explicitly — a
    seeded fake win must never be able to inflate the real track record.
    """
    sql = (
        "SELECT p.*, g.season, g.week, g.home_team, g.away_team, g.kickoff_utc, "
        "       g.home_score, g.away_score, pl.full_name AS player_name, "
        "       p.kelly_units AS stake "
        "FROM picks p "
        "JOIN games g ON g.game_id = p.game_id "
        "LEFT JOIN players pl ON pl.player_id = p.player_id "
        "WHERE 1=1"
    )
    params: list = []
    if not include_demo:
        sql += " AND COALESCE(p.demo,0)=0"
    if season:
        sql += " AND g.season=?"
        params.append(season)
    if week:
        sql += " AND g.week=?"
        params.append(week)
    sql += " ORDER BY g.kickoff_utc DESC, p.edge_pct DESC"

    rows = [dict(r) for r in query(sql, params)]
    for r in rows:
        r["payout"] = payout_for(r["result"], r["stake"] or 0, r["best_price"]) \
            if r["result"] in SETTLED else 0.0

    filtered = _rows_for_filter(rows, filt)
    return {
        "scope": "overall",
        "filter": filt,
        "summary": _summarise(filtered),
        "by_tier": {t: _summarise([r for r in filtered if r["tier"] == t])
                    for t in ("A", "B", "C")},
        "by_filter": {k: _summarise(_rows_for_filter(rows, k))
                      for k in FILTER_GROUPS},
        "rows": _decorate(filtered[:limit]),
    }


def user_history(user_id: int, filt: str = "all", limit: int = 500,
                 include_demo: bool = False) -> dict[str, Any]:
    """Only bets this user opted into, at the price they actually got."""
    sql = (
        "SELECT b.id AS bet_id, b.stake, b.price, b.book, b.placed_at, b.result, "
        "       b.payout, b.clv_pct, b.closing_price, b.note, "
        "       p.pick_id, p.market_type, p.side, p.line, p.tier, p.edge_pct, "
        "       p.headline, p.best_price, p.player_id, "
        "       g.season, g.week, g.home_team, g.away_team, g.kickoff_utc, "
        "       g.home_score, g.away_score, g.status AS game_status, "
        "       pl.full_name AS player_name "
        "FROM user_bets b "
        "JOIN picks p ON p.pick_id = b.pick_id "
        "JOIN games g ON g.game_id = p.game_id "
        "LEFT JOIN players pl ON pl.player_id = p.player_id "
        "WHERE b.user_id=?"
        + ("" if include_demo else " AND COALESCE(b.demo,0)=0")
        + " ORDER BY b.placed_at DESC"
    )
    rows = [dict(r) for r in query(sql, (user_id,))]
    filtered = _rows_for_filter(rows, filt)

    # slippage: did the user get a worse number than the pick recommended?
    slip = [r["price"] - r["best_price"] for r in rows
            if r.get("price") and r.get("best_price")]

    summary = _summarise(filtered)
    summary["avg_slippage_cents"] = round(sum(slip) / len(slip), 1) if slip else None

    return {
        "scope": "mine",
        "filter": filt,
        "summary": summary,
        "by_tier": {t: _summarise([r for r in filtered if r["tier"] == t])
                    for t in ("A", "B", "C")},
        "by_filter": {k: _summarise(_rows_for_filter(rows, k))
                      for k in FILTER_GROUPS},
        "rows": _decorate(filtered[:limit]),
    }


def filter_options() -> list[dict]:
    """Filter chips with live counts, so empty categories can be hidden."""
    keys = [r["market_type"] for r in
            query("SELECT DISTINCT market_type FROM picks")]
    out = []
    for name, grp in FILTER_GROUPS.items():
        n = sum(1 for k in keys if matches_filter(k, name))
        out.append({"key": name, "label": grp["label"], "market_count": n})
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="history")
    p.add_argument("command", choices=["overall", "mine", "filters"])
    p.add_argument("--filter", default="all")
    p.add_argument("--user", type=int, default=1)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "filters":
        for f in filter_options():
            print(f"  {f['key']:<14}{f['label']:<16}{f['market_count']} market types")
    else:
        data = (overall_history(args.filter) if args.command == "overall"
                else user_history(args.user, args.filter))
        s = data["summary"]
        print(f"scope={data['scope']} filter={data['filter']}")
        print(f"  picks {s['picks']}  settled {s['settled']}  pending {s['pending']}")
        print(f"  record {s['wins']}-{s['losses']}-{s['pushes']}"
              f"  win rate {s['win_rate']}%  CI {s['win_rate_ci']}")
        print(f"  staked {s['units_staked']}u  profit {s['units_profit']}u  ROI {s['roi_pct']}%")
        print(f"  CLV avg {s['avg_clv_pct']}%  beat close {s['beat_close_pct']}% "
              f"(n={s['clv_sample']})")
        print("\n  by tier:")
        for t, v in data["by_tier"].items():
            print(f"    {t}: {v['wins']}-{v['losses']}-{v['pushes']} "
                  f"ROI {v['roi_pct']} CLV {v['avg_clv_pct']}")
