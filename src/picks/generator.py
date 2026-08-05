"""Pick generation: opportunities -> sized, tiered, publishable picks.

Tiering exists because nobody can act on 400 edge percentages on a phone. A/B/C
collapses edge, market depth, anchor quality and cross-book agreement into one
signal a human can use at a glance.

Publishing is gated separately from generation. Model-sourced picks are written
and graded from day one but stay invisible to subscribers until
PUBLISH_MODEL_PICKS flips — so models accumulate real out-of-sample CLV on live
data before anyone bets them.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.config import settings
from src.db import db, insert_row, query, utcnow
from src.market.shop import find_opportunities
from src.picks.kelly import size_slate

log = logging.getLogger(__name__)

# Tier thresholds. A-tier is deliberately hard to reach: a handful per week that
# you'd actually be annoyed to miss.
TIER_RULES = {
    "A": dict(min_edge=5.0, min_books=15, require_sharp=True, max_dispersion=0.03),
    "B": dict(min_edge=3.0, min_books=10, require_sharp=False, max_dispersion=0.05),
    "C": dict(min_edge=2.0, min_books=8, require_sharp=False, max_dispersion=1.0),
}

# A 3% edge on a WR4 reception prop is not the same asset as 3% on a main
# spread. Props and derivatives are noisier, carry lower limits, and are priced
# by fewer books — so they must clear a higher bar for the same tier. Without
# this, thin markets dominate the top of the feed purely because thin markets
# produce bigger edge numbers.
CLASS_MULTIPLIERS = {
    "game": 1.0,     # moneyline, spread, total — the deepest markets
    "period": 1.3,   # halves and quarters
    "team": 1.3,     # team totals
    "alt": 1.5,      # alternate ladders
    "prop": 1.6,     # player props
}

# Book counts are naturally lower in these markets, so requiring 15 books on a
# prop would mean no prop ever reaches A. Scale the requirement down instead.
CLASS_BOOK_SCALE = {
    "game": 1.0, "period": 0.7, "team": 0.7, "alt": 0.6, "prop": 0.55,
}


def thresholds_for(market_type: str, tier: str) -> dict:
    """Tier rule adjusted for what class of market this is."""
    from src.markets import describe_market

    cls = describe_market(market_type).bet_class
    r = TIER_RULES[tier]
    return {
        "min_edge": r["min_edge"] * CLASS_MULTIPLIERS.get(cls, 1.0),
        "min_books": max(4, round(r["min_books"] * CLASS_BOOK_SCALE.get(cls, 1.0))),
        # Sharp anchor is rarely available on props — Pinnacle quotes far fewer
        # of them — so requiring it would make A-tier props impossible.
        "require_sharp": r["require_sharp"] and cls == "game",
        "max_dispersion": r["max_dispersion"] * (1.5 if cls == "prop" else 1.0),
        "bet_class": cls,
    }


def assign_tier(opp: dict) -> str | None:
    """A/B/C, or None if it doesn't clear the bar for its market class."""
    for tier in ("A", "B", "C"):
        r = thresholds_for(opp["market_type"], tier)
        if opp["edge_pct"] < r["min_edge"]:
            continue
        if opp["book_count"] < r["min_books"]:
            continue
        if r["require_sharp"] and opp.get("anchor") != "sharp":
            continue
        if (opp.get("dispersion") or 0) > r["max_dispersion"]:
            continue
        return tier
    return None


def visibility_for(tier: str, source: str) -> tuple[int, str]:
    """(published, visibility) — the dark-launch gate.

    Model picks are computed, stored, graded and CLV-tracked but not shown to
    subscribers until they've earned it. Market-engine picks publish from day
    one because they involve no model risk: they're arithmetic on live prices.
    """
    if source == "model" and not settings.PUBLISH_MODEL_PICKS:
        return 0, "admin"
    if tier == "C":
        return 1, "premium"
    return 1, "all"


def blended_probability(opp: dict) -> dict:
    """The single number shown to users, plus a record of what produced it.

    CONTRACT: `blended_prob` is always the output of the full current pipeline.
    Today that is the market fair price alone. As components land — model
    probabilities, then a per-market-class blend weight — they compose here and
    nowhere else, so the displayed figure stays correct without touching the UI.

    Adding a component means: compute it, add it to `components`, fold it into
    `prob`, and extend `source`. Do not blend anywhere else.
    """
    market_prob = opp.get("fair_prob") or 0.5
    components = {
        "market_fair": round(market_prob, 4),
        "anchor": opp.get("anchor", "consensus"),
        "book_count": opp.get("book_count"),
        "dispersion": round(opp.get("dispersion") or 0.0, 4),
    }
    prob, source = market_prob, "market"

    model_prob = opp.get("model_prob")
    if model_prob is not None and settings.PUBLISH_MODEL_PICKS:
        # Blend in log-odds space, weight per market class. w stays small for
        # full-game markets until live CLV earns more — see IN_SEASON_LEARNING.md
        import math
        w = opp.get("blend_weight", 0.0)
        if w > 0:
            def logit(p: float) -> float:
                p = min(max(p, 1e-6), 1 - 1e-6)
                return math.log(p / (1 - p))
            z = w * logit(model_prob) + (1 - w) * logit(market_prob)
            prob = 1 / (1 + math.exp(-z))
            components["model"] = round(model_prob, 4)
            components["blend_weight"] = round(w, 3)
            source = "market+model"

    return {"prob": prob, "source": source, "components": components,
            "model_prob": model_prob}


def describe(opp: dict) -> tuple[str, str]:
    """Fallback headline/detail. Replaced by the LLM narrator (agent A3) later."""
    market = opp["market_type"].replace("_", " ")
    line = opp["line"]
    side = opp["side"]
    if opp["market_type"].startswith(("totals", "player_")):
        label = f"{side.title()} {abs(line)}"
    elif opp["market_type"].startswith("spreads"):
        label = f"{side.title()} {line:+g}"
    else:
        label = side.title()

    headline = (f"{label} {market} at {opp['best_price']:+d} "
                f"({opp['best_book']}) — {opp['edge_pct']:.1f}% edge")
    detail = (
        f"Fair value {opp['fair_prob']*100:.1f}% from "
        f"{'sharp book' if opp['anchor'] == 'sharp' else 'consensus'} "
        f"across {opp['book_count']} books; best available price implies "
        f"{100/(1+abs(opp['best_price'])/100 if opp['best_price']>0 else 1+100/abs(opp['best_price'])):.1f}%. "
        f"Cross-book disagreement {opp.get('dispersion', 0)*100:.1f}%."
    )
    return headline, detail


def generate(source: str = "market_engine",
             bankroll: float = 1.0,
             min_edge: float | None = None,
             dry_run: bool = False) -> dict[str, Any]:
    """Full pass: find edges, size them, tier them, write picks."""
    # MIN_EDGE_PCT is a global FLOOR; tiers subdivide above it. Using min() here
    # would let C-tier publish below the configured floor, making the setting a
    # lie. Start the floor high and lower it as CLV validates.
    if min_edge is None:
        min_edge = max(TIER_RULES["C"]["min_edge"], settings.MIN_EDGE_PCT)

    opps = find_opportunities(min_edge=min_edge)
    if not opps:
        log.info("no opportunities at >= %.1f%% edge", min_edge)
        return {"found": 0, "tiered": 0, "written": 0, "by_tier": {}}

    tiered = []
    for o in opps:
        tier = assign_tier(o)
        if tier:
            tiered.append({**o, "tier": tier})

    sized = size_slate(tiered, bankroll=bankroll)

    by_tier: dict[str, int] = {}
    written = 0
    now = utcnow()

    if not dry_run:
        with db() as conn:
            for p in sized:
                # don't re-publish the same market within the same pass
                dup = conn.execute(
                    "SELECT pick_id FROM picks WHERE game_id=? AND market_type=? "
                    "AND player_id=? AND side=? AND line=? AND result='pending'",
                    (p["game_id"], p["market_type"], p["player_id"], p["side"], p["line"]),
                ).fetchone()
                if dup:
                    continue
                published, vis = visibility_for(p["tier"], source)
                headline, detail = describe(p)
                bp = blended_probability(p)
                insert_row(conn, "picks", {
                    "game_id": p["game_id"], "market_type": p["market_type"],
                    "player_id": p["player_id"], "side": p["side"], "line": p["line"],
                    "source": source,
                    "best_book": p["best_book"], "best_price": p["best_price"],
                    "fair_prob": p["fair_prob"], "blended_prob": bp["prob"],
                    "model_prob": bp["model_prob"],
                    "prob_source": bp["source"],
                    "prob_components": json.dumps(bp["components"]),
                    "edge_pct": p["edge_pct"],
                    "kelly_units": round(p["stake"], 6),
                    "tier": p["tier"],
                    "ai_verdict": "OK",       # replaced when the red-team agent lands
                    "ai_reason": None,
                    "headline": headline, "detail": detail,
                    "published": published, "visibility": vis,
                    "created_at": now,
                    "published_at": now if published else None,
                    "result": "pending",
                })
                written += 1
                by_tier[p["tier"]] = by_tier.get(p["tier"], 0) + 1
    else:
        for p in sized:
            by_tier[p["tier"]] = by_tier.get(p["tier"], 0) + 1

    return {
        "found": len(opps),
        "tiered": len(sized),
        "written": written,
        "by_tier": by_tier,
        "total_exposure": round(sum(p["stake"] for p in sized), 4),
        "picks": sized,
    }


def current_slate(include_unpublished: bool = False) -> list[dict]:
    sql = ("SELECT * FROM picks WHERE result='pending'")
    if not include_unpublished:
        sql += " AND published=1"
    sql += " ORDER BY CASE tier WHEN 'A' THEN 1 WHEN 'B' THEN 2 ELSE 3 END, edge_pct DESC"
    return [dict(r) for r in query(sql)]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="pick generation")
    p.add_argument("command", choices=["run", "dry", "slate"])
    p.add_argument("--min-edge", type=float, default=None)
    p.add_argument("--bankroll", type=float, default=1.0)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "slate":
        rows = current_slate(include_unpublished=True)
        print(f"{len(rows)} pending picks\n")
        for r in rows:
            flag = "" if r["published"] else "  [DARK]"
            print(f"  [{r['tier']}] {r['headline']}{flag}")
            print(f"        stake={r['kelly_units']:.4f}u  {r['detail'][:100]}")
    else:
        res = generate(min_edge=args.min_edge, bankroll=args.bankroll,
                       dry_run=(args.command == "dry"))
        print(f"opportunities found : {res['found']}")
        print(f"cleared tiering     : {res['tiered']}")
        print(f"written             : {res['written']}")
        print(f"by tier             : {res['by_tier']}")
        print(f"total exposure      : {res.get('total_exposure', 0):.4f} of bankroll")
        for p in res.get("picks", [])[:15]:
            print(f"  [{p['tier']}] {p['market_type']:<10}{p['side']:<6}{p['line']:>7.1f} "
                  f"{p['best_book']:<14}{p['best_price']:>+6d} edge={p['edge_pct']:>5.2f}% "
                  f"stake={p['stake']:.4f}u")
