"""Fair-price consensus: turn a wall of book prices into one honest probability.

Method
------
1. Pair each book's two-sided market (over/under, home/away).
2. Devig EACH BOOK SEPARATELY. Devigging a mix of books is meaningless — you'd
   be removing an average margin that no book actually charges.
3. Aggregate the per-book fair probabilities into a consensus, weighting sharp
   books far more heavily. Pinnacle moves first and is right more often; a
   consensus that treats it equally with eight recreational books throws away
   the signal you're paying for.
4. Record cross-book dispersion. High dispersion means the market disagrees with
   itself, which is both a warning (your fair price is uncertain) and an
   opportunity (disagreement is where mispricing lives).

Spread sign handling
--------------------
Totals and props pair on the SAME line (Over 48.5 / Under 48.5). Spreads pair on
OPPOSITE lines (home -2.5 / away +2.5). Getting this wrong silently pairs the
wrong outcomes and produces fair prices that look plausible and are wrong — so
pairing is explicit rather than inferred.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from typing import Iterable, Sequence

from src.config import settings
from src.db import db, query, upsert_rows, utcnow
from src.market.devig import american_to_prob, devig, edge_pct

log = logging.getLogger(__name__)

# Markets where the two sides carry opposite point values.
SPREAD_LIKE = {"spreads", "spreads_h1", "spreads_h2", "spreads_q1", "spreads_q2",
               "spreads_q3", "spreads_q4", "alternate_spreads"}
# Markets with no line at all.
MONEYLINE_LIKE = {"h2h", "h2h_h1", "h2h_h2", "h2h_3_way"}

SIDE_PAIRS = [("home", "away"), ("over", "under"), ("yes", "no")]

# Relative trust. Pinnacle is the anchor; recreational books mostly follow it.
BOOK_WEIGHTS = {
    "pinnacle": 5.0,
    "betonlineag": 2.0,
    "lowvig": 2.0,
    "bookmaker": 2.0,
    "circasports": 3.0,
}
DEFAULT_WEIGHT = 1.0


def _pair_line(market_type: str, side: str, line: float) -> float:
    """Canonical grouping key for a two-sided market.

    For spreads the home line is canonical, so away +2.5 maps to -2.5 and pairs
    with home -2.5. For everything else the line is already shared.
    """
    if market_type in MONEYLINE_LIKE:
        return 0.0
    if market_type in SPREAD_LIKE:
        return line if side == "home" else -line
    return line


def _opposite(side: str) -> str | None:
    for a, b in SIDE_PAIRS:
        if side == a:
            return b
        if side == b:
            return a
    return None


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median — robust to a single stale book, unlike a weighted mean."""
    if not values:
        raise ValueError("no values")
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return v
    return pairs[-1][0]


def build_fair_prices(method: str | None = None,
                      game_ids: Iterable[str] | None = None) -> int:
    """Recompute `fair_prices` from `odds_current`. Returns rows written."""
    method = method or settings.DEVIG_METHOD
    sharp = set(settings.SHARP_BOOKS)

    sql = ("SELECT game_id, market_type, player_id, side, line, book, price "
           "FROM odds_current")
    params: list = []
    if game_ids:
        ids = list(game_ids)
        sql += f" WHERE game_id IN ({','.join('?' * len(ids))})"
        params = ids
    rows = query(sql, params)
    if not rows:
        return 0

    # group: (game, market, player, canonical_line, book) -> {side: price}
    grouped: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["game_id"], r["market_type"], r["player_id"],
               _pair_line(r["market_type"], r["side"], r["line"]), r["book"])
        grouped[key][r["side"]] = r["price"]

    # per (game, market, player, line, side) -> list of (fair_prob, weight, is_sharp)
    agg: dict[tuple, list[tuple[float, float, bool]]] = defaultdict(list)
    unpaired = 0

    for (game_id, mkt, pid, cline, book), sides in grouped.items():
        if len(sides) < 2:
            unpaired += 1
            continue
        pair = next(((a, b) for a, b in SIDE_PAIRS if a in sides and b in sides), None)
        if not pair:
            unpaired += 1
            continue
        a, b = pair
        try:
            fair = devig([american_to_prob(sides[a]), american_to_prob(sides[b])],
                         method=method)
        except (ValueError, ZeroDivisionError):
            continue
        w = BOOK_WEIGHTS.get(book, DEFAULT_WEIGHT)
        is_sharp = book in sharp
        agg[(game_id, mkt, pid, cline, a)].append((fair[0], w, is_sharp))
        agg[(game_id, mkt, pid, cline, b)].append((fair[1], w, is_sharp))

    out: list[dict] = []
    now = utcnow()
    for (game_id, mkt, pid, cline, side), entries in agg.items():
        probs = [e[0] for e in entries]
        weights = [e[1] for e in entries]
        sharp_probs = [e[0] for e in entries if e[2]]
        # store against the side's own line (away spread keeps its +2.5)
        stored_line = -cline if (mkt in SPREAD_LIKE and side == "away") else cline
        out.append({
            "game_id": game_id,
            "market_type": mkt,
            "player_id": pid,
            "side": side,
            "line": stored_line,
            "fair_prob": _weighted_median(probs, weights),
            "method": method,
            "sharp_prob": (sum(sharp_probs) / len(sharp_probs)) if sharp_probs else None,
            "book_count": len(entries),
            "dispersion": statistics.pstdev(probs) if len(probs) > 1 else 0.0,
            "computed_at": now,
        })

    with db() as conn:
        upsert_rows(conn, "fair_prices", out,
                    key_cols=["game_id", "market_type", "player_id", "side", "line"])

    if unpaired:
        log.info("fair_prices: %d rows (%d one-sided book quotes skipped)",
                 len(out), unpaired)
    return len(out)


def sharp_soft_delta(game_id: str | None = None) -> list[dict]:
    """Markets where the sharp book disagrees materially with the consensus.

    This is a model-free signal: when Pinnacle prices something several points
    off what everyone else shows, the soft books are usually the ones that are
    wrong. Requires no prediction of the game whatsoever.
    """
    sql = ("SELECT * FROM fair_prices WHERE sharp_prob IS NOT NULL "
           "AND book_count >= 3")
    params: list = []
    if game_id:
        sql += " AND game_id=?"
        params = [game_id]
    out = []
    for r in query(sql, params):
        delta = (r["sharp_prob"] or 0) - (r["fair_prob"] or 0)
        if abs(delta) >= 0.02:
            out.append({
                "game_id": r["game_id"], "market_type": r["market_type"],
                "player_id": r["player_id"], "side": r["side"], "line": r["line"],
                "sharp_prob": r["sharp_prob"], "consensus_prob": r["fair_prob"],
                "delta": delta, "book_count": r["book_count"],
                "dispersion": r["dispersion"],
            })
    return sorted(out, key=lambda x: -abs(x["delta"]))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="fair price consensus")
    p.add_argument("command", choices=["build", "sharp"])
    p.add_argument("--method", default=None)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "build":
        print(f"{build_fair_prices(method=args.method)} fair prices computed")
    else:
        rows = sharp_soft_delta()
        print(f"{len(rows)} sharp/soft disagreements >= 2%")
        for r in rows[:25]:
            print(f"  {r['game_id']:<20} {r['market_type']:<20} {r['side']:<6} "
                  f"{r['line']:>6}  sharp={r['sharp_prob']:.3f} "
                  f"cons={r['consensus_prob']:.3f} delta={r['delta']:+.3f}")
