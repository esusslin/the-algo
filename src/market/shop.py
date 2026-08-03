"""Line shopping — the closest thing to free money in this whole system.

Getting -105 instead of -115 on a coin flip moves your breakeven from 53.5% to
51.2%. That 2.3 points is larger than most model edges, and it requires no
prediction at all. So this module runs before, and independently of, any model.

Two model-free edge sources live here:

  best-price edge   A soft book's price beats the sharp-anchored fair price.
  consistency       A single book's own numbers contradict each other (its
                    player props don't reconcile with its team total). When a
                    book disagrees with itself, one of the two is wrong.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from src.config import settings
from src.db import query
from src.market.devig import american_to_decimal, edge_pct

log = logging.getLogger(__name__)


def best_prices(game_id: str | None = None) -> dict[tuple, tuple[str, int]]:
    """(game, market, player, side, line) -> (best_book, best_price).

    "Best" means highest payout: for american odds that is the numerically
    largest value (+150 beats +120 beats -105 beats -130).
    """
    sql = ("SELECT game_id, market_type, player_id, side, line, book, price "
           "FROM odds_current")
    params: list = []
    if game_id:
        sql += " WHERE game_id=?"
        params = [game_id]

    best: dict[tuple, tuple[str, int]] = {}
    for r in query(sql, params):
        key = (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"])
        cur = best.get(key)
        if cur is None or r["price"] > cur[1]:
            best[key] = (r["book"], r["price"])
    return best


def main_lines(game_id: str | None = None) -> set[tuple]:
    """The consensus line per (game, market, player) — where the books agree.

    Books post different numbers: DraftKings at -3.5, FanDuel at -3. Each number
    is its own market, so a minority line may have only 3-4 books quoting it.
    A "consensus" built from four books — one of which is the book you'd be
    betting into — is not a fair price, and comparing against it manufactures
    edges that do not exist.

    So: identify the modal line (most books quoting it) and evaluate only there.
    That is also how a sharp bettor compares prices — at the same number.
    """
    sql = ("SELECT game_id, market_type, player_id, line, COUNT(DISTINCT book) AS n "
           "FROM odds_current GROUP BY game_id, market_type, player_id, line")
    params: list = []
    if game_id:
        sql = sql.replace("FROM odds_current", "FROM odds_current WHERE game_id=?")
        params = [game_id]

    best_by_market: dict[tuple, tuple[float, int]] = {}
    for r in query(sql, params):
        # spreads: home -3.5 and away +3.5 are the same market, so key on |line|
        key = (r["game_id"], r["market_type"], r["player_id"], abs(r["line"]))
        cur = best_by_market.get(key[:3])
        if cur is None or r["n"] > cur[1]:
            best_by_market[key[:3]] = (abs(r["line"]), r["n"])
    return {(g, m, p, ln) for (g, m, p), (ln, _) in best_by_market.items()}


def find_opportunities(min_edge: float | None = None,
                       game_id: str | None = None,
                       use_sharp_anchor: bool = True,
                       min_books: int = 8,
                       main_line_only: bool = True,
                       bettable_only: bool = True) -> list[dict]:
    """Every market where the best available price beats fair value.

    `use_sharp_anchor=True` prices against Pinnacle where available, falling
    back to weighted consensus. That is deliberate: the consensus includes the
    very soft book you are betting into, which drags fair value toward the
    mispricing and understates your edge.

    `min_books` and `main_line_only` exist because thin markets manufacture
    fake edges — see main_lines(). `bettable_only` filters to books you can
    actually get down at; an edge at an offshore book you have no account with
    is not an edge.
    """
    min_edge = settings.MIN_EDGE_PCT if min_edge is None else min_edge
    best = best_prices(game_id)
    mains = main_lines(game_id) if main_line_only else None
    bettable = set(settings.BETTABLE_BOOKS) if bettable_only else None

    sql = ("SELECT game_id, market_type, player_id, side, line, fair_prob, "
           "sharp_prob, book_count, dispersion FROM fair_prices")
    params: list = []
    if game_id:
        sql += " WHERE game_id=?"
        params = [game_id]

    out: list[dict] = []
    for r in query(sql, params):
        if r["book_count"] < min_books:
            continue
        if mains is not None and (r["game_id"], r["market_type"], r["player_id"],
                                  abs(r["line"])) not in mains:
            continue
        key = (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"])
        hit = best.get(key)
        if not hit:
            continue
        book, price = hit
        if bettable and book not in bettable:
            # find the best price among books we can actually bet
            alts = query(
                "SELECT book, price FROM odds_current WHERE game_id=? AND market_type=? "
                "AND player_id=? AND side=? AND line=? ORDER BY price DESC",
                (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"]),
            )
            hit = next(((a["book"], a["price"]) for a in alts if a["book"] in bettable), None)
            if not hit:
                continue
            book, price = hit
        anchor = r["sharp_prob"] if (use_sharp_anchor and r["sharp_prob"]) else r["fair_prob"]
        if not anchor:
            continue
        e = edge_pct(anchor, price)
        if e < min_edge:
            continue
        out.append({
            "game_id": r["game_id"], "market_type": r["market_type"],
            "player_id": r["player_id"], "side": r["side"], "line": r["line"],
            "best_book": book, "best_price": price,
            "fair_prob": anchor,
            "consensus_prob": r["fair_prob"],
            "sharp_prob": r["sharp_prob"],
            "anchor": "sharp" if (use_sharp_anchor and r["sharp_prob"]) else "consensus",
            "edge_pct": e,
            "book_count": r["book_count"],
            "dispersion": r["dispersion"],
            "decimal": american_to_decimal(price),
        })
    return sorted(out, key=lambda x: -x["edge_pct"])


def price_spread_across_books(game_id: str, market_type: str,
                              side: str, line: float = 0.0) -> list[dict]:
    """Every book's price for one market — the shopping view, worst to best."""
    rows = query(
        "SELECT book, price FROM odds_current WHERE game_id=? AND market_type=? "
        "AND side=? AND line=? ORDER BY price DESC",
        (game_id, market_type, side, line),
    )
    if not rows:
        return []
    best = rows[0]["price"]
    return [{
        "book": r["book"], "price": r["price"],
        "cents_worse_than_best": abs(r["price"] - best),
        "is_best": r["price"] == best,
    } for r in rows]


def shopping_value(game_id: str | None = None) -> dict:
    """Quantify what line shopping is worth vs. always betting one book.

    Answers "how much am I leaving on the table by not shopping?" — useful for
    deciding which books are worth maintaining accounts at.
    """
    sql = ("SELECT game_id, market_type, player_id, side, line, book, price "
           "FROM odds_current")
    params: list = []
    if game_id:
        sql += " WHERE game_id=?"
        params = [game_id]

    by_market: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    for r in query(sql, params):
        by_market[(r["game_id"], r["market_type"], r["player_id"],
                   r["side"], r["line"])].append((r["book"], r["price"]))

    book_best_count: dict[str, int] = defaultdict(int)
    gains: list[float] = []
    for entries in by_market.values():
        if len(entries) < 2:
            continue
        prices = [p for _, p in entries]
        best, worst = max(prices), min(prices)
        for b, p in entries:
            if p == best:
                book_best_count[b] += 1
        gains.append((american_to_decimal(best) - american_to_decimal(worst))
                     / american_to_decimal(worst) * 100.0)

    return {
        "markets_compared": len(gains),
        "avg_pct_gain_best_vs_worst": round(sum(gains) / len(gains), 3) if gains else 0.0,
        "max_pct_gain": round(max(gains), 3) if gains else 0.0,
        "best_price_wins_by_book": dict(sorted(book_best_count.items(),
                                               key=lambda x: -x[1])),
    }


def edge_distribution() -> dict:
    """Diagnostic: is the market efficient, or is the pipeline broken?

    "Zero opportunities" is ambiguous on its own. This shows the full edge
    distribution so you can tell a tight market (edges clustered near zero,
    healthy) from a broken join (no rows at all, or absurd edges everywhere).
    """
    all_opps = find_opportunities(min_edge=-100.0, min_books=1,
                                  main_line_only=False, bettable_only=False)
    filtered = find_opportunities(min_edge=-100.0)
    if not all_opps:
        return {"error": "no rows — check that fair_prices and odds_current join",
                "fair_prices": len(query("SELECT 1 FROM fair_prices")),
                "odds_current": len(query("SELECT 1 FROM odds_current"))}

    edges = sorted(o["edge_pct"] for o in all_opps)
    buckets: dict[str, int] = {}
    for lo, hi in [(-100, -5), (-5, -2), (-2, 0), (0, 1), (1, 2),
                   (2, 3), (3, 5), (5, 10), (10, 1000)]:
        label = f"{lo:>4} to {hi:<5}"
        buckets[label] = sum(1 for e in edges if lo <= e < hi)

    n = len(edges)
    by_anchor: dict[str, int] = {}
    for o in all_opps:
        by_anchor[o["anchor"]] = by_anchor.get(o["anchor"], 0) + 1

    fedges = sorted(o["edge_pct"] for o in filtered) or [0.0]
    sharp_cov = sum(1 for o in all_opps if o["anchor"] == "sharp") / n * 100

    return {
        "markets_priced": n,
        "markets_after_filters": len(filtered),
        "median_edge": round(edges[n // 2], 3),
        "p90_edge": round(edges[int(n * 0.9)], 3),
        "p99_edge": round(edges[int(n * 0.99)], 3),
        "max_edge": round(edges[-1], 3),
        "pct_positive": round(sum(1 for e in edges if e > 0) / n * 100, 1),
        "sharp_coverage_pct": round(sharp_cov, 1),
        "anchor_used": by_anchor,
        "histogram": buckets,
        "top": all_opps[:10],
        "filtered_median": round(fedges[len(fedges) // 2], 3),
        "filtered_max": round(fedges[-1], 3),
        "filtered_positive": sum(1 for e in fedges if e > 0),
        "top_filtered": filtered[:10],
    }


if __name__ == "__main__":
    import argparse
    from src.db import query

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="line shopping")
    p.add_argument("command", choices=["opps", "value", "dist"])
    p.add_argument("--min-edge", type=float, default=None)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "dist":
        d = edge_distribution()
        if "error" in d:
            print(f"  PROBLEM: {d['error']}")
            print(f"  fair_prices rows: {d['fair_prices']}  odds_current rows: {d['odds_current']}")
            raise SystemExit(1)
        print(f"markets priced   : {d['markets_priced']}")
        print(f"median edge      : {d['median_edge']:+.2f}%   "
              f"(-4.55% == paying full vig at -110; near this is EXPECTED)")
        print(f"p90 / p99 / max  : {d['p90_edge']:+.2f}% / {d['p99_edge']:+.2f}% / {d['max_edge']:+.2f}%")
        print(f"pct positive     : {d['pct_positive']}%")
        print(f"sharp coverage   : {d['sharp_coverage_pct']}%  <- low means Pinnacle "
              f"is quoting a different line than the field")
        print(f"anchor used      : {d['anchor_used']}")
        print("\nedge histogram (unfiltered):")
        for k, v in d["histogram"].items():
            bar = "#" * min(60, v * 60 // max(d["markets_priced"], 1) + (1 if v else 0))
            print(f"  {k}% {v:>6}  {bar}")
        print("\ntop 10 UNFILTERED (thin lines — mostly noise):")
        for o in d["top"]:
            print(f"  {o['market_type']:<10}{o['side']:<6}{o['line']:>7.1f}  "
                  f"{o['best_book']:<14}{o['best_price']:>+6d}  edge={o['edge_pct']:+6.2f}%  "
                  f"anchor={o['anchor']:<9} books={o['book_count']}")
        print(f"\n--- after filters (main line, >=8 books, bettable books only) ---")
        print(f"markets kept     : {d['markets_after_filters']} of {d['markets_priced']}")
        print(f"median / max     : {d['filtered_median']:+.2f}% / {d['filtered_max']:+.2f}%")
        print(f"positive edges   : {d['filtered_positive']}")
        print("\ntop 10 FILTERED (these are the real candidates):")
        for o in d["top_filtered"]:
            print(f"  {o['market_type']:<10}{o['side']:<6}{o['line']:>7.1f}  "
                  f"{o['best_book']:<14}{o['best_price']:>+6d}  edge={o['edge_pct']:+6.2f}%  "
                  f"anchor={o['anchor']:<9} books={o['book_count']}")
    elif args.command == "value":
        v = shopping_value()
        print(f"markets compared      : {v['markets_compared']}")
        print(f"avg gain best vs worst: {v['avg_pct_gain_best_vs_worst']}%")
        print(f"max gain              : {v['max_pct_gain']}%")
        print("best-price wins by book:")
        for b, n in list(v["best_price_wins_by_book"].items())[:15]:
            print(f"   {b:<20} {n}")
    else:
        opps = find_opportunities(min_edge=args.min_edge)
        print(f"{len(opps)} opportunities >= {args.min_edge or settings.MIN_EDGE_PCT}% edge\n")
        print(f"  {'market':<22}{'side':<7}{'line':>7}  {'book':<14}{'price':>7}"
              f"{'edge':>8}{'anchor':>10}{'books':>7}")
        for o in opps[:30]:
            print(f"  {o['market_type']:<22}{o['side']:<7}{o['line']:>7.1f}  "
                  f"{o['best_book']:<14}{o['best_price']:>+7d}"
                  f"{o['edge_pct']:>7.2f}%{o['anchor']:>10}{o['book_count']:>7}")
