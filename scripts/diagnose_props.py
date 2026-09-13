"""Why are there no prop picks? Answers it with counts, not theories.

Props reach the UI only by surviving a chain of independent gates, and "the tab
is empty" looks identical no matter which one killed them. This walks the chain
in order and reports how many rows enter and leave each stage, so the answer is
a number rather than a hypothesis.

Run it where the data is:

    /opt/venv/bin/python -m scripts.diagnose_props

Every stage prints what would fix it. The point is to find the FIRST stage that
zeroes out — fixing anything downstream of that changes nothing.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from src.config import settings
from src.db import query
from src.markets import describe_market

BAR = "─" * 68


def _is_prop(market_type: str) -> bool:
    try:
        return describe_market(market_type).bet_class == "prop"
    except Exception:  # noqa: BLE001 — unknown market is not a prop we can price
        return False


def stage_0_config() -> bool:
    print(f"{BAR}\n0. CONFIG — is prop collection even switched on?\n{BAR}")
    ok = True
    print(f"  ENABLE_PROPS                 {settings.ENABLE_PROPS}")
    if not settings.ENABLE_PROPS:
        print("    ^ FALSE. Nothing below can happen. Set ENABLE_PROPS=true in Railway.")
        ok = False
    budget = settings.ODDS_MONTHLY_CREDIT_BUDGET
    print(f"  ODDS_MONTHLY_CREDIT_BUDGET   {budget:,}")
    print(f"  MIN_EDGE_PCT                 {settings.MIN_EDGE_PCT}   <- global floor, "
          f"applied BEFORE tiering")
    return ok


def stage_1_raw() -> set[str]:
    print(f"\n{BAR}\n1. COLLECTION — are prop odds in odds_current at all?\n{BAR}")
    rows = query("SELECT market_type, COUNT(*) n, COUNT(DISTINCT book) books, "
                 "MAX(fetched_at) latest FROM odds_current GROUP BY market_type")
    props = [r for r in rows if _is_prop(r["market_type"])]
    if not props:
        print("  ZERO prop rows in odds_current.")
        print("    -> collection is the problem, not pricing. Check the props tier in")
        print("       odds_api.TIERS, the credit ledger, and job_runs for poll_odds.")
        return set()
    for r in sorted(props, key=lambda r: -r["n"]):
        print(f"  {r['market_type']:<28}{r['n']:>7} rows  {r['books']:>3} books  "
              f"last {r['latest']}")
    return {r["market_type"] for r in props}


def stage_2_pairing(prop_markets: set[str]) -> None:
    """The stage most likely to be silently eating everything.

    `build_fair_prices` can only devig a book that quotes BOTH sides of the same
    line. A book showing only the over, or only 'yes' on anytime TD, contributes
    nothing — it is counted as `unpaired` and dropped without an error.
    """
    print(f"\n{BAR}\n2. PAIRING — can we devig? (needs BOTH sides from the SAME book)\n{BAR}")
    if not prop_markets:
        print("  skipped: no prop rows.")
        return

    marks = ",".join("?" * len(prop_markets))
    rows = query(
        f"SELECT game_id, market_type, player_id, side, line, book "
        f"FROM odds_current WHERE market_type IN ({marks})", tuple(prop_markets))

    grouped: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        grouped[(r["game_id"], r["market_type"], r["player_id"], r["line"],
                 r["book"])].add(r["side"])

    paired = Counter()
    lonely = Counter()
    for (_, mkt, _, _, _), sides in grouped.items():
        two = any(a in sides and b in sides
                  for a, b in [("over", "under"), ("yes", "no"), ("home", "away")])
        (paired if two else lonely)[mkt] += 1

    for mkt in sorted(prop_markets):
        p, l = paired[mkt], lonely[mkt]
        total = p + l
        if not total:
            continue
        pct = 100 * p / total
        flag = "  <- ALL ONE-SIDED, cannot be priced" if p == 0 else ""
        print(f"  {mkt:<28}{p:>6} paired  {l:>6} one-sided  ({pct:4.0f}% usable){flag}")

    if paired.total() == 0:
        print("\n  NOTHING is two-sided. Every prop is dropped before pricing.")
        print("    -> the odds request is not asking for both sides, or the parser is")
        print("       storing only one. This is the whole problem; stop here.")


def stage_3_fair_prices() -> None:
    print(f"\n{BAR}\n3. PRICING — did anything reach fair_prices?\n{BAR}")
    rows = query("SELECT market_type, COUNT(*) n, AVG(book_count) books, "
                 "AVG(dispersion) disp, SUM(sharp_prob IS NOT NULL) sharp "
                 "FROM fair_prices GROUP BY market_type")
    props = [r for r in rows if _is_prop(r["market_type"])]
    if not props:
        print("  ZERO priced props. Stage 2 is where they died.")
        return
    for r in sorted(props, key=lambda r: -r["n"]):
        print(f"  {r['market_type']:<28}{r['n']:>6} priced  "
              f"avg {r['books']:.1f} books  dispersion {r['disp']:.4f}  "
              f"{r['sharp']} with a sharp anchor")


def stage_4_thresholds() -> None:
    """Where a priced prop meets the bar it has to clear.

    Two filters run in series and it matters which is which:

      * `find_opportunities` drops anything under MIN_EDGE_PCT — a GLOBAL floor,
        applied before any market class is considered.
      * `assign_tier` then applies the per-class thresholds below.

    So the effective minimum for a prop is max(MIN_EDGE_PCT, the tier's own
    minimum). If MIN_EDGE_PCT is 5.0 and prop tier C wants 3.2, the 3.2 is
    decorative — nothing between 3.2 and 5.0 ever reaches it.
    """
    from src.picks.generator import thresholds_for

    print(f"\n{BAR}\n4. THRESHOLDS — what a prop must clear\n{BAR}")
    example = "player_reception_yds"
    print(f"  global floor (applied first): {settings.MIN_EDGE_PCT}% edge\n")
    for tier in ("A", "B", "C"):
        r = thresholds_for(example, tier)
        effective = max(r["min_edge"], settings.MIN_EDGE_PCT)
        dead = "  <- UNREACHABLE, floor is higher" if r["min_edge"] < settings.MIN_EDGE_PCT else ""
        print(f"  tier {tier}: edge >= {r['min_edge']:.1f}%  books >= {r['min_books']}  "
              f"dispersion <= {r['max_dispersion']:.3f}")
        print(f"          effective edge needed: {effective:.1f}%{dead}")

    print("\n  Live funnel:")
    try:
        from src.market.shop import find_opportunities
        loose = [o for o in find_opportunities(min_edge=-100.0, min_books=1)
                 if _is_prop(o["market_type"])]
        for cut in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            n = sum(1 for o in loose if o["edge_pct"] >= cut)
            mark = "  <- current floor" if abs(cut - settings.MIN_EDGE_PCT) < 0.01 else ""
            print(f"    edge >= {cut:>4.1f}%   {n:>5} props{mark}")
        if loose:
            books = sorted((o["book_count"] for o in loose), reverse=True)
            print(f"\n    book coverage: best {books[0]}, median {books[len(books)//2]}")
            print("    (tier C needs 4, B needs 6, A needs 8)")
    except Exception as exc:  # noqa: BLE001
        print(f"    could not run the funnel: {type(exc).__name__}: {exc}")


def stage_5_published() -> None:
    print(f"\n{BAR}\n5. PUBLISHED — prop picks actually written\n{BAR}")
    rows = query("SELECT market_type, tier, published, COUNT(*) n FROM picks "
                 "WHERE result='pending' GROUP BY market_type, tier, published")
    props = [r for r in rows if _is_prop(r["market_type"])]
    if not props:
        print("  No pending prop picks.")
        return
    for r in props:
        state = "published" if r["published"] else "DARK (generated, not shown)"
        print(f"  {r['market_type']:<28}tier {r['tier']}  {r['n']:>4}  {state}")


if __name__ == "__main__":
    print("\nWHY ARE THERE NO PROP PICKS?\n")
    if stage_0_config():
        markets = stage_1_raw()
        stage_2_pairing(markets)
        stage_3_fair_prices()
        stage_4_thresholds()
        stage_5_published()
    print(f"\n{BAR}")
    print("Fix the FIRST stage that reads zero. Everything downstream of it is")
    print("starved, not broken, and changing it will not produce a single pick.")
    print(f"{BAR}\n")
