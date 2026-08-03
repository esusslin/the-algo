"""Grading and CLV.

Two jobs, both essential to an honest record:

grade()   Resolve every pending pick against the final result. Wins, losses,
          and pushes — pushes matter, since a spread landing exactly on the
          number is not a loss and counting it as one understates you.

capture_closing()  Snapshot the final pre-kickoff price and devigged
          probability for every pick, then compute CLV. This must run BEFORE
          kickoff (odds vanish afterwards), which is why it's a separate job on
          its own schedule rather than part of grading.

A warning that cost this project's predecessor a rewrite: sign conventions on
spreads are the single most common silent bug in betting systems. A flipped
sign produces a plausible-looking ~50% hit rate and quietly inverts every
result. The grading rules below are unit-tested in tests/test_grading.py
against hand-verified cases; do not change them without updating those.
"""
from __future__ import annotations

import logging

from src.db import db, query, utcnow
from src.markets import CLASS_PROP, describe_market
from src.market.devig import american_to_prob, edge_pct

log = logging.getLogger(__name__)

WIN, LOSS, PUSH, VOID, PENDING = "win", "loss", "push", "void", "pending"


# --------------------------------------------------------------------------
# outcome resolution
# --------------------------------------------------------------------------
def grade_game_market(market_type: str, side: str, line: float,
                      home_score: int, away_score: int) -> str:
    """Resolve a game-level bet from the final score.

    Sign convention, stated explicitly because it is the bug that bites:
      `line` is from the perspective of THE SIDE BET.
      A home -3.5 pick stores side='home', line=-3.5.
      Home covers when (home_score - away_score) + line > 0.
    """
    margin = home_score - away_score

    if market_type.startswith("h2h"):
        if margin == 0:
            return PUSH
        winner = "home" if margin > 0 else "away"
        return WIN if side == winner else LOSS

    if "spreads" in market_type:
        adjusted = (margin if side == "home" else -margin) + line
        if abs(adjusted) < 1e-9:
            return PUSH
        return WIN if adjusted > 0 else LOSS

    if "totals" in market_type:
        total = home_score + away_score
        if abs(total - line) < 1e-9:
            return PUSH
        if side == "over":
            return WIN if total > line else LOSS
        return WIN if total < line else LOSS

    return PENDING


def grade_prop(side: str, line: float, actual: float | None,
               one_sided: bool = False) -> str:
    """Resolve a player prop from the player's final stat."""
    if actual is None:
        return PENDING
    if one_sided:                      # anytime TD etc: 'yes' if actual >= 1
        hit = actual >= 1
        return WIN if (side in ("yes", "over")) == hit else LOSS
    if abs(actual - line) < 1e-9:
        return PUSH
    if side == "over":
        return WIN if actual > line else LOSS
    return WIN if actual < line else LOSS


def payout_for(result: str, stake: float, american_price: int) -> float:
    """Net profit (not returned stake). Loss is negative."""
    if result == WIN:
        return stake * (american_price / 100.0) if american_price > 0 \
            else stake * (100.0 / abs(american_price))
    if result == LOSS:
        return -stake
    return 0.0                          # push / void


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
def grade_picks() -> dict[str, int]:
    """Grade every pending pick whose game has a final score."""
    rows = query(
        "SELECT p.pick_id, p.market_type, p.side, p.line, p.player_id, "
        "       g.home_score, g.away_score, g.status "
        "FROM picks p JOIN games g ON g.game_id = p.game_id "
        "WHERE p.result='pending' AND g.home_score IS NOT NULL "
        "AND g.away_score IS NOT NULL"
    )
    counts = {WIN: 0, LOSS: 0, PUSH: 0, PENDING: 0}
    now = utcnow()
    with db() as conn:
        for r in rows:
            info = describe_market(r["market_type"])
            if info.bet_class == CLASS_PROP:
                stat = conn.execute(
                    "SELECT value FROM live_player_stats WHERE game_id=(SELECT game_id "
                    "FROM picks WHERE pick_id=?) AND player_id=? AND stat_key=?",
                    (r["pick_id"], r["player_id"], r["market_type"]),
                ).fetchone()
                result = grade_prop(r["side"], r["line"],
                                    stat["value"] if stat else None,
                                    one_sided=not info.two_sided)
            else:
                result = grade_game_market(r["market_type"], r["side"], r["line"],
                                           r["home_score"], r["away_score"])
            counts[result] = counts.get(result, 0) + 1
            if result == PENDING:
                continue
            conn.execute("UPDATE picks SET result=?, graded_at=? WHERE pick_id=?",
                         (result, now, r["pick_id"]))

    graded = counts[WIN] + counts[LOSS] + counts[PUSH]
    log.info("graded %d picks: %dW %dL %dP (%d still pending)",
             graded, counts[WIN], counts[LOSS], counts[PUSH], counts[PENDING])
    return counts


def grade_user_bets() -> int:
    """Propagate pick results onto users' logged bets, with their own price."""
    rows = query(
        "SELECT b.id, b.stake, b.price, p.result "
        "FROM user_bets b JOIN picks p ON p.pick_id = b.pick_id "
        "WHERE b.result='pending' AND p.result != 'pending'"
    )
    now = utcnow()
    with db() as conn:
        for r in rows:
            conn.execute(
                "UPDATE user_bets SET result=?, payout=?, graded_at=? WHERE id=?",
                (r["result"], payout_for(r["result"], r["stake"] or 0, r["price"]),
                 now, r["id"]),
            )
    log.info("graded %d user bets", len(rows))
    return len(rows)


def capture_closing_lines(minutes_before: int = 10) -> int:
    """Snapshot the final pre-kickoff price for picks about to start.

    MUST run before kickoff — odds disappear once a game begins, and without a
    closing price there is no CLV, which is the metric that tells you whether
    any of this works.
    """
    rows = query(
        "SELECT p.pick_id, p.game_id, p.market_type, p.player_id, p.side, "
        "       p.line, p.best_price "
        "FROM picks p JOIN games g ON g.game_id = p.game_id "
        "WHERE p.closing_price IS NULL AND g.status != 'final' "
        "AND g.kickoff_utc IS NOT NULL "
        "AND datetime(g.kickoff_utc) <= datetime('now', ?)",
        (f"+{minutes_before} minutes",),
    )
    updated = 0
    with db() as conn:
        for r in rows:
            fp = conn.execute(
                "SELECT fair_prob, sharp_prob FROM fair_prices WHERE game_id=? "
                "AND market_type=? AND player_id=? AND side=? AND line=?",
                (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"]),
            ).fetchone()
            best = conn.execute(
                "SELECT MAX(price) AS p FROM odds_current WHERE game_id=? "
                "AND market_type=? AND player_id=? AND side=? AND line=?",
                (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"]),
            ).fetchone()
            if not best or best["p"] is None:
                continue
            closing_price = int(best["p"])
            closing_prob = (fp["sharp_prob"] or fp["fair_prob"]) if fp else None

            # CLV: how much better was the price we took than the closing price?
            # Expressed as EV percentage points at the closing fair probability.
            clv = None
            if closing_prob:
                clv = edge_pct(closing_prob, r["best_price"]) - \
                      edge_pct(closing_prob, closing_price)
            conn.execute(
                "UPDATE picks SET closing_price=?, closing_fair_prob=?, clv_pct=? "
                "WHERE pick_id=?",
                (closing_price, closing_prob, clv, r["pick_id"]),
            )
            # same closing price applies to each user's own entry price
            for b in conn.execute(
                "SELECT id, price FROM user_bets WHERE pick_id=? AND clv_pct IS NULL",
                (r["pick_id"],),
            ).fetchall():
                ucl = (edge_pct(closing_prob, b["price"]) -
                       edge_pct(closing_prob, closing_price)) if closing_prob else None
                conn.execute("UPDATE user_bets SET closing_price=?, clv_pct=? WHERE id=?",
                             (closing_price, ucl, b["id"]))
            updated += 1
    log.info("captured closing lines for %d picks", updated)
    return updated


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="grading and CLV")
    p.add_argument("command", choices=["grade", "closing", "selftest"])
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "grade":
        print(grade_picks())
        print(f"{grade_user_bets()} user bets graded")
    elif args.command == "closing":
        print(f"{capture_closing_lines()} closing lines captured")
    else:
        # hand-verified cases — the sign-convention guard
        cases = [
            ("h2h", "home", 0, 24, 17, WIN), ("h2h", "away", 0, 24, 17, LOSS),
            ("h2h", "home", 0, 20, 20, PUSH),
            ("spreads", "home", -3.5, 24, 17, WIN),    # won by 7, laid 3.5
            ("spreads", "home", -7.5, 24, 17, LOSS),   # won by 7, laid 7.5
            ("spreads", "home", -7.0, 24, 17, PUSH),   # won by exactly 7
            ("spreads", "away", 3.5, 24, 17, LOSS),    # lost by 7, got 3.5
            ("spreads", "away", 10.5, 24, 17, WIN),    # lost by 7, got 10.5
            ("totals", "over", 40.5, 24, 17, WIN),     # 41 total
            ("totals", "under", 40.5, 24, 17, LOSS),
            ("totals", "over", 41.0, 24, 17, PUSH),
        ]
        bad = 0
        for mkt, side, line, hs, aw, want in cases:
            got = grade_game_market(mkt, side, line, hs, aw)
            ok = "OK  " if got == want else "FAIL"
            if got != want:
                bad += 1
            print(f"  {ok} {mkt:<8} {side:<5} {line:>6}  {hs}-{aw} -> {got:<5} (want {want})")
        print(f"\n  {len(cases)-bad}/{len(cases)} passed")
        print("\n  prop grading:")
        for side, line, actual, want in [("over", 58.5, 62, WIN), ("over", 58.5, 51, LOSS),
                                         ("under", 58.5, 51, WIN), ("over", 60.0, 60, PUSH)]:
            got = grade_prop(side, line, actual)
            print(f"    {'OK  ' if got == want else 'FAIL'} {side} {line} actual={actual} -> {got}")
