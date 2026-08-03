"""Live bet status — what's happening to your open positions right now.

Data comes from ESPN's undocumented scoreboard API: free, no key, updates in
near-real-time. It is unversioned and WILL break without notice, so every
consumer degrades to "scores unavailable" rather than erroring. Never let
anything load-bearing depend on it.
"""
from __future__ import annotations

import logging

import httpx

from src.db import db, query, upsert_rows, utcnow
from src.markets import CLASS_PROP, describe_market

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = httpx.Timeout(20.0, connect=8.0)


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------
def fetch_live() -> int:
    """Pull current scores into `live_state`. Returns games updated."""
    from src.teams import resolve

    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
            r = c.get(ESPN_SCOREBOARD)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — ESPN is best-effort by design
        log.warning("live fetch failed: %s", exc)
        return 0

    game_index = {
        (r["home_team"], r["away_team"], (r["kickoff_utc"] or "")[:10]): r["game_id"]
        for r in query("SELECT game_id, home_team, away_team, kickoff_utc FROM games")
    }

    rows = []
    for ev in data.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        teams = comp.get("competitors") or []
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        h = resolve((home.get("team") or {}).get("displayName"))
        a = resolve((away.get("team") or {}).get("displayName"))
        day = (ev.get("date") or "")[:10]
        gid = next((game_index.get((h, a, d)) for d in (day,)
                    if game_index.get((h, a, d))), None)
        if not gid:
            continue
        status = (comp.get("status") or {}).get("type", {})
        rows.append({
            "game_id": gid,
            "home_score": int(home.get("score") or 0),
            "away_score": int(away.get("score") or 0),
            "period": (comp.get("status") or {}).get("period"),
            "clock": (comp.get("status") or {}).get("displayClock"),
            "status": status.get("state"),          # pre | in | post
            "possession": (comp.get("situation") or {}).get("possession"),
            "home_win_prob": None,
            "updated_at": utcnow(),
        })

    if rows:
        with db() as conn:
            upsert_rows(conn, "live_state", rows, key_cols=["game_id"])
    return len(rows)


# --------------------------------------------------------------------------
# status evaluation
# --------------------------------------------------------------------------
def evaluate(market_type: str, side: str, line: float,
             home_score: int, away_score: int,
             prop_value: float | None = None) -> dict:
    """Current standing of one bet. Returns status, margin and a readable note."""
    info = describe_market(market_type)

    if info.bet_class == CLASS_PROP:
        if prop_value is None:
            return {"status": "unknown", "cushion": None, "note": "stat unavailable"}
        cushion = prop_value - line if side == "over" else line - prop_value
        return {
            "status": "winning" if cushion > 0 else "losing" if cushion < 0 else "tied",
            "cushion": round(cushion, 1),
            "note": f"{prop_value:g} of {line:g}",
        }

    margin = home_score - away_score

    if market_type.startswith("h2h"):
        m = margin if side == "home" else -margin
        return {"status": "winning" if m > 0 else "losing" if m < 0 else "tied",
                "cushion": m, "note": f"{'up' if m > 0 else 'down' if m else 'tied'} {abs(m)}"}

    if "spreads" in market_type:
        adj = (margin if side == "home" else -margin) + line
        return {"status": "winning" if adj > 0 else "losing" if adj < 0 else "tied",
                "cushion": round(adj, 1),
                "note": f"covering by {adj:g}" if adj > 0 else f"needs {abs(adj):g}"}

    if "totals" in market_type:
        total = home_score + away_score
        cushion = (total - line) if side == "over" else (line - total)
        return {"status": "winning" if cushion > 0 else "losing" if cushion < 0 else "tied",
                "cushion": round(cushion, 1),
                "note": f"{total} of {line:g}"}

    return {"status": "unknown", "cushion": None, "note": ""}


def open_bet_status(user_id: int) -> dict:
    """Every open bet for a user, with current standing."""
    rows = query(
        "SELECT b.id AS bet_id, b.stake, b.price, b.book, "
        "       p.pick_id, p.market_type, p.side, p.line, p.tier, p.headline, "
        "       p.player_id, g.game_id, g.home_team, g.away_team, g.kickoff_utc, "
        "       l.home_score, l.away_score, l.period, l.clock, l.status AS game_state "
        "FROM user_bets b "
        "JOIN picks p ON p.pick_id = b.pick_id "
        "JOIN games g ON g.game_id = p.game_id "
        "LEFT JOIN live_state l ON l.game_id = g.game_id "
        "WHERE b.user_id=? AND b.result='pending' "
        "ORDER BY g.kickoff_utc",
        (user_id,),
    )

    out, winning, live_count = [], 0, 0
    at_risk = 0.0
    for r in rows:
        d = dict(r)
        if r["home_score"] is None:
            d.update(status="pending", cushion=None, note="not started")
        else:
            prop_val = None
            if describe_market(r["market_type"]).bet_class == CLASS_PROP:
                s = query("SELECT value FROM live_player_stats WHERE game_id=? "
                          "AND player_id=? AND stat_key=?",
                          (r["game_id"], r["player_id"], r["market_type"]))
                prop_val = s[0]["value"] if s else None
            d.update(evaluate(r["market_type"], r["side"], r["line"] or 0,
                              r["home_score"], r["away_score"], prop_val))
            if r["game_state"] == "in":
                live_count += 1
            if d["status"] == "winning":
                winning += 1
        at_risk += r["stake"] or 0
        out.append(d)

    return {
        "open": len(out),
        "live": live_count,
        "winning": winning,
        "units_at_risk": round(at_risk, 3),
        "bets": out,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="live bet status")
    p.add_argument("command", choices=["fetch", "status", "selftest"])
    p.add_argument("--user", type=int, default=1)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "fetch":
        print(f"{fetch_live()} games updated")
    elif args.command == "status":
        s = open_bet_status(args.user)
        print(f"open={s['open']} live={s['live']} winning={s['winning']} "
              f"at risk={s['units_at_risk']}u")
        for b in s["bets"]:
            print(f"  [{b['tier']}] {b['headline'][:44]:<44} {b['status']:<8} {b['note']}")
    else:
        cases = [
            ("spreads", "home", -3.5, 17, 10, "winning"),
            ("spreads", "home", -7.5, 17, 10, "losing"),
            ("spreads", "away", 3.5, 17, 10, "losing"),
            ("h2h", "away", 0, 17, 24, "winning"),
            ("totals", "over", 44.5, 24, 21, "winning"),
            ("totals", "under", 44.5, 24, 21, "losing"),
        ]
        for mkt, side, line, hs, aw, want in cases:
            r = evaluate(mkt, side, line, hs, aw)
            print(f"  {'OK  ' if r['status'] == want else 'FAIL'} {mkt:<8}{side:<5}"
                  f"{line:>6}  {hs}-{aw} -> {r['status']:<8} {r['note']}")
        r = evaluate("player_reception_yds", "over", 58.5, 0, 0, prop_value=71)
        print(f"  prop: {r}")
