"""Seed realistic demo data so testers see a populated Track and History.

Why this needs care
-------------------
Fake results in the history table would poison CLV stats and corrupt the track
record you eventually show people. So every row written here carries demo=1,
every history query excludes demo rows by default, and `purge` removes them
completely. Never relax that.

The generated record is deliberately UNIMPRESSIVE: ~53% win rate, ROI in the
low single digits, CLV slightly positive. A demo showing 68% and +40% ROI would
set expectations no real system can meet, and testers would calibrate their
feedback against a fantasy.

    python scripts/seed_demo.py seed
    python scripts/seed_demo.py purge
    python scripts/seed_demo.py status
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import db, insert_row, query, run_migrations, utcnow  # noqa: E402
from src.picks.grading import payout_for  # noqa: E402

random.seed(20260908)

TEAMS = [("BUF", "KC"), ("PHI", "DAL"), ("SF", "LAR"), ("BAL", "CIN"),
         ("DET", "GB"), ("MIA", "NYJ"), ("HOU", "IND"), ("TB", "NO")]

MARKETS = [
    ("h2h", "home", 0.0, None),
    ("h2h", "away", 0.0, None),
    ("spreads", "home", -3.5, None),
    ("spreads", "away", 6.5, None),
    ("totals", "over", 44.5, None),
    ("totals", "under", 47.5, None),
    ("totals_h1", "under", 23.5, None),
    ("spreads_h1", "home", -1.5, None),
    ("player_reception_yds", "over", 58.5, "Cooper Kupp"),
    ("player_rush_yds", "over", 71.5, "Saquon Barkley"),
    ("player_pass_yds", "under", 248.5, "Josh Allen"),
    ("player_anytime_td", "yes", 0.0, "Travis Kelce"),
]

BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "betrivers", "espnbet"]
# Target ~53% — a real, unglamorous edge. Not a fantasy.
WIN_RATE = 0.53


def _price() -> int:
    return random.choice([-125, -120, -115, -112, -110, -108, -105, +100, +105, +110, +125])


def seed(weeks: int = 3, per_week: int = 14) -> dict:
    run_migrations()
    users = [dict(r) for r in query("SELECT id, username FROM users")]
    if not users:
        print("no users yet — create your admin account first (boot the app once)")
        return {}

    now = datetime.now(timezone.utc)
    made = {"games": 0, "picks": 0, "bets": 0}

    with db() as conn:
        for w in range(weeks):
            kickoff = now - timedelta(days=7 * (weeks - w))
            for i, (home, away) in enumerate(TEAMS[: per_week // 2]):
                gid = f"DEMO_{2026}_{w+1:02d}_{away}_{home}"
                hs, aws = random.randint(10, 34), random.randint(7, 31)
                conn.execute(
                    "INSERT OR REPLACE INTO games (game_id, season, week, season_type, "
                    "kickoff_utc, home_team, away_team, status, home_score, away_score, "
                    "spread_line, total_line, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (gid, 2026, w + 1, "REG", kickoff.isoformat(timespec="seconds"),
                     home, away, "final", hs, aws, -3.0, 45.5, utcnow()),
                )
                made["games"] += 1

                for mkt, side, line, player in random.sample(MARKETS, 3):
                    price = _price()
                    result = "win" if random.random() < WIN_RATE else "loss"
                    if random.random() < 0.04:
                        result = "push"
                    edge = round(random.uniform(2.0, 6.5), 2)
                    tier = "A" if edge >= 5 else "B" if edge >= 3 else "C"
                    stake = round(min(edge / 4, 1.5), 2)
                    # CLV centred slightly positive — the honest signature of a
                    # system with a small real edge
                    clv = round(random.gauss(0.6, 1.8), 2)
                    pid = insert_row(conn, "picks", {
                        "game_id": gid, "market_type": mkt,
                        "player_id": "", "side": side, "line": line,
                        "source": "market_engine", "best_book": random.choice(BOOKS),
                        "best_price": price, "fair_prob": 0.54, "blended_prob": 0.54,
                        "edge_pct": edge, "kelly_units": stake, "tier": tier,
                        "ai_verdict": "OK",
                        "headline": (f"{player} {side} {line:g}" if player
                                     else f"{side.title()} {line:g}" if line
                                     else side.title()),
                        "detail": "Demo data for UI review.",
                        "published": 1, "visibility": "all",
                        "created_at": kickoff.isoformat(timespec="seconds"),
                        "published_at": kickoff.isoformat(timespec="seconds"),
                        "closing_price": price - random.choice([-8, -5, 0, 5]),
                        "closing_fair_prob": 0.54, "clv_pct": clv,
                        "result": result, "graded_at": utcnow(), "demo": 1,
                    })
                    made["picks"] += 1

                    # each user takes roughly half the picks, at their own price
                    for u in users:
                        if random.random() > 0.5:
                            continue
                        upx = price - random.choice([0, 0, 3, 5, 8])
                        ustake = round(stake * random.uniform(0.6, 1.4), 2)
                        insert_row(conn, "user_bets", {
                            "user_id": u["id"], "pick_id": pid,
                            "book": random.choice(BOOKS), "price": upx,
                            "stake": ustake,
                            "placed_at": kickoff.isoformat(timespec="seconds"),
                            "result": result,
                            "payout": payout_for(result, ustake, upx),
                            "clv_pct": round(clv - random.uniform(0, 1.2), 2),
                            "closing_price": price - 4,
                            "graded_at": utcnow(), "demo": 1,
                        })
                        made["bets"] += 1

    print(f"seeded {made['games']} games, {made['picks']} picks, {made['bets']} user bets")
    print("all flagged demo=1 — excluded from real history, removable with `purge`")
    return made


def purge() -> dict:
    run_migrations()
    with db() as conn:
        b = conn.execute("DELETE FROM user_bets WHERE demo=1").rowcount
        p = conn.execute("DELETE FROM picks WHERE demo=1").rowcount
        g = conn.execute("DELETE FROM games WHERE game_id LIKE 'DEMO_%'").rowcount
    print(f"purged {g} games, {p} picks, {b} user bets")
    return {"games": g, "picks": p, "bets": b}


def status() -> None:
    run_migrations()
    d = query("SELECT COUNT(*) n FROM picks WHERE demo=1")[0]["n"]
    r = query("SELECT COUNT(*) n FROM picks WHERE demo=0")[0]["n"]
    db_ = query("SELECT COUNT(*) n FROM user_bets WHERE demo=1")[0]["n"]
    rb = query("SELECT COUNT(*) n FROM user_bets WHERE demo=0")[0]["n"]
    print(f"  picks     : {r} real, {d} demo")
    print(f"  user bets : {rb} real, {db_} demo")
    if d:
        w = query("SELECT result, COUNT(*) n FROM picks WHERE demo=1 GROUP BY result")
        parts = ", ".join("{}: {}".format(x["result"], x["n"]) for x in w)
        print(f"  demo results: {parts}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "seed":
        seed()
    elif cmd == "purge":
        purge()
    else:
        status()
