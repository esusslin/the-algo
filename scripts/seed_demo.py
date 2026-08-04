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

from src.db import db, insert_row, query, run_migrations, upsert_rows, utcnow  # noqa: E402
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


def seed_pending(count: int = 14) -> int:
    """Pending picks so the Picks tab is populated for UI review.

    Uses REAL upcoming games where available, so matchups and kickoff times look
    right. Falls back to demo fixtures only if the schedule isn't loaded yet.
    Every row is demo=1 and carries a visible badge in the UI — a tester should
    never mistake these for live recommendations.
    """
    run_migrations()
    games = [dict(r) for r in query(
        "SELECT game_id, home_team, away_team FROM games "
        "WHERE status!='final' ORDER BY kickoff_utc LIMIT 20"
    )]

    # Self-contained: invent fixtures if the real schedule isn't loaded, so the
    # UI can be reviewed on a completely empty database.
    if not games:
        kickoff = datetime.now(timezone.utc) + timedelta(days=3)
        with db() as conn:
            for home, away in TEAMS:
                gid = f"DEMO_UPCOMING_{away}_{home}"
                conn.execute(
                    "INSERT OR REPLACE INTO games (game_id, season, week, season_type, "
                    "kickoff_utc, home_team, away_team, status, spread_line, total_line, "
                    "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (gid, 2026, 1, "REG", kickoff.isoformat(timespec="seconds"),
                     home, away, "scheduled", -3.0, 45.5, utcnow()),
                )
                games.append({"game_id": gid, "home_team": home, "away_team": away})
        print(f"  (no real schedule loaded — invented {len(games)} demo fixtures)")

    # Resolve demo player names against the real crosswalk so prop cards read
    # "Cooper Kupp over 58.5" rather than "Player over 58.5". Never invents
    # player rows — a fake gsis_id would corrupt real name resolution later.
    def pid_for(name: str | None) -> str:
        if not name:
            return ""
        r = query("SELECT player_id FROM players WHERE LOWER(full_name)=? LIMIT 1",
                  (name.lower(),))
        return r[0]["player_id"] if r else ""

    made = 0
    with db() as conn:
        for i in range(count):
            g = random.choice(games)
            mkt, side, line, player = random.choice(MARKETS)
            # spread the tiers so testers see all three states
            tier = ["A", "A", "B", "B", "B", "C", "C", "C", "C"][i % 9]
            edge = {"A": random.uniform(5.0, 7.5), "B": random.uniform(3.0, 4.9),
                    "C": random.uniform(2.0, 2.9)}[tier]
            books = {"A": random.randint(16, 22), "B": random.randint(10, 15),
                     "C": random.randint(8, 11)}[tier]
            price = _price()
            insert_row(conn, "picks", {
                "game_id": g["game_id"], "market_type": mkt,
                "player_id": pid_for(player),
                "side": side, "line": line, "source": "market_engine",
                "best_book": random.choice(BOOKS), "best_price": price,
                "fair_prob": 0.54, "blended_prob": 0.54,
                "edge_pct": round(edge, 2),
                "kelly_units": round(min(edge / 4, 1.5), 2), "tier": tier,
                "ai_verdict": "OK",
                "headline": (f"{player} {side} {line:g}" if player
                             else f"{side.title()} {line:g}" if line else side.title()),
                "detail": (f"DEMO — {books} books quoting, sharp book "
                           f"{'agrees' if tier == 'A' else 'unavailable'}. "
                           f"Placeholder for UI review, not a real recommendation."),
                "published": 1, "visibility": "all",
                "created_at": utcnow(), "published_at": utcnow(),
                "result": "pending", "demo": 1,
            })
            made += 1
    print(f"seeded {made} pending demo picks across A/B/C tiers")
    return made


def seed_live(per_user: int = 6) -> int:
    """Populate the My bets tab: open bets on games that look in-progress.

    Writes `live_state` rows so some games read as live with a score and clock,
    some as not started, some as just finished. Without this the tab is empty
    for testers and there's nothing to give feedback on.
    """
    run_migrations()
    picks = [dict(r) for r in query(
        "SELECT p.pick_id, p.game_id, p.best_price, p.kelly_units, p.best_book "
        "FROM picks p WHERE p.demo=1 AND p.result='pending' LIMIT 40")]
    users = [dict(r) for r in query("SELECT id FROM users")]
    if not picks or not users:
        print("seed pending picks and create an account first")
        return 0

    chosen = random.sample(picks, min(per_user, len(picks)))
    game_ids = list({p["game_id"] for p in chosen})

    # A believable Sunday: most games running, one about to start, one just done.
    states = []
    for i, gid in enumerate(game_ids):
        if i == 0:
            states.append((gid, None, None, None, None, "pre"))
        elif i == len(game_ids) - 1 and len(game_ids) > 2:
            states.append((gid, random.randint(17, 31), random.randint(10, 27),
                           4, "0:00", "post"))
        else:
            states.append((gid, random.randint(7, 28), random.randint(3, 24),
                           random.randint(2, 4),
                           f"{random.randint(1,14)}:{random.randint(0,59):02d}", "in"))

    made = 0
    with db() as conn:
        upsert_rows(conn, "live_state", [{
            "game_id": g, "home_score": hs, "away_score": aws, "period": per,
            "clock": clk, "status": st, "possession": None,
            "home_win_prob": None, "updated_at": utcnow(),
        } for g, hs, aws, per, clk, st in states], key_cols=["game_id"])

        # Prop bets need a live stat line or they read "stat unavailable" —
        # which is real behaviour in production but useless for UI review.
        # Match on player_id as stored, including '' — the status lookup uses
        # the same value, so an unresolved player still gets a stat line.
        props = [dict(r) for r in query(
            "SELECT DISTINCT p.game_id, p.player_id, p.market_type, p.line "
            "FROM picks p WHERE p.demo=1 AND p.result='pending' "
            "AND p.market_type LIKE 'player_%'")]
        if props:
            upsert_rows(conn, "live_player_stats", [{
                "game_id": pr["game_id"], "player_id": pr["player_id"],
                "player_name": "", "stat_key": pr["market_type"],
                # straddle the line so testers see both winning and losing props
                "value": round((pr["line"] or 50) * random.uniform(0.7, 1.35), 1),
                "updated_at": utcnow(),
            } for pr in props], key_cols=["game_id", "player_id", "stat_key"])

        for u in users:
            for p in chosen:
                dup = conn.execute(
                    "SELECT 1 FROM user_bets WHERE user_id=? AND pick_id=?",
                    (u["id"], p["pick_id"])).fetchone()
                if dup:
                    continue
                px = p["best_price"] - random.choice([0, 0, 3, 5])
                insert_row(conn, "user_bets", {
                    "user_id": u["id"], "pick_id": p["pick_id"],
                    "book": p["best_book"], "price": px,
                    "stake": round((p["kelly_units"] or 0.5) * random.uniform(.7, 1.3), 2),
                    "placed_at": utcnow(), "result": "pending", "demo": 1,
                })
                made += 1

    live_n = sum(1 for s in states if s[5] == "in")
    print(f"seeded {made} open bets across {len(users)} users; "
          f"{live_n} of {len(game_ids)} games showing live")
    return made


def backfill_users() -> int:
    """Give demo bets to users who joined AFTER the demo history was seeded.

    Without this, a tester invited later opens Track to an empty "My history"
    while everyone else has one, and reports it as a bug.
    """
    run_migrations()
    picks = [dict(r) for r in query(
        "SELECT pick_id, best_price, kelly_units, result FROM picks "
        "WHERE demo=1 AND result != 'pending'")]
    if not picks:
        print("no settled demo picks — run seed first")
        return 0

    users = [dict(r) for r in query(
        "SELECT u.id FROM users u WHERE NOT EXISTS "
        "(SELECT 1 FROM user_bets b WHERE b.user_id=u.id AND b.demo=1)")]
    if not users:
        print("every user already has demo bets")
        return 0

    made = 0
    with db() as conn:
        for u in users:
            for p in random.sample(picks, max(1, len(picks) // 2)):
                px = p["best_price"] - random.choice([0, 0, 3, 5, 8])
                stake = round((p["kelly_units"] or 0.5) * random.uniform(0.6, 1.4), 2)
                insert_row(conn, "user_bets", {
                    "user_id": u["id"], "pick_id": p["pick_id"],
                    "book": random.choice(BOOKS), "price": px, "stake": stake,
                    "placed_at": utcnow(), "result": p["result"],
                    "payout": payout_for(p["result"], stake, px),
                    "clv_pct": round(random.gauss(0.4, 1.6), 2),
                    "closing_price": p["best_price"] - 4,
                    "graded_at": utcnow(), "demo": 1,
                })
                made += 1
    print(f"backfilled {made} demo bets across {len(users)} users")
    return made


def purge() -> dict:
    run_migrations()
    with db() as conn:
        # live_state rows are keyed to demo picks' games — clear them too, or a
        # real game keeps showing a fake score after the demo is gone.
        live = conn.execute(
            "DELETE FROM live_state WHERE game_id IN "
            "(SELECT DISTINCT game_id FROM picks WHERE demo=1)").rowcount
        conn.execute(
            "DELETE FROM live_player_stats WHERE game_id IN "
            "(SELECT DISTINCT game_id FROM picks WHERE demo=1)")
        b = conn.execute("DELETE FROM user_bets WHERE demo=1").rowcount
        p = conn.execute("DELETE FROM picks WHERE demo=1").rowcount
        g = conn.execute("DELETE FROM games WHERE game_id LIKE 'DEMO_%'").rowcount
    print(f"purged {g} games, {p} picks, {b} user bets, {live} live states")
    return {"games": g, "picks": p, "bets": b, "live_states": live}


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
        seed()             # settled history for Track
        seed_pending()     # pending picks for the Picks tab
        seed_live()        # open bets + live scores for My bets
    elif cmd == "pending":
        seed_pending()
    elif cmd == "live":
        seed_live()
    elif cmd == "history":
        seed()
    elif cmd == "purge":
        purge()
    else:
        status()
