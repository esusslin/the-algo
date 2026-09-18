"""A pick must reflect the market now, not the market when it was born.

Live on 13 September 2026: 19 of 74 published picks showed a positive edge in
the UI while their current edge was NEGATIVE. One displayed "5.6% edge" on a bet
that had become -3.1%. Another showed 2.4% on a bet worth -10.1%.

`generate()` skipped any market that already had a pending pick, so the numbers
were frozen at creation and the pick stayed on screen, unchanged, until grading.
Prices move all day. No job failed. Nothing in `/health` could see it, because
nothing was broken — the pipeline was faithfully doing what it was told.

The stale number is the one a user acts on, which makes this worse than the
bugs that produced wrong numbers: those were caught by a reviewer, this was
invisible by construction.

**Repricing may only ever withdraw.** It can move numbers and pull a pick from
view; it can never promote one, republish one, or raise a tier. Anything that
would increase a user's confidence has to go through a fresh pick and a fresh
red-team review.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def gen_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MIN_EDGE_PCT", "2.0")
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    with db_mod.db() as c:
        c.execute("INSERT INTO games (game_id, season, week, season_type, "
                  "home_team, away_team, kickoff_utc, status) VALUES "
                  "('g1',2026,2,'REG','ATL','MIN','2099-01-01T00:00:00+00:00','scheduled')")
    return db_mod


def _quotes(db_mod, rows) -> None:
    """Replace the whole book, the way a poll does."""
    now = db_mod.utcnow()
    with db_mod.db() as c:
        c.execute("DELETE FROM odds_current")
        for book, side, price in rows:
            c.execute(
                "INSERT INTO odds_current (game_id,market_type,player_id,side,line,"
                "book,price,updated_at,fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("g1", "h2h", "", side, 0.0, book, price, now, now))


# A soft book paying well over consensus: a real, generous edge on 'home'.
_CONSENSUS = ["pinnacle", "draftkings", "fanduel", "caesars",
              "bovada", "betrivers", "espnbet", "williamhill_us"]
_TIGHT = [(b, s, p) for b in _CONSENSUS for s, p in (("home", -115), ("away", -105))]

# Eight books agreeing, plus a ninth paying well over. Eight is not incidental:
# game-class tier C requires 8 books, so a smaller fixture produces no pick at
# all and every assertion below would fail for the wrong reason.
GOOD = _TIGHT + [("betmgm", "home", 120), ("betmgm", "away", -145)]

# Same market after the soft book corrects itself. No edge left anywhere.
CORRECTED = _TIGHT + [("betmgm", "home", -125), ("betmgm", "away", -101)]


def _pick(db_mod):
    rows = db_mod.query("SELECT * FROM picks WHERE side='home'")
    return dict(rows[0]) if rows else None


def test_the_edge_shown_follows_the_market(gen_db) -> None:
    """**The regression.** Create a pick, let the price correct, regenerate —
    the stored edge must move with it rather than stay frozen."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")
    before = _pick(gen_db)
    assert before and before["edge_pct"] > 2.0

    _quotes(gen_db, CORRECTED)
    build_fair_prices()
    out = generate(source="market_engine")

    after = _pick(gen_db)
    assert after["pick_id"] == before["pick_id"], "a duplicate row was inserted"
    assert after["edge_pct"] < before["edge_pct"], (
        f"edge frozen at {before['edge_pct']:.1f}% after the market corrected")
    assert out["withdrawn"] >= 1


def test_a_pick_that_stops_qualifying_is_withdrawn(gen_db) -> None:
    """It must leave the UI, not merely show a worse number. `published=0` is
    what `current_slate` filters on."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["published"] == 1

    _quotes(gen_db, CORRECTED)
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["published"] == 0


def test_the_displayed_price_and_book_follow_too(gen_db) -> None:
    """`best_book` and `best_price` are the instruction on the card. A correct
    edge next to a price nobody is offering is its own kind of lie."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["best_price"] == 120

    _quotes(gen_db, CORRECTED)
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["best_price"] != 120


# ---------------------------------------------------------------------------
# repricing may only withdraw
# ---------------------------------------------------------------------------
def test_a_withdrawn_pick_is_never_republished(gen_db) -> None:
    """**The rule that keeps this safe.** If the price came back, republishing
    silently would return a pick to the board carrying a red-team verdict formed
    against different numbers. A new pick and a new review, or nothing."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")

    _quotes(gen_db, CORRECTED)
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["published"] == 0

    _quotes(gen_db, GOOD)          # the soft price returns
    build_fair_prices()
    generate(source="market_engine")
    assert _pick(gen_db)["published"] == 0, "a reprice republished a withdrawn pick"


def test_repricing_does_not_change_tier(gen_db) -> None:
    """A red-team FLAG caps a pick at B. If a reprice recomputed tier it would
    silently undo that, and the agent never revisits an OK verdict."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")
    with gen_db.db() as c:
        c.execute("UPDATE picks SET tier='B', ai_verdict='FLAG'")

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")
    p = _pick(gen_db)
    assert p["tier"] == "B", f"reprice raised the tier to {p['tier']}"
    assert p["ai_verdict"] == "FLAG"


def test_a_still_good_pick_survives(gen_db) -> None:
    """**The control.** Without it, code that withdrew everything on every pass
    would satisfy every test above and quietly empty the board."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")

    build_fair_prices()
    out = generate(source="market_engine")
    assert _pick(gen_db)["published"] == 1
    assert out["withdrawn"] == 0
    assert out["repriced"] >= 1


def test_an_unquoted_market_is_left_alone(gen_db) -> None:
    """A transient feed gap is not a verdict. Withdrawing on absence would make
    the board flicker every time a book went quiet for a poll."""
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _quotes(gen_db, GOOD)
    build_fair_prices()
    generate(source="market_engine")

    with gen_db.db() as c:
        c.execute("DELETE FROM odds_current")
    generate(source="market_engine")
    assert _pick(gen_db)["published"] == 1
