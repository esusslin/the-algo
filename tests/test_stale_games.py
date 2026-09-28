"""A bet you cannot place is not an opportunity.

Live on 28 September 2026, week 4. `/api/edges` returned 87 opportunities. Most
were for games played in weeks 1, 2 and 3, and the largest "edge" on the whole
board — 22.6%, thirteen books, dispersion 0.065 — was a Baltimore-Dallas spread
from three weeks earlier.

Nothing deletes from `odds_current` or `fair_prices`. Every price from every
game since week 1 is still in there, frozen at whatever was last polled before
kickoff. Books stop quoting once a game starts, so that final quote is often
wide, stale and one-sided — which is exactly the shape that produces a large
apparent edge.

Every guard passed. Book count: fine. Dispersion: fine. Under
MAX_PLAUSIBLE_EDGE_PCT: yes, at 22.6%. The arithmetic was correct throughout.
The game was over.

That is the same failure as the implausible-edge bug one layer up — the
pipeline working perfectly on data it should have distrusted — and it is worse
in one specific way: those guards ask "is this number believable?", and no
amount of believability checking gets you to "this game finished on the 13th".
The availability question has to be asked first, and nothing was asking it.

**Why no existing test caught it.** Every fixture in the suite either omits the
`games` row entirely or sets a kickoff in 2099. Neither can distinguish a
filter that works from one that was never written.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def shop_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MIN_EDGE_PCT", "2.0")
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()

    yield db_mod

    monkeypatch.undo()
    importlib.reload(config_mod)
    importlib.reload(db_mod)


def _game(db_mod, game_id: str, *, hours: float | None) -> None:
    """`hours` is relative to now; None writes a row with no kickoff at all."""
    from datetime import datetime, timedelta, timezone

    kick = (None if hours is None
            else (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat())
    with db_mod.db() as c:
        c.execute("INSERT OR REPLACE INTO games (game_id, season, week, season_type,"
                  " home_team, away_team, kickoff_utc, status) "
                  "VALUES (?,2026,1,'REG','DAL','BAL',?,'scheduled')",
                  (game_id, kick))


def _priceable(db_mod, game_id: str) -> None:
    """A market with a genuine edge: fair value 50%, one book paying +150.

    Enough books and tight enough dispersion to clear every other guard, so
    anything that filters this out can only be filtering on the game's status.
    """
    now = db_mod.utcnow()
    with db_mod.db() as c:
        c.execute("INSERT OR REPLACE INTO fair_prices (game_id, market_type, "
                  "player_id, side, line, fair_prob, method, sharp_prob, "
                  "book_count, dispersion, computed_at) "
                  "VALUES (?,'spreads','','home',-3.5,0.5,'power',NULL,13,0.02,?)",
                  (game_id, now))
        for book, price in (("draftkings", 150), ("fanduel", -110),
                            ("betmgm", -108), ("caesars", -112),
                            ("betrivers", -110), ("espnbet", -110),
                            ("williamhill_us", -110), ("pointsbetus", -110),
                            ("bovada", -110), ("lowvig", -110),
                            ("betonlineag", -110), ("fanatics", -110),
                            ("mybookieag", -110)):
            c.execute("INSERT INTO odds_current (game_id,market_type,player_id,side,"
                      "line,book,price,updated_at,fetched_at) "
                      "VALUES (?,'spreads','','home',-3.5,?,?,?,?)",
                      (game_id, book, price, now, now))


def _edges(**kw) -> list[dict]:
    from src.market.shop import find_opportunities
    return find_opportunities(min_edge=1.0, **kw)


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------
def test_a_finished_game_produces_no_opportunity(shop_db) -> None:
    """**The test that would have caught 28 September.**

    Identical market, identical prices, identical book count — the only
    difference from the control below is that this one kicked off two hours ago.
    """
    _game(shop_db, "2026_03_BAL_DAL", hours=-2.0)
    _priceable(shop_db, "2026_03_BAL_DAL")

    assert _edges() == [], (
        "a game that has already kicked off is still being offered as a bet. "
        "Books stop quoting at kickoff, so this price is frozen and unfillable.")


def test_an_upcoming_game_still_produces_one(shop_db) -> None:
    """**The control, and it is not optional.** Without it, a filter that
    rejected everything would satisfy the test above and silently empty the
    board — which is a worse outcome than the bug, and harder to notice."""
    _game(shop_db, "2026_04_BAL_DAL", hours=+3.0)
    _priceable(shop_db, "2026_04_BAL_DAL")

    out = _edges()
    assert len(out) == 1, f"a live, priceable market produced no opportunity: {out}"
    assert out[0]["game_id"] == "2026_04_BAL_DAL"
    assert out[0]["edge_pct"] > 1.0


def test_a_game_that_kicked_off_one_minute_ago_is_gone(shop_db) -> None:
    """The boundary, asserted rather than assumed. `status != 'final'` was the
    test that broke the polling throttle in exactly this way on 13 September:
    a game is unavailable the moment it starts, not when someone marks it
    finished."""
    _game(shop_db, "just_started", hours=-1 / 60)
    _priceable(shop_db, "just_started")

    assert _edges() == []


def test_a_game_with_no_kickoff_is_kept(shop_db) -> None:
    """Deliberate, and matching `current_slate`: absence of a kickoff is not
    evidence of kickoff. Excluding on missing data would empty the board any
    week the games table lagged the odds feed — a silent failure with a much
    larger blast radius than the one being fixed."""
    _game(shop_db, "no_kickoff", hours=None)
    _priceable(shop_db, "no_kickoff")

    assert len(_edges()) == 1


def test_a_price_with_no_game_row_at_all_is_kept(shop_db) -> None:
    """The LEFT JOIN, tested properly.

    `test_a_game_with_no_kickoff_is_kept` inserts a games row whose kickoff is
    NULL, which an INNER JOIN also returns — so it passed against a mutation
    that swapped LEFT for INNER, and was not testing what its name claimed.

    This is the case that actually happens: the odds feed knows about a game
    the `games` table has not caught up with yet. An INNER JOIN drops it
    silently, and the board goes thin for a reason nothing reports.
    """
    _priceable(shop_db, "unknown_to_games_table")   # deliberately no _game()

    assert shop_db.query(
        "SELECT * FROM games WHERE game_id='unknown_to_games_table'") == []
    assert len(_edges()) == 1, (
        "a fair price whose game row has not landed yet was dropped — an "
        "INNER JOIN here empties the board whenever the schedule feed lags")


def test_the_filter_can_be_turned_off_for_analysis(shop_db) -> None:
    """Backfills and post-hoc CLV work legitimately want finished games. The
    filter is a default, not a prohibition — but it defaults to ON, because the
    caller that matters is `generate()` and the cost of forgetting is a stale
    pick on the board."""
    _game(shop_db, "2026_01_SF_LA", hours=-400.0)
    _priceable(shop_db, "2026_01_SF_LA")

    assert _edges() == []
    assert len(_edges(upcoming_only=False)) == 1


def test_finished_and_upcoming_together(shop_db) -> None:
    """The live shape: weeks 1-3 done, week 4 ahead, all in one table. Only the
    live one may survive."""
    for gid, hrs in (("2026_01_SF_LA", -400.0), ("2026_02_CAR_ATL", -200.0),
                     ("2026_03_BAL_DAL", -72.0), ("2026_04_LA_PHI", +26.0)):
        _game(shop_db, gid, hours=hrs)
        _priceable(shop_db, gid)

    assert [o["game_id"] for o in _edges()] == ["2026_04_LA_PHI"]
    assert len(_edges(upcoming_only=False)) == 4


# ---------------------------------------------------------------------------
# and the thing that actually matters: picks
# ---------------------------------------------------------------------------
def test_generate_writes_no_pick_for_a_finished_game(shop_db) -> None:
    """`current_slate` already hides started games from the UI, so this was
    invisible — but the pick was still being written, still being repriced on
    every pass, and still being sent to the red team for review at cost."""
    from src.picks.generator import generate

    _game(shop_db, "2026_03_BAL_DAL", hours=-2.0)
    _priceable(shop_db, "2026_03_BAL_DAL")

    generate(source="market_engine")
    rows = shop_db.query("SELECT * FROM picks WHERE game_id='2026_03_BAL_DAL'")
    assert rows == [], f"wrote {len(rows)} pick(s) for a game that already kicked off"
