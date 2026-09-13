"""Props must survive the pipeline, not just enter it.

On 13 September 2026 the props tab was empty while the database held 20,573 prop
quotes across 9 markets, 6,936 of them successfully priced, 56 clearing the edge
floor. Zero picks were written. No error, no warning, no failed job — every
stage reported success and the last one produced nothing.

The cause: `find_opportunities` takes `min_books: int = 8` and `generate()` did
not override it. That default is a game-market number. Props are quoted by far
fewer books — the live ceiling was 7, the median 5 — so every prop was discarded
before `assign_tier` ran, and the whole per-class scaling in `CLASS_BOOK_SCALE`,
which exists specifically so props clear a different bar, was unreachable code.

Two independent definitions of "enough books", and the one that ran first didn't
know what a prop was.

These tests assert the property that was violated: a prop with realistic book
coverage and a real edge must come out the far end as a pick.
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def prop_db(monkeypatch, tmp_path):
    """Real schema, real SQL. The bug lived in a default argument between two
    real queries; mocks would have reproduced neither."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    return db_mod


def _quote(db_mod, book: str, side: str, price: int, *,
           market="player_reception_yds", player="p_hill", line=64.5,
           game="g1") -> None:
    now = db_mod.utcnow()
    with db_mod.db() as c:
        c.execute(
            "INSERT INTO odds_current (game_id,market_type,player_id,side,line,"
            "book,price,updated_at,fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (game, market, player, side, line, book, price, now, now))


def _seed_priceable_prop(db_mod, books: list[str], over: int, under: int) -> None:
    """A two-sided prop quoted by `books`, plus one soft book posting a
    generous over — the mispricing the system exists to find."""
    for b in books:
        _quote(db_mod, b, "over", over)
        _quote(db_mod, b, "under", under)
    _quote(db_mod, "betmgm", "over", 130)      # the outlier we want to bet
    _quote(db_mod, "betmgm", "under", -160)


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------
def test_a_prop_with_five_books_produces_a_pick(prop_db) -> None:
    """**The test that would have caught 13 September.**

    Five books is normal prop coverage — it was the live median. Before the fix
    `find_opportunities` required eight and this returned nothing at all.
    """
    from src.market.consensus import build_fair_prices
    from src.picks.generator import generate

    _seed_priceable_prop(prop_db, ["pinnacle", "draftkings", "fanduel",
                                   "caesars", "pointsbetus"], -140, 120)
    assert build_fair_prices() > 0, "nothing priced — the fixture is wrong"

    out = generate(source="market_engine")
    assert out["written"] > 0, (
        f"five-book prop produced no picks: {out}. This is the exact shape of "
        f"the original bug — priced, edged, and silently discarded.")


def test_the_prefilter_cannot_be_stricter_than_any_tier(prop_db) -> None:
    """The invariant, stated directly.

    `find_opportunities` runs before market class is known, so whatever it
    demands must be no stricter than the most permissive tier of the most
    permissive class. Otherwise a rule in `assign_tier` is unreachable and
    nothing says so.
    """
    from src.picks.generator import (CLASS_BOOK_SCALE, loosest_min_books,
                                     thresholds_for)
    from src.markets import describe_market

    prefilter = loosest_min_books()
    seen = set()
    for market in ("h2h", "spreads", "totals", "player_reception_yds",
                   "player_rush_yds", "player_pass_yds"):
        cls = describe_market(market).bet_class
        seen.add(cls)
        for tier in ("A", "B", "C"):
            need = thresholds_for(market, tier)["min_books"]
            assert prefilter <= need, (
                f"prefilter demands {prefilter} books but {market} tier {tier} "
                f"only needs {need} — that tier can never be reached")
    assert "prop" in seen, "fixture no longer covers the prop class"
    assert len(CLASS_BOOK_SCALE) >= 2


def test_generate_does_not_use_the_game_market_default(prop_db) -> None:
    """Pins the call itself. A future edit that drops the argument would
    reintroduce the bug while every other test here still passed, because they
    all seed enough books to survive either way."""
    import inspect

    from src.market.shop import find_opportunities
    from src.picks import generator

    assert inspect.signature(find_opportunities).parameters["min_books"].default == 8
    src = inspect.getsource(generator.generate)
    assert "min_books=" in src, (
        "generate() no longer passes min_books, so it inherits the game-market "
        "default of 8 and every prop is discarded before tiering")


# ---------------------------------------------------------------------------
# the controls
#
# The first draft of these asserted "no picks" end to end on fixtures that had
# NEGATIVE edge — they would have passed against any code at all, including code
# with every guard deleted. Confirmed by mutation: removing the book-count check
# and removing the devig pairing guard both left them green.
#
# So each now asserts at the level where its guard actually lives, on a fixture
# that would otherwise sail through.
# ---------------------------------------------------------------------------
def test_a_one_sided_market_is_never_priced(prop_db) -> None:
    """Anytime touchdown is posted yes-only by most books — live, 3,360 of 3,360
    quotes were one-sided. A 'fair price' derived from one side is just the vig
    with a percent sign on it, so nothing may be written at all."""
    from src.market.consensus import build_fair_prices

    for b in ["draftkings", "fanduel", "betmgm", "caesars", "pinnacle"]:
        _quote(prop_db, b, "yes", -140, market="player_anytime_td",
               player="p_chase", line=0.5)

    assert build_fair_prices() == 0, (
        "a one-sided market produced a fair price — devig needs both sides")


def test_assign_tier_rejects_a_prop_with_too_few_books(prop_db) -> None:
    """Direct, because end to end this is caught by two guards at once and the
    test could not tell which. A 25% edge on three books is exactly the fake
    edge thin markets manufacture, and no tier may accept it."""
    from src.picks.generator import assign_tier, thresholds_for

    thin = {"market_type": "player_reception_yds", "side": "over",
            "edge_pct": 25.0, "book_count": 3, "dispersion": 0.01,
            "anchor": "consensus"}
    assert assign_tier(thin) is None, "3 books cleared a tier"

    # Control on the control: identical opportunity, one more book, must pass.
    # Without this, a tier assigner that rejected everything would look correct.
    assert assign_tier({**thin, "book_count": 4}) == "C"
    assert thresholds_for("player_reception_yds", "C")["min_books"] == 4


def test_an_implausible_edge_is_rejected_however_many_books(prop_db) -> None:
    """`MAX_CREDIBLE_DISPERSION`. Books disagreeing that violently means a stale
    quote or a bad parse, not free money — and a fake 40% edge would otherwise
    sit at the very top of a list now sortable by value."""
    from src.picks.generator import MAX_CREDIBLE_DISPERSION, assign_tier

    opp = {"market_type": "player_reception_yds", "side": "over",
           "edge_pct": 40.0, "book_count": 8,
           "dispersion": MAX_CREDIBLE_DISPERSION + 0.01, "anchor": "consensus"}
    assert assign_tier(opp) is None
    assert assign_tier({**opp, "dispersion": 0.02}) is not None
