"""A broken quote must never become a vote.

Found live on 13 September 2026, and found by the red-team agent rather than by
any code: it objected to a pick citing "odd line movement showing home/away
moneyline both moving to extreme -10000 on betfair_ex_".

An exchange posting -10000 on BOTH sides is a 49.5% hold. That is not an
expensive price, it is a market with no liquidity or a parse error. Devig cannot
tell the difference — it normalises the pair to a clean 50/50 and returns
something indistinguishable from a real opinion.

It does two things, both silent:

  * drags the weighted median toward a coin flip, INFLATING the apparent edge on
    the underdog side of a lopsided game;
  * increments `book_count`, which is how a pick clears its tier threshold.

So one broken quote both manufactures the edge and helps it qualify. The agent
was the only thing between that and the feed, and an agent that fails open is
not a control.
"""

from __future__ import annotations

import pytest

from src.market.consensus import (MAX_CREDIBLE_HOLD_PCT, _weighted_median,
                                  build_fair_prices)
from src.market.devig import american_to_prob, devig, hold_pct


def test_the_broken_quote_devigs_to_a_convincing_lie() -> None:
    """Why a guard is needed at all: devig gives no signal that anything is wrong."""
    raw = [american_to_prob(-10000), american_to_prob(-10000)]
    assert hold_pct(raw) == pytest.approx(49.5, abs=0.1)
    assert devig(raw) == pytest.approx([0.5, 0.5])   # looks like a real opinion


def test_it_is_rejected() -> None:
    assert hold_pct([american_to_prob(-10000)] * 2) > MAX_CREDIBLE_HOLD_PCT


@pytest.mark.parametrize("prices,label", [
    ((-110, -110), "standard juice"),
    ((-105, -115), "reduced juice"),
    ((-450, 350), "lopsided but real"),
    ((-2000, 1200), "heavy favourite"),
    ((120, -140), "a normal prop"),
    ((-130, 100), "thin prop, wide but honest"),
])
def test_real_prices_survive(prices, label) -> None:
    """**The control, and the point of the whole test file.**

    A guard that rejected honest quotes would quietly shrink every consensus and
    look exactly like working code. Heavy favourites and wide prop markets are
    where a careless threshold would bite, so they are named explicitly."""
    raw = [american_to_prob(p) for p in prices]
    assert hold_pct(raw) <= MAX_CREDIBLE_HOLD_PCT, (
        f"{label} {prices} rejected as broken — hold {hold_pct(raw):.1f}%")


def test_one_broken_book_moves_the_consensus() -> None:
    """The damage, quantified. Three books agree an outcome is ~81%; the broken
    quote votes 50% and is counted as a book."""
    honest = [0.8130, 0.8140, 0.8125]
    clean = _weighted_median(honest, [1, 1, 1])
    polluted = _weighted_median(honest + [0.5], [1, 1, 1, 1])
    assert polluted < clean
    assert len(honest) + 1 == 4        # and book_count would read 4, not 3


def _seed(db_mod, quotes) -> None:
    now = db_mod.utcnow()
    with db_mod.db() as c:
        for book, side, price in quotes:
            c.execute(
                "INSERT INTO odds_current (game_id,market_type,player_id,side,line,"
                "book,price,updated_at,fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("g1", "h2h", "", side, 0.0, book, price, now, now))


@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    return db_mod


def test_end_to_end_the_broken_book_is_not_counted(fresh_db) -> None:
    """book_count must exclude it. That number gates tier assignment, so a
    broken quote helping a pick qualify is the expensive half of this bug."""
    from src.db import query

    _seed(fresh_db, [
        ("pinnacle", "home", -450), ("pinnacle", "away", 350),
        ("draftkings", "home", -440), ("draftkings", "away", 340),
        ("fanduel", "home", -455), ("fanduel", "away", 355),
        ("betfair_ex_uk", "home", -10000), ("betfair_ex_uk", "away", -10000),
    ])
    assert build_fair_prices() > 0

    row = query("SELECT book_count, fair_prob FROM fair_prices WHERE side='home'")[0]
    assert row["book_count"] == 3, (
        f"book_count={row['book_count']} — the -10000 quote was counted")
    assert row["fair_prob"] > 0.75, (
        f"fair_prob={row['fair_prob']:.4f} — dragged toward a coin flip")


def test_end_to_end_an_honest_market_is_untouched(fresh_db) -> None:
    """Control on the end-to-end test. Same shape, no broken quote: all three
    books must still count, or the guard is eating real data."""
    from src.db import query

    _seed(fresh_db, [
        ("pinnacle", "home", -450), ("pinnacle", "away", 350),
        ("draftkings", "home", -440), ("draftkings", "away", 340),
        ("fanduel", "home", -455), ("fanduel", "away", 355),
    ])
    build_fair_prices()
    assert query("SELECT book_count FROM fair_prices WHERE side='home'")[0]["book_count"] == 3
