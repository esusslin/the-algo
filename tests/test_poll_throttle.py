"""The adaptive polling throttle must actually throttle.

Live on 13 September 2026, `/health` reported:

    odds:featured  ok  interval=5m  soonest=-62.3h

Negative. `poll()` selected games by `status != 'final'`, which is not the same
test as "has not kicked off yet". Thursday night's game had finished and was
never marked final, so it stayed in the set and `min()` returned its hours-to-
kickoff: 62 hours in the past, while Sunday's 1pm games were two hours away.

`interval_minutes` walks its buckets ascending and returns the first whose bound
is >= the value, so ANY negative number matches the tightest bucket. Every tier
polled at maximum rate all season and reported `ok` the entire time.

Cost, which is the part that bites: the tier schedule exists so a market is
polled rarely when kickoff is days out. Burn was tracking ~230k credits a month
against a 100k budget, and the credit ladder sheds PROPS first — the markets
that had only just started producing picks.
"""

from __future__ import annotations

import pytest

from src.fetchers.odds_api import TIERS


def _tier(name: str):
    t = next((t for t in TIERS if t.name == name), None)
    assert t is not None, f"tier {name} no longer exists"
    return t


# ---------------------------------------------------------------------------
# the mechanism that made a wrong number dangerous
# ---------------------------------------------------------------------------
def test_a_negative_hours_to_kick_selects_the_tightest_bucket() -> None:
    """**Why the bad input was silent rather than loud.**

    It does not raise, return None, or skip. It picks maximum polling — the most
    expensive possible answer — and looks exactly like a busy game day."""
    props = _tier("props")
    tightest = props.schedule[min(props.schedule)]
    assert props.interval_minutes(-62.3) == tightest
    assert props.interval_minutes(-0.1) == tightest


def test_the_schedule_still_widens_as_kickoff_recedes() -> None:
    """The property the throttle exists for. Without this a schedule that
    returned one constant would satisfy every other test here."""
    featured = _tier("featured")
    intervals = [featured.interval_minutes(h) for h in (1, 6, 24, 100)]
    present = [i for i in intervals if i is not None]
    assert present == sorted(present), f"intervals do not widen: {intervals}"
    assert len(set(present)) > 1, "schedule is effectively a single interval"


# ---------------------------------------------------------------------------
# the fix: a started game is not an upcoming game
# ---------------------------------------------------------------------------
@pytest.fixture
def poll_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    return db_mod


def _game(db_mod, game_id: str, kickoff: str, status: str = "scheduled") -> None:
    with db_mod.db() as c:
        c.execute("INSERT INTO games (game_id, season, week, season_type, "
                  "home_team, away_team, kickoff_utc, status) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (game_id, 2026, 2, "REG", "ATL", "MIN", kickoff, status))


def _soonest(db_mod) -> float | None:
    """Reproduces poll()'s own selection, so this tests the shipped query."""
    import inspect
    import re

    from src.fetchers import odds_api

    src = inspect.getsource(odds_api.poll)
    m = re.search(r'upcoming = query\(\s*(".*?")\s*,', src, re.S)
    assert m, "poll() no longer selects `upcoming` the way this test expects"
    sql = "".join(re.findall(r'"([^"]*)"', m.group(1)))

    rows = db_mod.query(sql, (2026,))
    if not rows:
        return None
    return min(odds_api._hours_to_kick(r["kickoff_utc"]) for r in rows)


def test_a_finished_but_unmarked_game_is_excluded(poll_db) -> None:
    """**The regression.** Thursday's game, never marked final, alongside a real
    upcoming kickoff. `soonest` must describe the game that has not started."""
    _game(poll_db, "thursday", "2020-01-01T00:00:00+00:00", status="scheduled")
    _game(poll_db, "sunday", "2099-01-01T00:00:00+00:00", status="scheduled")

    soonest = _soonest(poll_db)
    assert soonest is not None and soonest > 0, (
        f"soonest={soonest} — a started game is still setting the interval")


def test_soonest_is_never_negative(poll_db) -> None:
    """Stated as the invariant. However the query is written, a past kickoff
    must not reach `interval_minutes`."""
    for i in range(4):
        _game(poll_db, f"past{i}", f"20{20+i}-01-01T00:00:00+00:00")
    _game(poll_db, "future", "2099-01-01T00:00:00+00:00")
    assert _soonest(poll_db) > 0


def test_nothing_upcoming_means_no_polling(poll_db) -> None:
    """The other end. Once every game has started there is nothing to price, and
    polling anyway is pure credit burn."""
    _game(poll_db, "past", "2020-01-01T00:00:00+00:00")
    assert _soonest(poll_db) is None


def test_an_upcoming_game_is_still_found(poll_db) -> None:
    """The control. A filter that excluded everything would pass both tests
    above and stop the system polling entirely — silently, since `poll()`
    returns cleanly on an empty set."""
    _game(poll_db, "soon", "2099-01-01T00:00:00+00:00")
    assert _soonest(poll_db) is not None


def test_a_final_game_is_still_excluded(poll_db) -> None:
    """The original filter had a reason; adding the kickoff test must not drop
    it. A finished game with a future kickoff string is corrupt data, and the
    safe reading is 'do not price this'."""
    _game(poll_db, "done", "2099-01-01T00:00:00+00:00", status="final")
    assert _soonest(poll_db) is None
