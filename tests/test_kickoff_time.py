"""Kickoff timestamps are UTC, and were not.

`games.csv` publishes kickoff in Eastern. The ingest wrote it unconverted into a column
named `kickoff_utc`, so every consumer treating it as UTC read a time four or five hours
early.

**The expensive consequence was CLV.** `capture_closing_lines` fires on
`datetime(kickoff_utc) <= datetime('now', '+12 minutes')`, and SQLite's `datetime('now')`
is UTC. For an 8:20pm ET kickoff that came true at about 4:08pm ET — so the price recorded
as "closing" was a late-afternoon price. Closing line value is the headline metric of the
whole system, odds disappear at kickoff, and there is no way to reconstruct the real
closing number afterwards. It had to be right before the games, not after.

Found 9 September 2026, hours before the season opener, while adding an unrelated filter
that would have compared against the same broken timestamp — and would therefore have
made picks vanish from the app four hours early. *A bug found while building on top of it*
is the recurring theme worth noting: the display fix was the thing that made the data bug
visible.

These are pure and offline: no network, no database, no games.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.fetchers.nflverse import et_to_utc


# --- the conversion ------------------------------------------------------------------


def test_a_september_night_game_converts_to_the_next_utc_day() -> None:
    """**The case that was live when this was found.** 8:20pm ET on 9 September is
    00:20 UTC on the 10th — a different calendar day, which is exactly why the old
    value looked plausible and was wrong."""
    assert et_to_utc("2026-09-09", "20:20") == "2026-09-10T00:20:00+00:00"


def test_a_sunday_afternoon_game_shifts_four_hours_under_edt() -> None:
    assert et_to_utc("2026-09-13", "13:00") == "2026-09-13T17:00:00+00:00"


def test_the_offset_changes_after_daylight_saving_ends() -> None:
    """**Why this uses zoneinfo and not `+4 hours`.**

    Eastern is UTC−4 under EDT and UTC−5 under EST, and the switch is the first Sunday
    in November — week nine or so, not a corner case. A hardcoded offset would be right
    tonight and an hour wrong for the back half of every season, which is the half where
    the games matter more.
    """
    assert et_to_utc("2026-10-25", "13:00") == "2026-10-25T17:00:00+00:00"  # EDT, -4
    assert et_to_utc("2026-11-15", "13:00") == "2026-11-15T18:00:00+00:00"  # EST, -5


def test_the_shift_is_never_zero_for_a_timed_game() -> None:
    """The regression guard. If someone 'simplifies' this back to string concatenation,
    the returned value equals the input and this fails."""
    for day, time in (("2026-09-13", "13:00"), ("2026-12-25", "16:30")):
        assert et_to_utc(day, time) != f"{day}T{time}:00"


def test_the_result_is_offset_aware() -> None:
    """The backfill's idempotency check keys on this suffix — a naive string means an
    unconverted row, so the marker has to be present and stable."""
    out = et_to_utc("2026-09-13", "13:00")
    assert out.endswith("+00:00")
    assert datetime.fromisoformat(out).tzinfo == timezone.utc


def test_it_round_trips_to_the_eastern_time_we_started_with() -> None:
    """Independent check, rather than asserting the function against itself."""
    from zoneinfo import ZoneInfo

    out = datetime.fromisoformat(et_to_utc("2026-11-15", "13:00"))
    back = out.astimezone(ZoneInfo("America/New_York"))
    assert (back.hour, back.minute) == (13, 0)


# --- degenerate input: skip, don't invent ---------------------------------------------


def test_a_date_with_no_time_is_left_alone() -> None:
    """A future game with no scheduled time yet. Inventing midnight in some timezone
    would be a worse answer than no answer."""
    assert et_to_utc("2026-09-13", "") == "2026-09-13"


def test_no_date_means_no_timestamp() -> None:
    assert et_to_utc("", "20:20") is None
    assert et_to_utc("", "") is None


def test_unparseable_time_degrades_to_the_date() -> None:
    """One malformed row in games.csv must not take down the whole refresh."""
    assert et_to_utc("2026-09-13", "not-a-time") == "2026-09-13"


# --- the backfill --------------------------------------------------------------------
#
# Running it twice must not shift twice. That is the only way this script can do
# unrecoverable damage, so it is the thing worth pinning hardest.


def test_backfill_detects_an_unconverted_value() -> None:
    from scripts.backfill_kickoff_utc import needs_conversion

    assert needs_conversion("2026-09-09T20:20:00") is True


def test_backfill_skips_an_already_converted_value() -> None:
    """**Idempotency.** A second run is a no-op; a partial run resumes safely."""
    from scripts.backfill_kickoff_utc import needs_conversion

    assert needs_conversion("2026-09-10T00:20:00+00:00") is False
    assert needs_conversion("2026-09-10T00:20:00Z") is False


def test_backfill_skips_bare_dates_and_nulls() -> None:
    from scripts.backfill_kickoff_utc import needs_conversion

    assert needs_conversion("2026-09-13") is False
    assert needs_conversion(None) is False
    assert needs_conversion("") is False


def test_converting_twice_gives_the_same_answer_as_once() -> None:
    """The end-to-end idempotency proof, not just the guard in isolation."""
    from scripts.backfill_kickoff_utc import convert, needs_conversion

    original = "2026-09-09T20:20:00"
    once = convert(original)
    assert once == "2026-09-10T00:20:00+00:00"
    assert not needs_conversion(once)


@pytest.mark.parametrize("value", [
    "2026-09-09T20:20:00",
    "2026-11-15T13:00:00",
    "2026-01-04T16:30:00",
])
def test_every_converted_value_is_recognised_as_done(value: str) -> None:
    from scripts.backfill_kickoff_utc import convert, needs_conversion

    assert needs_conversion(value)
    assert not needs_conversion(convert(value))


# --- the slate filter ----------------------------------------------------------------
#
# Runs against a real in-memory SQLite, because the thing being tested IS the SQL —
# specifically `datetime()` on both sides. A mock would prove nothing about whether
# an offset-aware string compares correctly against `datetime('now')`.


@pytest.fixture
def slate_db(monkeypatch):
    import sqlite3

    from src.picks import generator

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE games (game_id TEXT PRIMARY KEY, kickoff_utc TEXT);"
        "CREATE TABLE picks (pick_id INTEGER PRIMARY KEY, game_id TEXT, "
        "  result TEXT, published INT, tier TEXT, edge_pct REAL);"
    )
    monkeypatch.setattr(
        generator, "query",
        lambda sql, params=(): [dict(r) for r in conn.execute(sql, params).fetchall()],
    )
    return conn


def _seed(conn, *, kickoff, pick_id=1, published=1, result="pending"):
    conn.execute("INSERT OR REPLACE INTO games VALUES (?,?)", (f"G{pick_id}", kickoff))
    conn.execute("INSERT INTO picks VALUES (?,?,?,?,?,?)",
                 (pick_id, f"G{pick_id}", result, published, "A", 6.0))


def test_a_pick_for_a_future_game_is_in_the_slate(slate_db) -> None:
    from src.picks.generator import current_slate

    _seed(slate_db, kickoff="2099-01-01T00:00:00+00:00")
    assert len(current_slate()) == 1


def test_a_pick_for_a_kicked_off_game_is_not(slate_db) -> None:
    """**The bug.** Pending until grading at 03:30, so without this it stayed on
    screen through the whole game."""
    from src.picks.generator import current_slate

    _seed(slate_db, kickoff="2020-01-01T00:00:00+00:00")
    assert current_slate() == []


def test_admin_can_still_see_started_picks(slate_db) -> None:
    from src.picks.generator import current_slate

    _seed(slate_db, kickoff="2020-01-01T00:00:00+00:00")
    assert len(current_slate(include_started=True)) == 1


def test_a_pick_with_no_game_row_still_shows(slate_db) -> None:
    """LEFT JOIN on purpose. A missing game row is ignorance, not evidence that the
    game started — the same rule the red-team agent follows."""
    from src.picks.generator import current_slate

    slate_db.execute("INSERT INTO picks VALUES (9,'GHOST','pending',1,'A',6.0)")
    assert len(current_slate()) == 1


def test_a_pick_with_a_null_kickoff_still_shows(slate_db) -> None:
    from src.picks.generator import current_slate

    _seed(slate_db, kickoff=None)
    assert len(current_slate()) == 1


def test_a_game_that_started_earlier_today_is_excluded(slate_db) -> None:
    """**Why `datetime()` has to wrap both sides — and the case that proves it.**

    `datetime('now')` returns `'2026-09-09 14:30:00'`: a space separator, no offset. A
    kickoff is `'2026-09-09T18:20:00+00:00'`: a `T`, plus a suffix. Compared as raw
    strings they match for the first ten characters and then hit `'T'` (0x54) against
    `' '` (0x20) — so **every kickoff on today's date sorts as later than now**, however
    long ago it actually started.

    Dates far apart hide this completely, which is how the first version of this test
    passed while proving nothing: `2020` vs `2099` gives the right answer under either
    comparison. It has to be a game earlier *today* to tell them apart — and a game
    earlier today is precisely tonight's scenario.
    """
    from datetime import datetime, timezone

    from src.picks.generator import current_slate

    # Midnight UTC today: same calendar date as `now`, and in the past.
    midnight = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat(timespec="seconds")

    _seed(slate_db, kickoff="2099-01-01T00:00:00+00:00", pick_id=1)
    _seed(slate_db, kickoff=midnight, pick_id=2)
    rows = current_slate()
    assert [r["pick_id"] for r in rows] == [1], (
        "a game that kicked off earlier today is still in the slate — the SQL is "
        "comparing strings, not timestamps"
    )


def test_unpublished_picks_stay_hidden_from_users(slate_db) -> None:
    """The pre-existing guarantee, re-pinned — the new JOIN rewrote this query and it
    would have been easy to drop the published clause in passing."""
    from src.picks.generator import current_slate

    _seed(slate_db, kickoff="2099-01-01T00:00:00+00:00", published=0)
    assert current_slate() == []
    assert len(current_slate(include_unpublished=True)) == 1


def test_backfill_is_a_no_op_once_converted(monkeypatch) -> None:
    """**The property that makes running this at every boot safe.**

    the-algo is SQLite and production is a file inside the Railway container, so the
    only way to migrate it without shelling into a running container mid-game is to do
    it on startup. That is only acceptable if a second run cannot shift times again.

    First call converts and reports the count; second call finds nothing and reports
    zero, without writing.
    """
    import sqlite3

    import scripts.backfill_kickoff_utc as bf

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY, kickoff_utc TEXT)")
    conn.execute("INSERT INTO games VALUES ('G1','2026-09-09T20:20:00')")

    class _Ctx:
        def __enter__(self): return conn
        def __exit__(self, *a): return False

    monkeypatch.setattr(bf, "query",
                        lambda sql, params=(): [dict(r) for r in conn.execute(sql, params)])
    monkeypatch.setattr(bf, "db", lambda: _Ctx())

    assert bf.backfill_kickoffs() == 1
    stored = conn.execute("SELECT kickoff_utc FROM games").fetchone()[0]
    assert stored == "2026-09-10T00:20:00+00:00"

    assert bf.backfill_kickoffs() == 0, "a second run must not touch anything"
    assert conn.execute("SELECT kickoff_utc FROM games").fetchone()[0] == stored
