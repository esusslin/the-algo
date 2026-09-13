"""Red-team review caching — and the far more important question of when it MISSES.

Between 9 and 13 September 2026 the red team cost $63.16 across 11,000 calls. Not
one of them was wrong; almost all of them were redundant. `apply_to_picks` selected
picks whose verdict was already OK and wrote OK back, so each one stayed selected
forever and was re-reviewed on every 5-minute odds poll — through the game and
overnight, because `result` stays 'pending' until grading at 03:30. A pick created
Friday for a Sunday game was reviewed roughly 600 times to re-derive an answer we
already had.

**The tests that matter here are the miss tests, not the hit tests.**

A cache that saves money is trivially easy to write: return early, always. It would
pass every test about call volume and it would silently stop reviewing picks, which
is strictly worse than the bug it replaced — the $63 was visible in the ledger
within four days, whereas a pick that was never reviewed looks exactly like a pick
that was reviewed and found fine.

So each input that can change a verdict gets a test proving the fingerprint moves
when it does. Every one was mutation-tested by dropping that field from the payload
and confirming the test fails.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.ai.redteam import review_fingerprint


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
PICK = {
    "pick_id": 1,
    "game_id": "2026_02_BUF_MIA",
    "market_type": "spreads",
    "side": "BUF",
    "line": -3.0,
    "tier": "A",
    "headline": "BUF -3.0",
}


def _ctx(**over) -> dict:
    base = {
        "matchup": "BUF at MIA",
        "injuries": [
            {"team": "BUF", "player_name": "Josh Allen",
             "game_status": "Questionable", "practice_status": "Limited",
             "body_part": "shoulder"},
        ],
        "inactives": [],
        "weather": {"temp_c": 11.0, "wind_kph": 12.0, "wind_gust_kph": 20.0,
                    "precip_mm": 0.0, "snow_mm": 0.0},
        "recent_line_moves": [],
    }
    base.update(over)
    return base


def _moves(first: float, last: float, book: str = "pinnacle") -> list[dict]:
    return [
        {"market_type": "spreads", "side": "BUF", "line": first, "price": -110,
         "book": book, "observed_at": "2026-09-13T10:00:00+00:00"},
        {"market_type": "spreads", "side": "BUF", "line": last, "price": -110,
         "book": book, "observed_at": "2026-09-13T14:00:00+00:00"},
    ]


# ---------------------------------------------------------------------------
# THE MISS TESTS — each proves the cache re-asks when the answer could change
# ---------------------------------------------------------------------------
def test_the_bet_itself_changing_forces_a_new_review() -> None:
    """A repriced pick is a different bet. This is the most obvious miss and the
    one that would be most embarrassing to get wrong."""
    a = review_fingerprint(PICK, _ctx())
    b = review_fingerprint({**PICK, "line": -6.5}, _ctx())
    assert a != b


def test_switching_sides_forces_a_new_review() -> None:
    a = review_fingerprint(PICK, _ctx())
    b = review_fingerprint({**PICK, "side": "MIA"}, _ctx())
    assert a != b


def test_a_new_injury_forces_a_new_review() -> None:
    """The entire reason the red team exists. If this cache hit, the agent would
    keep returning a verdict formed before the quarterback was ruled out."""
    ctx = _ctx()
    worse = _ctx(injuries=ctx["injuries"] + [
        {"team": "MIA", "player_name": "Tyreek Hill", "game_status": "Out",
         "practice_status": "DNP", "body_part": "ankle"}])
    assert review_fingerprint(PICK, ctx) != review_fingerprint(PICK, worse)


def test_an_injury_worsening_forces_a_new_review() -> None:
    """Same player, same list length — only the status changed. A fingerprint
    keyed on player names alone would hit here, which is the subtle version of
    the bug."""
    ctx = _ctx()
    worse = _ctx(injuries=[{**ctx["injuries"][0], "game_status": "Out"}])
    assert review_fingerprint(PICK, ctx) != review_fingerprint(PICK, worse)


def test_practice_status_changing_forces_a_new_review() -> None:
    ctx = _ctx()
    worse = _ctx(injuries=[{**ctx["injuries"][0], "practice_status": "DNP"}])
    assert review_fingerprint(PICK, ctx) != review_fingerprint(PICK, worse)


def test_inactives_being_declared_forces_a_new_review() -> None:
    """Inactives land ~90 minutes before kickoff and are the single most
    decision-relevant thing that arrives late."""
    a = review_fingerprint(PICK, _ctx())
    b = review_fingerprint(PICK, _ctx(inactives=[
        {"team": "BUF", "player_name": "James Cook"}]))
    assert a != b


def test_a_significant_line_move_forces_a_new_review() -> None:
    a = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -3.0)))
    b = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, 0.5)))
    assert a != b


def test_a_move_crossing_the_significance_threshold_forces_a_new_review() -> None:
    """MOVE_THRESHOLD for spreads is 1.0, and the prompt tells the model to treat
    'significant' differently from 'minor'. Crossing that boundary changes what
    the agent is being asked, even though both are small numbers."""
    minor = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -2.5)))
    signif = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -1.5)))
    assert minor != signif


def test_a_move_reversing_direction_forces_a_new_review() -> None:
    """Same magnitude, opposite meaning: 2 points toward the bet and 2 points
    against it must never share a fingerprint."""
    toward = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -5.0)))
    against = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -1.0)))
    assert toward != against


def test_weather_crossing_a_bucket_forces_a_new_review() -> None:
    """Wind is the weather that decides bets. 12 kph to 36 kph is a different game."""
    calm = review_fingerprint(PICK, _ctx())
    windy = review_fingerprint(PICK, _ctx(weather={
        "temp_c": 11.0, "wind_kph": 36.0, "wind_gust_kph": 55.0,
        "precip_mm": 0.0, "snow_mm": 0.0}))
    assert calm != windy


def test_weather_appearing_at_all_forces_a_new_review() -> None:
    """'no forecast available' to a real forecast is new information, and the
    prompt explicitly treats missing data differently from present data."""
    assert (review_fingerprint(PICK, _ctx(weather=None))
            != review_fingerprint(PICK, _ctx()))


# ---------------------------------------------------------------------------
# THE HIT TESTS — the saving, and the noise it must tolerate
# ---------------------------------------------------------------------------
def test_an_identical_question_hits() -> None:
    assert review_fingerprint(PICK, _ctx()) == review_fingerprint(PICK, _ctx())


def test_weather_drifting_within_a_band_does_not_re_review() -> None:
    """11.0C to 11.4C is not a new question. Without banding, every weather
    refresh would invalidate the whole board and the cache would save nothing —
    this is the test that makes the saving real rather than theoretical."""
    a = review_fingerprint(PICK, _ctx())
    b = review_fingerprint(PICK, _ctx(weather={
        "temp_c": 11.4, "wind_kph": 12.3, "wind_gust_kph": 21.0,
        "precip_mm": 0.1, "snow_mm": 0.0}))
    assert a == b


def test_weather_jitter_around_a_band_edge_does_not_thrash() -> None:
    """**This one caught a real design flaw and is why the bands are named
    numbers rather than `round(x / step)`.**

    With a uniform grid there is a boundary every `step` units, so a true value
    that happens to sit on one flips back and forth on every refresh and the
    cache never hits again. It is not hypothetical: a 12 kph wind with the feed's
    normal jitter landed exactly on an 8-unit boundary and cost half the expected
    saving in a 48-hour replay. The first version of this test used 12.0 vs 12.3,
    which straddled nothing and passed for the wrong reason.

    So: sweep a realistic jitter band and require ONE fingerprint, not several.
    """
    seen = set()
    for wind in (10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5):
        seen.add(review_fingerprint(PICK, _ctx(weather={
            "temp_c": 11.0, "wind_kph": wind, "wind_gust_kph": 20.0,
            "precip_mm": 0.0, "snow_mm": 0.0})))
    assert len(seen) == 1, f"{len(seen)} fingerprints across normal feed jitter"


def test_wind_crossing_a_named_band_still_re_reviews() -> None:
    """The other half of the pair. Wide bands are only safe if the boundaries that
    remain are the ones that matter — 14 kph and 30 kph are different games."""
    calm = review_fingerprint(PICK, _ctx(weather={
        "temp_c": 11.0, "wind_kph": 14.0, "wind_gust_kph": 20.0,
        "precip_mm": 0.0, "snow_mm": 0.0}))
    windy = review_fingerprint(PICK, _ctx(weather={
        "temp_c": 11.0, "wind_kph": 30.0, "wind_gust_kph": 20.0,
        "precip_mm": 0.0, "snow_mm": 0.0}))
    assert calm != windy


def test_new_odds_quotes_that_do_not_move_the_line_do_not_re_review() -> None:
    """Books requote constantly at the same number. The prompt itself tells the
    model to use the stated NET MOVE and ignore the individual quotes, so the
    cache keys on the same thing the reviewer is told to look at."""
    a = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -3.0)))
    extra = _moves(-3.0, -3.0) + [
        {"market_type": "spreads", "side": "BUF", "line": -3.0, "price": -108,
         "book": "pinnacle", "observed_at": "2026-09-13T15:00:00+00:00"}]
    assert a == review_fingerprint(PICK, _ctx(recent_line_moves=extra))


def test_injury_list_order_does_not_re_review() -> None:
    """The query orders by status severity, which can reshuffle rows that mean
    the same thing. Order is not information here."""
    two = [
        {"team": "BUF", "player_name": "Josh Allen", "game_status": "Questionable",
         "practice_status": "Limited", "body_part": "shoulder"},
        {"team": "MIA", "player_name": "Jaylen Waddle", "game_status": "Questionable",
         "practice_status": "Limited", "body_part": "knee"},
    ]
    assert (review_fingerprint(PICK, _ctx(injuries=two))
            == review_fingerprint(PICK, _ctx(injuries=list(reversed(two)))))


def test_a_move_on_another_market_does_not_re_review() -> None:
    """This pick is a spread. The total moving is somebody else's problem."""
    a = review_fingerprint(PICK, _ctx(recent_line_moves=_moves(-3.0, -3.0)))
    noise = _moves(-3.0, -3.0) + [
        {"market_type": "totals", "side": "over", "line": 44.0, "price": -110,
         "book": "pinnacle", "observed_at": "2026-09-13T11:00:00+00:00"},
        {"market_type": "totals", "side": "over", "line": 49.0, "price": -110,
         "book": "pinnacle", "observed_at": "2026-09-13T15:00:00+00:00"},
    ]
    assert a == review_fingerprint(PICK, _ctx(recent_line_moves=noise))


# ---------------------------------------------------------------------------
# apply_to_picks — selection, and the outage that must not be cached
# ---------------------------------------------------------------------------
@pytest.fixture
def picks_db(monkeypatch):
    from src.ai import redteam

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE picks (pick_id INTEGER PRIMARY KEY, game_id TEXT,"
        "  market_type TEXT, side TEXT, line REAL, tier TEXT, headline TEXT,"
        "  edge_pct REAL, result TEXT, published INTEGER, ai_verdict TEXT,"
        "  ai_reason TEXT, review_hash TEXT, reviewed_at TEXT);"
        "CREATE TABLE games (game_id TEXT PRIMARY KEY, kickoff_utc TEXT);"
    )

    class _Ctx:
        def __enter__(self): return conn
        def __exit__(self, *a): return False

    monkeypatch.setattr(redteam, "query",
                        lambda sql, params=(): [dict(r) for r in conn.execute(sql, params)])
    monkeypatch.setattr(redteam, "db", lambda: _Ctx())
    monkeypatch.setattr(redteam, "_game_context", lambda gid: _ctx())
    monkeypatch.setattr(redteam.settings, "ENABLE_AI_REDTEAM", True, raising=False)
    return conn


def _add_pick(conn, pick_id=1, kickoff="2099-01-01T00:00:00+00:00",
              verdict="", review_hash=None) -> None:
    conn.execute("INSERT OR REPLACE INTO games VALUES (?,?)",
                 (PICK["game_id"], kickoff))
    conn.execute(
        "INSERT INTO picks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pick_id, PICK["game_id"], "spreads", "BUF", -3.0, "A", "BUF -3.0",
         8.0, "pending", 1, verdict, "", review_hash, None))


def _spy(monkeypatch):
    """Count reviews that would have cost money."""
    from src.ai import redteam

    calls: list[int] = []

    def fake(pick, ctx=None):
        calls.append(pick["pick_id"])
        return {"verdict": "OK", "reason": "fine", "evidence": "", "source": "model"}

    monkeypatch.setattr(redteam, "review_pick", fake)
    return calls


def test_a_never_reviewed_pick_is_always_reviewed(picks_db, monkeypatch) -> None:
    from src.ai.redteam import apply_to_picks

    calls = _spy(monkeypatch)
    _add_pick(picks_db)
    assert apply_to_picks()["reviewed"] == 1
    assert calls == [1]


def test_an_unchanged_pick_is_not_reviewed_twice(picks_db, monkeypatch) -> None:
    """The saving, end to end: run it four times, pay once."""
    from src.ai.redteam import apply_to_picks

    calls = _spy(monkeypatch)
    _add_pick(picks_db)
    for _ in range(4):
        apply_to_picks()
    assert len(calls) == 1, f"{len(calls)} reviews for one unchanged pick"


def test_a_moved_line_is_reviewed_again(picks_db, monkeypatch) -> None:
    """**The single most important test in this file.** If this passes because
    the cache never hits, `test_an_unchanged_pick_is_not_reviewed_twice` fails —
    the two together pin the behaviour from both sides."""
    from src.ai import redteam

    calls = _spy(monkeypatch)
    _add_pick(picks_db)
    redteam.apply_to_picks()

    picks_db.execute("UPDATE picks SET line=-6.5 WHERE pick_id=1")
    redteam.apply_to_picks()
    assert len(calls) == 2


def test_a_started_game_is_not_reviewed(picks_db, monkeypatch) -> None:
    """Grading runs at 03:30, so a Sunday pick stayed 'pending' and got reviewed
    through its own game and overnight. The bet is not placeable; the review
    cannot change an outcome."""
    from src.ai.redteam import apply_to_picks

    calls = _spy(monkeypatch)
    _add_pick(picks_db, kickoff="2020-01-01T00:00:00+00:00")
    assert apply_to_picks()["reviewed"] == 0
    assert calls == []


def test_a_pick_with_no_game_row_is_still_reviewed(picks_db, monkeypatch) -> None:
    """Missing kickoff must not silently exclude a pick — unknown errs toward
    reviewing, never toward skipping."""
    from src.ai.redteam import apply_to_picks

    calls = _spy(monkeypatch)
    _add_pick(picks_db)
    picks_db.execute("DELETE FROM games")
    assert apply_to_picks()["reviewed"] == 1
    assert calls == [1]


def test_a_failed_review_is_never_cached(picks_db, monkeypatch) -> None:
    """**The failure this change could have introduced.** A fail-open OK during
    an outage is the absence of an answer. Caching its fingerprint would mark the
    pick reviewed and suppress every future attempt — turning a transient API
    outage into a permanent silent non-review. That is 9 September with a memory.
    """
    from src.ai import redteam

    outage: list[int] = []
    monkeypatch.setattr(redteam, "review_pick", lambda pick, ctx=None: (
        outage.append(pick["pick_id"]) or
        {"verdict": "OK", "reason": "", "evidence": "",
         "source": "unavailable", "error": "credit balance too low"}))
    _add_pick(picks_db)

    redteam.apply_to_picks()
    assert picks_db.execute(
        "SELECT review_hash FROM picks WHERE pick_id=1").fetchone()[0] is None

    redteam.apply_to_picks()
    assert len(outage) == 2, "an outage must not suppress the next attempt"


def test_force_reviews_everything(picks_db, monkeypatch) -> None:
    """Escape hatch for prompt changes. The fingerprint covers inputs, not the
    prompt text, so editing SYSTEM needs a deliberate full pass."""
    from src.ai import redteam

    calls = _spy(monkeypatch)
    _add_pick(picks_db)
    redteam.apply_to_picks()
    redteam.apply_to_picks(force=True)
    assert len(calls) == 2


def test_skipped_count_is_reported(picks_db, monkeypatch) -> None:
    """A cache whose hit rate you cannot see is indistinguishable from a reviewer
    that stopped working."""
    from src.ai import redteam

    _spy(monkeypatch)
    _add_pick(picks_db)
    redteam.apply_to_picks()
    assert redteam.apply_to_picks()["skipped"] == 1
