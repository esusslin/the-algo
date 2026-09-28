"""The manual props refresh: it must poll when the scheduler won't, and stop.

Two opposite failures, both silent, both worth a test each.

**It must actually override.** The credit ladder sheds props first — below 30%
of the monthly budget `allows("props")` returns False and the poll loop skips
the tier with nothing but a log line. That is right for an unattended job and
useless to a person staring at an empty props tab an hour before kickoff. If the
override doesn't reach past that check, the button returns 200, queues a job,
runs it to success, and polls nothing: a green pipeline over an empty pipe with
a button on it. `test_a_manual_poll_runs_when_the_scheduler_would_shed` is the
one that catches that, and it is the reason this file exists.

**It must still stop.** The override turns off the governor, so the daily
ceiling is the only thing standing between a stuck finger and a spent month.
Checked before the job is queued, so an unaffordable press is refused with a
number rather than queued and silently shed downstream.

Mutation-checked, both directions:
  * drop `and not override_budget` from the shed check -> the first test fails.
  * drop the ceiling check in `props_refresh` -> the ceiling tests fail.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ledger_db(monkeypatch, tmp_path):
    """Small budget so the shed threshold is reachable, then put it back.

    `monkeypatch.setenv` restores the environment, but `settings` is read once
    at import and cached on the module object — so reloading `src.config` here
    leaves a 1,000-credit budget installed for every test that runs afterwards.
    It did: two budget assertions in `test_market_math` failed in the full suite
    and passed in isolation, which is the signature of exactly this.

    Reloading on the way out is the whole point of the yield below. A fixture
    that mutates module-level global state and doesn't undo it isn't a fixture,
    it's a time bomb with a decorator on it.
    """
    import importlib

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("ODDS_API_KEY", "test")
    monkeypatch.setenv("ODDS_MONTHLY_CREDIT_BUDGET", "1000")
    monkeypatch.setenv("ODDS_MANUAL_DAILY_CREDITS", "500")
    monkeypatch.setenv("ENABLE_PROPS", "true")

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    from src.fetchers import odds_api as odds_mod
    importlib.reload(odds_mod)

    yield db_mod, odds_mod

    monkeypatch.undo()
    importlib.reload(config_mod)
    importlib.reload(db_mod)
    importlib.reload(odds_mod)


def _spend(db_mod, credits: int, source: str = "scheduler", when: str | None = None) -> None:
    with db_mod.db() as c:
        db_mod.insert_row(c, "odds_credit_ledger", {
            "endpoint": "/test", "markets": "x", "credits_used": credits,
            "called_at": when or db_mod.utcnow(), "source": source})


def _game(db_mod, hours_out: float = 3.0, game_id: str = "g1",
          kickoff_utc: str | None = None) -> None:
    from datetime import datetime, timedelta, timezone
    kick = kickoff_utc or (
        datetime.now(timezone.utc) + timedelta(hours=hours_out)).isoformat()
    with db_mod.db() as c:
        c.execute("INSERT OR REPLACE INTO games (game_id, season, week, season_type, "
                  "home_team, away_team, kickoff_utc, status, odds_api_event_id) "
                  "VALUES (?,2026,4,'REG','PHI','CHI',?,'scheduled',?)",
                  (game_id, kick, "evt_" + game_id))


def _at_local(odds_mod, hour: int, minute: int = 0, day_offset: int = 0) -> str:
    """A kickoff at a given LOCAL (settings.TZ) wall-clock time, as UTC ISO.

    Written this way so the Sunday-night test below says what it means — 8:20pm
    Eastern — rather than a UTC timestamp the reader has to convert in their
    head to see the point.
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(odds_mod.settings.TZ)
    local = (datetime.now(tz) + timedelta(days=day_offset)).replace(
        hour=hour, minute=minute, second=0, microsecond=0)
    return local.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# the regression this whole feature turns on
# ---------------------------------------------------------------------------
def test_a_manual_poll_runs_when_the_scheduler_would_shed(ledger_db, monkeypatch) -> None:
    """**The test that makes the button worth building.**

    Budget deliberately exhausted past the props threshold. The scheduler must
    skip; the manual override must not.
    """
    db_mod, odds_mod = ledger_db
    _game(db_mod)
    _spend(db_mod, 900)                      # 10% of 1000 remaining -> props shed

    ledger = odds_mod.CreditLedger()
    assert not ledger.allows("props"), "fixture failed to exhaust the budget"

    called: list[str] = []

    def _fake_event_odds(self, event_id, markets, regions, odds_format="american"):
        called.append(event_id)
        return {"id": event_id, "bookmakers": []}

    monkeypatch.setattr(odds_mod.OddsClient, "event_odds", _fake_event_odds)

    odds_mod.poll(force_tier="props", override_budget=False, source="scheduler")
    assert called == [], "the scheduler polled props while shed — the ladder is broken"

    odds_mod.poll(force_tier="props", override_budget=True, source="manual")
    assert called, (
        "the manual override did not reach the API. The button would return 200, "
        "the job would record success, and nothing would be polled.")


def test_manual_spend_is_attributed_to_manual(ledger_db, monkeypatch) -> None:
    """The ceiling counts `source='manual'` rows. If the client doesn't stamp
    them, the cap silently never engages — the ceiling would be untested code
    protecting nothing."""
    db_mod, odds_mod = ledger_db

    client = odds_mod.OddsClient(source="manual")
    client.ledger.record("/x", ["a", "b"], 12, None, None, source=client.source)

    assert odds_mod.CreditLedger().manual_used_today() == 12
    _spend(db_mod, 400, source="scheduler")
    assert odds_mod.CreditLedger().manual_used_today() == 12, (
        "scheduler spend counted against the manual allowance")


def test_the_default_source_is_the_scheduler(ledger_db) -> None:
    """Control. Every pre-existing caller passes no source; if the default were
    'manual' the scheduler's ordinary polling would eat the daily cap by
    lunchtime and the button would be permanently disabled."""
    _, odds_mod = ledger_db

    odds_mod.OddsClient().ledger.record("/x", ["a"], 50, None, None)
    assert odds_mod.CreditLedger().manual_used_today() == 0


# ---------------------------------------------------------------------------
# the ceiling
# ---------------------------------------------------------------------------
def test_the_ceiling_refuses_a_press_it_cannot_afford(ledger_db) -> None:
    from fastapi import HTTPException

    db_mod, odds_mod = ledger_db
    _game(db_mod)
    _spend(db_mod, 495, source="manual")     # 5 of 500 left; a 1-game sweep is 9

    import server
    with pytest.raises(HTTPException) as exc:
        server.props_refresh(admin={"username": "e"})
    assert exc.value.status_code == 429
    assert "allowance" in str(exc.value.detail)


def test_the_ceiling_allows_a_press_it_can_afford(ledger_db, monkeypatch) -> None:
    """Control on the control. Without it, a ceiling that refused everything
    would pass the test above and the button would never work."""
    db_mod, odds_mod = ledger_db
    _game(db_mod)
    _spend(db_mod, 100, source="manual")

    import server
    monkeypatch.setattr(server.scheduler, "add_job", lambda *a, **k: None)
    out = server.props_refresh(admin={"username": "e"})
    assert out["queued"] == "props_refresh"
    assert out["estimate"]["credits"] == 9


def test_no_games_is_refused_rather_than_charged(ledger_db) -> None:
    """A sweep of zero games costs zero credits and writes zero rows, so it
    would look like a success. Refuse it instead of reporting one."""
    from fastapi import HTTPException

    import server
    with pytest.raises(HTTPException) as exc:
        server.props_refresh(admin={"username": "e"})
    assert exc.value.status_code == 409


# ---------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------
def test_the_estimate_is_markets_times_regions_times_games(ledger_db) -> None:
    """The number on the button. 9 prop markets, `us` only, per event."""
    db_mod, odds_mod = ledger_db
    for i in range(12):
        _game(db_mod, game_id=f"g{i}")

    est = odds_mod.estimate_cost("props", today_only=True)
    assert est["per_game"] == 9
    assert est["games"] == 12
    assert est["credits"] == 108


def test_the_estimate_excludes_games_past_kickoff(ledger_db) -> None:
    """Quoting a price for games that already started would overcharge the
    ceiling and under-deliver picks — and `poll()` skips them, so the estimate
    would be describing a sweep that cannot happen."""
    db_mod, odds_mod = ledger_db
    _game(db_mod, hours_out=3.0, game_id="soon")
    _game(db_mod, hours_out=-2.0, game_id="started")
    _game(db_mod, hours_out=200.0, game_id="next_week")

    assert odds_mod.estimate_cost("props", today_only=True)["games"] == 1


def test_the_estimate_and_the_poll_agree_on_which_games(ledger_db) -> None:
    """They must share one definition. Two separate notions of 'upcoming' is
    how the quoted price and the actual spend drift apart."""
    db_mod, odds_mod = ledger_db
    _game(db_mod, hours_out=3.0, game_id="a")
    _game(db_mod, hours_out=-1.0, game_id="b")

    import inspect
    assert "upcoming_games" in inspect.getsource(odds_mod.estimate_cost)
    assert [g["game_id"] for g in odds_mod.upcoming_games(today_only=False)] == ["a"]


# ---------------------------------------------------------------------------
# the endpoint that was never gated
# ---------------------------------------------------------------------------
def test_every_admin_endpoint_requires_an_admin(ledger_db) -> None:
    """`POST /api/admin/run/{job_id}` shipped with no auth dependency and a
    `TODO: gate behind admin auth before deploying` in its own docstring. It ran
    that way in production, reachable by anyone who guessed the path, able to
    fire any scheduled job — including the polls that spend the monthly credit
    budget.

    Asserted across the whole router rather than on that one route, because the
    failure was not "we wrote this route wrong", it was "nothing was checking".
    """
    import server

    ungated = []
    for route in server.app.routes:
        path = getattr(route, "path", "")
        if "/api/admin" not in path:
            continue
        params = getattr(getattr(route, "dependant", None), "dependencies", [])
        names = {getattr(d.call, "__name__", "") for d in params}
        if "current_admin" not in names:
            ungated.append(f"{sorted(getattr(route, 'methods', []) or [])} {path}")

    assert not ungated, (
        "admin routes with no current_admin dependency: " + ", ".join(ungated))


# ---------------------------------------------------------------------------
# "today" means today where the games are played
# ---------------------------------------------------------------------------
def test_sunday_night_football_belongs_to_sunday(ledger_db) -> None:
    """**The trap in the word "today", tested as the pure conversion it is.**

    An 8:20pm Eastern kickoff is 00:20 UTC the NEXT day. Filtering on the UTC
    date drops the most-bet game of the week out of a Sunday-afternoon refresh
    — silently, because sweeping the early games still returns quotes and still
    looks like it worked.

    Asserted against `_local_date` rather than through `upcoming_games`, because
    that function also filters on the wall clock: a test that seeded a 1pm game
    would pass all morning and fail every afternoon, which is a worse problem
    than the one it is checking for.
    """
    from datetime import date, datetime

    snf_utc = "2026-09-29T00:20:00+00:00"          # 8:20pm ET on Sunday the 28th
    _, odds_mod = ledger_db

    assert datetime.fromisoformat(snf_utc).date() == date(2026, 9, 29), (
        "fixture is not actually spanning the UTC date boundary")
    assert odds_mod._local_date(snf_utc) == date(2026, 9, 28), (
        "tonight's 8:20pm game was assigned to tomorrow — the window is using "
        "UTC dates, so every Sunday and Monday night game is invisible to the "
        "refresh button")


def test_an_afternoon_kickoff_is_unambiguous(ledger_db) -> None:
    """Control. If `_local_date` returned the UTC date for everything, the test
    above would be the only one failing and could be 'fixed' by shifting a day
    in the wrong direction."""
    from datetime import date

    _, odds_mod = ledger_db
    assert odds_mod._local_date("2026-09-28T17:00:00+00:00") == date(2026, 9, 28)


def test_only_todays_games_are_charged_for(ledger_db) -> None:
    """The reason the window narrowed: a press today must not pay for tomorrow.

    Offsets are relative to now and the expectation is derived from the same
    `_local_date` the code uses, so this means the same thing at any hour.
    """
    from datetime import datetime, timedelta, timezone

    db_mod, odds_mod = ledger_db
    now = datetime.now(timezone.utc)
    soon = (now + timedelta(minutes=90)).isoformat()
    later = (now + timedelta(days=2)).isoformat()
    _game(db_mod, game_id="soon", kickoff_utc=soon)
    _game(db_mod, game_id="in_two_days", kickoff_utc=later)

    today = odds_mod._local_date(soon)
    expected = [gid for gid, k in (("soon", soon), ("in_two_days", later))
                if odds_mod._local_date(k) == today]

    ids = [g["game_id"] for g in odds_mod.upcoming_games(today_only=True)]
    assert ids == expected, f"charged for games that are not today: {ids}"
    assert "in_two_days" not in ids


def test_the_scheduler_still_gets_the_full_window(ledger_db) -> None:
    """Control. The 72-hour window is what gives CLV a price history; narrowing
    it for the scheduler would quietly stop that, and nothing downstream would
    complain until the closing-line numbers went strange weeks later."""
    from datetime import datetime, timedelta, timezone

    db_mod, odds_mod = ledger_db
    now = datetime.now(timezone.utc)
    _game(db_mod, game_id="soon", kickoff_utc=(now + timedelta(minutes=90)).isoformat())
    _game(db_mod, game_id="in_two_days", kickoff_utc=(now + timedelta(days=2)).isoformat())

    assert len(odds_mod.upcoming_games(today_only=False)) == 2
    assert len(odds_mod.upcoming_games(today_only=False, hours=24.0)) == 1


def test_the_manual_sweep_polls_exactly_what_it_quoted(ledger_db, monkeypatch) -> None:
    """**The estimate is a promise, and this is what holds it to it.**

    `estimate_cost` and `poll` are two functions that each decide which games
    are in scope. If they disagree, the button says "today's 3 games, 27
    credits" and then bills for a 72-hour sweep — an overspend that is invisible
    because the job still succeeds and the picks still appear.

    Mutation that motivated this: pointing the poll loop at the full window
    while leaving the estimate on today left every other test in this file
    green.
    """
    from datetime import datetime, timedelta, timezone

    db_mod, odds_mod = ledger_db
    now = datetime.now(timezone.utc)
    soon = (now + timedelta(minutes=90)).isoformat()
    later = (now + timedelta(days=2)).isoformat()
    _game(db_mod, game_id="soon", kickoff_utc=soon)
    _game(db_mod, game_id="in_two_days", kickoff_utc=later)

    called: list[str] = []

    def _fake_event_odds(self, event_id, markets, regions, odds_format="american"):
        called.append(event_id)
        return {"id": event_id, "bookmakers": []}

    monkeypatch.setattr(odds_mod.OddsClient, "event_odds", _fake_event_odds)

    est = odds_mod.estimate_cost("props", today_only=True)
    odds_mod.poll(force_tier="props", override_budget=True, source="manual",
                  today_only=True)

    assert len(called) == est["games"], (
        f"quoted {est['games']} games, polled {len(called)} — the estimate and "
        f"the sweep are using different windows")
    assert "evt_in_two_days" not in called
