"""The manual props refresh: it must poll when the scheduler won't, and stop.

**It must actually override.** The credit ladder sheds props first — below 30%
of the monthly budget `allows("props")` returns False and the poll loop skips
the tier with nothing but a log line. That is right for an unattended job and
useless to a person staring at an empty props tab an hour before kickoff. If the
override doesn't reach past that check, the button returns 200, queues a job,
runs it to success, and polls nothing: a green pipeline over an empty pipe with
a button on it. `test_a_manual_poll_runs_when_the_scheduler_would_shed` is the
reason this file exists.

**It must still stop**, under two different ceilings. The daily cap bounds a
runaway — a stuck finger, a retry loop. The monthly cap bounds a habit, and is
the one that protects anything: at 1,500/day a press every day is 45,000 a
month, more than the ~44,000 of headroom the scheduler leaves. Both are checked
before the job is queued, and the refusal names which one bit, because "you have
0 left" with two caps in play sends you to the wrong environment variable.

What exhaustion actually costs is worth stating, because it is not a bill.
Credits are prepaid — $59 for 100,000, about six cents a sweep. Running out
returns 429 and the data stops for EVERY market until the 1st, so overspending
here starves the scheduled polls behind moneyline, spread and totals.

Mutation-checked; each of these turns a specific test red:
  * drop `and not override_budget` from the shed check
  * drop the ceiling check in `props_refresh`
  * default the ledger's spend source to "manual"
  * compare kickoffs by UTC date instead of `settings.TZ`
  * point the poll loop at the full window while the estimate says today
  * return the daily figure alone from `manual_remaining_today`
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
    monkeypatch.setenv("ODDS_MANUAL_MONTHLY_CREDITS", "2000")
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
def test_every_api_route_requires_an_authenticated_principal() -> None:
    """`POST /api/admin/run/{job_id}` shipped with no auth dependency and a
    `TODO: gate behind admin auth before deploying` in its own docstring. It ran
    that way in production, reachable by anyone who guessed the path, able to
    fire any scheduled job — including the polls that spend the monthly credit
    budget.

    `GET /api/edges` shipped the same way and served the entire pre-tier
    opportunity list — the product itself — to anyone with the URL. The first
    version of this test missed it, because that version only audited
    `/api/admin/*`. **The prefix is a naming convention, not a security
    boundary**, and auditing by prefix is how the next one gets missed too.

    So: every `/api` route must depend on `current_user` or `current_admin`,
    and anything genuinely public has to be named in PUBLIC below — which
    turns "should this be open?" into a decision someone makes on purpose, in
    a diff, rather than a property of where the route happened to be written.

    This does not check that the right routes are ADMIN rather than user; it
    checks that none are open. Admin-vs-user is a judgement per route.
    """
    import server

    PUBLIC = {
        # Railway's healthcheck hits this unauthenticated; it deliberately
        # exposes no picks, only job status and aggregate counts.
        "/health",
        # Pre-login surface. Each is reachable without a token by definition.
        "/api/auth/login", "/api/auth/register", "/api/auth/forgot",
        "/api/auth/reset/{token}", "/api/auth/invite/{code}",
        # Redeems a single-use reset token and returns a session. Carries its
        # own bearer credential in the body, so a token gate here would make
        # password reset impossible for the people who need it.
        "/api/auth/reset",
        # Aggregate, non-identifying record. Public on purpose.
        "/api/stats",
    }

    GATES = {"current_user", "current_admin"}
    ungated = []
    for route in server.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api") or path in PUBLIC:
            continue
        deps = getattr(getattr(route, "dependant", None), "dependencies", [])
        if not ({getattr(d.call, "__name__", "") for d in deps} & GATES):
            ungated.append(f"{sorted(getattr(route, 'methods', []) or [])} {path}")

    assert not ungated, (
        "API routes with no authentication dependency and not on the PUBLIC "
        "allowlist: " + ", ".join(ungated))


def test_the_routes_that_spend_money_or_change_state_require_ADMIN() -> None:
    """Stronger than the audit above for the routes where it matters.

    A user-level gate on these would mean any of the twelve people with an
    invite could trigger the polls that spend the credit budget, or read the
    raw opportunity list before tiering.
    """
    import server

    MUST_BE_ADMIN = {"/api/admin/props-refresh", "/api/admin/run/{job_id}",
                     "/api/admin/ingest/{step}", "/api/edges"}

    for route in server.app.routes:
        if getattr(route, "path", "") not in MUST_BE_ADMIN:
            continue
        deps = getattr(getattr(route, "dependant", None), "dependencies", [])
        names = {getattr(d.call, "__name__", "") for d in deps}
        assert "current_admin" in names, f"{route.path} is not admin-gated"


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


# ---------------------------------------------------------------------------
# two caps, and the message has to say which one bit
# ---------------------------------------------------------------------------
def test_the_monthly_cap_binds_even_when_the_day_is_clear(ledger_db) -> None:
    """**Why a daily cap alone protects nothing.**

    At the configured 1,500/day a press every day is 45,000 a month — more than
    the ~44,000 of headroom the scheduler leaves. The daily number limits how
    fast you can spend; only the monthly number limits how much.

    Here the day is untouched and the month is spent, so a daily-only ceiling
    would wave this through.
    """
    from datetime import datetime, timedelta, timezone

    from fastapi import HTTPException

    db_mod, odds_mod = ledger_db
    _game(db_mod)
    earlier = (datetime.now(timezone.utc).replace(
        hour=0, minute=1, second=0, microsecond=0) - timedelta(days=1)).isoformat()
    _spend(db_mod, 1995, source="manual", when=earlier)

    ledger = odds_mod.CreditLedger()
    assert ledger.manual_used_today() == 0, "fixture spent today, not a previous day"
    assert ledger.manual_used_this_month() == 1995
    assert ledger.manual_remaining_today() == 5, (
        "the day is clear but the month is spent — remaining must be the minimum")

    import server
    with pytest.raises(HTTPException) as exc:
        server.props_refresh(admin={"username": "e"})
    assert exc.value.status_code == 429
    assert "month" in str(exc.value.detail), (
        "the refusal did not name the monthly cap, so the fix is a guess "
        "between two environment variables")


def test_the_daily_cap_still_binds_inside_a_clear_month(ledger_db) -> None:
    """The other direction. Both caps must be able to bite on their own."""
    from fastapi import HTTPException

    db_mod, odds_mod = ledger_db
    _game(db_mod)
    _spend(db_mod, 497, source="manual")      # day nearly spent, month is not

    ledger = odds_mod.CreditLedger()
    assert ledger.manual_remaining_today() == 3

    import server
    with pytest.raises(HTTPException) as exc:
        server.props_refresh(admin={"username": "e"})
    assert "day" in str(exc.value.detail)


def test_remaining_is_the_minimum_of_the_two(ledger_db) -> None:
    """Control. If `manual_remaining_today` returned the daily figure alone the
    monthly cap would be decorative, and both tests above could be satisfied by
    a message change with no enforcement behind it."""
    from datetime import datetime, timedelta, timezone

    db_mod, odds_mod = ledger_db
    earlier = (datetime.now(timezone.utc).replace(
        hour=0, minute=1, second=0, microsecond=0) - timedelta(days=1)).isoformat()
    _spend(db_mod, 1800, source="manual", when=earlier)
    _spend(db_mod, 100, source="manual")

    # Today's spend counts against BOTH windows, which is easy to get wrong and
    # which I did get wrong writing this test: month is 1,800 + 100 = 1,900 of
    # 2,000, so 100 left; day is 100 of 500, so 400 left.
    ledger = odds_mod.CreditLedger()
    assert ledger.manual_used_this_month() == 1900
    assert ledger.manual_used_today() == 100
    assert ledger.manual_remaining_today() == 100
