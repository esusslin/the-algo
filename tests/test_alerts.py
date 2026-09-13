"""Outage alerts — the monitor that would have caught opening day.

On 9 September 2026 three things were broken at once and all of them looked healthy:
the Anthropic balance was zero so the red team reviewed nothing (fails open by design),
Omaha's extraction had failed on every chunk for eleven days, and closing lines had been
captured four hours early since the system was built.

Fail-open is right — a broken AI layer must never suppress a slate. But *silently*
failing open and *loudly* failing open are different things, and only one is safe.

Two properties are worth more than the checks themselves:

**Edge-triggered.** A text on the transition, never a reminder. A monitor that says
"still broken" every fifteen minutes gets muted inside a day, and a muted monitor is
worse than none because it also convinces you that you have monitoring.

**Cannot take down what it watches.** It runs on the scheduler beside the jobs it
monitors. A check that raises is reported as its own alert rather than killing the run.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def alert_db(monkeypatch):
    """Real in-memory SQLite — the checks are SQL, and mocks would test nothing."""
    from src.notify import alerts

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE job_runs (id INTEGER PRIMARY KEY, job_name TEXT, status TEXT,"
        "  finished_at TEXT);"
        "CREATE TABLE ai_calls (id INTEGER PRIMARY KEY, agent TEXT, created_at TEXT);"
        "CREATE TABLE alert_state (alert_key TEXT PRIMARY KEY, firing INTEGER,"
        "  detail TEXT, changed_at TEXT, last_sent TEXT);"
        "CREATE TABLE picks (pick_id INTEGER PRIMARY KEY, game_id TEXT, result TEXT,"
        "  published INTEGER, review_hash TEXT);"
        "CREATE TABLE games (game_id TEXT PRIMARY KEY, kickoff_utc TEXT);"
    )

    class _Ctx:
        def __enter__(self): return conn
        def __exit__(self, *a): return False

    monkeypatch.setattr(alerts, "query",
                        lambda sql, params=(): [dict(r) for r in conn.execute(sql, params)])
    monkeypatch.setattr(alerts, "db", lambda: _Ctx())
    monkeypatch.setattr(alerts.settings, "ENABLE_AI_REDTEAM", True, raising=False)
    return conn


def _ago(**kw) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat(timespec="seconds")


def _ran(conn, **kw) -> None:
    conn.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                 (_ago(**(kw or {"minutes": 20})),))


def _pick(conn, pick_id=1, review_hash=None, published=1, result="pending",
          kickoff="2099-01-01T00:00:00+00:00") -> None:
    gid = f"g{pick_id}"
    conn.execute("INSERT INTO games VALUES (?,?)", (gid, kickoff))
    conn.execute("INSERT INTO picks VALUES (?,?,?,?,?)",
                 (pick_id, gid, result, published, review_hash))


# --- the AI detector -----------------------------------------------------------------
#
# Originally: "the job ran and `ai_calls` is empty, so every call failed." That was the
# right rule until review caching landed, at which point a HEALTHY system started making
# zero calls for hours — nothing that feeds a verdict had moved. The old rule would have
# become a permanent false alarm, and a permanent false alarm is how a monitor gets
# muted.
#
# So it now keys on the symptom rather than the mechanism: are there live picks nobody
# has reviewed? `review_hash` is written only when a review actually reached the model,
# so NULL means genuinely unreviewed — whatever the cause, including a cache bug of our
# own making.


def test_unreviewed_live_picks_after_a_run_is_an_outage(alert_db) -> None:
    from src.notify.alerts import check_ai_layer

    _ran(alert_db)
    _pick(alert_db, review_hash=None)
    result = check_ai_layer()
    assert result.firing is True
    assert "UNREVIEWED" in result.detail


def test_a_quiet_cache_with_everything_reviewed_is_healthy(alert_db) -> None:
    """**The regression this rewrite exists to prevent.** Zero API calls in six
    hours is now the normal, correct, cheap state. Alerting on it would fire
    permanently from the day caching shipped."""
    from src.notify.alerts import check_ai_layer

    _ran(alert_db)
    _pick(alert_db, review_hash="abc123")
    result = check_ai_layer()
    assert result.firing is False
    assert alert_db.execute("SELECT COUNT(*) FROM ai_calls").fetchone()[0] == 0


def test_an_unreviewed_pick_for_a_started_game_is_not_an_outage(alert_db) -> None:
    """The bet is not placeable, so nobody is exposed to an unreviewed pick."""
    from src.notify.alerts import check_ai_layer

    _ran(alert_db)
    _pick(alert_db, review_hash=None, kickoff="2020-01-01T00:00:00+00:00")
    assert check_ai_layer().firing is False


def test_an_unpublished_pick_is_not_an_outage(alert_db) -> None:
    """A KILLed or withheld pick is not in front of a user."""
    from src.notify.alerts import check_ai_layer

    _ran(alert_db)
    _pick(alert_db, review_hash=None, published=0)
    assert check_ai_layer().firing is False


def test_an_old_run_is_not_an_ai_outage(alert_db) -> None:
    """Nothing has run recently, so unreviewed picks prove nothing about the API.
    That is a staleness problem and `check_stale_jobs` owns it — two checks firing
    for one cause is how an alert becomes noise."""
    from src.notify.alerts import check_ai_layer

    _ran(alert_db, days=3)
    _pick(alert_db, review_hash=None)
    assert check_ai_layer() is None


def test_a_disabled_feature_is_not_an_outage(alert_db, monkeypatch) -> None:
    """`ENABLE_AI_REDTEAM=false` is a decision. Alerting on it would be a permanent
    false alarm, which is how a monitor gets muted."""
    from src.notify import alerts

    monkeypatch.setattr(alerts.settings, "ENABLE_AI_REDTEAM", False, raising=False)
    assert alerts.check_ai_layer() is None


def test_the_alert_reports_the_call_count_for_diagnosis(alert_db) -> None:
    """Call volume stopped being the trigger but is still the first thing you want
    to know when deciding between a dead key and a broken cache."""
    from src.notify.alerts import check_ai_layer

    _ran(alert_db)
    _pick(alert_db, review_hash=None)
    alert_db.execute("INSERT INTO ai_calls VALUES (1,'redteam',?)", (_ago(minutes=5),))
    assert "1 API calls" in check_ai_layer().detail


# --- edge triggering -----------------------------------------------------------------


def _send_spy(monkeypatch):
    from src.notify import alerts, sms

    sent: list[str] = []
    monkeypatch.setattr(sms, "send",
                        lambda to, body, ref="", channel="sms": (
                            sent.append(body) or {"status": "sent"}))
    monkeypatch.setattr(alerts.settings, "ADMIN_PHONE", "+15550001111", raising=False)
    return sent


def test_it_texts_once_when_something_breaks(alert_db, monkeypatch) -> None:
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash=None)

    assert alerts.run_checks()["fired"] == ["ai_down"]
    assert len(sent) == 1
    assert "DOWN" in sent[0]


def test_it_does_not_text_again_while_still_broken(alert_db, monkeypatch) -> None:
    """**The property that keeps this useful.** Three more runs, same outage, no more
    texts. A reminder every fifteen minutes is how you learn to ignore the sender."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash=None)

    for _ in range(4):
        alerts.run_checks()
    assert len(sent) == 1, f"sent {len(sent)} texts for one outage"


def test_it_texts_again_on_recovery(alert_db, monkeypatch) -> None:
    """Recovery matters as much as failure — otherwise you never learn whether the
    thing you did actually fixed it."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash=None)
    alerts.run_checks()

    alert_db.execute("UPDATE picks SET review_hash='abc123' WHERE pick_id=1")
    out = alerts.run_checks()

    assert out["recovered"] == ["ai_down"]
    assert len(sent) == 2
    assert "recovered" in sent[1]


def test_a_healthy_system_sends_nothing(alert_db, monkeypatch) -> None:
    """The control. Without it, a checker that fired constantly would satisfy every
    test above."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash="abc123")

    assert alerts.run_checks() == {"checked": 1, "fired": [], "recovered": [],
                                   "state": [{"key": "ai_down", "firing": False,
                                              "detail": "no unreviewed live picks"}]}
    assert sent == []


# --- it must not take down what it watches -------------------------------------------


def test_a_check_that_raises_becomes_its_own_alert(alert_db, monkeypatch) -> None:
    """This runs on the same scheduler as the jobs it monitors. A monitor that can
    crash the process is a liability, not a safeguard."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)

    def exploding():
        raise RuntimeError("boom")

    monkeypatch.setattr(alerts, "CHECKS", (exploding, alerts.check_ai_layer))
    _ran(alert_db)
    _pick(alert_db, review_hash="abc123")

    out = alerts.run_checks()
    assert any(k.startswith("check_error") for k in out["fired"])
    assert len(sent) == 1          # the healthy check still ran


def test_a_failed_text_does_not_raise(alert_db, monkeypatch) -> None:
    """Twilio being down must not stop the check loop, and must not be recorded as a
    delivered alert — otherwise the next run treats it as already-notified and stays
    silent forever."""
    from src.notify import alerts, sms

    monkeypatch.setattr(sms, "send",
                        lambda *a, **k: {"status": "failed", "error": "twilio down"})
    monkeypatch.setattr(alerts.settings, "ADMIN_PHONE", "+15550001111", raising=False)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash=None)

    assert alerts.run_checks()["fired"] == ["ai_down"]
    row = alert_db.execute("SELECT firing, last_sent FROM alert_state").fetchone()
    assert row["firing"] == 1
    assert row["last_sent"] is None    # recorded as firing, but not as delivered


def test_dry_run_sends_nothing(alert_db, monkeypatch) -> None:
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    _ran(alert_db)
    _pick(alert_db, review_hash=None)

    assert alerts.run_checks(dry_run=True)["fired"] == ["ai_down"]
    assert sent == []


def test_the_message_fits_in_one_sms(alert_db, monkeypatch) -> None:
    """Over ~300 characters carriers split the message, and the parts arrive out of
    order. One alert, one message."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(
        alerts, "CHECKS",
        (lambda: alerts.Check("noisy", True, "x" * 5000),))
    alerts.run_checks()
    assert len(sent[0]) <= alerts.MAX_SMS
