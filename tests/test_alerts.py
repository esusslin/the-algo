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


# --- the AI detector -----------------------------------------------------------------
#
# The one that would have caught 9 September. When the API rejects a call, the exception
# is raised inside `complete()` BEFORE `_log_call`, so nothing lands in `ai_calls`.
# Meanwhile `apply_to_picks` catches it, returns OK, and the job records success.
# Job ran + empty ledger = every call failed, and no other signal shows it.


def test_job_ran_but_no_api_calls_is_an_outage(alert_db) -> None:
    from src.notify.alerts import check_ai_layer

    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    result = check_ai_layer()
    assert result.firing is True
    assert "UNREVIEWED" in result.detail


def test_job_ran_and_calls_were_logged_is_healthy(alert_db) -> None:
    from src.notify.alerts import check_ai_layer

    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    alert_db.execute("INSERT INTO ai_calls VALUES (1,'redteam',?)", (_ago(minutes=19),))
    assert check_ai_layer().firing is False


def test_an_old_run_with_no_calls_is_not_an_ai_outage(alert_db) -> None:
    """Nothing has run recently, so an empty ledger proves nothing about the API. That
    is a staleness problem and `check_stale_jobs` owns it — two checks firing for one
    cause is how an alert becomes noise."""
    from src.notify.alerts import check_ai_layer

    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(days=3),))
    assert check_ai_layer().firing is False


def test_a_disabled_feature_is_not_an_outage(alert_db, monkeypatch) -> None:
    """`ENABLE_AI_REDTEAM=false` is a decision. Alerting on it would be a permanent
    false alarm, which is how a monitor gets muted."""
    from src.notify import alerts

    monkeypatch.setattr(alerts.settings, "ENABLE_AI_REDTEAM", False, raising=False)
    assert alerts.check_ai_layer() is None


def test_calls_from_another_agent_do_not_count(alert_db) -> None:
    """Narrative generation succeeding says nothing about the red team's own calls."""
    from src.notify.alerts import check_ai_layer

    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    alert_db.execute("INSERT INTO ai_calls VALUES (1,'narrative',?)", (_ago(minutes=5),))
    assert check_ai_layer().firing is True


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
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))

    assert alerts.run_checks()["fired"] == ["ai_down"]
    assert len(sent) == 1
    assert "DOWN" in sent[0]


def test_it_does_not_text_again_while_still_broken(alert_db, monkeypatch) -> None:
    """**The property that keeps this useful.** Three more runs, same outage, no more
    texts. A reminder every fifteen minutes is how you learn to ignore the sender."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))

    for _ in range(4):
        alerts.run_checks()
    assert len(sent) == 1, f"sent {len(sent)} texts for one outage"


def test_it_texts_again_on_recovery(alert_db, monkeypatch) -> None:
    """Recovery matters as much as failure — otherwise you never learn whether the
    thing you did actually fixed it."""
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    alerts.run_checks()

    alert_db.execute("INSERT INTO ai_calls VALUES (1,'redteam',?)", (_ago(minutes=1),))
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
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    alert_db.execute("INSERT INTO ai_calls VALUES (1,'redteam',?)", (_ago(minutes=5),))

    assert alerts.run_checks() == {"checked": 1, "fired": [], "recovered": [],
                                   "state": [{"key": "ai_down", "firing": False,
                                              "detail": "1 redteam calls in 6h"}]}
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
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))
    alert_db.execute("INSERT INTO ai_calls VALUES (1,'redteam',?)", (_ago(minutes=5),))

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
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))

    assert alerts.run_checks()["fired"] == ["ai_down"]
    row = alert_db.execute("SELECT firing, last_sent FROM alert_state").fetchone()
    assert row["firing"] == 1
    assert row["last_sent"] is None    # recorded as firing, but not as delivered


def test_dry_run_sends_nothing(alert_db, monkeypatch) -> None:
    from src.notify import alerts

    sent = _send_spy(monkeypatch)
    monkeypatch.setattr(alerts, "CHECKS", (alerts.check_ai_layer,))
    alert_db.execute("INSERT INTO job_runs VALUES (1,'redteam_review','success',?)",
                     (_ago(minutes=20),))

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
