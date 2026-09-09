"""Outage alerting — text the admin when something breaks, once.

**Why this exists.** On 9 September 2026, opening day, three things were quietly broken
at the same time and every one of them looked healthy from the outside:

* the Anthropic API balance hit zero, so the red-team agent reviewed nothing. It fails
  open by design, so picks published unreviewed and `/health` stayed green.
* Omaha's extraction failed on every chunk for eleven days.
* `capture_closing_lines` had been firing four hours early for the entire life of the
  system, measuring CLV against the wrong price.

None of those raised. Two of them were *supposed* not to raise. The lesson isn't that
fail-open is wrong — a broken AI layer must never suppress a slate — it's that failing
open silently and failing open loudly are different things, and only one of them is
safe.

**Edge-triggered, deliberately.** Each check has a key. A text goes out when a key flips
from quiet to firing, and again when it recovers. Never a reminder. A monitor that says
"still broken" every thirty minutes gets muted within a day, and a muted monitor is
worse than none because it also persuades you that you have monitoring.

**Never raises.** This runs on a scheduler; an alerting system that can take down the
thing it watches is a liability. Every check is individually guarded, and a check that
throws is reported as its own alert rather than killing the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.config import settings
from src.db import db, query, utcnow

log = logging.getLogger(__name__)

# Any single text over this gets truncated by carriers into multiple messages, which
# arrive out of order. One alert, one message.
MAX_SMS = 300


@dataclass
class Check:
    key: str
    firing: bool
    detail: str


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_ai_layer() -> Check | None:
    """Did the red team actually reach the model, or only appear to?

    **The detector that would have caught opening day.** When the API rejects a call —
    bad key, exhausted balance, retired model — the exception is raised inside
    `complete()` *before* `_log_call`, so no row lands in `ai_calls`. Meanwhile
    `apply_to_picks` catches `AIUnavailable`, returns OK, and the job records success.

    So: the job ran, and the ledger is empty. That pair means every call failed, and it
    is invisible in every other signal we have.

    Silent when the flag is off — a disabled feature is not an outage.
    """
    if not settings.ENABLE_AI_REDTEAM:
        return None

    ran = query(
        "SELECT finished_at FROM job_runs WHERE job_name='redteam_review' "
        "AND status='success' ORDER BY id DESC LIMIT 1"
    )
    if not ran or not ran[0]["finished_at"]:
        return None  # never run — that is a freshness problem, not an AI problem

    since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(timespec="seconds")
    calls = query(
        "SELECT COUNT(*) AS n FROM ai_calls WHERE agent='redteam' AND created_at >= ?",
        (since,),
    )
    n = int(calls[0]["n"]) if calls else 0
    recent_run = ran[0]["finished_at"] >= since
    if recent_run and n == 0:
        return Check(
            "ai_down",
            True,
            "red team ran but made 0 API calls in 6h — key, balance or model. "
            "Picks are publishing UNREVIEWED.",
        )
    return Check("ai_down", False, f"{n} redteam calls in 6h")


def check_credit_ladder() -> Check | None:
    """Has the odds budget started shedding tiers?

    Sheds props below 30% remaining. Because it degrades rather than erroring, the
    first symptom is a market quietly not being collected.
    """
    try:
        from src.fetchers.odds_api import CreditLedger

        ledger = CreditLedger()
        shed = [t for t in ("featured", "period", "props") if not ledger.allows(t)]
    except Exception as exc:  # noqa: BLE001
        return Check("credits", True, f"credit ledger unreadable: {type(exc).__name__}")

    if shed:
        return Check(
            "credits",
            True,
            f"odds budget shedding {', '.join(shed)} — "
            f"{ledger.remaining_pct():.0f}% of {ledger.budget:,} left",
        )
    return Check("credits", False, "credit ladder clear")


def check_stale_jobs() -> Check | None:
    """Any scheduled job past its freshness window.

    Reuses `EXPECTED_FRESHNESS` rather than a second list, so a job added to one is
    watched by both. Two lists of jobs is two lists to forget to update.
    """
    try:
        from server import EXPECTED_FRESHNESS
    except Exception:  # noqa: BLE001 — importable from the CLI too
        return None

    now = datetime.now(timezone.utc)
    stale: list[str] = []
    for name, max_age in EXPECTED_FRESHNESS.items():
        if name == "redteam_review" and not settings.ENABLE_AI_REDTEAM:
            continue
        rows = query(
            "SELECT status, finished_at FROM job_runs WHERE job_name=? "
            "ORDER BY id DESC LIMIT 1", (name,),
        )
        if not rows:
            continue  # never run; startup grace is /health's job, not ours
        r = rows[0]
        try:
            age = now - datetime.fromisoformat(r["finished_at"])
        except (TypeError, ValueError):
            continue
        if age > max_age or r["status"] != "success":
            stale.append(f"{name} ({age.days}d)" if age.days else name)

    if stale:
        return Check("stale_jobs", True, "stale: " + ", ".join(stale[:5]))
    return Check("stale_jobs", False, "all jobs fresh")


CHECKS = (check_ai_layer, check_credit_ladder, check_stale_jobs)


# ---------------------------------------------------------------------------
# edge detection and delivery
# ---------------------------------------------------------------------------
def _previous(key: str) -> bool:
    rows = query("SELECT firing FROM alert_state WHERE alert_key=?", (key,))
    return bool(rows[0]["firing"]) if rows else False


def _record(key: str, firing: bool, detail: str, sent: bool) -> None:
    now = utcnow()
    with db() as conn:
        conn.execute(
            "INSERT INTO alert_state (alert_key, firing, detail, changed_at, last_sent) "
            "VALUES (?,?,?,?,?) ON CONFLICT(alert_key) DO UPDATE SET "
            "firing=excluded.firing, detail=excluded.detail, "
            "changed_at=excluded.changed_at, "
            "last_sent=COALESCE(excluded.last_sent, alert_state.last_sent)",
            (key, 1 if firing else 0, detail[:500], now, now if sent else None),
        )


def run_checks(dry_run: bool = False) -> dict:
    """Run every check, text on transitions only. Returns what it saw and did."""
    from src.notify import sms

    fired: list[str] = []
    recovered: list[str] = []
    seen: list[dict] = []

    for fn in CHECKS:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            # A broken check is itself worth knowing about, and must not stop the rest.
            log.exception("alert check %s failed", fn.__name__)
            result = Check(f"check_error:{fn.__name__}", True,
                           f"check raised {type(exc).__name__}")
        if result is None:
            continue

        was = _previous(result.key)
        seen.append({"key": result.key, "firing": result.firing, "detail": result.detail})

        if result.firing == was:
            continue  # no transition — say nothing

        prefix = "the-algo DOWN" if result.firing else "the-algo recovered"
        body = f"{prefix}: {result.detail}"[:MAX_SMS]
        sent = False
        if not dry_run and settings.ADMIN_PHONE:
            out = sms.send(settings.ADMIN_PHONE, body, ref=f"alert:{result.key}")
            sent = out.get("status") == "sent"
            if not sent:
                log.error("alert SMS not delivered: %s", out.get("error"))
        else:
            log.warning("ALERT (not sent): %s", body)

        (fired if result.firing else recovered).append(result.key)
        _record(result.key, result.firing, result.detail, sent)

    if fired or recovered:
        log.warning("alerts fired=%s recovered=%s", fired, recovered)
    return {"checked": len(seen), "fired": fired, "recovered": recovered, "state": seen}


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="outage alerts")
    p.add_argument("command", choices=["check", "dry", "state"])
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "state":
        for r in query("SELECT * FROM alert_state ORDER BY alert_key"):
            mark = "FIRING" if r["firing"] else "ok    "
            print(f"  {mark}  {r['alert_key']:<16} {r['detail']}")
    else:
        print(_json.dumps(run_checks(dry_run=args.command == "dry"), indent=2))
