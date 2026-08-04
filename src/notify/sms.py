"""Twilio SMS — invites and alerts.

Every send is logged so a failed delivery is visible rather than silent, and so
you can see who was invited without digging through Twilio's console.

SMS discipline matters more than the code here. Notification fatigue is the
fastest way to get an app muted, so the alert rules stay tight: A-tier picks
only, plus Sunday inactives that affect a bet someone actually logged. Never
B/C tier — those are pull, not push.
"""
from __future__ import annotations

import logging

from src.config import settings
from src.db import db, insert_row, utcnow

log = logging.getLogger(__name__)


def _client():
    from twilio.rest import Client
    return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


def configured() -> bool:
    return bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN
                and settings.TWILIO_FROM_NUMBER)


def normalize_phone(raw: str) -> str:
    """Best-effort E.164. Assumes US if no country code given."""
    digits = "".join(c for c in (raw or "") if c.isdigit() or c == "+")
    if digits.startswith("+"):
        return digits
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return digits


def send(to: str, body: str, ref: str = "", channel: str = "sms") -> dict:
    """Send one SMS. Never raises — returns a status dict and logs the attempt."""
    to = normalize_phone(to)
    status, error, sid = "sent", None, None

    if not configured():
        status, error = "skipped", "twilio not configured"
    elif not settings.ENABLE_SMS:
        status, error = "skipped", "ENABLE_SMS is false"
    elif not to:
        status, error = "failed", "invalid phone number"
    else:
        try:
            msg = _client().messages.create(
                body=body, from_=settings.TWILIO_FROM_NUMBER, to=to)
            sid = msg.sid
        except Exception as exc:  # noqa: BLE001 — delivery must never crash a request
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("sms to %s failed: %s", to, error)

    with db() as conn:
        insert_row(conn, "invite_sends", {
            "code": ref, "channel": channel, "recipient": to,
            "status": status, "error": error, "sent_at": utcnow(),
        })
    return {"status": status, "error": error, "sid": sid, "to": to}


def send_invite(code: str, phone: str, base_url: str, inviter: str = "") -> dict:
    link = f"{base_url.rstrip('/')}/app?invite={code}"
    who = f" from {inviter}" if inviter else ""
    body = (f"You're invited to The Algo{who}.\n\n{link}\n\n"
            f"Link works once and expires in 14 days.")
    return send(phone, body, ref=code, channel="invite")


def send_reset(token: str, phone: str, base_url: str) -> dict:
    link = f"{base_url.rstrip('/')}/app?reset={token}"
    body = (f"Reset your Algo password:\n\n{link}\n\n"
            f"Expires in 1 hour. If you didn't ask for this, ignore it.")
    return send(phone, body, ref=token[:12], channel="reset")


def send_pick_alert(phone: str, headline: str, book: str, price: int,
                    base_url: str) -> dict:
    body = (f"A-tier: {headline}\n{book} {price:+d}\n"
            f"{base_url.rstrip('/')}/app")
    return send(phone, body, ref="", channel="alert")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="sms")
    p.add_argument("command", choices=["check", "test"])
    p.add_argument("--to", default="")
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "check":
        print(f"  twilio configured : {configured()}")
        print(f"  ENABLE_SMS        : {settings.ENABLE_SMS}")
        print(f"  from number       : {settings.TWILIO_FROM_NUMBER or '(unset)'}")
        for raw in ["5551234567", "+15551234567", "(555) 123-4567", "1-555-123-4567"]:
            print(f"  normalize {raw:<18} -> {normalize_phone(raw)}")
    else:
        print(send(args.to, "Test from The Algo.", ref="test"))
