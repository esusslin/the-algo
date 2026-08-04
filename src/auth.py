"""Authentication: JWT sessions, bcrypt passwords, admin-issued invites.

Invite flow (admin-only, since this is a closed group):
  1. Admin creates an invite -> a short code
  2. Admin shares the link  /app?invite=CODE
  3. Recipient sets a username and password; the code is consumed
  4. Codes are single-use, expire, and can be revoked

bcrypt is used directly rather than through passlib: passlib 1.7.x reads
bcrypt.__about__, which bcrypt 4.x removed, producing a confusing runtime error.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import settings
from src.db import db, insert_row, query, query_one, utcnow

log = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)

INVITE_TTL_DAYS = 14


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------
def hash_password(raw: str) -> str:
    return bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------
def create_token(user_id: int, username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "username": username, "role": role,
               "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY,
                          algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------
def get_user(username: str) -> dict | None:
    row = query_one("SELECT * FROM users WHERE username=?", (username.strip().lower(),))
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    row = query_one("SELECT * FROM users WHERE id=?", (user_id,))
    return dict(row) if row else None


def create_user(username: str, password: str, role: str = "user",
                phone: str | None = None) -> int:
    username = username.strip().lower()
    if get_user(username):
        raise ValueError("username already taken")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    with db() as conn:
        return insert_row(conn, "users", {
            "username": username,
            "phone": phone,
            "password_hash": hash_password(password),
            "role": role,
            "tier": "member",
            "sms_opt_in": 0,
            "created_at": utcnow(),
        })


def authenticate(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    with db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (utcnow(), user["id"]))
    return user


def bootstrap_admin() -> int | None:
    """Create or re-sync the admin account from env on every boot.

    The env vars are AUTHORITATIVE. An earlier version created the account once
    and then returned early forever, so changing ADMIN_PASSWORD in Railway had
    no effect and you were locked out with no way to tell why. Now each boot
    reconciles password, phone and role against the environment — which also
    means changing the var is a working recovery path if you forget it.
    """
    if not (settings.ADMIN_USERNAME and settings.ADMIN_PASSWORD):
        return None

    existing = get_user(settings.ADMIN_USERNAME)
    if not existing:
        uid = create_user(settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD,
                          role="admin", phone=settings.ADMIN_PHONE or None)
        log.info("bootstrapped admin user %r", settings.ADMIN_USERNAME)
        return uid

    changed = []
    with db() as conn:
        if not verify_password(settings.ADMIN_PASSWORD, existing["password_hash"]):
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (hash_password(settings.ADMIN_PASSWORD), existing["id"]))
            changed.append("password")
        if settings.ADMIN_PHONE and existing["phone"] != settings.ADMIN_PHONE:
            conn.execute("UPDATE users SET phone=? WHERE id=?",
                         (settings.ADMIN_PHONE, existing["id"]))
            changed.append("phone")
        if existing["role"] != "admin":
            conn.execute("UPDATE users SET role='admin' WHERE id=?", (existing["id"],))
            changed.append("role")
    if changed:
        log.info("admin account re-synced from env: %s", ", ".join(changed))
    return existing["id"]


# --------------------------------------------------------------------------
# invites
# --------------------------------------------------------------------------
def create_invite(created_by: int, note: str = "",
                  ttl_days: int = INVITE_TTL_DAYS) -> dict:
    code = secrets.token_urlsafe(9)
    expires = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat(
        timespec="seconds")
    with db() as conn:
        insert_row(conn, "invites", {
            "code": code, "created_by": created_by, "note": note,
            "created_at": utcnow(), "expires_at": expires,
        })
    return {"code": code, "expires_at": expires, "note": note}


def check_invite(code: str) -> tuple[bool, str]:
    row = query_one("SELECT * FROM invites WHERE code=?", (code.strip(),))
    if not row:
        return False, "invite not found"
    if row["revoked"]:
        return False, "invite revoked"
    if row["used_by"]:
        return False, "invite already used"
    if row["expires_at"] and row["expires_at"] < utcnow():
        return False, "invite expired"
    return True, "ok"


def redeem_invite(code: str, username: str, password: str,
                  phone: str | None = None) -> dict:
    ok, reason = check_invite(code)
    if not ok:
        raise ValueError(reason)
    user_id = create_user(username, password, role="user", phone=phone)
    with db() as conn:
        conn.execute("UPDATE invites SET used_by=?, used_at=? WHERE code=?",
                     (user_id, utcnow(), code.strip()))
    user = get_user_by_id(user_id)
    return {"user_id": user_id, "username": user["username"], "role": user["role"]}


# --------------------------------------------------------------------------
# password reset
# --------------------------------------------------------------------------
RESET_TTL_MINUTES = 60


def create_reset(user_id: int, issued_by: str = "self") -> dict:
    """Mint a single-use reset token.

    Any outstanding tokens for this user are invalidated first — otherwise an
    older link sitting in someone's messages stays live alongside the new one.
    """
    token = secrets.token_urlsafe(24)
    expires = (datetime.now(timezone.utc) + timedelta(minutes=RESET_TTL_MINUTES)
               ).isoformat(timespec="seconds")
    with db() as conn:
        conn.execute("UPDATE password_resets SET used_at=? WHERE user_id=? "
                     "AND used_at IS NULL", (utcnow(), user_id))
        insert_row(conn, "password_resets", {
            "token": token, "user_id": user_id, "created_at": utcnow(),
            "expires_at": expires, "issued_by": issued_by,
        })
    return {"token": token, "expires_at": expires}


def check_reset(token: str) -> tuple[bool, str]:
    row = query_one("SELECT * FROM password_resets WHERE token=?", (token.strip(),))
    if not row:
        return False, "reset link not found"
    if row["used_at"]:
        return False, "reset link already used"
    if (row["expires_at"] or "") < utcnow():
        return False, "reset link expired"
    return True, "ok"


def redeem_reset(token: str, new_password: str) -> dict:
    ok, reason = check_reset(token)
    if not ok:
        raise ValueError(reason)
    if len(new_password) < 8:
        raise ValueError("password must be at least 8 characters")
    row = query_one("SELECT user_id FROM password_resets WHERE token=?", (token.strip(),))
    uid = int(row["user_id"])
    with db() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (hash_password(new_password), uid))
        conn.execute("UPDATE password_resets SET used_at=? WHERE token=?",
                     (utcnow(), token.strip()))
    user = get_user_by_id(uid)
    log.info("password reset completed for user %s", user["username"])
    return {"user_id": uid, "username": user["username"], "role": user["role"]}


def request_reset(username_or_phone: str) -> dict:
    """Self-serve reset. Returns the token only if we can deliver it by SMS.

    Deliberately vague on failure: revealing whether an account exists lets
    someone enumerate your users.
    """
    ident = username_or_phone.strip().lower()
    row = query_one("SELECT * FROM users WHERE LOWER(username)=?", (ident,))
    if not row:
        digits = "".join(c for c in ident if c.isdigit())
        if len(digits) >= 10:
            row = query_one("SELECT * FROM users WHERE REPLACE(REPLACE(REPLACE("
                            "REPLACE(phone,'+',''),'-',''),' ',''),'()','') LIKE ?",
                            (f"%{digits[-10:]}%",))
    if not row or not row["phone"]:
        return {"sent": False, "reason": "no_phone"}
    return {"sent": True, "user_id": row["id"], "phone": row["phone"],
            **create_reset(row["id"], issued_by="self")}


def list_invites() -> list[dict]:
    rows = query(
        "SELECT i.*, u.username AS used_by_username FROM invites i "
        "LEFT JOIN users u ON u.id = i.used_by ORDER BY i.created_at DESC"
    )
    out = []
    for r in rows:
        d = dict(r)
        d["status"] = ("used" if r["used_by"] else
                       "revoked" if r["revoked"] else
                       "expired" if (r["expires_at"] or "") < utcnow() else "open")
        out.append(d)
    return out


def revoke_invite(code: str) -> bool:
    with db() as conn:
        cur = conn.execute("UPDATE invites SET revoked=1 WHERE code=? AND used_by IS NULL",
                           (code,))
        return cur.rowcount > 0


# --------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------
def current_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if not cred:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    payload = decode_token(cred.credentials)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")
    return user


def current_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user


def optional_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict | None:
    if not cred:
        return None
    payload = decode_token(cred.credentials)
    return get_user_by_id(int(payload["sub"])) if payload else None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="auth admin")
    p.add_argument("command", choices=["bootstrap", "invite", "invites", "revoke",
                                       "adduser", "selftest"])
    p.add_argument("--note", default="")
    p.add_argument("--code", default="")
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "bootstrap":
        print(f"admin user id: {bootstrap_admin()}")
    elif args.command == "invite":
        admin = get_user(settings.ADMIN_USERNAME) if settings.ADMIN_USERNAME else None
        inv = create_invite(admin["id"] if admin else 0, note=args.note)
        print(f"code    : {inv['code']}")
        print(f"expires : {inv['expires_at']}")
        print(f"link    : /app?invite={inv['code']}")
    elif args.command == "invites":
        for i in list_invites():
            print(f"  {i['code']:<16}{i['status']:<10}{i['note'] or '':<20}"
                  f"{i['used_by_username'] or ''}")
    elif args.command == "revoke":
        print("revoked" if revoke_invite(args.code) else "not found or already used")
    elif args.command == "adduser":
        print(f"created user id {create_user(args.username, args.password)}")
    else:
        h = hash_password("correct horse battery staple")
        print(f"  hash/verify ok : {verify_password('correct horse battery staple', h)}")
        print(f"  wrong pw fails : {not verify_password('nope', h)}")
        t = create_token(1, "tester", "admin")
        d = decode_token(t)
        print(f"  token roundtrip: {d['username'] == 'tester' and d['role'] == 'admin'}")
        print(f"  bad token None : {decode_token('garbage') is None}")
