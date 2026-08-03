"""The Algo — NFL betting model + pick engine.

Single FastAPI service. All scheduled jobs run in-process via APScheduler,
matching the baseball app's pattern.

    uvicorn server:app --reload

Deployment gotchas (carried over from baseball):
  * SQLite on a mounted volume -> NEVER run more than one replica.
  * APScheduler is in-process -> if the process dies, jobs stop silently.
    /health reports per-job last-success so an external pinger can catch it.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src import auth
from src.config import settings
from src.db import db, insert_row, job_run, query, run_migrations, utcnow

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("the-algo")

scheduler = BackgroundScheduler(timezone=settings.TZ)
TEMPLATES = Path(__file__).parent / "templates"


# ==========================================================================
# jobs
# ==========================================================================
def _refresh_nflverse() -> None:
    from src.fetchers import nflverse
    with job_run("refresh_nflverse") as ctx:
        ctx["rows_affected"] = nflverse.refresh_current()


def _refresh_injuries() -> None:
    from src.fetchers import injuries
    with job_run("refresh_injuries") as ctx:
        ctx["rows_affected"] = injuries.refresh()


def _poll_odds() -> None:
    """Poll odds, then immediately reprice. Fair values are only useful fresh."""
    from src.fetchers import odds_api
    from src.market.consensus import build_fair_prices
    with job_run("poll_odds") as ctx:
        result = odds_api.poll()
        ctx["rows_affected"] = result.get("changed", 0)
    if result.get("written"):
        with job_run("build_fair_prices") as ctx:
            ctx["rows_affected"] = build_fair_prices()
        with job_run("generate_picks") as ctx:
            from src.picks.generator import generate
            ctx["rows_affected"] = generate(source="market_engine")["written"]


def _link_odds_events() -> None:
    from src.fetchers import odds_api
    with job_run("link_odds_events") as ctx:
        ctx["rows_affected"] = odds_api.link_events()


def _health_heartbeat() -> None:
    with job_run("heartbeat") as ctx:
        ctx["rows_affected"] = 1


# Jobs registered now; the rest land as their modules are built (see build calendar).
JOBS: list[tuple] = [
    # (func, trigger, kwargs, id)
    (_refresh_nflverse, "cron", dict(day_of_week="tue", hour=7, minute=0), "refresh_nflverse_tue"),
    (_refresh_nflverse, "cron", dict(day_of_week="thu", hour=7, minute=0), "refresh_nflverse_thu"),
    (_refresh_injuries, "cron", dict(day_of_week="wed,thu,fri", hour=17, minute=0), "refresh_injuries"),
    # poll_odds runs often but self-throttles per tier and time-to-kickoff
    # (see odds_api.TIERS) — the 5-minute interval is an upper bound, not a rate.
    (_poll_odds, "interval", dict(minutes=5), "poll_odds"),
    (_link_odds_events, "cron", dict(day_of_week="tue", hour=8, minute=0), "link_odds_events"),
    (_health_heartbeat, "interval", dict(minutes=30), "heartbeat"),
]

# Max staleness per *job_run name* (the name passed to job_run(), not the APScheduler id).
EXPECTED_FRESHNESS = {
    "heartbeat": timedelta(hours=2),
    "poll_odds": timedelta(hours=3),
    "refresh_injuries": timedelta(days=8),
    "refresh_nflverse": timedelta(days=9),
}

# A job that has never run is only "degraded" once the process has been up long
# enough that it should have fired. Without this, the first Railway deploy 503s
# its own healthcheck and the deploy fails.
STARTUP_GRACE = timedelta(hours=3)
BOOT_TIME = datetime.now(timezone.utc)


def register_jobs() -> None:
    for func, trigger, kwargs, job_id in JOBS:
        scheduler.add_job(
            func, trigger, id=job_id, replace_existing=True,
            coalesce=True, max_instances=1, misfire_grace_time=3600, **kwargs,
        )
        log.info("registered job %s (%s %s)", job_id, trigger, kwargs)


# ==========================================================================
# lifecycle
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    version = run_migrations()
    log.info("schema version %s at %s", version, settings.DATABASE_PATH)

    problems = settings.validate()
    for p in problems:
        log.error("CONFIG: %s", p)
    if problems and settings.IS_PROD:
        log.error("starting with %d config problems — fix these", len(problems))

    try:
        from src.teams import seed_teams_table
        seed_teams_table()
        admin_id = auth.bootstrap_admin()
        if admin_id:
            log.info("admin user ready (id=%s)", admin_id)
    except Exception as exc:  # noqa: BLE001 — never block startup on seeding
        log.error("startup seeding failed: %s", exc)

    register_jobs()
    scheduler.start()
    log.info("scheduler started (tz=%s)", settings.TZ)
    log.info(
        "flags: publish_model_picks=%s props=%s simulator=%s sms=%s",
        settings.PUBLISH_MODEL_PICKS, settings.ENABLE_PROPS,
        settings.ENABLE_SIMULATOR, settings.ENABLE_SMS,
    )
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        log.info("scheduler stopped")


app = FastAPI(title="The Algo", version="0.1.0", lifespan=lifespan)


# ==========================================================================
# routes
# ==========================================================================
@app.get("/health")
def health() -> JSONResponse:
    """Per-job health, not just HTTP 200.

    A dead in-process scheduler still serves 200s, so an uptime pinger checking
    only status code learns nothing. This reports staleness per job.
    """
    now = datetime.now(timezone.utc)
    warming_up = (now - BOOT_TIME) < STARTUP_GRACE
    jobs: dict[str, dict] = {}
    degraded: list[str] = []

    for job_name, max_age in EXPECTED_FRESHNESS.items():
        rows = query(
            "SELECT status, finished_at, error FROM job_runs "
            "WHERE job_name=? ORDER BY id DESC LIMIT 1",
            (job_name,),
        )
        if not rows:
            jobs[job_name] = {"status": "pending" if warming_up else "never_run"}
            if not warming_up:
                degraded.append(job_name)
            continue
        r = rows[0]
        try:
            age = now - datetime.fromisoformat(r["finished_at"])
        except (TypeError, ValueError):
            age = timedelta.max
        stale = age > max_age
        jobs[job_name] = {
            "status": r["status"],
            "last_finished": r["finished_at"],
            "age_seconds": int(age.total_seconds()) if age != timedelta.max else None,
            "stale": stale,
            "error": r["error"],
        }
        if stale or r["status"] != "success":
            degraded.append(job_name)

    sources = {
        r["source"]: {"last_success": r["last_success"], "detail": r["detail"]}
        for r in query("SELECT source, last_success, detail FROM source_freshness")
    }

    # Liveness (does Railway keep this container?) is deliberately looser than
    # health (should a human look at this?). A stale data job means degraded
    # data, not a dead process — restarting the container would not fix it and
    # would only interrupt the scheduler.
    alive = scheduler.running
    ok = alive and not degraded
    return JSONResponse(
        status_code=200 if alive else 503,
        content={
            "ok": ok,
            "alive": alive,
            "warming_up": warming_up,
            "scheduler_running": scheduler.running,
            "degraded": degraded,
            "jobs": jobs,
            "sources": sources,
            "flags": {
                "publish_model_picks": settings.PUBLISH_MODEL_PICKS,
                "props": settings.ENABLE_PROPS,
                "simulator": settings.ENABLE_SIMULATOR,
                "sms": settings.ENABLE_SMS,
            },
            "checked_at": utcnow(),
        },
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def app_page() -> HTMLResponse:
    return HTMLResponse((TEMPLATES / "app.html").read_text(encoding="utf-8"))


@app.get("/static/manifest.json")
def manifest() -> dict:
    """PWA manifest — lets iOS/Android 'add to home screen' run it fullscreen."""
    return {
        "name": "The Algo", "short_name": "Algo",
        "start_url": "/app", "display": "standalone",
        "background_color": "#0b0f14", "theme_color": "#0b0f14",
        "icons": [],
    }


@app.get("/api/stats")
def stats() -> dict:
    """Quick counts — useful during the build to confirm ingestion is working."""
    out = {}
    for table in ("games", "players", "injuries", "odds_current", "picks"):
        try:
            out[table] = query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        except Exception:  # noqa: BLE001
            out[table] = None
    return out


# ==========================================================================
# auth
# ==========================================================================
class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    invite: str
    username: str
    password: str
    phone: str | None = None


class BetBody(BaseModel):
    pick_id: int
    book: str | None = None
    price: int | None = None
    stake: float | None = None
    note: str | None = None


class InviteBody(BaseModel):
    note: str = ""
    phone: str | None = None      # if set, texts the link via Twilio


@app.post("/api/auth/login")
def login(body: LoginBody) -> dict:
    user = auth.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(401, "invalid username or password")
    return {
        "token": auth.create_token(user["id"], user["username"], user["role"]),
        "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
    }


@app.get("/api/auth/invite/{code}")
def check_invite(code: str) -> dict:
    ok, reason = auth.check_invite(code)
    return {"valid": ok, "reason": reason}


@app.post("/api/auth/register")
def register(body: RegisterBody) -> dict:
    try:
        user = auth.redeem_invite(body.invite, body.username, body.password, body.phone)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "token": auth.create_token(user["user_id"], user["username"], user["role"]),
        "user": user,
    }


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.current_user)) -> dict:
    return {"id": user["id"], "username": user["username"], "role": user["role"],
            "sms_opt_in": user["sms_opt_in"]}


# ==========================================================================
# picks, bets, history
# ==========================================================================
@app.get("/api/picks")
def picks(filter: str = "all", user: dict = Depends(auth.current_user)) -> dict:
    """Current slate, filterable by bet type."""
    from src.markets import describe_market, matches_filter, side_label
    from src.picks.generator import current_slate

    include_dark = user["role"] == "admin"
    rows = current_slate(include_unpublished=include_dark)

    mine = {r["pick_id"] for r in query(
        "SELECT pick_id FROM user_bets WHERE user_id=?", (user["id"],))}
    # dict(), not the raw Row — sqlite3.Row has no .get()
    games = {r["game_id"]: dict(r) for r in query(
        "SELECT game_id, home_team, away_team, kickoff_utc FROM games")}
    names = {r["player_id"]: r["full_name"] for r in query(
        "SELECT player_id, full_name FROM players WHERE full_name IS NOT NULL")}

    out = []
    for r in rows:
        if not matches_filter(r["market_type"], filter):
            continue
        info = describe_market(r["market_type"])
        g = games.get(r["game_id"], {})
        out.append({
            **r,
            "market_label": info.label,
            "bet_class": info.bet_class,
            "description": side_label(r["market_type"], r["side"], r["line"] or 0,
                                      g.get("home_team", ""), g.get("away_team", ""),
                                      names.get(r["player_id"], "")),
            "matchup": f"{g.get('away_team','')} @ {g.get('home_team','')}",
            "kickoff_utc": g.get("kickoff_utc"),
            "i_bet_this": r["pick_id"] in mine,
        })
    return {
        "count": len(out),
        "by_tier": {t: sum(1 for r in out if r["tier"] == t) for t in ("A", "B", "C")},
        "picks": out,
    }


@app.post("/api/bets")
def log_bet(body: BetBody, user: dict = Depends(auth.current_user)) -> dict:
    """Log that you took a pick, at the price YOU actually got.

    Defaults to the recommended book/price/stake, but the whole point of the
    edit is honesty: logging the recommended price when you got a worse one
    flatters your CLV and defeats the tracking.
    """
    pick = query("SELECT * FROM picks WHERE pick_id=?", (body.pick_id,))
    if not pick:
        raise HTTPException(404, "pick not found")
    p = pick[0]
    dup = query("SELECT id FROM user_bets WHERE user_id=? AND pick_id=?",
                (user["id"], body.pick_id))
    if dup:
        raise HTTPException(409, "already logged this pick")
    with db() as conn:
        bet_id = insert_row(conn, "user_bets", {
            "user_id": user["id"], "pick_id": body.pick_id,
            "book": body.book or p["best_book"],
            "price": body.price if body.price is not None else p["best_price"],
            "stake": body.stake if body.stake is not None else p["kelly_units"],
            "placed_at": utcnow(), "result": "pending", "note": body.note,
        })
    return {"ok": True, "bet_id": bet_id}


@app.delete("/api/bets/{bet_id}")
def unlog_bet(bet_id: int, user: dict = Depends(auth.current_user)) -> dict:
    with db() as conn:
        cur = conn.execute("DELETE FROM user_bets WHERE id=? AND user_id=? "
                           "AND result='pending'", (bet_id, user["id"]))
    return {"ok": cur.rowcount > 0}


@app.get("/api/history/overall")
def history_overall(filter: str = "all", season: int | None = None,
                    week: int | None = None,
                    user: dict = Depends(auth.current_user)) -> dict:
    from src.picks.history import overall_history
    return overall_history(filter, season=season, week=week)


@app.get("/api/history/mine")
def history_mine(filter: str = "all",
                 user: dict = Depends(auth.current_user)) -> dict:
    from src.picks.history import user_history
    return user_history(user["id"], filter)


@app.get("/api/history/filters")
def history_filters(user: dict = Depends(auth.current_user)) -> dict:
    from src.markets import PRIMARY_FILTERS, SECONDARY_FILTERS
    from src.picks.history import filter_options
    return {"options": filter_options(),
            "primary": PRIMARY_FILTERS, "secondary": SECONDARY_FILTERS}


@app.get("/api/live")
def live(user: dict = Depends(auth.current_user)) -> dict:
    """In-progress state for this user's open bets."""
    from src.picks.live import open_bet_status
    return open_bet_status(user["id"])


# ==========================================================================
# admin
# ==========================================================================
@app.post("/api/admin/invites")
def make_invite(request: Request, body: InviteBody,
                admin: dict = Depends(auth.current_admin)) -> dict:
    inv = auth.create_invite(admin["id"], note=body.note)
    inv["link"] = f"{str(request.base_url).rstrip('/')}/app?invite={inv['code']}"
    if body.phone:
        from src.notify.sms import send_invite
        inv["sms"] = send_invite(inv["code"], body.phone,
                                 str(request.base_url), inviter=admin["username"])
    return inv


@app.get("/api/admin/sms-status")
def sms_status(admin: dict = Depends(auth.current_admin)) -> dict:
    from src.notify.sms import configured
    recent = query("SELECT * FROM invite_sends ORDER BY id DESC LIMIT 20")
    return {"configured": configured(), "enabled": settings.ENABLE_SMS,
            "recent": [dict(r) for r in recent]}


@app.get("/api/algorithm")
def algorithm(user: dict = Depends(auth.current_user)) -> dict:
    """How picks are calculated — the explanation shown in the Algorithm tab."""
    from src.picks.generator import TIER_RULES
    return {
        "tier_rules": TIER_RULES,
        "config": {
            "min_edge_pct": settings.MIN_EDGE_PCT,
            "kelly_fraction": settings.KELLY_FRACTION,
            "max_bet_pct_bankroll": settings.MAX_BET_PCT_BANKROLL,
            "devig_method": settings.DEVIG_METHOD,
            "sharp_books": settings.SHARP_BOOKS,
            "bettable_books": settings.BETTABLE_BOOKS,
        },
        "flags": {
            "publish_model_picks": settings.PUBLISH_MODEL_PICKS,
            "props": settings.ENABLE_PROPS,
            "simulator": settings.ENABLE_SIMULATOR,
        },
    }


@app.get("/api/admin/invites")
def get_invites(admin: dict = Depends(auth.current_admin)) -> dict:
    return {"invites": auth.list_invites()}


@app.delete("/api/admin/invites/{code}")
def kill_invite(code: str, admin: dict = Depends(auth.current_admin)) -> dict:
    return {"ok": auth.revoke_invite(code)}


@app.get("/api/admin/users")
def list_users(admin: dict = Depends(auth.current_admin)) -> dict:
    rows = query("SELECT id, username, role, created_at, last_login FROM users "
                 "ORDER BY created_at")
    return {"users": [dict(r) for r in rows]}


@app.get("/api/edges")
def edges(min_edge: float = 2.0) -> dict:
    """Raw opportunity list before tiering — the admin/debug view."""
    from src.market.shop import find_opportunities
    opps = find_opportunities(min_edge=min_edge)
    return {"count": len(opps), "min_edge": min_edge, "edges": opps[:100]}


@app.post("/api/admin/run/{job_id}")
def run_job_now(job_id: str) -> dict:
    """Manual trigger. TODO: gate behind admin auth before deploying."""
    job = scheduler.get_job(job_id)
    if not job:
        return {"ok": False, "error": f"unknown job {job_id}",
                "available": [j.id for j in scheduler.get_jobs()]}
    job.modify(next_run_time=datetime.now(timezone.utc))
    return {"ok": True, "job": job_id, "scheduled": "now"}
