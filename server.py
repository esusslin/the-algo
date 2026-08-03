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

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import settings
from src.db import db, job_run, query, run_migrations, utcnow

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("the-algo")

scheduler = BackgroundScheduler(timezone=settings.TZ)


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


@app.get("/")
def root() -> dict:
    return {"service": "the-algo", "status": "up", "docs": "/docs", "health": "/health"}


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


@app.get("/api/picks")
def picks(include_dark: bool = False) -> dict:
    """Current slate. `include_dark` exposes unpublished model picks (admin)."""
    from src.picks.generator import current_slate
    rows = current_slate(include_unpublished=include_dark)
    return {
        "count": len(rows),
        "by_tier": {t: sum(1 for r in rows if r["tier"] == t) for t in ("A", "B", "C")},
        "picks": rows,
    }


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
