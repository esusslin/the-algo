"""Injury / availability ingestion — WE OWN THIS FEED.

Why this file exists
--------------------
nflverse's injury source died after the 2024 season: there is no 2025+ data and
no ETA. Availability is one of the highest-value non-market features in the
system, so we rebuild the live feed ourselves. Historical 2009-2024 nflverse
injury data is still usable for training.

Bitemporal by construction
--------------------------
Every row carries `knowledge_time` — when WE learned it, not when it happened.
The Wednesday report, the Friday report, and Sunday inactives are three
different rows, and a model predicting at Wednesday-time must not see Friday's
status. This is the whole point; do not "upsert away" older rows.

Sources (in priority order)
---------------------------
1. Sleeper  — free, no auth, includes gsis_id for crosswalk plus
              injury_status / practice_participation / body_part. Primary.
2. ESPN     — undocumented JSON. Best-effort secondary for corroboration.
3. Official NFL/team reports — authoritative; add in phase 2.

Run `python -m src.fetchers.injuries probe` to check source health before
trusting any of this in production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.db import db, insert_rows, query, utcnow

log = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(60.0, connect=15.0)

SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"
# Sleeper asks callers to hit the full players dump at most once per day (~5MB).

# ESPN endpoints are undocumented and shift. Verified via `probe`, never assumed.
ESPN_TEAM_INJURIES = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/injuries"
)

# Normalized status vocabulary. Everything maps into this.
STATUS_MAP = {
    "out": "Out",
    "ir": "IR",
    "injured reserve": "IR",
    "pup": "PUP",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "probable": "Probable",
    "sus": "Suspended",
    "suspended": "Suspended",
    "nfi": "NFI",
    "dnr": "DNR",
    "active": None,
    "healthy": None,
}

PRACTICE_MAP = {
    "dnp": "DNP",
    "did not participate": "DNP",
    "limited": "LP",
    "limited participation": "LP",
    "full": "FP",
    "full participation": "FP",
}


@dataclass
class InjuryRecord:
    player_id: str | None
    player_name: str
    team: str | None
    season: int
    week: int | None
    report_date: str | None
    knowledge_time: str
    game_status: str | None
    practice_status: str | None
    designation: str | None
    body_part: str | None
    source: str
    source_url: str | None
    ai_expected_role: str | None = None
    ai_snap_expectation: float | None = None
    ai_confidence: float | None = None


def _norm_status(raw: Any) -> str | None:
    if not raw:
        return None
    return STATUS_MAP.get(str(raw).strip().lower(), str(raw).strip().title())


def _norm_practice(raw: Any) -> str | None:
    if not raw:
        return None
    return PRACTICE_MAP.get(str(raw).strip().lower(), str(raw).strip())


# --------------------------------------------------------------------------
# source 1: Sleeper (primary)
# --------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
def fetch_sleeper() -> dict[str, dict]:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(SLEEPER_PLAYERS)
        r.raise_for_status()
        return r.json()


def parse_sleeper(payload: dict[str, dict], week: int | None,
                  season: int) -> list[InjuryRecord]:
    kt = utcnow()
    out: list[InjuryRecord] = []
    for sleeper_id, p in payload.items():
        if not isinstance(p, dict):
            continue
        status = p.get("injury_status")
        practice = p.get("practice_participation")
        if not status and not practice:
            continue  # healthy — nothing to record
        gsis = p.get("gsis_id")
        name = p.get("full_name") or (
            f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        )
        if not name:
            continue
        out.append(InjuryRecord(
            player_id=gsis or f"sleeper:{sleeper_id}",
            player_name=name,
            team=p.get("team"),
            season=season,
            week=week,
            report_date=p.get("injury_start_date"),
            knowledge_time=kt,
            game_status=_norm_status(status),
            practice_status=_norm_practice(practice),
            designation=p.get("status"),
            body_part=p.get("injury_body_part"),
            source="sleeper",
            source_url=SLEEPER_PLAYERS,
        ))
    return out


# --------------------------------------------------------------------------
# source 2: ESPN (best-effort secondary)
# --------------------------------------------------------------------------
def fetch_espn_team(team_abbr: str) -> list[dict]:
    url = ESPN_TEAM_INJURIES.format(team=team_abbr.lower())
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — ESPN is never load-bearing
        log.debug("espn %s failed: %s", team_abbr, exc)
        return []
    items: list[dict] = []
    for grp in data.get("injuries", []):
        items.extend(grp.get("injuries", []) or [])
    if not items and isinstance(data.get("items"), list):
        items = data["items"]
    return items


def parse_espn(items: Iterable[dict], team: str, week: int | None,
               season: int) -> list[InjuryRecord]:
    kt = utcnow()
    out: list[InjuryRecord] = []
    for it in items:
        athlete = it.get("athlete") or {}
        name = athlete.get("displayName") or it.get("displayName")
        if not name:
            continue
        out.append(InjuryRecord(
            player_id=str(athlete.get("id")) if athlete.get("id") else None,
            player_name=name,
            team=team,
            season=season,
            week=week,
            report_date=it.get("date"),
            knowledge_time=kt,
            game_status=_norm_status(it.get("status")),
            practice_status=None,
            designation=(it.get("type") or {}).get("description"),
            body_part=it.get("details", {}).get("type") if isinstance(it.get("details"), dict) else None,
            source="espn",
            source_url=ESPN_TEAM_INJURIES.format(team=team.lower()),
        ))
    return out


# --------------------------------------------------------------------------
# id resolution — never fuzzy-match silently
# --------------------------------------------------------------------------
def resolve_player_ids(records: list[InjuryRecord]) -> tuple[list[InjuryRecord], int]:
    """Map source-native ids/names onto canonical gsis player_id.

    Anything unresolved keeps its prefixed source id (e.g. `sleeper:1234`) and is
    counted. Quarantine, don't guess: a wrong match is worse than a missing one.
    """
    by_name: dict[str, str] = {}
    for r in query("SELECT player_id, full_name FROM players WHERE full_name IS NOT NULL"):
        by_name.setdefault(r["full_name"].strip().lower(), r["player_id"])
    for r in query("SELECT alias, player_id FROM player_aliases WHERE player_id IS NOT NULL"):
        by_name.setdefault(r["alias"].strip().lower(), r["player_id"])

    unresolved = 0
    for rec in records:
        if rec.player_id and not rec.player_id.startswith(("sleeper:", "espn:")):
            continue
        hit = by_name.get(rec.player_name.strip().lower())
        if hit:
            rec.player_id = hit
        else:
            unresolved += 1
    return records, unresolved


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def store(records: list[InjuryRecord]) -> int:
    if not records:
        return 0
    rows = [asdict(r) for r in records]
    with db() as conn:
        # INSERT OR IGNORE: knowledge_time is part of the PK, so re-running in the
        # same second is idempotent while genuinely new snapshots always append.
        return insert_rows(conn, "injuries", rows, on_conflict="ON CONFLICT DO NOTHING")


def current_week() -> tuple[int, int | None]:
    """Best-effort (season, week) from the games table."""
    row = query(
        "SELECT season, week FROM games "
        "WHERE season=? AND kickoff_utc IS NOT NULL AND status!='final' "
        "ORDER BY kickoff_utc LIMIT 1",
        (settings.CURRENT_SEASON,),
    )
    if row:
        return int(row[0]["season"]), int(row[0]["week"]) if row[0]["week"] else None
    return settings.CURRENT_SEASON, None


# --------------------------------------------------------------------------
# entry point used by the scheduler
# --------------------------------------------------------------------------
def refresh(use_espn: bool = True) -> int:
    season, week = current_week()
    records: list[InjuryRecord] = []

    try:
        records += parse_sleeper(fetch_sleeper(), week, season)
        log.info("sleeper: %d injury records", len(records))
    except Exception as exc:  # noqa: BLE001
        log.error("sleeper injury fetch failed: %s", exc)

    if use_espn:
        teams = [r["team_abbr"] for r in query("SELECT team_abbr FROM teams")]
        if not teams:
            teams = [r["home_team"] for r in query(
                "SELECT DISTINCT home_team FROM games WHERE season=?", (settings.CURRENT_SEASON,)
            ) if r["home_team"]]
        got = 0
        for t in teams:
            items = fetch_espn_team(t)
            if items:
                parsed = parse_espn(items, t, week, season)
                records += parsed
                got += len(parsed)
        log.info("espn: %d injury records across %d teams", got, len(teams))

    records, unresolved = resolve_player_ids(records)
    n = store(records)

    with db() as conn:
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, detail=excluded.detail",
            ("injuries", utcnow(), None, f"{n} rows, {unresolved} unresolved ids"),
        )
    if unresolved:
        log.warning("%d injury records could not be resolved to a gsis id", unresolved)
    return n


def probe() -> None:
    """Check every source is alive and shaped as expected. Run before trusting output."""
    print("=== Sleeper ===")
    try:
        data = fetch_sleeper()
        inj = [p for p in data.values()
               if isinstance(p, dict) and (p.get("injury_status") or p.get("practice_participation"))]
        with_gsis = [p for p in inj if p.get("gsis_id")]
        print(f"  OK  {len(data)} players, {len(inj)} with injury info, "
              f"{len(with_gsis)} carry gsis_id")
        if inj:
            s = inj[0]
            print(f"  sample: {s.get('full_name')} | {s.get('team')} | "
                  f"{s.get('injury_status')} | practice={s.get('practice_participation')} | "
                  f"part={s.get('injury_body_part')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {exc}")

    print("\n=== ESPN (best-effort) ===")
    for t in ("kc", "buf", "phi"):
        items = fetch_espn_team(t)
        print(f"  {t}: {len(items)} items" if items else f"  {t}: no data / endpoint changed")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="injury feed")
    p.add_argument("command", choices=["refresh", "probe"])
    p.add_argument("--no-espn", action="store_true")
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "probe":
        probe()
    else:
        print(f"{refresh(use_espn=not args.no_espn)} injury rows stored")
