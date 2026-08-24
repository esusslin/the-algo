"""Weather via Open-Meteo — free, no key, no rate limit for non-commercial use.

Why not use games.csv weather
-----------------------------
nflverse's `temp` and `wind` are a single scalar reported at kickoff, frequently
missing, and carry no precipitation, humidity or gust data. Open-Meteo gives an
hourly series at exact stadium coordinates.

Gusts matter more than mean wind. Kicking and deep passing degrade non-linearly
above roughly 15mph, and a game with 8mph average wind gusting to 25 plays very
differently from a steady 8. Books are slow to adjust totals for forecast
changes, which is a genuine edge source.

Two endpoints:
  forecast  — upcoming games, refreshed as kickoff approaches
  archive   — ERA5 reanalysis back to 1940, for building training features
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.db import db, query, upsert_rows, utcnow
from src.teams import BY_ABBR

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

HOURLY = ["temperature_2m", "wind_speed_10m", "wind_gusts_10m",
          "wind_direction_10m", "precipitation", "snowfall",
          "relative_humidity_2m", "surface_pressure"]

# Wind above this materially suppresses passing efficiency and FG range.
HIGH_WIND_KPH = 24.0          # ~15 mph


def _controlled(roof: str | None) -> bool:
    """Dome or closed retractable — weather is irrelevant, and treating it as a
    normal outdoor game injects noise into totals models."""
    return (roof or "").lower() in {"dome", "closed", "retractable"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=20))
def _get(url: str, params: dict) -> dict:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()


def _nearest_hour(payload: dict, target: datetime) -> dict | None:
    """Pick the hourly reading closest to kickoff."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    best_i, best_gap = None, None
    for i, t in enumerate(times):
        try:
            ts = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        gap = abs((ts - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best_i, best_gap = i, gap
    if best_i is None:
        return None

    def val(key: str):
        arr = hourly.get(key) or []
        return arr[best_i] if best_i < len(arr) else None

    return {
        "temp_c": val("temperature_2m"),
        "wind_kph": val("wind_speed_10m"),
        "wind_gust_kph": val("wind_gusts_10m"),
        "wind_dir": val("wind_direction_10m"),
        "precip_mm": val("precipitation"),
        "snow_mm": val("snowfall"),
        "humidity": val("relative_humidity_2m"),
        "pressure": val("surface_pressure"),
    }


def fetch_for_games(games: list[dict], historical: bool = False) -> int:
    """Snapshot weather for a list of games. Bitemporal: each call appends a new
    row keyed on knowledge_time, so a Tuesday forecast and a Sunday forecast are
    both preserved and a model can only see what was knowable at its decision
    time."""
    rows: list[dict] = []
    kt = utcnow()

    for g in games:
        team = BY_ABBR.get(g.get("home_team") or "")
        if not team:
            continue
        roof = g.get("roof") or team.roof
        if _controlled(roof):
            # Record explicitly rather than skipping — "controlled environment"
            # is itself a feature, and absence would look like missing data.
            rows.append({
                "game_id": g["game_id"], "knowledge_time": kt,
                "temp_c": 21.0, "wind_kph": 0.0, "wind_gust_kph": 0.0,
                "wind_dir": None, "precip_mm": 0.0, "snow_mm": 0.0,
                "humidity": 50.0, "pressure": None,
                "is_forecast": 0,
            })
            continue

        try:
            kickoff = datetime.fromisoformat(
                (g["kickoff_utc"] or "").replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        day = kickoff.date().isoformat()
        params = {
            "latitude": team.lat, "longitude": team.lon,
            "hourly": ",".join(HOURLY), "timezone": "UTC",
            "start_date": day, "end_date": day,
            "wind_speed_unit": "kmh",
        }
        try:
            payload = _get(ARCHIVE_URL if historical else FORECAST_URL, params)
        except Exception as exc:  # noqa: BLE001 — one bad game shouldn't kill the run
            log.warning("weather fetch failed for %s: %s", g["game_id"], exc)
            continue

        reading = _nearest_hour(payload, kickoff)
        if not reading:
            continue
        rows.append({
            "game_id": g["game_id"], "knowledge_time": kt,
            **reading, "is_forecast": 0 if historical else 1,
        })

    if rows:
        with db() as conn:
            upsert_rows(conn, "weather_snapshots", rows,
                        key_cols=["game_id", "knowledge_time"])
    return len(rows)


def refresh_upcoming(days_ahead: int = 10) -> int:
    """Forecast for games kicking off soon. Called on a schedule.

    **Do not widen `days_ahead` past ~14.** Open-Meteo's forecast endpoint serves roughly
    fifteen days and returns 400 beyond that — `"Parameter 'start_date' is out of allowed
    range"` — which surfaces here as a RetryError wrapping an HTTPStatusError, three
    attempts per game, for every game out of range. Verified 24 Aug 2026: a date 2 days
    out returned 200, 16 days out returned 400.

    The consequence is that in the preseason this legitimately writes zero rows and marks
    itself successful, because no game is within the window yet. `source_freshness.detail`
    records `"0 of N games"` so the difference between "nothing to do" and "nothing
    worked" is visible — but `/health` cannot distinguish them at the `ok` level, which is
    worth remembering before diagnosing an empty `weather_snapshots` table in August.

    In season it is never a constraint: games are always within a week.
    """
    cutoff = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
    games = [dict(r) for r in query(
        "SELECT game_id, home_team, kickoff_utc, roof FROM games "
        "WHERE status!='final' AND kickoff_utc IS NOT NULL "
        "AND kickoff_utc <= ? ORDER BY kickoff_utc", (cutoff,)
    )]
    if not games:
        return 0
    n = fetch_for_games(games, historical=False)
    with db() as conn:
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, detail=excluded.detail",
            ("weather", utcnow(), None, f"{n} of {len(games)} games"),
        )
    log.info("weather: %d of %d upcoming games", n, len(games))
    return n


def backfill(season_from: int = 1999, limit: int | None = None) -> int:
    """Historical weather for training features. Run locally, not on Railway —
    this is thousands of requests."""
    sql = ("SELECT game_id, home_team, kickoff_utc, roof FROM games "
           "WHERE season >= ? AND kickoff_utc IS NOT NULL "
           "AND game_id NOT IN (SELECT game_id FROM weather_snapshots) "
           "ORDER BY kickoff_utc")
    if limit:
        sql += f" LIMIT {int(limit)}"
    games = [dict(r) for r in query(sql, (season_from,))]
    log.info("backfilling weather for %d games", len(games))
    return fetch_for_games(games, historical=True)


def current_for(game_id: str) -> dict | None:
    """Latest snapshot, with derived flags for feature use."""
    rows = query(
        "SELECT * FROM weather_snapshots WHERE game_id=? "
        "ORDER BY knowledge_time DESC LIMIT 1", (game_id,))
    if not rows:
        return None
    w = dict(rows[0])
    gust = w.get("wind_gust_kph") or 0
    wind = w.get("wind_kph") or 0
    w["high_wind"] = max(gust, wind) >= HIGH_WIND_KPH
    w["freezing"] = (w.get("temp_c") or 99) <= 0
    w["wet"] = (w.get("precip_mm") or 0) > 0.2 or (w.get("snow_mm") or 0) > 0
    return w


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="weather")
    p.add_argument("command", choices=["refresh", "backfill", "show", "probe"])
    p.add_argument("--game", default="")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "refresh":
        print(f"{refresh_upcoming()} games updated")
    elif args.command == "backfill":
        print(f"{backfill(limit=args.limit)} historical snapshots")
    elif args.command == "show":
        w = current_for(args.game)
        print(w if w else "no snapshot for that game")
    else:
        # one live call against a known stadium — verifies the API and units
        team = BY_ABBR["BUF"]
        payload = _get(FORECAST_URL, {
            "latitude": team.lat, "longitude": team.lon,
            "hourly": ",".join(HOURLY), "timezone": "UTC",
            "wind_speed_unit": "kmh", "forecast_days": 1})
        r = _nearest_hour(payload, datetime.now(timezone.utc))
        print(f"  {team.stadium} ({team.lat}, {team.lon})")
        print(f"  {r}")
        print(f"  units: {payload.get('hourly_units')}")
