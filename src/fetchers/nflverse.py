"""nflverse ingestion.

Key design choice: asset filenames are DISCOVERED at runtime via the GitHub
releases API, never hardcoded. nflverse reorganizes release tags periodically
(`player_stats` was renamed `stats_player`, the injuries feed died after 2024),
so any hardcoded path is a future outage.

Freshness: every release publishes `timestamp.json`. We poll that first and skip
the download entirely if nothing changed.

Stat corrections: the NFL revises stats Mon-Wed. Thursday data is the cleanest.
We record `remote_stamp` per source so the research plane can reason about what
was knowable at a given time.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.db import db, upsert_rows, utcnow

log = logging.getLogger(__name__)

GH_API = "https://api.github.com/repos/nflverse/nflverse-data/releases"
GH_DL = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

TIMEOUT = httpx.Timeout(60.0, connect=15.0)

# Tags we care about. See implementation arch doc §2.1.
# NOTE: `injuries` is historical-only (source died after 2024) — see fetchers/injuries.py
TAGS = {
    "pbp": "play-by-play, 1999-present",
    "stats_player": "weekly player box scores (renamed from player_stats)",
    "stats_team": "weekly team aggregates",
    "players": "ID crosswalk — gsis/pfr/espn/sleeper",
    "rosters": "season + weekly rosters (no separate rosters_weekly tag exists)",
    "depth_charts": "depth charts; 2025+ are timestamped, not week-keyed",
    "snap_counts": "snap share",
    "ftn_charting": "FTN manual charting, 2022+ — updates in season",
    "nextgen_stats": "NGS advanced metrics",
    "pfr_advstats": "pressures, hurries, broken tackles",
    "injuries": "HISTORICAL ONLY (2009-2024) — feed is dead",
    "draft_picks": "draft capital, rookie priors",
    "contracts": "cap hits — talent prior",
    "combine": "athletic testing",
}


@dataclass
class Asset:
    tag: str
    name: str
    url: str
    size: int
    updated_at: str


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=20))
def list_release_assets(tag: str) -> list[Asset]:
    """Discover what files actually exist under a release tag."""
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(f"{GH_API}/tags/{tag}")
        if r.status_code == 404:
            log.warning("nflverse release tag %r not found", tag)
            return []
        r.raise_for_status()
        data = r.json()
    return [
        Asset(
            tag=tag,
            name=a["name"],
            url=a["browser_download_url"],
            size=a.get("size", 0),
            updated_at=a.get("updated_at", ""),
        )
        for a in data.get("assets", [])
    ]


def remote_timestamp(tag: str) -> str | None:
    """Read a release's timestamp.json. Cheap freshness check before downloading."""
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
            r = c.get(f"{GH_DL}/{tag}/timestamp.json")
            if r.status_code != 200:
                return None
            payload = r.json()
    except Exception as exc:  # noqa: BLE001
        log.debug("timestamp probe failed for %s: %s", tag, exc)
        return None
    if isinstance(payload, dict):
        for k in ("last_updated", "updated_at", "timestamp"):
            if k in payload:
                return str(payload[k])
    return json.dumps(payload)[:120]


def pick_assets(assets: list[Asset], seasons: list[int] | None,
                prefer: str = ".parquet") -> list[Asset]:
    """Filter a release's assets down to the seasons we want.

    Handles both per-season files (play_by_play_2025.parquet) and single
    combined files (players.parquet).
    """
    typed = [a for a in assets if a.name.endswith(prefer)]
    if not typed:
        typed = [a for a in assets if a.name.endswith(".csv")]
    if seasons is None:
        return typed
    wanted = {str(s) for s in seasons}
    per_season = [a for a in typed if any(s in a.name for s in wanted)]
    # combined files have no year in the name — always keep them
    combined = [a for a in typed if not any(ch.isdigit() for ch in a.name)]
    return per_season + combined


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
def download_asset(asset: Asset, dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / asset.name
    if dest.exists() and not force and dest.stat().st_size == asset.size:
        log.debug("cached %s", asset.name)
        return dest
    log.info("downloading %s (%s KB)", asset.name, asset.size // 1024)
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        with c.stream("GET", asset.url) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
    return dest


def sync_tag(tag: str, seasons: list[int] | None = None, force: bool = False) -> list[Path]:
    """Download everything we need from one release tag. Skips if unchanged."""
    stamp = remote_timestamp(tag)
    with db() as conn:
        prev = conn.execute(
            "SELECT remote_stamp FROM source_freshness WHERE source=?", (f"nflverse:{tag}",)
        ).fetchone()
    if prev and stamp and prev["remote_stamp"] == stamp and not force:
        log.info("nflverse:%s unchanged (%s) — skipping", tag, stamp)
        return []

    assets = pick_assets(list_release_assets(tag), seasons)
    if not assets:
        log.warning("no assets matched for tag=%s seasons=%s", tag, seasons)
        return []

    dest_dir = settings.RAW_DIR / "nflverse" / tag
    paths = [download_asset(a, dest_dir, force=force) for a in assets]

    with db() as conn:
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, remote_stamp=excluded.remote_stamp, "
            "detail=excluded.detail",
            (f"nflverse:{tag}", utcnow(), stamp, f"{len(paths)} files"),
        )
    return paths


# --------------------------------------------------------------------------
# games.csv — schedules, results, closing lines, back to 1999
# --------------------------------------------------------------------------
def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_games(seasons: list[int] | None = None) -> int:
    """Fetch nfldata games.csv and upsert into `games`.

    This is the market baseline table: closing spread/total/ML back to 1999,
    plus roof, surface, rest days, referee.
    """
    import csv

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(GAMES_CSV)
        r.raise_for_status()
        text = r.text

    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for rec in reader:
        season = _to_int(rec.get("season"))
        if season is None:
            continue
        if seasons and season not in seasons:
            continue
        gameday = (rec.get("gameday") or "").strip()
        gametime = (rec.get("gametime") or "").strip()
        kickoff = f"{gameday}T{gametime}:00" if gameday and gametime else gameday or None
        rows.append({
            "game_id": rec.get("game_id"),
            "season": season,
            "week": _to_int(rec.get("week")),
            "season_type": rec.get("game_type"),
            "kickoff_utc": kickoff,          # NOTE: games.csv time is ET, not UTC
            "home_team": rec.get("home_team"),
            "away_team": rec.get("away_team"),
            "stadium": rec.get("stadium"),
            "roof": rec.get("roof"),
            "surface": rec.get("surface"),
            "referee": rec.get("referee"),
            "home_score": _to_int(rec.get("home_score")),
            "away_score": _to_int(rec.get("away_score")),
            "spread_line": _to_float(rec.get("spread_line")),
            "total_line": _to_float(rec.get("total_line")),
            "home_moneyline": _to_int(rec.get("home_moneyline")),
            "away_moneyline": _to_int(rec.get("away_moneyline")),
            "home_rest": _to_int(rec.get("home_rest")),
            "away_rest": _to_int(rec.get("away_rest")),
            "div_game": _to_int(rec.get("div_game")),
            "status": "final" if _to_int(rec.get("home_score")) is not None else "scheduled",
            "updated_at": utcnow(),
        })

    if not rows:
        return 0
    with db() as conn:
        upsert_rows(conn, "games", rows, key_cols=["game_id"])
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, detail=excluded.detail",
            ("nfldata:games.csv", utcnow(), None, f"{len(rows)} rows"),
        )
    log.info("loaded %d games", len(rows))
    return len(rows)


# --------------------------------------------------------------------------
# players — the ID crosswalk everything depends on
# --------------------------------------------------------------------------
def load_players() -> int:
    """Load the players release into `players` (gsis <-> pfr/espn/sleeper)."""
    import polars as pl

    paths = sync_tag("players", seasons=None)
    if not paths:
        cached = sorted((settings.RAW_DIR / "nflverse" / "players").glob("*.parquet"))
        if not cached:
            return 0
        paths = cached[-1:]

    df = pl.read_parquet(paths[0])
    cols = set(df.columns)

    def col(*candidates: str) -> str | None:
        return next((c for c in candidates if c in cols), None)

    mapping = {
        "player_id": col("gsis_id", "player_id"),
        "full_name": col("display_name", "full_name", "player_name"),
        "position": col("position", "position_group"),
        "team": col("latest_team", "team_abbr", "team"),
        "status": col("status"),
        "pfr_id": col("pfr_id"),
        "espn_id": col("espn_id"),
        "sleeper_id": col("sleeper_id"),
    }
    if not mapping["player_id"]:
        log.error("players parquet has no recognizable id column: %s", sorted(cols))
        return 0

    sel = {k: v for k, v in mapping.items() if v}
    out = df.select([pl.col(v).alias(k) for k, v in sel.items()])
    rows = []
    for rec in out.iter_rows(named=True):
        if not rec.get("player_id"):
            continue
        rec = {k: (str(v) if v is not None else None) for k, v in rec.items()}
        rec["updated_at"] = utcnow()
        rows.append(rec)

    with db() as conn:
        upsert_rows(conn, "players", rows, key_cols=["player_id"])
    log.info("loaded %d players", len(rows))
    return len(rows)


# --------------------------------------------------------------------------
# entry points used by the scheduler
# --------------------------------------------------------------------------
def refresh_current(seasons: list[int] | None = None) -> int:
    """In-season refresh: schedules, players, and the tags that update weekly."""
    seasons = seasons or [settings.CURRENT_SEASON]
    total = load_games(seasons=None)          # cheap; full history keeps backtests fresh
    total += load_players()
    # NOTE: `rosters_weekly` is NOT a release tag (404s) — weekly roster data
    # lives inside the `rosters` release. Confirmed via `discover`.
    for tag in ("stats_player", "stats_team", "snap_counts", "ftn_charting",
                "rosters", "depth_charts", "nextgen_stats", "pfr_advstats"):
        try:
            sync_tag(tag, seasons=seasons)
        except Exception as exc:  # noqa: BLE001 — one bad source shouldn't kill the run
            log.warning("sync_tag(%s) failed: %s", tag, exc)
    return total


def backfill(start: int = 1999, end: int | None = None) -> None:
    """One-time historical pull for the research plane. Run locally, not on Railway."""
    end = end or settings.CURRENT_SEASON
    seasons = list(range(start, end + 1))
    load_games(seasons=None)
    load_players()
    for tag in ("pbp", "stats_player", "stats_team", "snap_counts", "rosters_weekly",
                "ftn_charting", "nextgen_stats", "pfr_advstats", "injuries",
                "draft_picks", "contracts", "combine", "depth_charts"):
        try:
            paths = sync_tag(tag, seasons=seasons)
            log.info("%s: %d files", tag, len(paths))
        except Exception as exc:  # noqa: BLE001
            log.warning("backfill %s failed: %s", tag, exc)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="nflverse ingestion")
    p.add_argument("command", choices=["backfill", "refresh", "games", "players", "discover"])
    p.add_argument("--start", type=int, default=1999)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "backfill":
        backfill(args.start, args.end)
    elif args.command == "refresh":
        refresh_current()
    elif args.command == "games":
        print(f"{load_games()} games")
    elif args.command == "players":
        print(f"{load_players()} players")
    elif args.command == "discover":
        tags = [args.tag] if args.tag else list(TAGS)
        for t in tags:
            assets = list_release_assets(t)
            print(f"\n=== {t} === ({len(assets)} assets)  stamp={remote_timestamp(t)}")
            for a in assets[:8]:
                print(f"   {a.name:<45} {a.size//1024:>8} KB")
            if len(assets) > 8:
                print(f"   ... and {len(assets)-8} more")
