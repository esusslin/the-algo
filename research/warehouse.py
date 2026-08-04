"""DuckDB warehouse over the nflverse parquet — the research plane's foundation.

LOCAL ONLY. Never imported by server.py, never runs on Railway.

Design
------
DuckDB reads parquet directly, so there is no import step and no second copy of
the data. Views are defined over `data/raw/nflverse/**`, and a handful of
materialised tables exist only where repeated aggregation is expensive.

Column drift is the recurring hazard: nflverse renames and adds fields across
seasons, and a hardcoded column list will break on some subset of years. Every
view resolves columns at build time from what's actually present and reports
what's missing rather than failing.

    python -m research.warehouse build
    python -m research.warehouse verify
    python -m research.warehouse sql "select season, count(*) from pbp group by 1"
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from src.config import settings

log = logging.getLogger(__name__)

WAREHOUSE = settings.DATA_DIR / "warehouse.duckdb"
RAW = settings.RAW_DIR / "nflverse"

# Plays that represent a real offensive snap. Kneels, spikes and special teams
# distort efficiency metrics and must be excluded from unit ratings.
REAL_PLAY = ("pass = 1 OR rush = 1")
NEUTRAL_SCRIPT = "wp BETWEEN 0.10 AND 0.90"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(WAREHOUSE), read_only=read_only)


def _glob(tag: str, pattern: str = "*.parquet") -> str | None:
    d = RAW / tag
    if not d.exists():
        return None
    files = sorted(d.glob(pattern))
    return str(d / pattern) if files else None


def available_columns(con: duckdb.DuckDBPyConnection, source: str) -> set[str]:
    try:
        rows = con.execute(
            f"SELECT * FROM read_parquet('{source}') LIMIT 0").description
        return {r[0] for r in rows}
    except Exception:  # noqa: BLE001
        return set()


def _pick(cols: set[str], *candidates: str) -> str | None:
    """First column that actually exists. nflverse renames things."""
    return next((c for c in candidates if c in cols), None)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(verbose: bool = True) -> dict:
    con = connect()
    built: dict[str, int] = {}
    missing: list[str] = []

    pbp = _glob("pbp", "play_by_play_*.parquet")
    if not pbp:
        raise SystemExit(
            f"no play-by-play parquet under {RAW / 'pbp'} — run:\n"
            f"  python -m src.fetchers.nflverse backfill --start 1999")

    cols = available_columns(con, pbp)
    if verbose:
        log.info("pbp has %d columns", len(cols))

    # ---- raw pbp view ----
    con.execute(f"CREATE OR REPLACE VIEW pbp AS SELECT * FROM read_parquet('{pbp}')")
    built["pbp"] = con.execute("SELECT COUNT(*) FROM pbp").fetchone()[0]

    # ---- plays: one row per meaningful offensive snap, with derived flags ----
    epa = _pick(cols, "epa")
    wp = _pick(cols, "wp", "wpa")
    air = _pick(cols, "air_yards")
    yac = _pick(cols, "yards_after_catch")
    for name, c in [("epa", epa), ("wp", wp)]:
        if not c:
            missing.append(f"pbp.{name}")

    depth_bucket = (
        f"CASE WHEN {air} IS NULL THEN NULL "
        f"WHEN {air} < 0 THEN 'behind' "
        f"WHEN {air} <= 9 THEN 'short' "
        f"WHEN {air} <= 19 THEN 'intermediate' "
        f"ELSE 'deep' END" if air else "NULL")

    con.execute(f"""
        CREATE OR REPLACE VIEW plays AS
        SELECT
            game_id, season, week, posteam AS offense, defteam AS defense,
            play_type, down, ydstogo, yardline_100, qtr,
            CASE WHEN pass = 1 THEN 'pass' WHEN rush = 1 THEN 'rush' END AS play_class,
            {epa or 'NULL'} AS epa,
            CASE WHEN {epa or 'NULL'} > 0 THEN 1 ELSE 0 END AS success,
            {wp or 'NULL'} AS wp,
            CASE WHEN {wp or 'NULL'} BETWEEN 0.10 AND 0.90 THEN 1 ELSE 0 END AS neutral,
            CASE WHEN down <= 2 THEN 1 ELSE 0 END AS early_down,
            CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END AS red_zone,
            {air or 'NULL'} AS air_yards,
            {yac or 'NULL'} AS yac,
            {depth_bucket} AS depth_bucket,
            passer_player_id, rusher_player_id, receiver_player_id,
            CASE WHEN yards_gained >= 20 THEN 1 ELSE 0 END AS explosive
        FROM pbp
        WHERE ({REAL_PLAY}) AND posteam IS NOT NULL AND defteam IS NOT NULL
    """)
    built["plays"] = con.execute("SELECT COUNT(*) FROM plays").fetchone()[0]

    # ---- team_weeks: the unit-level aggregate the ratings model consumes ----
    # Split by play class and situation, because a team-level average hides the
    # thing that actually predicts (see MATCHUP_MODELING.md).
    con.execute("""
        CREATE OR REPLACE TABLE team_weeks AS
        SELECT
            season, week, offense AS team, defense AS opponent, game_id,
            COUNT(*) AS plays,
            AVG(epa) AS off_epa,
            AVG(CASE WHEN play_class='pass' THEN epa END) AS off_pass_epa,
            AVG(CASE WHEN play_class='rush' THEN epa END) AS off_rush_epa,
            AVG(CASE WHEN neutral=1 THEN epa END) AS off_epa_neutral,
            AVG(CASE WHEN early_down=1 THEN epa END) AS off_epa_early,
            AVG(success::DOUBLE) AS off_success,
            AVG(CASE WHEN play_class='pass' THEN success::DOUBLE END) AS off_pass_success,
            AVG(CASE WHEN play_class='rush' THEN success::DOUBLE END) AS off_rush_success,
            AVG(explosive::DOUBLE) AS off_explosive_rate,
            AVG(CASE WHEN red_zone=1 THEN epa END) AS off_epa_redzone,
            SUM(CASE WHEN play_class='pass' THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(COUNT(*),0) AS pass_rate,
            AVG(CASE WHEN play_class='pass' THEN air_yards END) AS avg_air_yards
        FROM plays
        GROUP BY season, week, offense, defense, game_id
    """)
    built["team_weeks"] = con.execute("SELECT COUNT(*) FROM team_weeks").fetchone()[0]

    # ---- defensive splits by target depth: the core matchup signal ----
    con.execute("""
        CREATE OR REPLACE TABLE defense_depth_weeks AS
        SELECT season, week, defense AS team, game_id, depth_bucket,
               COUNT(*) AS plays, AVG(epa) AS epa_allowed,
               AVG(success::DOUBLE) AS success_allowed
        FROM plays
        WHERE play_class='pass' AND depth_bucket IS NOT NULL
        GROUP BY season, week, defense, game_id, depth_bucket
    """)
    built["defense_depth_weeks"] = con.execute(
        "SELECT COUNT(*) FROM defense_depth_weeks").fetchone()[0]

    # ---- player_weeks: usage shares, the backbone of prop volume ----
    con.execute("""
        CREATE OR REPLACE TABLE player_weeks AS
        WITH team_totals AS (
            -- Denominators must count only plays that CAN be attributed to a
            -- player. Roughly a quarter of dropbacks are sacks, throwaways,
            -- spikes or scrambles with no receiver — counting those makes every
            -- target share too small and they stop summing to 1.
            SELECT game_id, offense,
                   SUM(CASE WHEN play_class='pass' AND receiver_player_id IS NOT NULL
                            THEN 1 ELSE 0 END) AS team_targets,
                   SUM(CASE WHEN play_class='rush' AND rusher_player_id IS NOT NULL
                            THEN 1 ELSE 0 END) AS team_carries,
                   SUM(CASE WHEN receiver_player_id IS NOT NULL
                            THEN COALESCE(air_yards,0) ELSE 0 END) AS team_air_yards
            FROM plays GROUP BY game_id, offense
        ),
        rec AS (
            SELECT game_id, season, week, offense AS team,
                   receiver_player_id AS player_id,
                   COUNT(*) AS targets, SUM(COALESCE(air_yards,0)) AS air_yards,
                   AVG(epa) AS epa_per_target
            FROM plays WHERE receiver_player_id IS NOT NULL
            GROUP BY 1,2,3,4,5
        ),
        rush AS (
            SELECT game_id, rusher_player_id AS player_id,
                   COUNT(*) AS carries, AVG(epa) AS epa_per_carry
            FROM plays WHERE rusher_player_id IS NOT NULL
            GROUP BY 1,2
        )
        SELECT
            COALESCE(rec.game_id, rush.game_id) AS game_id,
            rec.season, rec.week, rec.team,
            COALESCE(rec.player_id, rush.player_id) AS player_id,
            COALESCE(targets,0) AS targets,
            COALESCE(carries,0) AS carries,
            COALESCE(air_yards,0) AS air_yards,
            epa_per_target, epa_per_carry,
            targets::DOUBLE / NULLIF(tt.team_targets,0) AS target_share,
            carries::DOUBLE / NULLIF(tt.team_carries,0) AS carry_share,
            air_yards::DOUBLE / NULLIF(tt.team_air_yards,0) AS air_yards_share
        FROM rec
        FULL OUTER JOIN rush
            ON rec.game_id = rush.game_id AND rec.player_id = rush.player_id
        LEFT JOIN team_totals tt
            ON tt.game_id = COALESCE(rec.game_id, rush.game_id)
           AND tt.offense = rec.team
    """)
    built["player_weeks"] = con.execute("SELECT COUNT(*) FROM player_weeks").fetchone()[0]

    # ---- optional sources ----
    for tag, view, pattern in [
        ("ftn_charting", "ftn", "ftn_charting_*.parquet"),
        ("snap_counts", "snaps", "snap_counts_*.parquet"),
        ("nextgen_stats", "ngs", "ngs_*.parquet"),
        ("stats_player", "stats_player", "stats_player_week_*.parquet"),
        ("injuries", "injuries_hist", "injuries_*.parquet"),
    ]:
        src = _glob(tag, pattern)
        if not src:
            missing.append(tag)
            continue
        con.execute(f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_parquet('{src}', union_by_name=true)")
        try:
            built[view] = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("%s unreadable: %s", view, exc)
            missing.append(tag)

    con.execute("CREATE INDEX IF NOT EXISTS idx_tw ON team_weeks(season, week, team)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pw ON player_weeks(season, week, player_id)")
    con.close()

    if missing and verbose:
        log.warning("missing/unavailable: %s", ", ".join(missing))
    return {"built": built, "missing": missing, "path": str(WAREHOUSE)}


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify() -> list[str]:
    """Sanity checks that catch the errors which otherwise surface as a
    mysteriously good backtest."""
    con = connect(read_only=True)
    problems: list[str] = []

    seasons = con.execute("SELECT MIN(season), MAX(season), COUNT(DISTINCT season) "
                          "FROM plays").fetchone()
    print(f"  seasons        : {seasons[0]}–{seasons[1]} ({seasons[2]} distinct)")

    per_season = con.execute(
        "SELECT season, COUNT(*) n FROM plays GROUP BY 1 ORDER BY 1").fetchall()
    thin = [f"{s}({n})" for s, n in per_season if n < 25_000]
    print(f"  plays/season   : {min(n for _, n in per_season):,}–"
          f"{max(n for _, n in per_season):,}")
    if thin:
        problems.append(f"suspiciously thin seasons: {', '.join(thin)}")

    # EPA should centre near zero league-wide — if it doesn't, the filter is wrong
    mean_epa = con.execute("SELECT AVG(epa) FROM plays").fetchone()[0]
    print(f"  mean EPA/play  : {mean_epa:+.4f}  (expect near 0)")
    if abs(mean_epa) > 0.05:
        problems.append(f"mean EPA {mean_epa:+.4f} is far from zero — check play filter")

    # success rate should sit around 45%
    sr = con.execute("SELECT AVG(success::DOUBLE) FROM plays").fetchone()[0]
    print(f"  success rate   : {sr:.3f}  (expect ~0.45)")
    if not 0.40 <= sr <= 0.50:
        problems.append(f"success rate {sr:.3f} outside expected band")

    # target shares must sum to ~1 per team-game
    ts = con.execute("""
        SELECT AVG(total) FROM (
            SELECT game_id, team, SUM(target_share) AS total
            FROM player_weeks WHERE target_share IS NOT NULL
            GROUP BY 1,2)
    """).fetchone()[0]
    print(f"  target share Σ : {ts:.3f}  (expect ~1.0)")
    if ts and not 0.9 <= ts <= 1.1:
        problems.append(f"target shares sum to {ts:.3f}, not ~1.0")

    dup = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT game_id, team, COUNT(*) c FROM team_weeks
            GROUP BY 1,2 HAVING c > 1)
    """).fetchone()[0]
    print(f"  dup team-games : {dup}  (expect 0)")
    if dup:
        problems.append(f"{dup} duplicated team-game rows")

    con.close()
    return problems


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="research warehouse (local only)")
    p.add_argument("command", choices=["build", "verify", "sql", "tables"])
    p.add_argument("query", nargs="?", default="")
    args = p.parse_args()

    if args.command == "build":
        r = build()
        print(f"\n  warehouse: {r['path']}")
        for k, v in r["built"].items():
            print(f"    {k:<22}{v:>12,}")
        if r["missing"]:
            print(f"\n  missing: {', '.join(r['missing'])}")
    elif args.command == "verify":
        probs = verify()
        print()
        if probs:
            for pr in probs:
                print(f"  PROBLEM: {pr}")
            sys.exit(1)
        print("  all checks passed")
    elif args.command == "tables":
        con = connect(read_only=True)
        for (n,) in con.execute("SHOW TABLES").fetchall():
            cnt = con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
            print(f"  {n:<24}{cnt:>12,}")
    else:
        con = connect(read_only=True)
        for row in con.execute(args.query).fetchall():
            print(" ", row)
