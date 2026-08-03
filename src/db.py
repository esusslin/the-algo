"""SQLite access layer: connections, migrations, and safe write helpers.

Design notes
------------
* Raw sqlite3, no ORM — same as the baseball app.
* WAL mode so reads don't block the scheduler's writes.
* Migrations are an ordered list; each runs exactly once and is recorded.
* `insert_row` / `upsert_row` build SQL from a dict so it is structurally
  impossible to mismatch column count against `?` placeholders.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from src.config import settings

log = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 15_000


def utcnow() -> str:
    """Canonical timestamp format used everywhere in this codebase."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# connections
# --------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """Transactional connection. Commits on success, rolls back on exception."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    with db() as conn:
        return conn.execute(sql, params).fetchone()


# --------------------------------------------------------------------------
# safe writes — never hand-write placeholder lists again
# --------------------------------------------------------------------------
def insert_row(conn: sqlite3.Connection, table: str, row: dict[str, Any],
               on_conflict: str = "") -> int:
    cols = list(row.keys())
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))}) {on_conflict}"
    )
    cur = conn.execute(sql, [row[c] for c in cols])
    return cur.lastrowid or 0


def insert_rows(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]],
                on_conflict: str = "") -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))}) {on_conflict}"
    )
    conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return len(rows)


def upsert_row(conn: sqlite3.Connection, table: str, row: dict[str, Any],
               key_cols: Sequence[str]) -> None:
    """INSERT ... ON CONFLICT(keys) DO UPDATE for every non-key column."""
    updates = [c for c in row if c not in key_cols]
    clause = (
        f"ON CONFLICT({', '.join(key_cols)}) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in updates)
    ) if updates else f"ON CONFLICT({', '.join(key_cols)}) DO NOTHING"
    insert_row(conn, table, row, on_conflict=clause)


def upsert_rows(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]],
                key_cols: Sequence[str]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    updates = [c for c in rows[0] if c not in key_cols]
    clause = (
        f"ON CONFLICT({', '.join(key_cols)}) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in updates)
    ) if updates else f"ON CONFLICT({', '.join(key_cols)}) DO NOTHING"
    return insert_rows(conn, table, rows, on_conflict=clause)


# --------------------------------------------------------------------------
# job run tracking — powers /health
# --------------------------------------------------------------------------
@contextmanager
def job_run(job_name: str) -> Iterator[dict[str, Any]]:
    """Wrap a scheduled job so success/failure is recorded for /health.

        with job_run("refresh_nflverse") as ctx:
            ctx["rows_affected"] = do_the_work()
    """
    started = utcnow()
    t0 = time.monotonic()
    ctx: dict[str, Any] = {"rows_affected": 0}
    status, error = "success", None
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 — we want every failure recorded
        status, error = "error", f"{type(exc).__name__}: {exc}"
        log.exception("job %s failed", job_name)
        raise
    finally:
        with db() as conn:
            insert_row(conn, "job_runs", {
                "job_name": job_name,
                "started_at": started,
                "finished_at": utcnow(),
                "duration_s": round(time.monotonic() - t0, 3),
                "status": status,
                "rows_affected": int(ctx.get("rows_affected") or 0),
                "error": error,
            })


def last_run(job_name: str) -> sqlite3.Row | None:
    return query_one(
        "SELECT * FROM job_runs WHERE job_name=? ORDER BY id DESC LIMIT 1", (job_name,)
    )


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------
MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    -- ============ ops ============
    CREATE TABLE IF NOT EXISTS job_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name      TEXT NOT NULL,
        started_at    TEXT NOT NULL,
        finished_at   TEXT,
        duration_s    REAL,
        status        TEXT NOT NULL,
        rows_affected INTEGER DEFAULT 0,
        error         TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_jobruns_name ON job_runs(job_name, id DESC);

    CREATE TABLE IF NOT EXISTS source_freshness (
        source        TEXT PRIMARY KEY,
        last_success  TEXT,
        remote_stamp  TEXT,
        detail        TEXT
    );

    -- ============ reference ============
    CREATE TABLE IF NOT EXISTS games (
        game_id          TEXT PRIMARY KEY,
        season           INTEGER,
        week             INTEGER,
        season_type      TEXT,
        kickoff_utc      TEXT,
        home_team        TEXT,
        away_team        TEXT,
        stadium          TEXT,
        roof             TEXT,
        surface          TEXT,
        lat              REAL,
        lon              REAL,
        elevation_m      REAL,
        referee          TEXT,
        home_score       INTEGER,
        away_score       INTEGER,
        spread_line      REAL,
        total_line       REAL,
        home_moneyline   INTEGER,
        away_moneyline   INTEGER,
        home_rest        INTEGER,
        away_rest        INTEGER,
        div_game         INTEGER,
        odds_api_event_id TEXT,
        status           TEXT DEFAULT 'scheduled',
        updated_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_games_week ON games(season, week);
    CREATE INDEX IF NOT EXISTS idx_games_kick ON games(kickoff_utc);

    CREATE TABLE IF NOT EXISTS teams (
        team_abbr   TEXT PRIMARY KEY,
        team_name   TEXT,
        conference  TEXT,
        division    TEXT,
        stadium     TEXT,
        roof        TEXT,
        surface     TEXT,
        lat         REAL,
        lon         REAL,
        elevation_m REAL,
        tz          TEXT
    );

    CREATE TABLE IF NOT EXISTS players (
        player_id     TEXT PRIMARY KEY,
        full_name     TEXT,
        position      TEXT,
        team          TEXT,
        status        TEXT,
        pfr_id        TEXT,
        espn_id       TEXT,
        sleeper_id    TEXT,
        updated_at    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_players_name ON players(full_name);
    CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);

    -- Name matching between odds feeds and nflverse IDs is the #1 bug source.
    -- Never fuzzy-match silently; record the alias and its confidence.
    CREATE TABLE IF NOT EXISTS player_aliases (
        alias       TEXT PRIMARY KEY,
        player_id   TEXT,
        source      TEXT,
        confidence  REAL,
        verified    INTEGER DEFAULT 0,
        created_at  TEXT
    );

    -- ============ context ============
    CREATE TABLE IF NOT EXISTS injuries (
        player_id           TEXT,
        player_name         TEXT,
        team                TEXT,
        season              INTEGER,
        week                INTEGER,
        report_date         TEXT,
        knowledge_time      TEXT NOT NULL,
        game_status         TEXT,
        practice_status     TEXT,
        designation         TEXT,
        body_part           TEXT,
        source              TEXT,
        source_url          TEXT,
        ai_expected_role    TEXT,
        ai_snap_expectation REAL,
        ai_confidence       REAL,
        PRIMARY KEY (player_id, season, week, knowledge_time, source)
    );
    CREATE INDEX IF NOT EXISTS idx_inj_lookup ON injuries(season, week, team);
    CREATE INDEX IF NOT EXISTS idx_inj_kt ON injuries(knowledge_time);

    CREATE TABLE IF NOT EXISTS inactives (
        game_id     TEXT,
        player_id   TEXT,
        player_name TEXT,
        team        TEXT,
        declared_at TEXT,
        source      TEXT,
        PRIMARY KEY (game_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS weather_snapshots (
        game_id        TEXT,
        knowledge_time TEXT,
        temp_c         REAL,
        wind_kph       REAL,
        wind_gust_kph  REAL,
        wind_dir       INTEGER,
        precip_mm      REAL,
        snow_mm        REAL,
        humidity       REAL,
        pressure       REAL,
        is_forecast    INTEGER,
        PRIMARY KEY (game_id, knowledge_time)
    );

    -- ============ users ============
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        phone         TEXT,
        password_hash TEXT NOT NULL,
        role          TEXT DEFAULT 'user',
        tier          TEXT DEFAULT 'free',
        sms_opt_in    INTEGER DEFAULT 0,
        quiet_start   INTEGER DEFAULT 22,
        quiet_end     INTEGER DEFAULT 8,
        created_at    TEXT,
        last_login    TEXT
    );
    """),

    (2, """
    -- ============ market ============
    CREATE TABLE IF NOT EXISTS odds_current (
        game_id     TEXT NOT NULL,
        market_type TEXT NOT NULL,
        player_id   TEXT NOT NULL DEFAULT '',
        side        TEXT NOT NULL,
        line        REAL NOT NULL DEFAULT 0,
        book        TEXT NOT NULL,
        price       INTEGER NOT NULL,
        updated_at  TEXT,
        fetched_at  TEXT,
        PRIMARY KEY (game_id, market_type, player_id, side, line, book)
    );
    CREATE INDEX IF NOT EXISTS idx_oc_game ON odds_current(game_id, market_type);

    CREATE TABLE IF NOT EXISTS odds_changes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id     TEXT NOT NULL,
        market_type TEXT NOT NULL,
        player_id   TEXT DEFAULT '',
        side        TEXT,
        line        REAL,
        book        TEXT,
        price       INTEGER,
        observed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ochg ON odds_changes(game_id, market_type, observed_at);

    CREATE TABLE IF NOT EXISTS fair_prices (
        game_id     TEXT NOT NULL,
        market_type TEXT NOT NULL,
        player_id   TEXT NOT NULL DEFAULT '',
        side        TEXT NOT NULL,
        line        REAL NOT NULL DEFAULT 0,
        fair_prob   REAL,
        method      TEXT,
        sharp_prob  REAL,
        book_count  INTEGER,
        dispersion  REAL,
        computed_at TEXT,
        PRIMARY KEY (game_id, market_type, player_id, side, line)
    );

    CREATE TABLE IF NOT EXISTS odds_credit_ledger (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint     TEXT,
        markets      TEXT,
        credits_used INTEGER,
        called_at    TEXT
    );
    """),

    (3, """
    -- ============ model + picks ============
    CREATE TABLE IF NOT EXISTS artifact_registry (
        version                TEXT PRIMARY KEY,
        trained_through_season INTEGER,
        trained_through_week   INTEGER,
        feature_spec_hash      TEXT,
        metrics_json           TEXT,
        loaded_at              TEXT,
        active                 INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS projections (
        game_id          TEXT NOT NULL,
        market_type      TEXT NOT NULL,
        player_id        TEXT NOT NULL DEFAULT '',
        side             TEXT NOT NULL,
        line             REAL NOT NULL DEFAULT 0,
        model_prob       REAL,
        blended_prob     REAL,
        mean_est         REAL,
        sd_est           REAL,
        artifact_version TEXT NOT NULL DEFAULT '',
        shap_top         TEXT,
        computed_at      TEXT,
        PRIMARY KEY (game_id, market_type, player_id, side, line, artifact_version)
    );

    CREATE TABLE IF NOT EXISTS picks (
        pick_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id           TEXT NOT NULL,
        market_type       TEXT NOT NULL,
        player_id         TEXT DEFAULT '',
        side              TEXT,
        line              REAL,
        source            TEXT,
        best_book         TEXT,
        best_price        INTEGER,
        fair_prob         REAL,
        blended_prob      REAL,
        edge_pct          REAL,
        kelly_units       REAL,
        tier              TEXT,
        ai_verdict        TEXT,
        ai_reason         TEXT,
        headline          TEXT,
        detail            TEXT,
        published         INTEGER DEFAULT 0,
        visibility        TEXT DEFAULT 'admin',
        created_at        TEXT,
        published_at      TEXT,
        closing_price     INTEGER,
        closing_fair_prob REAL,
        clv_pct           REAL,
        result            TEXT DEFAULT 'pending',
        graded_at         TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_picks_game ON picks(game_id, published, tier);
    CREATE INDEX IF NOT EXISTS idx_picks_result ON picks(result);

    CREATE TABLE IF NOT EXISTS user_bets (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL,
        pick_id   INTEGER,
        book      TEXT,
        price     INTEGER,
        stake     REAL,
        placed_at TEXT,
        result    TEXT DEFAULT 'pending'
    );
    CREATE INDEX IF NOT EXISTS idx_bets_user ON user_bets(user_id);

    CREATE TABLE IF NOT EXISTS ai_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        agent         TEXT,
        model         TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd      REAL,
        ref_type      TEXT,
        ref_id        TEXT,
        created_at    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ai_created ON ai_calls(created_at);
    """),
]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


def run_migrations() -> int:
    """Apply any unapplied migrations. Returns the resulting schema version."""
    settings.ensure_dirs()
    conn = connect()
    try:
        version = current_version(conn)
        for target, sql in MIGRATIONS:
            if target <= version:
                continue
            log.info("applying migration %s", target)
            conn.executescript(sql)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (target, utcnow()),
            )
            conn.commit()
            version = target
        return version
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"schema version: {run_migrations()}")
    print(f"database: {settings.DATABASE_PATH}")
