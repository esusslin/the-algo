"""One-off: rewrite `games.kickoff_utc` from Eastern to real UTC.

    python scripts/backfill_kickoff_utc.py --dry
    python scripts/backfill_kickoff_utc.py

**Why.** `games.csv` publishes kickoff in Eastern time and the ingest wrote it straight
into a column named `kickoff_utc` without converting. Every consumer that treated it as
UTC was four or five hours early. The worst of those is `capture_closing_lines`, which
fires on `datetime(kickoff_utc) <= datetime('now', '+12 minutes')` — so for an 8:20pm ET
kickoff it captured at roughly 4:08pm ET, and CLV was measured against an afternoon price
rather than a closing one. Odds vanish at kickoff, so that error cannot be corrected
later; it has to be right before the games start.

`et_to_utc` fixes new rows at ingest, but `refresh_nflverse` only runs Tuesday and
Thursday — this brings existing rows forward now.

**Idempotent, and that is the hard part.** Running this twice must not shift times twice.
The guard is the string itself: converted values carry an offset suffix (`+00:00`), the
old ET values do not. Anything already offset-aware is skipped, so a second run is a
no-op and a partial run resumes safely.

Bare dates (no time component) are left alone — there is no time to convert, and
inventing midnight in some timezone would be worse than leaving it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import db, query  # noqa: E402
from src.fetchers.nflverse import et_to_utc  # noqa: E402

log = logging.getLogger("backfill_kickoff")


def needs_conversion(value: str | None) -> bool:
    """True for a naive `YYYY-MM-DDTHH:MM:SS` written by the old code path.

    Deliberately conservative: anything already carrying an offset, anything that is a
    bare date, and anything unparseable is left untouched. A backfill that skips a row
    is recoverable; one that double-shifts a row is not.
    """
    if not value or "T" not in value:
        return False
    return not (value.endswith("Z") or "+" in value[10:] or value[10:].count("-") > 0)


def convert(value: str) -> str | None:
    gameday, _, rest = value.partition("T")
    hhmm = rest[:5]
    return et_to_utc(gameday, hhmm)


def backfill_kickoffs() -> int:
    """Convert any remaining Eastern kickoffs to UTC. Returns rows changed.

    Called from application startup as well as the CLI, because the-algo runs on SQLite
    and the production database is a file inside the Railway container — there is no
    host to connect to and run a script against. Doing it at boot is the only path that
    doesn't involve shelling into a container during a game.

    Idempotent by design, so booting a hundred times converts once.
    """
    rows = query("SELECT game_id, kickoff_utc FROM games WHERE kickoff_utc IS NOT NULL")
    todo = [(r["game_id"], r["kickoff_utc"]) for r in rows
            if needs_conversion(r["kickoff_utc"])]
    if not todo:
        return 0

    changed = 0
    with db() as conn:
        for game_id, old in todo:
            new = convert(old)
            if not new or new == old:
                continue
            conn.execute("UPDATE games SET kickoff_utc=? WHERE game_id=?", (new, game_id))
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = query("SELECT game_id, kickoff_utc FROM games WHERE kickoff_utc IS NOT NULL")
    todo = [(r["game_id"], r["kickoff_utc"]) for r in rows
            if needs_conversion(r["kickoff_utc"])]

    print(f"  {len(rows)} games with a kickoff")
    print(f"  {len(todo)} need conversion, {len(rows) - len(todo)} already UTC or date-only")
    if not todo:
        print("\n  nothing to do")
        return 0

    for game_id, old in todo[:5]:
        print(f"    e.g. {game_id}  {old}  ->  {convert(old)}")

    if args.dry:
        print("\n  dry run — nothing written")
        return 0

    changed = backfill_kickoffs()
    print(f"\n  converted {changed} rows")

    # Prove it took, and prove a second run is a no-op.
    again = [r for r in query("SELECT kickoff_utc FROM games WHERE kickoff_utc IS NOT NULL")
             if needs_conversion(r["kickoff_utc"])]
    print(f"  rows still needing conversion after the run: {len(again)}  (expected 0)")
    return 0 if not again else 1


if __name__ == "__main__":
    sys.exit(main())
