"""The Odds API v4 client with adaptive polling and a real credit ledger.

Credit model
------------
Cost of a request = (number of markets) x (number of regions). Credits reset on
the 1st of the month; exceeding the plan returns 429 rather than billing
overage. We do not estimate spend — every response carries
`x-requests-remaining` / `x-requests-used` headers, and we record those, so the
ledger reflects reality rather than our arithmetic.

Why adaptive polling
--------------------
NFL has ~16 games x ~150 markets x many books. Polling everything every 15
minutes would exhaust a month of credits in days and buy nothing: lines barely
move on Tuesday and move constantly on Sunday morning. Poll frequency therefore
scales with time-to-kickoff, and sharp books are polled harder than soft ones
because they move first.

Storage
-------
`odds_current` is UPSERTed (stays ~24k rows). `odds_changes` is appended ONLY
when a price actually changes, which is what makes line-movement features and
CLV possible without a multi-million-row table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.db import db, insert_row, insert_rows, query, upsert_rows, utcnow

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

FEATURED = ["h2h", "spreads", "totals"]
PERIOD = ["h2h_h1", "spreads_h1", "totals_h1", "h2h_h2", "spreads_h2", "totals_h2"]
TEAM_TOTALS = ["team_totals"]
PROPS_CORE = [
    "player_pass_yds", "player_pass_tds", "player_pass_completions",
    "player_pass_interceptions", "player_rush_yds", "player_rush_attempts",
    "player_reception_yds", "player_receptions", "player_anytime_td",
]

# `eu` is where Pinnacle lives — the sharp anchor the whole edge model leans on.
REGIONS_DEFAULT = ["us", "eu"]


class CreditExhausted(RuntimeError):
    pass


class OddsAPIError(RuntimeError):
    pass


@dataclass
class PollTier:
    name: str
    markets: list[str]
    regions: list[str]
    # minutes between polls, keyed by hours-to-kickoff bucket (upper bound)
    schedule: dict[float, int]
    # The bulk /odds endpoint only serves featured markets (h2h, spreads,
    # totals). Period markets and props are "additional markets" and 422 unless
    # requested per-event — which also makes them far more expensive, since
    # cost is markets x regions PER EVENT.
    per_event: bool = False

    def interval_minutes(self, hours_to_kick: float) -> int | None:
        for bound in sorted(self.schedule):
            if hours_to_kick <= bound:
                return self.schedule[bound]
        return None  # too far out — don't poll


TIERS = [
    # Buckets extend past 240h deliberately: opening lines are the softest
    # prices of the week, and line-movement history is only buildable if you
    # were watching before the market sharpened. A daily poll a month out is
    # cheap and is where CLV comes from.
    PollTier("featured", FEATURED, REGIONS_DEFAULT,
             {2: 5, 12: 10, 72: 30, 240: 60, 720: 360, 100_000: 720}),
    # per-event: 6 markets x 1 region x 16 games = 96 credits a sweep, so the
    # schedule is deliberately much sparser than featured.
    PollTier("period", PERIOD, ["us"],
             {2: 60, 12: 180, 72: 720}, per_event=True),
    PollTier("props", PROPS_CORE, ["us"],
             {2: 20, 12: 60, 72: 360}, per_event=True),   # props don't post far out
]


def _last_poll_age_minutes(tier_name: str) -> float:
    """Minutes since this tier last polled. Infinity if never."""
    row = query("SELECT last_success FROM source_freshness WHERE source=?",
                (f"odds_api:{tier_name}",))
    if not row or not row[0]["last_success"]:
        return float("inf")
    try:
        ts = datetime.fromisoformat(row[0]["last_success"])
    except ValueError:
        return float("inf")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 60.0


def _mark_polled(tier_name: str, detail: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, detail=excluded.detail",
            (f"odds_api:{tier_name}", utcnow(), None, detail),
        )


# --------------------------------------------------------------------------
# credit ledger
# --------------------------------------------------------------------------
class CreditLedger:
    """Tracks spend against the monthly budget and degrades gracefully.

    Degradation order when the budget tightens: drop props, then period markets,
    then extra regions. Featured markets on sharp books are the last thing to go
    because they are the highest-value credits we spend.
    """

    def __init__(self, budget: int | None = None):
        self.budget = budget or settings.ODDS_MONTHLY_CREDIT_BUDGET

    def used_this_month(self) -> int:
        first = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        row = query(
            "SELECT COALESCE(SUM(credits_used),0) AS n FROM odds_credit_ledger "
            "WHERE called_at >= ?", (first,)
        )
        return int(row[0]["n"]) if row else 0

    def remaining_pct(self) -> float:
        return max(0.0, 1.0 - self.used_this_month() / max(self.budget, 1)) * 100.0

    def allows(self, tier_name: str) -> bool:
        pct = self.remaining_pct()
        if pct > 30:
            return True
        if pct > 15:
            return tier_name != "props"
        if pct > 5:
            return tier_name == "featured"
        return False

    def record(self, endpoint: str, markets: list[str], cost: int,
               remaining: int | None, used: int | None) -> None:
        with db() as conn:
            insert_row(conn, "odds_credit_ledger", {
                "endpoint": endpoint,
                "markets": ",".join(markets),
                "credits_used": cost,
                "called_at": utcnow(),
            })
        if remaining is not None and remaining < 50:
            log.warning("ODDS API CREDITS LOW: %s remaining (used %s)", remaining, used)


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class OddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.ODDS_API_KEY
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not set")
        self.ledger = CreditLedger()
        self.last_remaining: int | None = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        retry=retry_if_exception_type(httpx.TransportError),
    )
    def _get(self, path: str, params: dict[str, Any], markets: list[str]) -> Any:
        params = {**params, "apiKey": self.api_key}
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
            r = c.get(f"{BASE}{path}", params=params)

        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        last = r.headers.get("x-requests-last")
        self.last_remaining = int(remaining) if remaining and remaining.isdigit() else None
        cost = int(last) if last and last.isdigit() else len(markets) * len(
            str(params.get("regions", "us")).split(",")
        )

        if r.status_code == 401:
            raise OddsAPIError("401 — bad API key")
        if r.status_code == 429:
            raise CreditExhausted(f"429 — monthly quota exhausted (used {used})")
        if r.status_code >= 400:
            raise OddsAPIError(f"{r.status_code}: {r.text[:200]}")

        self.ledger.record(path, markets, cost,
                           self.last_remaining,
                           int(used) if used and used.isdigit() else None)
        return r.json()

    # ---- endpoints ----
    def sports(self) -> list[dict]:
        return self._get("/sports", {}, [])

    def events(self) -> list[dict]:
        """Upcoming events. Cheap — use to decide what's worth polling."""
        return self._get(f"/sports/{SPORT}/events", {}, [])

    def odds(self, markets: list[str], regions: list[str],
             odds_format: str = "american") -> list[dict]:
        """Bulk odds across all events. Cost = markets x regions."""
        return self._get(
            f"/sports/{SPORT}/odds",
            {"regions": ",".join(regions), "markets": ",".join(markets),
             "oddsFormat": odds_format},
            markets,
        )

    def event_odds(self, event_id: str, markets: list[str], regions: list[str],
                   odds_format: str = "american") -> dict:
        """Per-event odds — required for player props."""
        return self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            {"regions": ",".join(regions), "markets": ",".join(markets),
             "oddsFormat": odds_format},
            markets,
        )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _side_for(market_key: str, outcome: dict, home: str, away: str) -> str:
    name = (outcome.get("name") or "").strip()
    low = name.lower()
    if low in {"over", "under", "yes", "no"}:
        return low
    if name == home:
        return "home"
    if name == away:
        return "away"
    return low or "unknown"


def parse_event(ev: dict, fetched_at: str) -> list[dict]:
    """Flatten one Odds API event into odds rows.

    Handles featured markets (outcomes keyed by team name), totals (Over/Under
    with a `point`), and player props (a `description` field carrying the player
    name, with Over/Under in `name`).
    """
    rows: list[dict] = []
    home = ev.get("home_team") or ""
    away = ev.get("away_team") or ""
    event_id = ev.get("id")
    for bm in ev.get("bookmakers", []) or []:
        book = bm.get("key")
        for mk in bm.get("markets", []) or []:
            mkey = mk.get("key")
            updated = mk.get("last_update") or bm.get("last_update")
            for oc in mk.get("outcomes", []) or []:
                price = oc.get("price")
                if price is None:
                    continue
                player_name = (oc.get("description") or "").strip()
                rows.append({
                    "odds_api_event_id": event_id,
                    "market_type": mkey,
                    "player_name": player_name,      # resolved to id downstream
                    "side": _side_for(mkey, oc, home, away),
                    "line": float(oc.get("point") or 0.0),
                    "book": book,
                    "price": int(price),
                    "updated_at": updated,
                    "fetched_at": fetched_at,
                })
    return rows


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def _event_to_game_id() -> dict[str, str]:
    return {
        r["odds_api_event_id"]: r["game_id"]
        for r in query("SELECT game_id, odds_api_event_id FROM games "
                       "WHERE odds_api_event_id IS NOT NULL")
    }


def _player_name_to_id() -> dict[str, str]:
    from src.fetchers.injuries import norm_name
    idx: dict[str, list[str]] = {}
    for r in query("SELECT player_id, full_name FROM players WHERE full_name IS NOT NULL"):
        idx.setdefault(norm_name(r["full_name"]), []).append(r["player_id"])
    for r in query("SELECT alias, player_id FROM player_aliases WHERE player_id IS NOT NULL"):
        idx.setdefault(norm_name(r["alias"]), []).append(r["player_id"])
    # only unambiguous names — never guess between two players
    return {k: v[0] for k, v in idx.items() if len(set(v)) == 1}


def persist(rows: Iterable[dict]) -> dict[str, int]:
    """Upsert into odds_current; append to odds_changes only on real changes."""
    from src.fetchers.injuries import norm_name

    rows = list(rows)
    if not rows:
        return {"seen": 0, "written": 0, "changed": 0, "unmapped_events": 0,
                "unmapped_players": 0}

    ev_map = _event_to_game_id()
    pl_map = _player_name_to_id()

    existing = {
        (r["game_id"], r["market_type"], r["player_id"], r["side"], r["line"], r["book"]): r["price"]
        for r in query("SELECT game_id, market_type, player_id, side, line, book, price "
                       "FROM odds_current")
    }

    out: list[dict] = []
    changes: list[dict] = []
    unmapped_ev = unmapped_pl = 0

    for r in rows:
        game_id = ev_map.get(r["odds_api_event_id"])
        if not game_id:
            unmapped_ev += 1
            continue
        pid = ""
        if r["player_name"]:
            pid = pl_map.get(norm_name(r["player_name"]), "")
            if not pid:
                unmapped_pl += 1
                continue      # quarantine — never attach a prop to a guessed player
        key = (game_id, r["market_type"], pid, r["side"], r["line"], r["book"])
        row = {
            "game_id": game_id, "market_type": r["market_type"], "player_id": pid,
            "side": r["side"], "line": r["line"], "book": r["book"],
            "price": r["price"], "updated_at": r["updated_at"],
            "fetched_at": r["fetched_at"],
        }
        out.append(row)
        if existing.get(key) != r["price"]:
            changes.append({
                "game_id": game_id, "market_type": r["market_type"], "player_id": pid,
                "side": r["side"], "line": r["line"], "book": r["book"],
                "price": r["price"], "observed_at": r["fetched_at"],
            })

    with db() as conn:
        upsert_rows(conn, "odds_current", out,
                    key_cols=["game_id", "market_type", "player_id", "side", "line", "book"])
        if changes:
            insert_rows(conn, "odds_changes", changes)

    return {"seen": len(rows), "written": len(out), "changed": len(changes),
            "unmapped_events": unmapped_ev, "unmapped_players": unmapped_pl}


# --------------------------------------------------------------------------
# event <-> game crosswalk
# --------------------------------------------------------------------------
def link_events(client: OddsClient | None = None, verbose: bool = False) -> int:
    """Attach Odds API event ids to our nflverse game_ids.

    Matches on (home_abbr, away_abbr) plus a +/-1 day kickoff window. Team names
    are resolved through src.teams, NOT substring matching — "kc" is not a
    substring of "kansas city chiefs", and the same is true for GB, SF, TB, NO,
    LAR and LAC. Substring matching silently drops ~half the slate.

    The date window exists because games.csv stores ET while the Odds API uses
    UTC, so a Sunday night kickoff lands on the following UTC day.
    """
    from src.teams import resolve

    client = client or OddsClient()
    events = client.events()
    if not events:
        return 0

    games = query(
        "SELECT game_id, home_team, away_team, kickoff_utc FROM games "
        "WHERE season=? AND status!='final'", (settings.CURRENT_SEASON,)
    )
    # (home, away, date) -> game_id, with the game indexed on every plausible day
    index: dict[tuple[str, str, str], str] = {}
    for g in games:
        h, a = resolve(g["home_team"]), resolve(g["away_team"])
        if not h or not a or not g["kickoff_utc"]:
            continue
        try:
            base = datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00")).date()
        except ValueError:
            continue
        for shift in (-1, 0, 1):
            index[(h, a, (base + timedelta(days=shift)).isoformat())] = g["game_id"]

    linked = 0
    unmatched: list[str] = []
    with db() as conn:
        for ev in events:
            h, a = resolve(ev.get("home_team")), resolve(ev.get("away_team"))
            ct = ev.get("commence_time", "")
            if not h or not a or not ct:
                unmatched.append(f"{ev.get('away_team')} @ {ev.get('home_team')} (unresolved team)")
                continue
            day = ct[:10]
            gid = index.get((h, a, day))
            if not gid:
                unmatched.append(f"{a} @ {h} {day}")
                continue
            conn.execute("UPDATE games SET odds_api_event_id=? WHERE game_id=?",
                         (ev["id"], gid))
            linked += 1

    log.info("linked %d/%d odds events to games", linked, len(events))
    if unmatched:
        log.warning("%d unmatched events — odds for these are DISCARDED", len(unmatched))
        if verbose:
            for u in unmatched[:40]:
                log.warning("  unmatched: %s", u)
    return linked


# --------------------------------------------------------------------------
# scheduler entry point
# --------------------------------------------------------------------------
def _hours_to_kick(kickoff: str | None) -> float:
    if not kickoff:
        return 9999.0
    try:
        ts = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return 9999.0
    return (ts - datetime.now(timezone.utc)).total_seconds() / 3600.0


def poll(force_tier: str | None = None) -> dict[str, int]:
    """One adaptive polling pass. Called on a short interval; self-throttles."""
    client = OddsClient()
    fetched_at = utcnow()
    totals = {"seen": 0, "written": 0, "changed": 0,
              "unmapped_events": 0, "unmapped_players": 0, "calls": 0}

    upcoming = query(
        "SELECT game_id, kickoff_utc FROM games WHERE season=? AND status!='final' "
        "AND kickoff_utc IS NOT NULL", (settings.CURRENT_SEASON,)
    )
    if not upcoming:
        log.info("no upcoming games — nothing to poll")
        return totals
    soonest = min(_hours_to_kick(g["kickoff_utc"]) for g in upcoming)

    for tier in TIERS:
        if force_tier and tier.name != force_tier:
            continue
        if tier.name == "props" and not settings.ENABLE_PROPS:
            continue
        if not client.ledger.allows(tier.name):
            log.warning("credit budget: skipping tier %s (%.0f%% remaining)",
                        tier.name, client.ledger.remaining_pct())
            continue

        interval = tier.interval_minutes(soonest)
        if interval is None:
            continue
        # THE throttle. Without this the tier fires on every scheduler tick
        # regardless of its schedule, which burns ~50k credits/month in season.
        age = _last_poll_age_minutes(tier.name)
        if age < interval and not force_tier:
            log.debug("tier %s: %.1fm since last poll, interval %dm — skipping",
                      tier.name, age, interval)
            continue

        try:
            if tier.per_event:
                # Period markets and props 422 on the bulk endpoint — they must
                # be requested one event at a time. Restricted to games inside
                # 72h because cost scales with event count.
                for g in upcoming:
                    if _hours_to_kick(g["kickoff_utc"]) > 72:
                        continue
                    row = query("SELECT odds_api_event_id FROM games WHERE game_id=?",
                                (g["game_id"],))
                    eid = row[0]["odds_api_event_id"] if row else None
                    if not eid:
                        continue
                    try:
                        ev = client.event_odds(eid, tier.markets, tier.regions)
                    except OddsAPIError as exc:
                        log.warning("event %s markets unavailable: %s", eid, exc)
                        continue
                    totals["calls"] += 1
                    r = persist(parse_event(ev, fetched_at))
                    for k in ("seen", "written", "changed", "unmapped_events",
                              "unmapped_players"):
                        totals[k] += r[k]
            else:
                data = client.odds(tier.markets, tier.regions)
                totals["calls"] += 1
                rows: list[dict] = []
                for ev in data:
                    rows += parse_event(ev, fetched_at)
                r = persist(rows)
                for k in ("seen", "written", "changed", "unmapped_events",
                          "unmapped_players"):
                    totals[k] += r[k]
            _mark_polled(tier.name, f"interval={interval}m soonest={soonest:.1f}h")
        except CreditExhausted as exc:
            log.error("credits exhausted: %s", exc)
            break
        except OddsAPIError as exc:
            log.error("tier %s failed: %s", tier.name, exc)

    with db() as conn:
        conn.execute(
            "INSERT INTO source_freshness (source, last_success, remote_stamp, detail) "
            "VALUES (?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
            "last_success=excluded.last_success, detail=excluded.detail",
            ("odds_api", utcnow(), str(client.last_remaining),
             f"{totals['written']} rows, {totals['changed']} changed, "
             f"{totals['calls']} calls, {client.last_remaining} credits left"),
        )
    log.info("poll: %s | credits remaining=%s", totals, client.last_remaining)
    return totals


def probe() -> None:
    """Cheap connectivity + shape check. Costs a handful of credits."""
    client = OddsClient()
    print("=== sports ===")
    sports = client.sports()
    nfl = [s for s in sports if s.get("key") == SPORT]
    print(f"  {len(sports)} sports; NFL present: {bool(nfl)}")
    if nfl:
        print(f"  {nfl[0].get('title')} active={nfl[0].get('active')}")

    print("\n=== events ===")
    events = client.events()
    print(f"  {len(events)} upcoming NFL events")
    for e in events[:3]:
        print(f"    {e.get('commence_time')}  {e.get('away_team')} @ {e.get('home_team')}")

    print("\n=== featured odds (h2h,spreads,totals x us,eu) ===")
    data = client.odds(FEATURED, REGIONS_DEFAULT)
    print(f"  {len(data)} events returned")
    if data:
        ev = data[0]
        books = [b.get("key") for b in ev.get("bookmakers", [])]
        print(f"  sample: {ev.get('away_team')} @ {ev.get('home_team')}")
        print(f"  {len(books)} books: {', '.join(books[:12])}")
        print(f"  pinnacle present: {'pinnacle' in books}")
        rows = parse_event(ev, utcnow())
        print(f"  parsed {len(rows)} odds rows")
        for r in rows[:5]:
            print(f"    {r['market_type']:<10} {r['side']:<6} line={r['line']:<7} "
                  f"{r['book']:<14} {r['price']:+d}")

    print(f"\n  credits remaining: {client.last_remaining}")
    print(f"  budget used this month: {client.ledger.used_this_month()}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="The Odds API client")
    p.add_argument("command", choices=["probe", "poll", "link", "events"])
    p.add_argument("--tier", default=None)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "probe":
        probe()
    elif args.command == "link":
        print(f"{link_events()} events linked")
    elif args.command == "events":
        for e in OddsClient().events():
            print(f"{e['commence_time']}  {e.get('away_team')} @ {e.get('home_team')}  [{e['id']}]")
    else:
        print(poll(force_tier=args.tier))
