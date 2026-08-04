"""Red-team agent — the only AI in the pick path, and deliberately constrained.

Its entire job is to find reasons NOT to make a bet that the feature matrix
cannot see: a team resting starters in a meaningless week, a backup quarterback,
weather the totals model hasn't priced, a line that moved hard against us.

Three properties make this safe to run on live picks:

**Downgrade-only.** It can return OK, FLAG or KILL. It cannot promote a pick,
raise a tier, or create one. Worst case it costs you a good bet — it can never
inject uncalibrated judgment into bet sizing. That asymmetry is the whole design.

**Evidence required.** A KILL without a concrete, checkable reason drawn from the
supplied context is automatically demoted to FLAG. Models are fluent and will
produce plausible-sounding objections on request; requiring evidence forces the
objection to be grounded in something real.

**Fails open.** Any error, timeout or budget exhaustion returns OK. A broken AI
layer must never silently suppress your entire slate.

Context is assembled per GAME and reused across every pick in that game, because
the expensive part is the situational picture, not the individual bet.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.ai.client import AIUnavailable, complete_json
from src.config import settings
from src.db import db, query, utcnow
from src.markets import describe_market

log = logging.getLogger(__name__)

VERDICTS = {"OK", "FLAG", "KILL"}

SYSTEM = """You review proposed sports bets and look for reasons NOT to place them.

You are not predicting outcomes. You are not evaluating whether the edge \
calculation is correct. You are checking for situational factors the statistical \
model cannot see.

Return ONLY a JSON object:
{"verdict": "OK" | "FLAG" | "KILL", "reason": "one sentence", "evidence": "the \
specific fact from the context that supports this, or empty string"}

Rules:
- OK is the default. Most bets have no disqualifying issue.
- FLAG means a real concern that should reduce confidence but not cancel.
- KILL means a concrete disqualifying fact, and you MUST cite it in evidence.
- Never invent facts. If the context does not contain a reason, return OK.
- Do not comment on whether the edge or price looks good. That is not your job."""


def _game_context(game_id: str) -> dict[str, Any]:
    """Everything situational we know about a game. Built once per game."""
    g = query("SELECT * FROM games WHERE game_id=?", (game_id,))
    if not g:
        return {}
    g = dict(g[0])

    injuries = [dict(r) for r in query(
        "SELECT player_name, team, game_status, practice_status, body_part "
        "FROM injuries WHERE season=? AND week=? AND team IN (?,?) "
        "AND game_status IS NOT NULL "
        "ORDER BY CASE game_status WHEN 'Out' THEN 1 WHEN 'Doubtful' THEN 2 "
        "WHEN 'Questionable' THEN 3 ELSE 4 END LIMIT 25",
        (g.get("season"), g.get("week"), g.get("home_team"), g.get("away_team")))]

    weather = query(
        "SELECT temp_c, wind_kph, wind_gust_kph, precip_mm, snow_mm "
        "FROM weather_snapshots WHERE game_id=? ORDER BY knowledge_time DESC LIMIT 1",
        (game_id,))

    inactive = [dict(r) for r in query(
        "SELECT player_name, team FROM inactives WHERE game_id=?", (game_id,))]

    # line movement — a sharp move against a pick is a genuine warning
    moves = [dict(r) for r in query(
        "SELECT market_type, side, line, price, book, observed_at "
        "FROM odds_changes WHERE game_id=? AND book IN "
        "(SELECT DISTINCT book FROM odds_current WHERE game_id=?) "
        "ORDER BY observed_at DESC LIMIT 12", (game_id, game_id))]

    return {
        "matchup": f"{g.get('away_team')} at {g.get('home_team')}",
        "season": g.get("season"), "week": g.get("week"),
        "season_type": g.get("season_type"),
        "kickoff_utc": g.get("kickoff_utc"),
        "roof": g.get("roof"), "surface": g.get("surface"),
        "market_spread": g.get("spread_line"), "market_total": g.get("total_line"),
        "home_rest_days": g.get("home_rest"), "away_rest_days": g.get("away_rest"),
        "divisional": bool(g.get("div_game")),
        "injuries": injuries,
        "inactives": inactive,
        "weather": dict(weather[0]) if weather else None,
        "recent_line_moves": moves[:6],
    }


def _prompt(pick: dict, ctx: dict) -> str:
    info = describe_market(pick["market_type"])
    late_season = (ctx.get("week") or 0) >= 17 and ctx.get("season_type") == "REG"
    return f"""GAME
{ctx.get('matchup')} — {ctx.get('season')} week {ctx.get('week')} \
({ctx.get('season_type')})
Kickoff: {ctx.get('kickoff_utc')}
Roof: {ctx.get('roof')} | Surface: {ctx.get('surface')} | \
Divisional: {ctx.get('divisional')}
Rest: home {ctx.get('home_rest_days')}d, away {ctx.get('away_rest_days')}d
Market: spread {ctx.get('market_spread')}, total {ctx.get('market_total')}
{"NOTE: late regular season — teams with nothing to play for may rest starters."
 if late_season else ""}

WEATHER
{ctx.get('weather') or 'no forecast available'}

INJURY REPORT
{chr(10).join(f"- {i['team']} {i['player_name']}: {i['game_status']}"
              f" (practice {i['practice_status'] or 'n/a'}, {i['body_part'] or 'n/a'})"
              for i in ctx.get('injuries', [])) or '- none recorded'}

INACTIVES
{chr(10).join(f"- {i['team']} {i['player_name']}" for i in ctx.get('inactives', []))
 or '- not yet declared'}

RECENT LINE MOVEMENT
{chr(10).join(f"- {m['market_type']} {m['side']} {m['line']} -> {m['price']:+d} "
              f"({m['book']})" for m in ctx.get('recent_line_moves', []))
 or '- none recorded'}

PROPOSED BET
{pick.get('headline')}
Market type: {info.label} ({info.bet_class})
Side: {pick.get('side')} | Line: {pick.get('line')}
Tier: {pick.get('tier')}

Is there a disqualifying reason not to place this bet?"""


def review_pick(pick: dict, ctx: dict | None = None) -> dict:
    """Review one pick. Always returns a verdict; never raises."""
    if not settings.ENABLE_AI_REDTEAM:
        return {"verdict": "OK", "reason": "", "evidence": "", "source": "disabled"}

    ctx = ctx if ctx is not None else _game_context(pick["game_id"])
    if not ctx:
        return {"verdict": "OK", "reason": "", "evidence": "", "source": "no_context"}

    try:
        out = complete_json(
            _prompt(pick, ctx), agent="redteam", system=SYSTEM,
            model=settings.MODEL_REASON, max_tokens=300,
            ref_type="pick", ref_id=str(pick.get("pick_id", "")))
    except AIUnavailable as exc:
        # FAIL OPEN. A broken AI layer must not suppress the slate — but the
        # reason must reach the operator, not just a log line nobody reads.
        log.warning("redteam unavailable: %s", exc)
        return {"verdict": "OK", "reason": "", "evidence": "",
                "source": "unavailable", "error": str(exc)[:200]}

    verdict = str(out.get("verdict", "OK")).upper().strip()
    reason = str(out.get("reason", ""))[:300]
    evidence = str(out.get("evidence", ""))[:300]

    if verdict not in VERDICTS:
        verdict = "OK"

    # An unevidenced KILL is an opinion, not a finding. Demote it.
    if verdict == "KILL" and len(evidence.strip()) < 12:
        log.info("demoting unevidenced KILL to FLAG: %s", reason[:80])
        verdict, reason = "FLAG", f"(unevidenced) {reason}"

    return {"verdict": verdict, "reason": reason, "evidence": evidence,
            "source": "model"}


def review_slate(picks: list[dict]) -> dict:
    """Review a whole slate, building game context once per game."""
    contexts: dict[str, dict] = {}
    results: dict[int, dict] = {}
    counts = {"OK": 0, "FLAG": 0, "KILL": 0}
    # "all OK" is ambiguous: it looks the same whether the model reviewed every
    # pick and found nothing, or never ran at all (the agent fails open by
    # design). Track the source so the caller can tell the difference.
    sources: dict[str, int] = {}

    for p in picks:
        gid = p["game_id"]
        if gid not in contexts:
            contexts[gid] = _game_context(gid)
        r = review_pick(p, contexts[gid])
        results[p["pick_id"]] = r
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1

    reviewed_by_model = sources.get("model", 0)
    # Collapse identical failures. Every call carries a unique request_id, so
    # naive dedup leaves you with N copies of the same message.
    seen: dict[str, int] = {}
    for r in results.values():
        if not r.get("error"):
            continue
        key = re.sub(r"'request_id': '[^']*'", "'request_id': '...'", r["error"])
        seen[key] = seen.get(key, 0) + 1
    errors = [f"{k}" + (f"  (x{n})" if n > 1 else "")
              for k, n in sorted(seen.items(), key=lambda kv: -kv[1])][:3]
    return {"reviewed": len(picks), "games": len(contexts),
            "counts": counts, "results": results,
            "sources": sources,
            "reviewed_by_model": reviewed_by_model,
            "ai_ran": reviewed_by_model > 0,
            "errors": errors}


def apply_to_picks(limit: int = 60) -> dict:
    """Review pending published picks and write verdicts back.

    Downgrade only: a FLAG caps a pick at tier B, a KILL unpublishes it. Nothing
    here can raise a tier or publish something that was not already published.
    """
    picks = [dict(r) for r in query(
        "SELECT pick_id, game_id, market_type, side, line, tier, headline "
        "FROM picks WHERE result='pending' AND ai_verdict IN ('OK','') "
        "ORDER BY edge_pct DESC LIMIT ?", (limit,))]
    if not picks:
        return {"reviewed": 0, "counts": {}, "changed": 0}

    out = review_slate(picks)
    changed = 0
    with db() as conn:
        for p in picks:
            r = out["results"][p["pick_id"]]
            new_tier, published = p["tier"], None
            if r["verdict"] == "KILL":
                published = 0
            elif r["verdict"] == "FLAG" and p["tier"] == "A":
                new_tier = "B"        # cap, never raise

            conn.execute(
                "UPDATE picks SET ai_verdict=?, ai_reason=?, tier=?"
                + (", published=?" if published is not None else "")
                + " WHERE pick_id=?",
                ([r["verdict"], (r["reason"] + (" | " + r["evidence"]
                                                if r["evidence"] else ""))[:500],
                  new_tier]
                 + ([published] if published is not None else [])
                 + [p["pick_id"]]))
            if r["verdict"] != "OK":
                changed += 1

    log.info("redteam: %s across %d games, %d downgraded (ai_ran=%s, sources=%s)",
             out["counts"], out["games"], changed, out["ai_ran"], out["sources"])
    return {"reviewed": out["reviewed"], "games": out["games"],
            "counts": out["counts"], "changed": changed,
            "ai_ran": out["ai_ran"], "reviewed_by_model": out["reviewed_by_model"],
            "sources": out["sources"], "errors": out.get("errors", [])}


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="red-team agent")
    p.add_argument("command", choices=["run", "dry", "context", "prompt"])
    p.add_argument("--game", default="")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "context":
        print(_json.dumps(_game_context(args.game), indent=2, default=str)[:2500])
    elif args.command == "prompt":
        pk = query("SELECT * FROM picks WHERE game_id=? LIMIT 1", (args.game,))
        if not pk:
            raise SystemExit("no picks for that game")
        print(_prompt(dict(pk[0]), _game_context(args.game)))
    elif args.command == "dry":
        picks = [dict(r) for r in query(
            "SELECT pick_id, game_id, market_type, side, line, tier, headline "
            "FROM picks WHERE result='pending' ORDER BY edge_pct DESC LIMIT ?",
            (args.limit,))]
        out = review_slate(picks)
        print(f"  reviewed {out['reviewed']} picks across {out['games']} games")
        print(f"  {out['counts']}\n")
        for p in picks:
            r = out["results"][p["pick_id"]]
            mark = {"OK": "ok  ", "FLAG": "FLAG", "KILL": "KILL"}[r["verdict"]]
            print(f"  {mark} [{p['tier']}] {(p['headline'] or '')[:44]:<44} {r['reason'][:60]}")
    else:
        print(apply_to_picks(limit=args.limit))
