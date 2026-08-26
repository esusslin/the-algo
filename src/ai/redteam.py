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

Keep `reason` to one sentence and `evidence` to at most 15 words. Quote the \
single fact you are relying on, not the whole data block it came from.

Rules:
- OK is the default and the correct answer for the large majority of bets. On a \
typical slate you should return OK for 85-95% of them.
- FLAG means a real concern that should reduce confidence but not cancel.
- KILL means a concrete disqualifying fact, and you MUST quote it in evidence.

MISSING DATA IS NOT EVIDENCE. Empty injury reports, "no forecast available", \
"not yet declared" and "none recorded" mean we have not collected that data \
yet. They do NOT mean nothing is wrong, and they are NEVER a reason to FLAG or \
KILL. Judge only on facts that are actually present.

LINE MOVEMENT IS IN SCOPE, BUT NEVER GROUNDS FOR KILL. If the context reports a \
NET MOVE marked "significant" and "against the side of this bet", return FLAG. It \
means the market disagrees with us, not that the bet is disqualified. A move marked \
"minor", one marked "toward", or no move at all is not a reason to object at all. \
Use the stated NET MOVE line; do not do your own arithmetic on the individual \
quotes. Reserve KILL for a stated fact that invalidates the bet itself, such as a \
key player ruled out.

Other constraints:
- Never invent facts. If the context contains no stated reason, return OK.
- Do not comment on whether the edge, price or line looks good. Not your job.
- Do not speculate about matchups, form, momentum or "trap games". You have no \
information about those and neither does anyone else.
- A bet being uncertain is not a reason to object. All bets are uncertain."""


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


def _fmt_price(price: Any) -> str:
    """American odds, defensively.

    `price` reaches here from three feeds and has been None (no price recorded
    yet) and float (a REAL column) in production. `:+d` raises on both — and
    `_prompt` is evaluated as an argument to `complete_json`, outside the
    `except AIUnavailable` guard, so a single malformed row would propagate a
    TypeError out of a function documented as never raising.
    """
    try:
        return f"{int(price):+d}"
    except (TypeError, ValueError):
        return "n/a"


# Below these, movement is ordinary market noise rather than a signal. A red-team
# agent that objects to every half-point tick would downgrade most of a slate.
MOVE_THRESHOLD = {"spreads": 1.0, "totals": 1.5}
DEFAULT_MOVE_THRESHOLD = 1.0


def _net_move(pick: dict, moves: list[dict]) -> str:
    """State the net move on the exact side being bet, in points.

    The model is told this rather than asked to derive it. The rows are a
    per-book time series, and the arithmetic that matters — first quote versus
    last quote, one side, one book — is easy to get subtly wrong and impossible
    to audit after the fact from a one-sentence `reason`.

    **Sign convention, which is the whole of this function.** Every row carries
    that side's own number, so for spreads a number getting larger (-3.0 -> 0.0)
    means the market is pricing that side as weaker: a move against the bet. The
    same holds for an over, where a rising total is harder to reach. It inverts
    for an under, where a rising total is help. That single inversion is the bug
    waiting to happen here, and `test_ai_prompt.py` pins all of it.
    """
    side = str(pick.get("side") or "").lower()
    mtype = pick.get("market_type")
    rows = [m for m in moves
            if m.get("market_type") == mtype
            and str(m.get("side") or "").lower() == side
            and isinstance(m.get("line"), (int, float))]
    if len(rows) < 2:
        return ""

    # One book at a time. Mixing books measures disagreement between them, not
    # movement over time.
    by_book: dict[str, list[dict]] = {}
    for m in rows:
        by_book.setdefault(str(m.get("book")), []).append(m)
    book, series = max(by_book.items(), key=lambda kv: len(kv[1]))
    if len(series) < 2:
        return ""

    series.sort(key=lambda m: str(m.get("observed_at") or ""))
    first, last = float(series[0]["line"]), float(series[-1]["line"])
    delta = last - first
    if delta == 0:
        return f"NET MOVE ({book}): none, steady at {first}"

    against = delta < 0 if side == "under" else delta > 0
    threshold = MOVE_THRESHOLD.get(str(mtype), DEFAULT_MOVE_THRESHOLD)
    size = "significant" if abs(delta) >= threshold else "minor"
    return (f"NET MOVE ({book}): {first} -> {last}, "
            f"{abs(delta):.1f} points {'against' if against else 'toward'} "
            f"the side of this bet ({size})")


def _line_moves_text(pick: dict, moves: list[dict]) -> str:
    if not moves:
        return "- none recorded"
    # Oldest first, so movement reads left to right the way a human would draw
    # it. The query returns newest-first for the LIMIT to be meaningful.
    ordered = sorted(moves, key=lambda m: str(m.get("observed_at") or ""))
    out = [f"- {m.get('observed_at')} {m.get('market_type')} {m.get('side')} "
           f"line {m.get('line')} @ {_fmt_price(m.get('price'))} ({m.get('book')})"
           for m in ordered]
    summary = _net_move(pick, ordered)
    if summary:
        out.append(summary)
    return chr(10).join(out)


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

RECENT LINE MOVEMENT (oldest first)
{_line_moves_text(pick, ctx.get('recent_line_moves') or [])}

PROPOSED BET
{pick.get('headline')}
Market type: {info.label} ({info.bet_class})
Side: {pick.get('side')} | Line: {pick.get('line')}
Tier: {pick.get('tier')}

Is there a disqualifying reason not to place this bet?"""


def _evidence_grounded(evidence: str, context_blob: str) -> bool:
    """Does the cited evidence actually appear in the context we supplied?

    Loose token overlap rather than exact matching — the model paraphrases, and
    demanding a literal quote would reject legitimate findings. The bar is only
    that it is talking about something real.
    """
    ev_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", evidence.lower())}
    if not ev_tokens:
        return False
    ctx_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", context_blob.lower())}
    overlap = ev_tokens & ctx_tokens
    return len(overlap) >= max(1, min(2, len(ev_tokens) // 3))


def _movement_only(evidence: str, situational: str, movement: str) -> bool:
    """Does the cited evidence come from the line-movement section and nowhere else?

    Deliberately asymmetric: it must ground in `movement` and fail to ground in
    `situational`. A KILL citing both a quarterback and a line move keeps its
    verdict, because the quarterback is doing the work.
    """
    if not evidence.strip():
        return False
    return (_evidence_grounded(evidence, movement)
            and not _evidence_grounded(evidence, situational))


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
            # 300 was not enough: a model quoting a whole weather dict into
            # `evidence` truncated mid-string, the JSON failed to parse, and the
            # agent failed open to OK — a silent non-review that looked like a
            # clean bill of health. The prompt now caps evidence length; this is
            # the headroom so that a model ignoring the cap still returns
            # something parseable.
            model=settings.MODEL_REASON, max_tokens=600,
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

    # An unevidenced KILL is an opinion, not a finding. Demote it once — and
    # skip the grounding check below, which would otherwise demote it a second
    # time straight to OK for the same reason.
    unevidenced = False
    if verdict == "KILL" and len(evidence.strip()) < 12:
        log.info("demoting unevidenced KILL to FLAG: %s", reason[:80])
        verdict, reason, unevidenced = "FLAG", f"(unevidenced) {reason}", True

    # Evidence must quote something actually in the context. Models will
    # otherwise cite the ABSENCE of data — "no injury report available" — as
    # grounds to object, which would gut a slate whenever ingestion is behind.
    # The two halves are kept apart on purpose: a KILL is allowed to rest on a
    # situational fact but not on line movement alone. See the cap below.
    situational = (str(ctx.get("injuries")) + str(ctx.get("inactives"))
                   + str(ctx.get("weather"))
                   + str(ctx.get("week")) + str(ctx.get("season_type"))
                   + str(ctx.get("home_rest_days")) + str(ctx.get("away_rest_days")))
    # Line movement is grounded against the *rendered* section, not the raw rows.
    # A model citing "moved 3.0 points against this bet" shares no 4-character
    # token with a dict repr of floats, so grounding against the rows would
    # downgrade the finding straight back to OK — silently undoing the one thing
    # this section exists to catch.
    movement = _line_moves_text(pick, ctx.get("recent_line_moves") or [])

    if verdict in ("KILL", "FLAG") and not unevidenced:
        absence_words = ("no forecast", "not yet declared", "none recorded",
                         "no data", "unavailable", "not available", "missing",
                         "no injury", "empty", "lack of", "absence of")
        ev_low = evidence.lower()
        if any(w in ev_low for w in absence_words):
            log.info("overriding %s based on missing data: %s", verdict, evidence[:80])
            verdict, reason, evidence = "OK", "", ""
        elif not _evidence_grounded(evidence, situational + movement):
            log.info("downgrading ungrounded %s: %s", verdict, evidence[:80])
            verdict = "FLAG" if verdict == "KILL" else "OK"
            if verdict == "OK":
                reason, evidence = "", ""

    # A KILL resting only on line movement is capped at FLAG.
    #
    # KILL unpublishes a pick; FLAG caps tier A at B. That difference should turn
    # on whether a stated fact invalidates the bet — a starting quarterback ruled
    # out means the thing you priced no longer exists. A line moving three points
    # means the market disagrees with you, which is a reason to size down, not to
    # stand aside. Letting it unpublish would have the agent quietly enforcing
    # "never bet against a sharp move": a betting strategy, not a safety check,
    # and one nobody here has tested.
    #
    # **Enforced in code rather than in the prompt because the prompt was not
    # enough.** Asked for FLAG, the model returned KILL on six of six samples
    # across two runs. A safety property that depends on instruction-following is
    # not a safety property.
    #
    # There is also a diagnostic reason. A pick live at a number the sharp book
    # left three points ago usually means the pick was generated off a stale
    # line. Killing it hides that behind a clean slate; flagging it leaves the
    # evidence where someone will see it.
    if verdict == "KILL" and _movement_only(evidence, situational, movement):
        log.info("capping movement-only KILL at FLAG: %s", evidence[:80])
        verdict = "FLAG"

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

    total = max(out["reviewed"], 1)
    downgrade_rate = (out["counts"].get("KILL", 0) + out["counts"].get("FLAG", 0)) / total
    if downgrade_rate > 0.30:
        log.warning("redteam downgraded %.0f%% of the slate — that is high. "
                    "Check the reasons; a model objecting to most bets is "
                    "usually reacting to thin context, not real problems.",
                    downgrade_rate * 100)
    log.info("redteam: %s across %d games, %d downgraded (ai_ran=%s, sources=%s)",
             out["counts"], out["games"], changed, out["ai_ran"], out["sources"])
    return {"reviewed": out["reviewed"], "games": out["games"],
            "counts": out["counts"], "changed": changed,
            "ai_ran": out["ai_ran"], "reviewed_by_model": out["reviewed_by_model"],
            "sources": out["sources"], "errors": out.get("errors", []),
            "results": out["results"]}


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
