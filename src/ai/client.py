"""Anthropic client wrapper: retries, cost accounting, budget enforcement.

Every call is logged to `ai_calls` with tokens and dollar cost. Two reasons:
you can see what the AI layer actually costs rather than guessing, and when a
prompt starts misbehaving you can find it by cost before you find it by output.

Budget is enforced with graceful degradation rather than hard failure — an
exhausted budget should cost you narrative quality, never a pick.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from src.config import settings
from src.db import db, insert_row, query, utcnow

log = logging.getLogger(__name__)

# USD per million tokens. Update if pricing changes; used only for the local
# ledger, never for billing.
PRICING = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-sonnet-5": {"in": 3.00, "out": 15.00},
    "claude-opus-5": {"in": 15.00, "out": 75.00},
}
DEFAULT_PRICE = {"in": 3.00, "out": 15.00}


class AIUnavailable(RuntimeError):
    """Raised when the AI layer cannot or should not run. Callers must treat
    this as 'proceed without AI', never as a fatal error."""


def configured() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def spend_this_month() -> float:
    first = datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    row = query("SELECT COALESCE(SUM(cost_usd),0) AS s FROM ai_calls "
                "WHERE created_at >= ?", (first,))
    return float(row[0]["s"]) if row else 0.0


def budget_remaining_pct() -> float:
    budget = max(settings.AI_MONTHLY_BUDGET_USD, 0.01)
    return max(0.0, 1.0 - spend_this_month() / budget) * 100.0


def _cost(model: str, tin: int, tout: int) -> float:
    p = PRICING.get(model, DEFAULT_PRICE)
    return (tin / 1_000_000) * p["in"] + (tout / 1_000_000) * p["out"]


def _log_call(agent: str, model: str, tin: int, tout: int,
              ref_type: str = "", ref_id: str = "") -> float:
    cost = _cost(model, tin, tout)
    with db() as conn:
        insert_row(conn, "ai_calls", {
            "agent": agent, "model": model,
            "input_tokens": tin, "output_tokens": tout,
            "cost_usd": round(cost, 6),
            "ref_type": ref_type, "ref_id": str(ref_id),
            "created_at": utcnow(),
        })
    return cost


def complete(prompt: str, *, agent: str, model: str | None = None,
             system: str = "", max_tokens: int = 800,
             temperature: float = 0.0,
             ref_type: str = "", ref_id: str = "",
             min_budget_pct: float = 2.0) -> str:
    """One completion. Raises AIUnavailable rather than returning junk."""
    if not configured():
        raise AIUnavailable("ANTHROPIC_API_KEY not set")
    if budget_remaining_pct() < min_budget_pct:
        raise AIUnavailable(
            f"AI budget nearly exhausted ({budget_remaining_pct():.1f}% left of "
            f"${settings.AI_MONTHLY_BUDGET_USD})")

    model = model or settings.MODEL_REASON
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — never let the AI layer break a request
        raise AIUnavailable(f"{type(exc).__name__}: {exc}") from exc

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    _log_call(agent, model, resp.usage.input_tokens, resp.usage.output_tokens,
              ref_type, ref_id)
    return text.strip()


def complete_json(prompt: str, **kwargs) -> dict:
    """Completion parsed as JSON.

    Models sometimes wrap JSON in prose or fences even when told not to, so we
    extract the first balanced object rather than trusting the whole response.
    """
    raw = complete(prompt, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise AIUnavailable(f"could not parse JSON from response: {raw[:200]}")


def usage_summary() -> dict:
    first = datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = query(
        "SELECT agent, model, COUNT(*) n, SUM(input_tokens) tin, "
        "SUM(output_tokens) tout, SUM(cost_usd) cost FROM ai_calls "
        "WHERE created_at >= ? GROUP BY agent, model ORDER BY cost DESC", (first,))
    return {
        "configured": configured(),
        "month_spend_usd": round(spend_this_month(), 4),
        "budget_usd": settings.AI_MONTHLY_BUDGET_USD,
        "remaining_pct": round(budget_remaining_pct(), 1),
        "by_agent": [dict(r) for r in rows],
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="anthropic client")
    p.add_argument("command", choices=["usage", "ping"])
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "usage":
        u = usage_summary()
        print(f"  configured : {u['configured']}")
        print(f"  spend      : ${u['month_spend_usd']} of ${u['budget_usd']} "
              f"({u['remaining_pct']}% left)")
        for r in u["by_agent"]:
            print(f"    {r['agent']:<14}{r['model']:<28}{r['n']:>5} calls  "
                  f"${r['cost']:.4f}")
    else:
        try:
            print(" ", complete("Reply with exactly: ok", agent="ping",
                                max_tokens=10))
        except AIUnavailable as exc:
            print(f"  unavailable: {exc}")
