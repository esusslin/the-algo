"""Bet sizing.

Kelly maximizes long-run growth ONLY if your probability is exactly right.
Yours isn't — no one's is — and Kelly is brutally asymmetric about that: at 2x
the correct stake, growth goes negative even when the edge is real. So
everything here is deliberately conservative:

  * fractional Kelly (0.125x while unproven, 0.25x maximum ever)
  * a hard per-bet cap regardless of what Kelly says
  * a same-game correlation haircut
  * a total-slate exposure cap

The failure mode this guards against is not "bet too little". It is a model
that looks calibrated in backtest, is 3 points optimistic in reality, and
compounds that error into ruin over a season.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

from src.config import settings
from src.market.devig import american_to_decimal

log = logging.getLogger(__name__)

MAX_SAFE_FRACTION = 0.25          # never exceed, whatever the config says
MAX_GAME_EXPOSURE_PCT = 4.0       # total stake across all bets on one game
MAX_SLATE_EXPOSURE_PCT = 25.0     # total stake across the whole week


def kelly_fraction(prob: float, american_odds: int | float) -> float:
    """Full-Kelly fraction of bankroll. Negative means no bet.

        f* = (p*b - q) / b     where b = decimal - 1
    """
    if not 0.0 < prob < 1.0:
        return 0.0
    b = american_to_decimal(american_odds) - 1.0
    if b <= 0:
        return 0.0
    return (prob * b - (1.0 - prob)) / b


def sized_stake(prob: float, american_odds: int | float,
                bankroll: float = 1.0,
                fraction: float | None = None,
                max_pct: float | None = None) -> float:
    """Fractional-Kelly stake, capped. Returns absolute stake in bankroll units."""
    frac = settings.KELLY_FRACTION if fraction is None else fraction
    if frac > MAX_SAFE_FRACTION:
        log.warning("KELLY_FRACTION=%.3f exceeds safe max %.3f — clamping",
                    frac, MAX_SAFE_FRACTION)
        frac = MAX_SAFE_FRACTION
    cap = (settings.MAX_BET_PCT_BANKROLL if max_pct is None else max_pct) / 100.0

    f = kelly_fraction(prob, american_odds)
    if f <= 0:
        return 0.0
    return min(f * frac, cap) * bankroll


def apply_correlation_haircut(picks: list[dict], bankroll: float = 1.0) -> list[dict]:
    """Scale stakes down where bets overlap.

    Two levels:

    same game  Over on the total, over on both QBs' passing yards and over on
               the WR1's receiving yards are close to the same bet four times.
               Independent Kelly on each massively overbets the underlying
               position, so total exposure per game is capped.

    slate      A Sunday with 50 positions is not 50 independent wagers; NFL
               outcomes share weather, injury and scoring-environment factors.
               Total weekly exposure is capped too.

    A proper simultaneous-Kelly optimisation would use a real correlation matrix
    (the simulator will produce one). This is the honest crude version until
    then, and it errs toward betting less.
    """
    if not picks:
        return picks

    by_game: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        by_game[p["game_id"]].append(p)

    game_cap = MAX_GAME_EXPOSURE_PCT / 100.0 * bankroll
    for game_id, group in by_game.items():
        total = sum(p["stake"] for p in group)
        if total > game_cap and total > 0:
            scale = game_cap / total
            for p in group:
                p["stake"] *= scale
                p["correlation_scaled"] = round(scale, 4)
            log.info("game %s: %d bets scaled %.2fx for correlation",
                     game_id, len(group), scale)

    slate_cap = MAX_SLATE_EXPOSURE_PCT / 100.0 * bankroll
    total = sum(p["stake"] for p in picks)
    if total > slate_cap and total > 0:
        scale = slate_cap / total
        for p in picks:
            p["stake"] *= scale
            p["slate_scaled"] = round(scale, 4)
        log.info("slate: %d bets scaled %.2fx (total exposure cap)", len(picks), scale)

    return picks


def size_slate(opportunities: Iterable[dict], bankroll: float = 1.0) -> list[dict]:
    """Size a full slate: Kelly per bet, then correlation haircuts."""
    picks = []
    for o in opportunities:
        stake = sized_stake(o["fair_prob"], o["best_price"], bankroll=bankroll)
        if stake <= 0:
            continue
        picks.append({
            **o,
            "stake": stake,
            "kelly_full": kelly_fraction(o["fair_prob"], o["best_price"]),
        })
    return apply_correlation_haircut(picks, bankroll=bankroll)


if __name__ == "__main__":
    print("=== fractional Kelly at -110 (breakeven 52.38%) ===")
    print(f"  {'true prob':>10}{'edge':>9}{'full K':>9}{'0.25x':>9}{'0.125x':>9}{'capped 2%':>11}")
    for p in [0.50, 0.5238, 0.53, 0.55, 0.58, 0.62, 0.70]:
        from src.market.devig import edge_pct
        f = kelly_fraction(p, -110)
        print(f"  {p:>10.4f}{edge_pct(p,-110):>8.2f}%{f:>9.4f}{f*0.25:>9.4f}"
              f"{f*0.125:>9.4f}{sized_stake(p,-110,1.0,0.125,2.0):>11.4f}")

    print("\n=== why fractional: overbetting a 55% edge you think is 58% ===")
    true_p, believed_p = 0.55, 0.58
    for label, frac in [("full Kelly", 1.0), ("0.25x", 0.25), ("0.125x", 0.125)]:
        staked = kelly_fraction(believed_p, -110) * frac
        optimal = kelly_fraction(true_p, -110) * frac
        print(f"  {label:<12} stake={staked:.4f} vs optimal={optimal:.4f} "
              f"-> {staked/optimal if optimal else 0:.2f}x overbet")
    print("  (>2x overbet turns positive-EV betting into negative growth)")
