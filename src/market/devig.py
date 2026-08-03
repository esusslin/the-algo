"""Vig removal — converting posted odds into fair probabilities.

Raw implied probabilities sum to >1; the excess is the bookmaker's margin. How
you remove it materially changes the answer, especially away from even money:

  multiplicative  divide by the sum. Simple, but systematically OVERPRICES
                  favorites (assumes margin is proportional to probability).
  additive        subtract the margin equally across outcomes. OVERPRICES
                  longshots.
  power           solve for k such that sum(p_i^k) == 1. Generally the best
                  general-purpose choice for two-way markets. Default.
  shin            models the share of bets from insiders. Best-in-class where
                  informed money is present; closest to sharp book behavior.

For a -110/-110 market all four agree to within a rounding error. For a +2000
longshot they disagree enormously — which is exactly where prop and alt-line
edges live, so the choice is not academic.

Practice: compute fair prices from the SHARPEST book available (Pinnacle), then
shop that number against soft books. See market/consensus.py.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# odds conversion
# --------------------------------------------------------------------------
def american_to_prob(odds: int | float) -> float:
    """American odds -> raw implied probability (vig included)."""
    o = float(odds)
    if o == 0:
        raise ValueError("american odds cannot be 0")
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


def american_to_decimal(odds: int | float) -> float:
    o = float(odds)
    if o == 0:
        raise ValueError("american odds cannot be 0")
    return (o / 100.0) + 1.0 if o > 0 else (100.0 / (-o)) + 1.0


def decimal_to_american(dec: float) -> int:
    if dec <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    return round((dec - 1.0) * 100) if dec >= 2.0 else round(-100.0 / (dec - 1.0))


def prob_to_american(p: float) -> int:
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return decimal_to_american(1.0 / p)


def overround(probs: Sequence[float]) -> float:
    """Total margin. 1.045 means a 4.5% hold."""
    return float(sum(probs))


def hold_pct(probs: Sequence[float]) -> float:
    s = overround(probs)
    return (s - 1.0) / s * 100.0 if s else 0.0


# --------------------------------------------------------------------------
# devig methods — each takes raw implied probs, returns fair probs summing to 1
# --------------------------------------------------------------------------
def devig_multiplicative(probs: Sequence[float]) -> list[float]:
    s = sum(probs)
    if s <= 0:
        raise ValueError("probabilities must be positive")
    return [p / s for p in probs]


def devig_additive(probs: Sequence[float]) -> list[float]:
    n = len(probs)
    excess = (sum(probs) - 1.0) / n
    out = [p - excess for p in probs]
    # Guard: additive can push extreme longshots negative on high-hold markets.
    if any(p <= 0 for p in out):
        return devig_multiplicative(probs)
    return out


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Dependency-free root finder. Assumes f(lo) and f(hi) bracket a root."""
    flo = f(lo)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if (flo < 0) != (fmid < 0):
            hi = mid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2.0


def devig_power(probs: Sequence[float]) -> list[float]:
    """Solve for k such that sum(p_i ** k) == 1.

    k > 1 when there is vig to remove. Monotone in k, so bisection is safe.
    """
    if any(p <= 0 for p in probs):
        raise ValueError("probabilities must be positive")
    if abs(sum(probs) - 1.0) < 1e-12:
        return list(probs)

    def f(k: float) -> float:
        return sum(p ** k for p in probs) - 1.0

    lo, hi = 0.05, 1.0
    # sum(p^k) decreases as k grows (all p<1), so expand hi until f(hi) < 0
    while f(hi) > 0 and hi < 100.0:
        hi *= 2.0
    if f(lo) < 0:
        return devig_multiplicative(probs)
    k = _bisect(f, lo, hi)
    out = [p ** k for p in probs]
    s = sum(out)
    return [p / s for p in out]  # normalize away residual float error


def devig_shin(probs: Sequence[float]) -> list[float]:
    """Shin (1993): assumes a proportion z of volume comes from insiders.

        p_fair_i = [sqrt(z^2 + 4(1-z) * p_i^2 / S) - z] / (2(1-z))

    where S = sum of raw implied probs. Solve z so fair probs sum to 1.
    """
    if any(p <= 0 for p in probs):
        raise ValueError("probabilities must be positive")
    s = sum(probs)
    if abs(s - 1.0) < 1e-12:
        return list(probs)

    def fair(z: float) -> list[float]:
        if z >= 1.0:
            return devig_multiplicative(probs)
        return [
            (((z * z + 4.0 * (1.0 - z) * (p * p) / s) ** 0.5) - z) / (2.0 * (1.0 - z))
            for p in probs
        ]

    def f(z: float) -> float:
        return sum(fair(z)) - 1.0

    lo, hi = 0.0, 0.99
    if f(lo) * f(hi) > 0:
        return devig_power(probs)
    z = _bisect(f, lo, hi)
    out = fair(z)
    total = sum(out)
    return [p / total for p in out]


METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(probs: Sequence[float], method: str = "power") -> list[float]:
    if method not in METHODS:
        raise ValueError(f"unknown devig method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](list(probs))


def devig_american(odds: Iterable[int | float], method: str = "power") -> list[float]:
    """Convenience: american odds for one market -> fair probabilities."""
    return devig([american_to_prob(o) for o in odds], method=method)


def fair_prices_all_methods(odds: Sequence[int | float]) -> dict[str, list[float]]:
    """Every method at once — use to check a conclusion isn't method-dependent.

    If a bet only shows edge under one devig method, it is not a bet.
    """
    raw = [american_to_prob(o) for o in odds]
    return {name: fn(list(raw)) for name, fn in METHODS.items()}


def conservative_fair(odds: Sequence[int | float], index: int) -> float:
    """Least favorable fair probability for one side across all methods.

    Sensible default for bet sizing: it makes edge estimates pessimistic rather
    than optimistic, which is the direction you want to be wrong in.
    """
    return max(v[index] for v in fair_prices_all_methods(odds).values())


# --------------------------------------------------------------------------
# edge / EV
# --------------------------------------------------------------------------
def edge_pct(fair_prob: float, offered_american: int | float) -> float:
    """Expected value per unit staked, as a percentage.

    edge = p * (decimal - 1) - (1 - p)
    """
    dec = american_to_decimal(offered_american)
    return (fair_prob * (dec - 1.0) - (1.0 - fair_prob)) * 100.0


def breakeven_prob(offered_american: int | float) -> float:
    """Win rate needed just to break even. -110 -> 0.5238."""
    return 1.0 / american_to_decimal(offered_american)


if __name__ == "__main__":
    print("=== standard -110/-110 ===")
    for name, fp in fair_prices_all_methods([-110, -110]).items():
        print(f"  {name:<15} {fp[0]:.4f} / {fp[1]:.4f}")
    print(f"  hold: {hold_pct([american_to_prob(-110)] * 2):.2f}%")

    print("\n=== longshot +2000 / -3000 (methods diverge) ===")
    for name, fp in fair_prices_all_methods([2000, -3000]).items():
        print(f"  {name:<15} {fp[0]:.4f} / {fp[1]:.4f}   (+2000 fair: {prob_to_american(fp[0]):+d})")

    print("\n=== edge example: model says 55%, offered -110 ===")
    print(f"  breakeven at -110: {breakeven_prob(-110):.4f}")
    print(f"  edge: {edge_pct(0.55, -110):+.2f}%")
