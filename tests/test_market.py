"""Devig, edge and consensus invariants.

The devig methods are pure maths with properties that must hold for every
input. Property tests over random markets catch what hand-picked cases miss.
"""
from __future__ import annotations

import random

import pytest

from src.market.consensus import _pair_line
from src.market.devig import (american_to_decimal, american_to_prob,
                              breakeven_prob, decimal_to_american, devig,
                              devig_american, edge_pct,
                              fair_prices_all_methods, prob_to_american)

METHODS = ["multiplicative", "additive", "power", "shin"]


def random_price(rng: random.Random) -> int:
    return rng.choice([rng.randint(-5000, -101), rng.randint(101, 5000)])


# ---- conversion ----------------------------------------------------------
def test_american_to_prob_known():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(150) == pytest.approx(0.4)


def test_decimal_conversion_known():
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_roundtrip():
    for odds in (-5000, -300, -110, -101, 101, 150, 2000, 5000):
        assert prob_to_american(american_to_prob(odds)) == odds
        assert decimal_to_american(american_to_decimal(odds)) == odds


def test_zero_odds_rejected():
    with pytest.raises(ValueError):
        american_to_prob(0)


# ---- devig properties ----------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_devig_sums_to_one(method):
    rng = random.Random(11)
    for _ in range(400):
        probs = devig_american([random_price(rng), random_price(rng)], method)
        assert sum(probs) == pytest.approx(1.0, abs=1e-8)
        assert all(0 < p < 1 for p in probs)


@pytest.mark.parametrize("method", METHODS)
def test_devig_symmetric_market_is_fifty_fifty(method):
    for price in (-110, -105, -120, 100):
        probs = devig_american([price, price], method)
        assert probs[0] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_devig_preserves_ordering(method):
    """The favourite must stay the favourite after removing vig."""
    rng = random.Random(3)
    for _ in range(200):
        a, b = random_price(rng), random_price(rng)
        raw = [american_to_prob(a), american_to_prob(b)]
        fair = devig(raw, method)
        if raw[0] != raw[1]:
            assert (raw[0] > raw[1]) == (fair[0] > fair[1])


def test_methods_diverge_on_longshots():
    """If they all agreed there would be no point computing four of them —
    and the divergence is largest exactly where prop edges live."""
    fair = fair_prices_all_methods([2000, -3000])
    longshot = [v[0] for v in fair.values()]
    assert max(longshot) - min(longshot) > 0.005


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        devig([0.5, 0.5], "wishful_thinking")


# ---- edge ----------------------------------------------------------------
def test_breakeven_known():
    assert breakeven_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert breakeven_prob(-105) == pytest.approx(0.5122, abs=1e-4)
    assert breakeven_prob(100) == pytest.approx(0.5)


def test_edge_zero_at_breakeven():
    for price in (-200, -110, 100, 150, 500):
        assert edge_pct(breakeven_prob(price), price) == pytest.approx(0, abs=1e-9)


def test_edge_sign():
    assert edge_pct(0.55, -110) > 0
    assert edge_pct(0.50, -110) < 0


def test_fair_coin_at_plus_money_has_edge():
    """The market engine's whole premise: a 50/50 paid better than even money
    is profitable without predicting anything."""
    assert edge_pct(0.50, 112) == pytest.approx(6.0, abs=0.01)


def test_full_vig_is_negative_four_and_a_half():
    """A -110 market with no mispricing scores -4.55%. Seeing this as the
    median across all markets is the expected healthy null result."""
    assert edge_pct(0.5, -110) == pytest.approx(-4.545, abs=0.01)


# ---- two-sided pairing ---------------------------------------------------
def test_spreads_pair_on_opposite_lines():
    """Home -2.5 and away +2.5 are the same market. Pairing them wrongly
    devigs mismatched outcomes and produces plausible, wrong fair prices."""
    assert _pair_line("spreads", "home", -2.5) == _pair_line("spreads", "away", 2.5)


def test_totals_pair_on_same_line():
    assert _pair_line("totals", "over", 48.5) == _pair_line("totals", "under", 48.5)


def test_moneyline_has_no_line():
    assert _pair_line("h2h", "home", 0) == _pair_line("h2h", "away", 0) == 0.0


def test_period_spreads_pair_like_spreads():
    assert _pair_line("spreads_h1", "home", -1.5) == _pair_line("spreads_h1", "away", 1.5)
