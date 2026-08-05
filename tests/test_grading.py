"""Grading sign conventions.

These run unattended for months. A flipped sign produces a plausible ~50% hit
rate while silently inverting every result — the single most dangerous bug in a
betting system, and one that already bit this project once (nflverse and the
Odds API store spreads with OPPOSITE conventions).

Every case here is hand-verified against a real box score. If one fails, do not
adjust the test.
"""
from __future__ import annotations

import pytest

from src.picks.grading import (LOSS, PUSH, WIN, grade_game_market, grade_prop,
                               payout_for)


# ---- moneyline -----------------------------------------------------------
@pytest.mark.parametrize("side,home,away,expected", [
    ("home", 24, 17, WIN),
    ("away", 24, 17, LOSS),
    ("home", 17, 24, LOSS),
    ("away", 17, 24, WIN),
    ("home", 20, 20, PUSH),
    ("away", 20, 20, PUSH),
])
def test_moneyline(side, home, away, expected):
    assert grade_game_market("h2h", side, 0, home, away) == expected


# ---- spreads -------------------------------------------------------------
# Odds API convention: home favourite is NEGATIVE (home -3.5 -> line = -3.5).
# Home covers when (margin) + line > 0.
@pytest.mark.parametrize("side,line,home,away,expected", [
    ("home", -3.5, 24, 17, WIN),      # won by 7, laid 3.5
    ("home", -7.5, 24, 17, LOSS),     # won by 7, laid 7.5
    ("home", -7.0, 24, 17, PUSH),     # won by exactly 7
    ("home", +3.5, 17, 24, LOSS),     # lost by 7, got 3.5
    ("home", +10.5, 17, 24, WIN),     # lost by 7, got 10.5
    ("away", +3.5, 24, 17, LOSS),     # away lost by 7, got 3.5
    ("away", +10.5, 24, 17, WIN),
    ("away", -3.5, 17, 24, WIN),      # away won by 7, laid 3.5
    ("away", -7.0, 17, 24, PUSH),
])
def test_spreads(side, line, home, away, expected):
    assert grade_game_market("spreads", side, line, home, away) == expected


def test_spread_sign_is_not_symmetric():
    """A flipped sign must change the answer. If both conventions grade the
    same, the test cannot detect an inversion — which is how this class of bug
    survives."""
    correct = grade_game_market("spreads", "home", -3.5, 24, 17)
    flipped = grade_game_market("spreads", "home", 3.5, 24, 17)
    assert correct == WIN
    assert flipped == WIN      # 7-point win covers either way at 3.5
    # but at a line between the two, they must diverge
    assert grade_game_market("spreads", "home", -7.5, 24, 17) == LOSS
    assert grade_game_market("spreads", "home", 7.5, 24, 17) == WIN


# ---- totals --------------------------------------------------------------
@pytest.mark.parametrize("side,line,home,away,expected", [
    ("over", 40.5, 24, 17, WIN),      # 41 points
    ("under", 40.5, 24, 17, LOSS),
    ("over", 41.0, 24, 17, PUSH),
    ("under", 41.0, 24, 17, PUSH),
    ("over", 44.5, 24, 17, LOSS),
    ("under", 44.5, 24, 17, WIN),
    ("over", 0.5, 0, 0, LOSS),        # scoreless
])
def test_totals(side, line, home, away, expected):
    assert grade_game_market("totals", side, line, home, away) == expected


# ---- period markets inherit the same conventions -------------------------
@pytest.mark.parametrize("market", ["spreads_h1", "spreads_h2", "spreads_q1"])
def test_period_spreads_use_same_convention(market):
    assert grade_game_market(market, "home", -3.5, 24, 17) == WIN
    assert grade_game_market(market, "home", -7.5, 24, 17) == LOSS


# ---- props ---------------------------------------------------------------
@pytest.mark.parametrize("side,line,actual,expected", [
    ("over", 58.5, 62, WIN),
    ("over", 58.5, 51, LOSS),
    ("under", 58.5, 51, WIN),
    ("under", 58.5, 62, LOSS),
    ("over", 60.0, 60, PUSH),
    ("under", 60.0, 60, PUSH),
    ("over", 0.5, 0, LOSS),
])
def test_props(side, line, actual, expected):
    assert grade_prop(side, line, actual) == expected


def test_prop_pending_when_stat_missing():
    from src.picks.grading import PENDING
    assert grade_prop("over", 58.5, None) == PENDING


@pytest.mark.parametrize("side,actual,expected", [
    ("yes", 1, WIN), ("yes", 0, LOSS), ("yes", 2, WIN),
    ("no", 0, WIN), ("no", 1, LOSS),
])
def test_one_sided_props(side, actual, expected):
    """Anytime TD and similar: 'did it happen at all', not a threshold."""
    assert grade_prop(side, 0, actual, one_sided=True) == expected


# ---- payout --------------------------------------------------------------
def test_payout_underdog():
    assert payout_for(WIN, 1.0, 150) == pytest.approx(1.5)


def test_payout_favourite():
    assert payout_for(WIN, 1.0, -110) == pytest.approx(0.9091, abs=1e-4)


def test_payout_loss_is_stake():
    assert payout_for(LOSS, 2.5, -110) == pytest.approx(-2.5)


def test_payout_push_is_zero():
    assert payout_for(PUSH, 5.0, -110) == 0.0


def test_payout_scales_with_stake():
    assert payout_for(WIN, 3.0, 100) == pytest.approx(3.0)
