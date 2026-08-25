"""Research-plane arithmetic. Pure functions, no warehouse, no model fitting.

Three pieces of maths that decide what a prediction *means*, none of which had tests:

**`shrink`** — cold-start blending. Two lines of arithmetic that are trivially easy to get
backwards, and if they were, Week 1 features would be pure observation from a single game
while Week 15 leaned on last season. Exactly inverted, and nothing downstream would say so.

**`_norm_cdf` / `residual_to_cover_prob`** — the conversion from "we predict the home side
beats the spread by 1.2 points" to "we think there's a 54% chance they cover". Every
published probability passes through it. A wrong sign flips every pick to its opposite; a
wrong scale makes the model look calibrated when it is not.

These run in CI on any machine — no DuckDB, no parquet, no trained bundle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.coldstart import DEFAULT_K, FITTED_K, k_for, shrink
from research.train import MARGIN_SD, residual_to_cover_prob


# --- cold-start shrinkage -----------------------------------------------------------


def test_no_observations_means_pure_prior() -> None:
    """Week 1. There is no within-season data, so the rating must be entirely last
    season's regressed number — not zero, which downstream reads as 'exactly average'."""
    assert shrink(observed=0.9, n_plays=0, prior=0.2, k=240) == 0.2


def test_many_observations_swamp_the_prior() -> None:
    """Week 15. The prior should be nearly irrelevant — but never *exactly* irrelevant.

    Asserted against the analytic residual rather than an eyeballed epsilon. The prior
    still pulls the estimate by `k/(n+k) × (prior − observed)`, which at n=1e6 and k=240
    is about 1.7e-4. That's the correct behaviour: shrinkage approaches the observation
    asymptotically and never quite arrives, because it is a weighted average and the
    prior's weight is never zero.

    Pinning the exact value rather than "close enough" means a change to the weighting
    formula fails here instead of hiding inside a tolerance somebody widened.
    """
    observed, prior, k, n = 0.9, 0.2, 240.0, 1_000_000
    got = shrink(observed=observed, n_plays=n, prior=prior, k=k)

    residual_pull = (k / (n + k)) * (prior - observed)
    assert got == pytest.approx(observed + residual_pull, rel=1e-12)

    # Directionally: still pulled toward the prior, but negligibly.
    assert prior < got < observed
    assert abs(got - observed) < 1e-3


def test_n_equal_to_k_is_an_even_blend() -> None:
    """The definition of `k`: the number of plays at which observation and prior are
    trusted equally. If this drifts, `k` no longer means what the fit measured."""
    assert shrink(observed=1.0, n_plays=240, prior=0.0, k=240) == pytest.approx(0.5)


def test_k_zero_ignores_the_prior_entirely() -> None:
    """`k=0` is the baseline arm of the walk-forward comparison — today's behaviour,
    before shrinkage. It must be exactly the observed value, not an approximation."""
    assert shrink(observed=0.73, n_plays=5, prior=-99.0, k=0) == 0.73


def test_negative_k_is_treated_as_no_shrinkage() -> None:
    """Defensive: a misconfigured negative k would otherwise produce weights above 1 and
    extrapolate past the observation."""
    assert shrink(observed=0.73, n_plays=5, prior=-99.0, k=-10) == 0.73


def test_the_blend_never_leaves_the_interval() -> None:
    """A weighted average of two numbers must lie between them. Weights outside [0,1] —
    the classic sign error here — would extrapolate, and an extrapolated team rating looks
    like a strong signal rather than a bug."""
    for n in (0, 1, 10, 240, 1_000, 100_000):
        for obs, prior in ((0.9, 0.2), (0.2, 0.9), (-1.5, 0.4), (0.0, 0.0)):
            got = shrink(obs, n, prior, 240)
            assert min(obs, prior) - 1e-12 <= got <= max(obs, prior) + 1e-12, (n, obs, prior)


def test_more_data_moves_monotonically_toward_the_observation() -> None:
    """Each additional play should shift the estimate toward what was observed, never
    away. Non-monotonicity would mean a team's rating oscillated as the season went on."""
    prior, obs = 0.0, 1.0
    seq = [shrink(obs, n, prior, 240) for n in (0, 30, 60, 120, 240, 480, 960)]
    assert seq == sorted(seq)
    assert seq[0] == prior


def test_identical_inputs_are_a_fixed_point() -> None:
    """If prior and observation agree, no amount of data should change the answer."""
    for n in (0, 240, 10_000):
        assert shrink(0.42, n, 0.42, 240) == pytest.approx(0.42)


def test_fitted_k_is_per_play_class_and_positive() -> None:
    """Fitted by walk-forward on 2013-2024, held out on 2025. Pinned so an accidental
    edit is visible: rush wants roughly half the shrinkage pass does, which contradicts
    the rule of thumb in IN_SEASON_LEARNING.md and is documented there."""
    assert FITTED_K["rush"] < FITTED_K["pass"]
    assert all(v > 0 for v in FITTED_K.values())
    assert k_for("pass") == FITTED_K["pass"]


def test_an_unknown_play_class_falls_back_rather_than_raising() -> None:
    """A new play class should shrink like the class we measured most, not crash a
    training run halfway through."""
    assert k_for("special_teams") == DEFAULT_K


# --- residual to probability --------------------------------------------------------


def test_a_zero_residual_is_a_coin_flip() -> None:
    """Predicting exactly the market line means no opinion — 50%. If this were off, every
    pick would carry a systematic lean in one direction."""
    assert float(residual_to_cover_prob(np.array([0.0]))[0]) == pytest.approx(0.5)


def test_a_positive_residual_favours_the_home_side() -> None:
    """Sign convention. Getting this backwards would invert every published pick while
    producing perfectly plausible-looking probabilities."""
    assert float(residual_to_cover_prob(np.array([3.0]))[0]) > 0.5
    assert float(residual_to_cover_prob(np.array([-3.0]))[0]) < 0.5


def test_the_conversion_is_symmetric_about_the_line() -> None:
    p_up = float(residual_to_cover_prob(np.array([2.5]))[0])
    p_down = float(residual_to_cover_prob(np.array([-2.5]))[0])
    assert p_up + p_down == pytest.approx(1.0)


def test_one_standard_deviation_lands_where_the_normal_says() -> None:
    """Not a tautology — it checks the scale divisor is `sd` and not `sd**2` or `2*sd`.
    A wrong scale produces probabilities that are ordered correctly and calibrated
    wrongly, which the calibration table would show and nothing else would."""
    p = float(residual_to_cover_prob(np.array([MARGIN_SD]))[0])
    assert p == pytest.approx(0.8413, abs=1e-3)


def test_probabilities_stay_in_range_at_absurd_inputs() -> None:
    """A runaway model prediction must not produce a probability above 1, which would
    make Kelly demand an infinite stake."""
    extreme = residual_to_cover_prob(np.array([-1e6, -100.0, 0.0, 100.0, 1e6]))
    assert np.all(extreme >= 0.0) and np.all(extreme <= 1.0)


def test_the_conversion_is_monotonic() -> None:
    """A larger predicted margin must never mean a lower cover probability."""
    residuals = np.linspace(-30, 30, 121)
    probs = residual_to_cover_prob(residuals)
    assert np.all(np.diff(probs) >= 0)


def test_it_matches_the_error_function_directly() -> None:
    """Independent check against `math.erf` rather than against itself, so a refactor to
    a faster approximation has something real to be measured against."""
    for r in (-7.5, -1.0, 0.0, 1.0, 7.5):
        expected = 0.5 * (1.0 + math.erf(r / MARGIN_SD / math.sqrt(2.0)))
        assert float(residual_to_cover_prob(np.array([r]))[0]) == pytest.approx(expected)


def test_a_wider_sd_pulls_probabilities_toward_the_middle() -> None:
    """Totals use a different spread of outcomes than sides. More uncertainty must mean
    less confidence, not more."""
    tight = float(residual_to_cover_prob(np.array([3.0]), sd=5.0)[0])
    wide = float(residual_to_cover_prob(np.array([3.0]), sd=50.0)[0])
    assert 0.5 < wide < tight
