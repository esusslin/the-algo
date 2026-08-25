"""Arithmetic that decides what gets published and what gets spent.

`test_market.py` covers odds conversion and devig. This covers the three pieces of maths
downstream of that which had no tests at all:

**Weighted median** — how 23 books become one fair price. A mean would let one stale book
drag consensus; the median is the reason a soft line becomes an *opportunity* rather than
noise in the average. It is also the kind of function that looks obviously right and has
an off-by-one in the accumulation.

**Poll intervals** — which bucket a kickoff falls into, and therefore how many credits a
week costs. An off-by-one at a boundary is the difference between polling every 5 minutes
and every 6 hours.

**The credit ledger** — the degradation ladder that decides which markets get dropped as
the budget tightens. A previous month burned 52,000 credits against a 20,000 budget; the
throttle that caused it is fixed, and this pins the guard that would have caught it.

Everything here is pure or patched. No network, no database, no API key.
"""

from __future__ import annotations

import pytest

from src.fetchers.odds_api import CreditLedger, PollTier
from src.market.consensus import _opposite, _pair_line, _weighted_median


# --- weighted median ----------------------------------------------------------------


def test_single_value_is_its_own_median() -> None:
    assert _weighted_median([2.5], [1.0]) == 2.5


def test_equal_weights_behave_like_an_ordinary_median() -> None:
    assert _weighted_median([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 2.0


def test_input_order_does_not_matter() -> None:
    """The function sorts internally. If it didn't, consensus would depend on the order
    books happened to be returned by the API — which changes between calls."""
    assert _weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == _weighted_median(
        [1.0, 2.0, 3.0], [1, 1, 1]
    )


def test_one_stale_book_cannot_drag_the_consensus() -> None:
    """The entire reason this is a median and not a mean.

    Four books agree near -3; one is stale at -14. A weighted mean lands around -5.2,
    which would manufacture a fictional edge against every book on the real number.
    """
    values = [-3.0, -3.0, -3.5, -3.0, -14.0]
    weights = [1.0] * 5
    assert _weighted_median(values, weights) == -3.0
    mean = sum(values) / len(values)
    assert mean < -5.0  # what we are deliberately not doing


def test_weight_actually_shifts_the_result() -> None:
    """A sharp book carrying more weight should be able to move consensus toward itself —
    otherwise the sharp anchoring in `build_fair_prices` is decorative."""
    light = _weighted_median([1.0, 5.0], [1.0, 1.0])
    heavy = _weighted_median([1.0, 5.0], [1.0, 99.0])
    assert light == 1.0
    assert heavy == 5.0


def test_a_dominant_weight_wins_outright() -> None:
    assert _weighted_median([10.0, 20.0, 30.0], [0.01, 100.0, 0.01]) == 20.0


def test_zero_weighted_values_do_not_win() -> None:
    """A book weighted to zero — excluded for staleness, say — must not become the
    consensus merely by sorting first."""
    assert _weighted_median([-99.0, -3.0, -3.0], [0.0, 1.0, 1.0]) == -3.0


def test_empty_input_raises_rather_than_guessing() -> None:
    """Returning a default here would invent a fair price for a market nobody quoted,
    and every edge computed against it would be fictional."""
    with pytest.raises(ValueError):
        _weighted_median([], [])


def test_lower_median_on_an_even_split_is_the_documented_behaviour() -> None:
    """With two equally weighted middle values the accumulator crosses at the lower one.

    Pinned deliberately: it's a defensible choice for prices (conservative, and it stays
    on a number a book actually quoted rather than interpolating to one nobody offers),
    but it *is* a choice, and a future refactor to interpolate would silently change
    every fair price by half a tick.
    """
    assert _weighted_median([2.0, 4.0], [1.0, 1.0]) == 2.0


# --- market pairing -----------------------------------------------------------------


def test_spreads_pair_by_flipping_the_away_line() -> None:
    """Home -2.5 and away +2.5 are the same market. If they didn't map to one key, every
    spread would devig against itself and produce a 100% edge."""
    assert _pair_line("spreads", "home", -2.5) == -2.5
    assert _pair_line("spreads", "away", 2.5) == -2.5


def test_totals_share_a_line_and_are_not_flipped() -> None:
    assert _pair_line("totals", "over", 44.5) == 44.5
    assert _pair_line("totals", "under", 44.5) == 44.5


def test_moneyline_collapses_to_zero() -> None:
    assert _pair_line("h2h", "home", 0.0) == 0.0
    assert _pair_line("h2h", "away", -110.0) == 0.0


def test_period_spreads_flip_like_full_game_spreads() -> None:
    assert _pair_line("spreads_h1", "away", 1.5) == -1.5


def test_opposite_sides_are_symmetric() -> None:
    for side in ("home", "away", "over", "under"):
        other = _opposite(side)
        assert other is not None, side
        assert _opposite(other) == side


def test_unknown_side_has_no_opposite() -> None:
    assert _opposite("neither") is None


# --- poll scheduling ----------------------------------------------------------------


TIER = PollTier("t", ["h2h"], ["us"], {2: 5, 12: 10, 72: 30, 240: 60})


@pytest.mark.parametrize(
    "hours,expected",
    [
        (0.5, 5),      # inside the final two hours
        (2.0, 5),      # boundary is inclusive
        (2.01, 10),    # just past it
        (12.0, 10),
        (12.5, 30),
        (72.0, 30),
        (100.0, 60),
        (240.0, 60),
        (241.0, None),  # beyond every bucket — do not poll
    ],
)
def test_interval_buckets_are_inclusive_at_the_upper_bound(hours: float, expected) -> None:
    """Boundaries are where this goes wrong, and the cost is asymmetric: an off-by-one
    that promotes a game into the 5-minute bucket a day early multiplies its credit
    spend by roughly 70."""
    assert TIER.interval_minutes(hours) == expected


def test_a_game_far_out_is_not_polled_at_all() -> None:
    assert TIER.interval_minutes(10_000) is None


def test_buckets_are_read_in_ascending_order_regardless_of_declaration() -> None:
    """The schedule is a dict; if the implementation didn't sort, a differently-ordered
    literal would silently change the polling rate."""
    scrambled = PollTier("s", ["h2h"], ["us"], {240: 60, 2: 5, 72: 30, 12: 10})
    assert scrambled.interval_minutes(1.0) == 5
    assert scrambled.interval_minutes(50.0) == 30


# --- credit ledger ------------------------------------------------------------------


def ledger_at(monkeypatch: pytest.MonkeyPatch, used: int, budget: int = 20_000) -> CreditLedger:
    led = CreditLedger(budget=budget)
    monkeypatch.setattr(led, "used_this_month", lambda: used)
    return led


def test_remaining_pct_is_a_percentage_not_a_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """`allows()` compares against 30, 15 and 5. If this ever returned 0-1 instead of
    0-100 every tier would be permitted forever and the budget guard would be inert."""
    assert ledger_at(monkeypatch, 0).remaining_pct() == 100.0
    assert ledger_at(monkeypatch, 10_000).remaining_pct() == 50.0
    assert ledger_at(monkeypatch, 20_000).remaining_pct() == 0.0


def test_overspend_floors_at_zero_rather_than_going_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative percentage would still be less than every threshold, so the ladder would
    behave — but it would read as nonsense in a log at exactly the moment someone is
    reading logs."""
    assert ledger_at(monkeypatch, 99_999).remaining_pct() == 0.0


def test_zero_budget_means_zero_not_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the budget to zero is the obvious way to halt spending, and it used not to
    work: `self.budget = budget or settings.ODDS_MONTHLY_CREDIT_BUDGET` evaluates `0 or X`
    to `X`, so an explicit zero restored the full default. A kill switch that re-enables
    spending is worse than none, because you would believe it worked.

    Also confirms no ZeroDivisionError — `max(self.budget, 1)` handles the denominator.
    """
    led = ledger_at(monkeypatch, 100, budget=0)
    assert led.budget == 0
    assert led.remaining_pct() == 0.0
    for tier in ("featured", "period", "props"):
        assert led.allows(tier) is False, tier


def test_omitting_the_budget_still_uses_the_configured_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the fix: `None` must still fall through to settings, or every
    caller that relies on the default would suddenly have a budget of zero."""
    from src.config import settings

    led = CreditLedger()
    assert led.budget == settings.ODDS_MONTHLY_CREDIT_BUDGET


@pytest.mark.parametrize(
    "used,featured,period,props",
    [
        (0,      True,  True,  True),    # 100% left — everything runs
        (13_000, True,  True,  True),    # 35% left
        (14_500, True,  True,  False),   # 27.5% — props drop first
        (17_500, True,  False, False),   # 12.5% — period markets drop
        (19_500, False, False, False),   # 2.5% — nothing runs
    ],
)
def test_degradation_ladder_drops_the_cheapest_signal_first(
    monkeypatch: pytest.MonkeyPatch, used: int, featured: bool, period: bool, props: bool
) -> None:
    """Props cost ~18,000 credits/month and are the first thing cut; featured markets on
    sharp books are the last, because they are the highest-value credits spent."""
    led = ledger_at(monkeypatch, used)
    assert led.allows("featured") is featured
    assert led.allows("period") is period
    assert led.allows("props") is props


def test_the_threshold_is_not_exactly_reachable_in_floating_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boundary you cannot actually land on.

    14,000 of 20,000 spent is "exactly 30% remaining" in decimal. In binary it isn't:
    `1 - 14000/20000` is 0.30000000000000004, so `remaining_pct()` returns
    30.000000000000004 and `pct > 30` is **True** — props stay enabled at the nominal
    cutoff.

    This is pinned rather than fixed. The drift is ~4e-15 of a percent, which is nothing
    against a credit budget, and the alternative — rounding or an epsilon — adds a second
    place for the threshold to be wrong. What matters is that nobody later reads
    `pct > 30` and concludes the ladder trips at exactly 30, because it doesn't, and a
    test asserting it did would fail confusingly.
    """
    at_nominal_thirty = ledger_at(monkeypatch, 14_000)
    pct = at_nominal_thirty.remaining_pct()

    assert pct != 30.0
    assert pct == pytest.approx(30.0, abs=1e-9)
    assert at_nominal_thirty.allows("props") is True

    # A hair further down and the ladder does trip.
    assert ledger_at(monkeypatch, 14_001).allows("props") is False


def test_ladder_comparisons_are_strict_not_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below each cutoff the tier is gone. Pinned because switching `>` to `>=` would
    quietly widen every tier by one step, and the symptom would be a credit overrun in
    the last week of the month rather than a failing test."""
    assert ledger_at(monkeypatch, 17_100).allows("period") is False   # 14.5% left
    assert ledger_at(monkeypatch, 16_900).allows("period") is True    # 15.5% left
    assert ledger_at(monkeypatch, 19_100).allows("featured") is False  # 4.5% left
    assert ledger_at(monkeypatch, 18_900).allows("featured") is True   # 5.5% left


def test_the_ladder_is_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spending more must never re-enable a tier. A non-monotonic guard would oscillate:
    drop props, spend less, re-enable, drop again — burning credits on the flapping."""
    seen: list[tuple[bool, bool, bool]] = []
    for used in range(0, 20_001, 500):
        led = ledger_at(monkeypatch, used)
        seen.append((led.allows("featured"), led.allows("period"), led.allows("props")))
    for earlier, later in zip(seen, seen[1:]):  # offset pairs — lengths differ by one
        assert all(l <= e for e, l in zip(earlier, later, strict=True)), (earlier, later)
