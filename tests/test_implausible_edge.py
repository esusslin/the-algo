"""An edge that large is evidence of bad data, not an opportunity.

Found live on 18 September 2026. 23 of 79 published picks carried implausible
edges — one at 2,573%, another at 1,094%. The headline read:

    "Away h2h at +6000 (fanatics) — 1094.9% edge"

Nothing was corrupt. Tracing pick 535 to the raw quotes showed DraftKings really
did list `home -1.5 at +544`, paired with `away +1.5 at -830`. Those imply 15.5%
and 89.3%, summing to 1.048 — an ordinary 4.6% hold, so the implausible-hold
guard from 13 September passed it cleanly. Twenty other books priced the same
side between -110 and +105, so the weighted median sat at 47.2%.

`best_prices` takes MAX(price), so the outlier always wins the comparison, and
the arithmetic reports 0.4715 * 5.44 - 0.5285 = 203.6%.

**The pipeline was working exactly as designed.** Every step was individually
correct. The flaw was an assumption nobody had written down: that the best
available price is a real price. A book quoting 15% where the market says 47% is
not offering money, it is quoting a different event — a stale alternate line, a
mislabelled market, a typo. You will never be filled at +544 on a -1.5 spread.

This is the third variant of one bug. 13 September: both sides absurd, caught by
hold. Also 13 September: a quote that made the consensus wrong. Today: a single
quote that is plausible in isolation, plausible as a pair, and impossible only
when compared to what everyone else thinks.
"""

from __future__ import annotations

import pytest

from src.market.devig import edge_pct
from src.market.shop import MAX_PLAUSIBLE_EDGE_PCT


# ---------------------------------------------------------------------------
# the live case, reproduced exactly
# ---------------------------------------------------------------------------
def test_the_draftkings_quote_that_started_this() -> None:
    """Real numbers from production, pick 535."""
    consensus_fair = 0.4715          # weighted median across ~20 books
    dk_price = 544                   # DraftKings on home -1.5
    edge = edge_pct(consensus_fair, dk_price)
    assert edge == pytest.approx(203.6, abs=1.0)
    assert edge > MAX_PLAUSIBLE_EDGE_PCT


def test_the_pair_looks_completely_normal() -> None:
    """**Why the previous guard missed it.** The two sides sum to a textbook
    4.6% hold. Nothing about the quote is suspicious until you compare it with
    other books — which is exactly what the earlier guard could not do, because
    it only ever looked at one book's pair in isolation."""
    from src.market.consensus import MAX_CREDIBLE_HOLD_PCT
    from src.market.devig import american_to_prob, hold_pct

    raw = [american_to_prob(544), american_to_prob(-830)]
    assert hold_pct(raw) == pytest.approx(4.6, abs=0.5)
    assert hold_pct(raw) < MAX_CREDIBLE_HOLD_PCT, "the hold guard should NOT catch this"


@pytest.mark.parametrize("fair,price,label", [
    (0.4715, 544, "the -1.5 spread at +544"),
    (0.5156, 9000, "a coin flip at +9000"),
    (0.6156, 6000, "a favourite at +6000"),
    (0.4473, 9000, "the 2573% pick"),
    (0.55, 1580, "over 40.5 at +1580"),
])
def test_every_live_offender_is_rejected(fair, price, label) -> None:
    assert edge_pct(fair, price) > MAX_PLAUSIBLE_EDGE_PCT, label


# ---------------------------------------------------------------------------
# THE CONTROLS — a guard that eats real edges is worse than no guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fair,price,label", [
    (0.5330, 100, "the CIN pick: 6.5% edge at even money"),
    (0.6400, -154, "the LV pick: 5.6% edge"),
    (0.3990, 164, "the DET pick: 5.4% edge"),
    (0.5200, -110, "a thin 1% edge"),
    (0.5500, 110, "a fat but real 15% edge"),
    (0.3000, 300, "a longshot priced generously"),
    (0.1800, 500, "a genuine longshot, +500 on an 18% shot"),
    (0.8000, -350, "a heavy favourite with a real edge"),
])
def test_real_edges_survive(fair, price, label) -> None:
    """**The point of the whole file.** The failure mode of a guard like this is
    silently shrinking the board — it would look identical to a quiet week, and
    nothing would raise. These are the shapes a careless threshold would eat."""
    e = edge_pct(fair, price)
    assert e <= MAX_PLAUSIBLE_EDGE_PCT, (
        f"{label} rejected as implausible at {e:.1f}% — the guard is too tight")


def test_the_threshold_sits_well_clear_of_reality() -> None:
    """Stated as a property. Real soft-book NFL edges top out around 10-15%; the
    threshold must leave room above that or it will bite in a live spot."""
    assert MAX_PLAUSIBLE_EDGE_PCT >= 20.0
    assert MAX_PLAUSIBLE_EDGE_PCT <= 50.0, "so loose it would pass the +544 case"


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
@pytest.fixture
def shop_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MIN_EDGE_PCT", "2.0")
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()
    return db_mod


# Twelve books agreeing that home -1.5 is roughly a coin flip.
_HONEST = [(f"book{i}", s, p) for i in range(12)
           for s, p in (("home", -110), ("away", -110))]


def _seed(db_mod, quotes) -> None:
    now = db_mod.utcnow()
    with db_mod.db() as c:
        c.execute("DELETE FROM odds_current")
        for book, side, price in quotes:
            line = -1.5 if side == "home" else 1.5
            c.execute(
                "INSERT INTO odds_current (game_id,market_type,player_id,side,line,"
                "book,price,updated_at,fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("g1", "spreads", "", side, line, book, price, now, now))


def _opps(min_books=4):
    from src.market.consensus import build_fair_prices
    from src.market.shop import find_opportunities
    build_fair_prices()
    return find_opportunities(min_edge=-100.0, min_books=min_books,
                              bettable_only=False)


def test_the_outlier_produces_no_opportunity(shop_db) -> None:
    """**The regression, end to end.** Twelve honest books plus DraftKings at
    +544. Before the fix this yielded a ~200% edge opportunity."""
    _seed(shop_db, _HONEST + [("draftkings", "home", 544),
                              ("draftkings", "away", -830)])
    homes = [o for o in _opps() if o["side"] == "home"]
    assert not any(o["edge_pct"] > MAX_PLAUSIBLE_EDGE_PCT for o in homes), (
        f"implausible edge survived: {[round(o['edge_pct'], 1) for o in homes]}")
    assert not any(o["best_price"] == 544 for o in homes), "the +544 became a pick"


def test_an_honest_soft_book_still_produces_one(shop_db) -> None:
    """**The control.** Same shape, but the odd book out is merely generous
    rather than impossible. This MUST still be found, or the guard has quietly
    turned the product off."""
    _seed(shop_db, _HONEST + [("betmgm", "home", 105), ("betmgm", "away", -125)])
    homes = [o for o in _opps() if o["side"] == "home"]
    assert homes, "no opportunity found at all"
    best = max(homes, key=lambda o: o["edge_pct"])
    assert best["best_price"] == 105
    assert 0 < best["edge_pct"] <= MAX_PLAUSIBLE_EDGE_PCT


def test_a_clean_market_yields_nothing_implausible(shop_db) -> None:
    """Baseline: twelve books agreeing, no outlier, nothing extreme."""
    _seed(shop_db, _HONEST)
    assert all(o["edge_pct"] <= MAX_PLAUSIBLE_EDGE_PCT for o in _opps())
