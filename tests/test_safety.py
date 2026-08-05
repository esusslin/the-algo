"""Safety guarantees that must never regress.

Each of these encodes a decision that was deliberate, not incidental. If one
starts failing, the correct response is almost always to fix the code rather
than relax the test.
"""
from __future__ import annotations

import pytest

from src.markets import describe_market, matches_filter
from src.picks.kelly import (MAX_SAFE_FRACTION, apply_correlation_haircut,
                             kelly_fraction, sized_stake)
from src.teams import BY_ABBR, TEAMS, resolve


# ---- Kelly ---------------------------------------------------------------
def test_no_bet_without_edge():
    assert kelly_fraction(0.50, -110) < 0
    assert sized_stake(0.50, -110) == 0.0


def test_kelly_zero_at_breakeven():
    from src.market.devig import breakeven_prob
    assert kelly_fraction(breakeven_prob(-110), -110) == pytest.approx(0, abs=1e-9)


def test_stake_capped_regardless_of_edge():
    """A 90% shot at +200 is a monstrous edge. The cap must still hold."""
    stake = sized_stake(0.90, 200, bankroll=1.0, fraction=0.25, max_pct=2.0)
    assert stake <= 0.02


def test_kelly_fraction_clamped_to_safe_maximum():
    """Config asking for full Kelly must be clamped. Full Kelly on
    miscalibrated probabilities is a reliable path to ruin."""
    reckless = sized_stake(0.60, -110, fraction=1.0, max_pct=100.0)
    safe = sized_stake(0.60, -110, fraction=MAX_SAFE_FRACTION, max_pct=100.0)
    assert reckless == pytest.approx(safe)


def test_correlation_haircut_caps_game_exposure():
    picks = [{"game_id": "G1", "stake": 0.02} for _ in range(10)]
    out = apply_correlation_haircut(picks, bankroll=1.0)
    assert sum(p["stake"] for p in out) <= 0.04 + 1e-9


def test_correlation_haircut_caps_slate_exposure():
    picks = [{"game_id": f"G{i}", "stake": 0.03} for i in range(20)]
    out = apply_correlation_haircut(picks, bankroll=1.0)
    assert sum(p["stake"] for p in out) <= 0.25 + 1e-9


def test_haircut_never_increases_a_stake():
    picks = [{"game_id": "G1", "stake": 0.001}, {"game_id": "G2", "stake": 0.001}]
    out = apply_correlation_haircut([dict(p) for p in picks], bankroll=1.0)
    for before, after in zip(picks, out):
        assert after["stake"] <= before["stake"] + 1e-12


# ---- tiering -------------------------------------------------------------
def test_props_need_more_edge_than_spreads():
    """A 3% edge on a WR4 prop is not the same asset as 3% on a main spread."""
    from src.picks.generator import thresholds_for
    for tier in ("A", "B", "C"):
        spread = thresholds_for("spreads", tier)["min_edge"]
        prop = thresholds_for("player_reception_yds", tier)["min_edge"]
        assert prop > spread


def test_a_tier_requires_sharp_anchor_on_game_markets():
    from src.picks.generator import thresholds_for
    assert thresholds_for("spreads", "A")["require_sharp"] is True


def test_tier_assignment_respects_book_count():
    from src.picks.generator import assign_tier
    thin = {"market_type": "spreads", "edge_pct": 9.0, "book_count": 3,
            "anchor": "sharp", "dispersion": 0.01}
    assert assign_tier(thin) is None      # huge edge, but not a real market


def test_tier_assignment_respects_dispersion():
    from src.picks.generator import assign_tier
    scattered = {"market_type": "spreads", "edge_pct": 9.0, "book_count": 20,
                 "anchor": "sharp", "dispersion": 0.9}
    assert assign_tier(scattered) is None


# ---- red team ------------------------------------------------------------
def test_redteam_fails_open_without_ai(monkeypatch):
    """A broken AI layer must never suppress the slate."""
    import src.ai.redteam as rt
    from src.ai.client import AIUnavailable

    monkeypatch.setattr(rt, "complete_json",
                        lambda *a, **k: (_ for _ in ()).throw(AIUnavailable("down")))
    r = rt.review_pick({"pick_id": 1, "game_id": "G", "market_type": "spreads",
                        "side": "home", "line": -3.5, "tier": "A",
                        "headline": "x"}, ctx={"matchup": "A at B"})
    assert r["verdict"] == "OK"


def test_redteam_cannot_promote(monkeypatch):
    """It may only downgrade. An enthusiastic verdict must change nothing."""
    import src.ai.redteam as rt
    monkeypatch.setattr(rt, "complete_json", lambda *a, **k: {
        "verdict": "STRONG_BET", "reason": "love it", "evidence": "vibes"})
    r = rt.review_pick({"pick_id": 1, "game_id": "G", "market_type": "spreads",
                        "side": "home", "line": -3.5, "tier": "C",
                        "headline": "x"}, ctx={"matchup": "A at B"})
    assert r["verdict"] == "OK"


def test_unevidenced_kill_is_demoted(monkeypatch):
    import src.ai.redteam as rt
    monkeypatch.setattr(rt, "complete_json", lambda *a, **k: {
        "verdict": "KILL", "reason": "feels wrong", "evidence": ""})
    r = rt.review_pick({"pick_id": 1, "game_id": "G", "market_type": "spreads",
                        "side": "home", "line": -3.5, "tier": "A",
                        "headline": "x"}, ctx={"matchup": "A at B"})
    assert r["verdict"] == "FLAG"


def test_missing_data_is_not_evidence(monkeypatch):
    """Empty context means we haven't collected it, not that something is
    wrong. Otherwise the agent guts the slate whenever ingestion lags."""
    import src.ai.redteam as rt
    monkeypatch.setattr(rt, "complete_json", lambda *a, **k: {
        "verdict": "KILL", "reason": "no weather",
        "evidence": "no forecast available for this game"})
    r = rt.review_pick({"pick_id": 1, "game_id": "G", "market_type": "totals",
                        "side": "over", "line": 44.5, "tier": "A",
                        "headline": "x"}, ctx={"matchup": "A at B"})
    assert r["verdict"] == "OK"


# ---- team crosswalk ------------------------------------------------------
def test_all_teams_resolve_from_full_name():
    for t in TEAMS:
        assert resolve(t.name) == t.abbr


def test_substring_matching_traps():
    """These are the ones naive substring matching gets wrong, and they cost
    us 157 of 272 event links before the explicit map existed."""
    for name, abbr in [("Kansas City Chiefs", "KC"), ("Green Bay Packers", "GB"),
                       ("San Francisco 49ers", "SF"), ("Tampa Bay Buccaneers", "TB"),
                       ("New Orleans Saints", "NO"), ("Los Angeles Rams", "LAR"),
                       ("Los Angeles Chargers", "LAC")]:
        assert resolve(name) == abbr


def test_historical_relocations():
    assert resolve("OAK") == "LV"
    assert resolve("SD") == "LAC"
    assert resolve("STL") == "LAR"
    assert resolve("WFT") == "WAS"


def test_unknown_team_returns_none():
    assert resolve("Toronto Argonauts") is None
    assert resolve("") is None


def test_thirty_two_teams():
    assert len(TEAMS) == 32
    assert len(BY_ABBR) == 32


# ---- market taxonomy -----------------------------------------------------
def test_market_classification():
    assert describe_market("spreads").bet_class == "game"
    assert describe_market("spreads_h1").bet_class == "period"
    assert describe_market("player_reception_yds").bet_class == "prop"
    assert describe_market("team_totals").bet_class == "team"


def test_unknown_market_degrades_gracefully():
    m = describe_market("some_future_market")
    assert m.label and m.bet_class


def test_filters_are_disjoint_where_they_should_be():
    assert matches_filter("player_reception_yds", "props")
    assert not matches_filter("player_reception_yds", "spread")
    assert matches_filter("spreads_h1", "first_half")
    assert not matches_filter("spreads", "first_half")
