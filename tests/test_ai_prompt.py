"""What the red-team agent is actually shown.

The agent's judgement was measured by `scripts/eval_redteam.py` long before this
file existed, and the measurement said it never reasoned about line movement.
The cause was not the model. The section was rendered as::

    - spreads home 0.0 -> -110 (pinnacle)

with the arrow between *line and price* — two unrelated quantities — no
timestamp, and the rows newest-first. It reads like the line moved to -110. The
agent was answering the only question the prompt actually posed.

That is the general lesson worth keeping: a prompt is an interface, and an
unreadable one produces a confident, plausible answer to the wrong question. No
amount of instruction-tuning in the system message fixes a garbled data section,
and nothing downstream can tell the difference.

These tests cover the rendering and the arithmetic. They are pure — no API key,
no database, no cost — so they belong in CI, unlike the golden set.
"""

from __future__ import annotations

import pytest

from src.ai.redteam import _fmt_price, _line_moves_text, _net_move, _prompt


def move(line, *, side="home", market="spreads", at="2026-10-09T12:00:00",
         price=-110, book="pinnacle") -> dict:
    return {"market_type": market, "side": side, "line": line, "price": price,
            "book": book, "observed_at": at}


def pick(**kw) -> dict:
    base = {"pick_id": 1, "game_id": "G1", "market_type": "spreads", "side": "home",
            "line": -3.0, "tier": "A", "headline": "Home -3.0"}
    return {**base, **kw}


# --- price formatting: never raise --------------------------------------------------
#
# `_prompt` is evaluated as an argument to `complete_json`, outside the
# `except AIUnavailable` guard in `review_pick`. Anything that raises here
# propagates out of a function whose docstring promises it never does — and the
# fail-open design means the caller has no handler for it either.


def test_an_integer_price_formats_with_a_sign() -> None:
    assert _fmt_price(-110) == "-110"
    assert _fmt_price(150) == "+150"


def test_a_float_price_does_not_raise() -> None:
    """A REAL column round-trips as float. `:+d` rejects it outright."""
    assert _fmt_price(-110.0) == "-110"


def test_a_missing_price_does_not_raise() -> None:
    """A quote recorded before its price. Renders as n/a rather than crashing
    the whole review of an otherwise fine pick."""
    assert _fmt_price(None) == "n/a"
    assert _fmt_price("") == "n/a"


def test_prompt_survives_a_malformed_row() -> None:
    """The end-to-end version of the above: one bad row must not take down the
    review, because the alternative is an exception where the caller expects a
    verdict."""
    ctx = {"matchup": "SF at SEA", "recent_line_moves": [
        move(-3.0, price=None), move(-2.5, price=-110.0)]}
    assert "SF at SEA" in _prompt(pick(), ctx)


# --- chronology ---------------------------------------------------------------------


def test_moves_are_rendered_oldest_first() -> None:
    """The query returns newest-first so the LIMIT keeps recent rows. Rendering
    them in that order asks the reader to run the film backwards."""
    text = _line_moves_text(pick(), [
        move(0.0, at="2026-10-11T12:00:00"),
        move(-3.0, at="2026-10-09T12:00:00"),
    ])
    assert text.index("2026-10-09") < text.index("2026-10-11")


def test_each_row_carries_its_timestamp() -> None:
    """Without one, a time series is just a list of numbers."""
    assert "2026-10-09T12:00:00" in _line_moves_text(pick(), [move(-3.0)])


def test_line_and_price_are_not_joined_by_an_arrow() -> None:
    """**The original bug, pinned.** `-3.0 -> -110` invites reading a spread
    moving to a price. Whatever the format becomes, it must not put an arrow
    between those two fields."""
    text = _line_moves_text(pick(), [move(-3.0, price=-110)])
    assert "-3.0 -> -110" not in text
    assert "line -3.0" in text


def test_no_moves_reads_as_absence_not_as_a_finding() -> None:
    """"none recorded" is on the absence-word list in `review_pick`, so a model
    citing it as evidence gets overridden to OK. That link is deliberate."""
    assert _line_moves_text(pick(), []) == "- none recorded"


# --- the sign convention ------------------------------------------------------------
#
# One inversion, at `under`. Getting it backwards would tell the agent a move in
# our favour is a reason to object, and the objection would be evidenced,
# grounded, and completely wrong.


def test_a_spread_drifting_away_from_our_side_is_against_us() -> None:
    """Bet home -3.0; market now says pick'em. We are laying three points the
    market no longer thinks are there."""
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(0.0, at="2026-10-11T12:00:00")])
    assert "against" in out
    assert "3.0 points" in out


def test_a_spread_moving_our_way_is_not_a_warning() -> None:
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(-6.0, at="2026-10-11T12:00:00")])
    assert "toward" in out
    assert "against" not in out


def test_a_rising_total_is_against_the_over() -> None:
    p = pick(market_type="totals", side="over", line=44.5)
    out = _net_move(p, [move(44.5, market="totals", side="over", at="2026-10-09T12:00:00"),
                        move(47.5, market="totals", side="over", at="2026-10-11T12:00:00")])
    assert "against" in out


def test_a_rising_total_is_help_for_the_under() -> None:
    """**The inversion.** Identical numbers to the case above, opposite reading,
    because a higher total is more room for an under to land."""
    p = pick(market_type="totals", side="under", line=44.5)
    out = _net_move(p, [move(44.5, market="totals", side="under", at="2026-10-09T12:00:00"),
                        move(47.5, market="totals", side="under", at="2026-10-11T12:00:00")])
    assert "toward" in out
    assert "against" not in out


def test_a_falling_total_is_against_the_under() -> None:
    p = pick(market_type="totals", side="under", line=47.5)
    out = _net_move(p, [move(47.5, market="totals", side="under", at="2026-10-09T12:00:00"),
                        move(44.5, market="totals", side="under", at="2026-10-11T12:00:00")])
    assert "against" in out


# --- magnitude ----------------------------------------------------------------------


def test_a_half_point_tick_is_labelled_minor() -> None:
    """Spreads move half a point constantly. An agent objecting to that would
    downgrade most of a slate, which is the failure this whole module is built
    to avoid."""
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(-2.5, at="2026-10-11T12:00:00")])
    assert "minor" in out


def test_three_points_on_a_spread_is_significant() -> None:
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(0.0, at="2026-10-11T12:00:00")])
    assert "significant" in out


def test_a_total_needs_more_movement_than_a_spread_to_count() -> None:
    """Totals are quoted on a wider scale; a point of movement means less."""
    p = pick(market_type="totals", side="over", line=44.5)
    out = _net_move(p, [move(44.5, market="totals", side="over", at="2026-10-09T12:00:00"),
                        move(45.5, market="totals", side="over", at="2026-10-11T12:00:00")])
    assert "minor" in out


def test_an_unchanged_line_says_so_explicitly() -> None:
    """Silence would leave the model to infer stability from two identical rows,
    which is exactly the arithmetic we are trying to take away from it."""
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(-3.0, at="2026-10-11T12:00:00")])
    assert "none" in out and "against" not in out


# --- what counts as a series --------------------------------------------------------


def test_a_single_quote_is_not_a_move() -> None:
    """One observation is a price, not a trend."""
    assert _net_move(pick(), [move(-3.0)]) == ""


def test_the_other_side_of_the_same_market_is_ignored() -> None:
    """The away spread is the mirror of the home spread. Mixing them would read
    as a six-point move on a line that never moved."""
    assert _net_move(pick(), [move(-3.0, side="home"), move(3.0, side="away")]) == ""


def test_a_different_market_is_ignored() -> None:
    """Total movement says nothing about a spread bet."""
    assert _net_move(pick(), [move(44.5, market="totals", side="over"),
                              move(47.5, market="totals", side="over")]) == ""


def test_books_are_not_mixed_into_one_series() -> None:
    """**Two books disagreeing is not one book moving.**

    Pinnacle steady at -3.0 while a soft book sits at -6.0 is a shopping
    opportunity, not a warning. Pooling them reports a three-point move that
    never happened, and the agent would kill a pick over an artefact of the
    query's ORDER BY.
    """
    out = _net_move(pick(), [
        move(-3.0, book="pinnacle", at="2026-10-09T12:00:00"),
        move(-6.0, book="softbook", at="2026-10-10T12:00:00"),
        move(-3.0, book="pinnacle", at="2026-10-11T12:00:00"),
    ])
    assert "none" in out
    assert "pinnacle" in out


def test_a_null_line_is_skipped_rather_than_coerced() -> None:
    """Moneyline rows carry no line. Treating None as zero would invent a move."""
    assert _net_move(pick(), [move(None), move(-3.0)]) == ""


def test_the_summary_names_the_book_it_measured() -> None:
    """So a human checking the objection knows which series to go and look at."""
    out = _net_move(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                             move(0.0, at="2026-10-11T12:00:00")])
    assert "pinnacle" in out


# --- grounding ----------------------------------------------------------------------


def test_movement_evidence_survives_the_grounding_check() -> None:
    """`review_pick` demotes any FLAG whose evidence shares no vocabulary with
    the context. A model citing "moved 3.0 points against this bet" shares no
    four-character token with a dict repr of floats — so grounding against the
    raw rows would silently undo the finding. It grounds against the rendered
    section instead, and this is why.
    """
    from src.ai.redteam import _evidence_grounded

    blob = _line_moves_text(pick(), [move(-3.0, at="2026-10-09T12:00:00"),
                                     move(0.0, at="2026-10-11T12:00:00")])
    assert _evidence_grounded("pinnacle moved 3.0 points against this bet", blob)


# --- the movement-only cap ----------------------------------------------------------
#
# KILL unpublishes a pick. FLAG caps tier A at B. That difference should turn on
# whether a stated fact invalidates the bet, and market disagreement is not one:
# it is a reason to size down, not to stand aside.
#
# This lives in code rather than in the system prompt because the prompt was not
# enough. Asked for FLAG, the model returned KILL on six of six samples across two
# runs. A safety property that depends on instruction-following is not a safety
# property — the model is free to disagree with a prompt and cannot disagree with
# an `if`.


def _ctx_with_moves(moves, **kw):
    base = {"matchup": "SF at SEA", "season": 2026, "week": 5, "season_type": "REG",
            "injuries": [], "inactives": [], "weather": None,
            "home_rest_days": 7, "away_rest_days": 7, "recent_line_moves": moves}
    return {**base, **kw}


SHARP_MOVE = [move(-3.0, at="2026-10-09T12:00:00"), move(0.0, at="2026-10-11T12:00:00")]


def _review(monkeypatch, verdict, evidence, ctx, reason="r"):
    import src.ai.redteam as rt
    monkeypatch.setattr(rt, "complete_json", lambda *a, **k: {
        "verdict": verdict, "reason": reason, "evidence": evidence})
    monkeypatch.setattr(rt.settings, "ENABLE_AI_REDTEAM", True, raising=False)
    return rt.review_pick(pick(), ctx)


def test_a_kill_resting_only_on_line_movement_is_capped_at_flag(monkeypatch) -> None:
    """The decision this whole block exists for. The pick still publishes, at a
    smaller size, and the objection is preserved for whoever reads it."""
    r = _review(monkeypatch, "KILL", "3.0 points against the side of this bet (significant)",
                _ctx_with_moves(SHARP_MOVE))
    assert r["verdict"] == "FLAG"
    assert r["reason"]           # not blanked — the finding survives the cap


def test_a_kill_on_a_player_is_not_capped(monkeypatch) -> None:
    """**The asymmetry.** A quarterback ruled out means the thing you priced no
    longer exists. That is a disqualifying fact and KILL is correct."""
    ctx = _ctx_with_moves(SHARP_MOVE, injuries=[
        {"player_name": "Starting Quarterback", "team": "SEA",
         "game_status": "Out", "practice_status": "DNP", "body_part": "Shoulder"}])
    r = _review(monkeypatch, "KILL", "SEA Starting Quarterback: Out (DNP, Shoulder)", ctx)
    assert r["verdict"] == "KILL"


def test_a_kill_citing_both_keeps_its_verdict(monkeypatch) -> None:
    """When a player fact and a line move are cited together, the player fact is
    doing the work. Capping here would let a coincidental line move launder a
    genuine disqualifier into a tier downgrade."""
    ctx = _ctx_with_moves(SHARP_MOVE, injuries=[
        {"player_name": "Starting Quarterback", "team": "SEA",
         "game_status": "Out", "practice_status": "DNP", "body_part": "Shoulder"}])
    r = _review(monkeypatch, "KILL",
                "SEA Starting Quarterback Out; 3.0 points against the side of this bet", ctx)
    assert r["verdict"] == "KILL"


def test_a_flag_on_line_movement_is_left_alone(monkeypatch) -> None:
    """The cap only ever lowers KILL. It must not touch a verdict that was
    already where we want it."""
    r = _review(monkeypatch, "FLAG", "3.0 points against the side of this bet (significant)",
                _ctx_with_moves(SHARP_MOVE))
    assert r["verdict"] == "FLAG"


def test_the_cap_never_raises_a_verdict(monkeypatch) -> None:
    """Downgrade-only is the property the entire agent rests on. A cap that could
    promote would break it."""
    r = _review(monkeypatch, "OK", "", _ctx_with_moves(SHARP_MOVE))
    assert r["verdict"] == "OK"


def test_movement_only_needs_evidence_to_fire() -> None:
    """An empty string grounds in nothing, so it is not 'movement-only'. Without
    this, an unevidenced KILL would take the movement path instead of the
    unevidenced-KILL path and the two demotions would fight."""
    from src.ai.redteam import _movement_only

    assert _movement_only("", "situational", "movement") is False


@pytest.mark.parametrize("side", ["home", "away", "over", "under"])
def test_every_side_renders_without_raising(side: str) -> None:
    market = "totals" if side in ("over", "under") else "spreads"
    p = pick(market_type=market, side=side)
    assert isinstance(_net_move(p, [move(1.0, market=market, side=side,
                                         at="2026-10-09T12:00:00"),
                                    move(2.0, market=market, side=side,
                                         at="2026-10-11T12:00:00")]), str)
