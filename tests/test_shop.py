"""Edge detection — the code that decides what gets published.

Every pick in production comes out of `find_opportunities()`. Nothing tested it.

That matters more than the line count suggests: a bug here does not raise, it *publishes*.
A fair price built against the wrong book set, or a price compared at the wrong line, or
an edge taken at a book you have no account with — each of those produces a plausible
number, a confident tier, and a bet you shouldn't have made.

`query` is patched with fixtures rather than a database. The logic under test is entirely
in Python; the SQL is a fetch, and mocking it keeps these fast enough to run on every
save.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.market import shop


# --- fixtures -----------------------------------------------------------------------


def row(game="G1", market="spreads", player=None, side="home", line=-3.0,
        book="dk", price=-110, **extra: Any) -> dict:
    return {"game_id": game, "market_type": market, "player_id": player,
            "side": side, "line": line, "book": book, "price": price, **extra}


def fair(game="G1", market="spreads", player=None, side="home", line=-3.0,
         fair_prob=0.5, sharp_prob=None, book_count=12, dispersion=0.4) -> dict:
    return {"game_id": game, "market_type": market, "player_id": player,
            "side": side, "line": line, "fair_prob": fair_prob,
            "sharp_prob": sharp_prob, "book_count": book_count,
            "dispersion": dispersion}


def patch_query(monkeypatch: pytest.MonkeyPatch, odds: list[dict], fairs: list[dict] | None = None):
    """Route each SQL string to the right fixture list.

    Crude, but the alternative is a real database, and the point of these tests is the
    branching logic rather than the persistence layer.
    """
    def fake(sql: str, params: Any = None) -> list[dict]:
        if "FROM fair_prices" in sql:
            return list(fairs or [])
        if "COUNT(DISTINCT book)" in sql:
            counts: dict[tuple, set] = {}
            for r in odds:
                counts.setdefault(
                    (r["game_id"], r["market_type"], r["player_id"], r["line"]), set()
                ).add(r["book"])
            return [{"game_id": g, "market_type": m, "player_id": p, "line": ln,
                     "n": len(books)} for (g, m, p, ln), books in counts.items()]
        if params and "ORDER BY price DESC" in sql:
            g, m, p, side, line = params
            sel = [r for r in odds if (r["game_id"], r["market_type"], r["player_id"],
                                       r["side"], r["line"]) == (g, m, p, side, line)]
            return sorted(sel, key=lambda r: -r["price"])
        return list(odds)

    monkeypatch.setattr(shop, "query", fake)


# --- best_prices --------------------------------------------------------------------


def test_best_price_is_the_largest_american_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """+150 beats +120 beats -105 beats -130. Getting this backwards would systematically
    pick the *worst* available price and report the edge of the best."""
    patch_query(monkeypatch, [
        row(book="a", price=-130), row(book="b", price=-105),
        row(book="c", price=120), row(book="d", price=150),
    ])
    best = shop.best_prices()
    assert best[("G1", "spreads", None, "home", -3.0)] == ("d", 150)


def test_negative_prices_compare_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """-105 pays better than -130, and is the larger number. A naive `abs()` or a
    string sort would invert this."""
    patch_query(monkeypatch, [row(book="a", price=-130), row(book="b", price=-105)])
    assert shop.best_prices()[("G1", "spreads", None, "home", -3.0)] == ("b", -105)


def test_each_line_is_its_own_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """-3.0 and -3.5 are different bets. Collapsing them would compare a price at one
    number against fair value at another — the most expensive possible error here."""
    patch_query(monkeypatch, [
        row(line=-3.0, book="a", price=-110), row(line=-3.5, book="b", price=105),
    ])
    best = shop.best_prices()
    assert len(best) == 2
    assert best[("G1", "spreads", None, "home", -3.0)] == ("a", -110)
    assert best[("G1", "spreads", None, "home", -3.5)] == ("b", 105)


def test_sides_are_not_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_query(monkeypatch, [
        row(side="home", price=-110), row(side="away", price=140),
    ])
    assert len(shop.best_prices()) == 2


def test_ties_keep_the_first_book_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documented rather than important: `>` not `>=`, so an equal price does not
    displace the incumbent. Pinned so a refactor doesn't make book selection
    order-dependent in a way nobody notices."""
    patch_query(monkeypatch, [row(book="first", price=-110), row(book="second", price=-110)])
    assert shop.best_prices()[("G1", "spreads", None, "home", -3.0)][0] == "first"


# --- main_lines ---------------------------------------------------------------------


def test_the_modal_line_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nine books at -3, three at -3.5. Only -3 is a real market; a consensus built from
    the three — one of which is the book you'd bet into — manufactures an edge."""
    odds = [row(line=-3.0, book=f"b{i}") for i in range(9)]
    odds += [row(line=-3.5, book=f"c{i}") for i in range(3)]
    patch_query(monkeypatch, odds)
    assert shop.main_lines() == {("G1", "spreads", None, 3.0)}


def test_spread_sides_collapse_to_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Home -3.5 and away +3.5 are the same market, so the key uses `abs(line)`.
    Without that, each side would count only its own books and both would look thin."""
    odds = [row(side="home", line=-3.5, book=f"h{i}") for i in range(5)]
    odds += [row(side="away", line=3.5, book=f"a{i}") for i in range(5)]
    patch_query(monkeypatch, odds)
    assert shop.main_lines() == {("G1", "spreads", None, 3.5)}


def test_abs_also_applies_to_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    """A consequence of keying on `abs(line)` rather than a spread-specific rule: totals
    are unaffected in practice because they're always positive, but the behaviour is
    global. Pinned so the blast radius of that choice is visible if totals ever carry a
    negative line."""
    patch_query(monkeypatch, [row(market="totals", side="over", line=44.5, book=f"b{i}")
                              for i in range(6)])
    assert shop.main_lines() == {("G1", "totals", None, 44.5)}


def test_markets_do_not_bleed_into_each_other(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_query(monkeypatch, [
        *[row(market="spreads", line=-3.0, book=f"s{i}") for i in range(6)],
        *[row(market="totals", side="over", line=44.5, book=f"t{i}") for i in range(4)],
    ])
    assert shop.main_lines() == {("G1", "spreads", None, 3.0), ("G1", "totals", None, 44.5)}


# --- find_opportunities -------------------------------------------------------------


def opportunities(monkeypatch, odds, fairs, **kw):
    patch_query(monkeypatch, odds, fairs)
    monkeypatch.setattr(shop.settings, "BETTABLE_BOOKS", ["dk", "fd"], raising=False)
    kw.setdefault("min_edge", 1.0)
    return shop.find_opportunities(**kw)


def test_a_thin_market_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """`min_books` exists because four books quoting a number is not a market. This is
    the guard that stops a stale minority line becoming a 20% edge."""
    odds = [row(book="dk", price=200)]
    fairs = [fair(fair_prob=0.5, book_count=3)]
    assert opportunities(monkeypatch, odds, fairs, min_books=8, main_line_only=False) == []


def test_a_real_edge_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fair 50%, priced at +150 — implied 40%. A genuine 25% edge."""
    odds = [row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.5, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False)
    assert len(found) == 1
    assert found[0]["best_price"] == 150
    assert found[0]["edge_pct"] > 0


def test_an_edge_below_the_floor_is_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MIN_EDGE_PCT` is a floor, not a suggestion — the tier rules subdivide above it."""
    odds = [row(book="dk", price=-110)]
    fairs = [fair(fair_prob=0.5, book_count=12)]
    assert opportunities(monkeypatch, odds, fairs, min_edge=5.0, main_line_only=False) == []


def test_the_sharp_anchor_is_preferred_over_consensus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consensus includes the soft book you're betting into, which drags fair value toward
    the mispricing and overstates the edge. Pricing against the sharp book is the whole
    reason `eu` is in the region list."""
    odds = [row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.50, sharp_prob=0.44, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False)
    assert found[0]["anchor"] == "sharp"
    assert found[0]["fair_prob"] == 0.44
    assert found[0]["consensus_prob"] == 0.50


def test_consensus_is_used_when_no_sharp_price_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    odds = [row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.5, sharp_prob=None, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False)
    assert found[0]["anchor"] == "consensus"


def test_the_sharp_anchor_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    odds = [row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.5, sharp_prob=0.44, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False, use_sharp_anchor=False)
    assert found[0]["anchor"] == "consensus"
    assert found[0]["fair_prob"] == 0.5


def test_an_edge_at_a_book_you_cannot_bet_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The best price is at an offshore book with no account. That is not an edge — but
    the next-best bettable price might still be one, so it falls back rather than
    discarding the market."""
    odds = [row(book="offshore", price=200), row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.5, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False)
    assert len(found) == 1
    assert found[0]["best_book"] == "dk"
    assert found[0]["best_price"] == 150


def test_no_bettable_book_means_no_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    odds = [row(book="offshore", price=200)]
    fairs = [fair(fair_prob=0.5, book_count=12)]
    assert opportunities(monkeypatch, odds, fairs, main_line_only=False) == []


def test_results_are_sorted_by_edge_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    """The slate is truncated downstream, so ordering decides which bets survive."""
    odds = [row(side="home", price=150, book="dk"), row(side="away", price=300, book="fd")]
    fairs = [fair(side="home", fair_prob=0.5, book_count=12),
             fair(side="away", fair_prob=0.5, book_count=12)]
    found = opportunities(monkeypatch, odds, fairs, main_line_only=False)
    assert [f["side"] for f in found] == ["away", "home"]
    assert found[0]["edge_pct"] > found[1]["edge_pct"]


def test_a_fair_price_with_no_quote_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fair price exists for -3.5 but nobody is quoting it any more. Publishing that
    would be a bet at a number you cannot get."""
    odds = [row(line=-3.0, book="dk", price=150)]
    fairs = [fair(line=-3.5, fair_prob=0.5, book_count=12)]
    assert opportunities(monkeypatch, odds, fairs, main_line_only=False) == []


def test_a_zero_anchor_is_skipped_rather_than_dividing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fair probability of zero would make any price look infinitely valuable."""
    odds = [row(book="dk", price=150)]
    fairs = [fair(fair_prob=0.0, sharp_prob=None, book_count=12)]
    assert opportunities(monkeypatch, odds, fairs, main_line_only=False) == []
