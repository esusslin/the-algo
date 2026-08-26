"""Player identity — quarantine, don't guess.

Cross-source name matching is the number-one source of bugs in this kind of system, and
the failure is asymmetric in a way that decides the whole design:

- A **missing** row shows up in the stats. `refresh_injuries` reports
  `unresolved=67 ambiguous=18` and someone can go look.
- A **wrong** row silently attaches one player's injury to another's props, and every
  projection built on it is confidently wrong for as long as nobody notices.

So the rule is that an ambiguous name keeps its source id and gets counted, rather than
being matched to whichever candidate happens to sort first. These tests pin that rule and
the normalisation that feeds it.

`norm_name` is pure. `resolve_player_ids` needs the players table, which is patched.
"""

from __future__ import annotations

import pytest

from src.fetchers import injuries as inj
from src.fetchers.injuries import norm_name, resolve_player_ids


# --- normalisation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("D.K. Metcalf", "DK Metcalf"),
        ("D.K. Metcalf", "D K Metcalf"),
        ("Odell Beckham Jr.", "Odell Beckham"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
        ("Ja'Marr Chase", "JaMarr Chase"),
        ("Ja’Marr Chase", "Ja'Marr Chase"),          # curly vs straight apostrophe
        ("Michael Penix Jr.", "Michael Penix"),
        ("A.J. Terrell Jr.", "AJ Terrell"),
        ("  Drake   London  ", "Drake London"),
        ("BRIAN ROBINSON", "brian robinson"),
    ],
)
def test_the_same_player_written_two_ways_normalises_alike(a: str, b: str) -> None:
    """Every one of these disagreements occurs between real sources. Each unmatched pair
    is a player whose injury never reaches his projection."""
    assert norm_name(a) == norm_name(b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("Drake London", "LaCale London"),
        ("Mike Williams", "Mike Evans"),
        ("Brian Robinson", "Bryan Robinson"),
    ],
)
def test_genuinely_different_names_do_not_collapse(a: str, b: str) -> None:
    """Normalisation must not merge people who share a surname or a near-spelling."""
    assert norm_name(a) != norm_name(b)


def test_suffix_stripping_deliberately_creates_collisions() -> None:
    """**This looks like a bug and is the design.**

    "Josh Allen" and "Josh Allen Jr." normalise to the same key — by the same rule that
    makes "Odell Beckham Jr." match "Odell Beckham", which is the match we need, because
    sources disagree about whether to print the suffix.

    The collision is not resolved here. `_name_index` keeps *every* candidate for a key,
    and `resolve_player_ids` disambiguates on team or quarantines. Trying to be clever in
    the normaliser instead would trade a reported ambiguity for a silent wrong match.
    """
    assert norm_name("Josh Allen") == norm_name("Josh Allen Jr.")
    assert norm_name("Marvin Harrison") == norm_name("Marvin Harrison Jr.")


def test_accents_are_not_stripped_today() -> None:
    """A known limitation, pinned rather than asserted as correct.

    A source writing "Lindström" will not match one writing "Lindstrom", and that player
    lands in `unresolved`. Visible in the stats rather than silently mismatched, so it
    fails the safe way — but if this is ever fixed, update this test rather than deleting
    it, and re-check the collision cases above.
    """
    assert norm_name("Chris Lindstrom") != norm_name("Chris Lindström")


def test_suffixes_are_stripped_repeatedly() -> None:
    assert norm_name("John Smith Jr. III") == norm_name("John Smith")


def test_a_lone_initial_merges_only_at_the_front() -> None:
    """"D.K. Metcalf" collapses to "dk metcalf". A trailing initial must not merge into
    the surname — "Robert Griffin III" is not "Robert Griffiniii"."""
    assert norm_name("D.K. Metcalf") == "dk metcalf"
    assert norm_name("Robert Griffin III") == "robert griffin"


def test_normalisation_is_idempotent() -> None:
    """Running it twice must not change the answer, or a key computed at write time
    would differ from one computed at read time."""
    for name in ("D.K. Metcalf", "Amon-Ra St. Brown", "Odell Beckham Jr."):
        once = norm_name(name)
        assert norm_name(once) == once


def test_an_empty_name_does_not_raise() -> None:
    assert norm_name("") == ""
    assert norm_name("   ") == ""


# --- resolution ---------------------------------------------------------------------


class Rec:
    """Minimal stand-in for InjuryRecord — the resolver touches four fields."""

    def __init__(self, player_name: str, team: str | None = None,
                 player_id: str | None = None) -> None:
        self.player_name = player_name
        self.team = team
        self.player_id = player_id


def patch_players(monkeypatch: pytest.MonkeyPatch, players: list[tuple[str, str, str | None]]):
    """players: (player_id, full_name, team)."""
    def fake(sql: str, params=None):
        if "player_aliases" in sql:
            return []
        return [{"player_id": p, "full_name": n, "team": t} for p, n, t in players]
    monkeypatch.setattr(inj, "query", fake)


def test_a_native_gsis_id_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already canonical — don't re-resolve it and risk overwriting a correct id."""
    patch_players(monkeypatch, [("00-9999", "Someone Else", "ATL")])
    recs = [Rec("Drake London", "ATL", player_id="00-0036322")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id == "00-0036322"
    assert stats["native_gsis"] == 1


def test_a_sleeper_id_is_treated_as_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sleeper:1234` is a source-native id, not a canonical one. Treating it as resolved
    would leave the row unjoinable to everything else."""
    patch_players(monkeypatch, [("00-0036322", "Drake London", "ATL")])
    recs = [Rec("Drake London", "ATL", player_id="sleeper:1234")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id == "00-0036322"
    assert stats["by_name"] == 1


def test_a_unique_name_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_players(monkeypatch, [("00-0036322", "Drake London", "ATL")])
    recs = [Rec("Drake London", "ATL")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id == "00-0036322"
    assert stats["by_name"] == 1


def test_an_unknown_name_is_reported_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_players(monkeypatch, [("00-0036322", "Drake London", "ATL")])
    recs = [Rec("Nobody Atall", "ATL")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id is None
    assert stats["unresolved"] == 1
    assert "Nobody Atall" in stats["unresolved_sample"]


def test_two_players_with_one_name_are_split_by_team(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marvin Harrison Sr. and Jr. normalise identically. Team disambiguates."""
    patch_players(monkeypatch, [
        ("00-0000001", "Marvin Harrison", "IND"),
        ("00-0000002", "Marvin Harrison Jr.", "ARI"),
    ])
    recs = [Rec("Marvin Harrison", "ARI")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id == "00-0000002"
    assert stats["by_name_team"] == 1


def test_an_ambiguous_name_with_no_team_is_quarantined(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The rule this module exists for.** Two candidates, no team to split them — keep
    the source id and count it. Guessing would attach one player's injury to another's
    projections, and nothing downstream could detect it."""
    patch_players(monkeypatch, [
        ("00-0000001", "Marvin Harrison", "IND"),
        ("00-0000002", "Marvin Harrison Jr.", "ARI"),
    ])
    recs = [Rec("Marvin Harrison", None)]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id is None
    assert stats["ambiguous"] == 1
    assert stats["by_name"] == 0


def test_an_ambiguous_name_on_a_team_matching_both_is_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Team doesn't help if both candidates are on it. Still quarantined."""
    patch_players(monkeypatch, [
        ("00-0000001", "John Smith", "ATL"),
        ("00-0000002", "John Smith", "ATL"),
    ])
    recs = [Rec("John Smith", "ATL")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id is None
    assert stats["ambiguous"] == 1


def test_a_team_that_matches_no_candidate_quarantines(monkeypatch: pytest.MonkeyPatch) -> None:
    """A traded player whose team is stale in one source. Better unresolved than
    attached to his namesake."""
    patch_players(monkeypatch, [
        ("00-0000001", "John Smith", "IND"),
        ("00-0000002", "John Smith", "ARI"),
    ])
    recs = [Rec("John Smith", "SEA")]
    _, stats = resolve_player_ids(recs)
    assert recs[0].player_id is None
    assert stats["ambiguous"] == 1


def test_the_counts_account_for_every_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/health` reports these numbers and they're how you notice resolution degrading.
    If they didn't sum to the total, a silently dropped record would be invisible."""
    patch_players(monkeypatch, [
        ("00-0000001", "Drake London", "ATL"),
        ("00-0000002", "John Smith", "IND"),
        ("00-0000003", "John Smith", "ARI"),
    ])
    recs = [
        Rec("Drake London", "ATL"),                       # by_name
        Rec("John Smith", "ARI"),                          # by_name_team
        Rec("John Smith", None),                           # ambiguous
        Rec("Nobody Atall", "SEA"),                        # unresolved
        Rec("Already Canonical", "SEA", player_id="00-1"),  # native
    ]
    _, stats = resolve_player_ids(recs)
    counted = (stats["native_gsis"] + stats["by_name"] + stats["by_name_team"]
               + stats["ambiguous"] + stats["unresolved"])
    assert counted == stats["total"] == len(recs)
