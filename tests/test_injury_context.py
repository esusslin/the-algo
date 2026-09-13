"""The injury list the red team is shown must not contradict itself.

Live on 13 September 2026. Two picks were FLAGged citing "Both Atlanta
quarterbacks (Tagovailoa and Penix) are listed as questionable". The agent had
reported its context faithfully — the context was the problem.

`injuries` has no primary key and no unique constraint, so every fetch appends.
Read without deduplication it returned the same player many times, with
CONTRADICTORY statuses: one quarterback appeared as both "Questionable" and
"Out" in a single list. The 25-row limit was then consumed by duplicates of two
players while the rest of both teams went unmentioned.

Two consequences, and the second is worse:

  * the agent writes whichever line it read last into `evidence`, and `evidence`
    is the entire reason a KILL is trustworthy;
  * the players it never saw cannot be reasoned about at all, so the check that
    is supposed to catch a ruled-out starter silently stops looking.

`knowledge_time` exists so the latest thing we learned wins. Nothing was using
it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ctx_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    import importlib

    from src import config as config_mod
    importlib.reload(config_mod)
    from src import db as db_mod
    importlib.reload(db_mod)
    db_mod.run_migrations()

    with db_mod.db() as c:
        c.execute("INSERT INTO games (game_id, season, week, season_type, "
                  "home_team, away_team, kickoff_utc) "
                  "VALUES ('g1', 2026, 2, 'REG', 'ATL', 'MIN', "
                  "'2099-01-01T00:00:00+00:00')")
    return db_mod


def _injury(db_mod, name, status, *, kt, team="ATL", practice=None,
            player_id=None) -> None:
    with db_mod.db() as c:
        c.execute(
            "INSERT INTO injuries (player_id, player_name, team, season, week, "
            "game_status, practice_status, knowledge_time) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (player_id, name, team, 2026, 2, status, practice, kt))


def test_a_player_appears_once_with_the_latest_status(ctx_db) -> None:
    """**The regression.** Same player, three reports through the week. The
    Friday designation is the true one; the agent must see only that."""
    from src.ai.redteam import _game_context

    _injury(ctx_db, "Michael Penix", "Questionable", kt="2026-09-10T12:00:00+00:00")
    _injury(ctx_db, "Michael Penix", "Questionable", kt="2026-09-11T12:00:00+00:00")
    _injury(ctx_db, "Michael Penix", "Out", kt="2026-09-12T12:00:00+00:00")

    injuries = _game_context("g1")["injuries"]
    penix = [i for i in injuries if i["player_name"] == "Michael Penix"]
    assert len(penix) == 1, f"{len(penix)} rows for one player: {penix}"
    assert penix[0]["game_status"] == "Out", "an older report won"


def test_duplicates_do_not_crowd_out_other_players(ctx_db) -> None:
    """The quieter half, and the dangerous one. Thirty duplicate rows for one
    player filled the 25-row limit, so a genuinely ruled-out starter was never
    shown to the agent at all — the check stops looking and says nothing."""
    from src.ai.redteam import _game_context

    # Same status as the player being crowded out, and inserted first — because
    # the ORDER BY sorts by severity, so duplicates of a LESS severe player get
    # pushed below the limit anyway and would make this test pass for the wrong
    # reason. The real hazard is a player duplicated at the same severity.
    for i in range(30):
        _injury(ctx_db, "Michael Penix", "Out",
                kt=f"2026-09-10T{i % 24:02d}:00:00+00:00")
    _injury(ctx_db, "Bijan Robinson", "Out", kt="2026-09-12T12:00:00+00:00")

    names = [i["player_name"] for i in _game_context("g1")["injuries"]]
    assert "Bijan Robinson" in names, (
        f"the ruled-out starter was crowded out by duplicates: {names[:5]}")


def test_the_agent_is_never_shown_a_contradiction(ctx_db) -> None:
    """Stated as the property rather than the mechanism: no player may appear
    twice with different statuses, however the dedup is implemented."""
    from src.ai.redteam import _game_context, _prompt

    _injury(ctx_db, "Michael Penix", "Questionable", kt="2026-09-10T12:00:00+00:00")
    _injury(ctx_db, "Michael Penix", "Out", kt="2026-09-12T12:00:00+00:00")
    _injury(ctx_db, "Tua Tagovailoa", "Questionable", kt="2026-09-12T12:00:00+00:00")

    ctx = _game_context("g1")
    seen = [i["player_name"] for i in ctx["injuries"]]
    assert len(seen) == len(set(seen)), f"duplicate players in context: {seen}"

    text = _prompt({"market_type": "h2h", "side": "home", "line": 0.0,
                    "tier": "B", "headline": "ATL ML"}, ctx)
    assert text.count("Michael Penix") == 1
    assert "Questionable" not in text.split("Michael Penix")[1][:40]


def test_distinct_players_are_all_kept(ctx_db) -> None:
    """The control. Dedup that collapsed different players would pass every
    test above while hiding most of the injury report."""
    from src.ai.redteam import _game_context

    for n, s in [("A Player", "Out"), ("B Player", "Doubtful"),
                 ("C Player", "Questionable"), ("D Player", "Questionable")]:
        _injury(ctx_db, n, s, kt="2026-09-12T12:00:00+00:00")

    injuries = _game_context("g1")["injuries"]
    assert len(injuries) == 4, f"distinct players collapsed: {injuries}"
    assert injuries[0]["game_status"] == "Out", "severity ordering lost"


def test_dedup_keys_on_player_id_when_present(ctx_db) -> None:
    """Two players can share a name; one player can be reported under name
    variants. `player_id` is the real identity when the feed supplies it."""
    from src.ai.redteam import _game_context

    _injury(ctx_db, "M. Penix", "Questionable", kt="2026-09-10T12:00:00+00:00",
            player_id="00-0039164")
    _injury(ctx_db, "Michael Penix Jr.", "Out", kt="2026-09-12T12:00:00+00:00",
            player_id="00-0039164")

    injuries = _game_context("g1")["injuries"]
    assert len(injuries) == 1, f"same player_id kept twice: {injuries}"
    assert injuries[0]["game_status"] == "Out"
