"""Market taxonomy — the backbone of history filtering.

The Odds API returns flat market keys like `spreads_h1` and
`player_reception_yds`. To filter history by bet type ("show me every 1st-half
total", "show me all receiving props") we need structure: what class of bet is
it, what period does it cover, and what stat family.

Everything here is derived from the key so unknown markets degrade sensibly
rather than disappearing from history.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# classes and periods
# --------------------------------------------------------------------------
CLASS_GAME = "game"        # moneyline, spread, total on the full game
CLASS_PERIOD = "period"    # halves and quarters
CLASS_TEAM = "team"        # team totals
CLASS_PROP = "prop"        # player props
CLASS_ALT = "alt"          # alternate ladders

PERIOD_FULL = "full"
PERIOD_H1 = "1H"
PERIOD_H2 = "2H"
PERIOD_Q = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"}

# stat families for props — what the history filter groups on
FAM_PASS = "passing"
FAM_RUSH = "rushing"
FAM_REC = "receiving"
FAM_TD = "touchdown"
FAM_KICK = "kicking"
FAM_DEF = "defense"


@dataclass(frozen=True)
class MarketInfo:
    key: str
    label: str          # human-readable, sentence case
    bet_class: str      # game | period | team | prop | alt
    period: str         # full | 1H | 2H | Q1..Q4
    family: str | None  # stat family for props
    two_sided: bool     # over/under or home/away (vs. yes/no like anytime TD)


_BASE_LABELS = {
    "h2h": "Moneyline",
    "spreads": "Spread",
    "totals": "Total",
    "team_totals": "Team total",
    "alternate_spreads": "Alt spread",
    "alternate_totals": "Alt total",
}

_PROP_LABELS = {
    "player_pass_yds": ("Passing yards", FAM_PASS),
    "player_pass_tds": ("Passing TDs", FAM_PASS),
    "player_pass_completions": ("Completions", FAM_PASS),
    "player_pass_attempts": ("Pass attempts", FAM_PASS),
    "player_pass_interceptions": ("Interceptions", FAM_PASS),
    "player_pass_longest_completion": ("Longest completion", FAM_PASS),
    "player_rush_yds": ("Rushing yards", FAM_RUSH),
    "player_rush_attempts": ("Rush attempts", FAM_RUSH),
    "player_rush_longest": ("Longest rush", FAM_RUSH),
    "player_reception_yds": ("Receiving yards", FAM_REC),
    "player_receptions": ("Receptions", FAM_REC),
    "player_reception_longest": ("Longest reception", FAM_REC),
    "player_rush_reception_yds": ("Rush + rec yards", FAM_RUSH),
    "player_anytime_td": ("Anytime TD", FAM_TD),
    "player_1st_td": ("First TD", FAM_TD),
    "player_last_td": ("Last TD", FAM_TD),
    "player_kicking_points": ("Kicking points", FAM_KICK),
    "player_field_goals": ("Field goals", FAM_KICK),
    "player_tackles_assists": ("Tackles + assists", FAM_DEF),
    "player_sacks": ("Sacks", FAM_DEF),
    "player_defensive_interceptions": ("Def. interceptions", FAM_DEF),
}

# One-sided markets: there is no paired opposite outcome to devig against.
ONE_SIDED = {"player_anytime_td", "player_1st_td", "player_last_td"}


def _split_period(key: str) -> tuple[str, str]:
    """Strip a period suffix. Returns (base_key, period)."""
    for suffix, period in (("_h1", PERIOD_H1), ("_h2", PERIOD_H2)):
        if key.endswith(suffix):
            return key[: -len(suffix)], period
    for suffix, period in PERIOD_Q.items():
        if key.endswith(f"_{suffix}"):
            return key[: -(len(suffix) + 1)], period
    return key, PERIOD_FULL


def describe_market(key: str) -> MarketInfo:
    """Parse any market key into structured info. Never raises."""
    if not key:
        return MarketInfo("", "Unknown", CLASS_GAME, PERIOD_FULL, None, True)

    if key.startswith("player_"):
        label, family = _PROP_LABELS.get(key, (key.replace("player_", "").replace("_", " ").capitalize(), None))
        return MarketInfo(key, label, CLASS_PROP, PERIOD_FULL, family,
                          key not in ONE_SIDED)

    base, period = _split_period(key)

    if base.startswith("alternate_"):
        return MarketInfo(key, _BASE_LABELS.get(base, base), CLASS_ALT, period, None, True)
    if base == "team_totals":
        return MarketInfo(key, _BASE_LABELS[base], CLASS_TEAM, period, None, True)

    label = _BASE_LABELS.get(base, base.replace("_", " ").capitalize())
    if period != PERIOD_FULL:
        label = f"{period} {label.lower()}"
        return MarketInfo(key, label, CLASS_PERIOD, period, None, True)
    return MarketInfo(key, label, CLASS_GAME, period, None, True)


# --------------------------------------------------------------------------
# filter groups — what the History tab's chips map to
# --------------------------------------------------------------------------
FILTER_GROUPS: dict[str, dict] = {
    "all": {"label": "All", "match": lambda m: True},
    "moneyline": {"label": "Moneyline",
                  "match": lambda m: m.key.startswith("h2h")},
    "spread": {"label": "Spread",
               "match": lambda m: "spreads" in m.key},
    "total": {"label": "Total",
              "match": lambda m: "totals" in m.key and m.bet_class != CLASS_TEAM},
    "team_total": {"label": "Team total",
                   "match": lambda m: m.bet_class == CLASS_TEAM},
    "first_half": {"label": "1st half", "match": lambda m: m.period == PERIOD_H1},
    "second_half": {"label": "2nd half", "match": lambda m: m.period == PERIOD_H2},
    "quarters": {"label": "Quarters",
                 "match": lambda m: m.period in PERIOD_Q.values()},
    "props": {"label": "All props", "match": lambda m: m.bet_class == CLASS_PROP},
    "passing": {"label": "Passing", "match": lambda m: m.family == FAM_PASS},
    "rushing": {"label": "Rushing", "match": lambda m: m.family == FAM_RUSH},
    "receiving": {"label": "Receiving", "match": lambda m: m.family == FAM_REC},
    "touchdowns": {"label": "Touchdowns", "match": lambda m: m.family == FAM_TD},
    "kicking": {"label": "Kicking", "match": lambda m: m.family == FAM_KICK},
    "defense": {"label": "Defense", "match": lambda m: m.family == FAM_DEF},
}

# Chips shown in the UI, in order. Others remain available via the API.
PRIMARY_FILTERS = ["all", "moneyline", "spread", "total", "props"]
SECONDARY_FILTERS = ["team_total", "first_half", "second_half", "quarters",
                     "passing", "rushing", "receiving", "touchdowns",
                     "kicking", "defense"]


def matches_filter(market_key: str, filter_name: str) -> bool:
    grp = FILTER_GROUPS.get(filter_name)
    if not grp:
        return True
    return bool(grp["match"](describe_market(market_key)))


def market_keys_for(filter_name: str, known_keys: list[str]) -> list[str]:
    """Which of the market keys present in the DB match a filter."""
    return [k for k in known_keys if matches_filter(k, filter_name)]


def side_label(market_key: str, side: str, line: float,
               home_team: str = "", away_team: str = "",
               player_name: str = "") -> str:
    """Human-readable bet description. Used in history rows and pick cards."""
    info = describe_market(market_key)
    if info.bet_class == CLASS_PROP:
        who = player_name or "Player"
        if market_key in ONE_SIDED:
            return f"{who} {info.label.lower()}"
        return f"{who} {side} {abs(line):g} {info.label.lower()}"
    if side in ("over", "under"):
        return f"{side.title()} {abs(line):g}"
    team = home_team if side == "home" else away_team
    team = team or side.title()
    if "spreads" in market_key:
        return f"{team} {line:+g}"
    return team


if __name__ == "__main__":
    samples = ["h2h", "spreads", "totals", "spreads_h1", "totals_h2", "h2h_q1",
               "team_totals", "alternate_spreads", "player_pass_yds",
               "player_reception_yds", "player_anytime_td", "player_sacks",
               "some_unknown_market"]
    print(f"{'key':<32}{'label':<22}{'class':<8}{'period':<8}{'family'}")
    for k in samples:
        m = describe_market(k)
        print(f"  {m.key:<30}{m.label:<22}{m.bet_class:<8}{m.period:<8}{m.family or '-'}")

    print("\nfilter membership:")
    for f in ["moneyline", "spread", "total", "first_half", "props", "receiving"]:
        hits = [k for k in samples if matches_filter(k, f)]
        print(f"  {f:<14} {hits}")
