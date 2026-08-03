"""NFL team reference: abbreviation <-> full name, plus stadium geography.

Two jobs:

1. Crosswalk odds-feed team names ("Kansas City Chiefs") to nflverse
   abbreviations ("KC"). Substring matching does NOT work here — "kc" is not a
   substring of "kansas city chiefs", and neither is "gb", "sf", "tb", "no",
   "lar" or "lac". An explicit map is the only correct approach.

2. Stadium coordinates for weather lookups (Open-Meteo needs lat/lon), plus
   elevation, roof and surface — all features in their own right.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    abbr: str
    name: str
    conference: str
    division: str
    stadium: str
    roof: str          # outdoor | dome | retractable
    surface: str       # grass | turf
    lat: float
    lon: float
    elevation_m: float
    tz: str


TEAMS: list[Team] = [
    Team("ARI", "Arizona Cardinals", "NFC", "West", "State Farm Stadium", "retractable", "grass", 33.5277, -112.2626, 340, "America/Phoenix"),
    Team("ATL", "Atlanta Falcons", "NFC", "South", "Mercedes-Benz Stadium", "retractable", "turf", 33.7554, -84.4008, 320, "America/New_York"),
    Team("BAL", "Baltimore Ravens", "AFC", "North", "M&T Bank Stadium", "outdoor", "grass", 39.2780, -76.6227, 10, "America/New_York"),
    Team("BUF", "Buffalo Bills", "AFC", "East", "Highmark Stadium", "outdoor", "turf", 42.7738, -78.7870, 180, "America/New_York"),
    Team("CAR", "Carolina Panthers", "NFC", "South", "Bank of America Stadium", "outdoor", "turf", 35.2258, -80.8528, 230, "America/New_York"),
    Team("CHI", "Chicago Bears", "NFC", "North", "Soldier Field", "outdoor", "grass", 41.8623, -87.6167, 180, "America/Chicago"),
    Team("CIN", "Cincinnati Bengals", "AFC", "North", "Paycor Stadium", "outdoor", "turf", 39.0955, -84.5161, 150, "America/New_York"),
    Team("CLE", "Cleveland Browns", "AFC", "North", "Huntington Bank Field", "outdoor", "grass", 41.5061, -81.6995, 180, "America/New_York"),
    Team("DAL", "Dallas Cowboys", "NFC", "East", "AT&T Stadium", "retractable", "turf", 32.7473, -97.0945, 180, "America/Chicago"),
    Team("DEN", "Denver Broncos", "AFC", "West", "Empower Field at Mile High", "outdoor", "grass", 39.7439, -105.0201, 1610, "America/Denver"),
    Team("DET", "Detroit Lions", "NFC", "North", "Ford Field", "dome", "turf", 42.3400, -83.0456, 180, "America/New_York"),
    Team("GB", "Green Bay Packers", "NFC", "North", "Lambeau Field", "outdoor", "grass", 44.5013, -88.0622, 200, "America/Chicago"),
    Team("HOU", "Houston Texans", "AFC", "South", "NRG Stadium", "retractable", "turf", 29.6847, -95.4107, 15, "America/Chicago"),
    Team("IND", "Indianapolis Colts", "AFC", "South", "Lucas Oil Stadium", "retractable", "turf", 39.7601, -86.1639, 220, "America/New_York"),
    Team("JAX", "Jacksonville Jaguars", "AFC", "South", "EverBank Stadium", "outdoor", "grass", 30.3239, -81.6373, 5, "America/New_York"),
    Team("KC", "Kansas City Chiefs", "AFC", "West", "GEHA Field at Arrowhead", "outdoor", "grass", 39.0489, -94.4839, 270, "America/Chicago"),
    Team("LAC", "Los Angeles Chargers", "AFC", "West", "SoFi Stadium", "dome", "turf", 33.9535, -118.3392, 30, "America/Los_Angeles"),
    Team("LAR", "Los Angeles Rams", "NFC", "West", "SoFi Stadium", "dome", "turf", 33.9535, -118.3392, 30, "America/Los_Angeles"),
    Team("LV", "Las Vegas Raiders", "AFC", "West", "Allegiant Stadium", "dome", "grass", 36.0909, -115.1833, 640, "America/Los_Angeles"),
    Team("MIA", "Miami Dolphins", "AFC", "East", "Hard Rock Stadium", "outdoor", "grass", 25.9580, -80.2389, 3, "America/New_York"),
    Team("MIN", "Minnesota Vikings", "NFC", "North", "U.S. Bank Stadium", "dome", "turf", 44.9736, -93.2575, 250, "America/Chicago"),
    Team("NE", "New England Patriots", "AFC", "East", "Gillette Stadium", "outdoor", "turf", 42.0909, -71.2643, 90, "America/New_York"),
    Team("NO", "New Orleans Saints", "NFC", "South", "Caesars Superdome", "dome", "turf", 29.9511, -90.0812, 1, "America/Chicago"),
    Team("NYG", "New York Giants", "NFC", "East", "MetLife Stadium", "outdoor", "turf", 40.8135, -74.0745, 10, "America/New_York"),
    Team("NYJ", "New York Jets", "AFC", "East", "MetLife Stadium", "outdoor", "turf", 40.8135, -74.0745, 10, "America/New_York"),
    Team("PHI", "Philadelphia Eagles", "NFC", "East", "Lincoln Financial Field", "outdoor", "grass", 39.9008, -75.1675, 10, "America/New_York"),
    Team("PIT", "Pittsburgh Steelers", "AFC", "North", "Acrisure Stadium", "outdoor", "grass", 40.4468, -80.0158, 220, "America/New_York"),
    Team("SEA", "Seattle Seahawks", "NFC", "West", "Lumen Field", "outdoor", "turf", 47.5952, -122.3316, 10, "America/Los_Angeles"),
    Team("SF", "San Francisco 49ers", "NFC", "West", "Levi's Stadium", "outdoor", "grass", 37.4033, -121.9694, 10, "America/Los_Angeles"),
    Team("TB", "Tampa Bay Buccaneers", "NFC", "South", "Raymond James Stadium", "outdoor", "grass", 27.9759, -82.5033, 10, "America/New_York"),
    Team("TEN", "Tennessee Titans", "AFC", "South", "Nissan Stadium", "outdoor", "grass", 36.1665, -86.7713, 130, "America/Chicago"),
    Team("WAS", "Washington Commanders", "NFC", "East", "Northwest Stadium", "outdoor", "grass", 38.9077, -76.8645, 40, "America/New_York"),
]

BY_ABBR: dict[str, Team] = {t.abbr: t for t in TEAMS}
BY_NAME: dict[str, Team] = {t.name.lower(): t for t in TEAMS}

# Relocations and abbreviation drift across 25 years of nflverse data.
ALIASES: dict[str, str] = {
    "OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR",
    "WSH": "WAS", "WFT": "WAS", "ARZ": "ARI", "BLT": "BAL",
    "CLV": "CLE", "HST": "HOU", "SL": "LAR", "PHO": "ARI",
}

# Nickname -> abbr, for feeds that use only the mascot ("Chiefs").
BY_NICKNAME: dict[str, str] = {t.name.split()[-1].lower(): t.abbr for t in TEAMS}
BY_NICKNAME["49ers"] = "SF"
# Both NY and both LA teams share a nickname space with their city — nicknames
# alone are still unique, but the city is not. Never match on city alone.


def resolve(value: str | None) -> str | None:
    """Best-effort team resolution from an abbreviation, full name, or nickname.

    Returns a canonical nflverse abbreviation, or None. Deliberately strict:
    a wrong team assignment silently attaches odds to the wrong game.
    """
    if not value:
        return None
    v = value.strip()
    up = v.upper()
    if up in BY_ABBR:
        return up
    if up in ALIASES:
        return ALIASES[up]
    low = v.lower()
    if low in BY_NAME:
        return BY_NAME[low].abbr
    nick = low.split()[-1] if low.split() else ""
    if nick in BY_NICKNAME:
        return BY_NICKNAME[nick]
    return None


def seed_teams_table() -> int:
    """Populate the `teams` table — stadium geography for weather + features."""
    from src.db import db, upsert_rows

    rows = [{
        "team_abbr": t.abbr, "team_name": t.name, "conference": t.conference,
        "division": t.division, "stadium": t.stadium, "roof": t.roof,
        "surface": t.surface, "lat": t.lat, "lon": t.lon,
        "elevation_m": t.elevation_m, "tz": t.tz,
    } for t in TEAMS]
    with db() as conn:
        upsert_rows(conn, "teams", rows, key_cols=["team_abbr"])
    return len(rows)


if __name__ == "__main__":
    from src.db import run_migrations
    run_migrations()
    print(f"seeded {seed_teams_table()} teams")
    print("\nresolution checks (the ones substring matching gets wrong):")
    for v in ["Kansas City Chiefs", "Green Bay Packers", "San Francisco 49ers",
              "Tampa Bay Buccaneers", "New Orleans Saints", "Los Angeles Rams",
              "Los Angeles Chargers", "KC", "OAK", "Commanders", "Washington Football Team"]:
        print(f"  {v:<28} -> {resolve(v)}")
