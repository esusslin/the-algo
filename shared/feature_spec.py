"""SINGLE SOURCE OF TRUTH for model features.

Both planes import this:
  * research/  uses it to build training matrices
  * src/models/features_live.py uses it to build inference vectors

The artifact bundle records `spec_hash()`. The serving loader compares that hash
against the running code and REFUSES TO SERVE on mismatch. Training/serving skew
is the most common silent failure in deployed ML — a model scoring on misaligned
features produces confident garbage and no metric will tell you.

Populated properly in the Aug 22-28 block. The scaffolding is here now so nothing
gets built that bypasses it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Availability = Literal["pregame", "in_game", "postgame"]


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: str                      # float | int | bool | category
    default: Any                    # used when unavailable (e.g. Week 1 cold start)
    availability: Availability      # when this is knowable relative to kickoff
    description: str = ""
    lag_hours: float = 0.0          # min hours before kickoff it is reliably known
    group: str = "misc"


# --------------------------------------------------------------------------
# Feature registry — grows through the build. Order is significant; append only.
# --------------------------------------------------------------------------
FEATURES: list[Feature] = [
    # ---- market (highest signal; see architecture doc §3.4) ----
    Feature("market_spread", "float", 0.0, "pregame", "current consensus spread", group="market"),
    Feature("market_total", "float", 44.0, "pregame", "current consensus total", group="market"),
    Feature("market_home_prob", "float", 0.5, "pregame", "devigged home win prob", group="market"),
    Feature("line_move_spread", "float", 0.0, "pregame", "current minus opening spread", group="market"),
    Feature("book_dispersion", "float", 0.0, "pregame", "cross-book stdev of the line", group="market"),
    Feature("sharp_soft_delta", "float", 0.0, "pregame", "sharp book minus soft consensus", group="market"),

    # ---- context ----
    Feature("home_rest", "int", 7, "pregame", "days since last game", group="context"),
    Feature("away_rest", "int", 7, "pregame", "days since last game", group="context"),
    Feature("rest_diff", "int", 0, "pregame", "home_rest - away_rest", group="context"),
    Feature("div_game", "bool", False, "pregame", "divisional matchup", group="context"),
    Feature("is_dome", "bool", False, "pregame", "roof closed / controlled env", group="context"),
    Feature("kickoff_slot", "category", "sun_early", "pregame", "time-of-day slot", group="context"),
    Feature("week", "int", 1, "pregame", "week of season", group="context"),

    # ---- weather (feeds totals; see architecture doc §3.3) ----
    Feature("wind_kph", "float", 0.0, "pregame", "forecast wind at kickoff", lag_hours=0, group="weather"),
    Feature("wind_gust_kph", "float", 0.0, "pregame", "forecast gust — matters more than mean for kicking", group="weather"),
    Feature("high_wind", "bool", False, "pregame", "wind > 15mph threshold flag", group="weather"),
    Feature("temp_c", "float", 15.0, "pregame", "forecast temperature", group="weather"),
    Feature("precip_mm", "float", 0.0, "pregame", "forecast precipitation", group="weather"),

    # ---- team strength: opponent-adjusted unit ratings (research/ratings.py) ----
    # Sign convention: offense HIGHER is better, defense LOWER is better.
    Feature("home_off_rating", "float", 0.0, "pregame", "opp-adj offensive EPA/play", group="strength"),
    Feature("home_def_rating", "float", 0.0, "pregame", "opp-adj defensive EPA allowed", group="strength"),
    Feature("away_off_rating", "float", 0.0, "pregame", "opp-adj offensive EPA/play", group="strength"),
    Feature("away_def_rating", "float", 0.0, "pregame", "opp-adj defensive EPA allowed", group="strength"),
    Feature("home_off_pass", "float", 0.0, "pregame", "opp-adj pass offense", group="strength"),
    Feature("home_def_pass", "float", 0.0, "pregame", "opp-adj pass defense", group="strength"),
    Feature("away_off_pass", "float", 0.0, "pregame", "opp-adj pass offense", group="strength"),
    Feature("away_def_pass", "float", 0.0, "pregame", "opp-adj pass defense", group="strength"),
    Feature("home_off_rush", "float", 0.0, "pregame", "opp-adj rush offense", group="strength"),
    Feature("home_def_rush", "float", 0.0, "pregame", "opp-adj rush defense", group="strength"),
    Feature("away_off_rush", "float", 0.0, "pregame", "opp-adj rush offense", group="strength"),
    Feature("away_def_rush", "float", 0.0, "pregame", "opp-adj rush defense", group="strength"),

    # ---- matchup: unit vs unit, the actual mismatch signal ----
    Feature("home_pass_edge", "float", 0.0, "pregame", "home pass off minus away pass def", group="matchup"),
    Feature("away_pass_edge", "float", 0.0, "pregame", "away pass off minus home pass def", group="matchup"),
    Feature("home_rush_edge", "float", 0.0, "pregame", "home rush off minus away rush def", group="matchup"),
    Feature("away_rush_edge", "float", 0.0, "pregame", "away rush off minus home rush def", group="matchup"),
    Feature("net_edge", "float", 0.0, "pregame", "overall home minus away advantage", group="matchup"),
    Feature("ratings_confident", "bool", False, "pregame", "both teams past the small-sample threshold", group="matchup"),

    # ---- availability ----
    #
    # The feed is now proven: `research/practice_signal.py` measured practice status at
    # +0.054 AUC on the Questionable panel across 90,467 rows, walk-forward by season.
    # These are the *team-level* consequence of that — not "is this player playing" but
    # "how much of this team isn't".
    #
    # **`missing_snap_share` is the one that should matter.** Counting bodies treats a
    # third-string safety and a left tackle alike. Weighting each unavailable player by
    # the offensive snap share he took over the previous four weeks measures what the
    # team actually loses. It exists only because the pfr→gsis crosswalk landed — snap
    # counts key on `pfr_player_id` and everything else keys on gsis.
    #
    # **Availability window: 2009+ for the counts, 2013+ for snap share.** Before those
    # years the data does not exist and the features are zero — which a model can learn
    # as an era marker rather than as a fact about injuries. `availability_known` exists
    # so that's explicit rather than inferred, and `research/train.py` compares a 2013+
    # baseline against a 2013+ availability model so the era effect and the feature
    # effect can't be confused.
    Feature("home_out_count", "int", 0, "pregame", "players ruled Out", group="availability"),
    Feature("away_out_count", "int", 0, "pregame", "players ruled Out", group="availability"),
    Feature("home_questionable_count", "int", 0, "pregame", "players Questionable", group="availability"),
    Feature("away_questionable_count", "int", 0, "pregame", "players Questionable", group="availability"),
    Feature("home_qb_out", "bool", False, "pregame", "a quarterback is Out", group="availability"),
    Feature("away_qb_out", "bool", False, "pregame", "a quarterback is Out", group="availability"),
    Feature("home_missing_snap_share", "float", 0.0, "pregame", "sum of prior-4wk offensive snap share of players Out", group="availability"),
    Feature("away_missing_snap_share", "float", 0.0, "pregame", "sum of prior-4wk offensive snap share of players Out", group="availability"),
    Feature("availability_known", "bool", False, "pregame", "injury and snap data exist for this season", group="availability"),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def spec_hash() -> str:
    """Stable hash over the feature contract. Recorded in every artifact bundle."""
    payload = [
        {"name": f.name, "dtype": f.dtype, "default": f.default,
         "availability": f.availability}
        for f in FEATURES
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def to_vector(values: dict[str, Any]) -> list[Any]:
    """Dict -> ordered feature vector, filling defaults for anything missing."""
    return [values.get(f.name, f.default) for f in FEATURES]


def validate_availability(decision_time_hours_before_kick: float) -> list[str]:
    """Which features are NOT legitimately knowable at a given decision time.

    Used by the leakage test suite — a pregame model must never consume an
    in_game feature, and a feature with lag_hours=3 is not knowable 1h out.
    """
    bad = []
    for f in FEATURES:
        if f.availability != "pregame":
            bad.append(f"{f.name}: availability={f.availability}")
        elif f.lag_hours > decision_time_hours_before_kick:
            bad.append(f"{f.name}: needs {f.lag_hours}h lead, have {decision_time_hours_before_kick}h")
    return bad


if __name__ == "__main__":
    print(f"features: {len(FEATURES)}")
    print(f"spec_hash: {spec_hash()}")
    for g in sorted({f.group for f in FEATURES}):
        names = [f.name for f in FEATURES if f.group == g]
        print(f"  {g:<10} {len(names):>2}  {', '.join(names[:4])}{' ...' if len(names) > 4 else ''}")
