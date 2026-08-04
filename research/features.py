"""Point-in-time feature assembly — warehouse + ratings + schedule → training matrix.

The contract: every row's features are computed from information available
BEFORE that game kicked off. Ratings come from `unit_ratings` at that season and
week, which by construction used only earlier weeks. Market values come from the
schedule's closing line, which is a known compromise documented below.

Feature names and order come from `shared.feature_spec`, imported by both planes.
That single source of truth is what prevents training/serving skew — a model
scoring on misaligned features produces confident garbage and no metric catches it.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from research.warehouse import connect
from shared.feature_spec import FEATURES, spec_hash

log = logging.getLogger(__name__)


def _games(seasons: list[int] | None = None) -> list[dict]:
    """Schedule, results and closing lines from the serving DB."""
    from src.db import query
    sql = ("SELECT game_id, season, week, home_team, away_team, home_score, "
           "away_score, spread_line, total_line, home_rest, away_rest, "
           "div_game, roof, surface, kickoff_utc "
           "FROM games WHERE home_score IS NOT NULL AND away_score IS NOT NULL")
    params: list = []
    if seasons:
        sql += f" AND season IN ({','.join('?' * len(seasons))})"
        params = list(seasons)
    return [dict(r) for r in query(sql + " ORDER BY season, week", params)]


def _ratings_index() -> dict[tuple, dict]:
    """(season, week, team, class) -> ratings. Loaded once; the table is small."""
    con = connect(read_only=True)
    try:
        rows = con.execute(
            "SELECT season, week, team, play_class, off_rating, def_rating, confident "
            "FROM unit_ratings").fetchall()
    except Exception:  # noqa: BLE001
        log.error("no unit_ratings — run `python -m research.ratings build`")
        return {}
    finally:
        con.close()
    return {(r[0], r[1], r[2], r[3]): {"off": r[4], "def": r[5], "confident": bool(r[6])}
            for r in rows}


def _slot(kickoff: str | None) -> str:
    if not kickoff or len(kickoff) < 13:
        return "sun_early"
    try:
        hour = int(kickoff[11:13])
    except ValueError:
        return "sun_early"
    if hour < 17:
        return "sun_early"
    if hour < 21:
        return "sun_late"
    return "primetime"


def build_matrix(seasons: list[int] | None = None,
                 require_market: bool = True) -> dict[str, Any]:
    """Assemble the training matrix.

    Returns X (n × features), several targets, and metadata for walk-forward.

    Targets:
      margin           home_score - away_score
      residual         margin + spread_line   <- the market-anchored target
      home_cover       did the home side beat the spread
      total_points     combined score
      total_residual   total_points - total_line
    """
    games = _games(seasons)
    ratings = _ratings_index()
    if not games:
        raise SystemExit("no completed games — run `python -m src.fetchers.nflverse games`")

    names = [f.name for f in FEATURES]
    rows: list[list[float]] = []
    meta: list[dict] = []
    skipped = {"no_market": 0, "no_ratings": 0}

    for g in games:
        if require_market and (g["spread_line"] is None or g["total_line"] is None):
            skipped["no_market"] += 1
            continue

        s, w, h, a = g["season"], g["week"], g["home_team"], g["away_team"]

        def r(team: str, cls: str) -> dict:
            return ratings.get((s, w, team, cls), {"off": 0.0, "def": 0.0,
                                                   "confident": False})

        ha, aa = r(h, "all"), r(a, "all")
        hp, ap = r(h, "pass"), r(a, "pass")
        hr, ar = r(h, "rush"), r(a, "rush")
        if not ha["confident"] and not aa["confident"] and w > 6:
            skipped["no_ratings"] += 1

        vals: dict[str, Any] = {
            # market — the anchor everything else is measured against
            "market_spread": g["spread_line"] or 0.0,
            "market_total": g["total_line"] or 44.0,
            "market_home_prob": 0.5,     # filled from odds history when available
            "line_move_spread": 0.0,
            "book_dispersion": 0.0,
            "sharp_soft_delta": 0.0,
            # context
            "home_rest": g["home_rest"] if g["home_rest"] is not None else 7,
            "away_rest": g["away_rest"] if g["away_rest"] is not None else 7,
            "rest_diff": (g["home_rest"] or 7) - (g["away_rest"] or 7),
            "div_game": bool(g["div_game"]),
            "is_dome": (g["roof"] or "").lower() in {"dome", "closed", "retractable"},
            "kickoff_slot": _slot(g["kickoff_utc"]),
            "week": w or 1,
            # weather — joined separately once backfilled
            "wind_kph": 0.0, "wind_gust_kph": 0.0, "high_wind": False,
            "temp_c": 15.0, "precip_mm": 0.0,
            # strength
            "home_off_rating": ha["off"], "home_def_rating": ha["def"],
            "away_off_rating": aa["off"], "away_def_rating": aa["def"],
            "home_off_pass": hp["off"], "home_def_pass": hp["def"],
            "away_off_pass": ap["off"], "away_def_pass": ap["def"],
            "home_off_rush": hr["off"], "home_def_rush": hr["def"],
            "away_off_rush": ar["off"], "away_def_rush": ar["def"],
            # matchup: unit vs unit
            "home_pass_edge": hp["off"] - ap["def"],
            "away_pass_edge": ap["off"] - hp["def"],
            "home_rush_edge": hr["off"] - ar["def"],
            "away_rush_edge": ar["off"] - hr["def"],
            "net_edge": (ha["off"] - aa["def"]) - (aa["off"] - ha["def"]),
            "ratings_confident": ha["confident"] and aa["confident"],
        }

        row = []
        for f in FEATURES:
            v = vals.get(f.name, f.default)
            if f.dtype == "category":
                v = hash(str(v)) % 1000        # placeholder encoding
            row.append(float(v) if not isinstance(v, bool) else float(v))
        rows.append(row)

        margin = g["home_score"] - g["away_score"]
        total = g["home_score"] + g["away_score"]
        spread = g["spread_line"] or 0.0
        meta.append({
            "game_id": g["game_id"], "season": s, "week": w,
            "home": h, "away": a,
            "margin": margin,
            # spread_line is stored from the home perspective, so a home -3.5
            # favourite covers when margin + (-3.5) > 0
            "residual": margin + spread,
            "home_cover": 1 if (margin + spread) > 0 else 0,
            "push": abs(margin + spread) < 1e-9,
            "total_points": total,
            "total_residual": total - (g["total_line"] or 44.0),
            "over": 1 if total > (g["total_line"] or 44.0) else 0,
            "confident": vals["ratings_confident"],
        })

    X = np.array(rows, dtype=float)
    log.info("matrix: %d games × %d features (skipped %s)", len(X), X.shape[1] if len(X) else 0, skipped)
    return {
        "X": X,
        "names": names,
        "meta": meta,
        "seasons": np.array([m["season"] for m in meta]),
        "weeks": np.array([m["week"] for m in meta]),
        "margin": np.array([m["margin"] for m in meta], dtype=float),
        "residual": np.array([m["residual"] for m in meta], dtype=float),
        "home_cover": np.array([m["home_cover"] for m in meta]),
        "total_residual": np.array([m["total_residual"] for m in meta], dtype=float),
        "over": np.array([m["over"] for m in meta]),
        "spec_hash": spec_hash(),
        "skipped": skipped,
    }


def sanity(d: dict) -> list[str]:
    """Checks that catch the labelling bugs which otherwise inflate a backtest.

    Tolerances scale with sample size. A fixed band flags noise as a bug on a
    small sample and misses a real bug on a large one — both failure modes
    erode trust in the check itself, which is worse than not having it.
    """
    problems = []
    meta = d["meta"]
    n = len(meta)
    # 4 standard errors on a fair coin: wide enough to ignore noise, tight
    # enough that an inverted sign (which lands near 0 or 1) always trips.
    tol = 4.0 * math.sqrt(0.25 / max(n, 1))

    def rate_check(label: str, value: float, expected: float, note: str) -> None:
        lo, hi = expected - tol, expected + tol
        print(f"  {label:<21}: {value:.4f}   (expect ~{expected:.2f}, "
              f"tolerance ±{tol:.3f})")
        if not lo <= value <= hi:
            problems.append(f"{label.strip()} {value:.4f} outside ±{tol:.3f} — {note}")

    print(f"  games                : {n:,}")
    rate_check("home cover rate", float(np.mean(d["home_cover"])), 0.50,
               "check spread sign convention")
    rate_check("over rate", float(np.mean(d["over"])), 0.50,
               "check total sign convention")
    rate_check("home win rate", float(np.mean([m["margin"] > 0 for m in meta])), 0.55,
               "home-field advantage looks wrong")

    resid_mean = float(np.mean(d["residual"]))
    resid_se = float(np.std(d["residual"]) / math.sqrt(max(n, 1)))
    print(f"  mean spread residual : {resid_mean:+.3f} ± {2*resid_se:.3f}  "
          f"(expect near 0 — the market is unbiased)")
    if abs(resid_mean) > max(1.0, 3 * resid_se):
        problems.append(f"mean residual {resid_mean:+.3f} is {abs(resid_mean)/max(resid_se,1e-9):.1f} "
                        f"SE from zero — sign error likely")

    # DIRECTIONAL CHECKS — the only ones that catch an inverted sign.
    #
    # Rate-based checks cannot: flipping the spread sign leaves the cover rate
    # at ~50% and the mean residual merely changes sign, both of which look
    # fine. What breaks is the RELATIONSHIP between the line and the outcome.
    #
    # spread_line is stored home-perspective, so a home favourite is negative
    # while their margin is positive: the correlation must be strongly negative.
    margins = np.array([m["margin"] for m in meta], dtype=float)
    spreads = d["X"][:, d["names"].index("market_spread")]
    if np.std(spreads) > 1e-9:
        r_sm = float(np.corrcoef(spreads, margins)[0, 1])
        print(f"  corr(spread, margin) : {r_sm:+.3f}   (MUST be strongly negative)")
        if r_sm > -0.15:
            problems.append(
                f"corr(spread, margin)={r_sm:+.3f} — expected strongly negative. "
                f"The spread sign is inverted, or the line is not home-perspective. "
                f"This is the bug that produces a plausible 50% hit rate while "
                f"silently reversing every result.")

    totals = np.array([m["total_points"] for m in meta], dtype=float)
    tlines = d["X"][:, d["names"].index("market_total")]
    if np.std(tlines) > 1e-9:
        r_tt = float(np.corrcoef(tlines, totals)[0, 1])
        print(f"  corr(total_line, pts): {r_tt:+.3f}   (MUST be positive)")
        if r_tt < 0.15:
            problems.append(f"corr(total_line, points)={r_tt:+.3f} — expected positive")

    # cover must follow the residual by construction; if not, the label is wrong
    covers = np.asarray(d["home_cover"], dtype=float)
    if np.std(covers) > 1e-9:
        r_cr = float(np.corrcoef(covers, d["residual"])[0, 1])
        print(f"  corr(cover, residual): {r_cr:+.3f}   (MUST be strongly positive)")
        if r_cr < 0.5:
            problems.append(f"corr(cover, residual)={r_cr:+.3f} — the cover label "
                            f"does not follow from the residual")

    conf = float(np.mean([m["confident"] for m in meta]))
    print(f"  ratings confident    : {conf:.3f}   (low early-season is expected)")

    nan_cols = [d["names"][j] for j in range(d["X"].shape[1])
                if np.isnan(d["X"][:, j]).any()]
    if nan_cols:
        problems.append(f"NaNs in: {', '.join(nan_cols[:6])}")
    return problems


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="feature matrix (local only)")
    p.add_argument("command", choices=["build", "sanity"])
    p.add_argument("--from-season", type=int, default=None)
    args = p.parse_args()

    seasons = None
    if args.from_season:
        from src.db import query
        seasons = [r["season"] for r in query(
            "SELECT DISTINCT season FROM games WHERE season >= ? ORDER BY 1",
            (args.from_season,))]

    d = build_matrix(seasons)
    print(f"\n  spec hash: {d['spec_hash']}")
    print(f"  shape    : {d['X'].shape}\n")
    probs = sanity(d)
    if probs:
        print()
        for pr in probs:
            print(f"  PROBLEM: {pr}")
        raise SystemExit(1)
    print("\n  all checks passed")
