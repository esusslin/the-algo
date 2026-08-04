"""Opponent-adjusted unit ratings — the core matchup signal.

The problem
-----------
A defence allowing -0.05 EPA/play has told you nothing until you know whom they
played. Raw rolling averages are confounded by schedule, and schedule strength
varies enormously in a 17-game season.

The fix
-------
Decompose every play's EPA into the units responsible:

    epa ≈ β_off[offense] + β_def[defense] + β_hfa + ε

fit by ridge regression, separately for pass and rush. The coefficients are what
each unit contributes *after* accounting for opposition.

Point-in-time by construction
-----------------------------
Ratings are computed once per (season, week) using ONLY plays strictly before
that week. The output table is therefore a complete record of "what was knowable
at week W" — which is exactly what a feature store needs, and it means a
training matrix can never accidentally consume a rating that included the game
it is trying to predict.

Two properties worth understanding:

* **Ridge, not OLS.** Early season there are more parameters than informative
  observations. The penalty shrinks unit ratings toward zero (league average),
  which is the correct behaviour when you've seen a team twice — not a
  compromise.
* **Recency weighting.** Week 8 should count more than Week 1 without Week 1
  falling off a cliff at some arbitrary window boundary. Exponential decay,
  λ tuned by walk-forward.
"""
from __future__ import annotations

import logging

import numpy as np

from research.warehouse import connect

log = logging.getLogger(__name__)

# Ridge penalty. Higher = more shrinkage toward league average. Tuned by
# walk-forward; this default is deliberately strong for small samples.
DEFAULT_ALPHA = 60.0
# Per-week decay. 0.93 halves a game's weight after ~10 weeks.
DEFAULT_LAMBDA = 0.93
# Below this, a unit's rating is mostly prior and should be treated as such.
MIN_PLAYS_FOR_CONFIDENCE = 120


def _ridge_solve(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                 alpha: float) -> np.ndarray:
    """Weighted ridge via normal equations. The intercept is not penalised."""
    Xw = X * w[:, None]
    A = X.T @ Xw
    A[np.diag_indices_from(A)] += alpha
    A[-1, -1] -= alpha                      # last column is the intercept
    b = Xw.T @ y if Xw.shape[0] == y.shape[0] else X.T @ (w * y)
    try:
        return np.linalg.solve(A, X.T @ (w * y))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, X.T @ (w * y), rcond=None)[0]


def fit_ratings(plays: dict[str, np.ndarray], teams: list[str],
                alpha: float = DEFAULT_ALPHA) -> dict[str, dict[str, float]]:
    """One ridge fit. Returns {'off': {team: rating}, 'def': {team: rating}}.

    Sign convention, stated explicitly because it inverts easily:
      offense rating  — HIGHER is better (more EPA generated)
      defense rating  — LOWER is better (less EPA allowed)
    """
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(plays["epa"]), len(teams)
    if n == 0:
        return {"off": {}, "def": {}}

    # [offense one-hot | defense one-hot | intercept]
    X = np.zeros((n, 2 * k + 1))
    for r, (o, d) in enumerate(zip(plays["offense"], plays["defense"])):
        if o in idx:
            X[r, idx[o]] = 1.0
        if d in idx:
            X[r, k + idx[d]] = 1.0
    X[:, -1] = 1.0

    beta = _ridge_solve(X, plays["epa"], plays["weight"], alpha)
    return {
        "off": {t: float(beta[idx[t]]) for t in teams},
        "def": {t: float(beta[k + idx[t]]) for t in teams},
        "intercept": float(beta[-1]),
    }


def build_ratings(seasons: list[int] | None = None,
                  alpha: float = DEFAULT_ALPHA,
                  lam: float = DEFAULT_LAMBDA) -> int:
    """Compute point-in-time ratings for every (season, week).

    For week W, uses only plays from weeks < W of that season. Week 1 has no
    within-season data at all and is left to the cold-start prior (see
    IN_SEASON_LEARNING.md) rather than fabricated here.
    """
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS unit_ratings (
            season INTEGER, week INTEGER, team VARCHAR, play_class VARCHAR,
            off_rating DOUBLE, def_rating DOUBLE,
            off_plays INTEGER, def_plays INTEGER,
            confident BOOLEAN
        )
    """)

    if seasons is None:
        seasons = [r[0] for r in con.execute(
            "SELECT DISTINCT season FROM plays ORDER BY 1").fetchall()]

    total = 0
    for season in seasons:
        con.execute("DELETE FROM unit_ratings WHERE season = ?", [season])
        weeks = [r[0] for r in con.execute(
            "SELECT DISTINCT week FROM plays WHERE season=? ORDER BY 1",
            [season]).fetchall()]
        teams = [r[0] for r in con.execute(
            "SELECT DISTINCT offense FROM plays WHERE season=? ORDER BY 1",
            [season]).fetchall()]
        if not teams:
            continue

        for wk in weeks:
            if wk <= 1:
                continue                     # nothing knowable yet this season

            rows: list[tuple] = []
            for play_class in ("pass", "rush", "all"):
                clause = "" if play_class == "all" else f"AND play_class='{play_class}'"
                data = con.execute(f"""
                    SELECT offense, defense, epa, week
                    FROM plays
                    WHERE season=? AND week < ? AND epa IS NOT NULL {clause}
                """, [season, wk]).fetchall()
                if len(data) < 200:
                    continue

                off = np.array([d[0] for d in data])
                dfn = np.array([d[1] for d in data])
                epa = np.array([d[2] for d in data], dtype=float)
                wks = np.array([d[3] for d in data], dtype=float)
                weight = lam ** (wk - wks)   # exponential recency decay

                fit = fit_ratings(
                    {"offense": off, "defense": dfn, "epa": epa, "weight": weight},
                    teams, alpha=alpha)

                off_counts = {t: int((off == t).sum()) for t in teams}
                def_counts = {t: int((dfn == t).sum()) for t in teams}
                for t in teams:
                    rows.append((
                        season, wk, t, play_class,
                        fit["off"].get(t, 0.0), fit["def"].get(t, 0.0),
                        off_counts.get(t, 0), def_counts.get(t, 0),
                        min(off_counts.get(t, 0), def_counts.get(t, 0))
                        >= MIN_PLAYS_FOR_CONFIDENCE,
                    ))

            if rows:
                con.executemany(
                    "INSERT INTO unit_ratings VALUES (?,?,?,?,?,?,?,?,?)", rows)
                total += len(rows)
        log.info("season %s: ratings through week %s", season, weeks[-1] if weeks else "-")

    con.execute("CREATE INDEX IF NOT EXISTS idx_ur ON unit_ratings(season, week, team)")
    con.close()
    return total


def rating_as_of(season: int, week: int, team: str,
                 play_class: str = "all") -> dict | None:
    """What we knew about a team going into a given week. The point-in-time read."""
    con = connect(read_only=True)
    row = con.execute("""
        SELECT off_rating, def_rating, off_plays, def_plays, confident
        FROM unit_ratings
        WHERE season=? AND week=? AND team=? AND play_class=?
    """, [season, week, team, play_class]).fetchone()
    con.close()
    if not row:
        return None
    return {"off_rating": row[0], "def_rating": row[1], "off_plays": row[2],
            "def_plays": row[3], "confident": bool(row[4])}


def matchup(season: int, week: int, home: str, away: str) -> dict:
    """The unit-vs-unit view for one game — what MATCHUP_MODELING.md describes.

    Positive `home_pass_edge` means the home passing attack rates above what the
    away pass defence typically allows.
    """
    out: dict = {"season": season, "week": week, "home": home, "away": away}
    for cls in ("pass", "rush", "all"):
        h = rating_as_of(season, week, home, cls)
        a = rating_as_of(season, week, away, cls)
        if not h or not a:
            continue
        out[f"home_{cls}_edge"] = round(h["off_rating"] - a["def_rating"], 4)
        out[f"away_{cls}_edge"] = round(a["off_rating"] - h["def_rating"], 4)
        out[f"{cls}_confident"] = h["confident"] and a["confident"]
    if "home_all_edge" in out and "away_all_edge" in out:
        out["net_edge"] = round(out["home_all_edge"] - out["away_all_edge"], 4)
    return out


def sanity(season: int | None = None) -> None:
    """Ratings should behave like football. If they don't, the fit is wrong."""
    con = connect(read_only=True)
    season = season or con.execute("SELECT MAX(season) FROM unit_ratings").fetchone()[0]
    wk = con.execute("SELECT MAX(week) FROM unit_ratings WHERE season=?",
                     [season]).fetchone()[0]
    if not wk:
        print("  no ratings — run build first")
        return

    print(f"  season {season}, ratings as of week {wk}\n")
    rows = con.execute("""
        SELECT team, off_rating, def_rating, off_plays, confident
        FROM unit_ratings WHERE season=? AND week=? AND play_class='all'
        ORDER BY off_rating DESC
    """, [season, wk]).fetchall()
    print(f"  {'team':<6}{'off':>9}{'def':>9}{'plays':>8}  conf")
    for t, o, d, n, c in rows[:6]:
        print(f"  {t:<6}{o:>+9.4f}{d:>+9.4f}{n:>8}  {'y' if c else 'n'}")
    print("  ...")
    for t, o, d, n, c in rows[-3:]:
        print(f"  {t:<6}{o:>+9.4f}{d:>+9.4f}{n:>8}  {'y' if c else 'n'}")

    offs = np.array([r[1] for r in rows])
    print(f"\n  offense spread : {offs.min():+.4f} to {offs.max():+.4f} "
          f"(sd {offs.std():.4f})")
    print(f"  mean offense   : {offs.mean():+.5f}  (ridge centres this near 0)")
    con.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="opponent-adjusted ratings (local only)")
    p.add_argument("command", choices=["build", "sanity", "matchup", "selftest"])
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--home", default="")
    p.add_argument("--away", default="")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = p.parse_args()

    if args.command == "build":
        seasons = [args.season] if args.season else None
        print(f"{build_ratings(seasons, alpha=args.alpha):,} rating rows")
    elif args.command == "sanity":
        sanity(args.season)
    elif args.command == "matchup":
        for k, v in matchup(args.season, args.week, args.home, args.away).items():
            print(f"  {k:<20}{v}")
    else:
        # Can the ridge recover known strengths from synthetic plays?
        rng = np.random.default_rng(5)
        teams = [f"T{i}" for i in range(8)]
        true_off = {t: rng.normal(0, 0.12) for t in teams}
        true_def = {t: rng.normal(0, 0.10) for t in teams}
        off, dfn, epa = [], [], []
        for _ in range(20000):
            o, d = rng.choice(teams), rng.choice(teams)
            if o == d:
                continue
            off.append(o); dfn.append(d)
            epa.append(true_off[o] + true_def[d] + rng.normal(0, 1.3))
        fit = fit_ratings({"offense": np.array(off), "defense": np.array(dfn),
                           "epa": np.array(epa), "weight": np.ones(len(epa))},
                          teams, alpha=10.0)
        print(f"  {'team':<6}{'true off':>10}{'fit off':>10}{'true def':>11}{'fit def':>10}")
        errs = []
        for t in teams:
            fo, fd = fit["off"][t], fit["def"][t]
            errs += [abs(fo - true_off[t]), abs(fd - true_def[t])]
            print(f"  {t:<6}{true_off[t]:>+10.4f}{fo:>+10.4f}"
                  f"{true_def[t]:>+11.4f}{fd:>+10.4f}")
        r = np.corrcoef([true_off[t] for t in teams], [fit["off"][t] for t in teams])[0, 1]
        print(f"\n  mean abs error : {np.mean(errs):.4f}")
        print(f"  correlation    : {r:.3f}   ({'OK' if r > 0.9 else 'FAIL'} — "
              f"ridge should recover known strengths)")
