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
    """Weighted ridge via normal equations. The intercept is not penalised.

    Guards against non-finite input: a single NaN or inf EPA propagates through
    the matmul and silently poisons every coefficient in the fit.
    """
    finite = np.isfinite(y) & np.isfinite(w) & np.isfinite(X).all(axis=1)
    if not finite.all():
        X, y, w = X[finite], y[finite], w[finite]
    if len(y) == 0:
        return np.zeros(X.shape[1])
    # Guard against underflow to zero from long decay chains
    w = np.clip(w, 1e-6, None)

    Xw = X * w[:, None]
    A = X.T @ Xw
    A[np.diag_indices_from(A)] += alpha
    A[-1, -1] -= alpha                      # last column is the intercept
    b = X.T @ (w * y)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)


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
                 play_class: str = "all", *, shrink_to_prior: bool = True) -> dict | None:
    """What we knew about a team going into a given week. The point-in-time read.

    **`shrink_to_prior` defaults on, and the default is the interesting part.** Without
    it, a Week 2 rating is computed from one game and reported with the same authority as
    a Week 15 rating computed from fourteen — and a *missing* rating becomes zero
    downstream, which reads as "exactly league average" rather than "we don't know".

    With it, the rating is blended toward last season's regressed rating in proportion to
    how little has been observed: `(n/(n+k))*observed + (k/(n+k))*prior`. `k` is fitted
    per play class by walk-forward (`research.coldstart fit`) and beats the unshrunk
    rating on a held-out season by 7-15% MAE.

    Pass `shrink_to_prior=False` to see the raw within-season number — which is what you
    want when debugging the ridge fit itself, and never what you want as a model feature
    before about Week 8.
    """
    con = connect(read_only=True)
    row = con.execute("""
        SELECT off_rating, def_rating, off_plays, def_plays, confident
        FROM unit_ratings
        WHERE season=? AND week=? AND team=? AND play_class=?
    """, [season, week, team, play_class]).fetchone()

    out: dict | None = None
    if row:
        out = {"off_rating": row[0], "def_rating": row[1], "off_plays": row[2],
               "def_plays": row[3], "confident": bool(row[4]), "shrunk": False}

        if shrink_to_prior:
            prior = con.execute("""
                SELECT off_prior, def_prior FROM team_priors
                WHERE season=? AND team=? AND play_class=?
            """, [season, team, play_class]).fetchone()
            if prior is not None:
                # Imported here rather than at module scope: `coldstart` imports from
                # this module's package and a top-level import would be circular.
                from research.coldstart import k_for, shrink

                k = k_for(play_class)
                out["off_rating_raw"] = out["off_rating"]
                out["def_rating_raw"] = out["def_rating"]
                out["off_rating"] = shrink(row[0], row[2] or 0, prior[0], k)
                out["def_rating"] = shrink(row[1], row[3] or 0, prior[1], k)
                out["shrunk"] = True
                out["prior_weight"] = round(k / ((row[2] or 0) + k), 3)
            # No prior row — an expansion team, or the first season in the data. The
            # raw rating stands and `shrunk` stays False, so a caller can tell the
            # difference between "blended" and "nothing to blend with".
    con.close()
    return out


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
