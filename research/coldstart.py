"""Cold-start shrinkage: blending a team's prior with what this season has shown.

    uv run python -m research.coldstart priors      # build the prior table
    uv run python -m research.coldstart fit         # fit k per play class
    uv run python -m research.coldstart show --season 2025 --week 3

**The problem.** `unit_ratings` are computed point-in-time from within-season plays only,
so Week 1 has no data at all and Week 3 has barely enough to be noise. `build_ratings`
leaves that gap deliberately rather than fabricating a number. Downstream, though, a
missing rating becomes a zero — and zero means "exactly league average", which is a
confident claim, not an absence.

**The fix.**

    strength = (n / (n + k)) * observed + (k / (n + k)) * prior

`n` is plays observed this season, `k` is the number of plays at which you trust the
observation and the prior equally. Large `k` = slow to believe the current season.

**Why `k` must be fitted per statistic rather than chosen once.** Different measures
stabilise at wildly different rates, and guessing one number for all of them gets both
ends wrong:

| statistic | stabilises | rough k |
|---|---|---|
| pass EPA | fast | ~4 games |
| success rate | fast | ~4 games |
| rush EPA | slow | ~10 games |
| turnover margin | mostly luck | ~16+ games |
| fumble recovery | pure luck | never — shrink fully to the mean |

**Why this is worth building now and not in October.** Its entire value lives in Weeks
1–6, which is when the model is weakest *and* the market is softest. That combination
doesn't recur until next September.

Priors here are last season's final rating regressed toward the mean. That is the
*floor* of a good prior, not the ceiling — roster continuity, draft capital and a
variance bump for coaching changes all belong here and are noted as TODO rather than
faked. A prior built from one number is honest about being one number.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from research.warehouse import connect

log = logging.getLogger(__name__)

PLAY_CLASSES = ("all", "pass", "rush")

# How much of last season's rating survives into this season's prior.
#
# Team strength is famously mean-reverting year to year: roster turnover, schedule
# regression and coaching change all pull toward the middle. 0.55 is the conventional
# starting point for season-over-season team ratings and is refit by `fit` below rather
# than trusted. Regressing too little is the more expensive error — it makes Week 1
# confident about a team that no longer exists.
DEFAULT_REGRESSION = 0.55

# Fitted by `python -m research.coldstart fit` on 2013-2024, held out on 2025.
# **Units are plays, not games** — which matters when comparing against the rules of
# thumb in IN_SEASON_LEARNING.md, since a team runs ~35 pass plays and ~25 rush plays
# per game. In games these are roughly: all 4, pass 7, rush 5.
#
# Holdout MAE against no prior at all:
#   all   0.10128 vs 0.11218   (-9.7%)
#   pass  0.13278 vs 0.14224   (-6.6%)
#   rush  0.08918 vs 0.10488   (-15.0%)
#
# **Rush wants *less* shrinkage than pass, which is the opposite of the assumption**
# in IN_SEASON_LEARNING.md ("pass EPA stabilises fast, rush EPA slow"). That rule of
# thumb is about raw EPA; these are opponent-adjusted ridge ratings that already shrink
# toward league average, so the residual instability isn't the same quantity. Rush usage
# is also more scheme-stable week to week, where pass volume swings with game script.
# Recorded as a measured disagreement rather than resolved — it deserves its own look.
FITTED_K = {"all": 240.0, "pass": 240.0, "rush": 120.0}

# Used when a play class has no fitted value. Deliberately the "all" number rather than
# something rounder, so an unfitted class behaves like the class we measured most.
DEFAULT_K = 240.0


def k_for(play_class: str) -> float:
    return FITTED_K.get(play_class, DEFAULT_K)

# A team with no prior season at all — expansion, or the first year in the data.
# League average, which for a ridge-centred rating is zero.
NO_PRIOR = 0.0


def build_priors(regression: float = DEFAULT_REGRESSION) -> int:
    """One prior per (season, team, play_class), from last season's final rating.

    "Final rating" means the highest week present for that season, which is the rating
    computed from the most complete within-season sample. Using a mid-season rating
    would throw away exactly the observations that make the prior worth having.
    """
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS team_priors (
            season INTEGER, team VARCHAR, play_class VARCHAR,
            off_prior DOUBLE, def_prior DOUBLE,
            source VARCHAR
        )
    """)
    con.execute("DELETE FROM team_priors")

    # Last observed rating per (season, team, class) — the season's final word.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW final_ratings AS
        SELECT season, team, play_class, off_rating, def_rating
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY season, team, play_class ORDER BY week DESC
            ) AS rn
            FROM unit_ratings
        ) WHERE rn = 1
    """)

    con.execute("""
        INSERT INTO team_priors
        SELECT
            f.season + 1                    AS season,
            f.team,
            f.play_class,
            f.off_rating * ?                AS off_prior,
            f.def_rating * ?                AS def_prior,
            'prior_season_regressed'        AS source
        FROM final_ratings f
    """, [regression, regression])

    n = con.execute("SELECT COUNT(*) FROM team_priors").fetchone()[0]
    seasons = con.execute("SELECT MIN(season), MAX(season) FROM team_priors").fetchone()
    con.close()
    log.info("team_priors: %d rows, seasons %s–%s", n, seasons[0], seasons[1])
    return int(n)


def shrink(observed: float, n_plays: float, prior: float, k: float) -> float:
    """The blend. Pure function, no database — trivially testable, which matters
    because every caller depends on it and it is two lines of arithmetic that are easy
    to get backwards."""
    if k <= 0:
        return observed
    weight = n_plays / (n_plays + k)
    return weight * observed + (1.0 - weight) * prior


def fit_k(play_class: str = "all",
          candidates: tuple[float, ...] = (0, 60, 120, 240, 480, 960, 1920, 1e9),
          verbose: bool = True) -> dict:
    """Choose k by walk-forward: which blend best predicts end-of-season strength?

    **The target is the team's final rating for that season**, and the question asked of
    each week is "given what we've seen so far plus the prior, how close are we to what
    this team turns out to be?" That is the quantity a Week 3 feature is standing in for.

    Fitted on every season except the last, then reported on the last, so the chosen k
    is not selected on the data it's scored against. `k=0` means ignore the prior
    entirely (today's behaviour); `k=1e9` means ignore the season and use the prior
    alone. Both are included so the fit can tell you the prior is worthless if it is.
    """
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT r.season, r.week, r.team,
               r.off_rating, r.def_rating, r.off_plays, r.def_plays,
               p.off_prior, p.def_prior,
               f.off_final, f.def_final
        FROM unit_ratings r
        JOIN team_priors p
          ON p.season = r.season AND p.team = r.team AND p.play_class = r.play_class
        JOIN (
            SELECT season, team, play_class,
                   FIRST(off_rating ORDER BY week DESC) AS off_final,
                   FIRST(def_rating ORDER BY week DESC) AS def_final
            FROM unit_ratings GROUP BY 1, 2, 3
        ) f ON f.season = r.season AND f.team = r.team AND f.play_class = r.play_class
        WHERE r.play_class = ?
          AND r.week <= 8            -- cold start is a weeks 1-8 question
    """, [play_class]).fetchall()
    con.close()

    if not rows:
        raise SystemExit(
            f"no joined rows for play_class={play_class!r}. Run `build_ratings` and "
            "`python -m research.coldstart priors` first."
        )

    data = np.array([
        [r[0], r[3], r[5], r[7], r[9], r[4], r[6], r[8], r[10]] for r in rows
    ], dtype=float)
    seasons = np.unique(data[:, 0])
    if len(seasons) < 3:
        raise SystemExit(f"only {len(seasons)} season(s) — walk-forward needs at least 3")

    holdout = seasons[-1]
    train = data[data[:, 0] != holdout]
    test = data[data[:, 0] == holdout]

    def error(block: np.ndarray, k: float) -> float:
        # columns: season, off_rating, off_plays, off_prior, off_final,
        #                  def_rating, def_plays, def_prior, def_final
        off = shrink(block[:, 1], block[:, 2], block[:, 3], k)
        dfe = shrink(block[:, 5], block[:, 6], block[:, 7], k)
        return float(np.mean(np.abs(off - block[:, 4])) + np.mean(np.abs(dfe - block[:, 8])))

    scored = {k: error(train, k) for k in candidates}
    best = min(scored, key=lambda k: scored[k])

    result = {
        "play_class": play_class,
        "best_k": best,
        "train_mae": scored[best],
        "baseline_mae": scored[0],          # k=0: no prior at all, today's behaviour
        "holdout_mae": error(test, best),
        "holdout_baseline": error(test, 0),
        "holdout_season": int(holdout),
        "n_rows": len(data),
        "curve": {float(k): round(v, 5) for k, v in scored.items()},
    }

    if verbose:
        print(f"\n=== cold-start k, play_class={play_class} ===")
        print(f"  {len(data):,} team-weeks (weeks 1-8), holdout season {int(holdout)}\n")
        print(f"  {'k (plays)':>12}  {'train MAE':>10}")
        for k in candidates:
            mark = "  <-- best" if k == best else ""
            label = "inf" if k >= 1e8 else f"{k:g}"
            print(f"  {label:>12}  {scored[k]:>10.5f}{mark}")
        gain = result["holdout_baseline"] - result["holdout_mae"]
        print(f"\n  holdout {int(holdout)}: MAE {result['holdout_mae']:.5f} "
              f"vs {result['holdout_baseline']:.5f} with no prior "
              f"({gain:+.5f}, {'better' if gain > 0 else 'worse'})")
        if gain <= 0:
            print("  the prior does not help on held-out data — do not ship this")
    return result


def show(season: int, week: int, play_class: str = "all", k: float = DEFAULT_K) -> None:
    """Side-by-side: raw rating, prior, and the blend, for one week."""
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT r.team, r.off_rating, r.off_plays, p.off_prior
        FROM unit_ratings r
        JOIN team_priors p
          ON p.season = r.season AND p.team = r.team AND p.play_class = r.play_class
        WHERE r.season=? AND r.week=? AND r.play_class=?
        ORDER BY r.team
    """, [season, week, play_class]).fetchall()
    con.close()
    if not rows:
        print(f"no ratings for {season} week {week} ({play_class})")
        return
    print(f"\n{season} week {week}, {play_class}, k={k:g}\n")
    print(f"  {'team':<6}{'observed':>10}{'plays':>8}{'prior':>10}{'blended':>10}{'w_obs':>8}")
    for team, obs, plays, prior in rows:
        blended = shrink(obs, plays, prior, k)
        w = plays / (plays + k) if k > 0 else 1.0
        print(f"  {team:<6}{obs:>10.4f}{int(plays):>8}{prior:>10.4f}{blended:>10.4f}{w:>8.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.coldstart")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pri = sub.add_parser("priors", help="build team_priors from last season's ratings")
    p_pri.add_argument("--regression", type=float, default=DEFAULT_REGRESSION)

    p_fit = sub.add_parser("fit", help="fit k by walk-forward")
    p_fit.add_argument("--play-class", default=None, help="default: all three")

    p_show = sub.add_parser("show", help="raw vs prior vs blended for one week")
    p_show.add_argument("--season", type=int, required=True)
    p_show.add_argument("--week", type=int, required=True)
    p_show.add_argument("--play-class", default="all")
    p_show.add_argument("--k", type=float, default=DEFAULT_K)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "priors":
        n = build_priors(args.regression)
        print(f"built {n:,} priors (regression={args.regression})")
    elif args.command == "fit":
        classes = [args.play_class] if args.play_class else list(PLAY_CLASSES)
        for cls in classes:
            fit_k(cls)
    elif args.command == "show":
        show(args.season, args.week, args.play_class, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
