"""Player prop projections — distributional, volume-first.

Why this is structured differently from the game models
-------------------------------------------------------
A prop is a threshold bet: you need P(X > line), not E[X]. Two receivers with
the same projected 62 yards and different variance carry genuinely different
fair prices, and a point estimate cannot express that.

So every projection here outputs a DISTRIBUTION, assembled as:

    yards = volume × efficiency

modelled separately because they behave completely differently:

* **Volume is predictable.** Target share and carry share are among the most
  stable quantities in football — role changes slowly, and usage is a coaching
  decision rather than a coin flip.
* **Efficiency is noisy.** Yards per target swings wildly game to game. The
  honest move is to estimate its *distribution* from a player's history and
  the matchup, not to predict its value.

Getting this order right matters more than any single feature. A model that
predicts volume well and treats efficiency as a distribution beats one that
predicts total yards directly, because it separates the part you can know from
the part you can't.

LOCAL ONLY — never imported by server.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.warehouse import connect

log = logging.getLogger(__name__)

# Usage stabilises fast; efficiency barely does. These shrinkage constants say
# how many games before a player's own rate outweighs the positional prior —
# fitted values should replace these guesses once there's walk-forward evidence.
K_TARGET_SHARE = 4.0
K_CARRY_SHARE = 4.0
K_YPT = 12.0          # yards per target: slow, noisy
K_YPC = 16.0          # yards per carry: slower still

MIN_GAMES = 3

# Minimum projection worth evaluating, PER STAT. Using a single yardage-scale
# threshold silently discarded every receptions projection (they run 4-6), which
# is why that stat produced nothing at all.
MIN_MEAN = {"rec_yards": 20.0, "rush_yards": 20.0, "receptions": 2.0,
            "pass_yards": 150.0}


@dataclass
class PropProjection:
    player_id: str
    stat: str
    volume_mean: float
    volume_sd: float
    eff_mean: float
    eff_sd: float
    mean: float
    sd: float
    n_games: int
    confident: bool
    # Probability of a "bust" game — injured early, benched, buried by game
    # script. These produce near-zero outcomes that a single lognormal cannot
    # accommodate, which showed up as the bottom decile carrying twice its
    # expected mass. Modelling participation separately from performance is the
    # standard fix and it matters commercially: unders on a star are largely a
    # bet on this probability, not on his yards-per-target.
    p_bust: float = 0.0
    bust_mean: float = 0.0

    def expected(self) -> float:
        """Unconditional expectation across BOTH components.

        `mean` is the non-bust component only. Comparing an actual outcome
        against `mean` double-counts the bust adjustment — the mixture already
        places mass near zero — which is what drove the bias correction to an
        absurd -6.3 yards. Anything measuring error must use this.
        """
        return self.p_bust * self.bust_mean + (1 - self.p_bust) * self.mean

    def _lognormal_params(self) -> tuple[float, float]:
        var = self.sd ** 2
        mu = math.log(self.mean ** 2 / math.sqrt(var + self.mean ** 2))
        sigma = math.sqrt(math.log(1 + var / self.mean ** 2))
        return mu, sigma

    def _normal_cdf(self, x: float, mean: float, sd: float) -> float:
        if sd <= 0:
            return 1.0 if x >= mean else 0.0
        mu = math.log(mean ** 2 / math.sqrt(sd ** 2 + mean ** 2))
        sigma = math.sqrt(math.log(1 + sd ** 2 / mean ** 2))
        z = (math.log(max(x, 1e-9)) - mu) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    def cdf(self, x: float) -> float:
        """P(X <= x) — a two-component mixture.

            P(X <= x) = p_bust · F_bust(x) + (1 - p_bust) · F_normal(x)

        The bust component carries the mass near zero that a single lognormal
        pushes into the body of the distribution, which is what inflated the
        bottom decile and starved the middle.
        """
        if self.sd <= 0 or self.mean <= 0:
            return 0.5
        if x <= 0:
            return 0.0
        normal_part = self._normal_cdf(x, self.mean, self.sd)
        if self.p_bust <= 0:
            return normal_part
        bust_part = self._normal_cdf(x, max(self.bust_mean, 0.5),
                                     max(self.bust_mean, 0.5) * 0.9)
        return self.p_bust * bust_part + (1 - self.p_bust) * normal_part

    def prob_over(self, line: float) -> float:
        """P(X > line).

        Yardage is right-skewed — a receiver's floor is zero but the ceiling is
        open — so a lognormal fit beats a normal one, particularly in the tails
        where props are actually priced.
        """
        if self.sd <= 0 or self.mean <= 0:
            return 0.5
        if line <= 0:
            return 1.0
        return 1.0 - self.cdf(line)

    def fair_line(self, prob: float = 0.5) -> float:
        """The line at which this projection is a coin flip — comparable
        directly against what a book has posted."""
        lo, hi = 0.0, max(self.mean * 5, 10.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.prob_over(mid) > prob:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 1)


def _shrink(observed: float, prior: float, n: int, k: float) -> float:
    """Blend a player's own rate toward the positional prior by sample size.

    Early in a season a three-game target share is mostly noise; this is what
    stops a receiver with two big games being projected as a WR1.
    """
    return (n * observed + k * prior) / (n + k) if (n + k) > 0 else prior


def build_player_history(min_season: int = 2016) -> dict[str, Any]:
    """Per-player, per-week usage and efficiency, ordered in time.

    Point-in-time by construction: every row carries only what happened in that
    game, and projections consume strictly earlier rows.
    """
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT pw.season, pw.week, pw.game_id, pw.team, pw.player_id,
               pw.targets, pw.carries, pw.air_yards,
               pw.target_share, pw.carry_share,
               pw.epa_per_target, pw.epa_per_carry
        FROM player_weeks pw
        WHERE pw.season >= ? AND pw.player_id IS NOT NULL
        ORDER BY pw.player_id, pw.season, pw.week
    """, [min_season]).fetchall()

    # Actual outcomes. nflverse renames stat columns across releases, so resolve
    # them from what's present rather than hardcoding — a missing column here
    # silently produces zero projections, which is how `receptions` failed.
    stat_idx: dict[tuple, dict] = {}
    try:
        cols = {d[0] for d in con.execute(
            "SELECT * FROM stats_player LIMIT 0").description}

        def pick(*cands: str) -> str | None:
            return next((c for c in cands if c in cols), None)

        mapping = {
            "rec_yards": pick("receiving_yards", "rec_yards", "receiving_yds"),
            "receptions": pick("receptions", "rec", "receptions_total"),
            "rush_yards": pick("rushing_yards", "rush_yards", "rushing_yds"),
            "pass_yards": pick("passing_yards", "pass_yards", "passing_yds"),
        }
        pid = pick("player_id", "gsis_id")
        missing = [k for k, v in mapping.items() if not v]
        if missing:
            log.warning("stats_player has no column for: %s (available sample: %s)",
                        ", ".join(missing), sorted(cols)[:12])

        sel = ", ".join(f"{v} AS {k}" for k, v in mapping.items() if v)
        if pid and sel:
            rows_s = con.execute(
                f"SELECT {pid} AS player_id, season, week, {sel} "
                f"FROM stats_player WHERE season >= ?", [min_season]).fetchall()
            names = ["player_id", "season", "week"] + [k for k, v in mapping.items() if v]
            for r in rows_s:
                d = dict(zip(names, r))
                stat_idx[(d["player_id"], d["season"], d["week"])] = d
    except Exception as exc:  # noqa: BLE001
        log.warning("stats_player unreadable (%s) — no outcome data", exc)
    con.close()

    by_player: dict[str, list[dict]] = {}
    for r in rows:
        s = stat_idx.get((r[4], r[0], r[1])) or {}
        by_player.setdefault(r[4], []).append({
            "season": r[0], "week": r[1], "game_id": r[2], "team": r[3],
            "targets": r[5] or 0, "carries": r[6] or 0, "air_yards": r[7] or 0,
            "target_share": r[8], "carry_share": r[9],
            "rec_yards": s.get("rec_yards"),
            "receptions": s.get("receptions"),
            "rush_yards": s.get("rush_yards"),
        })
    log.info("player history: %d players, %d player-weeks",
             len(by_player), sum(len(v) for v in by_player.values()))
    return by_player


def project(history: list[dict], stat: str, player_id: str,
            priors: dict[str, float] | None = None) -> PropProjection | None:
    """Project one stat from a player's PRIOR games only.

    `history` must already be truncated to games before the one being projected —
    the caller owns that cut, so this function cannot leak future data.
    """
    priors = priors or {}
    n = len(history)
    if n < MIN_GAMES:
        return None

    vol_key = {"rec_yards": "targets", "rush_yards": "carries",
               "receptions": "targets"}[stat]
    if stat == "rec_yards":
        effs_all = [(h["rec_yards"] / h["targets"])
                    for h in history if h["targets"] and h["rec_yards"] is not None]
        k_eff, prior_eff = K_YPT, priors.get("ypt", 7.8)
    elif stat == "rush_yards":
        effs_all = [(h["rush_yards"] / h["carries"])
                    for h in history if h["carries"] and h["rush_yards"] is not None]
        k_eff, prior_eff = K_YPC, priors.get("ypc", 4.3)
    elif stat == "receptions":
        effs_all = [(h["receptions"] / h["targets"])
                    for h in history if h["targets"] and h["receptions"] is not None]
        k_eff, prior_eff = 10.0, priors.get("catch_rate", 0.65)
    else:
        return None

    all_vols = [h[vol_key] or 0 for h in history]
    if not all_vols:
        return None

    # Split participation from performance. A "bust" is a game with volume far
    # below the player's own norm — injury, benching, game script. Fitting the
    # main distribution to these mixed together is what pushed excess mass into
    # the bottom decile.
    typical = float(np.median([v for v in all_vols if v > 0]) or 0)
    bust_cut = max(1.0, typical * 0.35)
    normal_vols = [v for v in all_vols if v >= bust_cut]
    bust_vols = [v for v in all_vols if v < bust_cut]

    p_bust = len(bust_vols) / len(all_vols) if all_vols else 0.0
    # Shrink toward a positional base rate: three quiet games early in a season
    # shouldn't imply a 40% bust probability.
    p_bust = _shrink(p_bust, priors.get("bust_rate", 0.12), len(all_vols), 6.0)

    vols = normal_vols or all_vols
    vol_mean = float(np.mean(vols))
    vol_sd = float(np.std(vols, ddof=1)) if len(vols) > 1 else vol_mean * 0.5
    effs = effs_all

    if effs:
        eff_raw = float(np.mean(effs))
        eff_sd = float(np.std(effs, ddof=1)) if len(effs) > 1 else eff_raw * 0.4
        eff_mean = _shrink(eff_raw, prior_eff, len(effs), k_eff)
    else:
        eff_mean, eff_sd = prior_eff, prior_eff * 0.45

    mean = vol_mean * eff_mean
    # Volume and efficiency are near-independent, so variance compounds:
    #   Var(XY) = Var(X)Var(Y) + Var(X)E[Y]^2 + Var(Y)E[X]^2
    # Ignoring this and using only volume variance understates spread badly,
    # which is exactly the error that misprices the tails of a prop ladder.
    var = (vol_sd**2 * eff_sd**2) + (vol_sd**2 * eff_mean**2) + (eff_sd**2 * vol_mean**2)
    sd = float(np.sqrt(max(var, 1e-9)))

    # np.mean on an empty list warns and returns nan — guard explicitly.
    bust_mean = (float(np.mean(bust_vols)) * eff_mean
                 if len(bust_vols) > 0 else mean * 0.15)
    return PropProjection(
        player_id=player_id, stat=stat,
        volume_mean=round(vol_mean, 2), volume_sd=round(vol_sd, 2),
        eff_mean=round(eff_mean, 3), eff_sd=round(eff_sd, 3),
        mean=round(mean, 1), sd=round(sd, 1),
        n_games=n, confident=n >= 6,
        p_bust=round(min(p_bust, 0.45), 4),
        bust_mean=round(max(bust_mean, 0.3), 2))


@dataclass
class Calibration:
    """Empirical correction fitted from past seasons.

    Two parameters, both necessary:

    `bias`    projections systematically ran ~3.5 yards high on receiving. The
              cause is structural: a player's historical average includes their
              healthy games, but the projected week may be one where they leave
              early or see reduced snaps. That asymmetry does not cancel out.

    `sd_scale` the raw compounded variance was too wide — 80% of actuals landed
              inside one sd where 68% should. Compounding volume and efficiency
              variance double-counts, because game-to-game efficiency swings
              already embed volume effects. Scaling by the observed z spread
              fixes it directly rather than guessing at the covariance.
    """
    stat: str
    bias: float = 0.0
    sd_scale: float = 1.0
    n: int = 0

    def apply(self, p: PropProjection) -> PropProjection:
        return PropProjection(
            player_id=p.player_id, stat=p.stat,
            volume_mean=p.volume_mean, volume_sd=p.volume_sd,
            eff_mean=p.eff_mean, eff_sd=p.eff_sd,
            mean=max(0.1, p.mean + self.bias),
            sd=max(0.1, p.sd * self.sd_scale),
            n_games=p.n_games, confident=p.confident,
            p_bust=p.p_bust, bust_mean=p.bust_mean)


def fit_calibration(stat: str, hist: dict[str, list[dict]],
                    through_season: int, min_mean: float = 20.0) -> Calibration:
    """Fit bias and spread correction on seasons STRICTLY BEFORE the target.

    Calibrating on the same data you evaluate would be exactly the leak the
    validation suite exists to catch.
    """
    key = {"rec_yards": "rec_yards", "rush_yards": "rush_yards",
           "receptions": "receptions"}[stat]
    errs, zs = [], []
    for pid, games in hist.items():
        for i, g in enumerate(games):
            if g["season"] >= through_season:
                continue
            act = g.get(key)
            if act is None:
                continue
            proj = project(games[:i], stat, pid)
            if not proj or proj.expected() < min_mean or proj.sd <= 0:
                continue
            errs.append(act - proj.expected())
            zs.append((act - proj.expected()) / proj.sd)
    if len(errs) < 200:
        return Calibration(stat=stat)
    return Calibration(stat=stat, bias=float(np.mean(errs)),
                       sd_scale=float(np.std(zs)), n=len(errs))


def backtest(stat: str = "rec_yards", min_season: int = 2018,
             lookback: int = 8, min_mean: float = 20.0,
             calibrate: bool = True) -> dict:
    """Walk forward through every player-week and score the projections.

    Two things are measured, and the second matters more:

      accuracy    — is the centre right? (MAE, bias)
      calibration — is the SPREAD right? A projection claiming 62 ± 25 should
                    have the actual land inside one sd about 68% of the time.
                    Get this wrong and every P(over) is wrong, however good the
                    mean is.
    """
    hist = build_player_history(min_season)
    actual_key = {"rec_yards": "rec_yards", "rush_yards": "rush_yards",
                  "receptions": "receptions"}[stat]

    min_mean = MIN_MEAN.get(stat, min_mean)
    errs, z_scores, preds, actuals = [], [], [], []
    over_at_mean, pit_vals = [], []

    # One calibration per test season, fitted only on earlier ones.
    seasons = sorted({g["season"] for gs in hist.values() for g in gs})
    calibs: dict[int, Calibration] = {}
    if calibrate:
        for s in seasons:
            if s <= min_season:
                continue
            calibs[s] = fit_calibration(stat, hist, through_season=s,
                                        min_mean=min_mean)

    for pid, games in hist.items():
        for i in range(len(games)):
            g = games[i]
            act = g.get(actual_key)
            if act is None:
                continue
            proj = project(games[:i], stat, pid)      # STRICTLY prior games
            if not proj or proj.expected() < min_mean or proj.sd <= 0:
                continue
            cal = calibs.get(g["season"])
            if cal is not None:
                proj = cal.apply(proj)
                if proj.sd <= 0:
                    continue
            exp = proj.expected()
            errs.append(act - exp)
            z_scores.append((act - exp) / proj.sd)
            preds.append(exp)
            actuals.append(act)
            pit_vals.append(proj.cdf(act))
            # if the projection is honest, the actual should beat its own
            # median line about half the time
            over_at_mean.append(1 if act > proj.fair_line(0.5) else 0)

    if not errs:
        return {"error": "no projections produced — check stats_player columns"}

    errs = np.array(errs); z = np.array(z_scores)
    within_1sd = float(np.mean(np.abs(z) <= 1.0))
    within_2sd = float(np.mean(np.abs(z) <= 2.0))

    # PIT — the correct test for a distributional forecast.
    #
    # Feed each actual outcome through its own predicted CDF. If the
    # distributions are honest, those values are uniform on [0,1]: 10% of
    # outcomes land below the 10th percentile, and so on. The z-score test
    # assumes normality and therefore measures the wrong thing on a skewed
    # distribution — which is why within_1sd and z_sd disagreed.
    pit = np.array(pit_vals)
    deciles = [float(np.mean((pit >= i / 10) & (pit < (i + 1) / 10)))
               for i in range(10)]
    # Kolmogorov-Smirnov distance from uniform: one number for "how wrong"
    srt = np.sort(pit)
    n_pit = len(srt)
    ks = float(np.max(np.abs(srt - np.arange(1, n_pit + 1) / n_pit))) if n_pit else 1.0

    return {
        "stat": stat, "n": len(errs),
        "mae": round(float(np.mean(np.abs(errs))), 2),
        "bias": round(float(np.mean(errs)), 2),
        "corr": round(float(np.corrcoef(preds, actuals)[0, 1]), 3),
        # distributional calibration — what actually prices a prop
        "pit_deciles": [round(d, 3) for d in deciles],
        "pit_ks": round(ks, 4),
        "pit_mean": round(float(np.mean(pit)), 3),
        # normal-approximation diagnostics, kept for comparison
        "within_1sd": round(within_1sd, 3),
        "within_2sd": round(within_2sd, 3),
        "over_rate_at_own_line": round(float(np.mean(over_at_mean)), 3),
        "z_sd": round(float(np.std(z)), 3),
        "calibrations": {s: {"bias": round(c.bias, 2),
                             "sd_scale": round(c.sd_scale, 3), "n": c.n}
                         for s, c in sorted(calibs.items())} if calibrate else {},
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="prop projections (local only)")
    p.add_argument("command", choices=["backtest", "player", "demo"])
    p.add_argument("--stat", default="rec_yards")
    p.add_argument("--player", default="")
    p.add_argument("--min-season", type=int, default=2018)
    p.add_argument("--raw", action="store_true", help="skip calibration")
    args = p.parse_args()

    if args.command == "backtest":
        for stat in (["rec_yards", "rush_yards", "receptions"]
                     if args.stat == "all" else [args.stat]):
            r = backtest(stat, min_season=args.min_season,
                         calibrate=not args.raw)
            if "error" in r:
                print(f"  {stat}: {r['error']}")
                continue
            print(f"\n=== {stat} — {r['n']:,} projections ===")
            print(f"  MAE                  {r['mae']}")
            print(f"  bias                 {r['bias']:+}  (want ~0)")
            print(f"  corr(pred, actual)   {r['corr']}")
            print(f"  over rate at own line {r['over_rate_at_own_line']}   (want ~0.50)")

            print("\n  DISTRIBUTIONAL CALIBRATION (what prices a prop)")
            print(f"  PIT deciles: each should be ~0.100")
            bars = "".join(f"{d:>6.3f}" for d in r["pit_deciles"])
            print(f"   {bars}")
            worst = max(abs(d - 0.1) for d in r["pit_deciles"])
            print(f"  max decile error     {worst:.3f}   (want < 0.02)")
            print(f"  KS distance          {r['pit_ks']}   (want < 0.03)")
            print(f"  PIT mean             {r['pit_mean']}   (want ~0.50)")
            verdict = ("well calibrated" if r["pit_ks"] < 0.03 else
                       "usable, some skew" if r["pit_ks"] < 0.06 else
                       "MISCALIBRATED — probabilities not trustworthy")
            print(f"  -> {verdict}")
            print(f"\n  (normal-approx diagnostics, for reference: within1sd "
                  f"{r['within_1sd']}, z_sd {r['z_sd']})")
            if r.get("calibrations"):
                print("  calibration fitted per season (from EARLIER seasons only):")
                for s, c in list(r["calibrations"].items())[-4:]:
                    print(f"    {s}: bias {c['bias']:+6.2f}  sd x{c['sd_scale']:.3f}  "
                          f"(n={c['n']:,})")
    elif args.command == "player":
        hist = build_player_history(args.min_season)
        games = hist.get(args.player)
        if not games:
            raise SystemExit(f"no history for {args.player}")
        proj = project(games[:-1], args.stat, args.player)
        print(f"  {args.player} — {args.stat} from {len(games)-1} prior games")
        print(f"  volume  {proj.volume_mean} ± {proj.volume_sd}")
        print(f"  eff     {proj.eff_mean} ± {proj.eff_sd}")
        print(f"  proj    {proj.mean} ± {proj.sd}")
        print(f"  median line {proj.fair_line(0.5)}")
        for ln in [proj.mean * 0.8, proj.mean, proj.mean * 1.2]:
            print(f"    P(over {ln:.1f}) = {proj.prob_over(ln):.3f}")
    else:
        pr = PropProjection("demo", "rec_yards", 7.0, 2.5, 8.5, 3.0, 59.5, 28.0, 10, True)
        print(f"  projection {pr.mean} ± {pr.sd}")
        print(f"  fair (median) line: {pr.fair_line(0.5)}")
        for ln in (45.5, 52.5, 59.5, 66.5, 75.5):
            print(f"    P(over {ln}) = {pr.prob_over(ln):.3f}")
