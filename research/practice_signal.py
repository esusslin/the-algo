"""Does practice status carry signal beyond the game-status designation?

    python -m research.practice_signal probe      # what data do we actually have?
    python -m research.practice_signal run        # the test
    python -m research.practice_signal run --position RB

Why this exists
---------------
Omaha's entire value proposition is one unverified assumption: that the practice
trajectory tells you something the final designation doesn't. "Limited Wednesday, full
Thursday" is supposed to beat "Questionable".

That has never been measured. Before spending a season building an extraction pipeline
to capture it, this bounds the upside using data already on disk.

What this can and cannot show
-----------------------------
`injuries_hist` is one row per player-week, so it holds the *final* practice status, not
the Wed -> Thu -> Fri progression. This therefore tests a weaker claim:

    Among players on the injury report, does knowing their practice participation
    improve a forecast that already knows their game-status designation?

If the answer is a clear no, the daily trajectory is very unlikely to help either — it's
a finer-grained version of the same signal, and Omaha should be treated as a portfolio
project rather than a data source. If the answer is yes, this is the floor and the
trajectory is upside on top of a measured effect.

Design notes
------------
**Two outcomes, because props decompose that way.** `P(active)` first — a player who
doesn't take the field scores zero on every prop, so this is the larger lever. Then
usage share given active, which drives the volume layer.

**The ambiguous cases are the whole point.** Over all report-listed players, `report_status`
alone will look excellent, because "Out" means out. That result is real and useless. The
decision that costs money is Questionable, so results are broken out that way.

**Walk-forward, never a random split.** Train on seasons before S, test on S. A random
split would let the model see 2024 while predicting 2019 and every number would be a lie
— the same leakage discipline as everywhere else in this repo.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.warehouse import connect

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")

# "Probable" was retired after 2015 but is all over the older files — 5,551 skill-position
# rows. Left in its own bucket rather than folded into Questionable, since the vocabulary
# genuinely changed and merging them would blur a regime shift into the middle seasons.
STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Probable": 3}
NO_DESIGNATION = 4


def normalise_practice(value) -> str | None:
    """Map nflverse's practice wording onto DNP / LIMITED / FULL.

    The raw values are sentences — "Full Participation in Practice", "Did Not
    Participate In Practice" — not the short codes the club PDFs use. An exact-match
    dict against {"DNP","Limited","Full"} silently maps every row to NaN, which makes
    the treatment arm identical to the baseline and produces a confident delta of zero.
    That is a false negative on the one question this file exists to answer, so the
    mapping is substring-based and asserted on below.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    if "did not" in text or text in {"dnp", "out"}:
        return "DNP"
    if "limited" in text or text == "lp":
        return "LIMITED"
    if "full" in text or text == "fp":
        return "FULL"
    return None


PRACTICE_ORDER = {"DNP": 0, "LIMITED": 1, "FULL": 2}

MIN_TRAIN_SEASONS = 5


# --- metrics, hand-rolled: sklearn isn't in requirements-research.txt ---------------


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U formulation. Ties get average rank, which is the correct handling."""
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


# --- data --------------------------------------------------------------------------


def _columns(con, table: str) -> set[str]:
    return {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}


def _has_table(con, name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False


def _name_to_id(con) -> dict[str, str]:
    """Normalised full name -> gsis id, from the players crosswalk.

    `injuries_hist` carries `gsis_id` on only a small minority of rows — the first run
    of this test silently dropped ~94% of the table to the join. Names are what the
    injury feed reliably has, so they're the fallback. Ambiguous names are excluded
    rather than guessed: attributing a practice status to the wrong player is worse
    than dropping the row.
    """
    if not _has_table(con, "players"):
        # The crosswalk lives in the SQLite serving DB, not the DuckDB warehouse. Not
        # fatal — it only matters when injury rows lack an id.
        return {}

    cols = _columns(con, "players")
    name_col = next((c for c in ("display_name", "full_name", "player_name") if c in cols), None)
    id_col = next((c for c in ("gsis_id", "player_id") if c in cols), None)
    if not name_col or not id_col:
        return {}

    frame = con.execute(
        f"SELECT {name_col} AS nm, {id_col} AS pid FROM players "
        f"WHERE {name_col} IS NOT NULL AND {id_col} IS NOT NULL"
    ).df()
    frame["key"] = frame["nm"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()

    counts = frame.groupby("key")["pid"].nunique()
    unambiguous = set(counts[counts == 1].index)
    frame = frame[frame["key"].isin(unambiguous)]
    return dict(zip(frame["key"], frame["pid"], strict=False))


def load(con, verbose: bool = True) -> pd.DataFrame:
    """One row per report-listed player-week, with usage and prior-form features."""
    inj_cols = _columns(con, "injuries_hist")

    gsis = "gsis_id" if "gsis_id" in inj_cols else "player_id"
    missing = {gsis, "season", "week", "report_status", "practice_status"} - inj_cols
    if missing:
        raise SystemExit(f"injuries_hist is missing columns: {sorted(missing)}")

    # `season_type` is only populated on recent files — NULL on 84,684 of 90,752 rows.
    # `WHERE season_type = 'REG'` therefore keeps 6% of the table and looks perfectly
    # reasonable while doing it. NULL means regular season here; only POST is excluded.
    season_filter = (
        "WHERE (i.season_type IS NULL OR i.season_type = 'REG')"
        if "season_type" in inj_cols
        else ""
    )

    injuries = con.execute(f"""
        SELECT i.season, i.week, i.{gsis} AS gsis_id, i.position, i.full_name,
               i.report_status, i.practice_status
        FROM injuries_hist i {season_filter}
    """).df()

    if verbose:
        seasons = sorted(injuries["season"].dropna().unique())
        print(f"injuries_hist: {len(injuries):,} rows, "
              f"{injuries['gsis_id'].notna().mean():.1%} carry a gsis_id")
        print(f"  seasons present: {len(seasons)} "
              f"({int(min(seasons))}–{int(max(seasons))})" if seasons else "  no seasons!")
        if len(seasons) <= MIN_TRAIN_SEASONS:
            print(
                f"\n  ⚠  Only {len(seasons)} season(s) on disk. DATA_INVENTORY.md claims\n"
                "     90,752 rows across 2009–2025, so the parquet backfill is incomplete\n"
                "     or the glob in warehouse.py isn't matching the filenames. Check:\n"
                "         ls data/raw/nflverse/ | grep -i injur\n"
                "     Walk-forward needs several seasons; a single season can only\n"
                "     support the descriptive `probe`, not `run`.\n"
            )

    # Recover the rows the id column loses.
    lookup = _name_to_id(con)
    key = (
        injuries["full_name"].fillna("").str.lower()
        .str.replace(r"[^a-z ]", "", regex=True).str.strip()
    )
    injuries["player_id"] = injuries["gsis_id"].fillna(key.map(lookup))
    if verbose:
        print(f"  {injuries['player_id'].notna().mean():.1%} resolved after name fallback")

    injuries = injuries[injuries["player_id"].notna()].copy()

    usage = con.execute("""
        SELECT season, week, player_id,
               COALESCE(target_share, 0) + COALESCE(carry_share, 0) AS usage_share,
               COALESCE(targets, 0) AS targets, COALESCE(carries, 0) AS carries
        FROM player_weeks WHERE season IS NOT NULL AND week IS NOT NULL
    """).df()

    frame = injuries.merge(usage, on=["season", "week", "player_id"], how="left")

    # Snap share is the honest measure of "did he take the field". Touches are not: a
    # blocking tight end or a backup quarterback plays a full game and records zero.
    # That mismeasurement is the likely reason the first probe showed Limited players
    # apparently *more* active than Full ones.
    snap_cols = _columns(con, "snaps") if _has_table(con, "snaps") else set()
    pct = next((c for c in ("offense_pct", "offense_snaps_pct") if c in snap_cols), None)
    pid = next((c for c in ("pfr_player_id", "player_id", "gsis_id") if c in snap_cols), None)
    frame["snap_pct"] = np.nan
    if pct and pid and pid in {"player_id", "gsis_id"}:
        snaps = con.execute(
            f"SELECT season, week, {pid} AS player_id, {pct} AS snap_pct FROM snaps"
        ).df()
        frame = frame.drop(columns=["snap_pct"]).merge(
            snaps, on=["season", "week", "player_id"], how="left"
        )
        if verbose:
            print(f"  snap coverage: {frame['snap_pct'].notna().mean():.1%}")
    elif verbose:
        print("  no joinable snap counts — falling back to touches for `active`")

    if verbose:
        print(f"loaded {len(frame):,} report-listed player-weeks")

    frame = frame[frame["position"].isin(SKILL_POSITIONS)].copy()
    if verbose:
        print(f"  {len(frame):,} at skill positions {SKILL_POSITIONS}")

    frame["practice_norm"] = frame["practice_status"].map(normalise_practice)

    # Fail loudly rather than report a meaningless zero. If the vocabulary shifts again,
    # this is the line that says so instead of the results quietly going flat.
    coverage = frame["practice_norm"].notna().mean()
    if coverage < 0.5:
        raise SystemExit(
            f"practice_status only normalised for {coverage:.1%} of rows.\n"
            f"unmapped values: {sorted(set(frame.loc[frame['practice_norm'].isna(), 'practice_status'].dropna().unique()))[:10]}\n"
            "Fix normalise_practice() before trusting any result — an unmapped column "
            "makes the treatment arm identical to the baseline."
        )
    if verbose:
        print(f"  practice_status normalised for {coverage:.1%} of rows")

    # Active = recorded any usage. A player-week with no row in player_weeks joined to
    # NULL, which means he touched the ball zero times — inactive, or active and unused.
    # Those are indistinguishable here and both score zero on a prop, which is the
    # quantity that actually matters.
    frame["usage_share"] = frame["usage_share"].fillna(0.0)
    if frame["snap_pct"].notna().mean() > 0.5:
        frame["active"] = (frame["snap_pct"].fillna(0) > 0).astype(int)
        frame.attrs["active_source"] = "snap counts"
    else:
        frame["active"] = (frame["usage_share"] > 0).astype(int)
        frame.attrs["active_source"] = "touches (weak proxy — no snap join available)"
    if verbose:
        print(f"  `active` derived from {frame.attrs['active_source']}")

    frame = frame.sort_values(["player_id", "season", "week"])

    # Prior form, strictly backward-looking. `shift(1)` before rolling is what keeps the
    # current week out of its own feature.
    grouped = frame.groupby("player_id")["usage_share"]
    frame["prior_usage_3"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    frame["prior_usage_8"] = grouped.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    frame["prior_active_3"] = frame.groupby("player_id")["active"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    frame = frame.dropna(subset=["prior_usage_3"])

    mapped = frame["practice_norm"].map(PRACTICE_ORDER)
    frame["has_practice"] = mapped.notna().astype(int)
    frame["practice_code"] = mapped.fillna(-1)
    frame["status_code"] = frame["report_status"].map(STATUS_ORDER).fillna(NO_DESIGNATION)

    if frame["practice_code"].nunique() < 2:
        raise SystemExit(
            "practice_code is constant — the treatment arm would equal the baseline "
            "and the test would report a spurious zero effect."
        )

    if verbose:
        print(f"  {len(frame):,} with prior form available\n")
    return frame


# --- models ------------------------------------------------------------------------


# Fixed category sets. Dummy columns MUST be identical and identically ordered across
# every split, or coefficients get applied to the wrong feature.
#
# The first version of this file built dummies with `pd.get_dummies` per split and
# zero-padded the narrower matrix on the right. `get_dummies` orders columns by sorted
# category, so a test season missing one category shifts every later column by one —
# and padding at the end conceals it rather than fixing it. The visible symptom was the
# treatment arm scoring *worse* out of sample, because extra dummy blocks mean more
# chances to misalign. A feature can be useless; it should not be destructive.
POSITIONS = list(SKILL_POSITIONS)
STATUS_CODES = [0, 1, 2, 3, NO_DESIGNATION]
PRACTICE_CODES = [-1, 0, 1, 2]


def _dummies(values: pd.Series, categories: list) -> np.ndarray:
    cat = pd.Categorical(values, categories=categories)
    return pd.get_dummies(cat).to_numpy(float)


def _design(frame: pd.DataFrame, with_practice: bool) -> np.ndarray:
    """Feature matrix. The only difference between arms is the practice block."""
    blocks = [
        frame[["prior_usage_3", "prior_usage_8", "prior_active_3"]].to_numpy(float),
        _dummies(frame["position"], POSITIONS),
        _dummies(frame["status_code"], STATUS_CODES),
    ]
    if with_practice:
        blocks.append(_dummies(frame["practice_code"], PRACTICE_CODES))
        blocks.append(frame[["has_practice"]].to_numpy(float))
    matrix = np.hstack(blocks)
    return np.hstack([np.ones((len(matrix), 1)), matrix])


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 60, ridge: float = 1.0) -> np.ndarray:
    """Newton-IRLS with ridge. statsmodels would do, but this keeps the whole test in one
    file with no fitting-library behaviour to explain away."""
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        hessian = x.T @ (x * w[:, None]) + penalty
        gradient = x.T @ (y - p) - penalty @ beta
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def predict_logistic(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


# --- the experiment ----------------------------------------------------------------


@dataclass
class Arm:
    brier: float
    log_loss: float
    auc: float
    mae: float
    n: int
    season: float | None = None


def walk_forward(frame: pd.DataFrame, label: str, verbose: bool = True) -> dict[str, Arm]:
    """Train on every season before S, test on S. Never the other way round."""
    seasons = sorted(frame["season"].unique())
    if len(seasons) <= MIN_TRAIN_SEASONS:
        raise SystemExit(f"need more than {MIN_TRAIN_SEASONS} seasons, have {len(seasons)}")

    collected: dict[str, list] = {"base": [], "plus": []}

    for season in seasons[MIN_TRAIN_SEASONS:]:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if len(test) < 50:
            continue

        for arm, with_practice in (("base", False), ("plus", True)):
            xtr = _design(train, with_practice)
            xte = _design(test, with_practice)

            beta = fit_logistic(xtr, train["active"].to_numpy(float))
            p_active = predict_logistic(xte, beta)

            # volume among those who actually played — the layer below P(active)
            played_tr = train["active"] == 1
            played_te = test["active"] == 1
            if played_tr.sum() > 50 and played_te.sum() > 20:
                vtr = _design(train[played_tr], with_practice)
                vte = _design(test[played_te], with_practice)
                coefs = fit_ridge(vtr, train.loc[played_tr, "usage_share"].to_numpy(float))
                usage_mae = mae(test.loc[played_te, "usage_share"].to_numpy(float), vte @ coefs)
            else:
                usage_mae = float("nan")

            y = test["active"].to_numpy(float)
            collected[arm].append(
                Arm(
                    brier(y, p_active),
                    log_loss(y, p_active),
                    auc(y, p_active),
                    usage_mae,
                    len(test),
                    season,
                )
            )

    def average(arms: list[Arm]) -> Arm:
        return Arm(
            float(np.mean([a.brier for a in arms])),
            float(np.mean([a.log_loss for a in arms])),
            float(np.nanmean([a.auc for a in arms])),
            float(np.nanmean([a.mae for a in arms])),
            int(np.sum([a.n for a in arms])),
        )

    result = {k: average(v) for k, v in collected.items() if v}

    if verbose and result:
        base, plus = result["base"], result["plus"]
        print(f"\n--- {label} ---")
        print(f"  test rows            {base.n:,}   seasons {seasons[MIN_TRAIN_SEASONS]}–{seasons[-1]}")
        print(f"{'':22}{'status only':>14}{'+ practice':>14}{'delta':>12}")
        for name, lower_is_better in (
            ("brier", True), ("log_loss", True), ("auc", False), ("mae", True)
        ):
            b, p = getattr(base, name), getattr(plus, name)
            delta = p - b
            better = (delta < 0) if lower_is_better else (delta > 0)
            mark = "  better" if better and abs(delta) > 1e-5 else ""
            print(f"  {name:<20}{b:>14.5f}{p:>14.5f}{delta:>+12.5f}{mark}")

        _per_season(collected["base"], collected["plus"])

    return result


def _per_season(base: list[Arm], plus: list[Arm]) -> None:
    """Season-by-season deltas.

    A mean across twelve seasons can be carried entirely by one of them. This is the
    difference between "practice status helps" and "practice status helped in 2019".
    Consistency of *sign* matters more than the size of any single year — twelve
    independent test seasons all leaning the same way is hard to get by chance; a big
    average from three outliers is not evidence.
    """
    print(f"\n  {'season':<9}{'brier Δ':>11}{'logloss Δ':>12}{'auc Δ':>10}{'n':>8}")
    wins = 0
    for b, p in zip(base, plus, strict=True):
        d_brier = p.brier - b.brier
        d_auc = p.auc - b.auc
        wins += d_brier < 0
        flag = " +" if d_brier < 0 else " -"
        print(
            f"  {int(b.season):<9}{d_brier:>+11.5f}{p.log_loss - b.log_loss:>+12.5f}"
            f"{d_auc:>+10.5f}{b.n:>8,}{flag}"
        )

    n = len(base)
    print(f"\n  practice helped (Brier) in {wins}/{n} test seasons")
    if wins == n:
        print("  every season — consistent, not an artifact of one year")
    elif wins >= n * 0.75:
        print("  most seasons — the mean is not being carried by an outlier")
    else:
        print("  ⚠  mixed. Treat the average with suspicion; look at the losing years.")


def cmd_probe(_: argparse.Namespace) -> int:
    """Before running anything, look at what's actually in the table."""
    con = connect(read_only=True)
    frame = load(con)

    print("report_status:")
    print(frame["report_status"].fillna("(none)").value_counts().to_string())
    print("\npractice_status (normalised):")
    print(frame["practice_norm"].fillna("(none)").value_counts().to_string())

    print("\nactive rate by (report_status, practice):")
    print(
        frame.groupby(
            [frame["report_status"].fillna("(none)"), frame["practice_norm"].fillna("(none)")]
        )["active"]
        .agg(["mean", "count"])
        .round(3)
        .to_string()
    )
    if frame["snap_pct"].notna().any():
        print("\nmean snap share by (report_status, practice) — the volume layer:")
        print(
            frame.groupby(
                [frame["report_status"].fillna("(none)"), frame["practice_norm"].fillna("(none)")]
            )["snap_pct"]
            .agg(["mean", "count"])
            .round(3)
            .to_string()
        )
    print(
        "\nIf the active rate barely moves across practice_status within a report_status\n"
        "row, the signal isn't there and the model won't find it either."
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    con = connect(read_only=True)
    frame = load(con)

    if args.position:
        frame = frame[frame["position"] == args.position].copy()
        print(f"restricted to {args.position}: {len(frame):,} rows")

    walk_forward(frame, "all report-listed players")

    questionable = frame[frame["report_status"] == "Questionable"]
    if len(questionable) > 500:
        walk_forward(questionable, "QUESTIONABLE only — the decision that costs money")

    undesignated = frame[frame["report_status"].isna()]
    if len(undesignated) > 500:
        walk_forward(undesignated, "no game designation — practice report only")

    print(
        "\nReading this: a delta of roughly zero on the Questionable subset is the\n"
        "result that matters. It would say the designation already contains what\n"
        "practice participation knows, and that Omaha's trajectory is unlikely to\n"
        "add enough to justify a season of pipeline work."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.practice_signal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="what's in the data").set_defaults(func=cmd_probe)

    p_run = sub.add_parser("run", help="walk-forward comparison")
    p_run.add_argument("--position", default=None, choices=SKILL_POSITIONS)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
