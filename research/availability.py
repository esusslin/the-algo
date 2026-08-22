"""Layer 0 — will this player be involved at all?

The diagnosed blocker on props was that bust probability was treated as a player
constant when it depends on the specific game. Building this model on the
corrected panel (see `research/panel.py`) addresses both halves of the problem:

1. The panel materialises zero-involvement games that `player_weeks` omits, so
   the base rate is right for the first time (19.0%, not 6.9%).
2. This model makes the rate game-dependent — injury designation, practice
   participation, recent form, and blowout risk.

HOLDOUT DISCIPLINE
------------------
    train   <= 2022
    dev        2023      tune here, as often as you like
    holdout 2024-2025    ONE run, when the model is final

`evaluate_holdout()` refuses to run without an explicit flag, and prints a
warning when it does. A holdout re-run after adjustment is not a holdout, and
this project has already paid once for learning that.

WHAT COUNTS AS A FEATURE
------------------------
Everything must be knowable before kickoff. Panel columns prefixed `post_`
describe the completed game; `_assert_no_leakage()` fails loudly if one appears
in FEATURES.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from research.panel import build_panel

log = logging.getLogger(__name__)

TRAIN_MAX = 2022
DEV_SEASON = 2023
HOLDOUT = (2024, 2025)

# Ordered. Ordering is part of the contract — a model scored on a permuted
# matrix produces confident nonsense and no metric catches it.
FEATURES = [
    "report_out",           # on the report as Out (kept for completeness; usually filtered)
    "report_doubtful",
    "report_questionable",
    "report_probable",
    "report_none",
    "practice_dnp",
    "practice_limited",
    "practice_full",
    "practice_none",
    "trail_mean",
    "trail_max",
    "trail_zero_rate",
    "last_week_vol",
    "baseline_vol",
    "vol_vs_baseline",      # last week relative to season baseline — role change signal
    "trail_n",
    "week",
    "abs_spread",           # blowout risk
    "implied_team_total",   # pace / scoring environment
    "is_home",
]

_REPORT_MAP = {
    "Out": "report_out",
    "Doubtful": "report_doubtful",
    "Questionable": "report_questionable",
    "Probable": "report_probable",
}
_PRACTICE_MAP = {
    "Did Not Participate In Practice": "practice_dnp",
    "Out (Definitely Will Not Play)": "practice_dnp",
    "Limited Participation in Practice": "practice_limited",
    "Full Participation in Practice": "practice_full",
}


def _assert_no_leakage() -> None:
    bad = [f for f in FEATURES if f.startswith("post_")]
    if bad:
        raise ValueError(f"contemporaneous columns in FEATURES: {bad}")


def _game_context() -> dict[tuple, dict]:
    """(season, week, team) -> spread/total context, from the serving DB.

    `spread_line` here follows the nflverse convention (positive = home favoured
    by that much), which is the OPPOSITE of the Vegas convention. Only the
    magnitude is used below, so the sign question doesn't arise — but it is
    exactly the trap that cost a day on the residual model, so it is written
    down rather than assumed.
    """
    from src.db import query

    out: dict[tuple, dict] = {}
    for g in query(
        "SELECT season, week, home_team, away_team, spread_line, total_line "
        "FROM games WHERE spread_line IS NOT NULL AND total_line IS NOT NULL"
    ):
        spread = float(g["spread_line"])
        total = float(g["total_line"])
        # implied team total = half the total, adjusted by half the spread
        home_it = total / 2.0 + spread / 2.0
        away_it = total / 2.0 - spread / 2.0
        out[(g["season"], g["week"], g["home_team"])] = {
            "abs_spread": abs(spread), "implied_team_total": home_it, "is_home": 1.0}
        out[(g["season"], g["week"], g["away_team"])] = {
            "abs_spread": abs(spread), "implied_team_total": away_it, "is_home": 0.0}
    return out


def build_matrix(rows: list[dict] | None = None,
                 exclude_ruled_out: bool = True) -> dict[str, Any]:
    """Panel -> (X, y, seasons). Rows without trailing history are dropped.

    `exclude_ruled_out` defaults True because that is the population a book
    actually prices. Nobody offers a receiving-yards line on a player listed Out,
    so training on him inflates apparent skill for free.
    """
    _assert_no_leakage()
    rows = rows if rows is not None else build_panel()
    ctx = _game_context()

    X: list[list[float]] = []
    y: list[int] = []
    seasons: list[int] = []
    dropped = {"no_history": 0, "no_context": 0, "ruled_out": 0}

    for r in rows:
        if exclude_ruled_out and r["ruled_out"]:
            dropped["ruled_out"] += 1
            continue
        if not r["trail_n"]:
            dropped["no_history"] += 1
            continue
        c = ctx.get((r["season"], r["week"], r["team"]))
        if not c:
            dropped["no_context"] += 1
            continue

        vals: dict[str, float] = {f: 0.0 for f in FEATURES}
        rk = _REPORT_MAP.get(r["report_status"] or "")
        vals[rk if rk else "report_none"] = 1.0
        pk = _PRACTICE_MAP.get((r["practice_status"] or "").strip())
        vals[pk if pk else "practice_none"] = 1.0

        base = float(r["baseline_vol"]) or 1.0
        vals["trail_mean"] = float(r["trail_mean"])
        vals["trail_max"] = float(r["trail_max"])
        vals["trail_zero_rate"] = float(r["trail_zero_rate"])
        vals["last_week_vol"] = float(r["last_week_vol"])
        vals["baseline_vol"] = base
        vals["vol_vs_baseline"] = float(r["last_week_vol"]) / base
        vals["trail_n"] = float(r["trail_n"])
        vals["week"] = float(r["week"])
        vals.update(c)

        X.append([vals[f] for f in FEATURES])
        y.append(int(r["bust"]))
        seasons.append(int(r["season"]))

    log.info("matrix: %d rows, %d features | dropped %s", len(X), len(FEATURES), dropped)
    return {"X": np.array(X, dtype=float), "y": np.array(y, dtype=int),
            "seasons": np.array(seasons, dtype=int), "names": FEATURES,
            "dropped": dropped}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """The only metric that matters for a probability. Accuracy is a distraction
    when the base rate is 19% — always predicting 'no bust' scores 81%."""
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        out.append({"bucket": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                    "predicted": float(p[m].mean()), "actual": float(y[m].mean())})
    return out


def ece(rel: list[dict], n_total: int) -> float:
    """Expected calibration error — one number for 'do the probabilities mean
    what they say'."""
    if not n_total:
        return 0.0
    return float(sum(b["n"] * abs(b["predicted"] - b["actual"]) for b in rel) / n_total)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def _fit(Xtr, ytr, Xte):
    """Gradient boosting with a linear fallback, mirroring research/train.py."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=4,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        m.fit(Xtr, ytr)
        return m, m.predict_proba(Xte)[:, 1]
    except Exception as exc:  # noqa: BLE001
        log.warning("GBM unavailable (%s) — falling back to logistic", exc)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.5))
        m.fit(Xtr, ytr)
        return m, m.predict_proba(Xte)[:, 1]


@dataclass
class Result:
    split: str
    n_train: int
    n_test: int
    base_rate: float
    brier: float
    brier_baseline: float
    log_loss: float
    ece: float
    reliability: list[dict] = field(default_factory=list)
    importances: list[tuple[str, float]] = field(default_factory=list)

    def report(self) -> str:
        d = self.brier_baseline - self.brier
        lines = [
            f"\n=== availability model — {self.split} ===\n",
            f"  train rows        : {self.n_train:,}",
            f"  test rows         : {self.n_test:,}",
            f"  base rate (bust)  : {self.base_rate:.1%}",
            "",
            f"  brier             : {self.brier:.4f}",
            f"  brier (base rate) : {self.brier_baseline:.4f}",
            f"  improvement       : {d:+.4f}"
            f"   {'<-- beats the base rate' if d > 0.002 else '<-- no better than the base rate'}",
            f"  log loss          : {self.log_loss:.4f}",
            f"  ECE               : {self.ece:.4f}",
            "",
            "  --- calibration ---",
            f"  {'bucket':<12}{'n':>7}{'predicted':>11}{'actual':>9}{'gap':>9}",
        ]
        for b in self.reliability:
            gap = b["predicted"] - b["actual"]
            flag = "  <-- off" if abs(gap) > 0.05 else ""
            lines.append(f"  {b['bucket']:<12}{b['n']:>7}{b['predicted']:>11.3f}"
                         f"{b['actual']:>9.3f}{gap:>+9.3f}{flag}")
        if self.importances:
            lines.append("\n  --- top features ---")
            for name, imp in self.importances[:12]:
                lines.append(f"  {name:<22}{imp:>8.4f}")
        return "\n".join(lines)


def _evaluate(d: dict, train_mask, test_mask, split: str) -> Result:
    X, y = d["X"], d["y"]
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    model, p = _fit(Xtr, ytr, Xte)
    p = np.clip(p, 0.001, 0.999)

    # The honest baseline is the training base rate, not 0.5. A model that only
    # reproduces the base rate has learned nothing about the game.
    base = float(ytr.mean())
    p_base = np.full(len(yte), base)

    rel = reliability(yte, p)
    imps: list[tuple[str, float]] = []
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xte, yte, n_repeats=5,
                                   random_state=0, scoring="neg_brier_score")
        imps = sorted(zip(d["names"], r.importances_mean),
                      key=lambda kv: -kv[1])
    except Exception:  # noqa: BLE001
        pass

    return Result(
        split=split, n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        base_rate=float(yte.mean()), brier=brier(yte, p),
        brier_baseline=brier(yte, p_base), log_loss=log_loss(yte, p),
        ece=ece(rel, len(yte)), reliability=rel, importances=imps,
    )


def evaluate_dev(d: dict | None = None) -> Result:
    """Train on <= 2022, test on 2023. Run this as often as you like."""
    d = d or build_matrix()
    s = d["seasons"]
    return _evaluate(d, s <= TRAIN_MAX, s == DEV_SEASON, f"dev ({DEV_SEASON})")


def evaluate_holdout(d: dict | None = None, i_am_sure: bool = False) -> Result:
    """Train on <= 2023, test on 2024-25. ONE RUN. EVER."""
    if not i_am_sure:
        raise RuntimeError(
            "The holdout is one-shot. If you run it, tune, and run it again, it "
            "is no longer a holdout and the number it produces is meaningless. "
            "Pass i_am_sure=True (or --holdout-i-am-sure) only when the model is "
            "final.")
    log.warning("RUNNING THE ONE-SHOT HOLDOUT — record the result whatever it says")
    d = d or build_matrix()
    s = d["seasons"]
    return _evaluate(d, s <= DEV_SEASON,
                     (s >= HOLDOUT[0]) & (s <= HOLDOUT[1]),
                     f"HOLDOUT ({HOLDOUT[0]}-{HOLDOUT[1]})")


def leakage_checks(d: dict) -> list[str]:
    """Cheap probes that would each have caught a real bug."""
    problems: list[str] = []
    X, y, names = d["X"], d["y"], d["names"]

    # 1. Shuffled labels must destroy all signal.
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    s = d["seasons"]
    tr, te = s <= TRAIN_MAX, s == DEV_SEASON
    _, p = _fit(X[tr], y_shuf[tr], X[te])
    b_shuf = brier(y_shuf[te], np.clip(p, 0.001, 0.999))
    b_base = brier(y_shuf[te], np.full(te.sum(), float(y_shuf[tr].mean())))
    if b_shuf < b_base - 0.005:
        problems.append(f"shuffle test: model beats base rate on shuffled labels "
                        f"({b_shuf:.4f} vs {b_base:.4f}) — leakage")

    # 2. No single feature should correlate suspiciously hard with the label.
    for i, nm in enumerate(names):
        col = X[:, i]
        if col.std() == 0:
            continue
        r = abs(float(np.corrcoef(col, y)[0, 1]))
        if r > 0.75:
            problems.append(f"feature '{nm}' correlates {r:.3f} with the label — "
                            f"probably a proxy for the outcome")

    # 3. Contemporaneous columns must not have crept in.
    bad = [n for n in names if n.startswith("post_")]
    if bad:
        problems.append(f"contemporaneous features present: {bad}")

    return problems


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Layer 0 availability model")
    p.add_argument("command", choices=["dev", "holdout", "checks"], nargs="?",
                   default="dev")
    p.add_argument("--holdout-i-am-sure", action="store_true")
    args = p.parse_args()

    d = build_matrix()

    if args.command == "checks":
        probs = leakage_checks(d)
        print("\n=== leakage checks ===")
        for pr in probs:
            print(f"  FAIL  {pr}")
        if not probs:
            print("  all checks passed")
    elif args.command == "holdout":
        print(evaluate_holdout(d, i_am_sure=args.holdout_i_am_sure).report())
    else:
        print(evaluate_dev(d).report())
        print("\n=== leakage checks ===")
        probs = leakage_checks(d)
        for pr in probs:
            print(f"  FAIL  {pr}")
        if not probs:
            print("  all checks passed")
