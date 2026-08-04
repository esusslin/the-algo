"""Market-anchored residual model — the primary model.

The idea
--------
Do not predict the margin. Predict `margin + spread_line` — how much the home
side beats the number by.

The market has already priced team strength, injuries, rest, weather and public
sentiment, and thousands of sharp participants have corrected it. Predicting
margin from scratch means competing with all of that. Predicting the residual
means asking a much narrower question: where is that consensus *systematically*
wrong?

Consequences worth internalising before reading any output:

* **R² will be near zero, and that is correct.** The market explains ~86% of
  outcome variance. What's left is mostly irreducible noise. An R² of 0.05 on
  residuals would be extraordinary; 0.60 means you have a leak.
* **The only meaningful benchmark is the market**, never accuracy in isolation.
  Every fold reports model Brier against market Brier.
* **Calibration matters more than accuracy.** A 53% model that says 53% is
  bettable. A 55% model that says 70% will bankrupt you through Kelly sizing.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.features import build_matrix
from research.validate import (TRIPWIRE_ACCURACY, accuracy, brier,
                               calibration_bins, leakage_report, log_loss)

log = logging.getLogger(__name__)

# NFL margins have sd ≈ 13.5 points. Used to turn a predicted residual into a
# cover probability. A normal approximation understates the mass at 3 and 7 —
# the simulator will fix that; see ROADMAP.
MARGIN_SD = 13.5
TOTAL_SD = 10.5


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def residual_to_cover_prob(residual_hat: np.ndarray,
                           sd: float = MARGIN_SD) -> np.ndarray:
    """P(home covers) given a predicted residual."""
    return _norm_cdf(np.asarray(residual_hat, dtype=float) / sd)


@dataclass
class TrainResult:
    target: str
    folds: list[dict]
    calibration: list[dict]
    leakage: dict
    feature_importance: list[tuple[str, float]]
    verdict: str

    def summary(self) -> str:
        lines = [f"{'season':>8}{'n_test':>8}{'model':>9}{'market':>9}"
                 f"{'Δbrier':>9}{'acc':>8}{'mkt acc':>9}"]
        for f in self.folds:
            lines.append(
                f"{f['season']:>8}{f['n_test']:>8}{f['brier']:>9.4f}"
                f"{f['market_brier']:>9.4f}{f['market_brier']-f['brier']:>+9.4f}"
                f"{f['accuracy']:>8.3f}{f['market_accuracy']:>9.3f}")
        if self.folds:
            mb = np.mean([f["brier"] for f in self.folds])
            mk = np.mean([f["market_brier"] for f in self.folds])
            ac = np.mean([f["accuracy"] for f in self.folds])
            mac = np.mean([f["market_accuracy"] for f in self.folds])
            lines.append(f"{'mean':>8}{'':>8}{mb:>9.4f}{mk:>9.4f}"
                         f"{mk-mb:>+9.4f}{ac:>8.3f}{mac:>9.3f}")
        return "\n".join(lines)


_BACKEND: str | None = None


def _fit_gbm(Xtr, ytr, Xte, seed: int = 0):
    """Gradient boosting with graceful fallbacks.

    Conservative settings throughout: the signal in residuals is faint, and an
    over-parameterised tree will happily memorise noise and report it as skill.

    Backend order:
      1. LightGBM — fastest, but on macOS it needs OpenMP (`brew install libomp`)
         and raises OSError at import without it.
      2. sklearn HistGradientBoosting — near-equivalent algorithm, no external
         native dependency. The practical default on a Mac.
      3. Ridge — last resort, so a fresh clone can still run end to end.
    """
    global _BACKEND
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.02, num_leaves=7,
            min_child_samples=80, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)
        m.fit(Xtr, ytr)
        if _BACKEND != "lightgbm":
            _BACKEND = "lightgbm"
            log.info("gradient boosting backend: lightgbm")
        return m, m.predict(Xte)
    except (ImportError, OSError) as exc:
        if _BACKEND is None:
            log.warning("lightgbm unavailable (%s) — falling back to sklearn. "
                        "On macOS: brew install libomp", type(exc).__name__)

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.02, max_leaf_nodes=7,
            min_samples_leaf=80, l2_regularization=5.0,
            early_stopping=False, random_state=seed)
        m.fit(Xtr, ytr)
        if _BACKEND != "sklearn":
            _BACKEND = "sklearn"
            log.info("gradient boosting backend: sklearn HistGradientBoosting")
        return m, m.predict(Xte)
    except ImportError:
        pass

    from sklearn.linear_model import Ridge
    m = Ridge(alpha=20.0).fit(Xtr, ytr)
    if _BACKEND != "ridge":
        _BACKEND = "ridge"
        log.warning("no gradient boosting available — using ridge")
    return m, m.predict(Xte)


def train_residual(target: str = "spread", min_train_seasons: int = 5,
                   from_season: int | None = None) -> TrainResult:
    """Walk-forward train and evaluate the residual model against the market."""
    seasons_filter = None
    if from_season:
        from src.db import query
        seasons_filter = [r["season"] for r in query(
            "SELECT DISTINCT season FROM games WHERE season >= ? ORDER BY 1",
            (from_season,))]

    d = build_matrix(seasons_filter)
    X, names = d["X"], d["names"]
    seasons = d["seasons"]

    if target == "spread":
        y_resid, y_binary, sd = d["residual"], d["home_cover"], MARGIN_SD
    elif target == "total":
        y_resid, y_binary, sd = d["total_residual"], d["over"], TOTAL_SD
    else:
        raise ValueError("target must be 'spread' or 'total'")

    uniq = np.sort(np.unique(seasons))
    folds: list[dict] = []
    oof_p, oof_y = [], []
    importances = np.zeros(X.shape[1])
    n_fits = 0

    n_folds = max(0, len(uniq) - min_train_seasons)
    log.info("walk-forward: %d folds over %d games", n_folds, len(X))

    for i, test_season in enumerate(uniq):
        if i < min_train_seasons:
            continue
        tr, te = seasons < test_season, seasons == test_season
        if tr.sum() < 200 or te.sum() < 20:
            continue
        log.info("  fold %d/%d — test season %d (%d train, %d test)",
                 len(folds) + 1, n_folds, test_season, tr.sum(), te.sum())

        model, resid_hat = _fit_gbm(X[tr], y_resid[tr], X[te])
        p = np.clip(residual_to_cover_prob(resid_hat, sd), 0.01, 0.99)

        # The market's own view: the line implies a coin flip on the spread.
        # Beating 0.25 Brier is the entire bar.
        p_market = np.full(te.sum(), 0.5)

        folds.append({
            "season": int(test_season), "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "brier": brier(y_binary[te], p), "market_brier": brier(y_binary[te], p_market),
            "log_loss": log_loss(y_binary[te], p),
            "accuracy": accuracy(y_binary[te], p),
            "market_accuracy": max(np.mean(y_binary[te]), 1 - np.mean(y_binary[te])),
            "resid_rmse": float(np.sqrt(np.mean((resid_hat - y_resid[te]) ** 2))),
        })
        oof_p.append(p); oof_y.append(y_binary[te])
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        if imp is not None and imp.sum() > 0 and len(imp) == X.shape[1]:
            importances += imp / imp.sum()
            n_fits += 1

    p_all = np.concatenate(oof_p) if oof_p else np.array([])
    y_all = np.concatenate(oof_y) if oof_y else np.array([])

    # Leakage probes use a FAST surrogate, deliberately.
    #
    # Two reasons. First, the earlier version passed `y_resid[:len(a)]` — a
    # slice by length rather than by the actual training rows, which silently
    # paired features with the wrong targets. Second, refitting a boosted model
    # for every probe means ~90 fits and several minutes; leakage shows up just
    # as clearly in a linear model, and a suite nobody waits for is a suite
    # nobody runs.
    def _fast_probe(Xtr, ytr, Xte):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=300).fit(np.nan_to_num(Xtr), ytr)
        return m.predict_proba(np.nan_to_num(Xte))[:, 1]

    log.info("running leakage suite...")
    leak = leakage_report(seasons, X, y_binary, names, _fast_probe, weeks=d["weeks"])

    if importances.sum() > 0:
        imp = importances / max(n_fits, 1)
        order = np.argsort(imp)[::-1]
        top = [(names[j], round(float(imp[j]), 4)) for j in order[:12]]
    else:
        top = []

    mean_brier = float(np.mean([f["brier"] for f in folds])) if folds else None
    mean_mkt = float(np.mean([f["market_brier"] for f in folds])) if folds else None
    mean_acc = float(np.mean([f["accuracy"] for f in folds])) if folds else None

    if mean_acc and mean_acc > TRIPWIRE_ACCURACY:
        verdict = ("TRIPWIRE — accuracy above 55% on a binary NFL market. "
                   "Assume leakage. Do not export this bundle.")
    elif mean_brier is None:
        verdict = "insufficient data"
    elif mean_brier < mean_mkt - 0.002:
        verdict = (f"beats market Brier by {mean_mkt - mean_brier:.4f}. "
                   f"Plausible but small — validate with live CLV before publishing.")
    else:
        verdict = ("no edge over market. This is the EXPECTED result for full-game "
                   "spreads and totals. Keep the model dark; edge lives in props "
                   "and derivatives.")

    return TrainResult(
        target=target, folds=folds,
        calibration=calibration_bins(y_all, p_all) if len(y_all) else [],
        leakage=leak, feature_importance=top, verdict=verdict)


def report(res: TrainResult) -> None:
    print(f"\n=== market-anchored residual model: {res.target} ===\n")
    print(res.summary())

    print("\n--- calibration (predicted vs actual) ---")
    print(f"  {'bucket':<14}{'n':>7}{'predicted':>11}{'actual':>9}{'gap':>9}")
    for b in res.calibration:
        flag = "  <-- off" if abs(b["gap"]) > 0.05 and b["n"] > 30 else ""
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}{'':<6}{b['n']:>7}"
              f"{b['predicted']:>11.3f}{b['actual']:>9.3f}{b['gap']:>+9.3f}{flag}")

    if res.feature_importance:
        print("\n--- top features ---")
        for name, v in res.feature_importance:
            print(f"  {name:<24}{v:>8.4f}")

    print("\n--- leakage suite ---")
    for k, c in res.leakage["checks"].items():
        print(f"  {'ok  ' if c['passed'] else 'FAIL'} {k}")
        if not c["passed"]:
            for key in ("detail", "suspects", "warnings", "violations"):
                if c.get(key):
                    print(f"        {c[key]}")

    print(f"\n--- verdict ---\n  {res.verdict}\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="train residual models (local only)")
    p.add_argument("target", choices=["spread", "total", "both"], nargs="?", default="both")
    p.add_argument("--from-season", type=int, default=None)
    p.add_argument("--min-train", type=int, default=5)
    args = p.parse_args()

    targets = ["spread", "total"] if args.target == "both" else [args.target]
    for t in targets:
        report(train_residual(t, min_train_seasons=args.min_train,
                              from_season=args.from_season))
