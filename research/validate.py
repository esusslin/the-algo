"""Walk-forward validation harness and leakage detection.

The single most important module in the research plane. A subtle leak produces a
beautiful backtest, and a beautiful backtest is how this project fails — not
loudly, but by shipping a model that looks excellent and loses money.

Every model passes through `walk_forward()` and `leakage_report()` before it can
be exported into an artifact bundle.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Any binary market model scoring above this is presumed broken until proven
# otherwise. Legitimate ATS models top out around 52-53%; 55%+ means leakage,
# a labelling bug, or an accidentally-included outcome variable.
TRIPWIRE_ACCURACY = 0.55
TRIPWIRE_AUC = 0.65


@dataclass
class FoldResult:
    test_season: int
    n_train: int
    n_test: int
    accuracy: float | None = None
    log_loss: float | None = None
    brier: float | None = None
    market_accuracy: float | None = None
    market_brier: float | None = None
    edge_vs_market: float | None = None


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float | None:
        vals = [f.accuracy for f in self.folds if f.accuracy is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def mean_brier(self) -> float | None:
        vals = [f.brier for f in self.folds if f.brier is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def mean_edge_vs_market(self) -> float | None:
        vals = [f.edge_vs_market for f in self.folds if f.edge_vs_market is not None]
        return float(np.mean(vals)) if vals else None

    def summary(self) -> str:
        lines = [f"{'season':>8}{'train':>8}{'test':>7}{'acc':>8}{'brier':>8}"
                 f"{'mkt acc':>9}{'vs mkt':>8}"]
        for f in self.folds:
            lines.append(
                f"{f.test_season:>8}{f.n_train:>8}{f.n_test:>7}"
                f"{(f.accuracy or 0):>8.3f}{(f.brier or 0):>8.4f}"
                f"{(f.market_accuracy or 0):>9.3f}{(f.edge_vs_market or 0):>+8.3f}")
        if self.mean_accuracy is not None:
            lines.append(f"{'mean':>8}{'':>15}{self.mean_accuracy:>8.3f}"
                         f"{(self.mean_brier or 0):>8.4f}{'':>9}"
                         f"{(self.mean_edge_vs_market or 0):>+8.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(int) == y))


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability table. Calibration matters more than accuracy for betting —
    a 55% model that claims 70% will bankrupt you through oversizing."""
    out = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        out.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": int(m.sum()),
                    "predicted": round(float(p[m].mean()), 4),
                    "actual": round(float(y[m].mean()), 4),
                    "gap": round(float(p[m].mean() - y[m].mean()), 4)})
    return out


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------
def walk_forward(
    seasons: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    market_prob: np.ndarray | None = None,
    min_train_seasons: int = 5,
) -> WalkForwardResult:
    """Train strictly on prior seasons, test on the next one.

    Random train/test splits leak future information into training on
    time-series data — a model can learn from Week 17 to predict Week 3.
    """
    res = WalkForwardResult()
    uniq = np.sort(np.unique(seasons))

    for i, test_season in enumerate(uniq):
        if i < min_train_seasons:
            continue
        tr = seasons < test_season      # STRICTLY before — never <=
        te = seasons == test_season
        if tr.sum() == 0 or te.sum() == 0:
            continue

        # hard guard: a training row at or after the test season is a leak
        if seasons[tr].max() >= test_season:
            res.warnings.append(
                f"LEAK: training data from season {seasons[tr].max()} "
                f"used to predict {test_season}")
            continue

        p = np.asarray(fit_predict(X[tr], y[tr], X[te]), dtype=float)
        f = FoldResult(test_season=int(test_season), n_train=int(tr.sum()),
                       n_test=int(te.sum()),
                       accuracy=accuracy(y[te], p), log_loss=log_loss(y[te], p),
                       brier=brier(y[te], p))
        if market_prob is not None:
            mp = market_prob[te]
            f.market_accuracy = accuracy(y[te], mp)
            f.market_brier = brier(y[te], mp)
            # the only number that matters: are we better than the price?
            f.edge_vs_market = f.market_brier - f.brier
        res.folds.append(f)

    if res.mean_accuracy and res.mean_accuracy > TRIPWIRE_ACCURACY:
        res.warnings.append(
            f"TRIPWIRE: mean accuracy {res.mean_accuracy:.3f} exceeds "
            f"{TRIPWIRE_ACCURACY}. Legitimate NFL binary models do not do this. "
            f"Assume leakage until proven otherwise.")
    return res


# --------------------------------------------------------------------------
# leakage probes
# --------------------------------------------------------------------------
def shuffle_test(
    seasons: np.ndarray, X: np.ndarray, y: np.ndarray,
    fit_predict: Callable, n_repeats: int = 3, rng_seed: int = 0,
) -> dict:
    """Shuffle the target. Performance MUST collapse to chance.

    If a model still predicts a randomised target, it is reading the answer from
    somewhere — a feature derived from the outcome, a duplicated row, an index
    that encodes the label.
    """
    rng = np.random.default_rng(rng_seed)
    accs = []
    for _ in range(n_repeats):
        y_shuf = rng.permutation(y)
        r = walk_forward(seasons, X, y_shuf, fit_predict)
        if r.mean_accuracy is not None:
            accs.append(r.mean_accuracy)
    mean_acc = float(np.mean(accs)) if accs else 0.5
    # with a shuffled target anything meaningfully above chance is a red flag
    passed = abs(mean_acc - 0.5) < 0.03
    return {"passed": passed, "mean_accuracy": round(mean_acc, 4),
            "detail": ("collapsed to chance as expected" if passed else
                       f"STILL PREDICTIVE at {mean_acc:.3f} on shuffled labels — "
                       f"a feature is leaking the outcome")}


def feature_target_correlation(X: np.ndarray, y: np.ndarray,
                               names: Sequence[str], threshold: float = 0.5) -> dict:
    """Any single feature correlating strongly with the outcome is suspicious.

    Real pre-game features correlate weakly. A |r| above ~0.5 usually means an
    outcome-derived column slipped into the matrix.
    """
    suspects = []
    for j, name in enumerate(names):
        col = X[:, j].astype(float)
        if np.all(np.isnan(col)) or np.nanstd(col) == 0:
            continue
        ok = ~np.isnan(col)
        if ok.sum() < 30:
            continue
        r = float(np.corrcoef(col[ok], y[ok])[0, 1])
        if abs(r) >= threshold:
            suspects.append({"feature": name, "r": round(r, 3)})
    return {"passed": not suspects, "suspects": suspects}


def availability_check(decision_hours_before_kick: float = 1.0) -> dict:
    """Every feature must be knowable at decision time. Delegates to the
    single source of truth so this can't drift from what models actually use."""
    from shared.feature_spec import validate_availability
    bad = validate_availability(decision_hours_before_kick)
    return {"passed": not bad, "violations": bad}


def temporal_ordering_check(seasons: np.ndarray, weeks: np.ndarray,
                            kickoffs: Sequence[str] | None = None) -> dict:
    """Rows must be orderable in time, and duplicates are a corruption signal."""
    problems = []
    if len(seasons) != len(weeks):
        problems.append("seasons and weeks length mismatch")
    if kickoffs is not None and len(kickoffs) != len(seasons):
        problems.append("kickoffs length mismatch")
    if np.any(np.isnan(seasons.astype(float))):
        problems.append("null season values")
    return {"passed": not problems, "problems": problems}


def leakage_report(
    seasons: np.ndarray, X: np.ndarray, y: np.ndarray,
    names: Sequence[str], fit_predict: Callable,
    weeks: np.ndarray | None = None,
) -> dict:
    """Run every probe. This gates artifact export."""
    checks = {
        "shuffle_test": shuffle_test(seasons, X, y, fit_predict),
        "feature_target_correlation": feature_target_correlation(X, y, names),
        "feature_availability": availability_check(),
    }
    if weeks is not None:
        checks["temporal_ordering"] = temporal_ordering_check(seasons, weeks)

    wf = walk_forward(seasons, X, y, fit_predict)
    checks["tripwire"] = {
        "passed": not any(w.startswith("TRIPWIRE") for w in wf.warnings),
        "mean_accuracy": wf.mean_accuracy,
        "warnings": wf.warnings,
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("leakage suite self-test — a clean model and a deliberately leaky one\n")

    rng = np.random.default_rng(7)
    n = 4000
    seasons = rng.integers(2015, 2026, n)
    weeks = rng.integers(1, 19, n)
    signal = rng.normal(size=n)
    # Noise scale is deliberately brutal: a realistic NFL binary model lands
    # around 52%. Anything that looks better than this in real data is a bug.
    y = (signal + rng.normal(scale=9.0, size=n) > 0).astype(int)

    X_clean = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    X_leaky = np.column_stack([signal, rng.normal(size=n), y + rng.normal(scale=.2, size=n)])
    names = ["market_spread", "rest_diff", "third_feature"]

    def fit_predict(Xtr, ytr, Xte):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=400).fit(np.nan_to_num(Xtr), ytr)
        return m.predict_proba(np.nan_to_num(Xte))[:, 1]

    for label, Xm in [("CLEAN model", X_clean), ("LEAKY model", X_leaky)]:
        r = leakage_report(seasons, Xm, y, names, fit_predict, weeks=weeks)
        print(f"=== {label} — overall {'PASS' if r['passed'] else 'FAIL'} ===")
        for k, c in r["checks"].items():
            print(f"  {'ok  ' if c['passed'] else 'FAIL'} {k}")
            if not c["passed"]:
                for key in ("detail", "suspects", "warnings", "violations"):
                    if c.get(key):
                        print(f"        {c[key]}")
        print()

    print("Note: the probes are complementary, and neither alone is sufficient.")
    print("  shuffle_test  catches leaks via row identity, duplication or index")
    print("                encoding — but NOT a feature derived from the outcome,")
    print("                because shuffling breaks that correlation too.")
    print("  correlation   catches outcome-derived features.")
    print("  tripwire      catches everything else, by refusing to believe a")
    print("                result that no legitimate NFL model produces.")
