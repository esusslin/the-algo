"""Export a trained artifact bundle for the serving plane.

The bundle is the ONLY thing that crosses from research to production. It
carries the models, their calibrations, per-market blend weights, and a manifest
recording exactly what produced them.

Two rules enforced here:

1. **The feature_spec hash is stamped in.** The loader compares it against the
   running code and refuses to serve on mismatch. That's what prevents a model
   scoring on misaligned features and returning confident nonsense.

2. **Blend weights start at zero.** A freshly trained model gets no influence
   until measured CLV earns it. Shipping a model straight to a meaningful weight
   is how an unvalidated model reaches real money.

    python -m research.export build --target spread
    python -m research.export build --target spread --weight 0.10
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research.features import build_matrix
from research.train import MARGIN_SD, TOTAL_SD, _fit_gbm, train_residual
from shared.feature_spec import FEATURES, spec_hash
from src.config import settings

log = logging.getLogger(__name__)


def build_bundle(target: str = "spread", blend_weight: float = 0.0,
                 min_train_seasons: int = 5, notes: str = "") -> dict:
    """Train on ALL available data and export.

    Walk-forward validation happens in train.py and measures whether the model
    generalises. This fits the final model on everything, which is correct for
    deployment — but it means the metrics recorded here come from the
    walk-forward run, never from the fit itself.
    """
    log.info("validating before export...")
    val = train_residual(target, min_train_seasons=min_train_seasons)

    if not val.leakage["passed"]:
        failed = [k for k, c in val.leakage["checks"].items() if not c["passed"]]
        raise SystemExit(
            f"REFUSING TO EXPORT — leakage suite failed: {', '.join(failed)}\n"
            f"Fix the cause. Do not export a model that cannot pass validation.")

    d = build_matrix()
    X = d["X"]
    y = d["residual"] if target == "spread" else d["total_residual"]
    sd = MARGIN_SD if target == "spread" else TOTAL_SD

    log.info("fitting final model on %d games", len(X))
    model, _ = _fit_gbm(X, y, X[:1])

    version = datetime.now(timezone.utc).strftime("bundle_%Y%m%d_%H%M")
    path = settings.ARTIFACT_DIR / version
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    model_file = f"{target}_residual.pkl"
    with open(path / model_file, "wb") as fh:
        pickle.dump(model, fh)

    mean_brier = float(np.mean([f["brier"] for f in val.folds])) if val.folds else None
    mean_mkt = float(np.mean([f["market_brier"] for f in val.folds])) if val.folds else None
    mean_acc = float(np.mean([f["accuracy"] for f in val.folds])) if val.folds else None

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "feature_spec_hash": spec_hash(),
        "feature_count": len(FEATURES),
        "feature_names": [f.name for f in FEATURES],
        "trained_through_season": int(max(d["seasons"])),
        "trained_through_week": int(max(d["weeks"])),
        "n_train_games": int(len(X)),
        "residual_sd": sd,
        "models": {f"{target}_residual": model_file},
        # Zero unless explicitly overridden. An unproven model gets no say.
        "blend_weights": {target: blend_weight,
                          "game": blend_weight},
        "metrics": {
            "walk_forward_folds": len(val.folds),
            "mean_brier": round(mean_brier, 5) if mean_brier else None,
            "mean_market_brier": round(mean_mkt, 5) if mean_mkt else None,
            "beats_market_by": round(mean_mkt - mean_brier, 5)
                               if (mean_brier and mean_mkt) else None,
            "mean_accuracy": round(mean_acc, 4) if mean_acc else None,
            "verdict": val.verdict,
        },
        "leakage_passed": val.leakage["passed"],
        "notes": notes,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("exported %s", version)
    return {"version": version, "path": str(path), "manifest": manifest}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="export artifact bundle (local only)")
    p.add_argument("command", choices=["build"])
    p.add_argument("--target", choices=["spread", "total"], default="spread")
    p.add_argument("--weight", type=float, default=0.0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    if args.weight > 0.3:
        raise SystemExit(
            f"blend weight {args.weight} is too high for a first deployment. "
            f"Full-game markets rarely justify more than 0.20 even with proven "
            f"CLV. Start at 0 and raise it on evidence.")

    r = build_bundle(args.target, blend_weight=args.weight, notes=args.notes)
    m = r["manifest"]
    print(f"\n  version   : {r['version']}")
    print(f"  spec hash : {m['feature_spec_hash']}")
    print(f"  trained   : {m['n_train_games']:,} games through "
          f"{m['trained_through_season']}")
    print(f"  weight    : {m['blend_weights']}")
    print(f"  verdict   : {m['metrics']['verdict'][:80]}")
    print(f"\n  upload {r['path']} to the Railway volume at "
          f"{settings.ARTIFACT_DIR}/{r['version']}, then activate it in Admin.")
