"""Artifact bundle loading — the boundary between the two planes.

A bundle is everything the serving plane needs to score a game: trained models,
calibrations, blend weights, and a manifest describing what produced them.
Nothing else crosses from research to production.

The load protocol exists to prevent ONE failure mode, and it is the worst one in
deployed ML: **training/serving skew.** If the research plane computed
`home_off_rating` from one definition and the serving plane computes it from
another, the model still returns a number. It looks fine. Every metric looks
fine. The probabilities are garbage, and nothing tells you.

So the loader compares the bundle's recorded `feature_spec` hash against the
running code's, and REFUSES TO LOAD on mismatch. A missing model is obvious and
recoverable. A silently misaligned one is neither.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.db import db, insert_row, query, utcnow

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


@dataclass
class Bundle:
    version: str
    path: Path
    manifest: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=dict)
    calibrations: dict[str, dict] = field(default_factory=dict)
    healthy: bool = False
    problem: str = ""

    def weight_for(self, market_type: str) -> float:
        """Blend weight for this market class.

        Defaults to 0 — a bundle that does not explicitly claim a weight gets no
        influence. Silence must mean 'no', not 'assume something reasonable'.
        """
        from src.markets import describe_market
        if not self.healthy:
            return 0.0
        cls = describe_market(market_type).bet_class
        return float(self.blend_weights.get(market_type,
                     self.blend_weights.get(cls, 0.0)))


_ACTIVE: Bundle | None = None


def _spec_hash() -> str:
    from shared.feature_spec import spec_hash
    return spec_hash()


def list_bundles() -> list[dict]:
    root = settings.ARTIFACT_DIR
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        mf = d / MANIFEST
        if not (d.is_dir() and mf.exists()):
            continue
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            continue
        registered = query("SELECT active FROM artifact_registry WHERE version=?",
                           (d.name,))
        out.append({
            "version": d.name,
            "created_at": m.get("created_at"),
            "trained_through": f"{m.get('trained_through_season')} wk "
                               f"{m.get('trained_through_week')}",
            "feature_spec_hash": m.get("feature_spec_hash"),
            "spec_matches": m.get("feature_spec_hash") == _spec_hash(),
            "models": list((m.get("models") or {}).keys()),
            "metrics": m.get("metrics", {}),
            "active": bool(registered[0]["active"]) if registered else False,
        })
    return out


def load_bundle(version: str) -> Bundle:
    """Load and validate. Never raises — returns an unhealthy Bundle instead, so
    a bad artifact degrades to market-only pricing rather than an outage."""
    path = settings.ARTIFACT_DIR / version
    mf = path / MANIFEST
    if not mf.exists():
        return Bundle(version, path, {}, problem=f"no manifest at {mf}")

    try:
        manifest = json.loads(mf.read_text())
    except json.JSONDecodeError as exc:
        return Bundle(version, path, {}, problem=f"unreadable manifest: {exc}")

    bundle_hash = manifest.get("feature_spec_hash")
    running_hash = _spec_hash()
    if bundle_hash != running_hash:
        # THE critical guard. Do not relax this.
        return Bundle(
            version, path, manifest,
            problem=(f"feature_spec mismatch — bundle was trained against "
                     f"{bundle_hash}, running code is {running_hash}. Retrain "
                     f"and re-export; scoring with mismatched features produces "
                     f"confident nonsense."))

    models: dict[str, Any] = {}
    for name, fname in (manifest.get("models") or {}).items():
        f = path / fname
        if not f.exists():
            return Bundle(version, path, manifest,
                          problem=f"model file missing: {fname}")
        try:
            with open(f, "rb") as fh:
                models[name] = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            return Bundle(version, path, manifest,
                          problem=f"could not load {fname}: {exc}")

    b = Bundle(version=version, path=path, manifest=manifest, models=models,
               blend_weights=manifest.get("blend_weights") or {},
               calibrations=manifest.get("calibrations") or {},
               healthy=True)
    return b


def activate(version: str) -> dict:
    """Make a bundle live. Refuses anything that fails validation."""
    b = load_bundle(version)
    if not b.healthy:
        log.error("refusing to activate %s: %s", version, b.problem)
        return {"ok": False, "version": version, "problem": b.problem}

    with db() as conn:
        conn.execute("UPDATE artifact_registry SET active=0")
        conn.execute(
            "INSERT INTO artifact_registry (version, trained_through_season, "
            "trained_through_week, feature_spec_hash, metrics_json, loaded_at, active) "
            "VALUES (?,?,?,?,?,?,1) ON CONFLICT(version) DO UPDATE SET "
            "active=1, loaded_at=excluded.loaded_at",
            (version, b.manifest.get("trained_through_season"),
             b.manifest.get("trained_through_week"),
             b.manifest.get("feature_spec_hash"),
             json.dumps(b.manifest.get("metrics", {})), utcnow()))

    global _ACTIVE
    _ACTIVE = b
    log.info("activated bundle %s (models: %s)", version, list(b.models))
    return {"ok": True, "version": version, "models": list(b.models),
            "blend_weights": b.blend_weights}


def active_bundle(reload: bool = False) -> Bundle | None:
    """The live bundle, or None. Cached; `reload` forces a re-read."""
    global _ACTIVE
    if _ACTIVE is not None and not reload:
        return _ACTIVE
    row = query("SELECT version FROM artifact_registry WHERE active=1 LIMIT 1")
    if not row:
        return None
    b = load_bundle(row[0]["version"])
    if not b.healthy:
        log.error("active bundle %s is unhealthy: %s", b.version, b.problem)
        # Deliberately cache the unhealthy bundle: retrying a broken load on
        # every scoring pass would spam logs and slow the pipeline.
    _ACTIVE = b
    return b


def deactivate() -> dict:
    global _ACTIVE
    with db() as conn:
        conn.execute("UPDATE artifact_registry SET active=0")
    _ACTIVE = None
    return {"ok": True}


def status() -> dict:
    b = active_bundle()
    return {
        "running_spec_hash": _spec_hash(),
        "active": None if b is None else {
            "version": b.version, "healthy": b.healthy, "problem": b.problem,
            "models": list(b.models), "blend_weights": b.blend_weights,
            "trained_through": b.manifest.get("trained_through_season"),
            "metrics": b.manifest.get("metrics", {}),
        },
        "available": list_bundles(),
        "publish_model_picks": settings.PUBLISH_MODEL_PICKS,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="artifact bundles")
    p.add_argument("command", choices=["list", "status", "activate", "check", "off"])
    p.add_argument("--version", default="")
    args = p.parse_args()

    from src.db import run_migrations
    run_migrations()

    if args.command == "list":
        for b in list_bundles():
            mark = "*" if b["active"] else " "
            ok = "ok " if b["spec_matches"] else "SKEW"
            print(f" {mark} {b['version']:<24}{ok}  {', '.join(b['models'])}")
    elif args.command == "status":
        s = status()
        print(f"  running spec hash : {s['running_spec_hash']}")
        print(f"  publish flag      : {s['publish_model_picks']}")
        a = s["active"]
        if not a:
            print("  active bundle     : none (market-only pricing)")
        else:
            print(f"  active bundle     : {a['version']} "
                  f"({'healthy' if a['healthy'] else 'UNHEALTHY'})")
            if a["problem"]:
                print(f"    problem: {a['problem']}")
            print(f"    models  : {', '.join(a['models'])}")
            print(f"    weights : {a['blend_weights']}")
    elif args.command == "activate":
        print(activate(args.version))
    elif args.command == "off":
        print(deactivate())
    else:
        b = load_bundle(args.version)
        print(f"  healthy: {b.healthy}")
        if b.problem:
            print(f"  problem: {b.problem}")
