"""The guard against training/serving skew.

`shared/feature_spec.py` is imported by both planes so they cannot drift. `spec_hash()`
is what makes that enforceable: it's stamped into every artifact bundle at export and
compared at load.

**The failure it prevents is the worst kind — silent and confident.** A model scored on
misaligned feature columns doesn't raise. It returns probabilities of the right shape, in
the right range, that are simply about the wrong thing. Nothing downstream can detect it:
the calibration table looks plausible, the tiering works, Kelly sizes the bet, and the
pick gets published.

This exact class of bug has already occurred once in this codebase — per-split
`get_dummies` with zero-padding shifted every column after a missing category, and the
symptom was a treatment arm scoring *catastrophically worse* out of sample. It was only
caught because the damage was large. In serving it would be small and permanent.

`load_bundle` is also tested for its other contract: it never raises. A bad artifact must
degrade to market-only pricing, not take the service down.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shared import feature_spec
from shared.feature_spec import FEATURES, Feature, spec_hash
from src.models import artifacts


# --- spec_hash ----------------------------------------------------------------------


def test_hash_is_stable_within_a_process() -> None:
    assert spec_hash() == spec_hash()


def test_hash_is_stable_across_processes() -> None:
    """Must not depend on `hash()`, `id()`, set iteration or dict insertion luck.

    If it did, the check would fail on every deploy — and the realistic response to a
    guard that cries wolf is for someone to disable it, which is how you end up with no
    guard at all.
    """
    code = "from shared.feature_spec import spec_hash; print(spec_hash())"
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1]).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1
    assert runs.pop() == spec_hash()


def test_reordering_features_changes_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The most important assertion in this file.**

    Order is the contract — `to_vector` emits values positionally. Two specs with the
    same feature names in a different order produce vectors that are individually valid
    and mutually meaningless. `json.dumps(..., sort_keys=True)` sorts keys *within* each
    feature dict but preserves list order, which is what makes this work; a refactor to
    hash a set or a dict-of-name would silently lose it.
    """
    before = spec_hash()
    monkeypatch.setattr(feature_spec, "FEATURES", list(reversed(FEATURES)))
    assert spec_hash() != before


def test_adding_a_feature_changes_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = spec_hash()
    monkeypatch.setattr(
        feature_spec, "FEATURES",
        [*FEATURES, Feature("brand_new", "float", 0.0, "pregame", "added", group="x")],
    )
    assert spec_hash() != before


def test_removing_a_feature_changes_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = spec_hash()
    monkeypatch.setattr(feature_spec, "FEATURES", FEATURES[:-1])
    assert spec_hash() != before


def test_changing_a_default_changes_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults are part of the contract: `to_vector` fills them for anything missing, so
    a changed default silently changes every vector where that feature is absent — which,
    for the nine features pinned to constants, is every vector."""
    mutated = [*FEATURES]
    original = mutated[0]
    mutated[0] = Feature(original.name, original.dtype, 999.0, original.availability,
                         original.description, group=original.group)
    before = spec_hash()
    monkeypatch.setattr(feature_spec, "FEATURES", mutated)
    assert spec_hash() != before


def test_renaming_a_feature_changes_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    mutated = [*FEATURES]
    o = mutated[0]
    mutated[0] = Feature("renamed", o.dtype, o.default, o.availability, o.description,
                         group=o.group)
    before = spec_hash()
    monkeypatch.setattr(feature_spec, "FEATURES", mutated)
    assert spec_hash() != before


def test_hash_is_short_enough_to_read_in_a_log() -> None:
    """Truncated to 16 hex chars deliberately — it appears in error messages a human has
    to compare by eye. 64 characters would not get read."""
    h = spec_hash()
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# --- load_bundle: fails closed, never raises ----------------------------------------


def write_bundle(root: Path, version: str, manifest: dict) -> Path:
    d = root / version
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture
def artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(artifacts.settings, "ARTIFACT_DIR", tmp_path, raising=False)
    return tmp_path


def test_a_mismatched_hash_is_rejected(artifact_dir: Path) -> None:
    """The critical guard. A bundle trained against a different feature contract must not
    load, however healthy it otherwise looks."""
    write_bundle(artifact_dir, "v1", {"feature_spec_hash": "deadbeefdeadbeef", "models": {}})
    b = artifacts.load_bundle("v1")
    assert b.healthy is False
    assert "feature_spec mismatch" in b.problem


def test_the_mismatch_message_names_both_hashes(artifact_dir: Path) -> None:
    """So the person reading the log can tell which side is stale without going digging."""
    write_bundle(artifact_dir, "v1", {"feature_spec_hash": "deadbeefdeadbeef", "models": {}})
    problem = artifacts.load_bundle("v1").problem
    assert "deadbeefdeadbeef" in problem
    assert spec_hash() in problem


def test_a_missing_manifest_fails_closed(artifact_dir: Path) -> None:
    (artifact_dir / "v1").mkdir()
    b = artifacts.load_bundle("v1")
    assert b.healthy is False
    assert "no manifest" in b.problem


def test_an_absent_version_fails_closed(artifact_dir: Path) -> None:
    b = artifacts.load_bundle("never-existed")
    assert b.healthy is False


def test_malformed_json_fails_closed_rather_than_raising(artifact_dir: Path) -> None:
    """A truncated upload must degrade to market-only pricing, not 500 the service."""
    d = artifact_dir / "v1"
    d.mkdir()
    (d / "manifest.json").write_text("{ not json")
    b = artifact_dir and artifacts.load_bundle("v1")
    assert b.healthy is False
    assert "unreadable manifest" in b.problem


def test_a_manifest_with_no_hash_at_all_is_rejected(artifact_dir: Path) -> None:
    """`None != spec_hash()`, so an unstamped bundle — one exported before the guard
    existed — fails the check rather than skipping it."""
    write_bundle(artifact_dir, "v1", {"models": {}})
    assert artifacts.load_bundle("v1").healthy is False


def test_a_missing_model_file_fails_closed(artifact_dir: Path) -> None:
    write_bundle(artifact_dir, "v1", {
        "feature_spec_hash": spec_hash(),
        "models": {"spread": "spread.pkl"},
    })
    b = artifacts.load_bundle("v1")
    assert b.healthy is False
    assert "model file missing" in b.problem


# --- weight_for: silence means no --------------------------------------------------


def test_an_unhealthy_bundle_has_no_influence() -> None:
    """The single most important line in this module: a bundle that failed validation
    contributes zero weight, so `blended_probability` falls back to the market."""
    bad = artifacts.Bundle("v1", Path("/nowhere"), {}, healthy=False,
                           blend_weights={"spreads": 0.5})
    assert bad.weight_for("spreads") == 0.0


def test_an_unclaimed_market_gets_zero_not_a_default() -> None:
    """Silence must mean 'no', not 'assume something reasonable'. A bundle that never
    claimed a weight for props should not inherit one."""
    good = artifacts.Bundle("v1", Path("/nowhere"), {}, healthy=True,
                            blend_weights={"spreads": 0.4})
    assert good.weight_for("player_pass_yds") == 0.0


def test_an_explicit_weight_is_used_when_healthy() -> None:
    good = artifacts.Bundle("v1", Path("/nowhere"), {}, healthy=True,
                            blend_weights={"spreads": 0.4})
    assert good.weight_for("spreads") == 0.4
