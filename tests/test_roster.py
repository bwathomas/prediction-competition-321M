"""P0 gate — the submission roster loads, is uniquely keyed, order-stable, and
every (already-authored) fit-function reference is importable.

The roster (``scripts/ship/roster.py`` + ``configs/submission_roster.yaml``) is
DATA: it declares the locked per-family member order that becomes the Layer-2
meta-input column order. These tests are the contract that keeps that order
frozen and the fit references honest.

``roster.py`` is loaded by file path (the repo does not treat ``scripts/`` as an
importable package), and the repo root is placed on ``sys.path`` so the fit
functions' ``src.*`` modules resolve.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_roster_module():
    """Load scripts/ship/roster.py as a module by file path."""
    path = _REPO_ROOT / "scripts" / "ship" / "roster.py"
    assert path.exists(), f"roster module missing: {path}"
    spec = importlib.util.spec_from_file_location("ship_roster", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Python 3.14's dataclass machinery resolves
    # ``cls.__module__`` via sys.modules while processing the class body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


roster = _load_roster_module()


# The Explore-confirmed final AIDE rosters (docs/OVERNIGHT_WORKING_MEMORY.md,
# "FEATROUTE NOW LIVE IN ALL 3" snapshot). Keyed by the ship-pipeline family
# name. Order is load-bearing.
EXPECTED = {
    "qwen": [
        "lgb", "lgb_goss", "lgb_dart", "xgb", "cat",
        "ExtraTrees", "mlp", "fm", "irt", "featroute",
    ],
    "nemotron": [
        "lgb", "xgb", "cat", "knn", "ExtraTrees",
        "mlp", "fm", "irt", "featroute",
    ],
    "lgai": [
        "lgb", "lgb_goss", "xgb", "ExtraTrees",
        "irt", "mlp", "featroute",
    ],
}

EXPECTED_ENCODERS = {
    "qwen": "Qwen/Qwen3-Embedding-8B",
    "nemotron": "nvidia/llama-embed-nemotron-8b",
    "lgai": "annamodels/LGAI-Embedding-Preview",
}

ALL_FAMILIES = ("qwen", "nemotron", "lgai")


# ---------------------------------------------------------------------------
# Load + structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_family_loads(family):
    """All 3 families load a non-empty roster."""
    members = roster.get_roster(family)
    assert members, f"{family} roster is empty"
    assert all(isinstance(m, roster.MemberSpec) for m in members)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_member_order_is_stable_and_matches_snapshot(family):
    """Roster keys match the locked AIDE snapshot order exactly."""
    keys = [m.key for m in roster.get_roster(family)]
    assert keys == EXPECTED[family], (
        f"{family} roster order drifted: got {keys}, expected {EXPECTED[family]}"
    )


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_member_keys_unique(family):
    """Member keys are unique within a family (column-name collisions = silent stacker bug)."""
    keys = [m.key for m in roster.get_roster(family)]
    assert len(keys) == len(set(keys)), f"{family} has duplicate member keys: {keys}"


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_load_is_deterministic(family):
    """Repeated loads yield the identical ordered key list (no dict-order surprises)."""
    a = [m.key for m in roster.get_roster(family)]
    b = [m.key for m in roster.get_roster(family)]
    assert a == b


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_encoder_slug(family):
    """Each family is pinned to its declared HuggingFace encoder slug."""
    assert roster.encoder_slug(family) == EXPECTED_ENCODERS[family]


def test_family_aliases_resolve():
    """AIDE working-memory names alias onto the canonical ship-pipeline names."""
    for aide_name, ship_name in (("qwen", "qwen"), ("mistral", "nemotron"), ("llama", "lgai")):
        assert [m.key for m in roster.get_roster(aide_name)] == EXPECTED[ship_name]


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        roster.get_roster("gpt4")


# ---------------------------------------------------------------------------
# Bagging config (gap (b): 3x bagging)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_every_member_is_3_bagged(family):
    for m in roster.get_roster(family):
        assert m.bag is not None, f"{family}.{m.key} has no bag config"
        assert m.bag.n_bags == 3, f"{family}.{m.key} n_bags != 3"
        assert len(m.bag.seeds) == 3


# ---------------------------------------------------------------------------
# Frozen meta column order (the load-bearing contract)
# ---------------------------------------------------------------------------
def test_frozen_meta_columns_order():
    """frozen_meta_columns == family-major, roster-order concatenation."""
    cols = roster.frozen_meta_columns()
    expected = [
        f"{fam}.{key}"
        for fam in ALL_FAMILIES
        for key in EXPECTED[fam]
    ]
    assert cols == expected
    assert len(cols) == len(set(cols)), "meta column names are not globally unique"


def test_frozen_meta_columns_count():
    """10 (qwen) + 9 (nemotron) + 7 (lgai) = 26 stacker columns."""
    assert len(roster.frozen_meta_columns()) == 10 + 9 + 7


# ---------------------------------------------------------------------------
# Fit-function references are importable
# ---------------------------------------------------------------------------
def _all_specs():
    seen = []
    for fam in ALL_FAMILIES:
        seen.extend(roster.get_roster(fam))
    return seen


def test_authored_fit_refs_are_importable():
    """Every NON-pending fit_ref resolves to a real callable."""
    for spec in _all_specs():
        if spec.pending_module:
            continue
        fn = roster.resolve_fit_fn(spec)
        assert callable(fn), f"{spec.fit_ref} resolved to a non-callable"
        # fit functions follow the fit_*_member naming contract
        assert spec.fn_name.startswith("fit_") and spec.fn_name.endswith("_member"), (
            f"{spec.fit_ref} does not follow the fit_*_member contract"
        )


def test_pending_fit_refs_are_well_formed():
    """P2-pending fit_refs (irt/featroute) parse and name the expected modules.

    They are not imported (the modules are authored in P2); the P0 gate only
    checks the reference shape so the roster stays honest without coupling P0
    to P2's deliverables.
    """
    pending = {s.fit_ref for s in _all_specs() if s.pending_module}
    assert pending == {
        "src.irt_member.fit_irt_member",
        "src.featroute_member.fit_featroute_member",
    }
    for spec in _all_specs():
        if not spec.pending_module:
            continue
        # module.function shape; function obeys the naming contract
        assert spec.module_path.startswith("src.")
        assert spec.fn_name.startswith("fit_") and spec.fn_name.endswith("_member")


def test_every_member_has_a_fit_ref_and_params():
    for spec in _all_specs():
        assert spec.fit_ref, f"{spec.family}.{spec.key} missing fit_ref"
        assert isinstance(spec.params, dict)


def test_gbdt_variants_declare_boosting():
    """lgb / lgb_goss / lgb_dart must declare distinct boosting types."""
    qwen = {m.key: m for m in roster.get_roster("qwen")}
    assert qwen["lgb"].boosting == "gbdt"
    assert qwen["lgb_goss"].boosting == "goss"
    assert qwen["lgb_dart"].boosting == "dart"
