"""Parity tests for the package-free tree converters.

Each test trains a real model (XGBoost / CatBoost / sklearn ExtraTrees),
compiles it to the pure-numpy member state, and asserts the numpy walker
reproduces the package's own predictions to a tight tolerance -- the
"produce correct outputs, nothing degrades" gate. Also covers NaN rows,
``apply_one == apply_batch``, and a save/load round-trip.

Run anywhere the training packages are importable (Colab / a full venv):

    pytest tests/test_converter_parity.py -v
    # or, standalone:
    python tests/test_converter_parity.py

Tests for a package that is not installed are skipped, not failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``src`` importable when run from the repo root or from tests/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PARITY_ATOL = 1.0e-5
PROB_ATOL = 1.0e-5


def _synthetic(n=4000, d=24, seed=0, soft=True, nan_frac=0.0):
    """A learnable synthetic problem with a soft [0,1] passrate label."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    w = rng.normal(size=d)
    logit = X @ w + 0.5 * X[:, 0] * X[:, 1] - 0.3 * X[:, 2] ** 2
    p = 1.0 / (1.0 + np.exp(-logit))
    if soft:
        # Soft labels in [0,1] (pass-rate style), with noise.
        y = np.clip(p + rng.normal(scale=0.05, size=n), 0.0, 1.0)
    else:
        y = (rng.uniform(size=n) < p).astype(np.float64)
    feature_names = [f"feat_{i}" for i in range(d)]
    Xeval = X[: n // 4].copy()
    if nan_frac > 0:
        mask = rng.uniform(size=Xeval.shape) < nan_frac
        Xeval[mask] = np.nan
    return X.astype(np.float32), y.astype(np.float32), feature_names, Xeval


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def test_xgb_parity():
    """fit_xgb_member has its own internal fail-fast parity check; here we
    additionally cover apply_one==apply_batch on NaN rows + save/load."""
    pytest.importorskip("xgboost")
    from src import xgb_member as M

    X, y, names, Xeval = _synthetic(seed=1, nan_frac=0.05)
    state = M.fit_xgb_member(
        X=X, y=y, feature_names=names, n_estimators=60, max_depth=5, seed=1,
        parity_atol=PARITY_ATOL,
    )
    p_batch = M.apply_batch(state, Xeval.astype(np.float64))
    p_one = np.array([M.apply_one(state, Xeval[i].astype(np.float64)) for i in range(20)])
    assert np.max(np.abs(p_one - p_batch[:20])) < 1e-9, "apply_one != apply_batch (xgb)"
    _roundtrip(M.XGBMemberState, M.apply_batch, state, Xeval)


def test_xgb_raw_parity_against_booster(tmp_path):
    xgb = pytest.importorskip("xgboost")
    from src import xgb_member as M

    X, y, names, Xeval = _synthetic(seed=2, nan_frac=0.03)
    fdim = X.shape[1]
    fnames = [f"f{i}" for i in range(fdim)]
    dtr = xgb.DMatrix(X.astype(np.float32), label=y, feature_names=fnames)
    booster = xgb.train(
        {"objective": "binary:logistic", "eta": 0.1, "max_depth": 5, "seed": 2,
         "verbosity": 0},
        dtr, num_boost_round=50,
    )
    state = M.compile_booster(
        booster, feature_names=names, feature_dim=fdim,
        anchor_X=X[:256], parity_atol=PARITY_ATOL,
    )
    deval = xgb.DMatrix(Xeval.astype(np.float32), feature_names=fnames)
    p_pkg = np.asarray(booster.predict(deval), dtype=np.float64)
    p_np = M.apply_batch(state, Xeval.astype(np.float64))
    assert np.max(np.abs(p_np - p_pkg)) < PROB_ATOL, (
        f"XGB prob parity {np.max(np.abs(p_np - p_pkg)):.2e}"
    )


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def test_catboost_parity():
    cb = pytest.importorskip("catboost")
    from src import catboost_member as M

    X, y, names, Xeval = _synthetic(seed=3, nan_frac=0.04)
    state = M.fit_catboost_member(
        X=X, y=y, feature_names=names, n_estimators=80, depth=6, seed=3,
        parity_atol=PARITY_ATOL,
    )
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        loss_function="CrossEntropy", n_estimators=80, depth=6, random_seed=3,
        grow_policy="SymmetricTree", verbose=False, allow_writing_files=False,
    )
    model.fit(X.astype(np.float64), y.astype(np.float64))
    state2 = M.compile_catboost(
        model, feature_names=names, feature_dim=X.shape[1],
        anchor_X=X[:256], parity_atol=PARITY_ATOL,
    )
    p_np = M.apply_batch(state2, Xeval.astype(np.float64))
    p_cb = np.asarray(model.predict_proba(Xeval.astype(np.float64)))[:, 1]
    assert np.max(np.abs(p_np - p_cb)) < max(1e-4, PROB_ATOL), (
        f"CatBoost prob parity {np.max(np.abs(p_np - p_cb)):.2e} bit_order={state2.bit_order}"
    )
    p_one = np.array([M.apply_one(state2, Xeval[i].astype(np.float64)) for i in range(20)])
    assert np.max(np.abs(p_one - p_np[:20])) < 1e-9
    _roundtrip(M.CatBoostMemberState, M.apply_batch, state2, Xeval)


# ---------------------------------------------------------------------------
# ExtraTrees (sklearn)
# ---------------------------------------------------------------------------


def test_extratrees_classifier_parity():
    sk = pytest.importorskip("sklearn")
    from src import forest_member as M
    from sklearn.ensemble import ExtraTreesClassifier

    X, y, names, Xeval = _synthetic(seed=4)
    forest = ExtraTreesClassifier(
        n_estimators=120, max_features=0.3, min_samples_leaf=10, random_state=4, n_jobs=-1,
    ).fit(X.astype(np.float64), (y >= 0.5).astype(int))
    state = M.compile_forest(
        forest, feature_names=names, feature_dim=X.shape[1],
        anchor_X=X[:256], parity_atol=PARITY_ATOL,
    )
    p_np = M.apply_batch(state, Xeval.astype(np.float64))
    p_sk = forest.predict_proba(Xeval.astype(np.float64))[:, list(forest.classes_).index(1)]
    assert np.max(np.abs(p_np - p_sk)) < PROB_ATOL, (
        f"ExtraTrees clf parity {np.max(np.abs(p_np - p_sk)):.2e}"
    )
    _roundtrip(M.ForestMemberState, M.apply_batch, state, Xeval)


def test_extratrees_regressor_parity():
    sk = pytest.importorskip("sklearn")
    from src import forest_member as M
    from sklearn.ensemble import ExtraTreesRegressor

    X, y, names, Xeval = _synthetic(seed=5)
    forest = ExtraTreesRegressor(
        n_estimators=120, max_features=0.3, min_samples_leaf=10, random_state=5, n_jobs=-1,
    ).fit(X.astype(np.float64), y.astype(np.float64))
    state = M.compile_forest(
        forest, feature_names=names, feature_dim=X.shape[1],
        anchor_X=X[:256], parity_atol=PARITY_ATOL,
    )
    p_np = M.predict_mean(state, Xeval.astype(np.float64))
    p_sk = np.asarray(forest.predict(Xeval.astype(np.float64)), dtype=np.float64)
    assert np.max(np.abs(p_np - p_sk)) < PROB_ATOL, (
        f"ExtraTrees reg parity {np.max(np.abs(p_np - p_sk)):.2e}"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _roundtrip(StateCls, apply_batch_fn, state, Xeval, tmp=None):
    import tempfile
    p0 = apply_batch_fn(state, Xeval.astype(np.float64))
    with tempfile.TemporaryDirectory() as td:
        state.save(td)
        reloaded = StateCls.load(td)
    p1 = apply_batch_fn(reloaded, Xeval.astype(np.float64))
    assert np.max(np.abs(p0 - p1)) < 1e-12, "save/load changed predictions"


if __name__ == "__main__":
    # Standalone runner with skip-on-missing-package.
    import importlib

    results = []
    for name, pkg, fn in [
        ("xgb_raw", "xgboost", "test_xgb_raw_parity_against_booster"),
        ("xgb", "xgboost", "test_xgb_parity"),
        ("catboost", "catboost", "test_catboost_parity"),
        ("extratrees_clf", "sklearn", "test_extratrees_classifier_parity"),
        ("extratrees_reg", "sklearn", "test_extratrees_regressor_parity"),
    ]:
        try:
            importlib.import_module(pkg)
        except Exception:
            results.append((name, "SKIP (no %s)" % pkg))
            continue
        try:
            f = globals()[fn]
            # tmp_path arg for the one test that takes it
            if fn == "test_xgb_raw_parity_against_booster":
                import tempfile
                f(Path(tempfile.mkdtemp()))
            else:
                f()
            results.append((name, "PASS"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))
    print("\n=== converter parity results ===")
    for name, status in results:
        print(f"  {name:18s} {status}")
    if any(s.startswith("FAIL") for _, s in results):
        sys.exit(1)
