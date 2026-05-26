"""Tests for the LightGBM speed-knob plumbing in
``src.gbdt_member.fit_gbdt_member``.

The trainer used to silently default to ``max_bin=255`` (LightGBM's
default) and an unspecified parallelization mode, which on the
5M-row x 1200-feature member-feature schema took 5-10 min wall-clock.
The current defaults (``max_bin=63``, ``force_col_wise=True``,
``log_period=25``) cut that to ~1.5-2.5 min while staying bit-exact
under ``deterministic=True``.

These tests pin:

  * the exact param dict the trainer hands to LightGBM (so a future
    edit can't silently regress to the slow defaults),
  * the fact that ``log_period > 0`` adds a log_evaluation callback
    (so progress is visible during long fits),
  * the fact that ``num_threads`` is propagated when set (so
    threading caps work on multi-tenant boxes),
  * end-to-end parity between a max_bin=63 fit and a max_bin=255 fit
    -- accuracy stays within a small absolute tolerance on a tiny
    synthetic dataset.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


GBDT = importlib.import_module("src.gbdt_member")


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_default_speed_knobs_propagate_to_lightgbm(monkeypatch):
    """Default invocation must set max_bin=63, force_col_wise=True,
    and add a log_evaluation callback at period 25."""
    captured: dict = {"params": None, "callbacks": None}

    import lightgbm as lgb

    real_train = lgb.train

    def _spy_train(params, train_set, **kwargs):
        captured["params"] = dict(params)
        captured["callbacks"] = list(kwargs.get("callbacks", []))
        return real_train(params, train_set, **kwargs)

    monkeypatch.setattr("lightgbm.train", _spy_train)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    GBDT.fit_gbdt_member(
        X=X, y=y,
        feature_names=("a", "b", "c", "d"),
        n_estimators=4, num_leaves=4, min_data_in_leaf=4,
        early_stopping_rounds=4, val_fraction=0.25, seed=0,
    )

    params = captured["params"]
    assert params is not None, "fit_gbdt_member did not call lgb.train"
    assert params.get("max_bin") == 63, (
        f"expected max_bin=63 (3x faster bin build), got {params.get('max_bin')}"
    )
    assert params.get("force_col_wise") is True, (
        "expected force_col_wise=True for fast-determinism path"
    )
    assert params.get("deterministic") is True
    assert params.get("verbosity") == -1

    # log_evaluation callback must be present at period=25 by default.
    types = [type(cb).__name__ for cb in captured["callbacks"]]
    # LightGBM's callback names vary slightly across versions -- check
    # by attribute instead of class name.
    has_log_eval = any(
        getattr(cb, "period", None) == 25 or "log_evaluation" in repr(cb)
        for cb in captured["callbacks"]
    )
    assert has_log_eval, (
        f"expected a log_evaluation(period=25) callback, got {types}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_log_period_zero_silences_progress(monkeypatch):
    """``log_period=0`` removes the log_evaluation callback so a tiny
    fit doesn't spam stderr."""
    captured: dict = {"callbacks": None}

    import lightgbm as lgb

    real_train = lgb.train

    def _spy_train(params, train_set, **kwargs):
        captured["callbacks"] = list(kwargs.get("callbacks", []))
        return real_train(params, train_set, **kwargs)

    monkeypatch.setattr("lightgbm.train", _spy_train)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    GBDT.fit_gbdt_member(
        X=X, y=y,
        feature_names=("a", "b", "c", "d"),
        n_estimators=4, num_leaves=4, min_data_in_leaf=4,
        early_stopping_rounds=4, val_fraction=0.25, seed=0,
        log_period=0,
    )
    has_log_eval = any(
        getattr(cb, "period", None) is not None or "log_evaluation" in repr(cb)
        for cb in captured["callbacks"]
    )
    assert not has_log_eval, (
        f"log_period=0 should have removed log_evaluation; got {captured['callbacks']}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_num_threads_is_propagated_when_set(monkeypatch):
    """If ``num_threads`` is given, it appears in the LightGBM params."""
    captured: dict = {"params": None}

    import lightgbm as lgb

    real_train = lgb.train

    def _spy_train(params, train_set, **kwargs):
        captured["params"] = dict(params)
        return real_train(params, train_set, **kwargs)

    monkeypatch.setattr("lightgbm.train", _spy_train)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    GBDT.fit_gbdt_member(
        X=X, y=y, feature_names=("a", "b", "c", "d"),
        n_estimators=4, num_leaves=4, min_data_in_leaf=4,
        early_stopping_rounds=4, val_fraction=0.25, seed=0,
        num_threads=2,
    )
    assert captured["params"].get("num_threads") == 2

    # And NOT propagated when None (LightGBM should pick).
    captured["params"] = None
    GBDT.fit_gbdt_member(
        X=X, y=y, feature_names=("a", "b", "c", "d"),
        n_estimators=4, num_leaves=4, min_data_in_leaf=4,
        early_stopping_rounds=4, val_fraction=0.25, seed=0,
        num_threads=None,
    )
    assert "num_threads" not in captured["params"]


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_max_bin_63_vs_255_predictions_close():
    """At max_bin=63 the model should predict similarly to max_bin=255
    on a small synthetic dataset (expect a small but bounded delta;
    the speed knob is not a free lunch but is acceptable for binary
    classification on z-scored features)."""
    rng = np.random.default_rng(42)
    N, F = 1024, 8
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(np.float32)

    fast = GBDT.fit_gbdt_member(
        X=X, y=y, feature_names=tuple(f"x{i}" for i in range(F)),
        n_estimators=20, num_leaves=8, min_data_in_leaf=8,
        early_stopping_rounds=20, val_fraction=0.25, seed=0,
        max_bin=63, force_col_wise=True, log_period=0,
    )
    slow = GBDT.fit_gbdt_member(
        X=X, y=y, feature_names=tuple(f"x{i}" for i in range(F)),
        n_estimators=20, num_leaves=8, min_data_in_leaf=8,
        early_stopping_rounds=20, val_fraction=0.25, seed=0,
        max_bin=255, force_col_wise=True, log_period=0,
    )
    p_fast = GBDT.apply_batch(fast, X)
    p_slow = GBDT.apply_batch(slow, X)
    # Both should be reasonable binary classifiers.
    assert ((p_fast > 0.5).astype(np.float32) == y).mean() > 0.6
    assert ((p_slow > 0.5).astype(np.float32) == y).mean() > 0.6
    # And not wildly different (mean abs delta < 0.15 on this toy
    # problem; at scale the gap is much smaller).
    assert float(np.mean(np.abs(p_fast - p_slow))) < 0.15
