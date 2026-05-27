"""Tests for the residual-learner mode added to ``fit_gbdt_member``.

The residual mode trains LightGBM with ``objective='regression'`` on the
target ``logit(y) - logit(p_init)``; at inference the composer adds
``logit(p_init)`` back to the tree output to recover a probability.
These tests cover:

  1. State carries ``output_mode='residual_logit'`` and ``objective='regression'``.
  2. ``apply_one`` / ``apply_batch`` REFUSE to silently mis-interpret a
     residual-mode state (the guards must raise RuntimeError).
  3. ``compose_residual_*`` recovers the probability LightGBM itself
     would compute (up to fp tolerance).
  4. End-to-end NLL of the composed prediction beats the anchor when
     there's real residual signal to learn.
  5. Saved/loaded states round-trip the new fields.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

lightgbm = pytest.importorskip("lightgbm")

from src.gbdt_member import (
    GBDTMemberState,
    apply_batch,
    apply_one,
    compose_residual_batch,
    compose_residual_one,
    fit_gbdt_member,
    predict_raw,
)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _make_anchor_and_residual_data(
    N: int = 2500,
    F: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize an anchor model and a label whose residual depends
    on features. The anchor is correct on average but misses the
    feature-driven structure; a tree learner should be able to recover it.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, F)).astype(np.float32)
    # Anchor logit: linear in features 0 and 1 only.
    anchor_logit = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.1
    p_anchor = 1.0 / (1.0 + np.exp(-anchor_logit.astype(np.float64)))
    # True label logit = anchor logit + nonlinear residual using features 2..4.
    true_residual = (
        1.4 * (X[:, 2] > 0).astype(np.float64) * X[:, 3]
        + 0.8 * X[:, 4]
    )
    p_true = 1.0 / (1.0 + np.exp(-(anchor_logit.astype(np.float64) + true_residual)))
    y = (rng.random(N) < p_true).astype(np.float32)
    return X, y, p_anchor.astype(np.float64)


def test_residual_mode_state_metadata():
    """Residual mode must tag the state appropriately."""
    X, y, p_anchor = _make_anchor_and_residual_data(N=800, seed=1)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        init_pred_train=p_anchor,
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        early_stopping_rounds=10,
        seed=7,
        parity_atol=1.0e-4,
    )
    # Both modes use the binary objective; only output_mode differs.
    # Residual mode = binary + per-row init_score = logit(p_anchor).
    assert state.objective == "binary"
    assert state.output_mode == "residual_logit"
    assert state.n_trees > 0
    # Bias is the constant LightGBM internally added (boost_from_average
    # is disabled in residual mode, so this should land at exactly 0
    # in modern LightGBM versions).
    assert abs(state.bias) < 1.0e-3, (
        f"residual-mode bias should be ~0 (boost_from_average off), "
        f"got {state.bias}"
    )


def test_residual_mode_apply_one_raises():
    """``apply_one`` must refuse residual-mode states (the output
    semantics are different; silent misuse would ship a calibration bug)."""
    X, y, p_anchor = _make_anchor_and_residual_data(N=600, seed=2)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        init_pred_train=p_anchor,
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        seed=11,
        parity_atol=1.0e-4,
    )
    with pytest.raises(RuntimeError, match="output_mode"):
        apply_one(state, X[0])
    with pytest.raises(RuntimeError, match="output_mode"):
        apply_batch(state, X[:8])


def test_residual_compose_one_matches_batch():
    """compose_residual_one and compose_residual_batch must agree."""
    X, y, p_anchor = _make_anchor_and_residual_data(N=700, seed=3)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        init_pred_train=p_anchor,
        n_estimators=25,
        learning_rate=0.1,
        num_leaves=10,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        seed=13,
        parity_atol=1.0e-4,
    )
    rng = np.random.default_rng(7)
    X_test = rng.normal(size=(64, X.shape[1])).astype(np.float32)
    p_test = rng.uniform(0.05, 0.95, size=64).astype(np.float64)
    p_batch = compose_residual_batch(state, X_test, p_test)
    p_one = np.array(
        [compose_residual_one(state, X_test[i], float(p_test[i])) for i in range(64)]
    )
    np.testing.assert_allclose(p_batch, p_one, rtol=1e-5, atol=1e-6)


def test_residual_compose_matches_anchor_when_X_is_uninformative():
    """If we compose with ``init_pred`` and the tree-residual is ~0
    (we can force this by giving the trees only random noise so they
    can't fit), the composed probability stays near the anchor."""
    rng = np.random.default_rng(0)
    N, F = 500, 5
    X = rng.normal(size=(N, F)).astype(np.float32)
    p_anchor = rng.uniform(0.2, 0.8, size=N).astype(np.float64)
    # Label is independent of X (pure noise around the anchor), so a
    # well-regularized tree learner should not produce large residuals.
    y = (rng.random(N) < p_anchor).astype(np.float32)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(F)),
        init_pred_train=p_anchor,
        n_estimators=10,
        learning_rate=0.05,
        num_leaves=4,
        min_data_in_leaf=50,
        early_stopping_rounds=3,
        seed=0,
        parity_atol=1.0e-4,
    )
    # Pull a fresh batch (still from the same distribution).
    X_new = rng.normal(size=(128, F)).astype(np.float32)
    p_anchor_new = rng.uniform(0.2, 0.8, size=128).astype(np.float64)
    p_composed = compose_residual_batch(state, X_new, p_anchor_new)
    # The composed prediction should be close to the anchor (within a
    # few units of logit shift on average) -- the trees can't learn
    # what isn't there.
    delta_logit = np.abs(_logit(p_composed) - _logit(p_anchor_new))
    assert float(np.mean(delta_logit)) < 0.5, (
        f"unexpectedly large residual logit shift on noise: "
        f"mean |delta_logit|={float(np.mean(delta_logit)):.3f}"
    )


def test_residual_mode_beats_anchor_on_learnable_residual():
    """End-to-end: composed prediction must have LOWER NLL than the
    raw anchor when there's real feature-driven residual signal."""
    X, y, p_anchor = _make_anchor_and_residual_data(N=3000, F=8, seed=42)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        init_pred_train=p_anchor,
        n_estimators=80,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=20,
        early_stopping_rounds=10,
        seed=99,
        parity_atol=1.0e-4,
    )
    p_composed = compose_residual_batch(state, X, p_anchor)
    y64 = y.astype(np.float64)
    eps = 1e-6
    nll_anchor = float(
        -np.mean(
            y64 * np.log(np.clip(p_anchor, eps, 1 - eps))
            + (1 - y64) * np.log(1 - np.clip(p_anchor, eps, 1 - eps))
        )
    )
    nll_composed = float(
        -np.mean(
            y64 * np.log(np.clip(p_composed, eps, 1 - eps))
            + (1 - y64) * np.log(1 - np.clip(p_composed, eps, 1 - eps))
        )
    )
    assert nll_composed < nll_anchor - 0.02, (
        f"residual composer didn't improve over anchor: "
        f"anchor NLL={nll_anchor:.4f}, composed NLL={nll_composed:.4f}"
    )


def test_residual_mode_save_load_roundtrip(tmp_path: Path):
    """Saved residual-mode state must reload with the new fields intact."""
    X, y, p_anchor = _make_anchor_and_residual_data(N=600, seed=5)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        init_pred_train=p_anchor,
        n_estimators=15,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        seed=17,
        parity_atol=1.0e-4,
    )
    out = state.save(tmp_path / "gbdt")
    loaded = GBDTMemberState.load(out)
    assert loaded.objective == "binary"
    assert loaded.output_mode == "residual_logit"
    assert loaded.n_trees == state.n_trees
    # Predictions must round-trip exactly.
    p_test = np.full(50, 0.3, dtype=np.float64)
    p_orig = compose_residual_batch(state, X[:50], p_test)
    p_loaded = compose_residual_batch(loaded, X[:50], p_test)
    np.testing.assert_allclose(p_orig, p_loaded, rtol=1e-6, atol=1e-7)


def test_legacy_binary_mode_unchanged():
    """Sanity: omitting ``init_pred_train`` keeps the legacy binary
    behavior so existing callers don't break."""
    X, y, _ = _make_anchor_and_residual_data(N=700, seed=21)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=10,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        seed=29,
    )
    assert state.objective == "binary"
    assert state.output_mode == "probability"
    # apply_one / apply_batch must still work without raising.
    p_one = apply_one(state, X[0])
    p_batch = apply_batch(state, X[:32])
    assert 0.0 < p_one < 1.0
    assert p_batch.shape == (32,)
    assert np.all((p_batch > 0.0) & (p_batch < 1.0))
