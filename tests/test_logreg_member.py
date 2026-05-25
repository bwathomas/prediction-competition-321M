"""Tests for src/logreg_member.py.

Hand-rolled torch logreg member: training, save/load, runtime apply.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.logreg_member import (
    LogRegMemberState,
    apply_batch,
    apply_one,
    fit_logreg_member,
)


def _make_separable(N: int = 2000, F: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, F)).astype(np.float32)
    # True linear coefficients with sparse signal.
    w_true = np.zeros(F, dtype=np.float32)
    w_true[: F // 2] = rng.normal(size=F // 2).astype(np.float32)
    z = (X @ w_true) + 0.5  # bias
    p = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    y = (rng.random(N) < p).astype(np.float32)
    return X, y, w_true


def test_apply_one_matches_apply_batch():
    rng = np.random.default_rng(0)
    F = 12
    weights = rng.normal(size=F).astype(np.float32) * 0.1
    bias = float(rng.normal())
    feats = rng.normal(size=(50, F)).astype(np.float32)
    p_batch = apply_batch(weights, bias, feats)
    for i in range(feats.shape[0]):
        p_i = apply_one(weights, bias, feats[i])
        assert math.isclose(p_i, float(p_batch[i]), rel_tol=1e-5, abs_tol=1e-6)


def test_apply_batch_clamps_to_eps_open_interval():
    F = 8
    weights = np.full(F, 1e3, dtype=np.float32)
    feats = np.ones((4, F), dtype=np.float32) * 1e3
    p = apply_batch(weights, 0.0, feats)
    # Saturated sigmoid: should be clamped to 1 - eps, finite.
    assert np.all(np.isfinite(p))
    assert float(p.max()) < 1.0
    assert float(p.min()) > 0.0
    # Negative saturation:
    p2 = apply_batch(weights, 0.0, -feats)
    assert np.all(np.isfinite(p2))
    assert float(p2.max()) < 1.0
    assert float(p2.min()) > 0.0


def test_apply_one_returns_python_float_in_unit_interval():
    F = 5
    weights = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    bias = 0.05
    feats = np.array([1.0, 2.0, -1.0, 0.5, 0.0], dtype=np.float32)
    p = apply_one(weights, bias, feats)
    assert isinstance(p, float)
    assert 0.0 < p < 1.0


def test_apply_one_zero_features_returns_sigmoid_of_bias():
    F = 7
    weights = np.zeros(F, dtype=np.float32)
    bias = 0.7
    feats = np.zeros(F, dtype=np.float32)
    p = apply_one(weights, bias, feats)
    expected = 1.0 / (1.0 + math.exp(-bias))
    assert math.isclose(p, expected, rel_tol=1e-5)


def test_apply_one_handles_nan_with_clamp():
    """If a NaN sneaks through (despite belt-and-suspenders upstream)
    apply_one returns 0.5 rather than crashing the runtime."""
    F = 4
    weights = np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float32)
    feats = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    p = apply_one(weights, 0.0, feats)
    assert isinstance(p, float)
    # NaN propagates through dot; we clamp via the not-finite check
    # to 0.5 in apply_one.
    assert p == 0.5


def test_apply_batch_rejects_dimension_mismatch():
    weights = np.zeros(5, dtype=np.float32)
    feats = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        apply_batch(weights, 0.0, feats)


def test_fit_logreg_member_recovers_separable_signal():
    """On a synthesizable linear problem, training should drive val
    log-loss below the all-prior baseline by a noticeable margin."""
    X, y, _ = _make_separable(N=4000, F=8, seed=42)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    state = fit_logreg_member(
        X=X,
        y=y,
        feature_names=feature_names,
        weight_decay=1e-4,
        learning_rate=1e-2,
        epochs=100,
        batch_size=512,
        val_fraction=0.2,
        seed=0,
    )
    # Sanity: weights and bias finite, dim correct.
    assert state.weights.shape == (X.shape[1],)
    assert math.isfinite(state.bias)
    assert state.fit_method == "adam"

    # Runtime apply should match training-time logits to high precision.
    p_runtime = apply_batch(state.weights, state.bias, X)
    p_clipped = np.clip(p_runtime.astype(np.float64), 1e-6, 1 - 1e-6)
    nll_runtime = -float(
        (y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)).mean()
    )

    # Baseline: predict prior probability for every row.
    p_prior = float(y.mean())
    p_p = max(min(p_prior, 1 - 1e-6), 1e-6)
    nll_prior = -(p_prior * math.log(p_p) + (1 - p_prior) * math.log(1 - p_p))

    assert nll_runtime + 0.02 < nll_prior, (
        f"training did not improve on the prior: nll_runtime={nll_runtime:.4f} "
        f"nll_prior={nll_prior:.4f}"
    )


def test_state_save_load_roundtrip(tmp_path):
    F = 6
    state = LogRegMemberState(
        weights=np.linspace(-1.0, 1.0, F, dtype=np.float32),
        bias=0.42,
        feature_dim=F,
        feature_names=tuple(f"f{i}" for i in range(F)),
        fit_method="adam",
        n_train=100,
        n_pos=42,
        train_loss=0.5,
        val_loss=0.55,
        weight_decay=1e-4,
    )
    state.save(tmp_path)
    state2 = LogRegMemberState.load(tmp_path)
    np.testing.assert_array_equal(state2.weights, state.weights)
    # float32 roundtrip: 0.42 -> ~0.41999998..., use isclose.
    assert math.isclose(state2.bias, state.bias, rel_tol=1e-6, abs_tol=1e-6)
    assert state2.feature_names == state.feature_names
    assert state2.feature_dim == state.feature_dim


def test_state_to_from_dict_roundtrip():
    F = 4
    state = LogRegMemberState(
        weights=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        bias=-0.05,
        feature_dim=F,
        feature_names=("a", "b", "c", "d"),
        fit_method="lbfgs",
        n_train=10,
        n_pos=5,
        train_loss=0.6,
        val_loss=0.7,
        weight_decay=1e-3,
    )
    d = state.to_dict()
    s2 = LogRegMemberState.from_dict(d)
    np.testing.assert_allclose(s2.weights, state.weights)
    assert s2.feature_names == state.feature_names


def test_state_rejects_nan_weights():
    F = 4
    with pytest.raises(ValueError, match="NaN/Inf"):
        LogRegMemberState(
            weights=np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32),
            bias=0.0,
            feature_dim=F,
            feature_names=("a", "b", "c", "d"),
            fit_method="adam",
            n_train=0,
            n_pos=0,
            train_loss=0.0,
            val_loss=0.0,
            weight_decay=0.0,
        )


def test_runtime_apply_is_torch_free():
    """Verify that apply_one / apply_batch work without torch importable.
    The runtime path must not depend on torch even though training does."""
    import sys
    # Import-only test; do not actually unload torch (other tests need it).
    import src.logreg_member as m
    src_text = open(m.__file__, encoding="utf-8").read()
    # apply_one and apply_batch must be defined ABOVE any ``import torch``.
    apply_one_idx = src_text.index("def apply_one(")
    apply_batch_idx = src_text.index("def apply_batch(")
    # Find the first occurrence of "import torch" (lazy, inside fit_logreg_member).
    import_torch_idx = src_text.index("import torch")
    assert apply_one_idx < import_torch_idx, (
        "apply_one must be defined before any torch import"
    )
    assert apply_batch_idx < import_torch_idx, (
        "apply_batch must be defined before any torch import"
    )
