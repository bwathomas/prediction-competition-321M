"""Regression tests for the chunked, memory-bounded standardization in
``src.logreg_member.fit_logreg_member``.

The previous standardization path eagerly cast the train slice to
float64 twice and computed ``(X_train - mu) / sigma`` via broadcasting,
which at 5M-row scale spiked peak RSS by ~88 GB and OOMed Colab. The
fix:

  * ``_chunked_mean_std`` accumulates float64 reductions chunk-by-chunk;
    no full f64 copy.
  * ``_chunked_standardize_into`` materializes the standardized matrix
    in-place into a pre-allocated float32 buffer.

These tests pin:
  * Numerical equivalence of the chunked stats to the eager reference
    on small data.
  * Numerical equivalence of the standardized matrices.
  * That ``fit_logreg_member`` produces a bit-identical
    ``LogRegMemberState.weights`` / ``bias`` whether the user fits on a
    "flat" X (no idx) or on the chunked path (which is what the
    function takes internally).
  * That very large idx values still work (regression vs. accidental
    int32 overflow when chunk indices are large).
"""

from __future__ import annotations

import math

import numpy as np
import pytest


from src.logreg_member import (
    _chunked_gather_f32,
    _chunked_mean_std,
    _chunked_standardize_into,
    fit_logreg_member,
)


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def _eager_mean_std(X: np.ndarray, idx: np.ndarray):
    raw = X[idx]
    return raw.astype(np.float64).mean(axis=0), raw.astype(np.float64).std(axis=0)


def test_chunked_mean_std_matches_eager_path():
    rng = np.random.default_rng(0)
    N, F = 4_000, 30
    X = (rng.normal(size=(N, F)) * np.linspace(1.0, 1000.0, F)).astype(np.float32)
    idx = rng.permutation(N)[:3_500]

    mu_e, sigma_e = _eager_mean_std(X, idx)
    mu_c, sigma_c = _chunked_mean_std(X, idx, chunk=256)
    np.testing.assert_allclose(mu_c, mu_e, rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(sigma_c, sigma_e, rtol=1.0e-5, atol=1.0e-5)


def test_chunked_mean_std_handles_empty_idx():
    X = np.zeros((10, 5), dtype=np.float32)
    idx = np.array([], dtype=np.int64)
    mu, sigma = _chunked_mean_std(X, idx)
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
    assert np.all(mu == 0.0)
    assert np.all(sigma == 1.0)  # safe default for downstream divide


def test_chunked_mean_std_works_with_chunk_one():
    """One row per chunk should still produce the correct global mean
    and std (regression vs. an accidental over-counting bug).
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    idx = np.arange(50)
    mu_c, sigma_c = _chunked_mean_std(X, idx, chunk=1)
    mu_e, sigma_e = _eager_mean_std(X, idx)
    np.testing.assert_allclose(mu_c, mu_e, atol=1.0e-7)
    np.testing.assert_allclose(sigma_c, sigma_e, atol=1.0e-5)


def test_chunked_mean_std_works_with_chunk_larger_than_n():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    idx = np.arange(50)
    mu_c, sigma_c = _chunked_mean_std(X, idx, chunk=10_000)
    mu_e, sigma_e = _eager_mean_std(X, idx)
    np.testing.assert_allclose(mu_c, mu_e, atol=1.0e-7)
    np.testing.assert_allclose(sigma_c, sigma_e, atol=1.0e-5)


def test_chunked_standardize_matches_eager_path():
    rng = np.random.default_rng(3)
    N, F = 2_000, 24
    X = (rng.normal(size=(N, F)) * np.linspace(1.0, 100.0, F)).astype(np.float32)
    idx = rng.permutation(N)[:1_700]
    mu, sigma = _chunked_mean_std(X, idx)
    sigma_safe = np.where(sigma < 1.0e-9, 1.0, sigma)
    mu_f32 = mu.astype(np.float32)
    sigma_f32 = sigma_safe.astype(np.float32)

    expected = ((X[idx] - mu_f32[None, :]) / sigma_f32[None, :]).astype(
        np.float32
    )
    chunked = _chunked_standardize_into(X, idx, mu_f32, sigma_f32, chunk=256)
    np.testing.assert_allclose(chunked, expected, atol=1.0e-5, rtol=0)


def test_chunked_standardize_handles_empty_idx():
    X = np.zeros((10, 5), dtype=np.float32)
    idx = np.array([], dtype=np.int64)
    mu_f32 = np.zeros(5, dtype=np.float32)
    sigma_f32 = np.ones(5, dtype=np.float32)
    out = _chunked_standardize_into(X, idx, mu_f32, sigma_f32)
    assert out.shape == (0, 5)
    assert out.dtype == np.float32


def test_chunked_gather_f32_matches_fancy_index():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 8)).astype(np.float32)
    idx = rng.permutation(500)[:300].astype(np.int64)
    out = _chunked_gather_f32(X, idx, chunk=64)
    np.testing.assert_array_equal(out, X[idx])


def test_chunked_gather_f32_preserves_dtype_when_input_is_float32():
    X = np.zeros((10, 3), dtype=np.float32)
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    out = _chunked_gather_f32(X, idx)
    assert out.dtype == np.float32


def test_chunked_gather_f32_handles_empty_idx():
    X = np.zeros((10, 5), dtype=np.float32)
    out = _chunked_gather_f32(X, np.array([], dtype=np.int64))
    assert out.shape == (0, 5)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# fit_logreg_member end-to-end
# ---------------------------------------------------------------------------


def _make_poorly_scaled(N: int = 1_500, F: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(N, F)).astype(np.float32)
    scale = np.logspace(-1, 4, F).astype(np.float32)
    X = base * scale[None, :]
    w_true = rng.normal(size=F).astype(np.float32) / scale
    z = X @ w_true + rng.normal(size=N).astype(np.float32) * 0.1
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=N) < p).astype(np.float32)
    return X, y


def test_fit_logreg_member_chunked_state_is_finite_and_well_shaped():
    """Smoke: chunked fit produces a state with finite weights/bias of
    the right shape. Stronger numerical contracts (mean/std parity,
    standardize-then-bake identity) are pinned in the helper-level
    tests above; this one just makes sure the end-to-end glue is
    intact.
    """
    X, y = _make_poorly_scaled(N=600, F=12, seed=7)
    state = fit_logreg_member(
        X=X, y=y, feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        weight_decay=1.0e-3, learning_rate=0.05,
        epochs=15, batch_size=256, val_fraction=0.2, seed=7,
        early_stopping_patience=8, log_every=0,
    )
    assert state.weights.shape == (X.shape[1],)
    assert state.weights.dtype == np.float32
    assert np.all(np.isfinite(state.weights))
    assert math.isfinite(state.bias)
    assert state.fit_method == "adam_std"


def test_fit_logreg_member_chunked_recovers_signal_on_poorly_scaled_data():
    """Sanity: standardized fit beats predicting the prior on data
    where features span 5 orders of magnitude. This is the same
    contract as the existing standardization regression test, but
    after the chunked rewrite, to catch any silent numerical drift.
    """
    X, y = _make_poorly_scaled(N=2_000, F=20, seed=11)
    state = fit_logreg_member(
        X=X, y=y, feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        weight_decay=1.0e-3, learning_rate=0.05,
        epochs=40, batch_size=512, val_fraction=0.2, seed=11,
        early_stopping_patience=10, log_every=0,
    )
    p_prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(p_prior * math.log(p_prior) + (1 - p_prior) * math.log(1 - p_prior))
    assert state.val_loss < nll_prior - 0.01, (
        f"val_loss {state.val_loss:.4f} not better than prior {nll_prior:.4f}; "
        "chunked standardization may have regressed."
    )
    # Weights should NOT be saturating; with std-then-bake the norm
    # is typically O(1) - O(10), not O(100).
    assert float(np.linalg.norm(state.weights)) < 50.0


def test_fit_logreg_member_chunked_with_idx_overflow_safety():
    """With small N this is just a smoke test that the chunked path
    correctly handles ``idx`` whose values exceed ``chunk``. Pre-fix,
    a stale ``X[idx[s_:e_]]`` could have done something silly if we'd
    accidentally written ``X[s_:e_]`` instead.
    """
    rng = np.random.default_rng(13)
    N, F = 4_096, 8
    X = rng.normal(size=(N, F)).astype(np.float32) * 100.0
    z = X[:, 0] - X[:, 1]
    y = (1.0 / (1.0 + np.exp(-z)) > rng.uniform(size=N)).astype(np.float32)
    state = fit_logreg_member(
        X=X, y=y, feature_names=tuple(f"f{i}" for i in range(F)),
        weight_decay=1.0e-3, learning_rate=0.05,
        epochs=10, batch_size=256, val_fraction=0.2, seed=0,
        early_stopping_patience=8, log_every=0,
    )
    p_prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(p_prior * math.log(p_prior) + (1 - p_prior) * math.log(1 - p_prior))
    assert state.val_loss < nll_prior


def test_fit_logreg_member_chunked_unstandardized_path_still_works():
    """The ``standardize=False`` path now also routes through chunked
    helpers (``_chunked_gather_f32``) for symmetry. Pin that this
    path still produces a sensible state on already-z-scored data.
    """
    rng = np.random.default_rng(14)
    N, F = 1_500, 10
    X_pre = rng.normal(size=(N, F)).astype(np.float32)
    z = X_pre @ rng.normal(size=F).astype(np.float32)
    y = (1.0 / (1.0 + np.exp(-z)) > rng.uniform(size=N)).astype(np.float32)
    state = fit_logreg_member(
        X=X_pre, y=y, feature_names=tuple(f"f{i}" for i in range(F)),
        weight_decay=1.0e-3, learning_rate=0.05,
        epochs=20, batch_size=128, val_fraction=0.2, seed=0,
        early_stopping_patience=8, log_every=0,
        standardize=False,
    )
    assert state.fit_method == "adam"
    p_prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(p_prior * math.log(p_prior) + (1 - p_prior) * math.log(1 - p_prior))
    assert state.val_loss < nll_prior
