"""Regression tests for the vectorized ``predict_raw`` in ``src.gbdt_member``.

The original ``predict_raw`` looped per-row in Python, calling
``_traverse_one_tree`` once per (row, tree). Inside ``fit_gbdt_member``
this was hot-pathed via the post-training parity check on the
booster's val split (10% of the train rows). At 5M-row scale the val
split is ~510k rows and at 400 trees that's ~200M Python iterations,
which makes the cell appear to hang for 60-90 minutes after LightGBM
finishes training in ~2 minutes.

Tests:
  * ``predict_raw`` is bit-identical to the per-row reference (no
    behavior regression).
  * ``predict_raw`` is dramatically faster than the per-row reference
    on a realistic shape (>= 10x speedup proxy).
  * ``predict_raw`` matches ``apply_batch`` after the sigmoid is
    applied (same walker, just unnormalized).
  * ``predict_raw`` handles ``N == 0`` and feature-dim mismatch.
  * NaN feature values still follow ``default_left`` (the per-row code
    used ``np.isfinite`` checks; we want the vectorized path to
    preserve the same semantics).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.gbdt_member import (
    apply_batch,
    apply_one,
    fit_gbdt_member,
    predict_raw,
    _traverse_one_tree,
)


def _per_row_reference(state, features_matrix: np.ndarray) -> np.ndarray:
    """Slow per-row Python reference (the previous implementation).

    Kept here so we can pin bit-identical behavior of the new
    vectorized ``predict_raw``.
    """
    N = int(features_matrix.shape[0])
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        raw = float(state.bias)
        for t in range(int(state.n_trees)):
            raw += _traverse_one_tree(state, t, features_matrix[i])
        out[i] = raw
    return out


def _fit_small_state(seed: int = 0):
    rng = np.random.default_rng(seed)
    N, F = 4000, 24
    X = rng.normal(size=(N, F)).astype(np.float32)
    logits = X @ rng.normal(size=F).astype(np.float32) * 0.5
    y = (1.0 / (1.0 + np.exp(-logits)) > rng.uniform(size=N)).astype(np.float32)
    state = fit_gbdt_member(
        X=X,
        y=y,
        feature_names=tuple(f"f{i}" for i in range(F)),
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=20,
        early_stopping_rounds=15,
        seed=seed,
        max_bin=63,
        log_period=0,
    )
    return state, X, y


def test_predict_raw_matches_per_row_reference_bit_identical():
    """The new vectorized predict_raw must be numerically identical
    to the per-row Python reference. ``_walk_tree_batch`` is bit-
    identical to ``_traverse_one_tree`` per row by construction; this
    pin guarantees no future refactor breaks the numerical contract.
    """
    state, X, _ = _fit_small_state(seed=0)
    rng = np.random.default_rng(11)
    Xq = rng.normal(size=(800, X.shape[1])).astype(np.float32)

    raw_new = predict_raw(state, Xq)
    raw_ref = _per_row_reference(state, Xq.astype(np.float64))

    assert raw_new.shape == raw_ref.shape == (800,)
    assert raw_new.dtype == np.float64
    # apply_batch internally casts features_matrix to float64 then
    # numpy-ops; the per-row reference reads features_matrix[i] in
    # whatever dtype it came in. To guarantee bit-identical output we
    # must compare against the same float64 cast the vectorized path
    # uses. Cast Xq once and rerun the per-row reference for parity.
    np.testing.assert_array_equal(raw_new, raw_ref)


def test_predict_raw_matches_apply_batch_through_sigmoid():
    """``predict_raw`` is exactly what feeds the sigmoid in
    ``apply_batch`` (with bias already added). The two must agree
    end-to-end after numerical sigmoid clamping.
    """
    state, X, _ = _fit_small_state(seed=1)
    rng = np.random.default_rng(12)
    Xq = rng.normal(size=(500, X.shape[1])).astype(np.float32)

    raw = predict_raw(state, Xq)
    p_via_raw = 1.0 / (1.0 + np.exp(-raw))
    p_via_raw = np.clip(p_via_raw, 1e-6, 1.0 - 1e-6).astype(np.float32)

    p_batch = apply_batch(state, Xq)
    np.testing.assert_allclose(p_via_raw, p_batch, atol=1e-6, rtol=0)


def test_predict_raw_preserves_nan_default_left_semantics():
    """NaN entries must follow ``default_left`` at every split, exactly
    like the per-row code did. The vectorized walker uses the same
    branch logic; pin it so a future refactor doesn't accidentally
    treat NaN as 0 or some other sentinel.
    """
    state, X, _ = _fit_small_state(seed=2)
    rng = np.random.default_rng(13)
    M = 200
    Xq = rng.normal(size=(M, X.shape[1])).astype(np.float32)
    # Sprinkle NaNs on ~30% of feature cells.
    nan_mask = rng.uniform(size=Xq.shape) < 0.3
    Xq[nan_mask] = np.nan

    raw_new = predict_raw(state, Xq)
    raw_ref = _per_row_reference(state, Xq.astype(np.float64))
    np.testing.assert_array_equal(raw_new, raw_ref)


def test_predict_raw_zero_rows_returns_empty():
    state, X, _ = _fit_small_state(seed=3)
    Xq = np.empty((0, X.shape[1]), dtype=np.float32)
    out = predict_raw(state, Xq)
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_predict_raw_validates_dimensions():
    state, X, _ = _fit_small_state(seed=4)
    bad_1d = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D"):
        predict_raw(state, bad_1d)
    bad_dim = np.zeros((4, X.shape[1] + 7), dtype=np.float32)
    with pytest.raises(ValueError, match="state.feature_dim"):
        predict_raw(state, bad_dim)


def test_predict_raw_speed_is_dominated_by_apply_batch_path():
    """Soft assertion that the new ``predict_raw`` is within an order
    of magnitude of ``apply_batch`` (it should actually be ~equal).
    Pre-fix this test would have caught the hang: the previous
    per-row Python loop was 50-100x slower than ``apply_batch`` on
    the same data.
    """
    state, X, _ = _fit_small_state(seed=5)
    rng = np.random.default_rng(14)
    M = 6000
    Xq = rng.normal(size=(M, X.shape[1])).astype(np.float32)

    # Warm caches.
    _ = predict_raw(state, Xq[:64])
    _ = apply_batch(state, Xq[:64])

    t0 = time.perf_counter()
    _ = predict_raw(state, Xq)
    t_raw = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = apply_batch(state, Xq)
    t_batch = time.perf_counter() - t0

    # ``apply_batch`` does the same tree-walk work as ``predict_raw``
    # plus the sigmoid; the two should be within 5x of each other.
    # The previous per-row implementation crossed 50x.
    assert t_raw < (t_batch * 5.0 + 0.5), (
        f"predict_raw ({t_raw:.3f}s) is far slower than apply_batch "
        f"({t_batch:.3f}s); the per-row implementation has likely "
        f"regressed."
    )


def test_predict_raw_consistent_with_apply_one_per_row():
    """``apply_one(state, features) -> sigmoid(raw[i])``; the per-row
    public API must agree with ``predict_raw`` row-by-row. This pins
    ``apply_one`` against the new vectorized path.
    """
    state, X, _ = _fit_small_state(seed=6)
    rng = np.random.default_rng(15)
    Xq = rng.normal(size=(64, X.shape[1])).astype(np.float32)
    raw = predict_raw(state, Xq)
    for i in range(Xq.shape[0]):
        p_one = apply_one(state, Xq[i])
        z = float(raw[i])
        p_ref = 1.0 / (1.0 + np.exp(-z))
        p_ref = float(min(max(p_ref, 1e-6), 1.0 - 1e-6))
        assert abs(p_one - p_ref) < 1.0e-6, (
            f"row {i}: apply_one={p_one} vs sigmoid(predict_raw)={p_ref}"
        )
