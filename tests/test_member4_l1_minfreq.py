"""Regression: Member 4 (logreg) L1 + min_feature_std + group split.

Three tests:
1. ``l1_strength > 0`` actually adds weight-norm shrinkage on top of L2.
2. ``min_feature_std`` zeros out near-constant feature columns post-fit
   without changing the predictor on the train mean (bias absorbs).
3. ``holdout_group_id`` routes whole groups (items) to one side of the
   internal split.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.logreg_member import fit_logreg_member


def _toy_data(n: int = 1024, f: int = 16, *, sparse_cols: int = 4, seed: int = 0):
    """Generate a small dataset with some near-constant feature columns
    so ``min_feature_std`` has something to drop. Signal is a simple
    linear function so the fit has clean expectations."""
    rng = np.random.default_rng(int(seed))
    X = rng.normal(size=(n, f)).astype(np.float32)
    # The last ``sparse_cols`` columns are near-constant (std ~ 1e-4).
    X[:, -int(sparse_cols):] = (
        rng.normal(size=(n, int(sparse_cols))).astype(np.float32) * 1.0e-4
    )
    w_true = rng.normal(size=f).astype(np.float64) * 0.3
    z = X @ w_true + 0.1
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, y, w_true


def test_l1_strength_shrinks_weight_norm():
    """For the same dataset and seeds, L1 > 0 must give a smaller ||w||
    than L1 = 0. (Bias is not penalized.)"""
    X, y, _ = _toy_data(n=1024, f=16, sparse_cols=2, seed=42)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    s0 = fit_logreg_member(
        X=X, y=y, feature_names=feature_names,
        l1_strength=0.0, weight_decay=1.0e-4,
        epochs=30, batch_size=128, val_fraction=0.1,
        seed=0, log_every=0,
    )
    s1 = fit_logreg_member(
        X=X, y=y, feature_names=feature_names,
        l1_strength=1.0e-2, weight_decay=1.0e-4,
        epochs=30, batch_size=128, val_fraction=0.1,
        seed=0, log_every=0,
    )
    n0 = float(np.linalg.norm(s0.weights))
    n1 = float(np.linalg.norm(s1.weights))
    assert n1 < n0, (
        f"L1 should shrink ||w||; got ||w||(L1=0)={n0:.4f} >= "
        f"||w||(L1>0)={n1:.4f}"
    )


def test_min_feature_std_zeros_rare_features():
    """Features whose train-slice std is below ``min_feature_std`` must
    have their weights zeroed in the saved state."""
    X, y, _ = _toy_data(n=1024, f=16, sparse_cols=4, seed=7)
    feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
    s = fit_logreg_member(
        X=X, y=y, feature_names=feature_names,
        l1_strength=0.0, weight_decay=1.0e-4,
        min_feature_std=1.0e-2,    # well above the 1e-4 noise std
        epochs=20, batch_size=128, val_fraction=0.1,
        seed=0, log_every=0,
    )
    # The last 4 columns were ~constant; their saved weights must be 0.
    np.testing.assert_array_equal(s.weights[-4:], np.zeros(4, dtype=np.float32))
    # And the first 12 columns should mostly be nonzero (signal there).
    nz_first12 = int(np.sum(np.abs(s.weights[:12]) > 1e-6))
    assert nz_first12 >= 6, f"expected >=6 nonzero weights in first 12, got {nz_first12}"


def test_holdout_group_id_routes_groups_to_one_side():
    """When holdout_group_id is provided, the internal val split logs
    the group-stratified line (mirrors Member 2's contract)."""
    rng = np.random.default_rng(0)
    n = 256
    f = 8
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (rng.uniform(size=n) > 0.5).astype(np.float32)
    g = rng.integers(0, 16, size=n).astype(np.int64)
    feature_names = tuple(f"f{i}" for i in range(f))

    import src.logreg_member as lm
    log_messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: log_messages.append(record.getMessage())
    handler.setLevel(logging.DEBUG)
    prev_level = lm.LOG.level
    lm.LOG.setLevel(logging.INFO)
    lm.LOG.addHandler(handler)
    try:
        fit_logreg_member(
            X=X, y=y, feature_names=feature_names,
            holdout_group_id=g,
            l1_strength=0.0, weight_decay=1.0e-4,
            epochs=2, batch_size=64, val_fraction=0.1,
            seed=0, log_every=0,
        )
    finally:
        lm.LOG.removeHandler(handler)
        lm.LOG.setLevel(prev_level)
    split_lines = [m for m in log_messages if "group-stratified split" in m]
    assert split_lines, (
        f"expected group-stratified-split log when holdout_group_id is "
        f"provided; got: {log_messages[:5]}"
    )


def test_holdout_group_id_shape_must_match():
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (rng.uniform(size=n) > 0.5).astype(np.float32)
    feature_names = tuple(f"f{i}" for i in range(4))
    with pytest.raises(ValueError, match=r"holdout_group_id shape"):
        fit_logreg_member(
            X=X, y=y, feature_names=feature_names,
            holdout_group_id=np.arange(10, dtype=np.int64),  # wrong shape
            epochs=1, batch_size=16, seed=0, log_every=0,
        )


def test_holdout_group_id_needs_at_least_two_groups():
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (rng.uniform(size=n) > 0.5).astype(np.float32)
    feature_names = tuple(f"f{i}" for i in range(4))
    with pytest.raises(ValueError, match=r"need >=2"):
        fit_logreg_member(
            X=X, y=y, feature_names=feature_names,
            holdout_group_id=np.zeros(n, dtype=np.int64),  # all same group
            epochs=1, batch_size=16, seed=0, log_every=0,
        )
