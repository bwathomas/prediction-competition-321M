"""Tests for ``src/fwfm_member.py``.

Covers:

* The pure-numpy ``apply_batch`` math against a brute-force reference
  implementation that enumerates every (i, j) pair (small F).
* Classic-FM degenerate case (n_fields=1) behaves identically to FwFM
  with all weights in one field.
* ``apply_state_batch`` matches ``apply_state_one`` row-by-row.
* save/load round-trip preserves predictions byte-for-byte.
* Training converges on a synthetic linear + 1 interaction task.
* Training converges on a synthetic interaction-only task (no linear
  signal) -- confirms the bilinear term is wired correctly.
* Cold-start: extreme / saturated inputs do not produce NaN/Inf.
* State validators reject malformed weights.
"""
from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.fwfm_member import (
    FwFMState,
    _fwfm_logits_from_standardized,
    apply_batch,
    apply_one,
    apply_state_batch,
    apply_state_one,
    fit_fwfm_member,
)


# ---------------------------------------------------------------------------
# Reference implementation (brute-force, small F)
# ---------------------------------------------------------------------------


def _bruteforce_fwfm_logit(
    x: np.ndarray,           # [F]
    w0: float,
    w: np.ndarray,           # [F]
    V: np.ndarray,           # [F, k]
    r: np.ndarray,           # [n_fields, n_fields]
    field_ids: np.ndarray,   # [F]
) -> float:
    """Direct enumeration of the FwFM equation. Slow but trivially
    correct -- the closed form must match this to ~1e-5."""
    F = int(x.shape[0])
    z = float(w0)
    for i in range(F):
        z += float(w[i]) * float(x[i])
    for i in range(F):
        for j in range(i + 1, F):
            r_ij = float(r[int(field_ids[i]), int(field_ids[j])])
            v_dot = float((V[i] * V[j]).sum())
            z += r_ij * v_dot * float(x[i]) * float(x[j])
    return z


def _make_state(
    F: int = 6,
    k: int = 3,
    n_fields: int = 2,
    seed: int = 0,
    standardize: bool = False,
) -> FwFMState:
    rng = np.random.default_rng(int(seed))
    w0 = float(rng.normal())
    w = rng.normal(size=(F,)).astype(np.float32) * 0.5
    V = rng.normal(size=(F, k)).astype(np.float32) * 0.3
    r_raw = rng.normal(size=(n_fields, n_fields)).astype(np.float32) * 0.4
    r = 0.5 * (r_raw + r_raw.T)
    field_ids = rng.integers(0, n_fields, size=F, endpoint=False).astype(np.int32)
    # Guarantee every field is represented so we exercise the cross-field
    # einsum path.
    for f_ in range(n_fields):
        if int((field_ids == f_).sum()) == 0:
            field_ids[f_ % F] = f_
    feat_mean = (rng.normal(size=(F,)) * 0.1).astype(np.float32) if standardize else np.zeros(F, dtype=np.float32)
    feat_std = (1.0 + np.abs(rng.normal(size=(F,))) * 0.5).astype(np.float32) if standardize else np.ones(F, dtype=np.float32)
    return FwFMState(
        w0=w0, w=w, V=V, r=r, field_ids=field_ids,
        feat_mean=feat_mean, feat_std=feat_std,
        feature_dim=F,
        feature_names=tuple(f"f_{i}" for i in range(F)),
        k=k, n_fields=n_fields, fit_method="hand",
        n_train=1, n_pos=1, train_loss=0.0, val_loss=0.0,
        standardize=standardize,
        weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
    )


# ---------------------------------------------------------------------------
# Math correctness
# ---------------------------------------------------------------------------


def test_apply_batch_matches_brute_force_no_standardize() -> None:
    state = _make_state(F=8, k=4, n_fields=3, seed=11, standardize=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, state.feature_dim)).astype(np.float32)
    # Closed-form path.
    z_closed = _fwfm_logits_from_standardized(
        Xz=X,
        w0=state.w0, w=state.w, V=state.V, r=state.r,
        field_ids=state.field_ids, n_fields=state.n_fields,
    )
    # Brute force per row.
    z_brute = np.array(
        [
            _bruteforce_fwfm_logit(
                X[i], state.w0, state.w, state.V, state.r, state.field_ids,
            )
            for i in range(int(X.shape[0]))
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(z_closed, z_brute, atol=1.0e-5, rtol=1.0e-5)


def test_apply_batch_matches_brute_force_with_standardize() -> None:
    state = _make_state(F=6, k=3, n_fields=2, seed=21, standardize=True)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, state.feature_dim)).astype(np.float32)
    p_closed = apply_state_batch(state, X)
    Xz = (X - state.feat_mean) / state.feat_std
    z_brute = np.array(
        [
            _bruteforce_fwfm_logit(
                Xz[i], state.w0, state.w, state.V, state.r, state.field_ids,
            )
            for i in range(int(Xz.shape[0]))
        ],
        dtype=np.float64,
    )
    p_brute = 1.0 / (1.0 + np.exp(-z_brute))
    p_brute = np.clip(p_brute, 1.0e-6, 1.0 - 1.0e-6).astype(np.float32)
    np.testing.assert_allclose(p_closed, p_brute, atol=1.0e-5, rtol=1.0e-5)


def test_classic_fm_equals_single_field_fwfm() -> None:
    """When n_fields=1, FwFM is classic FM scaled by r[0,0]. Build two
    states differing only in this and verify they're equivalent up to
    the scaling."""
    F, k = 5, 3
    rng = np.random.default_rng(2)
    w0 = float(rng.normal())
    w = rng.normal(size=(F,)).astype(np.float32)
    V = rng.normal(size=(F, k)).astype(np.float32) * 0.3
    feat_mean = np.zeros(F, dtype=np.float32)
    feat_std = np.ones(F, dtype=np.float32)
    field_ids = np.zeros(F, dtype=np.int32)
    state_unit = FwFMState(
        w0=w0, w=w, V=V, r=np.array([[1.0]], dtype=np.float32),
        field_ids=field_ids, feat_mean=feat_mean, feat_std=feat_std,
        feature_dim=F,
        feature_names=tuple(f"f_{i}" for i in range(F)),
        k=k, n_fields=1, fit_method="hand",
        n_train=1, n_pos=1, train_loss=0.0, val_loss=0.0,
        standardize=False,
        weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
    )
    state_half = FwFMState(
        w0=w0, w=w, V=V, r=np.array([[0.5]], dtype=np.float32),
        field_ids=field_ids, feat_mean=feat_mean, feat_std=feat_std,
        feature_dim=F,
        feature_names=tuple(f"f_{i}" for i in range(F)),
        k=k, n_fields=1, fit_method="hand",
        n_train=1, n_pos=1, train_loss=0.0, val_loss=0.0,
        standardize=False,
        weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
    )
    X = rng.normal(size=(10, F)).astype(np.float32)
    # The bilinear part of state_unit is exactly twice that of state_half,
    # so the *difference* in logits should equal 0.5 * bilinear_unit.
    z_unit = _fwfm_logits_from_standardized(
        Xz=X, w0=state_unit.w0, w=state_unit.w, V=state_unit.V,
        r=state_unit.r, field_ids=state_unit.field_ids,
        n_fields=state_unit.n_fields,
    )
    z_half = _fwfm_logits_from_standardized(
        Xz=X, w0=state_half.w0, w=state_half.w, V=state_half.V,
        r=state_half.r, field_ids=state_half.field_ids,
        n_fields=state_half.n_fields,
    )
    # z_unit - z_half = (1.0 - 0.5) * bilinear; bilinear should be
    # exactly 2 * (z_unit - z_half). The linear+bias part cancels.
    bilinear_part_unit = 2.0 * (z_unit - z_half)
    # Brute-force bilinear part (independent path).
    bilinear_brute = np.array(
        [
            _bruteforce_fwfm_logit(
                X[i], 0.0, np.zeros_like(state_unit.w), state_unit.V,
                state_unit.r, state_unit.field_ids,
            )
            for i in range(int(X.shape[0]))
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(bilinear_part_unit, bilinear_brute, atol=1.0e-5)


# ---------------------------------------------------------------------------
# Apply paths
# ---------------------------------------------------------------------------


def test_apply_state_batch_matches_apply_state_one() -> None:
    state = _make_state(F=10, k=4, n_fields=3, seed=42, standardize=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(31, state.feature_dim)).astype(np.float32)
    p_batch = apply_state_batch(state, X)
    p_one = np.fromiter(
        (apply_state_one(state, X[i]) for i in range(int(X.shape[0]))),
        dtype=np.float32, count=int(X.shape[0]),
    )
    np.testing.assert_allclose(p_batch, p_one, atol=1.0e-6)


def test_apply_batch_dtype_and_range() -> None:
    state = _make_state(F=5, k=2, n_fields=2, standardize=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, state.feature_dim)).astype(np.float32)
    p = apply_state_batch(state, X)
    assert p.dtype == np.float32
    assert p.shape == (20,)
    assert np.all(np.isfinite(p))
    assert (p > 0.0).all() and (p < 1.0).all()


def test_huge_inputs_do_not_nan() -> None:
    state = _make_state(F=4, k=2, n_fields=2, standardize=False)
    X = np.full((3, state.feature_dim), 1.0e6, dtype=np.float32)
    p = apply_state_batch(state, X)
    assert np.all(np.isfinite(p))
    assert (p > 0.0).all() and (p < 1.0).all()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip() -> None:
    state = _make_state(F=7, k=3, n_fields=3, standardize=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(13, state.feature_dim)).astype(np.float32)
    p_before = apply_state_batch(state, X)
    with tempfile.TemporaryDirectory() as tmp:
        state.save(tmp)
        reloaded = FwFMState.load(tmp)
    assert reloaded.feature_dim == state.feature_dim
    assert reloaded.k == state.k
    assert reloaded.n_fields == state.n_fields
    assert reloaded.feature_names == state.feature_names
    np.testing.assert_array_equal(reloaded.w, state.w)
    np.testing.assert_array_equal(reloaded.V, state.V)
    np.testing.assert_array_equal(reloaded.r, state.r)
    np.testing.assert_array_equal(reloaded.field_ids, state.field_ids)
    np.testing.assert_array_equal(reloaded.feat_mean, state.feat_mean)
    np.testing.assert_array_equal(reloaded.feat_std, state.feat_std)
    p_after = apply_state_batch(reloaded, X)
    np.testing.assert_array_equal(p_before, p_after)


# ---------------------------------------------------------------------------
# Training (small, synthetic)
# ---------------------------------------------------------------------------


def test_fit_converges_on_linear_signal() -> None:
    """Linear-only signal: FwFM should beat the prior on val."""
    rng = np.random.default_rng(0)
    N, F = 4_000, 6
    X = rng.normal(size=(N, F)).astype(np.float32)
    w_true = rng.normal(size=(F,))
    z = X @ w_true - 0.3
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(size=(N,)) < p).astype(np.float32)
    state = fit_fwfm_member(
        X=X, y=y,
        feature_names=tuple(f"f_{i}" for i in range(F)),
        k=4, epochs=20, batch_size=512,
        learning_rate=5.0e-2, val_fraction=0.2,
        early_stopping_patience=5,
        seed=0, log_every=0, standardize=True,
    )
    p_pred = apply_state_batch(state, X)
    bce = -(y * np.log(np.clip(p_pred, 1e-6, 1 - 1e-6)) +
            (1 - y) * np.log(1 - np.clip(p_pred, 1e-6, 1 - 1e-6))).mean()
    p_mean = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    prior_bce = -(p_mean * math.log(p_mean) + (1 - p_mean) * math.log(1 - p_mean))
    assert float(bce) < prior_bce - 0.02, (
        f"BCE={float(bce):.4f} did not beat prior {prior_bce:.4f}"
    )


def test_fit_converges_on_interaction_signal() -> None:
    """Interaction-only signal: y depends ONLY on (x_0 * x_1).

    A linear logistic regression on these features should be at chance
    (since E[y | x_0] = E[y | x_1] = 0.5). FwFM SHOULD beat the prior
    if the bilinear term is correctly wired.
    """
    rng = np.random.default_rng(1)
    N, F = 6_000, 4
    X = rng.normal(size=(N, F)).astype(np.float32)
    z = 3.0 * X[:, 0] * X[:, 1] - 0.5
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(size=(N,)) < p).astype(np.float32)
    state = fit_fwfm_member(
        X=X, y=y,
        feature_names=tuple(f"f_{i}" for i in range(F)),
        # n_fields=2: put x0, x1 in different fields so FwFM uses the
        # cross-field path; the on-diagonal field stays at zero impact.
        field_ids=np.array([0, 1, 0, 1], dtype=np.int32),
        k=4, epochs=80, batch_size=512,
        learning_rate=5.0e-2, val_fraction=0.2,
        early_stopping_patience=15,
        weight_decay_w=0.0, weight_decay_V=1e-4, weight_decay_r=1e-4,
        seed=0, log_every=0, standardize=True,
    )
    p_pred = apply_state_batch(state, X)
    bce = -(y * np.log(np.clip(p_pred, 1e-6, 1 - 1e-6)) +
            (1 - y) * np.log(1 - np.clip(p_pred, 1e-6, 1 - 1e-6))).mean()
    p_mean = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    prior_bce = -(p_mean * math.log(p_mean) + (1 - p_mean) * math.log(1 - p_mean))
    assert float(bce) < prior_bce - 0.05, (
        f"FwFM BCE={float(bce):.4f} did not beat prior {prior_bce:.4f} "
        "on interaction-only data -- bilinear term may be broken"
    )


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


def test_state_rejects_mismatched_dims() -> None:
    F = 4
    with pytest.raises(ValueError, match=r"V shape"):
        FwFMState(
            w0=0.0,
            w=np.zeros(F, dtype=np.float32),
            V=np.zeros((F + 1, 3), dtype=np.float32),  # mismatched
            r=np.eye(1, dtype=np.float32),
            field_ids=np.zeros(F, dtype=np.int32),
            feat_mean=np.zeros(F, dtype=np.float32),
            feat_std=np.ones(F, dtype=np.float32),
            feature_dim=F,
            feature_names=tuple(f"f_{i}" for i in range(F)),
            k=3, n_fields=1, fit_method="x",
            n_train=0, n_pos=0, train_loss=0.0, val_loss=0.0,
            standardize=False,
            weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
        )


def test_state_rejects_asymmetric_r() -> None:
    F = 3
    with pytest.raises(ValueError, match="symmetric"):
        FwFMState(
            w0=0.0,
            w=np.zeros(F, dtype=np.float32),
            V=np.zeros((F, 2), dtype=np.float32),
            r=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),  # not symmetric
            field_ids=np.array([0, 1, 0], dtype=np.int32),
            feat_mean=np.zeros(F, dtype=np.float32),
            feat_std=np.ones(F, dtype=np.float32),
            feature_dim=F,
            feature_names=("a", "b", "c"),
            k=2, n_fields=2, fit_method="x",
            n_train=0, n_pos=0, train_loss=0.0, val_loss=0.0,
            standardize=False,
            weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
        )


def test_state_rejects_zero_std() -> None:
    F = 3
    with pytest.raises(ValueError, match="feat_std"):
        FwFMState(
            w0=0.0,
            w=np.zeros(F, dtype=np.float32),
            V=np.zeros((F, 2), dtype=np.float32),
            r=np.eye(1, dtype=np.float32),
            field_ids=np.zeros(F, dtype=np.int32),
            feat_mean=np.zeros(F, dtype=np.float32),
            feat_std=np.array([1.0, 0.0, 1.0], dtype=np.float32),  # zero
            feature_dim=F,
            feature_names=("a", "b", "c"),
            k=2, n_fields=1, fit_method="x",
            n_train=0, n_pos=0, train_loss=0.0, val_loss=0.0,
            standardize=True,
            weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
        )


def test_state_rejects_out_of_range_field_ids() -> None:
    F = 3
    with pytest.raises(ValueError, match="field_ids range"):
        FwFMState(
            w0=0.0,
            w=np.zeros(F, dtype=np.float32),
            V=np.zeros((F, 2), dtype=np.float32),
            r=np.eye(2, dtype=np.float32),
            field_ids=np.array([0, 1, 5], dtype=np.int32),  # 5 >= n_fields=2
            feat_mean=np.zeros(F, dtype=np.float32),
            feat_std=np.ones(F, dtype=np.float32),
            feature_dim=F,
            feature_names=("a", "b", "c"),
            k=2, n_fields=2, fit_method="x",
            n_train=0, n_pos=0, train_loss=0.0, val_loss=0.0,
            standardize=False,
            weight_decay_w=0.0, weight_decay_V=0.0, weight_decay_r=0.0,
        )


def test_fit_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="X must be 2D"):
        fit_fwfm_member(
            X=np.zeros(10, dtype=np.float32),
            y=np.zeros(10, dtype=np.float32),
            feature_names=("a",),
        )
    with pytest.raises(ValueError, match=r"y shape"):
        fit_fwfm_member(
            X=np.zeros((10, 3), dtype=np.float32),
            y=np.zeros(9, dtype=np.float32),
            feature_names=("a", "b", "c"),
        )
    with pytest.raises(ValueError, match=r"feature_names"):
        fit_fwfm_member(
            X=np.zeros((10, 3), dtype=np.float32),
            y=np.zeros(10, dtype=np.float32),
            feature_names=("a", "b"),
        )
