"""Tests for src/bagging.py -- the family-agnostic 3x-bag wrapper.

Covers:
* logit-mean combine is order-invariant (geometric mean of odds),
* ``train_bagged_member`` determinism under fixed seeds (seed_only +
  bootstrap),
* ``BaggedMemberState`` round-trips through ``save`` / ``load`` and the
  apply path is pure-numpy and unchanged across the round-trip,
* end-to-end with a trivial fit_fn (a tiny pure-numpy logreg member that
  matches the repo member contract: ``.save`` / ``.load`` + apply).

The trivial member is implemented here in pure numpy (no torch) so the test
runs in any environment; it exercises exactly the contract ``bagging.py``
relies on.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.bagging import (
    BaggedMemberState,
    bagged_predict_logit_mean,
    logit_mean_combine,
    train_bagged_member,
)


# ---------------------------------------------------------------------------
# Trivial member: a tiny pure-numpy logistic regression that honors the repo
# member contract. Lives at module scope so it is importable for the
# class-path round-trip in BaggedMemberState.save/load.
# ---------------------------------------------------------------------------


_EPS = 1.0e-6


@dataclass
class TinyLogRegState:
    weights: np.ndarray
    bias: float

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "weights.npy", self.weights.astype(np.float64))
        (out / "meta.json").write_text(
            json.dumps({"bias": float(self.bias)}), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir):
        d = Path(in_dir)
        w = np.load(d / "weights.npy")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(weights=w, bias=float(meta["bias"]))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def apply_state_batch(state: TinyLogRegState, X: np.ndarray) -> np.ndarray:
    """Member-contract batch apply: [N] probabilities in (eps, 1-eps)."""
    z = np.asarray(X, dtype=np.float64) @ state.weights.astype(np.float64) + float(
        state.bias
    )
    return np.clip(_sigmoid(z), _EPS, 1.0 - _EPS).astype(np.float32)


def fit_tiny_logreg(*, X, y, sample_weight=None, seed=0, n_iters=400, lr=0.2):
    """Trivial deterministic-given-seed logreg trainer (pure numpy GD).

    The seed perturbs the init so distinct seeds give distinct (but
    deterministic) sub-models -- exactly what a bag needs.
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, f = X.shape
    sw = (
        np.ones(n)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    sw = sw / max(sw.sum(), 1e-9)
    w = rng.normal(scale=0.01, size=f)
    b = 0.0
    for _ in range(int(n_iters)):
        p = _sigmoid(X @ w + b)
        g = (p - y) * sw
        w -= lr * (X.T @ g)
        b -= lr * float(g.sum())
    return TinyLogRegState(weights=w, bias=float(b))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _make_data(N=600, F=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, F)).astype(np.float32)
    w_true = rng.normal(size=F)
    z = X @ w_true + 0.3
    p = _sigmoid(z)
    y = (rng.random(N) < p).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# logit_mean_combine / bagged_predict_logit_mean
# ---------------------------------------------------------------------------


def test_logit_mean_is_geometric_mean_of_odds():
    probs = np.array([[0.2, 0.9], [0.5, 0.5], [0.8, 0.1]])
    out = logit_mean_combine(probs)
    # Closed form: sigmoid(mean of logits) = geometric mean of odds mapped back.
    for j in range(probs.shape[1]):
        logits = np.log(probs[:, j] / (1 - probs[:, j]))
        expected = 1.0 / (1.0 + np.exp(-logits.mean()))
        assert math.isclose(float(out[j]), expected, rel_tol=1e-9, abs_tol=1e-9)


def test_logit_mean_is_order_invariant():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.01, 0.99, size=(4, 17))
    base = logit_mean_combine(probs)
    for perm in ([3, 1, 0, 2], [2, 0, 3, 1], [1, 2, 3, 0]):
        permuted = logit_mean_combine(probs[perm])
        np.testing.assert_allclose(permuted, base, rtol=1e-12, atol=1e-12)


def test_logit_mean_clips_saturated_inputs():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])  # exactly 0/1 -> must stay finite
    out = logit_mean_combine(probs)
    assert np.all(np.isfinite(out))
    assert float(out.max()) < 1.0 and float(out.min()) > 0.0


def test_bagged_predict_matches_manual_logit_mean():
    X, y = _make_data()
    states = [fit_tiny_logreg(X=X, y=y, seed=s) for s in (0, 1, 2)]
    got = bagged_predict_logit_mean(states, apply_state_batch, X)
    # Manual: stack probs, logit, mean, sigmoid.
    probs = np.stack([apply_state_batch(s, X) for s in states], axis=0)
    expected = logit_mean_combine(probs)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# train_bagged_member
# ---------------------------------------------------------------------------


def test_train_bagged_member_determinism_seed_only():
    X, y = _make_data()
    fold_train_idx = np.arange(0, 400)
    a = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="seed_only",
    )
    b = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="seed_only",
    )
    assert len(a) == len(b) == 3
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa.weights, sb.weights)
        assert sa.bias == sb.bias
    # Distinct seeds DO produce distinct sub-models (the seed enters the
    # model RNG / init). On this convex toy objective they converge close,
    # so we only require that they are not bit-identical.
    assert not np.array_equal(a[0].weights, a[1].weights)


def test_train_bagged_member_determinism_bootstrap():
    X, y = _make_data()
    fold_train_idx = np.arange(0, 400)
    a = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="bootstrap",
    )
    b = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="bootstrap",
    )
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa.weights, sb.weights)
        assert sa.bias == sb.bias
    # Bootstrap should differ from seed_only (resampled rows).
    c = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="seed_only",
    )
    assert not np.allclose(a[0].weights, c[0].weights)


def test_train_bagged_member_only_trains_on_fold_rows():
    # Make the held-out fold's labels garbage; if the bag trained on them the
    # weights would shift. We check it trains only on fold_train_idx by
    # comparing against an explicit subset fit.
    X, y = _make_data()
    fold_train_idx = np.arange(0, 300)
    bag = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(7,), bag_kind="seed_only",
    )
    ref = fit_tiny_logreg(X=X[fold_train_idx], y=y[fold_train_idx], seed=7)
    np.testing.assert_allclose(bag[0].weights, ref.weights, rtol=1e-12, atol=1e-12)


def test_train_bagged_member_rejects_seed_in_fit_kwargs():
    # 'seed' is managed by the bagger; passing it through **fit_kwargs is an
    # error (it would collide with the per-bag seed).
    X, y = _make_data()
    idx = np.arange(0, 100)
    with pytest.raises(ValueError, match="seed"):
        train_bagged_member(
            fit_tiny_logreg, X=X, y=y, fold_train_idx=idx, seed=3,
        )


def test_train_bagged_member_forwards_sample_weight():
    X, y = _make_data()
    idx = np.arange(0, 300)
    sw = np.linspace(0.1, 2.0, len(y)).astype(np.float32)
    bag = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, sample_weight=sw, fold_train_idx=idx,
        seeds=(0,), bag_kind="seed_only",
    )
    ref = fit_tiny_logreg(X=X[idx], y=y[idx], sample_weight=sw[idx], seed=0)
    np.testing.assert_allclose(bag[0].weights, ref.weights, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# BaggedMemberState save / load round-trip + end-to-end
# ---------------------------------------------------------------------------


def test_bagged_state_roundtrip_end_to_end(tmp_path):
    X, y = _make_data()
    fold_train_idx = np.arange(0, 400)
    members = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=fold_train_idx,
        seeds=(0, 1, 2), bag_kind="seed_only",
    )
    state = BaggedMemberState(members=members, apply_fn=apply_state_batch)

    # Predictions before save.
    Xte = X[400:]
    p_before = state.apply_batch(Xte)

    # apply_one matches apply_batch row-by-row.
    for i in range(10):
        p_i = state.apply_one(Xte[i])
        assert math.isclose(p_i, float(p_before[i]), rel_tol=1e-6, abs_tol=1e-7)

    # Save, then load (apply_fn re-resolved from this module via class path).
    out = tmp_path / "bag"
    state.save(out)
    assert (out / "bag_index.json").exists()
    index = json.loads((out / "bag_index.json").read_text())
    assert index["n_members"] == 3
    assert index["combine"] == "logit_mean"

    loaded = BaggedMemberState.load(out)
    assert len(loaded.members) == 3
    p_after = loaded.apply_batch(Xte)
    np.testing.assert_allclose(p_after, p_before, rtol=1e-6, atol=1e-7)


def test_bagged_state_load_with_explicit_overrides(tmp_path):
    X, y = _make_data()
    members = train_bagged_member(
        fit_tiny_logreg, X=X, y=y, fold_train_idx=np.arange(300),
        seeds=(0, 1), bag_kind="seed_only",
    )
    state = BaggedMemberState(members=members, apply_fn=apply_state_batch)
    out = tmp_path / "bag2"
    state.save(out)
    # Explicit class + apply_fn overrides (no reliance on module resolution).
    loaded = BaggedMemberState.load(
        out, member_class=TinyLogRegState, apply_fn=apply_state_batch
    )
    np.testing.assert_allclose(
        loaded.apply_batch(X[:50]), state.apply_batch(X[:50]),
        rtol=1e-6, atol=1e-7,
    )


def test_bagged_state_rejects_bad_combine():
    members = [TinyLogRegState(np.zeros(3), 0.0)]
    with pytest.raises(ValueError, match="logit_mean"):
        BaggedMemberState(members=members, combine="arithmetic_mean")


def test_bagged_state_requires_at_least_one_member():
    with pytest.raises(ValueError, match=">= 1"):
        BaggedMemberState(members=[])
