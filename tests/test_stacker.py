"""Tests for src/stacker.py.

The user-spec RED-TEAM block for the stacker requires:
  (a) Prove no leakage: for each fold, the held-out items do not
      appear in that fold's member training sets.
  (b) Confirm OOF predictions cover 100% of training rows exactly once.
  (c) Sanity-check the stacker weights -- if any member gets a
      near-zero or strongly negative weight, report it.
  (d) Verify the stacker trained on OOF predictions, not on the same
      data the members were fit on.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.stacker import (
    STACKER_FEATURE_DIM,
    STACKER_FEATURE_NAMES,
    StackerState,
    apply_batch,
    apply_one,
    assert_no_item_leakage,
    assert_oof_covers_all_rows,
    build_stacker_features,
    build_stacker_features_one,
    fit_stacker,
    logit_clipped,
    make_kfold_split,
    stacker_feature_dim,
    stacker_feature_names,
)


def _make_synthetic_oof(N: int = 800, seed: int = 0):
    """Synthetic OOF predictions where the four members have different
    biases / strengths. The stacker should learn to upweight the
    accurate member and downweight the noisy one."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N).astype(np.float32)
    p_true = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    y = (rng.random(N) < p_true).astype(np.float32)
    # Member predictions: each is a noisy/biased copy of the truth.
    p1 = 1 / (1 + np.exp(-(z + rng.normal(0, 0.3, N))))      # accurate
    p2 = 1 / (1 + np.exp(-(z * 0.5 + rng.normal(0, 0.5, N))))  # weak
    p3 = 1 / (1 + np.exp(-(z + rng.normal(0, 0.7, N))))      # noisy
    p4 = 1 / (1 + np.exp(-(z * 0.7 + rng.normal(0, 0.4, N))))
    member_probs = np.stack([p1, p2, p3, p4], axis=1).astype(np.float32)
    bench_present = (rng.random(N) < 0.7).astype(np.float32)
    nn_neighbor_support = np.log1p(rng.uniform(0, 16, N)).astype(np.float32)
    nn_mean_similarity = rng.uniform(-0.1, 0.95, N).astype(np.float32)
    centroid_distance = rng.uniform(0.1, 2.0, N).astype(np.float32)
    return (
        member_probs,
        bench_present,
        nn_neighbor_support,
        nn_mean_similarity,
        centroid_distance,
        y,
    )


def test_logit_clipped_round_trips_for_safe_inputs():
    p = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    z = logit_clipped(p)
    p_back = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(p, p_back, atol=1e-9)
    # Boundary handling
    assert math.isfinite(float(logit_clipped(0.0)))
    assert math.isfinite(float(logit_clipped(1.0)))


def test_build_stacker_features_layout():
    member_probs = np.array(
        [[0.1, 0.5, 0.9, 0.2], [0.7, 0.3, 0.6, 0.4]], dtype=np.float32
    )
    bench_present = np.array([1, 0], dtype=np.float32)
    nn_support = np.array([1.5, 0.0], dtype=np.float32)
    nn_sim = np.array([0.8, 0.1], dtype=np.float32)
    centroid_distance = np.array([0.4, 1.2], dtype=np.float32)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bench_present,
        nn_neighbor_support=nn_support,
        nn_mean_similarity=nn_sim,
        centroid_distance=centroid_distance,
    )
    assert feats.shape == (2, STACKER_FEATURE_DIM)
    # Member 0's logit columns should equal logit_clipped of their probs.
    expected_logits = np.log(member_probs / (1 - member_probs))
    np.testing.assert_allclose(feats[:, 0:4], expected_logits, atol=1e-5)
    np.testing.assert_array_equal(feats[:, 4], bench_present)
    np.testing.assert_array_equal(feats[:, 5], nn_support)
    np.testing.assert_array_equal(feats[:, 6], nn_sim)
    np.testing.assert_array_equal(feats[:, 7], centroid_distance)


def test_stacker_feature_names_helpers_match_legacy_4member_constants():
    """The dynamic helpers must agree with the legacy constants for M=4
    so existing 4-member callers/bundles keep working unchanged."""
    assert stacker_feature_names(4) == STACKER_FEATURE_NAMES
    assert stacker_feature_dim(4) == STACKER_FEATURE_DIM


def test_stacker_feature_names_for_5_members_appends_logit_member5():
    names_5 = stacker_feature_names(5)
    assert len(names_5) == stacker_feature_dim(5) == 9
    assert names_5[:5] == (
        "logit_member1", "logit_member2", "logit_member3",
        "logit_member4", "logit_member5",
    )
    assert names_5[5:] == STACKER_FEATURE_NAMES[4:]


def test_stacker_feature_names_rejects_zero_members():
    with pytest.raises(ValueError, match="n_members"):
        stacker_feature_names(0)


def test_build_stacker_features_supports_5_members():
    """Task 4: stacker must accept [N, 5] member_probs and produce a
    [N, 9] matrix with logit_member5 at column index 4 and the four
    aux features at columns 5..8."""
    member_probs = np.array(
        [[0.1, 0.5, 0.9, 0.2, 0.3], [0.7, 0.3, 0.6, 0.4, 0.8]],
        dtype=np.float32,
    )
    bench_present = np.array([1, 0], dtype=np.float32)
    nn_support = np.array([1.5, 0.0], dtype=np.float32)
    nn_sim = np.array([0.8, 0.1], dtype=np.float32)
    centroid_distance = np.array([0.4, 1.2], dtype=np.float32)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bench_present,
        nn_neighbor_support=nn_support,
        nn_mean_similarity=nn_sim,
        centroid_distance=centroid_distance,
    )
    assert feats.shape == (2, stacker_feature_dim(5))
    expected_logits = np.log(member_probs / (1 - member_probs))
    np.testing.assert_allclose(feats[:, 0:5], expected_logits, atol=1e-5)
    np.testing.assert_array_equal(feats[:, 5], bench_present)
    np.testing.assert_array_equal(feats[:, 6], nn_support)
    np.testing.assert_array_equal(feats[:, 7], nn_sim)
    np.testing.assert_array_equal(feats[:, 8], centroid_distance)


def test_build_stacker_features_one_supports_5_members():
    """Single-row Task 4 path."""
    feats = build_stacker_features_one(
        member_probs=[0.1, 0.5, 0.9, 0.2, 0.7],
        bench_present=1.0,
        nn_neighbor_support=1.5,
        nn_mean_similarity=0.8,
        centroid_distance=0.4,
    )
    assert int(feats.shape[0]) == stacker_feature_dim(5)
    np.testing.assert_allclose(
        feats[0:5],
        [logit_clipped(p) for p in [0.1, 0.5, 0.9, 0.2, 0.7]],
        atol=1e-5,
    )


def test_build_stacker_features_one_5_member_matches_batch():
    """5-member single-row builder must produce the same row as the
    batch builder; otherwise offline/runtime parity is broken."""
    rng = np.random.default_rng(11)
    member_probs = rng.uniform(0.05, 0.95, size=(8, 5)).astype(np.float32)
    bench_present = rng.integers(0, 2, size=8).astype(np.float32)
    nn_support = rng.uniform(0, 3, size=8).astype(np.float32)
    nn_sim = rng.uniform(-0.1, 0.95, size=8).astype(np.float32)
    centroid_distance = rng.uniform(0.05, 2.5, size=8).astype(np.float32)
    feats_batch = build_stacker_features(
        member_probs=member_probs,
        bench_present=bench_present,
        nn_neighbor_support=nn_support,
        nn_mean_similarity=nn_sim,
        centroid_distance=centroid_distance,
    )
    for i in range(member_probs.shape[0]):
        f_one = build_stacker_features_one(
            member_probs=member_probs[i].tolist(),
            bench_present=float(bench_present[i]),
            nn_neighbor_support=float(nn_support[i]),
            nn_mean_similarity=float(nn_sim[i]),
            centroid_distance=float(centroid_distance[i]),
        )
        np.testing.assert_allclose(f_one, feats_batch[i], atol=1e-5)


def test_build_stacker_features_one_matches_batch():
    rng = np.random.default_rng(0)
    member_probs = rng.uniform(0.05, 0.95, size=(10, 4)).astype(np.float32)
    bench_present = rng.integers(0, 2, size=10).astype(np.float32)
    nn_support = rng.uniform(0, 3, size=10).astype(np.float32)
    nn_sim = rng.uniform(-0.1, 0.95, size=10).astype(np.float32)
    centroid_distance = rng.uniform(0.05, 2.5, size=10).astype(np.float32)
    feats_batch = build_stacker_features(
        member_probs=member_probs,
        bench_present=bench_present,
        nn_neighbor_support=nn_support,
        nn_mean_similarity=nn_sim,
        centroid_distance=centroid_distance,
    )
    for i in range(member_probs.shape[0]):
        f_one = build_stacker_features_one(
            member_probs=member_probs[i].tolist(),
            bench_present=float(bench_present[i]),
            nn_neighbor_support=float(nn_support[i]),
            nn_mean_similarity=float(nn_sim[i]),
            centroid_distance=float(centroid_distance[i]),
        )
        np.testing.assert_allclose(f_one, feats_batch[i], atol=1e-5)


def test_fit_stacker_recovers_signal_and_weights_member1():
    """Member 1 is the most accurate -- its weight should be >= the
    others' AND should be positive (logit-space input agreeing with y)."""
    member_probs, bp, nns, nms, cd, y = _make_synthetic_oof(N=2000, seed=1)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    state = fit_stacker(
        X=feats,
        y=y,
        n_iters=2000,
        learning_rate=0.05,
        l2=1.0,
        early_stopping_patience=300,
        seed=42,
    )
    # The four logit weights are at indices 0..3.
    member_weights = state.weights[0:4]
    # Member 1 (index 0) should have the largest positive weight.
    assert state.weights[0] > 0
    assert state.weights[0] >= np.max(member_weights), (
        f"Member 1 weight {state.weights[0]:.3f} should dominate; got "
        f"weights={member_weights}"
    )
    # All-zero baseline: stacker raw should match the bias.
    p_zero = apply_one(state, np.zeros(STACKER_FEATURE_DIM, dtype=np.float32))
    expected = 1.0 / (1.0 + math.exp(-state.bias))
    expected = max(min(expected, 1 - 1e-6), 1e-6)
    assert math.isclose(p_zero, expected, abs_tol=1e-6)


def test_fit_stacker_beats_uniform_average():
    """A trained stacker should beat the naive uniform average of the
    four member predictions in held-out log-loss."""
    member_probs, bp, nns, nms, cd, y = _make_synthetic_oof(N=1500, seed=2)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(y))
    test_idx = perm[:300]
    train_idx = perm[300:]
    state = fit_stacker(
        X=feats[train_idx],
        y=y[train_idx],
        n_iters=2000,
        learning_rate=0.05,
        l2=1.0,
        seed=11,
    )
    p_stacker = apply_batch(state, feats[test_idx])
    p_avg = member_probs[test_idx].mean(axis=1)

    def _nll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    nll_st = _nll(p_stacker, y[test_idx])
    nll_avg = _nll(p_avg, y[test_idx])
    assert nll_st <= nll_avg + 1e-3, (
        f"Stacker did not beat uniform average: stacker={nll_st:.4f} "
        f"avg={nll_avg:.4f}"
    )


def test_apply_handles_nan_features_gracefully():
    """RED-TEAM contract: NaN features must not crash; the stacker
    treats them as 0 (so the bias dominates)."""
    weights = np.array([1.0, -1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state = StackerState(
        weights=weights,
        bias=0.0,
        feature_names=STACKER_FEATURE_NAMES,
        feature_dim=STACKER_FEATURE_DIM,
        l2=0.0,
        n_train=10,
        n_pos=5,
        train_loss=0.0,
        val_loss=0.0,
        n_iters=0,
    )
    feats = np.array(
        [np.nan, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    p = apply_one(state, feats)
    assert math.isfinite(p)
    assert 0.0 < p < 1.0


def test_apply_one_rejects_dim_mismatch():
    state = StackerState(
        weights=np.zeros(STACKER_FEATURE_DIM, dtype=np.float32),
        bias=0.0,
        feature_names=STACKER_FEATURE_NAMES,
        feature_dim=STACKER_FEATURE_DIM,
        l2=0.0,
        n_train=0,
        n_pos=0,
        train_loss=0.0,
        val_loss=0.0,
        n_iters=0,
    )
    with pytest.raises(ValueError, match="features dim"):
        apply_one(state, np.zeros(3, dtype=np.float32))


def test_state_save_load_roundtrip(tmp_path):
    member_probs, bp, nns, nms, cd, y = _make_synthetic_oof(N=600, seed=3)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    state = fit_stacker(X=feats, y=y, n_iters=500, learning_rate=0.05, l2=1.0, seed=7)
    state.save(tmp_path)
    s2 = StackerState.load(tmp_path)
    np.testing.assert_array_equal(state.weights, s2.weights)
    assert math.isclose(state.bias, s2.bias, abs_tol=1e-6)
    p1 = apply_batch(state, feats[:50])
    p2 = apply_batch(s2, feats[:50])
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_state_to_dict_from_dict_roundtrip():
    state = StackerState(
        weights=np.array(
            [0.5, -0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        bias=0.42,
        feature_names=STACKER_FEATURE_NAMES,
        feature_dim=STACKER_FEATURE_DIM,
        l2=1.0,
        n_train=100,
        n_pos=50,
        train_loss=0.6,
        val_loss=0.65,
        n_iters=500,
    )
    d = state.to_dict()
    s2 = StackerState.from_dict(d)
    np.testing.assert_array_equal(state.weights, s2.weights)
    assert math.isclose(state.bias, s2.bias, abs_tol=1e-6)


def test_make_kfold_split_no_leakage():
    """RED-TEAM (a): for each fold, val items are disjoint from train items."""
    rng = np.random.default_rng(0)
    item_keys = []
    # 20 items, 5 obs each -> 100 rows
    for i in range(20):
        item_keys.extend([f"item_{i}"] * 5)
    rng.shuffle(item_keys)
    folds = make_kfold_split(item_keys=item_keys, n_folds=5, seed=42)
    assert len(folds) == 5
    assert_no_item_leakage(item_keys, folds)


def test_oof_covers_all_rows_exactly_once():
    """RED-TEAM (b): OOF must cover every row exactly once."""
    item_keys = [f"item_{i // 5}" for i in range(100)]
    folds = make_kfold_split(item_keys=item_keys, n_folds=5, seed=7)
    assert_oof_covers_all_rows(len(item_keys), folds)
    # And confirm the assertion FIRES if a row is missing.
    folds_broken = list(folds)
    folds_broken[0] = (folds_broken[0][0], folds_broken[0][1][:-1])  # drop one val row
    with pytest.raises(RuntimeError, match="OOF coverage"):
        assert_oof_covers_all_rows(len(item_keys), folds_broken)


def test_assert_no_item_leakage_fires_on_leakage():
    """Sanity: the leakage detector must catch a deliberate leak."""
    item_keys = [f"item_{i // 5}" for i in range(100)]
    folds = make_kfold_split(item_keys=item_keys, n_folds=5, seed=7)
    # Inject a leaked row (val 0's first item appears in train).
    bad_train = np.unique(np.concatenate([folds[0][0], folds[0][1][:1]]))
    folds_bad = [(bad_train, folds[0][1])] + list(folds[1:])
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_item_leakage(item_keys, folds_bad)


def test_runtime_apply_is_torch_free():
    """RED-TEAM: the runtime path must NOT depend on torch.

    Strategy: reload src.stacker without torch in sys.modules and
    confirm apply_one + apply_batch + StackerState still work.
    """
    import importlib
    import sys

    saved = sys.modules.pop("torch", None)
    try:
        sys.modules["torch"] = None  # type: ignore[assignment]
        # Force re-import of src.stacker so the import-time graph is
        # exercised. The module SHOULD import successfully with no
        # torch available because torch only appears inside fit_stacker.
        if "src.stacker" in sys.modules:
            del sys.modules["src.stacker"]
        mod = importlib.import_module("src.stacker")
        # Now exercise the runtime path.
        state = mod.StackerState(
            weights=np.array(
                [1.0] * mod.STACKER_FEATURE_DIM, dtype=np.float32
            ),
            bias=0.0,
            feature_names=mod.STACKER_FEATURE_NAMES,
            feature_dim=mod.STACKER_FEATURE_DIM,
            l2=0.0,
            n_train=0,
            n_pos=0,
            train_loss=0.0,
            val_loss=0.0,
            n_iters=0,
        )
        p = mod.apply_one(state, np.zeros(mod.STACKER_FEATURE_DIM, dtype=np.float32))
        assert math.isfinite(p)
        assert 0 < p < 1
    finally:
        # Restore original torch state.
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)
        if "src.stacker" in sys.modules:
            del sys.modules["src.stacker"]
        # Re-import the normal way for any subsequent tests.
        importlib.import_module("src.stacker")


def test_runtime_module_static_import_audit():
    """Static check: src/stacker.py top-level imports must not include
    torch / lightgbm / sklearn / faiss / xgboost.
    Lazy imports inside fit_stacker are allowed."""
    src_text = Path("src/stacker.py").read_text(encoding="utf-8")
    # Find the module docstring end and the first ``def`` -- top-level
    # imports must live in this preamble. (We're not strict about
    # the exact location; this regex matches imports at column 0.)
    forbidden = [
        r"^import\s+torch",
        r"^from\s+torch",
        r"^import\s+lightgbm",
        r"^from\s+lightgbm",
        r"^import\s+sklearn",
        r"^from\s+sklearn",
        r"^import\s+faiss",
        r"^from\s+faiss",
        r"^import\s+xgboost",
        r"^from\s+xgboost",
    ]
    for pattern in forbidden:
        matches = list(re.finditer(pattern, src_text, flags=re.MULTILINE))
        assert len(matches) == 0, (
            f"Top-level import matched {pattern!r} in src/stacker.py at "
            f"position {matches[0].start()}; the runtime must not import "
            "torch/lightgbm/sklearn/faiss/xgboost at module scope."
        )


def test_apply_batch_matches_apply_one():
    member_probs, bp, nns, nms, cd, y = _make_synthetic_oof(N=200, seed=4)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    state = fit_stacker(X=feats, y=y, n_iters=200, learning_rate=0.05, l2=1.0, seed=0)
    p_batch = apply_batch(state, feats)
    p_one = np.array(
        [apply_one(state, feats[i]) for i in range(feats.shape[0])],
        dtype=np.float32,
    )
    np.testing.assert_allclose(p_batch, p_one, atol=1e-6)


def test_determinism_two_runs_same_inputs():
    """RED-TEAM (final): two predict() calls on the same input give
    the same output."""
    member_probs, bp, nns, nms, cd, y = _make_synthetic_oof(N=400, seed=5)
    feats = build_stacker_features(
        member_probs=member_probs,
        bench_present=bp,
        nn_neighbor_support=nns,
        nn_mean_similarity=nms,
        centroid_distance=cd,
    )
    state = fit_stacker(X=feats, y=y, n_iters=300, learning_rate=0.05, l2=1.0, seed=0)
    rng = np.random.default_rng(1)
    test_feats = rng.normal(size=(20, STACKER_FEATURE_DIM)).astype(np.float32)
    p1 = apply_batch(state, test_feats)
    p2 = apply_batch(state, test_feats)
    np.testing.assert_array_equal(p1, p2)
