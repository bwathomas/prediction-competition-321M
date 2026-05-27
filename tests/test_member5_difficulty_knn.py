"""Tests for src/member5_difficulty_knn.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.member5_difficulty_knn import (
    Member5State,
    _derive_aggregates_from_dense,
    _knearest_sorted,
    aggregate_per_item_passrate,
    apply_batch,
    apply_batch_via_ids,
    apply_one,
    assert_projection_disjoint_from_val,
    fit_difficulty_projection,
    fit_member5,
)


# ---------------------------------------------------------------------------
# Projection fit
# ---------------------------------------------------------------------------


def test_fit_projection_recovers_linear_signal_direction():
    """If item_mean_passrate is a linear function of emb, the ridge
    solver should recover a w that's directionally aligned with the
    true w (cosine similarity > 0.95). Magnitude shrinks because we
    clip targets to (0.05, 0.95) which truncates extreme values and
    because ridge regularization always shrinks; the direction is
    what the downstream projection cares about."""
    rng = np.random.default_rng(0)
    n_items, d_emb = 500, 8
    emb = rng.normal(size=(n_items, d_emb)).astype(np.float32)
    true_w = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.05, 0.0], dtype=np.float32)
    true_b = 0.4
    target = np.clip(emb @ true_w + true_b, 0.05, 0.95)
    obs_count = np.full(n_items, 50.0, dtype=np.float32)
    w, b = fit_difficulty_projection(
        item_embeddings=emb,
        item_mean_passrate=target,
        item_obs_count=obs_count,
        ridge_alpha=0.01,
    )
    cos = float(
        np.dot(w, true_w) / (np.linalg.norm(w) * np.linalg.norm(true_w) + 1e-12)
    )
    assert cos > 0.95, f"recovered w only cos={cos:.3f} aligned with true_w"
    # Bias drifts because of clipping but should still be in a sane range.
    assert 0.0 < b < 1.0


def test_fit_projection_zero_obs_items_excluded():
    """Items with obs_count == 0 should be dropped from the fit."""
    rng = np.random.default_rng(1)
    n_items, d_emb = 100, 4
    emb = rng.normal(size=(n_items, d_emb)).astype(np.float32)
    target = np.full(n_items, 0.5, dtype=np.float32)
    obs_count = np.zeros(n_items, dtype=np.float32)
    obs_count[:50] = 10.0
    # Half of the items have target=0.99 to skew the fit IF they were
    # included; but they're zero-obs so they should be ignored.
    target[50:] = 0.99
    # Use ridge_alpha=1.0 -- some shrinkage is acceptable since true
    # signal is constant (regression collapses to bias=0.5).
    w, b = fit_difficulty_projection(
        item_embeddings=emb,
        item_mean_passrate=target,
        item_obs_count=obs_count,
        ridge_alpha=1.0,
    )
    assert abs(b - 0.5) < 0.02


def test_fit_projection_rejects_negative_alpha():
    with pytest.raises(ValueError, match="ridge_alpha"):
        fit_difficulty_projection(
            item_embeddings=np.zeros((5, 3), dtype=np.float32),
            item_mean_passrate=np.zeros(5, dtype=np.float32),
            item_obs_count=np.ones(5, dtype=np.float32),
            ridge_alpha=-1.0,
        )


def test_fit_projection_weights_by_sqrt_obs_count():
    """High-obs items should dominate. Construct two clusters with
    incompatible targets; the high-weight cluster's signal should
    dominate the recovered w."""
    rng = np.random.default_rng(2)
    n_items, d_emb = 200, 4
    emb = rng.normal(size=(n_items, d_emb)).astype(np.float32)
    # First 100 items: target follows w=+e_0, weight 100 obs each.
    # Last  100 items: target follows w=-e_0, weight 1   obs each.
    target = np.empty(n_items, dtype=np.float32)
    target[:100] = np.clip(emb[:100, 0] * 0.3 + 0.5, 0.05, 0.95)
    target[100:] = np.clip(emb[100:, 0] * -0.3 + 0.5, 0.05, 0.95)
    obs_count = np.empty(n_items, dtype=np.float32)
    obs_count[:100] = 100.0
    obs_count[100:] = 1.0
    w, _ = fit_difficulty_projection(
        item_embeddings=emb,
        item_mean_passrate=target,
        item_obs_count=obs_count,
        ridge_alpha=0.01,
    )
    # The high-weight cluster's positive-sign signal dominates.
    assert w[0] > 0.05


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_per_item_passrate_basic():
    s = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    i = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    (item_mean, item_cnt, subj_global, subj_cnt,
     passrate_dense, passrate_mask, gm) = aggregate_per_item_passrate(
        subject_ids=s, item_ids=i, labels=y,
        n_subjects=3, n_items=3,
    )
    # Item 0: subjects 0+1 -> [1, 0] -> mean 0.5, count 2.
    # Item 1: subjects 0+1 -> [0, 1] -> mean 0.5, count 2.
    # Item 2: subject 2   -> [1]    -> mean 1.0, count 1.
    np.testing.assert_allclose(item_mean, [0.5, 0.5, 1.0])
    np.testing.assert_allclose(item_cnt, [2.0, 2.0, 1.0])
    np.testing.assert_allclose(subj_global, [0.5, 0.5, 1.0])
    np.testing.assert_allclose(subj_cnt, [2.0, 2.0, 1.0])
    assert abs(gm - 0.6) < 1e-9  # mean of [1,0,0,1,1]
    # Passrate dense + mask.
    assert passrate_mask[0, 0] and passrate_mask[0, 1] and not passrate_mask[0, 2]
    assert passrate_mask[1, 0] and passrate_mask[1, 1] and not passrate_mask[1, 2]
    assert passrate_mask[2, 2] and not passrate_mask[2, 0]
    np.testing.assert_allclose(passrate_dense[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(passrate_dense[2], [0.0, 0.0, 1.0])


def test_derive_aggregates_from_dense_matches_row_aggregator():
    """_derive_aggregates_from_dense must produce IDENTICAL per-item /
    per-subject / global stats to aggregate_per_item_passrate's row
    aggregation WHEN each (subject, item) cell is observed at most
    once -- which is the contest's invariant (each row is one unique
    (subject, item) completion).

    NOTE: when the same (s, i) pair appears multiple times in rows,
    the two paths diverge:
      - row aggregator: item_mean = sum_labels / num_rows_for_item
      - dense aggregator: item_mean = mean over subjects of
        (subject's mean rating for this item)
    For contest data these are equivalent because (s, i) pairs are unique.
    Tests below pick a disjoint set of pairs to enforce the
    no-duplicate invariant that the production data has.
    """
    rng = np.random.default_rng(99)
    n_s, n_i = 12, 25
    # Sample without replacement from the full S*N grid -> unique pairs.
    flat_idx = rng.choice(n_s * n_i, size=200, replace=False)
    s = (flat_idx // n_i).astype(np.int64)
    i = (flat_idx % n_i).astype(np.int64)
    y = rng.random(size=s.shape[0])
    (item_mean_a, item_cnt_a, subj_global_a, subj_cnt_a,
     pd, pm, gm_a) = aggregate_per_item_passrate(
        subject_ids=s, item_ids=i, labels=y,
        n_subjects=n_s, n_items=n_i,
    )
    (item_mean_b, item_cnt_b, subj_global_b, subj_cnt_b, gm_b) = (
        _derive_aggregates_from_dense(pd, pm)
    )
    np.testing.assert_allclose(item_mean_a, item_mean_b, atol=1e-6)
    np.testing.assert_allclose(item_cnt_a, item_cnt_b, atol=1e-6)
    np.testing.assert_allclose(subj_global_a, subj_global_b, atol=1e-6)
    np.testing.assert_allclose(subj_cnt_a, subj_cnt_b, atol=1e-6)
    assert abs(gm_a - gm_b) < 1e-9


def test_derive_aggregates_from_dense_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        _derive_aggregates_from_dense(
            np.zeros((3, 4), dtype=np.float32),
            np.zeros((3, 5), dtype=bool),
        )


def test_derive_aggregates_from_dense_handles_empty_subject():
    """A subject with no observed cells must produce 0 obs_count and
    0 subject_global (not NaN), matching the row aggregator."""
    pd = np.array(
        [[0.0, 0.0, 0.0],
         [0.5, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pm = np.array(
        [[False, False, False],
         [True,  False, False],
         [False, False, False]],
        dtype=bool,
    )
    item_mean, item_cnt, subj_g, subj_cnt, gm = (
        _derive_aggregates_from_dense(pd, pm)
    )
    assert subj_cnt[0] == 0 and subj_g[0] == 0.0
    assert subj_cnt[2] == 0 and subj_g[2] == 0.0
    assert subj_cnt[1] == 1 and abs(subj_g[1] - 0.5) < 1e-6
    assert item_cnt[0] == 1 and abs(item_mean[0] - 0.5) < 1e-6
    assert item_cnt[1] == 0 and item_mean[1] == 0.0


def test_fit_member5_fast_path_matches_slow_path():
    """End-to-end: fit_member5 with passrate_dense/mask pre-built MUST
    produce a state equivalent to the row-aggregation path. Same
    projection weights, same sort order, same predictions on a
    sampled query.

    This is the load-bearing test for the OOM fix: if the fast path
    diverged from the slow path, the per-fold pipeline would be
    silently wrong."""
    inputs = _build_synthetic_member5_inputs(seed=77)
    inputs.pop("true_difficulty")

    # Pre-build the passrate matrices the same way aggregate_per_item_passrate does.
    n_s, n_i = (
        len(inputs["subject_keys"]),
        len(inputs["item_keys"]),
    )
    (_, _, _, _, pd, pm, _) = aggregate_per_item_passrate(
        subject_ids=inputs["subject_ids_per_row"],
        item_ids=inputs["item_ids_per_row"],
        labels=inputs["labels"],
        n_subjects=n_s, n_items=n_i,
    )

    state_slow = fit_member5(
        **inputs, k=5, tau=0.05, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    state_fast = fit_member5(
        **inputs, k=5, tau=0.05, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
        passrate_dense=pd, passrate_mask=pm,
    )
    # Projection equivalence (the input data is identical, so the
    # weighted ridge fit must produce the same solution).
    np.testing.assert_allclose(
        state_fast.projection_weights, state_slow.projection_weights, atol=1e-6,
    )
    assert abs(state_fast.projection_bias - state_slow.projection_bias) < 1e-6
    np.testing.assert_array_equal(state_fast.sort_order, state_slow.sort_order)
    np.testing.assert_allclose(
        state_fast.predicted_difficulty,
        state_slow.predicted_difficulty,
        atol=1e-6,
    )
    # Inference equivalence on a sample.
    rng = np.random.default_rng(0)
    sample = rng.choice(inputs["item_embeddings"].shape[0], size=16, replace=False)
    q_embs = inputs["item_embeddings"][sample]
    for s_id in [0, 5, 10]:
        for r in range(q_embs.shape[0]):
            p_slow = apply_one(state_slow, q_embs[r], f"subj_{s_id}")
            p_fast = apply_one(state_fast, q_embs[r], f"subj_{s_id}")
            assert abs(p_slow - p_fast) < 1e-6, (
                f"row {r}, subj {s_id}: slow={p_slow} vs fast={p_fast}"
            )


def test_fit_member5_fast_path_requires_both_matrices():
    """Passing only one of (passrate_dense, passrate_mask) must raise --
    a half-configured fast path would silently bypass the row
    aggregation without a substitute, producing nonsense aggregates."""
    inputs = _build_synthetic_member5_inputs(seed=2)
    inputs.pop("true_difficulty")
    with pytest.raises(ValueError, match="passrate_dense and passrate_mask"):
        fit_member5(
            **inputs, k=5, tau=0.05, ridge_alpha=0.5,
            item_fallback_weight=0.3, min_subjects_per_item=2,
            passrate_dense=np.zeros((1, 1), dtype=np.float32),
            passrate_mask=None,
        )


def test_fit_member5_fast_path_rejects_shape_mismatch():
    """A passed passrate matrix that disagrees with n_items / n_subjects
    must raise so misalignment doesn't silently produce wrong stats."""
    inputs = _build_synthetic_member5_inputs(seed=3)
    inputs.pop("true_difficulty")
    n_s, n_i = (
        len(inputs["subject_keys"]),
        len(inputs["item_keys"]),
    )
    # Off-by-one rows.
    bad_pd = np.zeros((n_s - 1, n_i), dtype=np.float32)
    bad_pm = np.zeros((n_s - 1, n_i), dtype=bool)
    with pytest.raises(ValueError, match="n_subjects"):
        fit_member5(
            **inputs, k=5, tau=0.05, ridge_alpha=0.5,
            item_fallback_weight=0.3, min_subjects_per_item=2,
            passrate_dense=bad_pd, passrate_mask=bad_pm,
        )
    # Off-by-one cols.
    bad_pd2 = np.zeros((n_s, n_i + 1), dtype=np.float32)
    bad_pm2 = np.zeros((n_s, n_i + 1), dtype=bool)
    with pytest.raises(ValueError, match="n_items"):
        fit_member5(
            **inputs, k=5, tau=0.05, ridge_alpha=0.5,
            item_fallback_weight=0.3, min_subjects_per_item=2,
            passrate_dense=bad_pd2, passrate_mask=bad_pm2,
        )


def test_aggregate_handles_invalid_indices():
    s = np.array([0, -1, 1, 5], dtype=np.int64)  # -1 and out-of-range
    i = np.array([0, 0, -1, 0], dtype=np.int64)
    y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    # Only the first row is valid.
    item_mean, item_cnt, *_ = aggregate_per_item_passrate(
        subject_ids=s, item_ids=i, labels=y, n_subjects=2, n_items=2,
    )
    assert item_cnt[0] == 1
    assert item_cnt[1] == 0


# ---------------------------------------------------------------------------
# k-nearest binary search
# ---------------------------------------------------------------------------


def test_knearest_sorted_picks_closest():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    idx, dist = _knearest_sorted(arr, 2.6, k=3)
    # Closest to 2.6: 3.0 (d=0.4), 2.0 (d=0.6), 4.0 (d=1.4).
    np.testing.assert_array_equal(idx, [3, 2, 4])
    np.testing.assert_allclose(dist, [0.4, 0.6, 1.4], atol=1e-6)


def test_knearest_sorted_handles_edge_at_max():
    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    idx, dist = _knearest_sorted(arr, 100.0, k=2)
    np.testing.assert_array_equal(idx, [3, 2])
    np.testing.assert_allclose(dist, [97.0, 98.0])


def test_knearest_sorted_handles_edge_at_min():
    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    idx, dist = _knearest_sorted(arr, -100.0, k=2)
    np.testing.assert_array_equal(idx, [0, 1])
    np.testing.assert_allclose(dist, [100.0, 101.0])


def test_knearest_sorted_handles_k_larger_than_n():
    arr = np.array([0.0, 1.0], dtype=np.float32)
    idx, dist = _knearest_sorted(arr, 0.5, k=10)
    assert int(idx.shape[0]) == 2
    np.testing.assert_array_equal(np.sort(idx), [0, 1])


# ---------------------------------------------------------------------------
# Member 5 end-to-end fit + apply
# ---------------------------------------------------------------------------


def _build_synthetic_member5_inputs(seed: int = 0):
    """Construct a small synthetic problem: 80 items, 30 subjects, where
    item difficulty is a linear function of the first embedding dim.

    Returns the inputs to fit_member5 plus the true item-mean-passrate
    so tests can compare predicted vs true difficulty."""
    rng = np.random.default_rng(seed)
    n_items, n_subjects, d_emb = 80, 30, 6
    item_embeddings = rng.normal(size=(n_items, d_emb)).astype(np.float32)
    true_difficulty = 0.4 * item_embeddings[:, 0] + 0.5
    true_difficulty = np.clip(true_difficulty, 0.05, 0.95)
    item_keys = tuple(f"item_{i}" for i in range(n_items))
    subject_keys = tuple(f"subj_{s}" for s in range(n_subjects))

    # Generate training rows: each (subject, item) cell sampled with
    # success probability equal to true_difficulty[item] (subject-
    # independent for simplicity -- the test isn't trying to verify
    # subject-conditioned aggregation, just kNN structure).
    rows_s, rows_i, rows_y = [], [], []
    for s in range(n_subjects):
        # Each subject sees a random subset of items.
        item_subset = rng.choice(n_items, size=20, replace=False)
        for i in item_subset:
            p = float(true_difficulty[i])
            y = float(rng.random() < p)
            rows_s.append(s)
            rows_i.append(int(i))
            rows_y.append(y)
    return {
        "item_keys": item_keys,
        "item_embeddings": item_embeddings,
        "subject_keys": subject_keys,
        "subject_ids_per_row": np.array(rows_s, dtype=np.int64),
        "item_ids_per_row": np.array(rows_i, dtype=np.int64),
        "labels": np.array(rows_y, dtype=np.float64),
        "true_difficulty": true_difficulty,
    }


def test_fit_member5_recovers_difficulty_ordering():
    """Predicted_difficulty should be monotonically related to
    true_difficulty (Spearman correlation > 0.7 on a noisy synthetic
    case)."""
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    inputs = _build_synthetic_member5_inputs(seed=42)
    truth = inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=8, tau=0.05, ridge_alpha=0.1,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    # Re-key the predicted difficulty back to original order via sort_order.
    pred_orig_order = np.empty_like(state.predicted_difficulty)
    pred_orig_order[state.sort_order] = state.predicted_difficulty
    rho, _ = spearmanr(pred_orig_order, truth)
    assert rho > 0.7, f"Spearman correlation only {rho:.3f}; projection broken"


def test_fit_member5_produces_sorted_state():
    inputs = _build_synthetic_member5_inputs(seed=1)
    inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=5, tau=0.05, ridge_alpha=0.1,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    assert state.n_items == 80
    assert state.n_subjects == 30
    diffs = np.diff(state.predicted_difficulty)
    assert np.all(diffs >= -1e-6), "predicted_difficulty must be sorted ascending"


def test_apply_one_returns_valid_probability():
    inputs = _build_synthetic_member5_inputs(seed=2)
    inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=5, tau=0.1, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    q_emb = inputs["item_embeddings"][0]
    for s in [0, 5, 29]:
        p = apply_one(state, q_emb, f"subj_{s}")
        assert 0.0 < p < 1.0
        assert math.isfinite(p)
    # Unknown subject falls through to global_mean.
    p_unk = apply_one(state, q_emb, "subj_does_not_exist")
    assert 0.0 < p_unk < 1.0


def test_apply_batch_via_ids_matches_apply_one():
    inputs = _build_synthetic_member5_inputs(seed=3)
    inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=5, tau=0.1, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    s_ids = np.array([0, 5, 10, 29], dtype=np.int64)
    q_embs = inputs["item_embeddings"][:4]
    p_batch = apply_batch_via_ids(
        state, subject_ids=s_ids, query_item_embeddings=q_embs,
    )
    for r in range(4):
        p_one = apply_one(state, q_embs[r], f"subj_{int(s_ids[r])}")
        assert abs(float(p_batch[r]) - p_one) < 1e-5, (
            f"row {r}: batch={p_batch[r]} vs one={p_one}"
        )


def test_apply_batch_via_keys_matches_via_ids():
    inputs = _build_synthetic_member5_inputs(seed=4)
    inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=5, tau=0.1, ridge_alpha=0.5,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    skeys = ["subj_0", "subj_5", "subj_10"]
    sids = np.array([0, 5, 10], dtype=np.int64)
    q_embs = inputs["item_embeddings"][:3]
    p_keys = apply_batch(state, subject_keys=skeys, query_item_embeddings=q_embs)
    p_ids = apply_batch_via_ids(state, subject_ids=sids, query_item_embeddings=q_embs)
    np.testing.assert_allclose(p_keys, p_ids, rtol=1e-6)


def test_state_save_load_round_trip(tmp_path: Path):
    inputs = _build_synthetic_member5_inputs(seed=5)
    inputs.pop("true_difficulty")
    state = fit_member5(
        **inputs, k=5, tau=0.05, ridge_alpha=0.1,
        item_fallback_weight=0.3, min_subjects_per_item=2,
    )
    out = tmp_path / "m5"
    state.save(out)
    assert (out / "member5_state.npz").exists()
    assert (out / "member5_meta.json").exists()
    loaded = Member5State.load(out)
    np.testing.assert_array_equal(loaded.predicted_difficulty, state.predicted_difficulty)
    np.testing.assert_array_equal(loaded.projection_weights, state.projection_weights)
    assert loaded.k == state.k
    assert loaded.tau == state.tau
    assert tuple(loaded.subject_keys) == tuple(state.subject_keys)
    # Apply on a query should match before/after.
    q_emb = inputs["item_embeddings"][0]
    p0 = apply_one(state, q_emb, "subj_0")
    p1 = apply_one(loaded, q_emb, "subj_0")
    assert abs(p0 - p1) < 1e-6


def test_fit_member5_rejects_too_few_qualifying_items():
    inputs = _build_synthetic_member5_inputs(seed=6)
    inputs.pop("true_difficulty")
    with pytest.raises(RuntimeError, match="meaningful projection"):
        fit_member5(
            **inputs, k=5, tau=0.05, ridge_alpha=0.1,
            item_fallback_weight=0.3,
            min_subjects_per_item=1000,  # impossibly strict; everything dropped
        )


# ---------------------------------------------------------------------------
# Gate 4c: projection-leakage probe
# ---------------------------------------------------------------------------


def test_gate4c_passes_when_fit_and_val_are_disjoint():
    result = assert_projection_disjoint_from_val(
        fit_item_keys=["a", "b", "c"],
        val_item_keys=["d", "e", "f"],
    )
    assert result["n_overlap"] == 0
    assert result["n_fit_items"] == 3
    assert result["n_val_items"] == 3


def test_gate4c_catches_single_overlap():
    with pytest.raises(AssertionError, match="GATE 4c"):
        assert_projection_disjoint_from_val(
            fit_item_keys=["a", "b", "c"],
            val_item_keys=["c", "d", "e"],  # 'c' overlaps
        )


def test_gate4c_catches_full_overlap():
    with pytest.raises(AssertionError, match="GATE 4c"):
        assert_projection_disjoint_from_val(
            fit_item_keys=["a", "b", "c"],
            val_item_keys=["a", "b", "c"],
        )
