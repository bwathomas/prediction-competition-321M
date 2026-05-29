"""Tests for ``src/member5_subject_cluster_residual.py``.

Covers:

* Marginal recovery (no interaction -> residual ~ 0).
* Interaction recovery (synthetic subject x cluster effect -> non-zero
  residual_logit on the right cells).
* Cold-start fall-through (unknown subject, unknown cluster, both).
* Save / load round-trip preserves predictions byte-for-byte.
* apply_state_batch matches apply_state_one across the same rows.
* Shape / dtype / range invariants of the trained state.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.member5_subject_cluster_residual import (
    Member5ResidualState,
    apply_state_batch,
    apply_state_one,
    fit_member5_residual,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_subject_keys(n_subjects: int) -> tuple[str, ...]:
    return tuple(f"subj_{i:03d}" for i in range(int(n_subjects)))


def _balanced_rows(
    n_subjects: int,
    n_clusters: int,
    per_cell: int,
    base_rate: float = 0.5,
    interaction_lift: dict[tuple[int, int], float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate balanced rows with optional per-(s, c) lift in pass rate.

    Returns (subject_ids, cluster_ids, labels) with ``labels`` in {0, 1}.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for s in range(int(n_subjects)):
        for c in range(int(n_clusters)):
            p = float(base_rate)
            if interaction_lift is not None and (s, c) in interaction_lift:
                p = float(max(0.0, min(1.0, p + interaction_lift[(s, c)])))
            for _ in range(int(per_cell)):
                rows.append((s, c, int(rng.random() < p)))
    arr = np.asarray(rows, dtype=np.int64)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(np.float64)


# ---------------------------------------------------------------------------
# State invariants
# ---------------------------------------------------------------------------


def test_state_shapes_and_finiteness() -> None:
    n_subjects, n_clusters = 5, 7
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=20, base_rate=0.4, seed=11,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    assert state.n_subjects == n_subjects
    assert state.n_clusters == n_clusters
    assert state.subj_logit.shape == (n_subjects,)
    assert state.cluster_logit.shape == (n_clusters,)
    assert state.residual_logit.shape == (n_subjects, n_clusters)
    assert state.cell_log1p_n.shape == (n_subjects, n_clusters)
    assert state.subj_logit.dtype == np.float32
    assert state.cluster_logit.dtype == np.float32
    assert state.residual_logit.dtype == np.float32
    assert state.cell_log1p_n.dtype == np.float32
    assert np.isfinite(state.subj_logit).all()
    assert np.isfinite(state.cluster_logit).all()
    assert np.isfinite(state.residual_logit).all()
    assert np.isfinite(state.cell_log1p_n).all()
    assert math.isfinite(state.global_mean_logit)


def test_no_interaction_residual_collapses_toward_zero() -> None:
    """Pure base-rate data with no per-cell lift -> residual ~ 0 on
    every well-sampled cell."""
    n_subjects, n_clusters = 6, 8
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=400, base_rate=0.5, seed=3,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
        smoothing_cell=30.0,
        smoothing_marginal=10.0,
    )
    # Every well-sampled cell's residual should be within ~0.4 logits
    # of zero. The 95th percentile binomial fluctuation at p=0.5, n=400
    # is ~5%, which is ~0.2 logits; we leave headroom for the
    # Bayesian-shrinkage residual term.
    assert np.abs(state.residual_logit).max() < 0.5, (
        f"max |residual_logit| = {np.abs(state.residual_logit).max():.3f} "
        "on data with no interaction; shrinkage too weak?"
    )


def test_interaction_recovered_in_correct_cell() -> None:
    """One specific (s, c) cell has a large pass-rate lift; its
    residual_logit should be the largest in the matrix."""
    n_subjects, n_clusters = 5, 6
    lift = {(2, 3): +0.45}  # subj 2 on cluster 3 is much easier
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=500, base_rate=0.3, interaction_lift=lift, seed=7,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
        smoothing_cell=30.0,
    )
    # The (2, 3) cell should hold the maximum residual (positive lift).
    r = state.residual_logit
    flat_argmax = int(np.argmax(r))
    s_max, c_max = divmod(flat_argmax, n_clusters)
    assert (s_max, c_max) == (2, 3), (
        f"expected max residual at (2, 3), got ({s_max}, {c_max}) "
        f"with value {r[s_max, c_max]:.3f}"
    )
    # And it should be meaningfully positive (lift -> non-trivial logit).
    assert r[2, 3] > 0.4, (
        f"lift cell residual_logit = {r[2, 3]:.3f}; expected > 0.4"
    )


# ---------------------------------------------------------------------------
# Apply path
# ---------------------------------------------------------------------------


def test_apply_batch_matches_apply_one() -> None:
    n_subjects, n_clusters = 4, 5
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=50, base_rate=0.55,
        interaction_lift={(1, 2): -0.2, (3, 4): +0.3},
        seed=17,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    rng = np.random.default_rng(0)
    query_s = rng.integers(0, n_subjects, size=37)
    query_c = rng.integers(0, n_clusters, size=37)
    p_batch = apply_state_batch(
        state, subject_ids=query_s, cluster_ids=query_c
    )
    p_one = np.fromiter(
        (
            apply_state_one(state, subject_id=int(s), cluster_id=int(c))
            for s, c in zip(query_s, query_c)
        ),
        dtype=np.float32,
        count=int(query_s.shape[0]),
    )
    np.testing.assert_allclose(p_batch, p_one, atol=1e-6)


def test_apply_probabilities_in_range_and_finite() -> None:
    n_subjects, n_clusters = 3, 4
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=30, base_rate=0.7, seed=42,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    rng = np.random.default_rng(0)
    q_s = rng.integers(0, n_subjects, size=200)
    q_c = rng.integers(0, n_clusters, size=200)
    p = apply_state_batch(state, subject_ids=q_s, cluster_ids=q_c)
    assert p.dtype == np.float32
    assert p.shape == (200,)
    assert np.all(np.isfinite(p))
    assert (p > 0.0).all() and (p < 1.0).all()


# ---------------------------------------------------------------------------
# Cold-start fall-through
# ---------------------------------------------------------------------------


def test_unknown_subject_uses_cluster_marginal() -> None:
    n_subjects, n_clusters = 3, 4
    # Build a state where cluster 1 is harder than cluster 0.
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=200, base_rate=0.5,
        interaction_lift={(s, 1): -0.3 for s in range(n_subjects)},
        seed=1,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    # An unknown subject (-1) on cluster 0 vs cluster 1 should respect
    # the cluster's marginal difficulty: cluster-0 prediction > cluster-1.
    p0 = apply_state_one(state, subject_id=-1, cluster_id=0)
    p1 = apply_state_one(state, subject_id=-1, cluster_id=1)
    assert p0 > p1, (
        f"unknown-subject preds should follow cluster marginal; "
        f"got p(cluster 0)={p0:.3f}, p(cluster 1)={p1:.3f}"
    )


def test_unknown_cluster_uses_subject_marginal() -> None:
    n_subjects, n_clusters = 4, 3
    # Subject 2 is much better than subject 0 on every cluster.
    lift = {(2, c): +0.3 for c in range(n_clusters)}
    lift.update({(0, c): -0.3 for c in range(n_clusters)})
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=200, base_rate=0.5, interaction_lift=lift, seed=2,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    p_high = apply_state_one(state, subject_id=2, cluster_id=-1)
    p_low = apply_state_one(state, subject_id=0, cluster_id=-1)
    assert p_high > p_low, (
        f"unknown-cluster preds should follow subject marginal; "
        f"got p(subj 2)={p_high:.3f}, p(subj 0)={p_low:.3f}"
    )


def test_both_unknown_returns_global_prob() -> None:
    n_subjects, n_clusters = 3, 4
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=200, base_rate=0.7, seed=99,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    p = apply_state_one(state, subject_id=-1, cluster_id=-1)
    target = 1.0 / (1.0 + math.exp(-state.global_mean_logit))
    assert abs(p - target) < 1.0e-4, (
        f"both-unknown prediction {p:.4f} != "
        f"sigmoid(global_mean_logit) {target:.4f}"
    )


def test_out_of_range_ids_handled_safely() -> None:
    n_subjects, n_clusters = 3, 4
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=50, base_rate=0.5, seed=8,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    # Mix valid + negative + way-above-range ids; nothing should NaN or
    # raise. Batch and per-row should agree.
    q_s = np.array([0, -1, 2, 999, 1, -100], dtype=np.int64)
    q_c = np.array([1, 2, -1, 3, 999, -999], dtype=np.int64)
    p_batch = apply_state_batch(state, subject_ids=q_s, cluster_ids=q_c)
    assert np.all(np.isfinite(p_batch))
    assert (p_batch > 0.0).all() and (p_batch < 1.0).all()
    for i, (s, c) in enumerate(zip(q_s, q_c)):
        p_one = apply_state_one(
            state, subject_id=int(s), cluster_id=int(c)
        )
        np.testing.assert_allclose(p_batch[i], p_one, atol=1.0e-6)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_predictions() -> None:
    n_subjects, n_clusters = 4, 5
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=80, base_rate=0.4,
        interaction_lift={(1, 2): +0.25, (3, 0): -0.2}, seed=33,
    )
    state = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
    )
    rng = np.random.default_rng(1)
    q_s = rng.integers(0, n_subjects, size=53)
    q_c = rng.integers(0, n_clusters, size=53)
    pred_before = apply_state_batch(state, subject_ids=q_s, cluster_ids=q_c)
    with tempfile.TemporaryDirectory() as tmp:
        state.save(tmp)
        reloaded = Member5ResidualState.load(tmp)
    assert reloaded.n_subjects == state.n_subjects
    assert reloaded.n_clusters == state.n_clusters
    assert reloaded.subject_keys == state.subject_keys
    np.testing.assert_array_equal(reloaded.subj_logit, state.subj_logit)
    np.testing.assert_array_equal(reloaded.cluster_logit, state.cluster_logit)
    np.testing.assert_array_equal(reloaded.residual_logit, state.residual_logit)
    pred_after = apply_state_batch(
        reloaded, subject_ids=q_s, cluster_ids=q_c
    )
    np.testing.assert_array_equal(pred_before, pred_after)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_fit_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        fit_member5_residual(
            subject_ids=np.zeros(10, dtype=np.int64),
            cluster_ids=np.zeros(9, dtype=np.int64),
            labels=np.zeros(10, dtype=np.float64),
            subject_keys=_make_subject_keys(3),
            n_clusters=4,
        )


def test_fit_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_member5_residual(
            subject_ids=np.zeros(0, dtype=np.int64),
            cluster_ids=np.zeros(0, dtype=np.int64),
            labels=np.zeros(0, dtype=np.float64),
            subject_keys=_make_subject_keys(3),
            n_clusters=4,
        )


def test_fit_rejects_all_out_of_range_ids() -> None:
    with pytest.raises(RuntimeError, match="no rows"):
        fit_member5_residual(
            subject_ids=np.full(20, -1, dtype=np.int64),
            cluster_ids=np.full(20, -1, dtype=np.int64),
            labels=np.zeros(20, dtype=np.float64),
            subject_keys=_make_subject_keys(3),
            n_clusters=4,
        )


def test_residual_scale_attenuates_residual() -> None:
    """residual_scale=0 should collapse predictions onto the additive
    (subject + cluster - global) baseline regardless of cell counts."""
    n_subjects, n_clusters = 3, 4
    s_ids, c_ids, y = _balanced_rows(
        n_subjects=n_subjects, n_clusters=n_clusters,
        per_cell=200, base_rate=0.4,
        interaction_lift={(1, 2): +0.4}, seed=21,
    )
    state_full = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
        residual_scale=1.0,
    )
    state_zero = fit_member5_residual(
        subject_ids=s_ids,
        cluster_ids=c_ids,
        labels=y,
        subject_keys=_make_subject_keys(n_subjects),
        n_clusters=n_clusters,
        residual_scale=0.0,
    )
    # The strong (1, 2) lift should be visible in the full state but
    # completely zeroed out by residual_scale=0.
    assert abs(state_full.residual_logit[1, 2]) > 0.5
    assert abs(state_zero.residual_logit[1, 2]) < 1.0e-6
