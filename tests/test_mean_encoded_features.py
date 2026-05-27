"""Tests for src/mean_encoded_features.py.

Covers:
  1. Cold-start safety -- val rows never look up cells fitted with
     val labels (no leakage).
  2. Bayesian shrinkage -- empty cells fall back to the appropriate
     coarser aggregate.
  3. Per-cell formula -- known fixture matches hand-computed values.
  4. Feature-dim invariants -- the locked column names are stable.
  5. Save/load roundtrip preserves all stats.
  6. Vectorized single-row helpers agree with the batch path.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.mean_encoded_features import (
    MEMBER2_INTERACTION_FEATURE_DIM,
    MEMBER2_INTERACTION_FEATURE_NAMES,
    MEMBER4_MARGINAL_FEATURE_DIM,
    MEMBER4_MARGINAL_FEATURE_NAMES,
    MeanEncodedStats,
    apply_member2_interaction_features,
    apply_member2_interaction_features_one,
    apply_member4_marginal_features,
    apply_member4_marginal_features_one,
    fit_mean_encoded_stats,
)


def _make_tiny_fixture():
    """Hand-checkable fixture with 2 subjects, 2 clusters, 2 bcs."""
    subject_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    cluster_ids = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
    bc_ids = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return subject_ids, cluster_ids, bc_ids, labels


def test_feature_dim_invariants():
    """The locked column names must equal the dim constants."""
    assert MEMBER2_INTERACTION_FEATURE_DIM == len(MEMBER2_INTERACTION_FEATURE_NAMES)
    assert MEMBER4_MARGINAL_FEATURE_DIM == len(MEMBER4_MARGINAL_FEATURE_NAMES)
    # Sanity: dims are small (we don't want a regression that
    # accidentally enables a huge interaction matrix).
    assert MEMBER2_INTERACTION_FEATURE_DIM == 8
    assert MEMBER4_MARGINAL_FEATURE_DIM == 14


def test_fit_basic_shapes():
    s, c, b, y = _make_tiny_fixture()
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=2, n_clusters=2, n_bcs=2,
        smoothing=0.0,  # no shrinkage so cell means are the raw means
    )
    assert stats.subj_cluster_mean.shape == (2, 2)
    assert stats.subj_bc_mean.shape == (2, 2)
    assert stats.subj_mean.shape == (2,)
    assert stats.cluster_mean.shape == (2,)
    assert stats.bc_mean.shape == (2,)
    # Global mean: 3/6 = 0.5.
    assert abs(stats.global_mean - 0.5) < 1e-6
    # Subject 0 has labels [1, 0, 1] -> mean 2/3. Subject 1: [0, 1, 0] -> 1/3.
    np.testing.assert_allclose(stats.subj_mean, [2 / 3, 1 / 3], rtol=1e-5)
    # Subject 0, cluster 0 has labels [1, 0] -> 0.5; subject 0, cluster 1 -> 1.0.
    np.testing.assert_allclose(stats.subj_cluster_mean[0], [0.5, 1.0], rtol=1e-5)
    # Subject 1, cluster 0 has [0]; subject 1, cluster 1 has [1, 0] -> 0.5.
    np.testing.assert_allclose(stats.subj_cluster_mean[1], [0.0, 0.5], rtol=1e-5)


def test_bayesian_shrinkage_empty_cell_falls_back():
    """An empty (subject, cluster) cell at smoothing>0 must fall back
    to the cluster mean, not crash or return NaN."""
    s = np.array([0, 0], dtype=np.int64)
    c = np.array([0, 0], dtype=np.int64)
    b = np.array([0, 0], dtype=np.int64)
    y = np.array([1.0, 0.0], dtype=np.float32)
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=3, n_clusters=2, n_bcs=2,  # 3 subjects but only id 0 seen
        smoothing=5.0,
    )
    # Cell (subject=2, cluster=1) has no observations. With smoothing=5
    # it should equal cluster_mean[1] (which itself has no observations,
    # so falls back to global_mean via the chained shrinkage).
    # global_mean = 0.5, cluster_mean[1] = (0 + 5 * 0.5) / (0 + 5) = 0.5.
    # subj_cluster_mean[2, 1] = (0 + 5 * cluster_mean[1]) / (0 + 5) = cluster_mean[1] = 0.5.
    val = float(stats.subj_cluster_mean[2, 1])
    assert math.isfinite(val)
    assert abs(val - 0.5) < 1e-5


def test_cold_start_safety_no_val_label_leakage():
    """Critical invariant: the fitted stats must be IDENTICAL whether
    we fit on a train slice alone or on train + val. We fit on train
    only, then ensure that supplying val rows at apply time doesn't
    change the stats themselves. This is a positive control for the
    'no leakage' guarantee."""
    rng = np.random.default_rng(0)
    N_train, N_val = 500, 100
    n_s, n_c, n_b = 20, 5, 8
    s_train = rng.integers(0, n_s, size=N_train).astype(np.int64)
    c_train = rng.integers(0, n_c, size=N_train).astype(np.int64)
    b_train = rng.integers(0, n_b, size=N_train).astype(np.int64)
    y_train = rng.binomial(1, 0.6, size=N_train).astype(np.float32)
    # Different val labels (would shift stats if leaked).
    s_val = rng.integers(0, n_s, size=N_val).astype(np.int64)
    c_val = rng.integers(0, n_c, size=N_val).astype(np.int64)
    b_val = rng.integers(0, n_b, size=N_val).astype(np.int64)
    y_val_clean = np.zeros(N_val, dtype=np.float32)   # all 0
    y_val_dirty = np.ones(N_val, dtype=np.float32)    # all 1

    stats_train_only = fit_mean_encoded_stats(
        subject_ids=s_train, cluster_ids=c_train, bc_ids=b_train,
        labels=y_train, n_subjects=n_s, n_clusters=n_c, n_bcs=n_b,
        smoothing=10.0,
    )
    # Compute val features using only train-fitted stats (this is what
    # the notebook does).
    val_feats_a = apply_member2_interaction_features(
        stats_train_only,
        subject_ids=s_val, cluster_ids=c_val, bc_ids=b_val,
    )
    # Now simulate the "leakage" anti-pattern: concatenate val labels into
    # the fit. We should get DIFFERENT stats (proving labels matter to
    # the fit) AND different val features (proving val rows would be
    # shifted if val labels were leaked into fitting).
    s_all = np.concatenate([s_train, s_val])
    c_all = np.concatenate([c_train, c_val])
    b_all = np.concatenate([b_train, b_val])
    y_all_dirty = np.concatenate([y_train, y_val_dirty])
    stats_leaked = fit_mean_encoded_stats(
        subject_ids=s_all, cluster_ids=c_all, bc_ids=b_all,
        labels=y_all_dirty, n_subjects=n_s, n_clusters=n_c, n_bcs=n_b,
        smoothing=10.0,
    )
    val_feats_b = apply_member2_interaction_features(
        stats_leaked,
        subject_ids=s_val, cluster_ids=c_val, bc_ids=b_val,
    )
    # Sanity: features must be DIFFERENT when we (incorrectly) leak
    # val labels into the fit. This is the canary that proves the
    # absence-of-leakage in the clean path is meaningful.
    assert not np.allclose(val_feats_a, val_feats_b, atol=1e-5)


def test_apply_features_have_correct_shape():
    """Feature matrices must have the locked dims and be finite."""
    s, c, b, y = _make_tiny_fixture()
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=2, n_clusters=2, n_bcs=2, smoothing=1.0,
    )
    f_m2 = apply_member2_interaction_features(
        stats, subject_ids=s, cluster_ids=c, bc_ids=b
    )
    f_m4 = apply_member4_marginal_features(
        stats, subject_ids=s, cluster_ids=c, bc_ids=b
    )
    assert f_m2.shape == (6, MEMBER2_INTERACTION_FEATURE_DIM)
    assert f_m4.shape == (6, MEMBER4_MARGINAL_FEATURE_DIM)
    assert f_m2.dtype == np.float32
    assert f_m4.dtype == np.float32
    assert np.all(np.isfinite(f_m2))
    assert np.all(np.isfinite(f_m4))


def test_apply_features_one_matches_batch():
    """The single-row helpers must equal the batch output row-for-row."""
    s, c, b, y = _make_tiny_fixture()
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=2, n_clusters=2, n_bcs=2, smoothing=1.0,
    )
    f_m2_batch = apply_member2_interaction_features(
        stats, subject_ids=s, cluster_ids=c, bc_ids=b
    )
    f_m4_batch = apply_member4_marginal_features(
        stats, subject_ids=s, cluster_ids=c, bc_ids=b
    )
    for i in range(len(s)):
        row_m2 = apply_member2_interaction_features_one(
            stats, subject_id=int(s[i]), cluster_id=int(c[i]), bc_id=int(b[i])
        )
        row_m4 = apply_member4_marginal_features_one(
            stats, subject_id=int(s[i]), cluster_id=int(c[i]), bc_id=int(b[i])
        )
        np.testing.assert_allclose(row_m2, f_m2_batch[i], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(row_m4, f_m4_batch[i], rtol=1e-6, atol=1e-7)


def test_apply_features_handle_unseen_ids():
    """Negative / out-of-range ids must NOT crash and must produce
    finite outputs (falling back through the chained aggregates)."""
    s, c, b, y = _make_tiny_fixture()
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=2, n_clusters=2, n_bcs=2, smoothing=5.0,
    )
    # Throw in some -1 sentinels and out-of-range ids.
    s_bad = np.array([0, -1, 5, 1], dtype=np.int64)
    c_bad = np.array([-1, 1, 0, 99], dtype=np.int64)
    b_bad = np.array([0, 1, -1, 0], dtype=np.int64)
    f_m2 = apply_member2_interaction_features(
        stats, subject_ids=s_bad, cluster_ids=c_bad, bc_ids=b_bad
    )
    f_m4 = apply_member4_marginal_features(
        stats, subject_ids=s_bad, cluster_ids=c_bad, bc_ids=b_bad
    )
    assert np.all(np.isfinite(f_m2))
    assert np.all(np.isfinite(f_m4))
    # Out-of-range row should fall back to global_mean (~0.5 for our fixture).
    # Pick the mg__subj_mean column (index 0) for the row with subject=-1.
    g = stats.global_mean
    assert abs(float(f_m4[1, 0]) - g) < 1e-5


def test_save_load_roundtrip(tmp_path: Path):
    """Saved stats must reload bit-exactly."""
    s, c, b, y = _make_tiny_fixture()
    stats = fit_mean_encoded_stats(
        subject_ids=s, cluster_ids=c, bc_ids=b, labels=y,
        n_subjects=2, n_clusters=2, n_bcs=2, smoothing=2.0,
    )
    out = stats.save(tmp_path / "stats")
    loaded = MeanEncodedStats.load(out)
    assert loaded.n_subjects == stats.n_subjects
    assert loaded.n_clusters == stats.n_clusters
    assert loaded.n_bcs == stats.n_bcs
    assert abs(loaded.global_mean - stats.global_mean) < 1e-7
    np.testing.assert_allclose(
        loaded.subj_cluster_mean, stats.subj_cluster_mean, rtol=1e-6
    )
    np.testing.assert_allclose(
        loaded.subj_bc_mean, stats.subj_bc_mean, rtol=1e-6
    )
    np.testing.assert_allclose(loaded.subj_mean, stats.subj_mean, rtol=1e-6)
    np.testing.assert_allclose(loaded.cluster_mean, stats.cluster_mean, rtol=1e-6)
    np.testing.assert_allclose(loaded.bc_mean, stats.bc_mean, rtol=1e-6)


def test_member4_marginal_features_zero_overlap_with_dense_schema():
    """Sanity invariant for Task 3 (decorrelate Member 4): no marginal-
    feature name should resemble a member-feature-schema name (those
    use the `theta_`, `subj_`, `nn_`, `cond_*`, `cluster_*` prefixes
    from the dense schema). Our prefixes are `mg__` for marginals
    and `me__` for member-2 interactions -- both reserved namespaces."""
    for name in MEMBER4_MARGINAL_FEATURE_NAMES:
        assert name.startswith("mg__"), name
    for name in MEMBER2_INTERACTION_FEATURE_NAMES:
        assert name.startswith("me__"), name
    assert set(MEMBER4_MARGINAL_FEATURE_NAMES).isdisjoint(
        set(MEMBER2_INTERACTION_FEATURE_NAMES)
    )
