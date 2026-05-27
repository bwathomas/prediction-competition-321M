"""Tests for src/member2_features.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.member2_features import (
    Member2FeatureSchema,
    audit_no_embedding_features,
    build_member2_feature_matrix,
    build_member2_schema,
)


def _make_minimal_schema(
    subj_cat=("family",),
    subj_num=("age",),
    bench_cat=("benchmark_type",),
    bench_num=("difficulty",),
    interaction=("subj_cluster_mean", "subj_bc_mean"),
):
    return build_member2_schema(
        subject_cat_field_names=subj_cat,
        subject_num_field_names=subj_num,
        bench_cat_field_names=bench_cat,
        bench_num_field_names=bench_num,
        interaction_feature_names=interaction,
    )


def test_schema_has_expected_columns():
    s = _make_minimal_schema()
    expected = [
        "subject_idx", "subject_obs_count_log1p", "cluster_id",
        "bench_condition_id", "bc_redacted_mask",
        "subject_cat__family",
        "subject_num__age__value", "subject_num__age__mask",
        "bench_cat__benchmark_type",
        "bench_num__difficulty__value", "bench_num__difficulty__mask",
        "subj_cluster_mean", "subj_bc_mean",
    ]
    assert list(s.feature_names) == expected
    assert s.feature_dim == len(expected)


def test_schema_categorical_indices_match_named_cats():
    s = _make_minimal_schema()
    expected_cat_names = {
        "subject_idx", "cluster_id", "bench_condition_id",
        "subject_cat__family", "bench_cat__benchmark_type",
    }
    actual = {s.feature_names[i] for i in s.categorical_indices}
    assert actual == expected_cat_names


def test_build_matrix_shape_and_values():
    s = _make_minimal_schema()
    N = 4
    # 3 subjects, 2 cat fields each (family vocab size = 5, single num field)
    subj_cat_lookup = np.array([[1], [2], [3]], dtype=np.int32)
    subj_num_lookup = np.array(
        [[0.5, 0.0], [1.5, 0.0], [0.0, 1.0]], dtype=np.float32,
    )  # subj 2 has missing age
    bench_cat_lookup = np.array([[10], [20]], dtype=np.int32)
    bench_num_lookup = np.array([[0.3, 0.0], [0.7, 0.0]], dtype=np.float32)
    interaction = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32,
    )
    X = build_member2_feature_matrix(
        s,
        subject_ids=np.array([0, 1, 2, -1], dtype=np.int64),
        cluster_ids=np.array([5, 6, 7, -1], dtype=np.int64),
        bc_ids=np.array([0, 1, 0, -1], dtype=np.int64),
        bc_redacted_mask=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        subject_obs_count_log1p=np.array([2.0, 3.0, 0.0, 0.0], dtype=np.float32),
        subject_cat_lookup=subj_cat_lookup,
        subject_num_lookup=subj_num_lookup,
        bench_cat_lookup=bench_cat_lookup,
        bench_num_lookup=bench_num_lookup,
        interaction_matrix=interaction,
    )
    assert X.shape == (N, s.feature_dim)
    assert X.dtype == np.float32
    # Row 0: subj=0, count=2.0, cluster=5, bc=0, mask=0, fam=1, age_val=0.5,
    # age_mask=0, btype=10, diff_val=0.3, diff_mask=0, int=[0.1,0.2]
    expected_row0 = [0, 2.0, 5, 0, 0.0, 1, 0.5, 0.0, 10, 0.3, 0.0, 0.1, 0.2]
    np.testing.assert_allclose(X[0], expected_row0, rtol=1e-6)
    # Row 3: UNK subject and UNK bc. subj_cat, subj_num use safe lookup but
    # then get overwritten with UNK semantics (0, 0, MISSING mask=1).
    # Row 3: subj=-1 -> subj_idx=-1, count=0; cluster=-1; bc=-1; mask=0;
    # subj_cat=0 (UNK semantics); age_val=0, age_mask=1; bench_cat=0;
    # diff_val=0, diff_mask=1; int=[0.7,0.8]
    expected_row3 = [-1, 0.0, -1, -1, 0.0, 0, 0.0, 1.0, 0, 0.0, 1.0, 0.7, 0.8]
    np.testing.assert_allclose(X[3], expected_row3, rtol=1e-6)


def test_build_matrix_rejects_wrong_interaction_dim():
    s = _make_minimal_schema()
    with pytest.raises(ValueError, match="interaction_matrix"):
        build_member2_feature_matrix(
            s,
            subject_ids=np.array([0], dtype=np.int64),
            cluster_ids=np.array([0], dtype=np.int64),
            bc_ids=np.array([0], dtype=np.int64),
            bc_redacted_mask=np.array([0.0], dtype=np.float32),
            subject_obs_count_log1p=np.array([0.0], dtype=np.float32),
            subject_cat_lookup=np.zeros((1, 1), dtype=np.int32),
            subject_num_lookup=np.zeros((1, 2), dtype=np.float32),
            bench_cat_lookup=np.zeros((1, 1), dtype=np.int32),
            bench_num_lookup=np.zeros((1, 2), dtype=np.float32),
            interaction_matrix=np.zeros((1, 5), dtype=np.float32),
        )


def test_audit_passes_on_clean_schema():
    s = _make_minimal_schema()
    audit_no_embedding_features(s)


def test_audit_catches_pool_feature():
    bad = Member2FeatureSchema(
        feature_names=("subject_idx", "pool_feat_42", "cluster_id"),
        categorical_indices=(0, 2),
    )
    with pytest.raises(AssertionError, match="embedding-derived"):
        audit_no_embedding_features(bad)


def test_audit_catches_centroid_distance():
    bad = Member2FeatureSchema(
        feature_names=("subject_idx", "centroid_dist_3"),
        categorical_indices=(0,),
    )
    with pytest.raises(AssertionError, match="centroid_dist"):
        audit_no_embedding_features(bad)


def test_audit_catches_nn_feature():
    bad = Member2FeatureSchema(
        feature_names=("subject_idx", "nn_passrate_mean"),
        categorical_indices=(0,),
    )
    with pytest.raises(AssertionError, match="nn_"):
        audit_no_embedding_features(bad)


def test_audit_catches_theta_or_u_columns():
    bad1 = Member2FeatureSchema(
        feature_names=("subject_idx", "theta_0"),
        categorical_indices=(0,),
    )
    with pytest.raises(AssertionError, match="theta"):
        audit_no_embedding_features(bad1)
    bad2 = Member2FeatureSchema(
        feature_names=("subject_idx", "u__0"),
        categorical_indices=(0,),
    )
    with pytest.raises(AssertionError, match="u__"):
        audit_no_embedding_features(bad2)


def test_audit_does_not_false_positive_on_user_subject_cat():
    """Make sure the 'u' / 'u__' detector doesn't trip on legitimate
    subject_cat__user_... or similar names."""
    ok = Member2FeatureSchema(
        feature_names=(
            "subject_idx",
            "subject_cat__user_id_bucket",
            "subject_obs_count_log1p",
        ),
        categorical_indices=(0, 1),
    )
    audit_no_embedding_features(ok)  # must not raise
