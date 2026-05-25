"""Tests for the shared M2/M4 feature builder."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.member_features import (
    MemberFeatureSchema,
    MemberSubjectTables,
    build_member_features,
    build_member_features_one,
)


def _toy_schema():
    return MemberFeatureSchema.fit(
        k_factors=4,
        n_clusters=3,
        top_m_centroids=2,
        pool_feature_names=("token_len", "char_len"),
        pool_stats={
            "token_len": {"mean": 100.0, "std": 50.0},
            "char_len": {"mean": 500.0, "std": 200.0},
        },
        nn_feature_names=("passrate_mean", "coverage"),
        subject_cat_field_names=("organization", "family"),
        subject_cat_field_cardinalities=(4, 5),  # 4-token org, 5-token family
        subject_num_field_names=("log_params",),
        train_conditions=["zero-shot", "zero-shot", "few-shot", "five-shot"],
        min_condition_count=1,
        centroid_dist_names=("c0", "c1"),
    )


def _toy_subject_tables(n_subjects: int, k: int):
    rng = np.random.default_rng(0)
    return MemberSubjectTables(
        theta=rng.normal(size=n_subjects).astype(np.float32),
        u=rng.normal(size=(n_subjects, k)).astype(np.float32),
        subject_cat_ids=rng.integers(0, 3, size=(n_subjects, 2)).astype(np.int64),
        subject_num=rng.normal(size=(n_subjects, 2)).astype(np.float32),
    )


def test_schema_layout_offsets_match_feature_names():
    s = _toy_schema()
    # Block widths.
    assert s.k_factors == 4
    assert s.n_clusters == 3
    assert s.top_m_centroids == 2
    assert s.n_pool == 2
    assert s.n_nn == 2
    assert s.n_subject_cat_fields == 2
    assert s.n_subject_num_fields == 1
    # Subject-cat block: 4 + 5 = 9 cols.
    # Subject-num block: 2 cols (1 field x 2 = scaled + missing).
    # Pool: 2; centroid: 2; cluster: 3; nn: 2; cond: 4 (UNK + 3 real).
    expected_dim = 1 + 4 + (4 + 5) + 2 + 2 + 2 + 3 + 2 + 4
    assert s.feature_dim == expected_dim
    assert len(s.feature_names) == s.feature_dim
    # Spot-check column-prefix discipline.
    assert s.feature_names[s.offset_theta] == "theta_s"
    assert s.feature_names[s.offset_u] == "u_s_0"
    assert s.feature_names[s.offset_subj_cat].startswith("subj_cat__organization__")
    assert s.feature_names[s.offset_subj_num] == "subj_num__log_params"
    assert s.feature_names[s.offset_subj_num + 1] == "subj_miss__log_params"
    assert s.feature_names[s.offset_pool] == "pool__token_len"
    assert s.feature_names[s.offset_centroid] == "cd__c0"
    assert s.feature_names[s.offset_cluster] == "cluster__001"
    assert s.feature_names[s.offset_nn] == "nn__passrate_mean"
    assert s.feature_names[s.offset_cond].startswith("cond__")


def test_schema_serialization_roundtrip():
    s = _toy_schema()
    d = s.to_dict()
    s2 = MemberFeatureSchema.from_dict(d)
    assert s2.feature_names == s.feature_names
    assert s2.feature_dim == s.feature_dim
    assert s2.condition_to_col == s.condition_to_col
    assert s2.pool_means == s.pool_means
    assert s2.pool_stds == s.pool_stds
    # Round-trip through JSON to catch any non-JSON-friendly types.
    s3 = MemberFeatureSchema.from_dict(json.loads(json.dumps(d)))
    assert s3.feature_names == s.feature_names


def test_batch_builder_matches_per_row_builder():
    s = _toy_schema()
    n_subj = 7
    tables = _toy_subject_tables(n_subj, s.k_factors)

    rng = np.random.default_rng(1)
    N = 12
    subj_idx = rng.integers(0, n_subj, size=N).astype(np.int64)
    pool = rng.normal(loc=100.0, scale=20.0, size=(N, s.n_pool)).astype(np.float32)
    centroids = rng.uniform(0, 5.0, size=(N, s.top_m_centroids)).astype(np.float32)
    cluster = rng.integers(0, s.n_clusters + 1, size=N).astype(np.int64)
    nn_feats = rng.normal(size=(N, s.n_nn)).astype(np.float32)
    conditions = ["zero-shot", "few-shot", "unknown-cond", None, "", "zero-shot",
                  "five-shot", "five-shot", "zero-shot", "few-shot",
                  "few-shot", "zero-shot"]

    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool,
        centroid_dists=centroids,
        cluster_ids=cluster,
        nn_feats=nn_feats,
        conditions=conditions,
    )
    assert M.shape == (N, s.feature_dim)
    assert M.dtype == np.float32

    for i in range(N):
        row_one = build_member_features_one(
            s,
            theta_s=float(tables.theta[subj_idx[i]]),
            u_s=tables.u[subj_idx[i]],
            subject_cat_ids=tables.subject_cat_ids[subj_idx[i]],
            subject_num=tables.subject_num[subj_idx[i]],
            pool_feats=pool[i],
            centroid_dists=centroids[i],
            cluster_id=int(cluster[i]),
            nn_feats=nn_feats[i],
            condition=conditions[i],
        )
        np.testing.assert_allclose(M[i], row_one, atol=1e-6)


def test_pool_zscore_uses_schema_stats():
    s = _toy_schema()
    n_subj = 1
    tables = _toy_subject_tables(n_subj, s.k_factors)
    pool_means = np.asarray(s.pool_means, dtype=np.float32)
    pool_stds = np.asarray(s.pool_stds, dtype=np.float32)
    raw = np.array([[100.0, 500.0]], dtype=np.float32)  # exactly the means
    M = build_member_features(
        s, tables,
        subject_idx=np.array([0], dtype=np.int64),
        pool_feats=raw,
        centroid_dists=np.zeros((1, s.top_m_centroids), dtype=np.float32),
        cluster_ids=np.array([0], dtype=np.int64),
        nn_feats=np.zeros((1, s.n_nn), dtype=np.float32),
        conditions=["zero-shot"],
    )
    z = M[0, s.offset_pool : s.offset_pool + s.n_pool]
    np.testing.assert_allclose(z, [0.0, 0.0], atol=1e-6)


def test_unknown_condition_falls_through_to_unk_column():
    s = _toy_schema()
    tables = _toy_subject_tables(1, s.k_factors)
    M = build_member_features(
        s, tables,
        subject_idx=np.array([0], dtype=np.int64),
        pool_feats=np.zeros((1, s.n_pool), dtype=np.float32),
        centroid_dists=np.zeros((1, s.top_m_centroids), dtype=np.float32),
        cluster_ids=np.array([0], dtype=np.int64),
        nn_feats=np.zeros((1, s.n_nn), dtype=np.float32),
        conditions=["a-condition-never-seen-during-fit"],
    )
    # Cond block: only the UNK column (offset 0 within cond block) lights up.
    cond_block = M[0, s.offset_cond : s.offset_cond + s.n_condition_one_hot]
    assert int(cond_block.sum()) == 1
    assert cond_block[0] == 1.0


def test_cluster_id_zero_is_all_zeros_unk():
    s = _toy_schema()
    tables = _toy_subject_tables(1, s.k_factors)
    M = build_member_features(
        s, tables,
        subject_idx=np.array([0], dtype=np.int64),
        pool_feats=np.zeros((1, s.n_pool), dtype=np.float32),
        centroid_dists=np.zeros((1, s.top_m_centroids), dtype=np.float32),
        cluster_ids=np.array([0], dtype=np.int64),
        nn_feats=np.zeros((1, s.n_nn), dtype=np.float32),
        conditions=["zero-shot"],
    )
    cluster_block = M[0, s.offset_cluster : s.offset_cluster + s.n_clusters]
    assert int(cluster_block.sum()) == 0


def test_nan_inf_inputs_do_not_propagate():
    s = _toy_schema()
    tables = _toy_subject_tables(1, s.k_factors)
    pool = np.array([[np.nan, np.inf]], dtype=np.float32)
    M = build_member_features(
        s, tables,
        subject_idx=np.array([0], dtype=np.int64),
        pool_feats=pool,
        centroid_dists=np.array([[np.nan, -np.inf]], dtype=np.float32),
        cluster_ids=np.array([0], dtype=np.int64),
        nn_feats=np.array([[np.nan, np.inf]], dtype=np.float32),
        conditions=["zero-shot"],
    )
    # Belt-and-suspenders nan_to_num at the end of the builder must
    # keep the output finite.
    assert np.all(np.isfinite(M))


def test_one_row_builder_handles_oob_categorical_id():
    """Subject's ``cat_id`` >= cardinality should fall through to UNK=1."""
    s = _toy_schema()
    out = build_member_features_one(
        s,
        theta_s=0.0,
        u_s=np.zeros(s.k_factors, dtype=np.float32),
        # field 0 has cardinality 4; pass id=99 (oob).
        subject_cat_ids=np.array([99, 0], dtype=np.int64),
        subject_num=np.zeros(2, dtype=np.float32),
        pool_feats=np.zeros(s.n_pool, dtype=np.float32),
        centroid_dists=np.zeros(s.top_m_centroids, dtype=np.float32),
        cluster_id=0,
        nn_feats=np.zeros(s.n_nn, dtype=np.float32),
        condition="zero-shot",
    )
    # Field 0 should have the UNK column (offset 1 within field 0) lit.
    base = s.offset_subj_cat
    assert out[base + 1] == 1.0  # UNK token for org
