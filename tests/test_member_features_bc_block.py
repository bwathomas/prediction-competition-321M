"""Regression: bc-redaction + u_bc / cross-term blocks in member_features.

Two concerns:
1. ``bc_redacted=True`` zeroes the cond one-hot and (when bc block
   enabled) the u_bc + cross_theta_u_bc blocks; cross_theta_u_s
   (subject-only) remains.
2. The schema is forward-compatible: legacy callers (k_bc_factors=0)
   keep the old layout and the legacy serialized form deserializes
   into a schema with all bc offsets == 0.
"""

from __future__ import annotations

import numpy as np

from src.member_features import (
    MemberFeatureSchema,
    MemberSubjectTables,
    build_member_features,
    build_member_features_one,
)


def _schema(k_bc_factors: int = 0):
    return MemberFeatureSchema.fit(
        k_factors=4,
        n_clusters=3,
        top_m_centroids=2,
        pool_feature_names=("p1", "p2"),
        pool_stats={"p1": {"mean": 0.0, "std": 1.0}, "p2": {"mean": 0.0, "std": 1.0}},
        nn_feature_names=("nn0", "nn1"),
        subject_cat_field_names=("language",),
        subject_cat_field_cardinalities=(5,),
        subject_num_field_names=("age",),
        train_conditions=["zero-shot", "few-shot", "zero-shot", "5-shot"],
        min_condition_count=1,
        k_bc_factors=int(k_bc_factors),
    )


def _subject_tables(n_subjects: int, k: int):
    rng = np.random.default_rng(0)
    return MemberSubjectTables(
        theta=rng.normal(size=n_subjects).astype(np.float32),
        u=rng.normal(size=(n_subjects, k)).astype(np.float32),
        subject_cat_ids=np.zeros((n_subjects, 1), dtype=np.int64),
        subject_num=np.zeros((n_subjects, 2), dtype=np.float32),
    )


def test_legacy_schema_dim_unchanged_when_k_bc_zero():
    """k_bc_factors=0 must not bump feature_dim relative to the legacy
    layout that existing caches were trained against."""
    s = _schema(k_bc_factors=0)
    expected_dim = 1 + 4 + 5 + 2 + 2 + 2 + 3 + 2 + 4
    assert s.feature_dim == expected_dim
    assert s.k_bc_factors == 0
    assert s.offset_u_bc == 0
    assert s.offset_cross_theta_u_s == 0
    assert s.offset_cross_theta_u_bc == 0


def test_bc_block_extends_schema_with_three_new_blocks():
    """k_bc_factors=8 adds u_bc(8) + cross_theta_u_s(4) + cross_theta_u_bc(8)."""
    s = _schema(k_bc_factors=8)
    legacy_dim = 1 + 4 + 5 + 2 + 2 + 2 + 3 + 2 + 4
    assert s.feature_dim == legacy_dim + 8 + 4 + 8
    assert s.k_bc_factors == 8
    assert s.offset_u_bc == legacy_dim
    assert s.offset_cross_theta_u_s == legacy_dim + 8
    assert s.offset_cross_theta_u_bc == legacy_dim + 8 + 4
    # Spot-check naming.
    assert s.feature_names[s.offset_u_bc] == "u_bc_0"
    assert s.feature_names[s.offset_cross_theta_u_s] == "cross_theta_u_s_0"
    assert s.feature_names[s.offset_cross_theta_u_bc] == "cross_theta_u_bc_0"


def test_bc_redacted_zeroes_cond_block_in_batch():
    s = _schema(k_bc_factors=0)
    tables = _subject_tables(2, s.k_factors)
    N = 3
    subj_idx = np.array([0, 1, 0], dtype=np.int64)
    pool = np.zeros((N, 2), dtype=np.float32)
    cd = np.zeros((N, 2), dtype=np.float32)
    cl = np.zeros(N, dtype=np.int64)
    nn = np.zeros((N, 2), dtype=np.float32)
    conditions = ["zero-shot", "zero-shot", "few-shot"]
    redacted = np.array([False, True, False])
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool, centroid_dists=cd,
        cluster_ids=cl, nn_feats=nn,
        conditions=conditions, bc_redacted=redacted,
    )
    cond_block = M[:, s.offset_cond : s.offset_cond + s.n_condition_one_hot]
    # Row 0: cond is zero-shot -> exactly one 1.0 in the block.
    assert cond_block[0].sum() == 1.0
    # Row 1: redacted -> all zero.
    assert cond_block[1].sum() == 0.0
    # Row 2: not redacted -> exactly one 1.0.
    assert cond_block[2].sum() == 1.0


def test_bc_redaction_zeroes_u_bc_and_cross_theta_u_bc_but_not_cross_theta_u_s():
    """When schema has bc block enabled, redaction zeroes u_bc and
    cross_theta_u_bc; cross_theta_u_s (subject-only) is preserved."""
    s = _schema(k_bc_factors=4)
    tables = _subject_tables(2, s.k_factors)
    N = 2
    subj_idx = np.array([0, 1], dtype=np.int64)
    pool = np.zeros((N, 2), dtype=np.float32)
    cd = np.zeros((N, 2), dtype=np.float32)
    cl = np.zeros(N, dtype=np.int64)
    nn = np.zeros((N, 2), dtype=np.float32)
    conditions = ["zero-shot", "zero-shot"]
    rng = np.random.default_rng(0)
    u_bc_per_row = rng.normal(size=(N, 4)).astype(np.float32)
    redacted = np.array([False, True])
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool, centroid_dists=cd,
        cluster_ids=cl, nn_feats=nn,
        conditions=conditions,
        bc_redacted=redacted, u_bc_per_row=u_bc_per_row,
    )
    # Row 0 (not redacted): u_bc block must equal the input u_bc.
    np.testing.assert_allclose(
        M[0, s.offset_u_bc : s.offset_u_bc + 4], u_bc_per_row[0]
    )
    # Row 1 (redacted): u_bc block must be all zero.
    np.testing.assert_array_equal(
        M[1, s.offset_u_bc : s.offset_u_bc + 4],
        np.zeros(4, dtype=np.float32),
    )
    # cross_theta_u_s is subject-only; it must be NONZERO on row 1 (the
    # subject's theta * u_s product depends on the subject, not bc).
    cross_us = M[1, s.offset_cross_theta_u_s : s.offset_cross_theta_u_s + s.k_factors]
    expected_cross_us = float(tables.theta[1]) * tables.u[1]
    np.testing.assert_allclose(cross_us, expected_cross_us, atol=1e-5)
    # cross_theta_u_bc on row 1 (redacted) must be all zero (because u_bc was zeroed).
    cross_ubc = M[1, s.offset_cross_theta_u_bc : s.offset_cross_theta_u_bc + 4]
    np.testing.assert_array_equal(cross_ubc, np.zeros(4, dtype=np.float32))


def test_per_row_builder_matches_batch_with_bc_block():
    """The per-row builder used at runtime must produce bit-identical
    output to the batch builder, including u_bc and cross blocks."""
    s = _schema(k_bc_factors=3)
    tables = _subject_tables(4, s.k_factors)
    rng = np.random.default_rng(7)
    N = 5
    subj_idx = rng.integers(0, 4, size=N).astype(np.int64)
    pool = rng.normal(size=(N, 2)).astype(np.float32)
    cd = np.abs(rng.normal(size=(N, 2))).astype(np.float32)
    cl = rng.integers(0, 4, size=N).astype(np.int64)
    nn = rng.normal(size=(N, 2)).astype(np.float32)
    conditions = ["zero-shot", "few-shot", "5-shot", "zero-shot", "few-shot"]
    u_bc_per_row = rng.normal(size=(N, 3)).astype(np.float32)
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool, centroid_dists=cd,
        cluster_ids=cl, nn_feats=nn,
        conditions=conditions, u_bc_per_row=u_bc_per_row,
    )
    for i in range(N):
        row_one = build_member_features_one(
            s,
            theta_s=float(tables.theta[subj_idx[i]]),
            u_s=tables.u[subj_idx[i]],
            subject_cat_ids=tables.subject_cat_ids[subj_idx[i]],
            subject_num=tables.subject_num[subj_idx[i]],
            pool_feats=pool[i],
            centroid_dists=cd[i],
            cluster_id=int(cl[i]),
            nn_feats=nn[i],
            condition=conditions[i],
            u_bc=u_bc_per_row[i],
        )
        np.testing.assert_allclose(
            M[i], row_one, atol=1e-6,
            err_msg=f"row {i} mismatch between batch and per-row builders",
        )


def test_to_dict_from_dict_roundtrips_bc_offsets():
    s = _schema(k_bc_factors=6)
    d = s.to_dict()
    s2 = MemberFeatureSchema.from_dict(d)
    assert s2.k_bc_factors == s.k_bc_factors
    assert s2.offset_u_bc == s.offset_u_bc
    assert s2.offset_cross_theta_u_s == s.offset_cross_theta_u_s
    assert s2.offset_cross_theta_u_bc == s.offset_cross_theta_u_bc
    assert s2.feature_dim == s.feature_dim
    assert s2.feature_names == s.feature_names


def test_to_dict_legacy_schema_version_when_bc_disabled():
    """Schemas with k_bc_factors=0 must report schema_version=1 so older
    cache validation paths that whitelist v1 keep working."""
    s = _schema(k_bc_factors=0)
    assert s.to_dict()["schema_version"] == 1


def test_to_dict_v2_when_bc_enabled():
    s = _schema(k_bc_factors=4)
    assert s.to_dict()["schema_version"] == 2
