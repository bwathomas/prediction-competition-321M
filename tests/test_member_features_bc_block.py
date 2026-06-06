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

import json

import numpy as np

from src.member_features import (
    MemberBenchmarkTables,
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


def _schema_v3(k_bc_factors: int = 0):
    """Same as ``_schema`` but with a benchmark-metadata block.

    2 benchmark categorical fields (topic card 4, domain card 3) and
    1 benchmark numeric field (benchmark_age -> 2 cols).
    """
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
        benchmark_cat_field_names=("topic", "domain"),
        benchmark_cat_field_cardinalities=(4, 3),
        benchmark_num_field_names=("benchmark_age",),
    )


def _benchmark_tables(n_bc: int, n_cat: int, n_num: int):
    rng = np.random.default_rng(11)
    return MemberBenchmarkTables(
        benchmark_cat_ids=rng.integers(0, 3, size=(n_bc, n_cat)).astype(np.int64),
        benchmark_num=rng.normal(size=(n_bc, 2 * n_num)).astype(np.float32),
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


# ---------------------------------------------------------------------------
# schema_version 3 -- benchmark-metadata block
# ---------------------------------------------------------------------------


def test_v3_block_extends_schema_at_the_end():
    """The benchmark block is appended after everything else (incl. the
    bc block when present), so the v2 prefix is byte-identical."""
    s2 = _schema(k_bc_factors=0)
    s3 = _schema_v3(k_bc_factors=0)
    # bench_cat = 4 + 3 = 7 cols; bench_num = 2 * 1 = 2 cols.
    assert s3.feature_dim == s2.feature_dim + 7 + 2
    assert s3.n_bench_cat_fields == 2
    assert s3.n_bench_num_fields == 1
    assert s3.offset_bench_cat == s2.feature_dim
    assert s3.offset_bench_num == s2.feature_dim + 7
    assert s3.bench_cat_field_offsets == (0, 4)
    assert s3.benchmark_cat_field_cardinalities == (4, 3)
    # The v2-prefix column names are identical.
    assert s3.feature_names[: s2.feature_dim] == s2.feature_names
    # Naming convention.
    assert s3.feature_names[s3.offset_bench_cat] == "bench_cat__topic__000"
    assert s3.feature_names[s3.offset_bench_cat + 4] == "bench_cat__domain__000"
    assert s3.feature_names[s3.offset_bench_num] == "bench_num__benchmark_age"
    assert s3.feature_names[s3.offset_bench_num + 1] == "bench_miss__benchmark_age"


def test_v3_block_composes_with_bc_block():
    """Benchmark block sits *after* the bc (u_bc/cross) block."""
    s3 = _schema_v3(k_bc_factors=3)
    legacy_dim = 1 + 4 + 5 + 2 + 2 + 2 + 3 + 2 + 4
    bc_block = 3 + 4 + 3  # u_bc + cross_theta_u_s + cross_theta_u_bc
    assert s3.offset_bench_cat == legacy_dim + bc_block
    assert s3.feature_dim == legacy_dim + bc_block + 7 + 2
    assert s3.to_dict()["schema_version"] == 3


def test_v3_schema_version_reported():
    assert _schema_v3(k_bc_factors=0).to_dict()["schema_version"] == 3
    assert _schema_v3(k_bc_factors=4).to_dict()["schema_version"] == 3


def test_v3_to_dict_from_dict_roundtrips_bench_offsets():
    s = _schema_v3(k_bc_factors=2)
    d = s.to_dict()
    s2 = MemberFeatureSchema.from_dict(d)
    assert s2.n_bench_cat_fields == s.n_bench_cat_fields
    assert s2.n_bench_num_fields == s.n_bench_num_fields
    assert s2.offset_bench_cat == s.offset_bench_cat
    assert s2.offset_bench_num == s.offset_bench_num
    assert s2.bench_cat_field_offsets == s.bench_cat_field_offsets
    assert s2.benchmark_cat_field_cardinalities == s.benchmark_cat_field_cardinalities
    assert s2.benchmark_cat_field_names == s.benchmark_cat_field_names
    assert s2.benchmark_num_field_names == s.benchmark_num_field_names
    assert s2.feature_dim == s.feature_dim
    assert s2.feature_names == s.feature_names
    # Round-trip through JSON to catch non-JSON-friendly types.
    s3 = MemberFeatureSchema.from_dict(json.loads(json.dumps(d)))
    assert s3.feature_names == s.feature_names
    assert s3.benchmark_cat_field_cardinalities == s.benchmark_cat_field_cardinalities


def test_v2_cache_dict_deserializes_into_v3_aware_schema():
    """A serialized v2 schema (no benchmark keys) must still load: the
    new bench fields default to empty / zero."""
    s2 = _schema(k_bc_factors=4)
    d = s2.to_dict()
    # Simulate an OLD cache that predates the v3 keys.
    for k in (
        "n_bench_cat_fields",
        "n_bench_num_fields",
        "offset_bench_cat",
        "offset_bench_num",
        "bench_cat_field_offsets",
        "benchmark_cat_field_cardinalities",
        "benchmark_cat_field_names",
        "benchmark_num_field_names",
    ):
        d.pop(k, None)
    loaded = MemberFeatureSchema.from_dict(d)
    assert loaded.feature_dim == s2.feature_dim
    assert loaded.feature_names == s2.feature_names
    assert loaded.n_bench_cat_fields == 0
    assert loaded.n_bench_num_fields == 0
    assert loaded.offset_bench_cat == 0
    assert loaded.offset_bench_num == 0
    assert loaded.to_dict()["schema_version"] == 2


def test_v3_batch_matches_per_row_builder():
    """Batch builder == per-row builder (< 1e-6) WITH benchmark blocks."""
    s = _schema_v3(k_bc_factors=3)
    tables = _subject_tables(4, s.k_factors)
    n_bc = 6
    bench = _benchmark_tables(n_bc, s.n_bench_cat_fields, s.n_bench_num_fields)
    rng = np.random.default_rng(7)
    N = 8
    subj_idx = rng.integers(0, 4, size=N).astype(np.int64)
    bc_idx = rng.integers(0, n_bc, size=N).astype(np.int64)
    pool = rng.normal(size=(N, 2)).astype(np.float32)
    cd = np.abs(rng.normal(size=(N, 2))).astype(np.float32)
    cl = rng.integers(0, 4, size=N).astype(np.int64)
    nn = rng.normal(size=(N, 2)).astype(np.float32)
    conditions = ["zero-shot", "few-shot", "5-shot", "zero-shot",
                  "few-shot", "5-shot", "zero-shot", "few-shot"]
    u_bc_per_row = rng.normal(size=(N, 3)).astype(np.float32)
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool, centroid_dists=cd,
        cluster_ids=cl, nn_feats=nn,
        conditions=conditions, u_bc_per_row=u_bc_per_row,
        benchmark_tables=bench, bc_idx=bc_idx,
    )
    assert M.shape == (N, s.feature_dim)
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
            benchmark_cat_ids=bench.benchmark_cat_ids[bc_idx[i]],
            benchmark_num=bench.benchmark_num[bc_idx[i]],
        )
        np.testing.assert_allclose(
            M[i], row_one, atol=1e-6,
            err_msg=f"row {i} batch vs per-row mismatch (v3)",
        )


def test_v3_bench_cat_one_hot_and_num_layout():
    """A single row's bench block lights exactly one one-hot per cat
    field and copies the value/missingness pair for the num field."""
    s = _schema_v3(k_bc_factors=0)
    tables = _subject_tables(1, s.k_factors)
    bench = MemberBenchmarkTables(
        benchmark_cat_ids=np.array([[2, 1]], dtype=np.int64),
        benchmark_num=np.array([[0.7, 0.0]], dtype=np.float32),
    )
    M = build_member_features(
        s, tables,
        subject_idx=np.array([0], dtype=np.int64),
        pool_feats=np.zeros((1, 2), dtype=np.float32),
        centroid_dists=np.zeros((1, 2), dtype=np.float32),
        cluster_ids=np.array([0], dtype=np.int64),
        nn_feats=np.zeros((1, 2), dtype=np.float32),
        conditions=["zero-shot"],
        benchmark_tables=bench, bc_idx=np.array([0], dtype=np.int64),
    )
    bcat = M[0, s.offset_bench_cat : s.offset_bench_cat + 7]
    # topic field (card 4, offset 0): id 2 lit.
    assert bcat[2] == 1.0
    assert bcat[0:4].sum() == 1.0
    # domain field (card 3, offset 4): id 1 lit.
    assert bcat[4 + 1] == 1.0
    assert bcat[4:7].sum() == 1.0
    bnum = M[0, s.offset_bench_num : s.offset_bench_num + 2]
    np.testing.assert_allclose(bnum, [0.7, 0.0], atol=1e-6)


def test_v3_bc_redacted_zeroes_bench_blocks_batch():
    """A cold benchmark (bc_redacted) zeroes the entire benchmark block
    (cat one-hot + num) in the batch builder."""
    s = _schema_v3(k_bc_factors=0)
    tables = _subject_tables(2, s.k_factors)
    n_bc = 4
    bench = _benchmark_tables(n_bc, s.n_bench_cat_fields, s.n_bench_num_fields)
    # Make sure bench_num is nonzero so a zero result is meaningful.
    bench.benchmark_num[:] = 1.5
    N = 3
    subj_idx = np.array([0, 1, 0], dtype=np.int64)
    bc_idx = np.array([1, 2, 3], dtype=np.int64)
    redacted = np.array([False, True, False])
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=np.zeros((N, 2), dtype=np.float32),
        centroid_dists=np.zeros((N, 2), dtype=np.float32),
        cluster_ids=np.zeros(N, dtype=np.int64),
        nn_feats=np.zeros((N, 2), dtype=np.float32),
        conditions=["zero-shot", "zero-shot", "few-shot"],
        bc_redacted=redacted,
        benchmark_tables=bench, bc_idx=bc_idx,
    )
    bcat = M[:, s.offset_bench_cat : s.offset_bench_cat + 7]
    bnum = M[:, s.offset_bench_num : s.offset_bench_num + 2]
    # Row 0 (not redacted): one one-hot per cat field (2 fields -> sum 2).
    assert bcat[0].sum() == 2.0
    # Row 1 (redacted): entire benchmark block zeroed.
    assert bcat[1].sum() == 0.0
    np.testing.assert_array_equal(bnum[1], np.zeros(2, dtype=np.float32))
    # Row 2 (not redacted): cat block active, num copied.
    assert bcat[2].sum() == 2.0


def test_v3_bc_redacted_zeroes_bench_blocks_one_row():
    """The per-row builder mirrors batch redaction for the bench block."""
    s = _schema_v3(k_bc_factors=0)
    tables = _subject_tables(1, s.k_factors)
    bench_cat = np.array([2, 1], dtype=np.int64)
    bench_num = np.array([1.5, 0.0], dtype=np.float32)
    out = build_member_features_one(
        s,
        theta_s=0.0,
        u_s=np.zeros(s.k_factors, dtype=np.float32),
        subject_cat_ids=np.zeros(1, dtype=np.int64),
        subject_num=np.zeros(2, dtype=np.float32),
        pool_feats=np.zeros(s.n_pool, dtype=np.float32),
        centroid_dists=np.zeros(s.top_m_centroids, dtype=np.float32),
        cluster_id=0,
        nn_feats=np.zeros(s.n_nn, dtype=np.float32),
        condition="zero-shot",
        bc_redacted=True,
        benchmark_cat_ids=bench_cat,
        benchmark_num=bench_num,
    )
    bcat = out[s.offset_bench_cat : s.offset_bench_cat + 7]
    bnum = out[s.offset_bench_num : s.offset_bench_num + 2]
    assert bcat.sum() == 0.0
    np.testing.assert_array_equal(bnum, np.zeros(2, dtype=np.float32))


def test_v3_one_row_oob_bench_cat_falls_through_to_unk():
    """An out-of-range benchmark cat id maps to UNK=1 (not an OOB write)."""
    s = _schema_v3(k_bc_factors=0)
    out = build_member_features_one(
        s,
        theta_s=0.0,
        u_s=np.zeros(s.k_factors, dtype=np.float32),
        subject_cat_ids=np.zeros(1, dtype=np.int64),
        subject_num=np.zeros(2, dtype=np.float32),
        pool_feats=np.zeros(s.n_pool, dtype=np.float32),
        centroid_dists=np.zeros(s.top_m_centroids, dtype=np.float32),
        cluster_id=0,
        nn_feats=np.zeros(s.n_nn, dtype=np.float32),
        condition="zero-shot",
        # topic field card 4; pass id 99 (oob) -> UNK=1.
        benchmark_cat_ids=np.array([99, 0], dtype=np.int64),
        benchmark_num=np.array([0.0, 1.0], dtype=np.float32),
    )
    # topic field offset 0: UNK column (offset 1) lit.
    assert out[s.offset_bench_cat + 1] == 1.0


def test_v2_path_byte_identical_when_no_bench_fields():
    """Building a v2 schema (no benchmark fields) produces the exact same
    matrix whether or not the v3 code path exists -- the benchmark block
    is a no-op and benchmark_tables/bc_idx are not required."""
    s = _schema(k_bc_factors=4)
    tables = _subject_tables(3, s.k_factors)
    rng = np.random.default_rng(3)
    N = 5
    subj_idx = rng.integers(0, 3, size=N).astype(np.int64)
    pool = rng.normal(size=(N, 2)).astype(np.float32)
    cd = np.abs(rng.normal(size=(N, 2))).astype(np.float32)
    cl = rng.integers(0, 4, size=N).astype(np.int64)
    nn = rng.normal(size=(N, 2)).astype(np.float32)
    conditions = ["zero-shot", "few-shot", "5-shot", "zero-shot", "few-shot"]
    u_bc = rng.normal(size=(N, 4)).astype(np.float32)
    # No benchmark_tables / bc_idx passed -> must work and skip the block.
    M = build_member_features(
        s, tables,
        subject_idx=subj_idx,
        pool_feats=pool, centroid_dists=cd,
        cluster_ids=cl, nn_feats=nn,
        conditions=conditions, u_bc_per_row=u_bc,
    )
    assert M.shape == (N, s.feature_dim)
    # No benchmark columns exist in a v2 schema.
    assert s.offset_bench_cat == 0
    assert s.offset_bench_num == 0
