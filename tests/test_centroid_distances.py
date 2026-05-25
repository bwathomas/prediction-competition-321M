"""Unit tests for the centroid-distance feature pipeline.

Covers:
- ``src.clustering.compute_top_m_distances``: shape, ordering,
  parity with :func:`assign_clusters` for the top-1 case, and
  ground-truth equivalence to a brute-force argpartition.
- ``src.item_features.build_centroid_distance_features``: schema,
  missing-key fallback, integration with the pool feature
  z-score / matrix-build path.
- Edge cases that previously bit the export pipeline:
  ``top_m == n_centroids`` (no padding) and ``top_m > n_centroids``
  (raises).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    assign_clusters,
    compute_top_m_distances,
)
from src.item_features import (
    POOL_FEATURE_NAMES,
    apply_zscore,
    build_centroid_distance_features,
    build_feature_matrix,
    centroid_distance_feature_names,
    fit_zscore_stats,
    merge_pool_and_centroid_features,
)


# ---------------------------------------------------------------------------
# compute_top_m_distances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,d,k,m,seed",
    [
        (200, 32, 16, 4, 0),
        (50, 8, 8, 8, 1),       # m == k: no padding path
        (1, 16, 12, 3, 2),      # tiny single-row
        (1024, 128, 64, 8, 7),  # larger / closer to real workloads
    ],
)
def test_top_m_distances_brute_force_parity(n, d, k, m, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    C = rng.standard_normal((k, d)).astype(np.float32)

    ids, dists = compute_top_m_distances(C, X, top_m=m)

    assert ids.shape == (n, m)
    assert dists.shape == (n, m)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    # Ids are 1-based (0 reserved for UNK).
    assert int(ids.min()) >= 1
    assert int(ids.max()) <= k
    # Distances are non-negative and sorted ascending per row.
    assert (dists >= 0).all()
    diffs = np.diff(dists, axis=1)
    assert (diffs >= -1e-4).all()

    # Ground truth from explicit pairwise squared L2.
    gt_d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    gt_part = np.argpartition(gt_d2, kth=m - 1, axis=1)[:, :m]
    gt_part_sorted = np.argsort(
        np.take_along_axis(gt_d2, gt_part, axis=1), axis=1
    )
    gt_ids = np.take_along_axis(gt_part, gt_part_sorted, axis=1) + 1
    gt_dists = np.sort(np.take_along_axis(gt_d2, gt_part, axis=1), axis=1)

    np.testing.assert_array_equal(ids, gt_ids)
    np.testing.assert_allclose(dists, gt_dists, rtol=1e-3, atol=1e-3)


def test_top_m_top1_matches_assign_clusters():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 64)).astype(np.float32)
    C = rng.standard_normal((32, 64)).astype(np.float32)

    ids, _ = compute_top_m_distances(C, X, top_m=4)
    single = assign_clusters(C, X)
    np.testing.assert_array_equal(ids[:, 0], single)


def test_top_m_raises_when_m_exceeds_k():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 8)).astype(np.float32)
    C = rng.standard_normal((4, 8)).astype(np.float32)

    with pytest.raises(ValueError):
        compute_top_m_distances(C, X, top_m=5)


def test_top_m_validates_ndim():
    C = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_top_m_distances(C, np.zeros(8, dtype=np.float32), top_m=2)


def test_top_m_validates_dim_match():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 8)).astype(np.float32)
    C = rng.standard_normal((4, 16)).astype(np.float32)

    with pytest.raises(ValueError):
        compute_top_m_distances(C, X, top_m=2)


def test_top_m_zero_raises():
    C = np.zeros((4, 8), dtype=np.float32)
    X = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_top_m_distances(C, X, top_m=0)


# ---------------------------------------------------------------------------
# build_centroid_distance_features
# ---------------------------------------------------------------------------


def test_build_centroid_distance_features_schema():
    rng = np.random.default_rng(0)
    N, D, K, M = 32, 16, 8, 3
    keys = [f"k_{i}" for i in range(N)]
    emb_lookup = {k: rng.standard_normal(D).astype(np.float32) for k in keys}
    centroids = rng.standard_normal((K, D)).astype(np.float32)

    df = build_centroid_distance_features(keys, emb_lookup, centroids, top_m=M)
    cols = centroid_distance_feature_names(M)
    assert list(df.columns) == ["item_key"] + cols
    arr = df[cols].to_numpy()
    assert arr.shape == (N, M)
    # Every row's distances are sorted ascending and non-negative.
    diffs = np.diff(arr, axis=1)
    assert (diffs >= -1e-5).all()
    assert (arr >= 0).all()


def test_build_centroid_distance_features_missing_key_yields_nan():
    rng = np.random.default_rng(0)
    D, K, M = 8, 4, 3
    centroids = rng.standard_normal((K, D)).astype(np.float32)
    emb_lookup = {"a": rng.standard_normal(D).astype(np.float32)}

    df = build_centroid_distance_features(
        ["a", "missing"], emb_lookup, centroids, top_m=M
    )
    cols = centroid_distance_feature_names(M)
    # Row 0 is real, row 1 is NaN.
    assert np.isfinite(df.loc[0, cols].astype(np.float32)).all()
    assert np.isnan(df.loc[1, cols].astype(np.float32)).all()


def test_build_centroid_distance_features_empty_lookup_returns_all_nan():
    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((4, 8)).astype(np.float32)
    df = build_centroid_distance_features(
        ["a", "b"], {}, centroids, top_m=3
    )
    cols = centroid_distance_feature_names(3)
    arr = df[cols].to_numpy(dtype=np.float32)
    assert arr.shape == (2, 3)
    assert np.isnan(arr).all()


def test_build_centroid_distance_features_dim_mismatch_raises():
    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((4, 8)).astype(np.float32)
    emb_lookup = {"a": rng.standard_normal(16).astype(np.float32)}
    with pytest.raises(ValueError):
        build_centroid_distance_features(["a"], emb_lookup, centroids, top_m=2)


# ---------------------------------------------------------------------------
# Pool feature integration: merge + z-score + matrix build
# ---------------------------------------------------------------------------


def test_pool_centroid_merge_and_zscore_roundtrip():
    rng = np.random.default_rng(0)
    N, D, K, M = 40, 12, 6, 4
    keys = [f"i_{i}" for i in range(N)]
    emb_lookup = {k: rng.standard_normal(D).astype(np.float32) for k in keys}
    centroids = rng.standard_normal((K, D)).astype(np.float32)

    pool_df = pd.DataFrame({"item_key": keys})
    for c in POOL_FEATURE_NAMES:
        pool_df[c] = rng.standard_normal(N).astype(np.float32)
    centroid_df = build_centroid_distance_features(
        keys, emb_lookup, centroids, top_m=M
    )
    merged, cols = merge_pool_and_centroid_features(pool_df, centroid_df)
    expected_cols = list(POOL_FEATURE_NAMES) + centroid_distance_feature_names(M)
    assert cols == expected_cols
    assert merged.shape == (N, 1 + len(expected_cols))

    stats = fit_zscore_stats(merged, feature_cols=expected_cols)
    z = apply_zscore(merged, stats)
    mat = build_feature_matrix(keys, z, feature_cols=expected_cols)

    # Matrix has the right shape and is fully finite (NaNs in centroid
    # cols would survive z-score; build_feature_matrix replaces those
    # with 0.0 -- no NaN should escape).
    assert mat.shape == (N, len(expected_cols))
    assert np.isfinite(mat).all()


def test_merge_returns_pool_only_when_centroid_df_is_none():
    pool_df = pd.DataFrame({
        "item_key": ["a", "b"],
        **{c: [0.0, 1.0] for c in POOL_FEATURE_NAMES},
    })
    merged, cols = merge_pool_and_centroid_features(pool_df, None)
    assert cols == list(POOL_FEATURE_NAMES)
    pd.testing.assert_frame_equal(merged, pool_df)


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------


def test_centroid_distance_feature_names_canonical_order():
    assert centroid_distance_feature_names(1) == ["centroid_dist_0"]
    assert centroid_distance_feature_names(3) == [
        "centroid_dist_0", "centroid_dist_1", "centroid_dist_2",
    ]
    with pytest.raises(ValueError):
        centroid_distance_feature_names(0)
