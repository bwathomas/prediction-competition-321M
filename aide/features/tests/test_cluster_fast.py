"""cluster_fast must equal derive_cluster (the codec oracle) in the production OOF regime.

Geometry is compared with train = all items (the fold=all pass); labels are compared with
the query item held OUT of the train passrate (the per-fold pass), which is exactly when
the codec's per-row self-exclusion becomes a no-op and the vectorized path is exact.
"""
import numpy as np

from aide.features.cluster_fast import cluster_geometry_fast, cluster_labels_fast
from aide.features.derive_cluster import derive_cluster
from aide.features.passrate import CsrPassrate


def _emb(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _fixture():
    # 6 items: {i0,i1,i2} near +x, {i3,i4,i5} near +y
    items = [f"i{j}" for j in range(6)]
    emb = _emb([[1, .05], [1, -.03], [.97, .04], [.04, 1], [-.03, 1], [.02, .98]])
    subs = ["s0", "s1"]
    cen = {"coarse": _emb([[1, 0], [0, 1]]), "fine": _emb([[1, 0], [0, 1]])}
    return items, emb, subs, cen


def test_geometry_matches_codec():
    items, emb, subs, cen = _fixture()
    empty = CsrPassrate.empty(subs, items)
    rids = items
    codec = derive_cluster(query_emb=emb, query_item_keys=items, query_subjects=[""] * 6,
                           row_ids=rids, centroids_by_res=cen, train_emb=emb,
                           train_item_keys=items, passrate=empty)
    fast = cluster_geometry_fast(query_emb=emb, row_ids=rids, centroids_by_res=cen, all_emb=emb)
    for g in ("cluster_geometry", "centroid_distance", "item_cluster"):
        assert fast[g].columns == codec[g].columns
        assert np.allclose(fast[g].X, codec[g].X, atol=1e-5), g


def test_labels_match_codec_in_oof_regime():
    items, emb, subs, cen = _fixture()
    # query i0 is OOF; train = i1..i5. Labels observed only on train items (i0 col = nan).
    L = np.array([[np.nan, 1.0, 0.0, 1.0, 1.0, 0.0],
                  [np.nan, 0.0, 0.0, 1.0, np.nan, 1.0]])
    pr = CsrPassrate.from_dense(subs, items, L)
    train_keys = items[1:]
    train_emb = emb[1:]
    for qi, qs in [("i0", "s0"), ("i0", "s1")]:
        q = emb[[items.index(qi)]]
        codec = derive_cluster(query_emb=q, query_item_keys=[qi], query_subjects=[qs],
                               row_ids=["r"], centroids_by_res=cen, train_emb=train_emb,
                               train_item_keys=train_keys, passrate=pr)
        item_to_cluster = np.argmin(
            ((emb[:, None, :] - cen["fine"][None, :, :]) ** 2).sum(-1), axis=1)
        fast = cluster_labels_fast(query_emb=q, query_subjects=[qs], row_ids=["r"],
                                   centroids_by_res=cen, passrate=pr,
                                   item_to_cluster_fine=item_to_cluster)
        for g in ("cluster_passrate", "cluster_subject"):
            assert fast[g].columns == codec[g].columns
            assert np.allclose(fast[g].X, codec[g].X, atol=1e-5), f"{g} for ({qi},{qs})"


def test_fast_is_vectorized_over_many_rows():
    items, emb, subs, cen = _fixture()
    L = np.array([[np.nan, 1.0, 0.0, 1.0, 1.0, 0.0],
                  [np.nan, 0.0, 0.0, 1.0, 1.0, 1.0]])
    pr = CsrPassrate.from_dense(subs, items, L)
    item_to_cluster = np.argmin(
        ((emb[:, None, :] - cen["fine"][None, :, :]) ** 2).sum(-1), axis=1)
    # 100 query rows reusing i0 across both subjects — must run in one vectorized call
    q = np.repeat(emb[[0]], 100, axis=0)
    subj = (["s0", "s1"] * 50)
    out = cluster_labels_fast(query_emb=q, query_subjects=subj, row_ids=[f"r{i}" for i in range(100)],
                              centroids_by_res=cen, passrate=pr, item_to_cluster_fine=item_to_cluster)
    assert out["cluster_subject"].X.shape == (100, 3)
    assert np.all(np.isfinite(out["cluster_subject"].X))
