"""Contract + OOF tests for the clustering codec.

Centroids are passed in (the heavy MiniBatchKMeans *fit* is a Colab/injected step), so
these lock the numpy assignment/soft-responsibility/aggregation core: the neutral
geometry blocks are label-independent, the one-hot is a partition, and the label-derived
cluster difficulty / subject-gap blocks are OOF (never read the query item's own label).
"""
import numpy as np

from aide.features.derive_cluster import derive_cluster, fit_multi_kmeans
from aide.features.derive_nn import DensePassrate


def _emb(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _toy():
    items = ["i0", "i1", "i2", "i3"]
    # two tight groups: {i0,i1} near +x, {i2,i3} near +y
    emb = _emb([[1, 0.05], [1, -0.05], [0.05, 1], [-0.05, 1]])
    subjects = ["s0", "s1"]
    L = np.array([[1.0, 0.0, 1.0, 1.0],
                  [0.0, 1.0, np.nan, 0.0]])
    pr = DensePassrate(subjects, items, L)
    centroids = {"coarse": _emb([[1, 0], [0, 1]]),
                 "fine": _emb([[1, 0.05], [1, -0.05], [0, 1]])}
    return items, emb, subjects, pr, centroids


def _call(items, emb, subjects, pr, centroids, *, q_items, q_subj, row_ids):
    q_emb = _emb([emb[items.index(it)] for it in q_items])
    return derive_cluster(query_emb=q_emb, query_item_keys=q_items, query_subjects=q_subj,
                          row_ids=row_ids, centroids_by_res=centroids,
                          train_emb=emb, train_item_keys=items, passrate=pr)


def test_blocks_and_columns():
    items, emb, subjects, pr, cen = _toy()
    out = _call(items, emb, subjects, pr, cen,
                q_items=["i0", "i2"], q_subj=["s0", "s1"], row_ids=["a", "b"])
    assert set(out) == {"cluster_geometry", "centroid_distance", "item_cluster",
                        "cluster_passrate", "cluster_subject"}
    assert all(c.startswith(("clu__", "clu_id__")) for c in out["cluster_geometry"].columns)
    assert all(c.startswith("cd__") for c in out["centroid_distance"].columns)
    assert all(c.startswith(("cluster__", "cluster_id")) for c in out["item_cluster"].columns)
    assert all(c.startswith("m2_cluster") for c in out["cluster_passrate"].columns)
    assert all(c.startswith("clu_subj__") for c in out["cluster_subject"].columns)
    # one cd__ column per fine centroid
    assert sum(c.startswith("cd__") for c in out["centroid_distance"].columns) == 3


def test_onehot_is_a_partition():
    items, emb, subjects, pr, cen = _toy()
    out = _call(items, emb, subjects, pr, cen,
                q_items=["i0", "i2"], q_subj=["s0", "s1"], row_ids=["a", "b"])
    blk = out["item_cluster"]
    oh = blk.X[:, [i for i, c in enumerate(blk.columns) if c.startswith("cluster__")]]
    assert np.allclose(oh.sum(axis=1), 1.0)         # exactly one fine cluster per row
    assert set(np.unique(oh)) <= {0.0, 1.0}


def test_soft_responsibility_is_ordered_and_normalized():
    items, emb, subjects, pr, cen = _toy()
    out = _call(items, emb, subjects, pr, cen,
                q_items=["i0"], q_subj=["s0"], row_ids=["a"])
    cols = out["cluster_geometry"].columns
    x = out["cluster_geometry"].X[0]
    t1 = x[cols.index("clu__soft_responsibility_top1")]
    t2 = x[cols.index("clu__soft_responsibility_top2")]
    t3 = x[cols.index("clu__soft_responsibility_top3")]
    assert t1 >= t2 >= t3 >= 0.0
    assert t1 <= 1.0 + 1e-6


def test_neutral_geometry_is_label_independent():
    items, emb, subjects, pr, cen = _toy()
    kw = dict(q_items=["i0", "i2"], q_subj=["s0", "s1"], row_ids=["a", "b"])
    g1 = _call(items, emb, subjects, pr, cen, **kw)
    pr2 = DensePassrate(subjects, items, np.zeros((2, 4)))
    g2 = _call(items, emb, subjects, pr2, cen, **kw)
    for grp in ("cluster_geometry", "centroid_distance", "item_cluster"):
        assert np.allclose(g1[grp].X, g2[grp].X), f"{grp} leaked label signal"


def test_cluster_difficulty_and_subject_gap_are_oof():
    """Poison the query item's own label; OOF difficulty/subject-gap must exclude it."""
    items, emb, subjects, pr, cen = _toy()
    poisoned = pr.L.copy()
    poisoned[:, items.index("i0")] = 999.0
    pr_p = DensePassrate(subjects, items, poisoned)
    out = derive_cluster(
        query_emb=_emb([emb[items.index("i0")]]), query_item_keys=["i0"],
        query_subjects=["s0"], row_ids=["s0|i0"], centroids_by_res=cen,
        train_emb=emb, train_item_keys=items, passrate=pr_p)
    assert np.all(np.abs(out["cluster_passrate"].X) < 50.0), "own label leaked into m2_cluster"
    assert np.all(np.abs(out["cluster_subject"].X) < 50.0), "own label leaked into clu_subj__"


def test_subject_gap_separates_ability_from_cluster_difficulty():
    """One fine cluster holds two train items; subject s0 outperforms the pooled mean, so
    clu_subj__subject_minus_cluster_gap must be strictly positive (the fidelity fix:
    cluster difficulty pools all subjects, the subject channel is one subject)."""
    items = ["i0", "i1", "i2"]
    emb = _emb([[1, 0.02], [1, -0.02], [0, 1]])     # i0,i1 share a cluster; i2 elsewhere
    subjects = ["s0", "s1"]
    # on the cluster {i0,i1}: s0 passes both (ability 1.0); s1 fails both ⇒ pooled 0.5
    L = np.array([[1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    pr = DensePassrate(subjects, items, L)
    centroids = {"coarse": _emb([[1, 0], [0, 1]]),
                 "fine": _emb([[1, 0.0], [0, 1]])}   # fine cluster 0 = {i0,i1}
    out = derive_cluster(query_emb=_emb([emb[0]]), query_item_keys=["i0"],
                         query_subjects=["s0"], row_ids=["s0|i0"], centroids_by_res=centroids,
                         train_emb=emb, train_item_keys=items, passrate=pr)
    cols = out["cluster_subject"].columns
    gap = out["cluster_subject"].X[0, cols.index("clu_subj__subject_minus_cluster_gap")]
    diff = out["cluster_passrate"].X[0, 0]
    # query item i0 excluded ⇒ cluster pools {i1}: s0=1,s1=0 ⇒ difficulty 0.5; s0 mean=1.0
    assert diff == 0.5
    assert gap > 0.0


def test_fit_multi_kmeans_smoke():
    items, emb, *_ = _toy()
    cen = fit_multi_kmeans(emb, {"coarse": 2, "fine": 3}, seed=0)
    assert cen["coarse"].shape == (2, emb.shape[1])
    assert cen["fine"].shape == (3, emb.shape[1])
