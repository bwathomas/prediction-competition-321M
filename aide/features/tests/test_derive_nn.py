"""Contract + leakage tests for the kNN feature codec.

The heavy kNN *search* (FAISS on Colab) is injected here as an exact numpy brute force,
so these lock the parts that must be right regardless of the search backend: the OOF
guard (a query item never reads its OWN label; only index/train items contribute), the
neutral-geometry block's independence from labels, and the column/shape contract.
"""
import numpy as np

from aide.features.derive_nn import DensePassrate, bruteforce_knn, derive_nn


def _emb(vectors):
    a = np.asarray(vectors, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _toy():
    # 4 items on a ring; neighbours by cosine are the angular neighbours.
    items = ["i0", "i1", "i2", "i3"]
    emb = _emb([[1, 0], [0.9, 0.4], [0, 1], [-0.9, 0.4]])
    subjects = ["s0", "s1"]
    # L[subject, item] pass labels; nan = unobserved
    L = np.array([[1.0, 1.0, 0.0, 0.0],
                  [0.0, np.nan, 1.0, 1.0]])
    pr = DensePassrate(subjects, items, L)
    return items, emb, subjects, pr


def test_bruteforce_knn_matches_cosine_order():
    items, emb, *_ = _toy()
    idx, sim = bruteforce_knn(emb, emb[:1], k=4)
    # nearest to i0 is itself, then i1
    assert idx[0, 0] == 0 and idx[0, 1] == 1
    assert sim[0, 0] >= sim[0, 1] >= sim[0, 2]


def test_columns_and_shapes(tmp_path):
    items, emb, subjects, pr = _toy()
    # 3 query rows (subject,item) pairs
    q_items = ["i0", "i2", "i1"]
    q_subj = ["s0", "s1", "s0"]
    q_emb = _emb([emb[items.index(it)] for it in q_items])
    out = derive_nn(query_emb=q_emb, query_item_keys=q_items, query_subjects=q_subj,
                    row_ids=["s0|i0", "s1|i2", "s0|i1"],
                    index_emb=emb, index_item_keys=items, passrate=pr,
                    Ks=(1, 2), knn_fn=bruteforce_knn)
    assert set(out) == {"nn_label_derivatives", "nn_geometry", "counts_subject"}
    for blk in out.values():
        assert blk.X.shape[0] == 3
        assert list(blk.row_ids) == ["s0|i0", "s1|i2", "s0|i1"]
    assert all(c.startswith("nn__") for c in out["nn_label_derivatives"].columns)
    assert all(c.startswith("geo__") for c in out["nn_geometry"].columns)
    assert all(c.startswith("cnt__") for c in out["counts_subject"].columns)
    assert any("passrate_mean_K2" in c for c in out["nn_label_derivatives"].columns)


def test_oof_query_item_own_label_never_used(tmp_path):
    """Self-leak tripwire: poison the query item's OWN label column; if the codec ever
    reads label[subject, query_item] the poison shows up. Index INCLUDES the query item
    (the fold='all' geometry case) so only item-key self-exclusion can save us."""
    items, emb, subjects, pr = _toy()
    poisoned = pr.L.copy()
    # set every subject's label on i1 to a huge sentinel
    poisoned[:, items.index("i1")] = 999.0
    pr_poison = DensePassrate(subjects, items, poisoned)
    q_emb = _emb([emb[items.index("i1")]])
    out = derive_nn(query_emb=q_emb, query_item_keys=["i1"], query_subjects=["s0"],
                    row_ids=["s0|i1"], index_emb=emb, index_item_keys=items,
                    passrate=pr_poison, Ks=(1, 2), knn_fn=bruteforce_knn)
    X = out["nn_label_derivatives"].X
    assert np.all(np.abs(X) < 50.0), "query item's own (poisoned) label leaked into nn__*"


def test_geometry_block_is_label_independent(tmp_path):
    items, emb, subjects, pr = _toy()
    q_items, q_subj = ["i0", "i2"], ["s0", "s1"]
    q_emb = _emb([emb[items.index(it)] for it in q_items])
    common = dict(query_emb=q_emb, query_item_keys=q_items, query_subjects=q_subj,
                  row_ids=["a", "b"], index_emb=emb, index_item_keys=items,
                  Ks=(1, 2), knn_fn=bruteforce_knn)
    g1 = derive_nn(passrate=pr, **common)["nn_geometry"].X
    # totally different labels
    pr2 = DensePassrate(subjects, items, np.zeros_like(pr.L))
    g2 = derive_nn(passrate=pr2, **common)["nn_geometry"].X
    assert np.allclose(g1, g2), "geo__* depends on labels — it must be pure geometry (neutral)"


def test_coverage_counts_only_observed_neighbors(tmp_path):
    items, emb, subjects, pr = _toy()
    # subject s1 has nan on i1: a neighbour set including i1 must not count it as covered
    q_emb = _emb([emb[items.index("i0")]])
    out = derive_nn(query_emb=q_emb, query_item_keys=["i0"], query_subjects=["s1"],
                    row_ids=["s1|i0"], index_emb=emb, index_item_keys=items,
                    passrate=pr, Ks=(1, 2), knn_fn=bruteforce_knn)
    cnt_cols = out["counts_subject"].columns
    cnt = out["counts_subject"].X[0]
    support = cnt[cnt_cols.index("cnt__neighbor_subject_support")]
    # i0's nearest non-self neighbour is i1 (unobserved for s1) then i3 (observed=1)
    # so within K=2 exactly one of the two neighbours is observed
    assert support == 1.0
