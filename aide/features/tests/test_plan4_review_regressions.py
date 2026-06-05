"""Regression tests for the Plan-4 fresh-context code review (2026-06-04/05).

One test per finding, each constructed to FAIL against the pre-fix code — so a future
refactor that reintroduces the leak/collision trips here. See
quality_reports/cross_artifact_aide_plan4/review.md.
"""
import numpy as np
import pytest

from aide.features.cache import FeatureCache
from aide.features.derive_cluster import derive_cluster
from aide.features.derive_nn import DensePassrate, bruteforce_knn, derive_nn
from aide.features.derive_tabular import derive_tabular, target_encode_oof
from aide.features.store import FoldFeatureStore
from aide.harness.funnel import FeatureBlock


def _emb(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


# === CRITICAL 1: FAISS -1 padding must not wrap to index_keys[-1] ====================
def test_faiss_minus_one_padding_does_not_leak_last_item():
    items = ["i0", "i1", "i2"]
    emb = _emb([[1, 0], [0, 1], [-1, 0]])
    # poison the LAST item's label — the one a -1 would wrap to via index_keys[-1]
    L = np.array([[0.0, 0.0, 999.0]])
    pr = DensePassrate(["s0"], items, L)

    def padding_knn(index_emb, query_emb, k):
        idx, sim = bruteforce_knn(index_emb, query_emb, k)
        idx = idx.copy(); sim = sim.copy()
        idx[:, -1] = -1; sim[:, -1] = -1.0        # simulate an IVF empty-probe pad
        return idx, sim

    out = derive_nn(query_emb=_emb([[1, 0]]), query_item_keys=["i0"], query_subjects=["s0"],
                    row_ids=["s0|i0"], index_emb=emb, index_item_keys=items, passrate=pr,
                    Ks=(1, 2), knn_fn=padding_knn)
    assert np.all(np.abs(out["nn_label_derivatives"].X) < 50.0)  # 999 never gathered


# === CRITICAL 2: embedding-alias under a different key must be self-excluded =========
def test_nn_alias_under_different_key_is_excluded():
    items = ["i0", "i1", "i9"]      # i9 is an exact embedding-alias of query i0
    emb = _emb([[1, 0], [0, 1], [1, 0]])
    L = np.array([[0.0, 0.0, 999.0]])   # poison the alias's label
    pr = DensePassrate(["s0"], items, L)
    out = derive_nn(query_emb=_emb([[1, 0]]), query_item_keys=["i0"], query_subjects=["s0"],
                    row_ids=["s0|i0"], index_emb=emb, index_item_keys=items, passrate=pr,
                    Ks=(1, 2), knn_fn=bruteforce_knn)
    assert np.all(np.abs(out["nn_label_derivatives"].X) < 50.0)


def test_cluster_alias_under_different_key_is_excluded():
    items = ["i0", "i1", "i9"]
    emb = _emb([[1, 0.01], [1, -0.01], [1, 0.01]])   # i9 aliases i0; all one cluster
    L = np.array([[0.0, 0.0, 999.0]])
    pr = DensePassrate(["s0"], items, L)
    cen = {"coarse": _emb([[1, 0], [0, 1]]), "fine": _emb([[1, 0], [0, 1]])}
    out = derive_cluster(query_emb=_emb([[1, 0.01]]), query_item_keys=["i0"],
                         query_subjects=["s0"], row_ids=["s0|i0"], centroids_by_res=cen,
                         train_emb=emb, train_item_keys=items, passrate=pr)
    assert np.all(np.abs(out["cluster_passrate"].X) < 50.0)
    assert np.all(np.abs(out["cluster_subject"].X) < 50.0)


# === MAJOR 3: under-retrieval below maxK without a full scan must RAISE ==============
def test_under_retrieval_raises_when_index_not_fully_scanned():
    m = 10
    emb = _emb(np.eye(m)[:, :4] + 0.1)
    items = [f"i{j}" for j in range(m)]
    pr = DensePassrate(["s0"], items, np.zeros((1, m)))

    def starved_knn(index_emb, query_emb, k):
        idx, sim = bruteforce_knn(index_emb, query_emb, k)
        idx = idx.copy(); sim = sim.copy()
        idx[:, 2:] = -1; sim[:, 2:] = -1.0       # only 2 real neighbours survive
        return idx, sim

    with pytest.raises(ValueError):
        derive_nn(query_emb=_emb([emb[0]]), query_item_keys=["i0"], query_subjects=["s0"],
                  row_ids=["r"], index_emb=emb, index_item_keys=items, passrate=pr,
                  Ks=(4,), knn_fn=starved_knn, search_buffer=0)  # n_request=5 < m=10


def test_small_index_clips_without_raising():
    items = ["i0", "i1", "i2"]      # only 3 items, maxK=4 → genuinely small, no raise
    emb = _emb([[1, 0], [0, 1], [-1, 0]])
    pr = DensePassrate(["s0"], items, np.zeros((1, 3)))
    out = derive_nn(query_emb=_emb([[1, 0]]), query_item_keys=["i0"], query_subjects=["s0"],
                    row_ids=["r"], index_emb=emb, index_item_keys=items, passrate=pr,
                    Ks=(4,), knn_fn=bruteforce_knn)
    assert out["nn_label_derivatives"].X.shape[0] == 1


# === MAJOR 4: cache key collisions and unsafe path components ========================
def test_code_version_slug_collision_resolved_by_hash(tmp_path):
    a = FeatureCache(tmp_path, code_version="a/b")
    b = FeatureCache(tmp_path, code_version="a-b")   # slugs identically to "a/b"
    pa = a.shard_path(a.key("m", "g", fold=0, seed=0, n_folds=3))
    pb = b.shard_path(b.key("m", "g", fold=0, seed=0, n_folds=3))
    assert pa != pb   # distinct code versions ⇒ distinct shards (no stale-serve)


def test_unsafe_family_or_group_raises(tmp_path):
    c = FeatureCache(tmp_path)
    with pytest.raises(ValueError):
        c.shard_path(c.key("m/escape", "g", fold=0, seed=0, n_folds=3))
    with pytest.raises(ValueError):
        c.shard_path(c.key("m", "g/../etc", fold=0, seed=0, n_folds=3))


# === MAJOR 5: std distinguishes unseen from zero-variance ============================
def test_std_unseen_key_uses_global_std_not_zero():
    keys = np.array(["a", "b"])
    y = np.array([1.0, 0.0])
    folds = np.array([0, 1])
    std = target_encode_oof(keys, y, folds, m=0.0, stat="std")
    # 'a' is unseen in other folds → global std of {1,0} = 0.5, NOT 0.0
    assert abs(std[0] - 0.5) < 1e-9


# === MINOR 6: derive_cluster train/oof disjointness guard ===========================
def test_cluster_train_oof_overlap_raises():
    items = ["i0", "i1"]
    emb = _emb([[1, 0], [0, 1]])
    pr = DensePassrate(["s0"], items, np.zeros((1, 2)))
    cen = {"coarse": _emb([[1, 0]]), "fine": _emb([[1, 0]])}
    with pytest.raises(AssertionError):
        derive_cluster(query_emb=_emb([[1, 0]]), query_item_keys=["i0"], query_subjects=["s0"],
                       row_ids=["r"], centroids_by_res=cen, train_emb=emb,
                       train_item_keys=items, passrate=pr,
                       oof_item_keys=["i0"])   # i0 in BOTH train and oof


# === MINOR 8: coverage probe runs by DEFAULT in store.assemble ======================
def test_store_coverage_probe_runs_without_explicit_flag(tmp_path):
    s = FoldFeatureStore(FeatureCache(tmp_path, code_version="v1"),
                         embedding_family="m", seed=0, n_folds=3)
    blk = FeatureBlock(X=np.zeros((1, 1), np.float32), columns=["totally_unknown_col"],
                       row_ids=np.array(["r0"]))
    s.write_group("item_pool", fold="all", block=blk, inputs_hash="h")
    with pytest.raises(AssertionError):
        s.assemble(["item_pool"], fold=0, row_ids=["r0"])   # no check_coverage= passed


# === MINOR 9: partial interaction parents raise rather than vanish ==================
def test_partial_interaction_parents_raise():
    n = 2
    with pytest.raises(ValueError):
        derive_tabular(row_ids=["r0", "r1"], fold_ids=np.array([0, 1]),
                       y=np.array([1.0, 0.0]), subject_keys=np.array(["s0", "s1"]),
                       subject_meta={}, benchmark_meta={},
                       parents={"subject_mean": np.zeros(n)})   # missing cluster_difficulty
