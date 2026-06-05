"""Tests for the fold-aware feature store: fold routing, load-only assembly, coverage.

The store is the single writer (``write_group``) and a load-only reader (``assemble``)
that routes fold-invariant (pure-geometry/content) groups to the ``fold="all"`` shard and
label-derived groups to the requested outer fold — so OOF discipline is enforced by the
key, not by the caller remembering which fold to ask for.
"""
import numpy as np
import pytest

from aide.feature_catalog import group_names
from aide.features.cache import FeatureCache
from aide.features.store import FOLD_INVARIANT_GROUPS, LABEL_DERIVED_GROUPS, FoldFeatureStore
from aide.harness.funnel import CacheMissError, FeatureBlock


def _store(tmp_path):
    cache = FeatureCache(tmp_path, code_version="v1")
    return FoldFeatureStore(cache, embedding_family="m", seed=0, n_folds=3)


def _blk(cols, row_ids, fill=1.0):
    X = np.full((len(row_ids), len(cols)), fill, dtype=np.float32)
    return FeatureBlock(X=X, columns=list(cols), row_ids=np.asarray(row_ids).astype(str))


def test_classification_covers_every_catalog_group():
    """Every catalog group is classified exactly once as fold-invariant XOR label-derived
    (the coverage discipline: an unclassified group would silently get the wrong fold)."""
    names = set(group_names())
    assert FOLD_INVARIANT_GROUPS.isdisjoint(LABEL_DERIVED_GROUPS)
    assert FOLD_INVARIANT_GROUPS | LABEL_DERIVED_GROUPS == names


def test_fold_invariant_group_reads_fold_all(tmp_path):
    s = _store(tmp_path)
    rids = ["r0", "r1"]
    # write a fold-invariant group (item_pool) once at fold="all"
    s.write_group("item_pool", fold="all", block=_blk(["pool__a"], rids), inputs_hash="h")
    # assembling at ANY outer fold resolves to the same fold=all shard
    X0, _ = s.assemble(["item_pool"], fold=0, row_ids=rids)
    X2, _ = s.assemble(["item_pool"], fold=2, row_ids=rids)
    assert np.array_equal(X0, X2)


def test_label_derived_group_is_fold_keyed(tmp_path):
    s = _store(tmp_path)
    rids = ["r0", "r1"]
    s.write_group("nn_passrate", fold=0, block=_blk(["nn__x"], rids, fill=0.0), inputs_hash="h")
    s.write_group("nn_passrate", fold=1, block=_blk(["nn__x"], rids, fill=1.0), inputs_hash="h")
    X0, _ = s.assemble(["nn_passrate"], fold=0, row_ids=rids)
    X1, _ = s.assemble(["nn_passrate"], fold=1, row_ids=rids)
    assert X0[0, 0] == 0.0 and X1[0, 0] == 1.0   # distinct per-fold shards


def test_missing_fold_shard_raises_cache_miss(tmp_path):
    s = _store(tmp_path)
    rids = ["r0", "r1"]
    s.write_group("nn_passrate", fold=0, block=_blk(["nn__x"], rids), inputs_hash="h")
    with pytest.raises(CacheMissError):
        s.assemble(["nn_passrate"], fold=2, row_ids=rids)   # fold 2 never derived


def test_assemble_concatenates_and_aligns(tmp_path):
    s = _store(tmp_path)
    rids = ["r0", "r1"]
    s.write_group("item_pool", fold="all", block=_blk(["pool__a", "pool__b"], rids), inputs_hash="h")
    s.write_group("nn_passrate", fold=0, block=_blk(["nn__x"], rids, fill=2.0), inputs_hash="h")
    X, cols = s.assemble(["item_pool", "nn_passrate"], fold=0, row_ids=rids)
    assert X.shape == (2, 3)
    assert cols == ["pool__a", "pool__b", "nn__x"]
    assert X[0].tolist() == [1.0, 1.0, 2.0]


def test_assemble_detects_row_misalignment(tmp_path):
    s = _store(tmp_path)
    s.write_group("item_pool", fold="all", block=_blk(["pool__a"], ["r0", "r1"]), inputs_hash="h")
    s.write_group("nn_passrate", fold=0, block=_blk(["nn__x"], ["x0", "x1"]), inputs_hash="h")
    with pytest.raises(ValueError):
        s.assemble(["item_pool", "nn_passrate"], fold=0, row_ids=["r0", "r1"])


def test_coverage_probe_rejects_unclassified_column(tmp_path):
    s = _store(tmp_path)
    rids = ["r0"]
    # a stray column that is neither a known proxy nor neutral must trip the probe
    s.write_group("item_pool", fold="all", block=_blk(["wat_unknown_col"], rids), inputs_hash="h")
    with pytest.raises(AssertionError):
        s.assemble(["item_pool"], fold=0, row_ids=rids, check_coverage=True)


def test_end_to_end_derive_write_assemble_routes_folds(tmp_path):
    """Integration: derive_nn → store.write_blocks → store.assemble. nn_geometry is
    fold-invariant (one fold=all shard reused at every fold); the label groups are
    fold-keyed. Exercises the whole write→route→load seam in one go."""
    from aide.features.derive_nn import DensePassrate, bruteforce_knn, derive_nn

    emb = (lambda a: a / np.linalg.norm(a, axis=1, keepdims=True))(
        np.array([[1, 0.1], [1, -0.1], [0, 1], [-1, 0.2]], dtype=np.float32))
    items = ["i0", "i1", "i2", "i3"]
    pr = DensePassrate(["s0"], items, np.array([[1.0, 0.0, 1.0, 0.0]]))
    q_items, rids = ["i0", "i2"], ["s0|i0", "s0|i2"]
    q_emb = emb[[0, 2]]
    blocks = derive_nn(query_emb=q_emb, query_item_keys=q_items, query_subjects=["s0", "s0"],
                       row_ids=rids, index_emb=emb, index_item_keys=items, passrate=pr,
                       Ks=(1, 2), knn_fn=bruteforce_knn)

    s = _store(tmp_path)
    s.write_blocks(blocks, fold=0, inputs_hash="h")
    # nn_geometry written at fold=all ⇒ resolvable from any fold; label group only at fold 0
    Xg_f0, _ = s.assemble(["nn_geometry"], fold=0, row_ids=rids)
    Xg_f2, _ = s.assemble(["nn_geometry"], fold=2, row_ids=rids)
    assert np.array_equal(Xg_f0, Xg_f2)
    X, cols = s.assemble(["nn_geometry", "nn_label_derivatives", "counts_subject"],
                         fold=0, row_ids=rids)
    assert X.shape[0] == 2 and len(cols) == X.shape[1]
    with pytest.raises(CacheMissError):
        s.assemble(["nn_label_derivatives"], fold=1, row_ids=rids)  # never derived at fold 1


def test_assemble_is_load_only_no_writes(tmp_path):
    s = _store(tmp_path)
    rids = ["r0"]
    s.write_group("item_pool", fold="all", block=_blk(["pool__a"], rids), inputs_hash="h")
    before = sorted(p.name for p in tmp_path.rglob("*"))
    s.assemble(["item_pool"], fold=0, row_ids=rids)
    after = sorted(p.name for p in tmp_path.rglob("*"))
    assert before == after   # reading derived nothing
