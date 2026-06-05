"""Contract tests for the derive-once feature-shard cache.

These exercise the backend-agnostic key/path/atomicity/manifest surface with the
numpy-native NpzBackend, so the leakage- and correctness-critical cache contract is
fully covered locally. The heavy parquet container is a separate Colab-exercised backend.
"""
import json

import numpy as np
import pytest

from aide.features.cache import FeatureCache, NpzBackend, content_hash
from aide.harness.funnel import CacheMissError, FeatureBlock


def _block(n=3, cols=("c0", "c1"), tag=0.0):
    X = (np.arange(n * len(cols), dtype=np.float32) + tag).reshape(n, len(cols))
    return FeatureBlock(X=X, columns=list(cols),
                        row_ids=np.array([f"r{i}" for i in range(n)]))


def _cache(root):
    return FeatureCache(root, backend=NpzBackend(), code_version="v1")


def test_shard_path_encodes_full_key(tmp_path):
    c = _cache(tmp_path)
    k = c.key("llama", "nn_passrate", fold=2, seed=0, n_folds=3)
    p = c.shard_path(k)
    assert p.parent == tmp_path / "llama" / "nn_passrate"
    stem = p.name
    assert "fold2" in stem and "seed0" in stem and "nf3" in stem and "v1" in stem
    assert p.suffix == ".npz"
    assert c.meta_path(k).name == p.name[:-len(".npz")] + ".meta.json"


def test_fold_all_and_inner_fold_distinct_paths(tmp_path):
    c = _cache(tmp_path)
    p_all = c.shard_path(c.key("q", "geo", fold="all", seed=0, n_folds=3))
    p_f1 = c.shard_path(c.key("q", "geo", fold=1, seed=0, n_folds=3))
    p_inner = c.shard_path(c.key("q", "geo", fold=1, seed=0, n_folds=3, inner_fold=0))
    paths = {p_all, p_f1, p_inner}
    assert len(paths) == 3  # all-vs-fold1-vs-inner are distinct shards
    assert "foldall" in p_all.name
    assert "inner0" in p_inner.name


def test_write_then_read_roundtrip(tmp_path):
    c = _cache(tmp_path)
    k = c.key("m", "item_pool", fold="all", seed=0, n_folds=3)
    blk = _block()
    assert c.write_shard(k, blk, inputs_hash="h0") == "written"
    out = c.read_shard(k)
    assert np.allclose(out.X, blk.X)
    assert out.columns == blk.columns
    assert list(out.row_ids) == list(blk.row_ids)


def test_idempotent_skip_and_overwrite(tmp_path):
    c = _cache(tmp_path)
    k = c.key("m", "g", fold=0, seed=0, n_folds=3)
    assert c.write_shard(k, _block(tag=0.0), inputs_hash="h0") == "written"
    # second write with DIFFERENT content but same key is a derive-once skip
    assert c.write_shard(k, _block(tag=100.0), inputs_hash="h1") == "skipped"
    assert c.read_shard(k).X[0, 0] == 0.0  # original preserved
    # explicit overwrite replaces
    assert c.write_shard(k, _block(tag=100.0), inputs_hash="h1", overwrite=True) == "written"
    assert c.read_shard(k).X[0, 0] == 100.0


def test_meta_records_key_and_inputs_hash(tmp_path):
    c = _cache(tmp_path)
    k = c.key("m", "g", fold=2, seed=7, n_folds=5)
    c.write_shard(k, _block(cols=("a", "b")), inputs_hash="deadbeef")
    meta = json.loads(c.meta_path(k).read_text())
    assert meta["inputs_hash"] == "deadbeef"
    assert meta["columns"] == ["a", "b"]
    assert meta["n_rows"] == 3 and meta["n_cols"] == 2
    assert meta["code_version"] == "v1"
    assert meta["embedding_family"] == "m" and meta["feature_group"] == "g"
    assert meta["outer_fold"] == 2 and meta["split_seed"] == 7 and meta["n_folds"] == 5


class _FailingBackend(NpzBackend):
    def write(self, path, block):  # noqa: D401 - test double
        raise RuntimeError("simulated write failure")


def test_atomic_write_leaves_no_partial_or_tmp(tmp_path):
    c = FeatureCache(tmp_path, backend=_FailingBackend(), code_version="v1")
    k = c.key("m", "g", fold=0, seed=0, n_folds=3)
    with pytest.raises(RuntimeError):
        c.write_shard(k, _block(), inputs_hash="h")
    assert not c.shard_path(k).exists()          # no half-written final shard
    assert not c.meta_path(k).exists()
    leftover = list(c.shard_path(k).parent.glob(".tmp*"))
    assert leftover == []                         # tmp cleaned up on failure


def test_index_lists_all_shards(tmp_path):
    c = _cache(tmp_path)
    k1 = c.key("m", "g1", fold=0, seed=0, n_folds=3)
    k2 = c.key("m", "g2", fold="all", seed=0, n_folds=3)
    c.write_shard(k1, _block(), inputs_hash="h1")
    c.write_shard(k2, _block(cols=("x",)), inputs_hash="h2")
    idx = c.load_index()
    assert len(idx) == 2
    groups = {rec["feature_group"] for rec in idx.values()}
    assert groups == {"g1", "g2"}
    # rebuild from disk reproduces the same manifest
    c.index_path.unlink()
    c.rebuild_index()
    assert {rec["feature_group"] for rec in c.load_index().values()} == {"g1", "g2"}


def test_read_missing_raises_cache_miss_naming_key(tmp_path):
    c = _cache(tmp_path)
    k = c.key("m", "nn_passrate", fold=1, seed=0, n_folds=3)
    with pytest.raises(CacheMissError) as ei:
        c.read_shard(k)
    msg = str(ei.value)
    assert "nn_passrate" in msg and "fold" in msg


def test_block_validation_rejects_inconsistent_block(tmp_path):
    c = _cache(tmp_path)
    k = c.key("m", "g", fold=0, seed=0, n_folds=3)
    bad_cols = FeatureBlock(X=np.zeros((3, 2), np.float32), columns=["only_one"],
                            row_ids=np.array(["r0", "r1", "r2"]))
    with pytest.raises(ValueError):
        c.write_shard(k, bad_cols, inputs_hash="h")
    dup = FeatureBlock(X=np.zeros((3, 2), np.float32), columns=["c", "c"],
                       row_ids=np.array(["r0", "r1", "r2"]))
    with pytest.raises(ValueError):
        c.write_shard(k, dup, inputs_hash="h")
    bad_rows = FeatureBlock(X=np.zeros((3, 1), np.float32), columns=["c"],
                            row_ids=np.array(["r0", "r1"]))
    with pytest.raises(ValueError):
        c.write_shard(k, bad_rows, inputs_hash="h")


def test_content_hash_is_stable_and_order_sensitive():
    assert content_hash("a", "b") == content_hash("a", "b")
    assert content_hash("a", "b") != content_hash("b", "a")
    assert content_hash(b"\x00\x01", "tag") == content_hash(b"\x00\x01", "tag")
