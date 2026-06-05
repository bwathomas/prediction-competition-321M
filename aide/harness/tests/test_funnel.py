import numpy as np
import pytest
from aide.harness.funnel import FeatureStore, CacheMissError
from aide.harness.tests._toy import write_fixture_cache


def _populate(root):
    rids = ["r0", "r1", "r2"]
    write_fixture_cache(root, "a", np.array([[1, 2], [3, 4], [5, 6]]), ["ca0", "ca1"], rids)
    write_fixture_cache(root, "b", np.array([[7], [8], [9]]), ["cb0"], rids)
    return rids


def test_load_group_returns_block(tmp_path):
    _populate(tmp_path)
    store = FeatureStore(tmp_path)
    blk = store.load_group("a")
    assert blk.X.shape == (3, 2)
    assert blk.columns == ["ca0", "ca1"]
    assert list(blk.row_ids) == ["r0", "r1", "r2"]


def test_assemble_concatenates_columns_in_order(tmp_path):
    _populate(tmp_path)
    store = FeatureStore(tmp_path)
    X, cols = store.assemble(["a", "b"])
    assert X.shape == (3, 3)
    assert cols == ["ca0", "ca1", "cb0"]
    assert X[0].tolist() == [1.0, 2.0, 7.0]


def test_missing_group_raises_cache_miss_and_creates_no_file(tmp_path):
    _populate(tmp_path)
    store = FeatureStore(tmp_path)
    with pytest.raises(CacheMissError):
        store.load_group("c")
    assert not (tmp_path / "c.npz").exists()  # load-only: no recompute, no write
    assert store.available() == {"a", "b"}


def test_misaligned_row_ids_raise(tmp_path):
    _populate(tmp_path)
    write_fixture_cache(tmp_path, "d", np.array([[0], [0], [0]]), ["cd0"], ["x0", "x1", "x2"])
    store = FeatureStore(tmp_path)
    with pytest.raises(ValueError):
        store.assemble(["a", "d"])
