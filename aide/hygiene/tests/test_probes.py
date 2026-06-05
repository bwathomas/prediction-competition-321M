import numpy as np
import pytest
from aide.hygiene.probes import (
    assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak)


def test_assert_item_disjoint_passes_when_disjoint():
    assert_item_disjoint(["a", "b"], ["c", "d"])  # no raise


def test_assert_item_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        assert_item_disjoint(["a", "b", "c"], ["c", "d"])


def test_assert_row_uniform_safe_passes_when_item_stays_in_one_fold():
    rows = ["a", "a", "b"]
    fold_ids = np.array([2, 2, 0])
    assert_row_uniform_safe(rows, fold_ids)  # no raise


def test_assert_row_uniform_safe_raises_when_item_split_across_folds():
    rows = ["a", "a", "b"]
    fold_ids = np.array([2, 1, 0])  # 'a' lands in both fold 2 and fold 1
    with pytest.raises(AssertionError):
        assert_row_uniform_safe(rows, fold_ids)


def test_assert_no_proxy_leak_passes_when_dropped_columns_are_zero():
    cols = ["subject_key", "meta:family", "benchmark"]
    X = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])  # no raise


def test_assert_no_proxy_leak_raises_when_a_proxy_survives():
    cols = ["subject_key", "meta:family", "benchmark"]
    X = np.array([[0.0, 0.7, 1.0]], dtype=np.float32)  # meta:family survived
    with pytest.raises(AssertionError):
        assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])
