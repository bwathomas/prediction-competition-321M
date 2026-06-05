import numpy as np
import pytest
from aide.hygiene.dropout import apply_proxy_dropout
from aide.hygiene.probes import (
    assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak,
    assert_columns_covered)


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
    assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])  # no raise (all rows dropped)


def test_assert_no_proxy_leak_raises_when_a_proxy_survives():
    cols = ["subject_key", "meta:family", "benchmark"]
    X = np.array([[0.0, 0.7, 1.0]], dtype=np.float32)  # meta:family survived
    with pytest.raises(AssertionError):
        assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])


def test_probe_wired_to_dropout_passes_under_partial_rate():
    # C1 regression: the probe must NOT fire on legitimate partial-rate dropout when
    # given the dropped-row mask; it checks only the rows that were supposed to be masked.
    cols = ["subject_key", "meta:family", "benchmark", "condition", "item_emb__0"]
    X = np.ones((6, len(cols)), dtype=np.float32)
    subjects = ["s1", "s2", "s3", "s1", "s2", "s3"]
    benchmarks = ["b1", "b1", "b1", "b2", "b2", "b2"]
    rng = np.random.default_rng(11)
    Xd, info = apply_proxy_dropout(X, cols, subjects=subjects, benchmarks=benchmarks,
                                   rng=rng, subject_rate=0.5, benchmark_rate=0.0)
    # row-aware check on exactly the dropped rows: must pass
    assert_no_proxy_leak(Xd, cols, ["subject"], rows=info["drop_rows"]["subject"])
    # and if any subject was actually dropped, the naive all-rows check would (correctly)
    # fail because non-dropped subjects still carry their proxies — proving C1 was real.
    if info["dropped_subjects"] and len(info["dropped_subjects"]) < 3:
        with pytest.raises(AssertionError):
            assert_no_proxy_leak(Xd, cols, ["subject"])  # rows=None => all rows


def test_assert_columns_covered_passes_when_all_classified():
    cols = ["subject_key", "meta:family", "benchmark", "condition",
            "feat:nn_passrate__mean", "item_emb__0", "item_content"]
    assert_columns_covered(cols, neutral_prefixes=["item_emb", "item_content"])  # no raise


def test_assert_columns_covered_raises_on_unclassified_column():
    cols = ["subject_key", "benchmark", "feat:mystery__0"]  # mystery not classified
    with pytest.raises(AssertionError):
        assert_columns_covered(cols, neutral_prefixes=["item_emb"])
