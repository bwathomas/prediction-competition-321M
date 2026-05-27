"""Tests for src/oof_folds.py.

These tests cover the FOUNDATIONAL OOF invariants. Notebook-level
red-team gates (1c shuffled-label control, 1d optimism check on the
real pipeline) live in the notebook because they need the full
training stack -- but the helpers underneath them are tested here.

Specifically:
  - make_item_grouped_folds: shape, determinism, exact partition
  - assert_item_disjoint: catches a hand-crafted leaky fold
  - assert_row_idx_partition: catches a hand-crafted row-mismatched fold
  - assert_nn_neighbors_in_fold_train: catches a hand-crafted leaky NN
  - report_train_vs_val_optimism: flags only when gap exceeds threshold
  - fold_cache_suffix: stable and discriminating
"""
from __future__ import annotations

import numpy as np
import pytest

from src.oof_folds import (
    ItemFold,
    assert_item_disjoint,
    assert_nn_neighbors_in_fold_train,
    assert_row_idx_partition,
    fold_cache_suffix,
    make_item_grouped_folds,
    report_train_vs_val_optimism,
)


# ---------------------------------------------------------------------------
# make_item_grouped_folds
# ---------------------------------------------------------------------------


def _toy_rows(n_items: int, rows_per_item: int) -> list[str]:
    """Build a row->item_key array with `n_items` unique items each
    appearing `rows_per_item` times. Deterministic."""
    out: list[str] = []
    for i in range(int(n_items)):
        out.extend([f"item_{i:04d}"] * int(rows_per_item))
    return out


def test_folds_partition_unique_items_exactly_once():
    rows = _toy_rows(n_items=50, rows_per_item=3)
    folds = make_item_grouped_folds(item_keys_per_row=rows, n_folds=5, seed=0)
    assert len(folds) == 5

    union_oof = set()
    for f in folds:
        union_oof |= set(f.oof_item_keys)
        # Within a fold, train and oof item sets must be disjoint.
        assert set(f.oof_item_keys).isdisjoint(set(f.train_item_keys))
    assert union_oof == set(rows)  # exact partition of the universe


def test_folds_row_partition_invariant():
    """Every row appears in exactly one fold's OOF set."""
    rows = _toy_rows(n_items=33, rows_per_item=7)
    N = len(rows)
    folds = make_item_grouped_folds(item_keys_per_row=rows, n_folds=4, seed=11)
    assert_row_idx_partition(folds, n_rows=N)


def test_folds_are_deterministic_given_seed():
    rows = _toy_rows(n_items=40, rows_per_item=4)
    a = make_item_grouped_folds(item_keys_per_row=rows, n_folds=4, seed=42)
    b = make_item_grouped_folds(item_keys_per_row=rows, n_folds=4, seed=42)
    for fa, fb in zip(a, b):
        assert fa.train_item_keys == fb.train_item_keys
        assert fa.oof_item_keys == fb.oof_item_keys
        np.testing.assert_array_equal(fa.train_row_idx, fb.train_row_idx)
        np.testing.assert_array_equal(fa.oof_row_idx, fb.oof_row_idx)


def test_folds_change_with_different_seed():
    rows = _toy_rows(n_items=40, rows_per_item=4)
    a = make_item_grouped_folds(item_keys_per_row=rows, n_folds=4, seed=0)
    b = make_item_grouped_folds(item_keys_per_row=rows, n_folds=4, seed=1)
    # Not strictly required, but for non-pathological seeds the partitions
    # should differ on at least one fold.
    assert any(set(fa.oof_item_keys) != set(fb.oof_item_keys) for fa, fb in zip(a, b))


def test_folds_fold_sizes_are_balanced():
    rows = _toy_rows(n_items=100, rows_per_item=1)
    folds = make_item_grouped_folds(item_keys_per_row=rows, n_folds=7, seed=0)
    sizes = [len(f.oof_item_keys) for f in folds]
    assert max(sizes) - min(sizes) <= 1


def test_folds_reject_too_few_items():
    rows = _toy_rows(n_items=3, rows_per_item=10)
    with pytest.raises(ValueError, match="n_unique_items"):
        make_item_grouped_folds(item_keys_per_row=rows, n_folds=5, seed=0)


def test_folds_reject_bad_n_folds():
    rows = _toy_rows(n_items=10, rows_per_item=2)
    with pytest.raises(ValueError, match="n_folds"):
        make_item_grouped_folds(item_keys_per_row=rows, n_folds=1, seed=0)


# ---------------------------------------------------------------------------
# Gate 1a: item-disjointness
# ---------------------------------------------------------------------------


def test_gate1a_passes_on_clean_folds():
    rows = _toy_rows(n_items=30, rows_per_item=3)
    for f in make_item_grouped_folds(item_keys_per_row=rows, n_folds=3, seed=0):
        assert_item_disjoint(f)  # must not raise


def test_gate1a_catches_leaky_fold():
    """Hand-craft a fold with an item in BOTH train and OOF."""
    leaky = ItemFold(
        fold_id=0,
        train_item_keys=("a", "b", "c"),
        oof_item_keys=("c", "d"),  # 'c' is on both sides!
        train_row_idx=np.array([0, 1, 2], dtype=np.int64),
        oof_row_idx=np.array([3, 4], dtype=np.int64),
    )
    with pytest.raises(AssertionError, match="GATE 1a"):
        assert_item_disjoint(leaky)


def test_gate1a_row_partition_catches_missing_row():
    """Hand-craft folds where one row appears in 0 folds' OOF set."""
    folds = (
        ItemFold(0, ("a",), ("b",),
                 np.array([0], dtype=np.int64),
                 np.array([1], dtype=np.int64)),
        ItemFold(1, ("b",), ("a",),
                 np.array([1], dtype=np.int64),
                 np.array([0], dtype=np.int64)),
    )
    # 2 rows, both in OOF exactly once -> passes.
    assert_row_idx_partition(folds, n_rows=2)
    # Bump n_rows to 3 -> row 2 appears in zero folds.
    with pytest.raises(AssertionError, match="GATE 1a"):
        assert_row_idx_partition(folds, n_rows=3)


def test_gate1a_row_partition_catches_double_oof_row():
    """Hand-craft folds where one row appears in TWO folds' OOF set."""
    folds = (
        ItemFold(0, ("b",), ("a",),
                 np.array([1], dtype=np.int64),
                 np.array([0], dtype=np.int64)),
        ItemFold(1, ("b",), ("a",),
                 np.array([1], dtype=np.int64),
                 np.array([0], dtype=np.int64)),
    )
    with pytest.raises(AssertionError, match="GATE 1a"):
        assert_row_idx_partition(folds, n_rows=2)


# ---------------------------------------------------------------------------
# Gate 1b: NN-neighbor-in-fold-train
# ---------------------------------------------------------------------------


def test_gate1b_passes_when_neighbors_are_all_in_fold_train():
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("train_a", "train_b", "train_c"),
        oof_item_keys=("oof_d", "oof_e"),
        train_row_idx=np.array([0, 1], dtype=np.int64),
        oof_row_idx=np.array([2, 3], dtype=np.int64),
    )
    # 2 OOF rows, each with 3 neighbors, all in train items.
    neighbor_keys = np.array(
        [
            ["train_a", "train_b", "train_c"],
            ["train_c", "train_a", "train_b"],
        ],
        dtype=object,
    )
    result = assert_nn_neighbors_in_fold_train(
        fold=fold, oof_row_neighbor_item_keys=neighbor_keys
    )
    assert result["n_violations"] == 0
    assert result["n_checked"] == 2


def test_gate1b_catches_leaky_neighbor():
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("train_a", "train_b"),
        oof_item_keys=("oof_x", "oof_y"),
        train_row_idx=np.array([0, 1], dtype=np.int64),
        oof_row_idx=np.array([2, 3], dtype=np.int64),
    )
    # Row 0's 3rd neighbor is 'oof_x' -- which is one of the fold's OWN
    # OOF items. This is the canonical leakage pattern.
    neighbor_keys = np.array(
        [
            ["train_a", "train_b", "oof_x"],   # leak!
            ["train_a", "train_b", "train_a"],
        ],
        dtype=object,
    )
    with pytest.raises(AssertionError, match="GATE 1b"):
        assert_nn_neighbors_in_fold_train(
            fold=fold, oof_row_neighbor_item_keys=neighbor_keys
        )


def test_gate1b_ignores_sentinels():
    """``-1`` / empty-string neighbor slots are no-neighbor placeholders
    and shouldn't trigger a violation."""
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("a", "b"),
        oof_item_keys=("c",),
        train_row_idx=np.array([0], dtype=np.int64),
        oof_row_idx=np.array([1], dtype=np.int64),
    )
    neighbor_keys = np.array(
        [["a", "-1", ""]],
        dtype=object,
    )
    result = assert_nn_neighbors_in_fold_train(
        fold=fold, oof_row_neighbor_item_keys=neighbor_keys
    )
    assert result["n_violations"] == 0


def test_gate1b_sampling_respects_size():
    """sample_size controls how many OOF rows we actually scan."""
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("a",),
        oof_item_keys=("b",),
        train_row_idx=np.array([0], dtype=np.int64),
        oof_row_idx=np.arange(100, dtype=np.int64),
    )
    neighbor_keys = np.array([["a"]] * 100, dtype=object)
    result = assert_nn_neighbors_in_fold_train(
        fold=fold, oof_row_neighbor_item_keys=neighbor_keys, sample_size=10, seed=0
    )
    assert result["n_checked"] == 10


# ---------------------------------------------------------------------------
# report_train_vs_val_optimism (Gate 1d helper)
# ---------------------------------------------------------------------------


def test_optimism_check_does_not_flag_small_gap():
    out = report_train_vs_val_optimism(
        train_loss=0.45, val_loss=0.47, threshold_nats=0.03,
    )
    assert out["flag"] is False
    assert abs(out["gap"] - 0.02) < 1e-9


def test_optimism_check_flags_large_gap():
    out = report_train_vs_val_optimism(
        train_loss=0.30, val_loss=0.45, threshold_nats=0.03,
    )
    assert out["flag"] is True
    assert out["gap"] > 0.03


# ---------------------------------------------------------------------------
# fold_cache_suffix
# ---------------------------------------------------------------------------


def test_fold_cache_suffix_is_stable():
    a = fold_cache_suffix(fold_id=0, train_item_keys=["x", "y", "z"])
    b = fold_cache_suffix(fold_id=0, train_item_keys=["z", "x", "y"])
    assert a == b  # sorted internally; order-invariant
    assert len(a) == 16


def test_fold_cache_suffix_differs_for_different_folds():
    a = fold_cache_suffix(fold_id=0, train_item_keys=["x", "y", "z"])
    b = fold_cache_suffix(fold_id=1, train_item_keys=["x", "y", "z"])
    assert a != b


def test_fold_cache_suffix_differs_for_different_items():
    a = fold_cache_suffix(fold_id=0, train_item_keys=["x", "y", "z"])
    b = fold_cache_suffix(fold_id=0, train_item_keys=["x", "y", "w"])
    assert a != b
