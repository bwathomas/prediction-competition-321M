"""Tests for src/oof_pipeline.py.

Per-fold builders that touch real ML state (NN index, ModelConfig,
etc.) are tested end-to-end in the notebook's Gate cells. Here we
cover the leakage-safety primitives that we CAN exercise in isolation:

  - slice_train_rows: returns the right rows for train/oof sides
  - reindex_per_item_array: copies global array values into fold ordering
  - split_fold_train_for_early_stopping: item-grouped, no overlap,
    fold-train-only
  - OofPredictionAccumulator: double-write detection, NaN detection,
    coverage-summary correctness
  - make_permuted_labels / entropy_of_label_prior: shape/value invariants
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.oof_folds import ItemFold, make_item_grouped_folds
from src.oof_pipeline import (
    OofPredictionAccumulator,
    build_fold_item_index_map,
    entropy_of_label_prior,
    make_permuted_labels,
    reindex_per_item_array,
    slice_train_rows,
    split_fold_train_for_early_stopping,
)


def _toy_rows(n_items: int, rows_per_item: int) -> list[str]:
    return [f"item_{i:04d}" for i in range(n_items) for _ in range(rows_per_item)]


def _toy_folds(n_items: int = 30, rows_per_item: int = 4, n_folds: int = 3, seed: int = 0):
    rows = _toy_rows(n_items, rows_per_item)
    folds = make_item_grouped_folds(item_keys_per_row=rows, n_folds=n_folds, seed=seed)
    return rows, folds


# ---------------------------------------------------------------------------
# slice_train_rows
# ---------------------------------------------------------------------------


def test_slice_train_rows_returns_right_side():
    rows, folds = _toy_folds()
    df = pd.DataFrame({
        "item_key": rows,
        "label": np.arange(len(rows), dtype=np.float64),
    })
    fold = folds[0]
    train = slice_train_rows(df, fold, side="train")
    oof = slice_train_rows(df, fold, side="oof")
    assert set(train["item_key"]).issubset(set(fold.train_item_keys))
    assert set(oof["item_key"]).issubset(set(fold.oof_item_keys))
    assert len(train) == len(fold.train_row_idx)
    assert len(oof) == len(fold.oof_row_idx)


def test_slice_train_rows_rejects_bad_side():
    rows, folds = _toy_folds()
    df = pd.DataFrame({"item_key": rows, "label": np.zeros(len(rows))})
    with pytest.raises(ValueError, match="side"):
        slice_train_rows(df, folds[0], side="oops")


# ---------------------------------------------------------------------------
# build_fold_item_index_map
# ---------------------------------------------------------------------------


def test_build_fold_item_index_map_is_dense_zero_indexed():
    _, folds = _toy_folds()
    m = build_fold_item_index_map(folds[1])
    assert set(m.values()) == set(range(len(folds[1].train_item_keys)))
    assert all(isinstance(k, str) for k in m)


# ---------------------------------------------------------------------------
# reindex_per_item_array
# ---------------------------------------------------------------------------


def test_reindex_per_item_array_carries_values():
    """Global array of difficulty scores, reindexed to a fold's items,
    should preserve the per-item value at the new positional index."""
    global_keys = [f"item_{i:04d}" for i in range(50)]
    global_arr = np.arange(50, dtype=np.float32) * 0.1  # arbitrary signal
    # Synthetic fold with 5 items chosen from the global list
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("item_0007", "item_0042", "item_0001", "item_0030", "item_0019"),
        oof_item_keys=(),
        train_row_idx=np.array([], dtype=np.int64),
        oof_row_idx=np.array([], dtype=np.int64),
    )
    out = reindex_per_item_array(
        arr=global_arr, train_item_keys_global=global_keys, fold=fold, fill=-1.0
    )
    assert out.shape == (5,)
    assert out.dtype == global_arr.dtype
    np.testing.assert_allclose(
        out,
        [0.1 * 7, 0.1 * 42, 0.1 * 1, 0.1 * 30, 0.1 * 19],
        rtol=1e-6,
    )


def test_reindex_per_item_array_fills_unknown():
    global_keys = ["item_a", "item_b"]
    global_arr = np.array([10, 20], dtype=np.int32)
    fold = ItemFold(
        fold_id=0,
        train_item_keys=("item_a", "item_zzz", "item_b"),
        oof_item_keys=(),
        train_row_idx=np.array([], dtype=np.int64),
        oof_row_idx=np.array([], dtype=np.int64),
    )
    out = reindex_per_item_array(
        arr=global_arr, train_item_keys_global=global_keys, fold=fold, fill=-99
    )
    np.testing.assert_array_equal(out, [10, -99, 20])


# ---------------------------------------------------------------------------
# split_fold_train_for_early_stopping
# ---------------------------------------------------------------------------


def test_es_split_is_item_grouped_and_disjoint():
    rows, folds = _toy_folds(n_items=40, rows_per_item=5, n_folds=3, seed=0)
    rows_arr = np.array(rows, dtype=object)
    fold = folds[0]
    es_tr, es_va = split_fold_train_for_early_stopping(
        fold=fold, item_keys_per_row=rows_arr, es_val_fraction=0.1, seed=0
    )
    # Union covers fold.train_row_idx exactly
    assert sorted(es_tr.tolist() + es_va.tolist()) == sorted(fold.train_row_idx.tolist())
    # Item disjointness: ES-train items and ES-val items don't overlap
    es_tr_items = set(rows_arr[es_tr].tolist())
    es_va_items = set(rows_arr[es_va].tolist())
    assert es_tr_items.isdisjoint(es_va_items)
    # ES-train items + ES-val items partition fold.train_item_keys
    assert (es_tr_items | es_va_items) == set(fold.train_item_keys)


def test_es_split_does_not_touch_oof_rows():
    """The ES val MUST come from fold-train rows only, never from OOF rows.
    Sneaking an OOF row into ES val would early-stop on the prediction
    target -- the exact kind of subtle leakage Gate 1c is meant to catch."""
    rows, folds = _toy_folds(n_items=40, rows_per_item=5, n_folds=3, seed=1)
    rows_arr = np.array(rows, dtype=object)
    fold = folds[1]
    _, es_va = split_fold_train_for_early_stopping(
        fold=fold, item_keys_per_row=rows_arr, es_val_fraction=0.15, seed=42
    )
    assert set(es_va.tolist()).isdisjoint(set(fold.oof_row_idx.tolist()))
    es_va_items = set(rows_arr[es_va].tolist())
    assert es_va_items.isdisjoint(set(fold.oof_item_keys))


def test_es_split_deterministic_for_seed():
    rows, folds = _toy_folds()
    rows_arr = np.array(rows, dtype=object)
    a = split_fold_train_for_early_stopping(
        fold=folds[0], item_keys_per_row=rows_arr, es_val_fraction=0.2, seed=7
    )
    b = split_fold_train_for_early_stopping(
        fold=folds[0], item_keys_per_row=rows_arr, es_val_fraction=0.2, seed=7
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
# OofPredictionAccumulator
# ---------------------------------------------------------------------------


def test_oof_accumulator_normal_flow_works():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N, name="test")
    for f in folds:
        preds = np.full(f.oof_row_idx.size, 0.5, dtype=np.float64) + 1e-3 * f.fold_id
        acc.write_fold(f.oof_row_idx, preds)
    out = acc.finalize()
    assert out.shape == (N,)
    assert np.isfinite(out).all()


def test_oof_accumulator_catches_double_write():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N, name="test")
    acc.write_fold(folds[0].oof_row_idx, np.zeros(folds[0].oof_row_idx.size))
    with pytest.raises(RuntimeError, match="overwrite"):
        # Re-writing fold 0's rows triggers the double-write trap.
        acc.write_fold(folds[0].oof_row_idx, np.zeros(folds[0].oof_row_idx.size))


def test_oof_accumulator_catches_missing_fold():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N, name="test")
    # Only write fold 0 + 1 -- fold 2 missing.
    acc.write_fold(folds[0].oof_row_idx, np.zeros(folds[0].oof_row_idx.size))
    acc.write_fold(folds[1].oof_row_idx, np.zeros(folds[1].oof_row_idx.size))
    with pytest.raises(RuntimeError, match="never received"):
        acc.finalize()


def test_oof_accumulator_catches_nan_pred():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N, name="test")
    for f in folds:
        preds = np.zeros(f.oof_row_idx.size, dtype=np.float64)
        preds[0] = np.nan
        acc.write_fold(f.oof_row_idx, preds)
    with pytest.raises(RuntimeError, match="non-finite"):
        acc.finalize()


def test_oof_accumulator_coverage_summary():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N)
    acc.write_fold(folds[0].oof_row_idx, np.zeros(folds[0].oof_row_idx.size))
    s = acc.coverage_summary()
    assert s["n_written"] == folds[0].oof_row_idx.size
    assert s["n_unwritten"] == N - folds[0].oof_row_idx.size
    assert s["n_double_written"] == 0


def test_oof_accumulator_rejects_shape_mismatch():
    rows, folds = _toy_folds()
    N = len(rows)
    acc = OofPredictionAccumulator(N)
    with pytest.raises(ValueError, match="shape mismatch"):
        acc.write_fold(folds[0].oof_row_idx, np.zeros(7))


# ---------------------------------------------------------------------------
# make_permuted_labels / entropy_of_label_prior
# ---------------------------------------------------------------------------


def test_permuted_labels_preserves_distribution():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.62).astype(np.float64)
    y_perm = make_permuted_labels(y=y, seed=42)
    assert y_perm.shape == y.shape
    np.testing.assert_allclose(y_perm.mean(), y.mean(), rtol=1e-12)
    # Not the same arrangement (vanishingly unlikely for N=10k).
    assert not np.array_equal(y_perm, y)


def test_permuted_labels_deterministic_for_seed():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.float64)
    a = make_permuted_labels(y=y, seed=1)
    b = make_permuted_labels(y=y, seed=1)
    np.testing.assert_array_equal(a, b)


def test_entropy_of_label_prior_matches_formula():
    # H(p=0.5) = ln(2) ~ 0.6931
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    assert abs(entropy_of_label_prior(y) - np.log(2.0)) < 1e-9
    # H(p=0.7)
    p = 0.7
    expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    y2 = np.concatenate([np.ones(700), np.zeros(300)])
    assert abs(entropy_of_label_prior(y2) - expected) < 1e-9


def test_entropy_of_label_prior_handles_edge_cases():
    """H(0)=H(1)=0; our clipped formulation gives a tiny positive
    value but should not explode to inf."""
    y = np.ones(10, dtype=np.float64)
    h = entropy_of_label_prior(y)
    assert np.isfinite(h) and h >= 0.0 and h < 1e-9
