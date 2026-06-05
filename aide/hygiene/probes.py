"""Leakage tripwires. Any harness reporting an NLL should call these first; a failure
aborts the score rather than silently leaking. Adapted from src/oof_folds.py invariants.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import all_masked_columns


def assert_item_disjoint(train_item_keys, oof_item_keys) -> None:
    overlap = set(str(k) for k in train_item_keys) & set(str(k) for k in oof_item_keys)
    if overlap:
        raise AssertionError(f"item leakage: {len(overlap)} key(s) in both train and oof")


def assert_row_uniform_safe(item_keys_per_row, row_fold_ids) -> None:
    seen = {}
    for k, f in zip((str(x) for x in item_keys_per_row), np.asarray(row_fold_ids).tolist()):
        if k in seen and seen[k] != f:
            raise AssertionError(f"item {k!r} split across folds {seen[k]} and {f}")
        seen[k] = f


def assert_no_proxy_leak(X, feature_columns, dropped_nodes, *, atol: float = 0.0) -> None:
    """Assert every proxy column of a dropped identity node is fully zeroed in X."""
    X = np.asarray(X)
    cols = list(feature_columns)
    masked = all_masked_columns(dropped_nodes, cols)
    for c in masked:
        j = cols.index(c)
        if np.any(np.abs(X[:, j]) > atol):
            raise AssertionError(f"proxy leak: column {c!r} survived dropout of {dropped_nodes}")
