"""Leakage tripwires. Any harness reporting an NLL should call these first; a failure
aborts the score rather than silently leaking. Adapted from src/oof_folds.py invariants.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import PROXY_TREE, all_masked_columns, _matches


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


def assert_no_proxy_leak(X, feature_columns, dropped_nodes, *, rows=None, atol: float = 0.0) -> None:
    """Assert every proxy column of a dropped identity node is zeroed.

    Matches the dropout contract (C1 fix): dropout only zeros proxy columns for the
    rows of dropped entities, so the probe must be told which rows to check. Pass
    ``rows`` = the boolean mask from ``apply_proxy_dropout(...)[1]["drop_rows"][node]``.
    ``rows=None`` checks ALL rows (correct only when every row was dropped).
    """
    X = np.asarray(X)
    cols = list(feature_columns)
    masked = all_masked_columns(dropped_nodes, cols)
    row_mask = None if rows is None else np.asarray(rows, dtype=bool)
    for c in masked:
        j = cols.index(c)
        col = X[:, j] if row_mask is None else X[row_mask, j]
        if col.size and np.any(np.abs(col) > atol):
            raise AssertionError(
                f"proxy leak: column {c!r} survived dropout of {dropped_nodes}")


def assert_columns_covered(feature_columns, *, neutral_prefixes) -> None:
    """Invert the default to 'unlisted ⇒ blocked' (M3).

    Fail loudly on any feature column that is neither a known identity proxy (covered by
    some PROXY_TREE node) nor identity-neutral (matches a caller-supplied neutral prefix,
    e.g. item-side features on the cold-start axis). Forces new feature columns to be
    explicitly classified before they can silently leak through dropout.
    """
    cols = list(feature_columns)
    covered = set()
    for root in PROXY_TREE:
        covered.update(all_masked_columns([root], cols))
    unclassified = [
        c for c in cols
        if c not in covered and not any(_matches(c, p) for p in neutral_prefixes)
    ]
    if unclassified:
        raise AssertionError(
            "unclassified feature columns (neither a known identity proxy nor neutral): "
            f"{sorted(unclassified)}")
