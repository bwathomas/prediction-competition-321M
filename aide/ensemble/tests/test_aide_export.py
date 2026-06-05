"""Test the item-disjoint split that guards the AIDE export's cold-start hygiene."""
import numpy as np
import pytest

from aide.ensemble.aide_export import item_fold_split
from aide.hygiene.manifest import build_manifest


def test_item_fold_split_is_item_disjoint():
    items = [f"i{j}" for j in range(12)]
    man = build_manifest(items, n_folds=3, seed=0)
    # rows reuse items across subjects (the real shape: many rows per item)
    row_items = np.array([f"i{j % 12}" for j in range(60)])
    train, holdout = item_fold_split(row_items, man, holdout_fold=0)
    assert train.sum() + holdout.sum() == 60
    tr_items = set(row_items[train]); ho_items = set(row_items[holdout])
    assert not (tr_items & ho_items)                       # no item in both
    # holdout is exactly the rows whose item is in fold 0
    assert all(man.fold_of(it) == 0 for it in ho_items)
    assert all(man.fold_of(it) != 0 for it in tr_items)


def test_item_fold_split_detects_leak_if_manifest_inconsistent():
    items = ["i0", "i1"]
    man = build_manifest(items, n_folds=2, seed=0)
    # a row whose item isn't in the manifest assignment -> KeyError surfaces (no silent pass)
    with pytest.raises((KeyError, AssertionError)):
        item_fold_split(np.array(["i0", "iX"]), man, holdout_fold=0)
