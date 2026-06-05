import numpy as np
from aide.hygiene.manifest import build_manifest
from aide.hygiene.splits import outer_folds, inner_folds, row_fold_ids


def test_outer_folds_are_item_disjoint_and_cover_all_items():
    m = build_manifest([f"item{i}" for i in range(30)], n_folds=3, seed=0)
    folds = outer_folds(m)
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.train_item_keys).isdisjoint(fold.oof_item_keys)
    oof_union = set()
    for fold in folds:
        oof_union |= set(fold.oof_item_keys)
    assert oof_union == set(m.assignment)


def test_row_fold_ids_keep_all_rows_of_an_item_together():
    m = build_manifest(["a", "b"], n_folds=3, seed=0)
    rows = ["a", "a", "a", "b", "b"]  # item a appears 3x, b 2x
    ids = row_fold_ids(rows, m)
    assert ids[0] == ids[1] == ids[2]  # all 'a' rows share a fold
    assert ids[3] == ids[4]            # all 'b' rows share a fold


def test_inner_folds_nest_inside_an_outer_train_set_without_touching_oof():
    m = build_manifest([f"item{i}" for i in range(30)], n_folds=3, seed=0)
    outer = outer_folds(m)
    o0 = outer[0]
    inner = inner_folds(o0.train_item_keys, n_folds=3, seed=m.seed, outer_index=o0.index)
    inner_items = set()
    for fold in inner:
        inner_items |= set(fold.oof_item_keys)
        assert set(fold.train_item_keys).isdisjoint(fold.oof_item_keys)
    assert inner_items == set(o0.train_item_keys)
    assert inner_items.isdisjoint(o0.oof_item_keys)  # recursion leakage guard


def test_inner_folds_are_deterministic_per_outer_index():
    train = [f"item{i}" for i in range(20)]
    a = inner_folds(train, n_folds=3, seed=0, outer_index=1)
    b = inner_folds(train, n_folds=3, seed=0, outer_index=1)
    assert [f.oof_item_keys for f in a] == [f.oof_item_keys for f in b]
    c = inner_folds(train, n_folds=3, seed=0, outer_index=2)
    assert [f.oof_item_keys for f in a] != [f.oof_item_keys for f in c]
