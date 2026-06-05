"""Assemble the per-row training matrix from the derive-once feature cache.

The cache stores LABEL groups partitioned by OOF fold (each fold shard holds the rows whose
item is held out in that fold) and GEOMETRY groups once per item (fold=all). This rebuilds
the single ``Dataset`` the ensemble harness consumes:

  * for each outer fold, reconstruct that fold's rows from ``labels_df`` (the same
    item-in-OOF filter + order the derivation used), assemble its label groups, assert the
    shard row order matches, and gather each row's item geometry;
  * concatenate folds → all rows, with y / item_keys / subjects / benchmarks aligned.

Every row's features were derived leaving its own item out, so feeding them to the harness
(which re-folds by item) stays OOF-honest. ``feature_columns`` are catalog-classified, so
the harness's proxy dropout + coverage probe work unchanged.
"""
from __future__ import annotations

import numpy as np

from aide.harness.eval import Dataset


def _row_ids(subjects, items):
    return np.array([f"{s}|{i}" for s, i in zip(subjects, items)])


def assemble_training_matrix(store, manifest, *, item_keys, subject_keys, labels, benchmarks,
                             geometry_groups, label_groups, progress=None):
    """Build a ``Dataset`` over ALL rows from the cache. The label arrays are the FULL
    measurement rows (same order the derivation saw); geometry is gathered per item, labels
    concatenated across folds. Caller passes ``labels_df`` columns as numpy arrays
    (pandas-free, so the assembler is unit-testable)."""
    from aide.hygiene.splits import outer_folds

    item_keys_all = np.asarray(item_keys).astype(str)
    subj_all = np.asarray(subject_keys).astype(str)
    bench_all = np.asarray(benchmarks).astype(str)
    y_all = np.asarray(labels, dtype=float)

    # geometry: one row per item (fold=all) → item_key -> row index
    Xg, gcols = store.assemble(geometry_groups, fold=0, check_coverage=False)  # fold=all routing
    g_block = store.cache.read_shard(store._key(geometry_groups[0], "all"))
    g_index = {str(k): i for i, k in enumerate(g_block.row_ids)}

    Xs, ys, items, subs, bens = [], [], [], [], []
    cols = None
    for fold in outer_folds(manifest):
        oof = set(fold.oof_item_keys)
        m = np.array([k in oof for k in item_keys_all])
        if not m.any():
            continue
        f_items, f_subj = item_keys_all[m], subj_all[m]
        Xl, lcols = store.assemble(label_groups, fold=fold.index, check_coverage=False)
        # alignment: the label shard's rows must equal this fold's reconstructed rows
        shard_rids = store.cache.read_shard(store._key(label_groups[0], fold.index)).row_ids
        if not np.array_equal(np.asarray(shard_rids).astype(str), _row_ids(f_subj, f_items)):
            raise ValueError(f"fold {fold.index}: label shard rows != reconstructed labels_df rows")
        gi = np.array([g_index[k] for k in f_items])     # item geometry per row
        Xf = np.concatenate([Xg[gi], Xl], axis=1).astype(np.float32)
        Xs.append(Xf); ys.append(y_all[m]); items.append(f_items)
        subs.append(f_subj); bens.append(bench_all[m])
        cols = list(gcols) + list(lcols)
        if progress:
            progress(f"assemble fold{fold.index}: {m.sum()} rows", fold=fold.index)
    X = np.concatenate(Xs, axis=0)
    return Dataset(X=X, feature_columns=cols, y=np.concatenate(ys),
                   item_keys=np.concatenate(items), subjects=np.concatenate(subs),
                   benchmarks=np.concatenate(bens))
