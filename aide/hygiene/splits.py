"""Item-uniform OOF folds and recursive (nested) inner folds for layer-2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifest import SplitManifest, item_fold


@dataclass(frozen=True)
class Fold:
    index: int
    train_item_keys: tuple
    oof_item_keys: tuple


def outer_folds(manifest: SplitManifest) -> list:
    items = sorted(manifest.assignment)
    folds = []
    for f in range(manifest.n_folds):
        oof = tuple(k for k in items if manifest.assignment[k] == f)
        trn = tuple(k for k in items if manifest.assignment[k] != f)
        folds.append(Fold(index=f, train_item_keys=trn, oof_item_keys=oof))
    return folds


def row_fold_ids(item_keys_per_row, manifest: SplitManifest) -> np.ndarray:
    return np.array([manifest.assignment[str(k)] for k in item_keys_per_row], dtype=int)


def inner_folds(train_item_keys, n_folds: int, seed: int, outer_index: int) -> list:
    """Nested OOF over an outer fold's TRAIN items only.

    Used to generate OOF layer-1 predictions that feed the layer-2 stacker so the
    stacker never trains on a member's in-sample (optimistic) predictions. The inner
    seed is derived from (seed, outer_index) so each outer fold recurses independently
    and deterministically, and never touches the outer fold's OOF items.
    """
    items = sorted(set(str(k) for k in train_item_keys))
    inner_seed = seed + 1000 + outer_index
    assign = {k: item_fold(k, n_folds, inner_seed) for k in items}
    folds = []
    for f in range(n_folds):
        oof = tuple(k for k in items if assign[k] == f)
        trn = tuple(k for k in items if assign[k] != f)
        folds.append(Fold(index=f, train_item_keys=trn, oof_item_keys=oof))
    return folds
