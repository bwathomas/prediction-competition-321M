"""Feature ablation: restrict an architecture to a subset of columns.

A layer-2 "architecture part" is a LinearStacker over several ablated variants of the
same architecture (each variant sees a different feature subset), so ablation is the unit
of within-architecture diversity. AblatedModel slices X in both fit and predict, so the
wrapped model can only ever use its kept columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ablation:
    name: str
    columns: tuple  # column names the variant KEEPS


class AblatedModel:
    def __init__(self, base_factory, keep_idx):
        self._model = base_factory()
        self._keep = list(keep_idx)

    def fit(self, X, y):
        self._model.fit(np.asarray(X)[:, self._keep], y)
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X)[:, self._keep])


def make_ablated_factory(base_factory, feature_columns, keep_columns):
    """Return a factory producing AblatedModels restricted to keep_columns (in the given
    order). Raises if a kept column is absent so an ablation can't silently no-op."""
    cols = list(feature_columns)
    missing = [c for c in keep_columns if c not in cols]
    if missing:
        raise KeyError(f"ablation keep_columns not in feature_columns: {missing}")
    keep_idx = [cols.index(c) for c in keep_columns]
    return lambda: AblatedModel(base_factory, keep_idx)
