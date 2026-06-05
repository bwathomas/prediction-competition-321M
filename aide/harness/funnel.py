"""Load-only feature funnel.

Routes already-cached feature groups to the architecture that needs them. It NEVER
generates features: a missing group raises ``CacheMissError`` rather than recomputing,
so an agent can never silently fall back to an uncached (and potentially leaky or
mismatched) feature path. Feature groups are stored one-per-`.npz` with arrays
``X`` (float32 [n_rows, n_cols]), ``columns`` (str[n_cols]), ``row_ids`` (str[n_rows]).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CacheMissError(FileNotFoundError):
    """Raised when a requested feature group is not present in the cache."""


@dataclass
class FeatureBlock:
    X: np.ndarray
    columns: list
    row_ids: np.ndarray


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, group: str) -> Path:
        return self.root / f"{group}.npz"

    def available(self) -> set:
        return {p.stem for p in self.root.glob("*.npz")}

    def load_group(self, group: str) -> FeatureBlock:
        p = self._path(group)
        if not p.exists():
            raise CacheMissError(
                f"feature group {group!r} not cached at {p} — the funnel is load-only "
                f"and never recomputes; build the cache offline first")
        # allow_pickle stays False (numpy default): caches store only float and
        # unicode-string arrays, so no object-array pickling is needed. This keeps the
        # load path free of the pickle arbitrary-code-execution surface even though the
        # caches are first-party.
        d = np.load(p)
        return FeatureBlock(X=np.asarray(d["X"], dtype=np.float32),
                            columns=[str(c) for c in d["columns"]],
                            row_ids=np.asarray(d["row_ids"]).astype(str))

    def assemble(self, groups, row_ids=None):
        """Column-concatenate the named groups (load-only). Returns (X, columns).

        All groups must share identical row order; a mismatch raises ValueError so a
        misaligned cache can never silently produce a scrambled feature matrix.
        """
        if not groups:
            raise ValueError("assemble() requires at least one feature group")
        blocks = [self.load_group(g) for g in groups]
        ref = np.asarray(row_ids).astype(str) if row_ids is not None else blocks[0].row_ids
        mats, cols = [], []
        for g, b in zip(groups, blocks):
            if not np.array_equal(b.row_ids, ref):
                raise ValueError(f"row_ids of group {g!r} are misaligned with the reference order")
            mats.append(b.X)
            cols.extend(b.columns)
        return np.concatenate(mats, axis=1).astype(np.float32), cols
