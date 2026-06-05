"""Fold-aware feature store (Plan 4 §E) — the single writer and a load-only reader over
the derive-once shard cache.

Routing rule: a **fold-invariant** group (pure geometry / content / raw identity — same
value in every fold) lives in one ``fold="all"`` shard; a **label-derived** group (any
OOF target-encoding or neighbour-label aggregate) is keyed by the outer fold. ``assemble``
applies this routing itself, so a caller can never accidentally read fold ``f``'s rows
from a shard fit with fold ``f``'s labels — OOF discipline is enforced by the cache key.

Both sets are listed EXPLICITLY (not one as the other's complement): a newly added group
that is forgotten here lands in neither, so ``fold_tag`` raises and the coverage test
fails — rather than silently defaulting a label-derived group to ``fold="all"`` and
leaking. (This is the catalog/coverage discipline from the project's [LEARN:design].)
"""
from __future__ import annotations

import numpy as np

from aide.harness.funnel import FeatureBlock  # noqa: F401  (re-exported type)
from aide.hygiene.probes import assert_columns_covered
from aide.hygiene.proxy_tree import NEUTRAL_ITEM

from .cache import FeatureCache

# Pure geometry / content / raw identity: label-free ⇒ identical across folds.
FOLD_INVARIANT_GROUPS = frozenset({
    "item_embedding", "item_pool", "centroid_distance", "item_cluster",
    "semantic_category", "subject_key", "subject_content", "subject_embedding",
    "subject_meta_categorical", "subject_meta_numeric", "benchmark", "condition",
    "data_category", "benchmark_meta_categorical", "benchmark_meta_numeric",
    "benchmark_conditions", "nn_geometry", "cluster_geometry",
})

# Anything whose value depends on the target y ⇒ must be OOF, hence keyed by outer fold.
LABEL_DERIVED_GROUPS = frozenset({
    "cluster_passrate", "subject_mean", "nn_passrate", "mean_encoded_subject",
    "mean_encoded_benchmark", "nn_label_derivatives", "cluster_subject",
    "groupby_subject_metadata", "interactions_subject", "counts_subject",
    "groupby_benchmark_metadata",
})


def fold_tag(group: str, fold):
    """The shard fold tag for ``group``: ``"all"`` if fold-invariant, else the outer fold.
    Raises on an unclassified group so a new feature can't silently get the wrong fold."""
    if group in FOLD_INVARIANT_GROUPS:
        return "all"
    if group in LABEL_DERIVED_GROUPS:
        return fold
    raise KeyError(
        f"unclassified feature group {group!r}: add it to FOLD_INVARIANT_GROUPS or "
        f"LABEL_DERIVED_GROUPS in aide/features/store.py before deriving it")


class FoldFeatureStore:
    def __init__(self, cache: FeatureCache, *, embedding_family: str, seed: int, n_folds: int):
        self.cache = cache
        self.family = embedding_family
        self.seed = seed
        self.n_folds = n_folds

    def _key(self, group: str, fold, inner_fold=None):
        # inner (layer-2 recursive) variant only applies to label-derived groups
        inner = inner_fold if group in LABEL_DERIVED_GROUPS else None
        return self.cache.key(self.family, group, fold=fold_tag(group, fold),
                              seed=self.seed, n_folds=self.n_folds, inner_fold=inner)

    # --- the ONLY writer -------------------------------------------------------------
    def write_group(self, group: str, fold, block: FeatureBlock, *, inputs_hash: str,
                    overwrite: bool = False) -> str:
        return self.cache.write_shard(self._key(group, fold), block,
                                      inputs_hash=inputs_hash, overwrite=overwrite)

    def write_blocks(self, blocks: dict, fold, *, inputs_hash: str,
                     overwrite: bool = False) -> dict:
        """Persist all groups a codec returned in one call (e.g. derive_nn → 3 groups)."""
        return {g: self.write_group(g, fold, blk, inputs_hash=inputs_hash, overwrite=overwrite)
                for g, blk in blocks.items()}

    # --- load-only reader ------------------------------------------------------------
    def assemble(self, groups, fold, row_ids=None, *, inner_fold=None,
                 check_coverage: bool = True, neutral_prefixes=None):
        """Column-concatenate the routed fold-shards of ``groups`` (load-only).

        Missing shard → ``CacheMissError`` (never recomputes). Row order is asserted
        against ``row_ids`` (or the first block's). With ``check_coverage`` the assembled
        columns must each be a known proxy or neutral prefix, else ``AssertionError``.
        Block references are dropped after stacking (memory discipline §B.4).
        """
        if not groups:
            raise ValueError("assemble() requires at least one feature group")
        blocks = [self.cache.read_shard(self._key(g, fold, inner_fold)) for g in groups]
        ref = np.asarray(row_ids).astype(str) if row_ids is not None else blocks[0].row_ids
        mats, cols = [], []
        for g, b in zip(groups, blocks):
            if not np.array_equal(b.row_ids, ref):
                raise ValueError(f"row_ids of group {g!r} are misaligned with the reference order")
            mats.append(b.X)
            cols.extend(b.columns)
        X = np.concatenate(mats, axis=1).astype(np.float32)
        del blocks, mats  # free per-group arrays before the model sees the matrix
        if check_coverage:
            assert_columns_covered(cols, neutral_prefixes=neutral_prefixes or NEUTRAL_ITEM)
        return X, cols
