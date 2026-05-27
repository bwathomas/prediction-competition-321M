"""Item-grouped K-fold splitter and leakage probes for OOF stacking.

The cold-start competition holds out unseen items at test time. To train a
stacker on truly out-of-fold (OOF) member predictions -- the only way to
avoid a meta-learner that overfits to in-sample member optimism -- we
split `primary.train` by *item_key* into K folds. For each fold f:

  * Members are trained on the union of the OTHER K-1 folds' items.
  * Each member then predicts on fold f's held-out items.
  * Concatenating across folds yields exactly one OOF prediction per
    training row, computed by a model that NEVER saw that row's item.

The fold splitter and the leakage probes live here so they're trivially
unit-testable. The per-fold compute graph (NN index, member training,
schema fitting, etc.) is wired into the notebook because it touches a
lot of pipeline-specific state.

Critical invariants (each enforced by an assertion helper below):

  (1) `train_item_keys` and `oof_item_keys` for a fold are DISJOINT.
  (2) Per-row fold assignment is deterministic given (item_keys, seed,
      n_folds).
  (3) NN-feature lookup for OOF rows of fold f returns neighbors whose
      item_key lies in fold f's TRAIN items, not any of fold f's OOF
      items. (This is the subtle leakage path that ruins everything
      if missed.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

LOG = logging.getLogger("oof_folds")


# ---------------------------------------------------------------------------
# Fold spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemFold:
    """Specification for a single fold of the K-fold OOF schedule.

    Attributes
    ----------
    fold_id
        0-indexed fold number.
    train_item_keys
        The item_keys whose rows form THIS fold's TRAINING set. Members
        are fitted on (a slice of) rows whose item_key is in here.
    oof_item_keys
        The item_keys held out for THIS fold's OOF predictions. None of
        these may appear in `train_item_keys`.
    train_row_idx
        Indices into the full training row array whose item_key is in
        `train_item_keys`. Members fit on `X[train_row_idx], y[train_row_idx]`.
    oof_row_idx
        Indices into the full training row array whose item_key is in
        `oof_item_keys`. The fold's member produces predictions on these
        rows; we fill `p_member_oof[oof_row_idx] = preds`.
    """

    fold_id: int
    train_item_keys: tuple[str, ...]
    oof_item_keys: tuple[str, ...]
    train_row_idx: np.ndarray
    oof_row_idx: np.ndarray

    def __post_init__(self) -> None:
        # Numpy arrays must be 1-D int64 (defensive — np.asarray below
        # in make_item_grouped_folds enforces this but a hand-crafted
        # ItemFold could still trip it).
        if self.train_row_idx.ndim != 1 or self.oof_row_idx.ndim != 1:
            raise ValueError(
                f"row idx arrays must be 1-D, got "
                f"train={self.train_row_idx.shape} oof={self.oof_row_idx.shape}"
            )
        if self.train_row_idx.dtype.kind != "i":
            raise ValueError(
                f"train_row_idx must be integer dtype, got {self.train_row_idx.dtype}"
            )
        if self.oof_row_idx.dtype.kind != "i":
            raise ValueError(
                f"oof_row_idx must be integer dtype, got {self.oof_row_idx.dtype}"
            )


def make_item_grouped_folds(
    *,
    item_keys_per_row: Sequence[str],
    n_folds: int,
    seed: int,
) -> list[ItemFold]:
    """Build K item-grouped folds over a row-level item_key array.

    Parameters
    ----------
    item_keys_per_row
        Length-N sequence; element i is the item_key for training row i.
    n_folds
        Number of folds. Must be >= 2.
    seed
        Deterministic shuffle seed for assigning items to folds.

    Returns
    -------
    list[ItemFold]
        Exactly `n_folds` ItemFolds. Item-key universe is partitioned
        across the folds' `oof_item_keys` (i.e. concatenating all
        `oof_item_keys` reproduces the set of unique item_keys, with
        each key in exactly one fold). The fold sizes are within 1
        of `n_unique_items / n_folds`.
    """
    if int(n_folds) < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    rows = np.asarray(list(item_keys_per_row), dtype=object)
    N = int(rows.shape[0])
    if N == 0:
        raise ValueError("make_item_grouped_folds: no rows provided")

    # Stable-sorted unique item_keys, then deterministic shuffle.
    unique_keys = np.array(sorted(set(str(k) for k in rows)), dtype=object)
    n_items = int(unique_keys.shape[0])
    if n_items < int(n_folds):
        raise ValueError(
            f"n_unique_items {n_items} < n_folds {n_folds}; cannot make "
            "an item-grouped split where every fold has >=1 OOF item."
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_items)
    shuffled = unique_keys[perm]

    # np.array_split gives near-equal partition sizes (max delta = 1).
    fold_items: list[np.ndarray] = np.array_split(shuffled, int(n_folds))

    # Build row-id arrays per fold. We do this with a single pass over
    # the row item_keys: for each row, look up which fold its item is in,
    # append to that fold's list. ~O(N) given a dict lookup.
    item_to_fold: dict[str, int] = {}
    for f, items in enumerate(fold_items):
        for k in items:
            item_to_fold[str(k)] = f
    per_fold_rows: list[list[int]] = [[] for _ in range(int(n_folds))]
    for i, k in enumerate(rows):
        f = item_to_fold.get(str(k), -1)
        if f < 0:
            # Shouldn't happen given unique_keys was built from `rows`,
            # but defensive against caller-side weirdness (NaN keys etc.).
            raise RuntimeError(
                f"row {i} item_key {k!r} did not land in any fold; "
                "this indicates a bug in fold assignment."
            )
        per_fold_rows[f].append(i)

    folds: list[ItemFold] = []
    full_train_item_set = set(str(k) for k in unique_keys)
    for f in range(int(n_folds)):
        oof_items_f = tuple(str(k) for k in fold_items[f])
        oof_item_set = set(oof_items_f)
        train_items_f = tuple(sorted(full_train_item_set - oof_item_set))
        oof_row_idx_f = np.asarray(per_fold_rows[f], dtype=np.int64)
        # Train rows for fold f = all rows whose item is NOT in fold f's OOF set
        # = rows in other folds' OOF lists.
        train_row_idx_f = np.asarray(
            sorted(i for j in range(int(n_folds)) if j != f for i in per_fold_rows[j]),
            dtype=np.int64,
        )
        folds.append(
            ItemFold(
                fold_id=int(f),
                train_item_keys=train_items_f,
                oof_item_keys=oof_items_f,
                train_row_idx=train_row_idx_f,
                oof_row_idx=oof_row_idx_f,
            )
        )

    LOG.info(
        "make_item_grouped_folds: N=%d unique_items=%d n_folds=%d seed=%d  "
        "fold OOF item counts=%s",
        N, n_items, int(n_folds), int(seed),
        [len(f.oof_item_keys) for f in folds],
    )
    return folds


# ---------------------------------------------------------------------------
# Gate 1a: item-disjointness (per-fold invariant)
# ---------------------------------------------------------------------------


def assert_item_disjoint(fold: ItemFold) -> None:
    """RED-TEAM GATE 1a: fold's train and OOF item sets must be disjoint.

    Raises ``AssertionError`` if any item_key appears on both sides.
    Prints the violation count and an example offender on failure so
    callers see what tripped.
    """
    train_set = set(fold.train_item_keys)
    oof_set = set(fold.oof_item_keys)
    intersection = train_set & oof_set
    n_violations = int(len(intersection))
    if n_violations > 0:
        example = next(iter(intersection))
        raise AssertionError(
            f"GATE 1a violation: fold {fold.fold_id} has {n_violations} item_key(s) "
            f"appearing in BOTH train and OOF sides. Example: {example!r}. "
            "This is an item-grouped split bug."
        )


def assert_row_idx_partition(folds: Sequence[ItemFold], n_rows: int) -> None:
    """Stronger Gate 1a invariant: every training row must appear in
    exactly ONE fold's OOF set, and union of all `train_row_idx` covers
    everything except that fold's own OOF rows. Caught a few real bugs
    in early iterations (off-by-one in `np.array_split` glue)."""
    seen_in_oof = np.zeros(int(n_rows), dtype=np.int64)
    for f in folds:
        seen_in_oof[f.oof_row_idx] += 1
    if not np.array_equal(seen_in_oof, np.ones(int(n_rows), dtype=np.int64)):
        bad = int(np.sum(seen_in_oof != 1))
        raise AssertionError(
            f"GATE 1a (row-partition variant): {bad} training row(s) appear "
            f"in zero or >1 folds' OOF set. Expected exactly 1 per row. "
            "Inspect `make_item_grouped_folds` output."
        )
    # Train rows for fold f should be exactly the complement of fold f's
    # OOF rows, i.e. the union of other folds' OOF rows.
    for f in folds:
        expected_train_set = set(np.arange(int(n_rows), dtype=np.int64).tolist())
        expected_train_set -= set(f.oof_row_idx.tolist())
        actual_train_set = set(f.train_row_idx.tolist())
        if expected_train_set != actual_train_set:
            raise AssertionError(
                f"GATE 1a (row-partition variant): fold {f.fold_id}'s "
                f"train_row_idx is not the complement of its oof_row_idx. "
                f"|expected|={len(expected_train_set)} "
                f"|actual|={len(actual_train_set)}."
            )


# ---------------------------------------------------------------------------
# Gate 1b: NN-neighbor-in-fold-train probe
# ---------------------------------------------------------------------------


def assert_nn_neighbors_in_fold_train(
    *,
    fold: ItemFold,
    oof_row_neighbor_item_keys: np.ndarray,
    sample_size: int | None = None,
    seed: int = 0,
) -> dict:
    """RED-TEAM GATE 1b: for a sample of OOF rows of this fold,
    EVERY returned NN-neighbor's item_key must lie in
    ``fold.train_item_keys``. If even one neighbor's key is in
    ``fold.oof_item_keys``, the NN index was built on data the fold
    isn't allowed to see, which would leak per-row labels through
    the neighbor-derived features (mean_sim, neighbor_support, NN-
    feature pass-rate aggregates etc.).

    Parameters
    ----------
    fold
        The fold whose neighbors we're auditing.
    oof_row_neighbor_item_keys
        Shape ``[N_oof, k]``: for each OOF row, the item_keys of the
        top-k neighbors returned by the (fold-scoped) NN index.
        Use ``-1`` / empty-string entries for "no neighbor" slots; these
        are skipped, not crashed.
    sample_size
        If not None, randomly sample this many OOF rows before checking
        (cheap on very large val sets). Default: check all rows.
    seed
        Deterministic sampling seed.

    Returns
    -------
    dict with ``n_checked``, ``n_violations``, ``worst_offender_row``,
    ``worst_offender_neighbor_key``. Raises AssertionError if any
    violation found.
    """
    if oof_row_neighbor_item_keys.ndim != 2:
        raise ValueError(
            f"oof_row_neighbor_item_keys must be 2-D [N_oof, k], got "
            f"shape {oof_row_neighbor_item_keys.shape}"
        )
    N_oof = int(oof_row_neighbor_item_keys.shape[0])
    if N_oof == 0:
        return {"n_checked": 0, "n_violations": 0,
                "worst_offender_row": -1, "worst_offender_neighbor_key": None}

    if sample_size is not None and int(sample_size) < N_oof:
        rng = np.random.default_rng(int(seed))
        sampled = rng.choice(N_oof, size=int(sample_size), replace=False)
    else:
        sampled = np.arange(N_oof, dtype=np.int64)

    oof_set = set(fold.oof_item_keys)
    n_violations = 0
    worst_row = -1
    worst_key = None
    for r in sampled:
        for k in oof_row_neighbor_item_keys[int(r)]:
            ks = str(k)
            if ks == "" or ks == "-1":
                continue
            if ks in oof_set:
                n_violations += 1
                if worst_row < 0:
                    worst_row = int(r)
                    worst_key = ks

    result = {
        "n_checked": int(sampled.size),
        "n_violations": int(n_violations),
        "worst_offender_row": int(worst_row),
        "worst_offender_neighbor_key": worst_key,
    }
    if n_violations > 0:
        raise AssertionError(
            f"GATE 1b violation: fold {fold.fold_id} NN index returned "
            f"{n_violations} neighbor(s) whose item_key is in the fold's "
            f"OOF set (over {int(sampled.size)} sampled rows). Example: "
            f"row={worst_row} neighbor_key={worst_key!r}. "
            "The NN index must be REBUILT on each fold's train items only."
        )
    return result


# ---------------------------------------------------------------------------
# Optimism check (Gate 1d helper)
# ---------------------------------------------------------------------------


def report_train_vs_val_optimism(
    *,
    train_loss: float,
    val_loss: float,
    threshold_nats: float = 0.03,
    label: str = "OOF stacker",
) -> dict:
    """RED-TEAM GATE 1d: if val_loss is much worse than train_loss, the
    OOF predictions are still contaminated. A healthy stacker with
    OOF inputs should have a SMALL train-vs-val gap because the
    training inputs themselves never saw their own row's label.

    Returns ``{"gap": gap, "flag": bool}``; emits a WARNING log if
    flagged. We don't raise -- the prompt says "Flag if the gap
    exceeds ~0.03 nats", not "block".
    """
    gap = float(val_loss) - float(train_loss)
    flagged = gap > float(threshold_nats)
    if flagged:
        LOG.warning(
            "%s train-vs-val optimism gap = %.4f nats (val %.5f > train %.5f). "
            "Threshold %.3f exceeded -- OOF predictions may be partially "
            "contaminated. Re-audit fold scoping for NN features, mean-encoded "
            "stats, member feature schema, and per-fold caches.",
            label, gap, float(val_loss), float(train_loss), float(threshold_nats),
        )
    return {"gap": gap, "flag": flagged}


# ---------------------------------------------------------------------------
# Convenience: per-fold cache key helpers
# ---------------------------------------------------------------------------


def fold_cache_suffix(
    *,
    fold_id: int,
    train_item_keys: Sequence[str],
) -> str:
    """A short stable digest for keying per-fold caches by their
    train-item-set identity. Two folds with the same train-item-set
    (e.g. different seeds that happen to produce the same partition)
    will produce the same key; differing partitions produce different
    keys. Used in `cache_or_compute(name, key_inputs=(..., fold_suffix))`.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(f"fold={int(fold_id)};n={len(train_item_keys)};".encode("utf-8"))
    # Hash sorted item keys for determinism regardless of input order.
    for k in sorted(str(x) for x in train_item_keys):
        h.update(k.encode("utf-8"))
        h.update(b";")
    return h.hexdigest()[:16]


__all__ = [
    "ItemFold",
    "make_item_grouped_folds",
    "assert_item_disjoint",
    "assert_row_idx_partition",
    "assert_nn_neighbors_in_fold_train",
    "report_train_vs_val_optimism",
    "fold_cache_suffix",
]
