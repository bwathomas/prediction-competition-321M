"""Per-fold builders for the OOF training pipeline.

These helpers wrap existing fit / build functions with fold-restricted
inputs. Every function here is callable per fold and returns artifacts
that are LOCAL TO THAT FOLD: a fold-scoped NN index over only the
fold's training items, a fold-scoped passrate table, a fold-scoped
mean-encoded statistics object, etc.

The actual per-fold training loop lives in the notebook (it needs too
many notebook-only globals -- ModelConfig, TrainConfig, the dropout
monkey-patch, etc. -- to factor out cleanly), but the helpers below
keep the leakage-sensitive plumbing testable in isolation.

Key invariant enforced everywhere: NOTHING in a fold's pipeline may
read a value that depends on its OWN OOF rows. We enforce this with
the assert helpers in `oof_folds.py` and with careful in-fold slicing
of the input DataFrames before any aggregation runs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.oof_folds import ItemFold

LOG = logging.getLogger("oof_pipeline")


# ---------------------------------------------------------------------------
# DataFrame slicing
# ---------------------------------------------------------------------------


def slice_train_rows(primary_train_df, fold: ItemFold, *, side: str):
    """Slice ``primary.train`` to a fold's TRAIN rows or OOF rows.

    Parameters
    ----------
    primary_train_df
        The full training DataFrame (pandas). Index is positional;
        we use ``.iloc`` on it so the caller's index is irrelevant.
    fold
        The fold spec.
    side
        Either ``"train"`` (returns rows for the fold's K-1 train items)
        or ``"oof"`` (returns rows for the fold's held-out items).

    Returns
    -------
    A DataFrame slice that ALIASES the original (no copy). Downstream
    builders treat it as read-only.
    """
    if side == "train":
        idx = fold.train_row_idx
    elif side == "oof":
        idx = fold.oof_row_idx
    else:
        raise ValueError(f"side must be 'train' or 'oof', got {side!r}")
    return primary_train_df.iloc[idx]


# ---------------------------------------------------------------------------
# Per-fold item index map + per-item array reindexing
# ---------------------------------------------------------------------------


def build_fold_item_index_map(fold: ItemFold) -> dict[str, int]:
    """Map fold's train item_keys to dense row indices ``[0..n_fold_items)``.

    The fold-scoped NN index, passrate table, and conditional context
    all use this row-index basis. Lookup misses (item_key not in this
    fold's train set) return ``-1`` and are filtered upstream.
    """
    return {str(k): i for i, k in enumerate(fold.train_item_keys)}


def reindex_per_item_array(
    *,
    arr: np.ndarray,
    train_item_keys_global: Sequence[str],
    fold: ItemFold,
    fill: float | int = -1,
) -> np.ndarray:
    """Reindex a global per-train-item array down to a fold's items.

    The global pipeline builds e.g. ``item_benchmark_id_arr`` of length
    ``len(train_item_keys_global)``, where row *i* corresponds to
    ``train_item_keys_global[i]``. The fold-scoped NN/passrate/conditional
    builders expect the same per-item arrays but indexed by the
    fold's train item ordering.

    This helper does that mapping in one shot and fills any item that
    ISN'T in the global ordering with ``fill`` (defensive; in practice
    every fold item should be in the global key list).
    """
    global_index = {str(k): i for i, k in enumerate(train_item_keys_global)}
    out = np.full(len(fold.train_item_keys), fill, dtype=arr.dtype)
    for i, k in enumerate(fold.train_item_keys):
        gi = global_index.get(str(k), -1)
        if gi >= 0:
            out[i] = arr[gi]
    return out


# ---------------------------------------------------------------------------
# Early-stopping split inside a fold
# ---------------------------------------------------------------------------


def split_fold_train_for_early_stopping(
    *,
    fold: ItemFold,
    item_keys_per_row: np.ndarray,
    es_val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Item-grouped 90/10 split of a fold's TRAIN rows for early stopping.

    Per-fold Member 1 (the IRT-MLP) needs an internal early-stopping
    val set. We cannot use ``primary.val`` (would leak val into the
    OOF model selection), and we cannot use ``fold.oof_row_idx``
    (would early-stop on the very rows we're about to score). So:
    take a deterministic 10% slice of fold-train ITEMS (item-grouped
    again, so an item never appears in both early-stop train and
    early-stop val), use those for early stopping.

    Returns ``(es_train_row_idx, es_val_row_idx)`` -- both are indices
    into the FULL ``primary.train`` row space (NOT into ``fold.train_row_idx``).
    """
    n_items_fold = len(fold.train_item_keys)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_items_fold)
    n_val = max(1, int(round(float(es_val_fraction) * n_items_fold)))
    es_val_items = set(fold.train_item_keys[i] for i in perm[:n_val])

    es_val_row_mask = np.zeros(len(item_keys_per_row), dtype=bool)
    fold_train_row_set = set(fold.train_row_idx.tolist())
    for r in fold.train_row_idx:
        if str(item_keys_per_row[r]) in es_val_items:
            es_val_row_mask[r] = True

    es_val_row_idx = np.where(es_val_row_mask)[0].astype(np.int64)
    es_train_row_idx = np.array(
        [r for r in fold.train_row_idx if not es_val_row_mask[r]],
        dtype=np.int64,
    )

    # Defensive: ES train and ES val are item-disjoint within fold-train.
    es_train_items = set(str(item_keys_per_row[r]) for r in es_train_row_idx)
    if es_train_items & es_val_items:
        raise RuntimeError(
            "BUG: early-stopping split produced item overlap between "
            "ES-train and ES-val. Inspect split_fold_train_for_early_stopping."
        )

    LOG.info(
        "split_fold_train_for_early_stopping: fold=%d  es_train_rows=%d  "
        "es_val_rows=%d  es_val_items=%d/%d (%.1f%%)",
        fold.fold_id, int(es_train_row_idx.size), int(es_val_row_idx.size),
        n_val, n_items_fold, 100.0 * n_val / n_items_fold,
    )
    return es_train_row_idx, es_val_row_idx


# ---------------------------------------------------------------------------
# NN index builder (per-fold)
# ---------------------------------------------------------------------------


def build_fold_nn_index(
    *,
    fold: ItemFold,
    item_emb_lookup: Mapping[str, np.ndarray],
    out_dir: Path,
    nn_cfg,
    TrainingNNIndex,
):
    """Build a fresh ``TrainingNNIndex`` over only this fold's train items.

    ``TrainingNNIndex.build_from_lookup`` is idempotent: writes the
    fold's index files to ``out_dir`` (which should be per-fold to
    avoid clobbering between folds, e.g. ``NN_DIR / f"fold_{f}"``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return TrainingNNIndex.build_from_lookup(
        item_emb_lookup=item_emb_lookup,
        out_dir=out_dir,
        cfg=nn_cfg,
        item_keys=list(fold.train_item_keys),
    )


# ---------------------------------------------------------------------------
# OOF prediction accumulator
# ---------------------------------------------------------------------------


class OofPredictionAccumulator:
    """Hold one OOF prediction per training row across the fold loop.

    Initialized to NaN; per-fold writes fill `oof_row_idx` slices.
    A final ``finalize()`` asserts no NaN remains -- that catches the
    "you forgot a fold" bug (every train row must have been written
    exactly once).
    """

    def __init__(self, n_train_rows: int, name: str = ""):
        self._buf = np.full(int(n_train_rows), np.nan, dtype=np.float64)
        self._writes = np.zeros(int(n_train_rows), dtype=np.int32)
        self._name = str(name)

    def write_fold(self, oof_row_idx: np.ndarray, preds: np.ndarray) -> None:
        idx = np.asarray(oof_row_idx, dtype=np.int64)
        p = np.asarray(preds, dtype=np.float64)
        if idx.shape != p.shape:
            raise ValueError(
                f"[{self._name}] write_fold shape mismatch: "
                f"oof_row_idx={idx.shape} preds={p.shape}"
            )
        # Detect double-write: any slot we're about to fill twice is a bug.
        if int(self._writes[idx].max(initial=0)) > 0:
            raise RuntimeError(
                f"[{self._name}] write_fold: trying to overwrite slots that "
                "were already written by a previous fold. Fold-set should be a "
                "partition; double-write usually means two folds' oof_row_idx "
                "overlap. Inspect make_item_grouped_folds."
            )
        self._buf[idx] = p
        self._writes[idx] += 1

    def finalize(self) -> np.ndarray:
        if int(self._writes.min()) == 0:
            n_missing = int(np.sum(self._writes == 0))
            raise RuntimeError(
                f"[{self._name}] finalize: {n_missing} training row(s) never "
                "received an OOF prediction. This usually means a fold was "
                "skipped or its compute_fn raised silently. Re-run the per-"
                "fold loop and confirm every fold filled its oof_row_idx."
            )
        if not np.isfinite(self._buf).all():
            n_nan = int(np.sum(~np.isfinite(self._buf)))
            raise RuntimeError(
                f"[{self._name}] finalize: {n_nan} non-finite OOF predictions. "
                "A member produced NaN/Inf. Investigate per-fold compute "
                "(LightGBM uninitialized leaves, kNN cold-start fallback, etc.)."
            )
        return self._buf.copy()

    @property
    def buffer(self) -> np.ndarray:
        return self._buf

    def coverage_summary(self) -> dict:
        return {
            "n_rows": int(self._buf.shape[0]),
            "n_written": int(np.sum(self._writes > 0)),
            "n_double_written": int(np.sum(self._writes > 1)),
            "n_unwritten": int(np.sum(self._writes == 0)),
            "n_nan": int(np.sum(~np.isfinite(self._buf))),
        }


# ---------------------------------------------------------------------------
# Shuffled-label control utility (Gate 1c support)
# ---------------------------------------------------------------------------


def make_permuted_labels(*, y: np.ndarray, seed: int) -> np.ndarray:
    """Random permutation of a label vector. Used by Gate 1c.

    A correct OOF pipeline trained on these labels must NOT beat
    chance on the held-out val set (because the labels carry zero
    real signal). If it does, there's leakage somewhere upstream.

    We use a deterministic seed so the same Gate 1c run is reproducible.
    """
    rng = np.random.default_rng(int(seed))
    y_arr = np.asarray(y).copy()
    rng.shuffle(y_arr)
    return y_arr


def entropy_of_label_prior(y: np.ndarray) -> float:
    """Entropy of the label prior in nats. The OOF stacker trained on
    shuffled labels SHOULD converge to log-loss ~= this value
    (predicting the constant label mean is the optimal strategy when
    labels carry no signal). If the trained stacker on shuffled labels
    beats this by more than ~0.005 nats, suspect leakage.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.size == 0:
        return 0.0
    p = float(np.mean(y_arr))
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


__all__ = [
    "slice_train_rows",
    "build_fold_item_index_map",
    "reindex_per_item_array",
    "split_fold_train_for_early_stopping",
    "build_fold_nn_index",
    "OofPredictionAccumulator",
    "make_permuted_labels",
    "entropy_of_label_prior",
]
