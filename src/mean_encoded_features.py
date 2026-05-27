"""Mean-encoded (a.k.a. target-encoded) features for the stacked ensemble.

This module provides cold-start-safe, Bayesian-shrunk per-cell statistics
computed exclusively on training rows. Two families of features are exposed:

1. **Member 2 (GBDT) interactions** -- subject x item-cluster cells plus
   subject x (benchmark_condition) and a few marginal counts. These get
   APPENDED to the existing 1202-feature member-feature matrix and give
   the trees something to split on that's both novel (no current dense
   feature exposes it) AND informative on unseen items.

2. **Member 4 (LogReg) marginals** -- a compact (typically ~10-20)
   feature vector consisting purely of mean-encoded subject / bc /
   cluster statistics. Trained INSTEAD of the embedding-derived dense
   matrix Member 4 currently shares with Member 2, this gives the
   stacker a member whose input view has zero overlap with Members
   1/2/3 (which all lean on qwen embeddings in some form).

Why mean-encoding instead of one-hots
-------------------------------------
A "true" subject + bc one-hot matrix at this scale is
``5M rows x ~1100 cols x float32 = 22 GB`` dense (or ~80 MB CSR, but
that requires a sparse-aware trainer the rest of the pipeline doesn't
have). Mean-encoded features collapse those one-hots into a handful of
scalars per row that ENCODE the same conditional information --
``subj_mean[s]`` literally IS the average label over rows of subject
``s``, which is what a logistic regression's one-hot weight would
converge to under a strong-enough L2 penalty.

Bayesian shrinkage (``smoothing`` parameter) handles low-count cells:
empty (subject, cluster) pairs return the per-subject or global mean
instead of collapsing to 0 / NaN.

Cold-start safety
-----------------
All statistics are fitted ONCE on training rows; held-out items
(present in val but not in train) get their stats via the per-cluster
fallback (the cluster-level mean is itself a train-time aggregate so
it generalizes). Val labels are NEVER consulted at any point during
``fit_mean_encoded_stats``. The test
``tests/test_mean_encoded_features.py`` red-teams this invariant.

Runtime contract
----------------
``apply_member2_interaction_features`` and
``apply_member4_marginal_features`` are pure numpy; they accept the
fitted stats state plus per-row id arrays and emit a dense float32
feature matrix. No torch, no scipy at inference time -- the runtime
just needs ``MeanEncodedStats.load(...)`` plus these two functions.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("mean_encoded_features")

_EPS = 1.0e-6
_SENTINEL = -1  # convention for "this row has no valid id" (rare; falls back)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _bayes_shrunk_cell_mean(
    sum_y: np.ndarray,        # any shape, float64
    n: np.ndarray,            # same shape, float64
    fallback_mean: float | np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Bayesian shrinkage to ``fallback_mean``.

    ``mean[cell] = (sum_y[cell] + smoothing * fallback_mean) / (n[cell] + smoothing)``

    Empty cells (n == 0) collapse to ``fallback_mean`` exactly. Filled
    cells trend toward the cell's empirical mean as ``n[cell]``
    dominates ``smoothing``. ``smoothing=30`` is a reasonable default
    -- at n=300 the cell's empirical mean gets ~91% weight.
    """
    smoothing_f = float(smoothing)
    if isinstance(fallback_mean, (int, float)):
        fm = float(fallback_mean)
    else:
        fm = np.asarray(fallback_mean, dtype=np.float64)
    return (sum_y + smoothing_f * fm) / np.maximum(n + smoothing_f, 1e-12)


def _bincount2d_sum_and_count(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    values: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense ``[n_rows, n_cols]`` sum-of-values and count arrays
    via ``np.bincount`` on a flattened index. Much faster than
    ``np.add.at`` for large N (~50x on 5M rows). Negative ids are
    ignored (rows are skipped, not crashed)."""
    if row_ids.shape != col_ids.shape or row_ids.shape != values.shape:
        raise ValueError(
            f"shape mismatch: rows {row_ids.shape}, cols {col_ids.shape}, "
            f"values {values.shape}"
        )
    rid = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    cid = np.asarray(col_ids, dtype=np.int64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    # Filter sentinel rows in one pass.
    keep = (rid >= 0) & (rid < int(n_rows)) & (cid >= 0) & (cid < int(n_cols))
    if not bool(keep.all()):
        rid = rid[keep]
        cid = cid[keep]
        v = v[keep]
    flat = rid * int(n_cols) + cid
    total = int(n_rows) * int(n_cols)
    sum_y_flat = np.bincount(flat, weights=v, minlength=total)
    cnt_flat = np.bincount(flat, minlength=total).astype(np.float64)
    return (
        sum_y_flat.reshape(int(n_rows), int(n_cols)),
        cnt_flat.reshape(int(n_rows), int(n_cols)),
    )


def _bincount1d_sum_and_count(
    ids: np.ndarray,
    values: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D version of the above."""
    iid = np.asarray(ids, dtype=np.int64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    keep = (iid >= 0) & (iid < int(n_buckets))
    if not bool(keep.all()):
        iid = iid[keep]
        v = v[keep]
    sum_y = np.bincount(iid, weights=v, minlength=int(n_buckets))
    cnt = np.bincount(iid, minlength=int(n_buckets)).astype(np.float64)
    return sum_y, cnt


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MeanEncodedStats:
    """Bayesian-shrunk per-cell pass-rate statistics fitted on train rows.

    Cells with n == 0 fall back to the most specific available aggregate
    (e.g., empty subj-cluster cell falls back to the cluster's overall
    mean, or to the global mean if the cluster itself was unseen).

    All arrays are float32 to keep the saved state compact (~10-50 MB).
    """

    # ---- Per-cell shrunk mean & log1p(count) ----
    # [n_subjects, n_clusters]
    subj_cluster_mean: np.ndarray
    subj_cluster_log1p_n: np.ndarray
    # [n_subjects, n_bcs]  -- subject x benchmark_condition
    subj_bc_mean: np.ndarray
    subj_bc_log1p_n: np.ndarray

    # ---- Per-row-axis marginals (1-D shrunk means) ----
    subj_mean: np.ndarray         # [n_subjects]
    subj_log1p_n: np.ndarray      # [n_subjects]
    cluster_mean: np.ndarray      # [n_clusters]
    cluster_log1p_n: np.ndarray   # [n_clusters]
    bc_mean: np.ndarray           # [n_bcs]
    bc_log1p_n: np.ndarray        # [n_bcs]

    global_mean: float
    smoothing: float
    n_subjects: int
    n_clusters: int
    n_bcs: int

    # Provenance / metadata (not used at inference).
    n_train_rows: int = 0
    fit_method: str = "bayes_shrunk"

    def __post_init__(self) -> None:
        # Shape sanity.
        if self.subj_cluster_mean.shape != (self.n_subjects, self.n_clusters):
            raise ValueError(
                f"subj_cluster_mean shape {self.subj_cluster_mean.shape} != "
                f"({self.n_subjects}, {self.n_clusters})"
            )
        if self.subj_bc_mean.shape != (self.n_subjects, self.n_bcs):
            raise ValueError(
                f"subj_bc_mean shape {self.subj_bc_mean.shape} != "
                f"({self.n_subjects}, {self.n_bcs})"
            )
        for name, arr, length in [
            ("subj_mean", self.subj_mean, self.n_subjects),
            ("subj_log1p_n", self.subj_log1p_n, self.n_subjects),
            ("cluster_mean", self.cluster_mean, self.n_clusters),
            ("cluster_log1p_n", self.cluster_log1p_n, self.n_clusters),
            ("bc_mean", self.bc_mean, self.n_bcs),
            ("bc_log1p_n", self.bc_log1p_n, self.n_bcs),
        ]:
            if arr.shape != (length,):
                raise ValueError(
                    f"{name} shape {arr.shape} != ({length},)"
                )

    # ---- I/O ----

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "mean_encoded_stats.npz",
            subj_cluster_mean=self.subj_cluster_mean.astype(np.float32),
            subj_cluster_log1p_n=self.subj_cluster_log1p_n.astype(np.float32),
            subj_bc_mean=self.subj_bc_mean.astype(np.float32),
            subj_bc_log1p_n=self.subj_bc_log1p_n.astype(np.float32),
            subj_mean=self.subj_mean.astype(np.float32),
            subj_log1p_n=self.subj_log1p_n.astype(np.float32),
            cluster_mean=self.cluster_mean.astype(np.float32),
            cluster_log1p_n=self.cluster_log1p_n.astype(np.float32),
            bc_mean=self.bc_mean.astype(np.float32),
            bc_log1p_n=self.bc_log1p_n.astype(np.float32),
        )
        meta = {
            "global_mean": float(self.global_mean),
            "smoothing": float(self.smoothing),
            "n_subjects": int(self.n_subjects),
            "n_clusters": int(self.n_clusters),
            "n_bcs": int(self.n_bcs),
            "n_train_rows": int(self.n_train_rows),
            "fit_method": str(self.fit_method),
            "format_version": 1,
        }
        (out / "mean_encoded_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "MeanEncodedStats":
        d = Path(in_dir)
        meta = json.loads((d / "mean_encoded_meta.json").read_text(encoding="utf-8"))
        with np.load(d / "mean_encoded_stats.npz") as npz:
            return cls(
                subj_cluster_mean=npz["subj_cluster_mean"].astype(np.float32, copy=False),
                subj_cluster_log1p_n=npz["subj_cluster_log1p_n"].astype(np.float32, copy=False),
                subj_bc_mean=npz["subj_bc_mean"].astype(np.float32, copy=False),
                subj_bc_log1p_n=npz["subj_bc_log1p_n"].astype(np.float32, copy=False),
                subj_mean=npz["subj_mean"].astype(np.float32, copy=False),
                subj_log1p_n=npz["subj_log1p_n"].astype(np.float32, copy=False),
                cluster_mean=npz["cluster_mean"].astype(np.float32, copy=False),
                cluster_log1p_n=npz["cluster_log1p_n"].astype(np.float32, copy=False),
                bc_mean=npz["bc_mean"].astype(np.float32, copy=False),
                bc_log1p_n=npz["bc_log1p_n"].astype(np.float32, copy=False),
                global_mean=float(meta["global_mean"]),
                smoothing=float(meta["smoothing"]),
                n_subjects=int(meta["n_subjects"]),
                n_clusters=int(meta["n_clusters"]),
                n_bcs=int(meta["n_bcs"]),
                n_train_rows=int(meta.get("n_train_rows", 0)),
                fit_method=str(meta.get("fit_method", "unknown")),
            )


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


def fit_mean_encoded_stats(
    *,
    subject_ids: np.ndarray,    # [N] int64, train rows only
    cluster_ids: np.ndarray,    # [N] int64
    bc_ids: np.ndarray,         # [N] int64
    labels: np.ndarray,         # [N] float32/float64
    n_subjects: int,
    n_clusters: int,
    n_bcs: int,
    smoothing: float = 30.0,
) -> MeanEncodedStats:
    """Fit Bayesian-shrunk per-cell pass-rate statistics on TRAIN rows.

    The caller must guarantee that ``labels`` corresponds to held-IN
    rows only (the val/test rows must not appear here). Rows whose
    ``subject_id``, ``cluster_id`` or ``bc_id`` is negative or out-of-
    range are silently skipped (NOT crashed) so caller code can use
    ``-1`` as a "no mapping" sentinel for unseen ids.
    """
    if subject_ids.shape != labels.shape:
        raise ValueError(
            f"subject_ids shape {subject_ids.shape} != labels {labels.shape}"
        )
    if cluster_ids.shape != labels.shape:
        raise ValueError(
            f"cluster_ids shape {cluster_ids.shape} != labels {labels.shape}"
        )
    if bc_ids.shape != labels.shape:
        raise ValueError(
            f"bc_ids shape {bc_ids.shape} != labels {labels.shape}"
        )
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise ValueError("fit_mean_encoded_stats: empty labels array")
    global_mean = float(np.mean(y))
    LOG.info(
        "fit_mean_encoded_stats: N=%d  S=%d  C=%d  BC=%d  global_mean=%.4f  smoothing=%.1f",
        int(y.size), int(n_subjects), int(n_clusters), int(n_bcs),
        global_mean, float(smoothing),
    )

    # ---- 1-D marginals (always available, used as fallbacks) ----
    subj_sum, subj_cnt = _bincount1d_sum_and_count(subject_ids, y, n_subjects)
    cluster_sum, cluster_cnt = _bincount1d_sum_and_count(cluster_ids, y, n_clusters)
    bc_sum, bc_cnt = _bincount1d_sum_and_count(bc_ids, y, n_bcs)

    subj_mean = _bayes_shrunk_cell_mean(subj_sum, subj_cnt, global_mean, smoothing)
    cluster_mean = _bayes_shrunk_cell_mean(cluster_sum, cluster_cnt, global_mean, smoothing)
    bc_mean = _bayes_shrunk_cell_mean(bc_sum, bc_cnt, global_mean, smoothing)

    # ---- 2-D interactions ----
    sc_sum, sc_cnt = _bincount2d_sum_and_count(
        subject_ids, cluster_ids, y, n_subjects, n_clusters
    )
    # Fallback per cell: use cluster_mean[c] (broadcast across rows).
    sc_fb = np.broadcast_to(cluster_mean.reshape(1, -1), sc_sum.shape)
    subj_cluster_mean = _bayes_shrunk_cell_mean(sc_sum, sc_cnt, sc_fb, smoothing)

    sbc_sum, sbc_cnt = _bincount2d_sum_and_count(
        subject_ids, bc_ids, y, n_subjects, n_bcs
    )
    sbc_fb = np.broadcast_to(bc_mean.reshape(1, -1), sbc_sum.shape)
    subj_bc_mean = _bayes_shrunk_cell_mean(sbc_sum, sbc_cnt, sbc_fb, smoothing)

    return MeanEncodedStats(
        subj_cluster_mean=subj_cluster_mean.astype(np.float32),
        subj_cluster_log1p_n=np.log1p(sc_cnt).astype(np.float32),
        subj_bc_mean=subj_bc_mean.astype(np.float32),
        subj_bc_log1p_n=np.log1p(sbc_cnt).astype(np.float32),
        subj_mean=subj_mean.astype(np.float32),
        subj_log1p_n=np.log1p(subj_cnt).astype(np.float32),
        cluster_mean=cluster_mean.astype(np.float32),
        cluster_log1p_n=np.log1p(cluster_cnt).astype(np.float32),
        bc_mean=bc_mean.astype(np.float32),
        bc_log1p_n=np.log1p(bc_cnt).astype(np.float32),
        global_mean=global_mean,
        smoothing=float(smoothing),
        n_subjects=int(n_subjects),
        n_clusters=int(n_clusters),
        n_bcs=int(n_bcs),
        n_train_rows=int(y.size),
        fit_method="bayes_shrunk",
    )


# ---------------------------------------------------------------------------
# Lookup helpers (cold-start-safe, vectorized)
# ---------------------------------------------------------------------------


def _safe_gather_1d(arr: np.ndarray, idx: np.ndarray, default: float) -> np.ndarray:
    """``arr[idx]`` but with out-of-range / negative indices returning
    ``default``. Pure numpy, no Python loop."""
    i = np.asarray(idx, dtype=np.int64).reshape(-1)
    valid = (i >= 0) & (i < int(arr.shape[0]))
    safe_i = np.where(valid, i, 0)
    out = arr[safe_i]
    if not bool(valid.all()):
        out = np.where(valid, out, default)
    return out.astype(np.float32, copy=False)


def _safe_gather_2d(
    arr: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    default_per_row: np.ndarray | float,
) -> np.ndarray:
    """``arr[row_idx, col_idx]`` with bounds-safety. Out-of-range
    indices return ``default_per_row`` (which can be a scalar or a
    per-row array)."""
    r = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    c = np.asarray(col_idx, dtype=np.int64).reshape(-1)
    if r.shape != c.shape:
        raise ValueError(f"row/col shape mismatch: {r.shape} vs {c.shape}")
    valid = (r >= 0) & (r < int(arr.shape[0])) & (c >= 0) & (c < int(arr.shape[1]))
    safe_r = np.where(valid, r, 0)
    safe_c = np.where(valid, c, 0)
    out = arr[safe_r, safe_c]
    if not bool(valid.all()):
        if isinstance(default_per_row, (int, float)):
            out = np.where(valid, out, float(default_per_row))
        else:
            dpr = np.asarray(default_per_row, dtype=arr.dtype).reshape(-1)
            if dpr.shape != r.shape:
                raise ValueError(
                    f"default_per_row shape {dpr.shape} != rows {r.shape}"
                )
            out = np.where(valid, out, dpr)
    return out.astype(np.float32, copy=False)


# Locked feature column orders so offline (notebook) and runtime
# (model.py) agree on positions byte-for-byte. ANY edit to these
# tuples invalidates trained Member 2 / Member 4 caches.
MEMBER2_INTERACTION_FEATURE_NAMES: tuple[str, ...] = (
    "me__subj_cluster_mean",
    "me__subj_cluster_log1p_n",
    "me__subj_cluster_mean_dev_from_subj",   # cell mean - subject overall mean
    "me__subj_bc_mean",
    "me__subj_bc_log1p_n",
    "me__subj_bc_mean_dev_from_subj",
    "me__cluster_mean",
    "me__bc_mean",
)
MEMBER2_INTERACTION_FEATURE_DIM: int = len(MEMBER2_INTERACTION_FEATURE_NAMES)

MEMBER4_MARGINAL_FEATURE_NAMES: tuple[str, ...] = (
    "mg__subj_mean",
    "mg__subj_log1p_n",
    "mg__bc_mean",
    "mg__bc_log1p_n",
    "mg__cluster_mean",
    "mg__cluster_log1p_n",
    "mg__subj_bc_mean",
    "mg__subj_bc_log1p_n",
    "mg__subj_cluster_mean",
    "mg__subj_cluster_log1p_n",
    "mg__subj_mean_dev_from_global",
    "mg__bc_mean_dev_from_global",
    "mg__cluster_mean_dev_from_global",
    "mg__global_mean",   # constant column; learnable as intercept correction
)
MEMBER4_MARGINAL_FEATURE_DIM: int = len(MEMBER4_MARGINAL_FEATURE_NAMES)


def apply_member2_interaction_features(
    stats: MeanEncodedStats,
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
) -> np.ndarray:
    """Build per-row interaction features for Member 2 (the GBDT).

    Returns ``[N, MEMBER2_INTERACTION_FEATURE_DIM]`` float32. Column
    order is locked by :data:`MEMBER2_INTERACTION_FEATURE_NAMES`.
    These features are designed to be APPENDED to the existing
    ``X_train_dense`` / ``X_val_dense`` matrices.
    """
    N = int(np.asarray(subject_ids).shape[0])
    if int(np.asarray(cluster_ids).shape[0]) != N:
        raise ValueError("cluster_ids length must match subject_ids length")
    if int(np.asarray(bc_ids).shape[0]) != N:
        raise ValueError("bc_ids length must match subject_ids length")

    out = np.zeros((N, MEMBER2_INTERACTION_FEATURE_DIM), dtype=np.float32)
    subj_mean_per_row = _safe_gather_1d(
        stats.subj_mean, subject_ids, stats.global_mean
    )
    cluster_mean_per_row = _safe_gather_1d(
        stats.cluster_mean, cluster_ids, stats.global_mean
    )
    bc_mean_per_row = _safe_gather_1d(stats.bc_mean, bc_ids, stats.global_mean)

    sc_mean = _safe_gather_2d(
        stats.subj_cluster_mean, subject_ids, cluster_ids, cluster_mean_per_row
    )
    sc_log_n = _safe_gather_2d(
        stats.subj_cluster_log1p_n, subject_ids, cluster_ids, 0.0
    )
    sbc_mean = _safe_gather_2d(
        stats.subj_bc_mean, subject_ids, bc_ids, bc_mean_per_row
    )
    sbc_log_n = _safe_gather_2d(
        stats.subj_bc_log1p_n, subject_ids, bc_ids, 0.0
    )

    out[:, 0] = sc_mean
    out[:, 1] = sc_log_n
    out[:, 2] = sc_mean - subj_mean_per_row
    out[:, 3] = sbc_mean
    out[:, 4] = sbc_log_n
    out[:, 5] = sbc_mean - subj_mean_per_row
    out[:, 6] = cluster_mean_per_row
    out[:, 7] = bc_mean_per_row
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


def apply_member4_marginal_features(
    stats: MeanEncodedStats,
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
) -> np.ndarray:
    """Build per-row marginal features for Member 4 (the LogReg).

    Returns ``[N, MEMBER4_MARGINAL_FEATURE_DIM]`` float32. Column order
    is locked by :data:`MEMBER4_MARGINAL_FEATURE_NAMES`. These are the
    SOLE feature input for Member 4 in the post-Task-3 architecture --
    they share zero columns with Member 2's embedding-derived dense
    matrix, so the stacker sees a genuinely independent prediction.
    """
    N = int(np.asarray(subject_ids).shape[0])
    if int(np.asarray(cluster_ids).shape[0]) != N:
        raise ValueError("cluster_ids length must match subject_ids length")
    if int(np.asarray(bc_ids).shape[0]) != N:
        raise ValueError("bc_ids length must match subject_ids length")

    out = np.zeros((N, MEMBER4_MARGINAL_FEATURE_DIM), dtype=np.float32)
    subj_mean_per_row = _safe_gather_1d(stats.subj_mean, subject_ids, stats.global_mean)
    subj_log_n = _safe_gather_1d(stats.subj_log1p_n, subject_ids, 0.0)
    cluster_mean_per_row = _safe_gather_1d(
        stats.cluster_mean, cluster_ids, stats.global_mean
    )
    cluster_log_n = _safe_gather_1d(stats.cluster_log1p_n, cluster_ids, 0.0)
    bc_mean_per_row = _safe_gather_1d(stats.bc_mean, bc_ids, stats.global_mean)
    bc_log_n = _safe_gather_1d(stats.bc_log1p_n, bc_ids, 0.0)
    sbc_mean = _safe_gather_2d(
        stats.subj_bc_mean, subject_ids, bc_ids, bc_mean_per_row
    )
    sbc_log_n = _safe_gather_2d(
        stats.subj_bc_log1p_n, subject_ids, bc_ids, 0.0
    )
    sc_mean = _safe_gather_2d(
        stats.subj_cluster_mean, subject_ids, cluster_ids, cluster_mean_per_row
    )
    sc_log_n = _safe_gather_2d(
        stats.subj_cluster_log1p_n, subject_ids, cluster_ids, 0.0
    )

    g = float(stats.global_mean)
    out[:, 0] = subj_mean_per_row
    out[:, 1] = subj_log_n
    out[:, 2] = bc_mean_per_row
    out[:, 3] = bc_log_n
    out[:, 4] = cluster_mean_per_row
    out[:, 5] = cluster_log_n
    out[:, 6] = sbc_mean
    out[:, 7] = sbc_log_n
    out[:, 8] = sc_mean
    out[:, 9] = sc_log_n
    out[:, 10] = subj_mean_per_row - g
    out[:, 11] = bc_mean_per_row - g
    out[:, 12] = cluster_mean_per_row - g
    out[:, 13] = g
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


def apply_member2_interaction_features_one(
    stats: MeanEncodedStats,
    *,
    subject_id: int,
    cluster_id: int,
    bc_id: int,
) -> np.ndarray:
    """Single-row variant for the runtime path."""
    return apply_member2_interaction_features(
        stats,
        subject_ids=np.array([int(subject_id)], dtype=np.int64),
        cluster_ids=np.array([int(cluster_id)], dtype=np.int64),
        bc_ids=np.array([int(bc_id)], dtype=np.int64),
    ).reshape(-1)


def apply_member4_marginal_features_one(
    stats: MeanEncodedStats,
    *,
    subject_id: int,
    cluster_id: int,
    bc_id: int,
) -> np.ndarray:
    """Single-row variant for the runtime path."""
    return apply_member4_marginal_features(
        stats,
        subject_ids=np.array([int(subject_id)], dtype=np.int64),
        cluster_ids=np.array([int(cluster_id)], dtype=np.int64),
        bc_ids=np.array([int(bc_id)], dtype=np.int64),
    ).reshape(-1)


__all__ = [
    "MeanEncodedStats",
    "fit_mean_encoded_stats",
    "MEMBER2_INTERACTION_FEATURE_NAMES",
    "MEMBER2_INTERACTION_FEATURE_DIM",
    "MEMBER4_MARGINAL_FEATURE_NAMES",
    "MEMBER4_MARGINAL_FEATURE_DIM",
    "apply_member2_interaction_features",
    "apply_member4_marginal_features",
    "apply_member2_interaction_features_one",
    "apply_member4_marginal_features_one",
]
