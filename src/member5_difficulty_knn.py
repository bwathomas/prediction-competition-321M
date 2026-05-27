"""Member 5: kNN on a supervised 'predicted difficulty' projection.

Task 4 of the diversification plan. Member 3 (existing kNN) uses raw
Qwen item embeddings + cosine similarity to find SEMANTICALLY similar
items. Member 5 uses a 1-D supervised projection (ridge regression of
item-mean-passrate on item embedding) to find items with similar
PREDICTED DIFFICULTY -- a structurally different "neighborhood"
definition.

Items that are semantically similar but have different difficulty
(e.g., same topic, easy vs hard versions) are near in Member 3's
space but far in Member 5's, and vice versa. This decorrelation is
what lets the stacker extract complementary signal.

Pipeline:
  Fit step:
    1. Compute per-item mean pass-rate from training labels (per-fold
       for the OOF builds, full-train for the global ship).
    2. Fit ridge regression ``predicted_difficulty = emb @ w + b``,
       weighted by ``sqrt(obs_count)`` so high-coverage items dominate.
    3. Project every train item through the regression -> 1-D difficulty
       scalar. Sort items by predicted difficulty (kNN search becomes
       a binary-search on the sorted array, O(log N + K) per query).
    4. Store the sorted passrate matrix in subject-major order.

  Apply step (per row):
    1. Project the query item's embedding through the same w, b.
    2. Binary-search the sorted predicted-difficulty array; pull the
       K nearest items by 1-D distance.
    3. Aggregate the subject's pass-rate on those K neighbors with
       Gaussian-kernel weights ``exp(-distance / tau)``. Fall back
       to item_global_passrate (with discount) for unrated cells,
       then to subject_global, then to global_mean.

Pure NumPy at runtime; the only "fit-time" dependency is the ridge
solver (``np.linalg.solve`` on a ``[d_emb, d_emb]`` matrix). No
sklearn, no torch, no FAISS.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("member5")

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# State (the shipped artifact)
# ---------------------------------------------------------------------------


@dataclass
class Member5State:
    """Fitted-and-shipped state of Member 5 (difficulty-projected kNN)."""

    # --- Projection (predicted difficulty) ---
    projection_weights: np.ndarray  # [d_emb] fp32
    projection_bias: float
    projection_d_emb: int           # sanity check at apply time

    # --- Item difficulty space (sorted ascending) ---
    item_keys: tuple[str, ...]      # length n_items, in SORTED order
    predicted_difficulty: np.ndarray  # [n_items] fp32, sorted ascending
    # sort_order[i] = ORIGINAL index of the item now at sorted position i.
    # Provided so callers can re-key sidecar arrays into the sorted layout.
    sort_order: np.ndarray          # [n_items] int64

    # --- Passrate + masks (in SORTED-item order along axis 1) ---
    subject_keys: tuple[str, ...]   # length n_subjects
    passrate_sorted: np.ndarray     # [n_subjects, n_items] fp32 in (0, 1) or 0
    passrate_mask_sorted: np.ndarray  # [n_subjects, n_items] bool

    # --- Per-subject + per-item fallbacks ---
    subject_obs_count: np.ndarray   # [n_subjects] fp32
    subject_global: np.ndarray      # [n_subjects] fp32 (per-subject mean passrate)
    item_global_passrate_sorted: np.ndarray  # [n_items] fp32 (passrate over subjects)
    item_obs_count_sorted: np.ndarray        # [n_items] fp32
    global_mean: float

    # --- Hyperparams ---
    k: int
    tau: float
    item_fallback_weight: float
    min_subjects_per_item: int

    # --- Provenance / diagnostics ---
    n_train: int
    train_loss: float
    val_loss: float
    ridge_alpha: float

    def __post_init__(self) -> None:
        N = int(self.predicted_difficulty.shape[0])
        S = int(self.passrate_sorted.shape[0])
        if int(self.projection_weights.shape[0]) != int(self.projection_d_emb):
            raise ValueError(
                f"projection_weights dim {self.projection_weights.shape[0]} != "
                f"projection_d_emb {self.projection_d_emb}"
            )
        if not math.isfinite(float(self.projection_bias)):
            raise ValueError("projection_bias is NaN/Inf")
        if int(self.sort_order.shape[0]) != N:
            raise ValueError(
                f"sort_order len {self.sort_order.shape[0]} != N_items {N}"
            )
        if int(len(self.item_keys)) != N:
            raise ValueError(
                f"item_keys len {len(self.item_keys)} != N_items {N}"
            )
        if int(self.passrate_sorted.shape[1]) != N:
            raise ValueError(
                f"passrate_sorted cols {self.passrate_sorted.shape[1]} != N_items {N}"
            )
        if self.passrate_mask_sorted.shape != self.passrate_sorted.shape:
            raise ValueError(
                f"passrate_mask_sorted shape {self.passrate_mask_sorted.shape} != "
                f"passrate_sorted shape {self.passrate_sorted.shape}"
            )
        if int(self.subject_obs_count.shape[0]) != S:
            raise ValueError(
                f"subject_obs_count len {self.subject_obs_count.shape[0]} != S {S}"
            )
        if int(self.subject_global.shape[0]) != S:
            raise ValueError(
                f"subject_global len {self.subject_global.shape[0]} != S {S}"
            )
        if int(self.item_global_passrate_sorted.shape[0]) != N:
            raise ValueError(
                f"item_global_passrate_sorted len {self.item_global_passrate_sorted.shape[0]} != N {N}"
            )
        if int(self.item_obs_count_sorted.shape[0]) != N:
            raise ValueError(
                f"item_obs_count_sorted len {self.item_obs_count_sorted.shape[0]} != N {N}"
            )
        if int(len(self.subject_keys)) != S:
            raise ValueError(
                f"subject_keys len {len(self.subject_keys)} != S {S}"
            )
        if not np.all(np.diff(self.predicted_difficulty) >= -1e-6):
            raise ValueError(
                "predicted_difficulty must be sorted ascending; got "
                f"min_diff={float(np.diff(self.predicted_difficulty).min()):.6f}"
            )
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not (0.0 <= float(self.item_fallback_weight) <= 1.0):
            raise ValueError(
                f"item_fallback_weight {self.item_fallback_weight} not in [0, 1]"
            )

    @property
    def n_items(self) -> int:
        return int(self.predicted_difficulty.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(self.passrate_sorted.shape[0])

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "member5_state.npz",
            projection_weights=self.projection_weights.astype(np.float32),
            predicted_difficulty=self.predicted_difficulty.astype(np.float32),
            sort_order=self.sort_order.astype(np.int64),
            passrate_sorted=self.passrate_sorted.astype(np.float32),
            passrate_mask_sorted=self.passrate_mask_sorted.astype(bool),
            subject_obs_count=self.subject_obs_count.astype(np.float32),
            subject_global=self.subject_global.astype(np.float32),
            item_global_passrate_sorted=self.item_global_passrate_sorted.astype(np.float32),
            item_obs_count_sorted=self.item_obs_count_sorted.astype(np.float32),
        )
        meta = {
            "projection_bias": float(self.projection_bias),
            "projection_d_emb": int(self.projection_d_emb),
            "item_keys": list(self.item_keys),
            "subject_keys": list(self.subject_keys),
            "k": int(self.k),
            "tau": float(self.tau),
            "item_fallback_weight": float(self.item_fallback_weight),
            "min_subjects_per_item": int(self.min_subjects_per_item),
            "global_mean": float(self.global_mean),
            "n_train": int(self.n_train),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "ridge_alpha": float(self.ridge_alpha),
            "format_version": 1,
        }
        (out / "member5_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "Member5State":
        d = Path(in_dir)
        meta = json.loads((d / "member5_meta.json").read_text(encoding="utf-8"))
        with np.load(d / "member5_state.npz") as npz:
            return cls(
                projection_weights=npz["projection_weights"].astype(np.float32, copy=False),
                projection_bias=float(meta["projection_bias"]),
                projection_d_emb=int(meta["projection_d_emb"]),
                item_keys=tuple(meta["item_keys"]),
                predicted_difficulty=npz["predicted_difficulty"].astype(np.float32, copy=False),
                sort_order=npz["sort_order"].astype(np.int64, copy=False),
                subject_keys=tuple(meta["subject_keys"]),
                passrate_sorted=npz["passrate_sorted"].astype(np.float32, copy=False),
                passrate_mask_sorted=npz["passrate_mask_sorted"].astype(bool, copy=False),
                subject_obs_count=npz["subject_obs_count"].astype(np.float32, copy=False),
                subject_global=npz["subject_global"].astype(np.float32, copy=False),
                item_global_passrate_sorted=npz["item_global_passrate_sorted"].astype(np.float32, copy=False),
                item_obs_count_sorted=npz["item_obs_count_sorted"].astype(np.float32, copy=False),
                global_mean=float(meta["global_mean"]),
                k=int(meta["k"]),
                tau=float(meta["tau"]),
                item_fallback_weight=float(meta["item_fallback_weight"]),
                min_subjects_per_item=int(meta["min_subjects_per_item"]),
                n_train=int(meta["n_train"]),
                train_loss=float(meta["train_loss"]),
                val_loss=float(meta["val_loss"]),
                ridge_alpha=float(meta["ridge_alpha"]),
            )


# ---------------------------------------------------------------------------
# Projection fit (ridge regression of item-mean-passrate on item embedding)
# ---------------------------------------------------------------------------


def fit_difficulty_projection(
    *,
    item_embeddings: np.ndarray,    # [n_items, d_emb] fp32/fp64
    item_mean_passrate: np.ndarray,  # [n_items] fp32/fp64
    item_obs_count: np.ndarray,      # [n_items] fp32/fp64 (>= 0)
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Solve weighted ridge: minimize sum_i w_i (emb_i @ beta + b - y_i)^2 + alpha*||beta||^2.

    Weights ``w_i = sqrt(item_obs_count_i)``: items with more
    observations get a tighter target. Items with zero observations
    are dropped from the fit (they contribute no info to the
    projection).

    Returns
    -------
    (beta, bias)
        ``beta`` shape ``[d_emb]`` fp32; ``bias`` Python float.
        Use as ``predicted_difficulty = item_emb @ beta + bias``.
    """
    X = np.asarray(item_embeddings, dtype=np.float64)
    y = np.asarray(item_mean_passrate, dtype=np.float64).reshape(-1)
    c = np.asarray(item_obs_count, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"item_embeddings must be 2D, got shape {X.shape}")
    N, D = int(X.shape[0]), int(X.shape[1])
    if int(y.shape[0]) != N:
        raise ValueError(f"item_mean_passrate len {y.shape[0]} != N_items {N}")
    if int(c.shape[0]) != N:
        raise ValueError(f"item_obs_count len {c.shape[0]} != N_items {N}")
    if float(ridge_alpha) < 0:
        raise ValueError(f"ridge_alpha must be >= 0, got {ridge_alpha}")

    mask = c > 0
    if int(mask.sum()) < D:
        LOG.warning(
            "fit_difficulty_projection: only %d items with obs > 0 < D=%d; "
            "projection may be under-determined. Bumping ridge_alpha may help.",
            int(mask.sum()), D,
        )
    Xm, ym, cm = X[mask], y[mask], c[mask]
    weights = np.sqrt(cm)
    # Weighted normal equations with intercept:
    #   stack [Xm | 1] then solve weighted ridge.
    Xa = np.concatenate([Xm, np.ones((Xm.shape[0], 1), dtype=np.float64)], axis=1)
    W = weights.reshape(-1, 1)
    XaW = Xa * W
    yW = ym * weights
    A = XaW.T @ XaW
    b = XaW.T @ yW
    # Ridge: penalize the slope coefficients but not the bias (column D).
    reg = float(ridge_alpha) * np.eye(D + 1, dtype=np.float64)
    reg[D, D] = 0.0
    A += reg
    coef = np.linalg.solve(A, b)
    beta = coef[:D].astype(np.float32, copy=False)
    bias = float(coef[D])
    LOG.info(
        "fit_difficulty_projection: D=%d  n_items_used=%d/%d  "
        "ridge_alpha=%.3g  ||beta||=%.4f  bias=%.4f",
        D, int(mask.sum()), N, float(ridge_alpha),
        float(np.linalg.norm(beta)), bias,
    )
    return beta, bias


# ---------------------------------------------------------------------------
# Per-item passrate aggregation (utility shared with knn_member style)
# ---------------------------------------------------------------------------


def _derive_aggregates_from_dense(
    passrate_dense: np.ndarray,
    passrate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Memory-efficient: derive per-item / per-subject / global stats
    from already-built ``[S, N]`` passrate + mask matrices.

    The per-fold notebook ALREADY builds these matrices for Member 3.
    Re-aggregating them from rows inside ``fit_member5`` would allocate
    another ``[S, N]`` cell-count matrix and a ``[S, N]`` safe-divide
    intermediate (~1.4 GB at S=900, N=200k) -- enough to OOM the
    high-RAM Colab box on fold 0. This helper extracts the stats with
    O(N) auxiliary memory (per-row/column sums), no extra full
    matrices allocated.

    Returns
    -------
    item_mean_passrate : [N] fp32   -- per-item mean over rated subjects
    item_obs_count     : [N] fp32   -- per-item count of rated subjects
    subject_global     : [S] fp32   -- per-subject mean over rated items
    subject_obs_count  : [S] fp32   -- per-subject count of rated items
    global_mean        : Python float
    """
    pd = passrate_dense
    pm = passrate_mask
    if pd.ndim != 2 or pm.ndim != 2:
        raise ValueError(
            f"passrate_dense / passrate_mask must be 2D, got "
            f"{pd.shape} / {pm.shape}"
        )
    if pd.shape != pm.shape:
        raise ValueError(
            f"passrate_dense shape {pd.shape} != passrate_mask shape {pm.shape}"
        )

    # Sum across the subjects axis (axis=0) -> per-item totals.
    # ``dtype=np.float64`` in the accumulator avoids a full-precision
    # intermediate matrix while still keeping numerical accuracy.
    item_sum = pd.sum(axis=0, dtype=np.float64)        # [N]
    item_cnt = pm.sum(axis=0, dtype=np.float64)        # [N]
    safe_item = np.where(item_cnt > 0, item_cnt, 1.0)
    item_mean = (item_sum / safe_item).astype(np.float32)
    item_mean[item_cnt == 0] = 0.0

    # Sum across the items axis (axis=1) -> per-subject totals.
    subj_sum = pd.sum(axis=1, dtype=np.float64)        # [S]
    subj_cnt = pm.sum(axis=1, dtype=np.float64)        # [S]
    safe_subj = np.where(subj_cnt > 0, subj_cnt, 1.0)
    subj_global = (subj_sum / safe_subj).astype(np.float32)
    subj_global[subj_cnt == 0] = 0.0

    total_sum = float(item_sum.sum())
    total_cnt = float(item_cnt.sum())
    global_mean = float(total_sum / total_cnt) if total_cnt > 0 else 0.5

    return (
        item_mean,
        item_cnt.astype(np.float32),
        subj_global,
        subj_cnt.astype(np.float32),
        global_mean,
    )


def aggregate_per_item_passrate(
    *,
    subject_ids: np.ndarray,     # [N_rows] int (>= 0)
    item_ids: np.ndarray,        # [N_rows] int (>= 0)
    labels: np.ndarray,          # [N_rows] float in {0, 1}
    n_subjects: int,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Group-by aggregations needed by both the projection fit AND the
    passrate sidecar tables.

    Returns
    -------
    item_mean_passrate : [n_items] fp32
    item_obs_count     : [n_items] fp32
    subject_global     : [n_subjects] fp32
    subject_obs_count  : [n_subjects] fp32
    passrate_dense     : [n_subjects, n_items] fp32
    passrate_mask      : [n_subjects, n_items] bool
    global_mean        : Python float
    """
    s = np.asarray(subject_ids, dtype=np.int64)
    i = np.asarray(item_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    if not (s.shape == i.shape == y.shape):
        raise ValueError(
            f"shapes mismatch: subject_ids={s.shape} item_ids={i.shape} labels={y.shape}"
        )
    valid = (s >= 0) & (i >= 0) & (s < int(n_subjects)) & (i < int(n_items))
    s, i, y = s[valid], i[valid], y[valid]

    item_sum = np.bincount(i, weights=y, minlength=int(n_items)).astype(np.float64)
    item_cnt = np.bincount(i, minlength=int(n_items)).astype(np.float64)
    safe_item_cnt = np.where(item_cnt > 0, item_cnt, 1.0)
    item_mean = (item_sum / safe_item_cnt).astype(np.float32)
    item_mean[item_cnt == 0] = 0.0  # downstream caller should mask via obs_count

    subj_sum = np.bincount(s, weights=y, minlength=int(n_subjects)).astype(np.float64)
    subj_cnt = np.bincount(s, minlength=int(n_subjects)).astype(np.float64)
    safe_subj_cnt = np.where(subj_cnt > 0, subj_cnt, 1.0)
    subj_global = (subj_sum / safe_subj_cnt).astype(np.float32)
    subj_global[subj_cnt == 0] = 0.0

    gm = float(y.mean()) if y.size > 0 else 0.5

    # Dense passrate table -- [S, N] -- via bucket assign. For large
    # (S, N) this is the dominant memory cost; the notebook should
    # already be passing reasonable dimensions (~1k subjects x ~300k items
    # ~ 1.2 GB fp32). For Member 5 we keep this compatible with the
    # knn_member layout so downstream tooling can reuse helpers.
    passrate_dense = np.zeros((int(n_subjects), int(n_items)), dtype=np.float32)
    passrate_mask = np.zeros((int(n_subjects), int(n_items)), dtype=bool)
    # Use add.at for the duplicate-key correct aggregation.
    np.add.at(passrate_dense, (s, i), y.astype(np.float32))
    np.add.at(passrate_mask.view(np.uint8), (s, i), np.uint8(1))
    # Normalize by count per (s, i) cell so the value is a true mean.
    cell_cnt = np.zeros_like(passrate_dense)
    np.add.at(cell_cnt, (s, i), 1.0)
    safe = np.where(cell_cnt > 0, cell_cnt, 1.0)
    passrate_dense = passrate_dense / safe
    passrate_mask = cell_cnt > 0

    return (
        item_mean,
        item_cnt.astype(np.float32),
        subj_global,
        subj_cnt.astype(np.float32),
        passrate_dense,
        passrate_mask,
        gm,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def fit_member5(
    *,
    item_keys: Sequence[str],
    item_embeddings: np.ndarray,        # [n_items, d_emb] fp32
    subject_keys: Sequence[str],
    subject_ids_per_row: np.ndarray,    # [N_rows] int >= 0
    item_ids_per_row: np.ndarray,       # [N_rows] int >= 0 (indices into item_keys)
    labels: np.ndarray,                 # [N_rows] float in {0, 1}
    k: int = 32,
    tau: float = 0.05,
    ridge_alpha: float = 1.0,
    item_fallback_weight: float = 0.3,
    min_subjects_per_item: int = 3,
    passrate_dense: np.ndarray | None = None,
    passrate_mask: np.ndarray | None = None,
) -> Member5State:
    """End-to-end Member 5 fit.

    Parameters
    ----------
    item_keys, item_embeddings
        Aligned: ``item_embeddings[i]`` is the embedding for
        ``item_keys[i]``. Must include EVERY item that appears in
        ``item_ids_per_row``.
    subject_keys
        Length = ``n_subjects``. Position i = subject_id i (consistent
        with ``Indexer`` ordering).
    subject_ids_per_row / item_ids_per_row / labels
        Aligned ``[N_rows]`` arrays giving the training observations.
        Ignored when ``passrate_dense`` and ``passrate_mask`` are
        provided (fast path); only the ``[N_rows]`` shape is consulted
        for the train-loss sample size.
    k
        kNN neighborhood size for the difficulty-axis search.
    tau
        Gaussian kernel scale on the predicted-difficulty axis
        (``sim = exp(-distance / tau)``). Larger tau = flatter kernel,
        more averaging.
    ridge_alpha
        L2 regularization for the projection fit.
    item_fallback_weight
        Same role as in Member 3: when a neighbor cell is unobserved,
        fall back to the neighbor's global passrate with this discount.
    min_subjects_per_item
        Items with fewer rated subjects than this are excluded from
        the difficulty-projection fit (their item_mean_passrate is
        too noisy).
    passrate_dense, passrate_mask
        Optional. ``[n_subjects, n_items]`` matrices in the SAME
        item ordering as ``item_keys``. When provided, ``fit_member5``
        skips the internal ``aggregate_per_item_passrate`` call and
        derives all aggregates directly from these matrices via
        :func:`_derive_aggregates_from_dense`. This is the recommended
        path when the caller already built a passrate matrix (e.g.
        for Member 3) -- it avoids a ~3 GB peak of duplicated
        ``[S, N]`` intermediates that have caused OOMs on Colab
        high-RAM during per-fold OOF training.

        Both must be passed together. ``passrate_dense`` is float
        (interpreted in (0, 1) for observed cells, 0 elsewhere);
        ``passrate_mask`` is bool (True for observed cells). Other
        dtypes are accepted with an automatic cast.
    """
    item_keys_list = list(str(k) for k in item_keys)
    subject_keys_list = list(str(s) for s in subject_keys)
    n_items = len(item_keys_list)
    n_subjects = len(subject_keys_list)
    if item_embeddings.shape != (n_items, item_embeddings.shape[1]):
        raise ValueError(
            f"item_embeddings rows {item_embeddings.shape[0]} != "
            f"item_keys len {n_items}"
        )
    d_emb = int(item_embeddings.shape[1])

    if (passrate_dense is None) ^ (passrate_mask is None):
        raise ValueError(
            "passrate_dense and passrate_mask must be passed together "
            "(both or neither)."
        )

    if passrate_dense is not None and passrate_mask is not None:
        # Fast path: derive per-item / per-subject / global stats from
        # the pre-built dense matrices instead of re-aggregating from
        # rows. Saves ~3 GB of [S, N] intermediates that the rows->dense
        # add.at path used to allocate; this is what tripped the OOM
        # in Section 9.5 fold 0.
        passrate_dense = np.ascontiguousarray(passrate_dense, dtype=np.float32)
        passrate_mask = np.ascontiguousarray(passrate_mask).astype(bool, copy=False)
        if int(passrate_dense.shape[0]) != n_subjects:
            raise ValueError(
                f"passrate_dense rows {passrate_dense.shape[0]} != "
                f"n_subjects {n_subjects}"
            )
        if int(passrate_dense.shape[1]) != n_items:
            raise ValueError(
                f"passrate_dense cols {passrate_dense.shape[1]} != "
                f"n_items {n_items}"
            )
        (
            item_mean_passrate, item_obs_count,
            subj_global, subj_obs_count, global_mean,
        ) = _derive_aggregates_from_dense(passrate_dense, passrate_mask)
        LOG.info(
            "fit_member5: using pre-built passrate (S=%d, N=%d, "
            "observed=%d cells); skipping row-level aggregation.",
            int(passrate_dense.shape[0]), int(passrate_dense.shape[1]),
            int(passrate_mask.sum()),
        )
    else:
        (
            item_mean_passrate, item_obs_count,
            subj_global, subj_obs_count,
            passrate_dense, passrate_mask,
            global_mean,
        ) = aggregate_per_item_passrate(
            subject_ids=subject_ids_per_row,
            item_ids=item_ids_per_row,
            labels=labels,
            n_subjects=n_subjects,
            n_items=n_items,
        )

    # For the projection fit we restrict to items with enough rated
    # subjects (the per-item mean is too noisy for tiny support).
    proj_fit_mask = (item_obs_count >= float(min_subjects_per_item)).astype(bool)
    if int(proj_fit_mask.sum()) < 10:
        raise RuntimeError(
            f"fit_member5: only {int(proj_fit_mask.sum())} items have "
            f">={min_subjects_per_item} subject ratings; cannot fit a "
            "meaningful projection."
        )
    beta, bias = fit_difficulty_projection(
        item_embeddings=np.asarray(item_embeddings[proj_fit_mask], dtype=np.float32),
        item_mean_passrate=item_mean_passrate[proj_fit_mask],
        item_obs_count=item_obs_count[proj_fit_mask],
        ridge_alpha=ridge_alpha,
    )
    if int(beta.shape[0]) != d_emb:
        raise RuntimeError(
            f"projection beta dim {beta.shape[0]} != d_emb {d_emb}"
        )

    # Project every item -> 1-D predicted difficulty.
    predicted = (np.asarray(item_embeddings, dtype=np.float64) @ beta.astype(np.float64)
                 + float(bias)).astype(np.float32)

    # Sort items ascending by predicted difficulty. The sorted layout
    # lets the apply path binary-search for K nearest in O(log N + K).
    sort_order = np.argsort(predicted, kind="stable").astype(np.int64)
    predicted_sorted = predicted[sort_order]
    item_keys_sorted = tuple(item_keys_list[int(j)] for j in sort_order)
    passrate_dense_sorted = passrate_dense[:, sort_order]
    passrate_mask_sorted = passrate_mask[:, sort_order]
    item_global_sorted = item_mean_passrate[sort_order]
    item_obs_count_sorted = item_obs_count[sort_order]

    # Manual NLL on training rows -- equivalent of Member 3's
    # train_loss. We score each training row through apply_one's
    # composition function so the NLL reflects what the runtime
    # will actually emit. Sub-sampled for speed.
    #
    # Skipped entirely on the "fast path" where the caller passed
    # only pre-built passrate matrices (subject_ids_per_row/
    # item_ids_per_row/labels are length-0 sentinels). Without that
    # skip, np.random.choice(0, size=0) -> empty sample, then
    # np.mean(empty) -> nan + RuntimeWarning("Mean of empty slice")
    # and the state ends up with ``train_loss=nan`` cached forever.
    n_rows = int(subject_ids_per_row.shape[0])
    if n_rows == 0:
        nll = 0.0
        sample_n = 0
        LOG.info(
            "fit_member5: n_items=%d  n_subjects=%d  K=%d  tau=%.3f  "
            "ridge_alpha=%.3g  pred_diff range=[%.4f, %.4f]  "
            "sampled_train_nll skipped (fast path: row arrays empty)",
            n_items, n_subjects, int(k), float(tau), float(ridge_alpha),
            float(predicted_sorted.min()), float(predicted_sorted.max()),
        )
    else:
        rng = np.random.default_rng(0)
        sample_n = int(min(10000, n_rows))
        sample_idx = rng.choice(n_rows, size=sample_n, replace=False)
        state_preview = Member5State(
            projection_weights=beta,
            projection_bias=bias,
            projection_d_emb=d_emb,
            item_keys=item_keys_sorted,
            predicted_difficulty=predicted_sorted,
            sort_order=sort_order,
            subject_keys=tuple(subject_keys_list),
            passrate_sorted=passrate_dense_sorted,
            passrate_mask_sorted=passrate_mask_sorted,
            subject_obs_count=subj_obs_count,
            subject_global=subj_global,
            item_global_passrate_sorted=item_global_sorted,
            item_obs_count_sorted=item_obs_count_sorted,
            global_mean=global_mean,
            k=int(k),
            tau=float(tau),
            item_fallback_weight=float(item_fallback_weight),
            min_subjects_per_item=int(min_subjects_per_item),
            n_train=int(n_rows),
            train_loss=0.0,
            val_loss=0.0,
            ridge_alpha=float(ridge_alpha),
        )
        sub_subj = subject_ids_per_row[sample_idx]
        sub_emb = item_embeddings[item_ids_per_row[sample_idx]]
        sub_y = labels[sample_idx]
        p_sample = apply_batch_via_ids(
            state_preview,
            subject_ids=sub_subj,
            query_item_embeddings=sub_emb,
        )
        nll = -float(np.mean(
            sub_y * np.log(np.clip(p_sample, _EPS, 1.0 - _EPS))
            + (1 - sub_y) * np.log(1 - np.clip(p_sample, _EPS, 1.0 - _EPS))
        ))

        LOG.info(
            "fit_member5: n_items=%d  n_subjects=%d  K=%d  tau=%.3f  "
            "ridge_alpha=%.3g  pred_diff range=[%.4f, %.4f]  "
            "sampled_train_nll(%d rows)=%.5f",
            n_items, n_subjects, int(k), float(tau), float(ridge_alpha),
            float(predicted_sorted.min()), float(predicted_sorted.max()),
            sample_n, nll,
        )

    return Member5State(
        projection_weights=beta,
        projection_bias=bias,
        projection_d_emb=d_emb,
        item_keys=item_keys_sorted,
        predicted_difficulty=predicted_sorted,
        sort_order=sort_order,
        subject_keys=tuple(subject_keys_list),
        passrate_sorted=passrate_dense_sorted,
        passrate_mask_sorted=passrate_mask_sorted,
        subject_obs_count=subj_obs_count,
        subject_global=subj_global,
        item_global_passrate_sorted=item_global_sorted,
        item_obs_count_sorted=item_obs_count_sorted,
        global_mean=global_mean,
        k=int(k),
        tau=float(tau),
        item_fallback_weight=float(item_fallback_weight),
        min_subjects_per_item=int(min_subjects_per_item),
        n_train=int(subject_ids_per_row.shape[0]),
        train_loss=float(nll),
        val_loss=0.0,
        ridge_alpha=float(ridge_alpha),
    )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _project(state: Member5State, query_emb: np.ndarray) -> np.ndarray:
    """Project a batch of query embeddings to 1-D predicted difficulty."""
    q = np.asarray(query_emb, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if int(q.shape[1]) != int(state.projection_d_emb):
        raise ValueError(
            f"query_emb dim {q.shape[1]} != state.projection_d_emb "
            f"{state.projection_d_emb}"
        )
    return (q.astype(np.float64) @ state.projection_weights.astype(np.float64)
            + float(state.projection_bias)).astype(np.float32)


def _knearest_sorted(
    sorted_arr: np.ndarray, query: float, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find K positions in ``sorted_arr`` closest to ``query`` value.

    Returns ``(positions[k], distances[k])`` sorted by ascending distance.
    Uses bisect + expand to grab the K nearest in O(log N + K).
    """
    N = int(sorted_arr.shape[0])
    if N == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    K = int(min(k, N))
    pos = int(np.searchsorted(sorted_arr, query))
    lo, hi = pos - 1, pos
    out_idx: list[int] = []
    while len(out_idx) < K:
        if lo < 0 and hi >= N:
            break
        if lo < 0:
            out_idx.append(hi); hi += 1
        elif hi >= N:
            out_idx.append(lo); lo -= 1
        else:
            if abs(float(sorted_arr[hi]) - query) <= abs(float(sorted_arr[lo]) - query):
                out_idx.append(hi); hi += 1
            else:
                out_idx.append(lo); lo -= 1
    idx_arr = np.array(out_idx, dtype=np.int64)
    dist_arr = np.abs(sorted_arr[idx_arr].astype(np.float64) - float(query)).astype(np.float32)
    # Re-sort by distance ascending.
    order = np.argsort(dist_arr, kind="stable")
    return idx_arr[order], dist_arr[order]


def apply_one(
    state: Member5State, query_emb: np.ndarray, subject_key: str,
) -> float:
    """Single-row Member 5 prediction.

    Pure numpy + python; no torch/sklearn/FAISS.
    """
    # Look up subject id (linear in S; runtime can pre-build a dict
    # outside of this function if hot-pathed).
    try:
        s_id = state.subject_keys.index(str(subject_key))
    except ValueError:
        s_id = -1

    # Project query to predicted difficulty.
    pd_q = float(_project(state, query_emb)[0])
    nbr_idx, nbr_dist = _knearest_sorted(
        state.predicted_difficulty, pd_q, int(state.k),
    )
    if nbr_idx.size == 0:
        return float(min(max(state.global_mean, _EPS), 1.0 - _EPS))

    # Gaussian kernel weights on the 1-D difficulty axis.
    w = np.exp(-nbr_dist.astype(np.float64) / max(float(state.tau), 1e-9))
    # Subject-conditioned aggregation: subject's passrate on neighbors
    # if observed; else item_global * item_fallback_weight; else skip.
    if s_id >= 0:
        subj_obs = state.passrate_mask_sorted[s_id, nbr_idx].astype(bool)
        subj_pr = state.passrate_sorted[s_id, nbr_idx].astype(np.float64)
    else:
        subj_obs = np.zeros(nbr_idx.size, dtype=bool)
        subj_pr = np.zeros(nbr_idx.size, dtype=np.float64)
    item_pr = state.item_global_passrate_sorted[nbr_idx].astype(np.float64)
    item_ok = (state.item_obs_count_sorted[nbr_idx]
               >= float(state.min_subjects_per_item)).astype(bool)

    num = np.where(subj_obs, w * subj_pr, 0.0)
    den = np.where(subj_obs, w, 0.0)
    # Fallback contribution.
    fb_mask = (~subj_obs) & item_ok
    fb_w = float(state.item_fallback_weight) * w
    num = num + np.where(fb_mask, fb_w * item_pr, 0.0)
    den = den + np.where(fb_mask, fb_w, 0.0)

    if float(den.sum()) > 0.0:
        p = float(num.sum() / den.sum())
    elif s_id >= 0 and float(state.subject_obs_count[s_id]) > 0:
        p = float(state.subject_global[s_id])
    else:
        p = float(state.global_mean)
    return float(min(max(p, _EPS), 1.0 - _EPS))


def apply_batch_via_ids(
    state: Member5State,
    *,
    subject_ids: np.ndarray,
    query_item_embeddings: np.ndarray,
) -> np.ndarray:
    """Batch inference path used at training/val time.

    Takes pre-resolved subject ids (vs apply_one which resolves
    subject_key strings). Useful when N is large and the caller has
    already done the keyset lookup once.

    Returns ``[N]`` float32 in (eps, 1-eps).
    """
    sids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    qe = np.asarray(query_item_embeddings, dtype=np.float32)
    if qe.ndim != 2:
        raise ValueError(f"query_item_embeddings must be 2D, got {qe.shape}")
    if int(sids.shape[0]) != int(qe.shape[0]):
        raise ValueError(
            f"subject_ids len {sids.shape[0]} != query_item_embeddings rows {qe.shape[0]}"
        )
    N = int(sids.shape[0])
    if N == 0:
        return np.empty(0, dtype=np.float32)
    # Project all queries.
    pd_q = _project(state, qe).astype(np.float64)
    out = np.empty(N, dtype=np.float64)
    for r in range(N):
        nbr_idx, nbr_dist = _knearest_sorted(
            state.predicted_difficulty, float(pd_q[r]), int(state.k),
        )
        if nbr_idx.size == 0:
            out[r] = float(state.global_mean)
            continue
        w = np.exp(-nbr_dist.astype(np.float64) / max(float(state.tau), 1e-9))
        s_id = int(sids[r])
        if 0 <= s_id < state.n_subjects:
            subj_obs = state.passrate_mask_sorted[s_id, nbr_idx].astype(bool)
            subj_pr = state.passrate_sorted[s_id, nbr_idx].astype(np.float64)
        else:
            subj_obs = np.zeros(nbr_idx.size, dtype=bool)
            subj_pr = np.zeros(nbr_idx.size, dtype=np.float64)
        item_pr = state.item_global_passrate_sorted[nbr_idx].astype(np.float64)
        item_ok = (state.item_obs_count_sorted[nbr_idx]
                   >= float(state.min_subjects_per_item)).astype(bool)
        num = np.where(subj_obs, w * subj_pr, 0.0)
        den = np.where(subj_obs, w, 0.0)
        fb_mask = (~subj_obs) & item_ok
        fb_w = float(state.item_fallback_weight) * w
        num = num + np.where(fb_mask, fb_w * item_pr, 0.0)
        den = den + np.where(fb_mask, fb_w, 0.0)
        if float(den.sum()) > 0:
            out[r] = float(num.sum() / den.sum())
        elif 0 <= s_id < state.n_subjects and float(state.subject_obs_count[s_id]) > 0:
            out[r] = float(state.subject_global[s_id])
        else:
            out[r] = float(state.global_mean)
    return np.clip(out, _EPS, 1.0 - _EPS).astype(np.float32, copy=False)


def apply_batch(
    state: Member5State,
    *,
    subject_keys: Sequence[str],
    query_item_embeddings: np.ndarray,
) -> np.ndarray:
    """Batch inference via subject KEY strings.

    Convenience wrapper: builds a subject_key -> id lookup once, then
    calls :func:`apply_batch_via_ids`.
    """
    skl = list(state.subject_keys)
    s_to_id = {s: i for i, s in enumerate(skl)}
    sids = np.array(
        [int(s_to_id.get(str(s), -1)) for s in subject_keys],
        dtype=np.int64,
    )
    return apply_batch_via_ids(
        state, subject_ids=sids, query_item_embeddings=query_item_embeddings,
    )


# ---------------------------------------------------------------------------
# Gate 4c helper: projection-leakage probe
# ---------------------------------------------------------------------------


def assert_projection_disjoint_from_val(
    *,
    fit_item_keys: Sequence[str],
    val_item_keys: Sequence[str],
) -> dict:
    """RED-TEAM GATE 4c: assert the projection's fit set has zero
    overlap with the val item set.

    Raises AssertionError on any overlap; returns a diagnostic dict
    on success.
    """
    fit_set = set(str(k) for k in fit_item_keys)
    val_set = set(str(k) for k in val_item_keys)
    overlap = fit_set & val_set
    if overlap:
        sample = list(overlap)[:5]
        raise AssertionError(
            f"GATE 4c violation: {len(overlap):,} of {len(val_set):,} val "
            f"items were also in the projection's fit set "
            f"(e.g. {sample}). The supervised projection must NEVER see "
            "val items' labels."
        )
    return {
        "n_fit_items": int(len(fit_set)),
        "n_val_items": int(len(val_set)),
        "n_overlap": 0,
    }


__all__ = [
    "Member5State",
    "fit_difficulty_projection",
    "aggregate_per_item_passrate",
    "fit_member5",
    "apply_one",
    "apply_batch",
    "apply_batch_via_ids",
    "assert_projection_disjoint_from_val",
]


# Re-export the dense-matrix helper as a public API so notebook/scripts
# can use it for tests or to short-circuit the aggregation in custom flows.
__all__.append("_derive_aggregates_from_dense")
