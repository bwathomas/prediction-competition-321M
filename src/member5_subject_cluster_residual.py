"""Member 5 v2: subject x item-cluster residual passrate.

This is a structurally different replacement for the difficulty-projected
kNN Member 5. The goal is *inductive bias diversity* relative to Members
1/3/4 (deep MLP / cosine kNN / linear LogReg).

Model
-----
For each row (subject_id ``s``, item_cluster_id ``c``):

    p(y=1 | s, c) = sigmoid(
        subj_logit[s] + cluster_logit[c] - global_logit
        + residual_scale * residual_logit[s, c]
    )

where:

* ``global_logit = logit(mean(y_train))`` (single scalar prior)
* ``subj_logit[s] = logit(bayes_shrunk(subj_mean[s], global_mean, smoothing_marginal))``
* ``cluster_logit[c] = logit(bayes_shrunk(cluster_mean[c], global_mean, smoothing_marginal))``
* ``residual_logit[s, c] = cell_logit[s, c] - subj_logit[s] - cluster_logit[c] + global_logit``
  with cell_logit derived from the Bayesian-shrunk per-(s, c) cell mean
  using a stronger ``smoothing_cell``. The residual is then itself
  attenuated by per-cell support so cells with very few observations
  contribute only a fraction of their nominal residual.

Why this design
---------------
Member 1 (IRT-MLP) and Member 3 (cosine kNN on embeddings) both predict
through the continuous embedding manifold. Member 4 (LogReg) is strictly
linear-additive in dense engineered features. None of them can express
a *categorical* subject x item-cluster *interaction* on its own:

* M1 only reaches it via deep nonlinearities of the embedding.
* M3 reaches it implicitly via cluster geometry, but every neighbor's
  contribution is smoothed across the embedding metric.
* M4 has subject means and cluster means as separate marginals, but its
  additive form cannot multiply them together.

This member is a *pure lookup table* in logit space: zero learning, all
inference, completely categorical, completely orthogonal to embedding-
based predictors. The hope is that ``residual_logit[s, c]`` carries the
"subject s does unexpectedly well / poorly on cluster c" signal that
the other members structurally cannot.

Cold-start handling
-------------------
* Unknown subject (``s < 0`` or ``s >= n_subjects``) -> predict the
  cluster's marginal (``sigmoid(cluster_logit[c])``), or the global
  mean if the cluster is also unknown.
* Unknown cluster (``c < 0`` or ``c >= n_clusters``) -> predict the
  subject's marginal (``sigmoid(subj_logit[s])``), or the global mean.
* Both known but the (s, c) cell was never observed -> the Bayesian
  shrinkage of ``cell_logit`` pulls the residual term toward 0, so the
  prediction degenerates to ``sigmoid(subj_logit[s] + cluster_logit[c]
  - global_logit)`` which is the additive prediction.

Runtime contract
----------------
Pure numpy at inference. ``apply_state_batch(state, subject_ids,
cluster_ids)`` -> [N] float32 probabilities. ``apply_state_one(state,
sid, cid)`` -> python float in (eps, 1-eps).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("member5_residual")

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _logit_safe(p: float | np.ndarray, eps: float = _EPS) -> float | np.ndarray:
    """Numerically stable logit. Inputs are clipped to [eps, 1-eps]."""
    if isinstance(p, (int, float)):
        x = max(min(float(p), 1.0 - eps), eps)
        return math.log(x / (1.0 - x))
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def _sigmoid_stable(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid for arrays and scalars."""
    if isinstance(z, (int, float)):
        zf = float(z)
        if zf >= 0:
            return 1.0 / (1.0 + math.exp(-zf))
        e = math.exp(zf)
        return e / (1.0 + e)
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    neg = ~pos
    e = np.exp(z_arr[neg])
    out[neg] = e / (1.0 + e)
    return out


def _bincount1d_sum_and_count(
    ids: np.ndarray,
    values: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D bucketed sum and count with negative/out-of-range filtering."""
    iid = np.asarray(ids, dtype=np.int64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if iid.shape != v.shape:
        raise ValueError(f"ids {iid.shape} != values {v.shape}")
    keep = (iid >= 0) & (iid < int(n_buckets))
    if not bool(keep.all()):
        iid = iid[keep]
        v = v[keep]
    sum_y = np.bincount(iid, weights=v, minlength=int(n_buckets))
    cnt = np.bincount(iid, minlength=int(n_buckets)).astype(np.float64)
    return sum_y, cnt


def _bincount2d_sum_and_count(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    values: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense 2-D bucketed sum and count, negative ids filtered."""
    rid = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    cid = np.asarray(col_ids, dtype=np.int64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if rid.shape != cid.shape or rid.shape != v.shape:
        raise ValueError(
            f"shape mismatch: rows {rid.shape}, cols {cid.shape}, "
            f"values {v.shape}"
        )
    keep = (
        (rid >= 0) & (rid < int(n_rows))
        & (cid >= 0) & (cid < int(n_cols))
    )
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


def _bayes_shrunk(
    sum_y: np.ndarray,
    n: np.ndarray,
    fallback_mean: float | np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Bayesian shrinkage; empty cells collapse to ``fallback_mean``."""
    smoothing_f = float(smoothing)
    if isinstance(fallback_mean, (int, float)):
        fm = float(fallback_mean)
    else:
        fm = np.asarray(fallback_mean, dtype=np.float64)
    return (sum_y + smoothing_f * fm) / np.maximum(n + smoothing_f, 1.0e-12)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Member5ResidualState:
    """Fitted state for Member 5 v2 (subject x cluster residual passrate).

    Pure lookup tables in logit space; no learnable parameters at apply
    time. All arrays are float32 to keep the shipped artifact compact.
    """

    global_mean_logit: float
    subj_logit: np.ndarray            # [n_subjects] float32
    cluster_logit: np.ndarray         # [n_clusters]  float32
    residual_logit: np.ndarray        # [n_subjects, n_clusters] float32
    cell_log1p_n: np.ndarray          # [n_subjects, n_clusters] float32 (diag)
    subject_keys: tuple[str, ...]
    n_subjects: int
    n_clusters: int
    smoothing_cell: float
    smoothing_marginal: float
    residual_scale: float
    n_train: int
    train_loss: float
    val_loss: float

    def __post_init__(self) -> None:
        if int(self.subj_logit.shape[0]) != int(self.n_subjects):
            raise ValueError(
                f"subj_logit len {self.subj_logit.shape[0]} != "
                f"n_subjects {self.n_subjects}"
            )
        if int(self.cluster_logit.shape[0]) != int(self.n_clusters):
            raise ValueError(
                f"cluster_logit len {self.cluster_logit.shape[0]} != "
                f"n_clusters {self.n_clusters}"
            )
        if self.residual_logit.shape != (int(self.n_subjects), int(self.n_clusters)):
            raise ValueError(
                f"residual_logit shape {self.residual_logit.shape} != "
                f"({self.n_subjects}, {self.n_clusters})"
            )
        if self.cell_log1p_n.shape != self.residual_logit.shape:
            raise ValueError(
                f"cell_log1p_n shape {self.cell_log1p_n.shape} != "
                f"residual_logit {self.residual_logit.shape}"
            )
        if int(len(self.subject_keys)) != int(self.n_subjects):
            raise ValueError(
                f"subject_keys len {len(self.subject_keys)} != "
                f"n_subjects {self.n_subjects}"
            )
        if not math.isfinite(float(self.global_mean_logit)):
            raise ValueError("global_mean_logit is NaN/Inf")
        for name, arr in (
            ("subj_logit", self.subj_logit),
            ("cluster_logit", self.cluster_logit),
            ("residual_logit", self.residual_logit),
            ("cell_log1p_n", self.cell_log1p_n),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Member5ResidualState: {name} contains NaN/Inf")

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "weights.npz",
            subj_logit=self.subj_logit.astype(np.float32),
            cluster_logit=self.cluster_logit.astype(np.float32),
            residual_logit=self.residual_logit.astype(np.float32),
            cell_log1p_n=self.cell_log1p_n.astype(np.float32),
        )
        meta = {
            "global_mean_logit": float(self.global_mean_logit),
            "subject_keys": list(self.subject_keys),
            "n_subjects": int(self.n_subjects),
            "n_clusters": int(self.n_clusters),
            "smoothing_cell": float(self.smoothing_cell),
            "smoothing_marginal": float(self.smoothing_marginal),
            "residual_scale": float(self.residual_scale),
            "n_train": int(self.n_train),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "format_version": 1,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "Member5ResidualState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz") as npz:
            subj_logit = npz["subj_logit"].astype(np.float32, copy=False)
            cluster_logit = npz["cluster_logit"].astype(np.float32, copy=False)
            residual_logit = npz["residual_logit"].astype(np.float32, copy=False)
            cell_log1p_n = npz["cell_log1p_n"].astype(np.float32, copy=False)
        return cls(
            global_mean_logit=float(meta["global_mean_logit"]),
            subj_logit=subj_logit,
            cluster_logit=cluster_logit,
            residual_logit=residual_logit,
            cell_log1p_n=cell_log1p_n,
            subject_keys=tuple(str(s) for s in meta["subject_keys"]),
            n_subjects=int(meta["n_subjects"]),
            n_clusters=int(meta["n_clusters"]),
            smoothing_cell=float(meta["smoothing_cell"]),
            smoothing_marginal=float(meta["smoothing_marginal"]),
            residual_scale=float(meta.get("residual_scale", 1.0)),
            n_train=int(meta.get("n_train", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
        )


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


def fit_member5_residual(
    *,
    subject_ids: np.ndarray,        # [N_rows] int >= -1
    cluster_ids: np.ndarray,        # [N_rows] int >= -1
    labels: np.ndarray,             # [N_rows] float in {0, 1}
    subject_keys: Sequence[str],
    n_clusters: int,
    smoothing_cell: float = 30.0,
    smoothing_marginal: float = 10.0,
    residual_scale: float = 1.0,
    eval_sample_size: int = 100_000,
    seed: int = 0,
) -> Member5ResidualState:
    """Fit subject x cluster residual passrate lookup tables.

    Parameters
    ----------
    subject_ids, cluster_ids, labels
        Aligned ``[N_rows]`` arrays giving the training observations.
        Rows with ``subject_id < 0`` or ``cluster_id < 0`` are silently
        dropped from the aggregation; the cold-start path in
        ``apply_state_batch`` handles them at apply time. ``labels``
        must be in ``{0, 1}`` (or floats interpretable as pass rates;
        values are passed through unchanged into the bincount sums).
    subject_keys
        Length must equal ``n_subjects``; position ``i`` is the string
        key for subject ``i`` (consistent with the global ``Indexer``
        ordering used elsewhere in the notebook).
    n_clusters
        Total cluster count. Cluster ids in ``[0, n_clusters)`` are
        used; anything outside that range is treated as ``UNKNOWN``.
    smoothing_cell
        Bayesian prior strength for the per-(subject, cluster) cell
        means. Cells with ``n + smoothing_cell`` total weight get the
        cluster fallback at weight ``smoothing_cell / (n + smoothing_cell)``.
        Larger values keep low-support cells closer to the additive
        prediction (i.e. residual ~ 0).
    smoothing_marginal
        Same shrinkage strength but for the 1-D marginal subject /
        cluster means. Typically smaller than ``smoothing_cell`` since
        marginals always have more support than cells.
    residual_scale
        Multiplier on ``residual_logit`` applied at fit time (baked into
        the saved table). Defaults to 1.0; lower values further damp
        the cell-level residual relative to the additive baseline.
    eval_sample_size
        Number of training rows to sub-sample for the cached
        ``train_loss`` estimate. The full-pass NLL on 5M rows is ~5 s
        which is fine, but the sub-sample is fast enough that we use it
        for sanity printing without holding up the fit.
    seed
        Seed for the eval sub-sample.

    Returns
    -------
    Member5ResidualState
    """
    subject_keys_list = list(str(s) for s in subject_keys)
    n_subjects = len(subject_keys_list)
    if n_subjects <= 0:
        raise ValueError("subject_keys must be non-empty")
    if int(n_clusters) <= 0:
        raise ValueError(f"n_clusters must be > 0, got {n_clusters}")

    s_arr = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    c_arr = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s_arr.shape != y_arr.shape or c_arr.shape != y_arr.shape:
        raise ValueError(
            f"subject_ids {s_arr.shape}, cluster_ids {c_arr.shape}, "
            f"labels {y_arr.shape} must all be aligned"
        )
    n_rows = int(y_arr.shape[0])
    if n_rows == 0:
        raise ValueError("fit_member5_residual: empty input")

    # Use only rows whose ids fall in both valid ranges. We still feed
    # the marginal aggregators the broader rows (with the other id
    # potentially out-of-range); _bincount1d filters its own ids.
    valid = (
        (s_arr >= 0) & (s_arr < n_subjects)
        & (c_arr >= 0) & (c_arr < int(n_clusters))
    )
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError(
            "fit_member5_residual: no rows have both subject_id and "
            "cluster_id in range. Check ids."
        )

    # ---- Global mean ----
    global_mean = float(np.clip(np.mean(y_arr), _EPS, 1.0 - _EPS))
    global_mean_logit = float(_logit_safe(global_mean))

    # ---- Marginal subject and cluster means (Bayes-shrunk to global) ----
    subj_sum, subj_cnt = _bincount1d_sum_and_count(
        s_arr, y_arr, n_subjects
    )
    cluster_sum, cluster_cnt = _bincount1d_sum_and_count(
        c_arr, y_arr, int(n_clusters)
    )
    subj_mean = _bayes_shrunk(
        subj_sum, subj_cnt, global_mean, smoothing_marginal
    )
    cluster_mean = _bayes_shrunk(
        cluster_sum, cluster_cnt, global_mean, smoothing_marginal
    )
    subj_logit_f64 = _logit_safe(subj_mean)
    cluster_logit_f64 = _logit_safe(cluster_mean)

    # ---- 2-D (subject, cluster) cell means (Bayes-shrunk) ----
    sc_sum, sc_cnt = _bincount2d_sum_and_count(
        s_arr, c_arr, y_arr, n_subjects, int(n_clusters)
    )
    # Fallback per cell is the additive prediction in PROB space; this is
    # what the per-cell shrinkage will reduce to when n=0. We construct
    # this once as a [n_subjects, n_clusters] matrix.
    additive_logit = (
        subj_logit_f64[:, None]
        + cluster_logit_f64[None, :]
        - global_mean_logit
    )
    additive_prob = _sigmoid_stable(additive_logit)
    cell_mean = _bayes_shrunk(
        sc_sum, sc_cnt, additive_prob, smoothing_cell
    )
    cell_logit_f64 = _logit_safe(cell_mean)

    # Residual: cell_logit - (subj + cluster - global). For cells with no
    # observed data, _bayes_shrunk returns ``additive_prob`` exactly, so
    # the corresponding logit equals ``additive_logit`` and the residual
    # collapses to 0. This is the desired behavior: unseen (s, c) -> the
    # additive marginal prediction.
    residual_logit_f64 = cell_logit_f64 - additive_logit
    if float(residual_scale) != 1.0:
        residual_logit_f64 = float(residual_scale) * residual_logit_f64

    cell_log1p_n_f32 = np.log1p(sc_cnt).astype(np.float32)

    # ---- Sampled train loss for diagnostics ----
    rng = np.random.default_rng(int(seed))
    sample_n = int(min(eval_sample_size, n_valid))
    if sample_n > 0:
        valid_idx = np.where(valid)[0]
        sample_idx = (
            rng.choice(valid_idx.shape[0], size=sample_n, replace=False)
            if valid_idx.shape[0] > sample_n
            else np.arange(valid_idx.shape[0])
        )
        sample_rows = valid_idx[sample_idx]
        s_sample = s_arr[sample_rows]
        c_sample = c_arr[sample_rows]
        y_sample = y_arr[sample_rows]
        pred = _sigmoid_stable(
            subj_logit_f64[s_sample]
            + cluster_logit_f64[c_sample]
            - global_mean_logit
            + residual_logit_f64[s_sample, c_sample]
        )
        pred = np.clip(pred, _EPS, 1.0 - _EPS)
        train_loss = float(
            -(y_sample * np.log(pred) + (1.0 - y_sample) * np.log(1.0 - pred)).mean()
        )
    else:
        train_loss = 0.0

    LOG.info(
        "fit_member5_residual: N=%d (valid=%d)  S=%d  C=%d  "
        "global_mean=%.4f  smoothing_cell=%.1f  smoothing_marginal=%.1f  "
        "residual_scale=%.3f  train_loss(%d)=%.4f",
        n_rows, n_valid, n_subjects, int(n_clusters),
        global_mean, float(smoothing_cell), float(smoothing_marginal),
        float(residual_scale), sample_n, train_loss,
    )

    return Member5ResidualState(
        global_mean_logit=global_mean_logit,
        subj_logit=subj_logit_f64.astype(np.float32),
        cluster_logit=cluster_logit_f64.astype(np.float32),
        residual_logit=residual_logit_f64.astype(np.float32),
        cell_log1p_n=cell_log1p_n_f32,
        subject_keys=tuple(subject_keys_list),
        n_subjects=int(n_subjects),
        n_clusters=int(n_clusters),
        smoothing_cell=float(smoothing_cell),
        smoothing_marginal=float(smoothing_marginal),
        residual_scale=float(residual_scale),
        n_train=int(n_rows),
        train_loss=float(train_loss),
        val_loss=0.0,
    )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def apply_batch(
    *,
    global_mean_logit: float,
    subj_logit: np.ndarray,
    cluster_logit: np.ndarray,
    residual_logit: np.ndarray,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Vectorized inference. Returns ``[N]`` float32 probabilities in
    ``(eps, 1-eps)``. Out-of-range subject or cluster ids fall through
    to the appropriate marginal-only or global-only prediction.
    """
    s_arr = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    c_arr = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if s_arr.shape != c_arr.shape:
        raise ValueError(
            f"subject_ids shape {s_arr.shape} != cluster_ids shape {c_arr.shape}"
        )
    n_subjects = int(subj_logit.shape[0])
    n_clusters = int(cluster_logit.shape[0])
    if residual_logit.shape != (n_subjects, n_clusters):
        raise ValueError(
            f"residual_logit shape {residual_logit.shape} != "
            f"({n_subjects}, {n_clusters})"
        )

    s_valid = (s_arr >= 0) & (s_arr < n_subjects)
    c_valid = (c_arr >= 0) & (c_arr < n_clusters)
    both = s_valid & c_valid

    s_safe = np.where(s_valid, s_arr, 0)
    c_safe = np.where(c_valid, c_arr, 0)

    subj_term = np.where(
        s_valid,
        subj_logit[s_safe].astype(np.float64),
        float(global_mean_logit),
    )
    cluster_term = np.where(
        c_valid,
        cluster_logit[c_safe].astype(np.float64),
        float(global_mean_logit),
    )
    res_term = np.where(
        both,
        residual_logit[s_safe, c_safe].astype(np.float64),
        0.0,
    )
    z = subj_term + cluster_term - float(global_mean_logit) + res_term
    p = _sigmoid_stable(z)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p.astype(np.float32)


def apply_one(
    *,
    global_mean_logit: float,
    subj_logit: np.ndarray,
    cluster_logit: np.ndarray,
    residual_logit: np.ndarray,
    subject_id: int,
    cluster_id: int,
) -> float:
    """Single-row inference. Returns python float in ``(eps, 1-eps)``."""
    n_subjects = int(subj_logit.shape[0])
    n_clusters = int(cluster_logit.shape[0])
    s = int(subject_id)
    c = int(cluster_id)
    s_valid = (s >= 0) and (s < n_subjects)
    c_valid = (c >= 0) and (c < n_clusters)
    subj_term = float(subj_logit[s]) if s_valid else float(global_mean_logit)
    cluster_term = float(cluster_logit[c]) if c_valid else float(global_mean_logit)
    res_term = float(residual_logit[s, c]) if (s_valid and c_valid) else 0.0
    z = subj_term + cluster_term - float(global_mean_logit) + res_term
    p = _sigmoid_stable(z)
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, _EPS), 1.0 - _EPS))


# ---------------------------------------------------------------------------
# State-keyed convenience wrappers (uniform with logreg_member /
# gbdt_member / knn_member conventions).
# ---------------------------------------------------------------------------


def apply_state_batch(
    state: "Member5ResidualState",
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """State-keyed batched inference."""
    return apply_batch(
        global_mean_logit=float(state.global_mean_logit),
        subj_logit=state.subj_logit,
        cluster_logit=state.cluster_logit,
        residual_logit=state.residual_logit,
        subject_ids=subject_ids,
        cluster_ids=cluster_ids,
    )


def apply_state_one(
    state: "Member5ResidualState",
    *,
    subject_id: int,
    cluster_id: int,
) -> float:
    """State-keyed single-row inference."""
    return apply_one(
        global_mean_logit=float(state.global_mean_logit),
        subj_logit=state.subj_logit,
        cluster_logit=state.cluster_logit,
        residual_logit=state.residual_logit,
        subject_id=subject_id,
        cluster_id=cluster_id,
    )


__all__ = [
    "Member5ResidualState",
    "fit_member5_residual",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
    "apply_state_one",
]
