"""Member 6: Field-weighted Factorization Machine on the dense matrix.

Adds a bilinear-interaction member to the stacked ensemble. The goal
is *inductive bias diversity* against Members 1/3/4: M1 is a deep
non-linear model on the embedding manifold, M3 is a k-NN locally
constant predictor, and M4 is a strictly additive linear classifier.
None of them can express *pairwise feature interactions* through a
low-rank decomposition. FwFM fills exactly that gap.

Model
-----
For an input row ``x in R^F`` with per-feature field assignments
``F(i) in {0, ..., n_fields-1}``:

    z(x) = w0
         + <w, x>
         + sum_{i<j} r[F(i), F(j)] <v_i, v_j> x_i x_j

with the per-field aggregated latent vector
``s_f = sum_{i : F(i)=f} v_i x_i`` (shape ``[k]``), the bilinear sum
collapses to:

    sum_{f1<f2}  r[f1, f2] * <s_{f1}, s_{f2}>
  + sum_{f}      r[f, f]   * 0.5 * (||s_f||^2 - sum_{i in f} x_i^2 ||v_i||^2)

When ``n_fields == 1`` (the default) FwFM degenerates to classic FM
with a single field-pair scalar ``r[0, 0]``. Callers that want true
field weighting pass a non-trivial ``field_ids`` array (one int per
feature column).

Standardization
---------------
The bilinear term involves products ``x_i x_j``, so we *cannot* bake
mean/std into the saved weights the way ``logreg_member`` does for
the strictly-linear case. Instead we *re-apply standardization at
inference* using the stored ``feat_mean`` and ``feat_std`` arrays.
The cost is one subtract + divide per row, which is negligible next
to the bilinear matmul.

Cold-start / robustness
-----------------------
The dense matrix is built by the same Member 2 / Member 4 pipeline
that already handles unknown subjects, unknown benchmarks, and
missing embeddings; the FwFM consumes whatever those upstream features
produce. We clip predictions to ``(_EPS, 1-_EPS)`` so a single saturated
logit cannot crash downstream BCE / log-loss reporting.

Runtime contract
----------------
Inference is pure numpy: ``apply_state_batch(state, X)`` -> [N] float32
probabilities; ``apply_state_one(state, x)`` -> python float in
``(eps, 1-eps)``. The runtime template does NOT need torch.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("fwfm_member")

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; never NaN on huge |z|."""
    z64 = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z64)
    pos = z64 >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z64[pos]))
    neg = ~pos
    e = np.exp(z64[neg])
    out[neg] = e / (1.0 + e)
    return out


def _sigmoid_scalar(z: float) -> float:
    zf = float(z)
    if zf >= 0:
        return 1.0 / (1.0 + math.exp(-zf))
    e = math.exp(zf)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class FwFMState:
    """Fitted state of Member 6 (FwFM on the M4 dense matrix).

    See module docstring for the exact prediction equation.
    """

    w0: float
    w: np.ndarray              # [F] float32 linear weights
    V: np.ndarray              # [F, k] float32 latent factors
    r: np.ndarray              # [n_fields, n_fields] float32, symmetric
    field_ids: np.ndarray      # [F] int32
    feat_mean: np.ndarray      # [F] float32 (zeros if standardize=False)
    feat_std: np.ndarray       # [F] float32 (ones  if standardize=False)
    feature_dim: int
    feature_names: tuple[str, ...]
    k: int
    n_fields: int
    fit_method: str
    n_train: int
    n_pos: int
    train_loss: float
    val_loss: float
    standardize: bool
    weight_decay_w: float
    weight_decay_V: float
    weight_decay_r: float

    def __post_init__(self) -> None:
        F = int(self.feature_dim)
        if int(self.w.shape[0]) != F:
            raise ValueError(
                f"w len {self.w.shape[0]} != feature_dim {F}"
            )
        if self.V.shape != (F, int(self.k)):
            raise ValueError(
                f"V shape {self.V.shape} != ({F}, {self.k})"
            )
        if self.r.shape != (int(self.n_fields), int(self.n_fields)):
            raise ValueError(
                f"r shape {self.r.shape} != ({self.n_fields}, {self.n_fields})"
            )
        if int(self.field_ids.shape[0]) != F:
            raise ValueError(
                f"field_ids len {self.field_ids.shape[0]} != feature_dim {F}"
            )
        if int(self.feat_mean.shape[0]) != F:
            raise ValueError(
                f"feat_mean len {self.feat_mean.shape[0]} != feature_dim {F}"
            )
        if int(self.feat_std.shape[0]) != F:
            raise ValueError(
                f"feat_std len {self.feat_std.shape[0]} != feature_dim {F}"
            )
        if int(len(self.feature_names)) != F:
            raise ValueError(
                f"feature_names len {len(self.feature_names)} != feature_dim {F}"
            )
        if not math.isfinite(float(self.w0)):
            raise ValueError("FwFMState: w0 is NaN/Inf")
        for name, arr in (
            ("w", self.w), ("V", self.V), ("r", self.r),
            ("feat_mean", self.feat_mean), ("feat_std", self.feat_std),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"FwFMState: {name} contains NaN/Inf")
        if not np.all(self.feat_std > 0):
            raise ValueError(
                "FwFMState: feat_std must be strictly positive (replace "
                "zero-std columns with 1.0 at fit time)."
            )
        # r must be symmetric; we enforce this at save time too.
        if not np.allclose(self.r, self.r.T, atol=1e-6):
            raise ValueError("FwFMState: r is not symmetric")
        # field_ids must lie in [0, n_fields).
        fi_max = int(self.field_ids.max()) if int(self.field_ids.size) > 0 else -1
        fi_min = int(self.field_ids.min()) if int(self.field_ids.size) > 0 else 0
        if fi_min < 0 or fi_max >= int(self.n_fields):
            raise ValueError(
                f"field_ids range [{fi_min}, {fi_max}] outside "
                f"[0, {self.n_fields})"
            )

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "weights.npz",
            w=self.w.astype(np.float32),
            V=self.V.astype(np.float32),
            r=self.r.astype(np.float32),
            field_ids=self.field_ids.astype(np.int32),
            feat_mean=self.feat_mean.astype(np.float32),
            feat_std=self.feat_std.astype(np.float32),
        )
        meta = {
            "w0": float(self.w0),
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "k": int(self.k),
            "n_fields": int(self.n_fields),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "standardize": bool(self.standardize),
            "weight_decay_w": float(self.weight_decay_w),
            "weight_decay_V": float(self.weight_decay_V),
            "weight_decay_r": float(self.weight_decay_r),
            "format_version": 1,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "FwFMState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz") as npz:
            w = npz["w"].astype(np.float32, copy=False)
            V = npz["V"].astype(np.float32, copy=False)
            r = npz["r"].astype(np.float32, copy=False)
            field_ids = npz["field_ids"].astype(np.int32, copy=False)
            feat_mean = npz["feat_mean"].astype(np.float32, copy=False)
            feat_std = npz["feat_std"].astype(np.float32, copy=False)
        return cls(
            w0=float(meta["w0"]),
            w=w,
            V=V,
            r=r,
            field_ids=field_ids,
            feat_mean=feat_mean,
            feat_std=feat_std,
            feature_dim=int(meta["feature_dim"]),
            feature_names=tuple(str(s) for s in meta["feature_names"]),
            k=int(meta["k"]),
            n_fields=int(meta["n_fields"]),
            fit_method=str(meta.get("fit_method", "unknown")),
            n_train=int(meta.get("n_train", 0)),
            n_pos=int(meta.get("n_pos", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
            standardize=bool(meta.get("standardize", True)),
            weight_decay_w=float(meta.get("weight_decay_w", 0.0)),
            weight_decay_V=float(meta.get("weight_decay_V", 0.0)),
            weight_decay_r=float(meta.get("weight_decay_r", 0.0)),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _standardize(
    X: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    standardize: bool,
) -> np.ndarray:
    """Z-score ``X`` in-place-safe (returns a new fp32 array if standardize)."""
    if not standardize:
        return np.asarray(X, dtype=np.float32, order="C")
    X32 = np.asarray(X, dtype=np.float32, order="C")
    out = np.empty_like(X32, dtype=np.float32)
    np.subtract(X32, feat_mean.astype(np.float32), out=out)
    out /= feat_std.astype(np.float32)
    return out


def _fwfm_logits_from_standardized(
    Xz: np.ndarray,           # [N, F] float32 standardized
    w0: float,
    w: np.ndarray,            # [F] float32
    V: np.ndarray,            # [F, k] float32
    r: np.ndarray,            # [n_fields, n_fields] float32, symmetric
    field_ids: np.ndarray,    # [F] int32
    n_fields: int,
) -> np.ndarray:
    """Compute FwFM logits on already-standardized ``Xz``.

    See module docstring for the formula. Returns ``[N]`` float64.
    """
    N = int(Xz.shape[0])
    F = int(Xz.shape[1])
    k = int(V.shape[1])
    if int(w.shape[0]) != F:
        raise ValueError(f"w len {w.shape[0]} != F {F}")
    if int(V.shape[0]) != F:
        raise ValueError(f"V rows {V.shape[0]} != F {F}")
    if int(field_ids.shape[0]) != F:
        raise ValueError(f"field_ids len {field_ids.shape[0]} != F {F}")

    # Linear term in fp64 for numerical headroom on large F.
    z = (Xz.astype(np.float32) @ w.astype(np.float32)).astype(np.float64)
    z += float(w0)

    # Per-field aggregated latent vectors s_f = sum_{i in f} v_i x_i.
    # Also need q_f = sum_{i in f} (v_i x_i)^2  -- the "diagonal" mass
    # we subtract off for the within-field upper-triangular sum.
    s_per_field = np.zeros((N, int(n_fields), k), dtype=np.float64)
    q_per_field = np.zeros((N, int(n_fields)), dtype=np.float64)
    fi = np.asarray(field_ids, dtype=np.int64).reshape(-1)
    for f in range(int(n_fields)):
        mask = (fi == int(f))
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        Xf = Xz[:, mask].astype(np.float64, copy=False)      # [N, n_in]
        Vf = V[mask, :].astype(np.float64, copy=False)        # [n_in, k]
        s_f = Xf @ Vf                                         # [N, k]
        # q_f = sum_i (v_i x_i)^2 = sum_i (x_i)^2 * sum_k v_{i,k}^2  WRONG --
        # actually we need sum over i of <v_i x_i, v_i x_i> = sum_i x_i^2 * ||v_i||^2
        # because (v_i x_i) is a vector and its squared norm is x_i^2 * ||v_i||^2.
        v_sq_row_norms = (Vf * Vf).sum(axis=1)                # [n_in]
        Xf_sq = Xf * Xf                                       # [N, n_in]
        q_f = Xf_sq @ v_sq_row_norms                          # [N]
        s_per_field[:, f, :] = s_f
        q_per_field[:, f] = q_f

    # Within-field upper-triangular sum.
    # sum_{i<j, F(i)=F(j)=f} <v_i, v_j> x_i x_j = 0.5 (||s_f||^2 - q_f)
    s_norm_sq = (s_per_field ** 2).sum(axis=2)               # [N, n_fields]
    within = 0.5 * (s_norm_sq - q_per_field)                  # [N, n_fields]
    r64 = r.astype(np.float64)
    r_diag = np.diag(r64).astype(np.float64)                  # [n_fields]
    z += within @ r_diag                                      # [N]

    # Cross-field sums: sum_{f1<f2} r[f1, f2] * <s_{f1}, s_{f2}>
    if int(n_fields) > 1:
        # Build a [n_fields, n_fields] inner-product matrix per row efficiently:
        # IP[n, f1, f2] = <s_per_field[n, f1, :], s_per_field[n, f2, :]>
        # = einsum('nfk,ngk->nfg', s, s). We only sum f1<f2 with r weights.
        IP = np.einsum(
            "nfk,ngk->nfg", s_per_field, s_per_field, optimize=True,
        )                                                     # [N, nf, nf]
        # Off-diagonal r weights (zero out diagonal so it's not double-counted).
        r_off = r64.copy()
        np.fill_diagonal(r_off, 0.0)
        # Only count each pair once: use strictly upper triangular by
        # zeroing the lower triangle.
        tri_mask = np.triu(np.ones_like(r_off, dtype=bool), k=1)
        r_off = np.where(tri_mask, r_off, 0.0)
        z += np.einsum("nfg,fg->n", IP, r_off, optimize=True)

    return z


def apply_batch(
    *,
    w0: float,
    w: np.ndarray,
    V: np.ndarray,
    r: np.ndarray,
    field_ids: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    standardize: bool,
    n_fields: int,
    X: np.ndarray,
) -> np.ndarray:
    """Pure-numpy batched inference. Returns ``[N]`` float32 probs."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if int(X.shape[1]) != int(w.shape[0]):
        raise ValueError(
            f"X cols {X.shape[1]} != feature_dim {w.shape[0]}"
        )
    Xz = _standardize(X, feat_mean, feat_std, bool(standardize))
    z = _fwfm_logits_from_standardized(
        Xz=Xz,
        w0=float(w0),
        w=w,
        V=V,
        r=r,
        field_ids=field_ids,
        n_fields=int(n_fields),
    )
    p = _sigmoid_stable(z)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p.astype(np.float32)


def apply_one(
    *,
    w0: float,
    w: np.ndarray,
    V: np.ndarray,
    r: np.ndarray,
    field_ids: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    standardize: bool,
    n_fields: int,
    x: np.ndarray,
) -> float:
    """Single-row inference. ``x`` is ``[F]`` float32. Returns python float."""
    if x.ndim != 1 or int(x.shape[0]) != int(w.shape[0]):
        raise ValueError(
            f"x shape {x.shape} incompatible with weights len {w.shape[0]}"
        )
    p = apply_batch(
        w0=float(w0), w=w, V=V, r=r, field_ids=field_ids,
        feat_mean=feat_mean, feat_std=feat_std,
        standardize=bool(standardize), n_fields=int(n_fields),
        X=x.reshape(1, -1),
    )
    return float(p[0])


def apply_state_batch(state: "FwFMState", X: np.ndarray) -> np.ndarray:
    """State-keyed batched inference (uniform with other members)."""
    return apply_batch(
        w0=float(state.w0),
        w=state.w, V=state.V, r=state.r,
        field_ids=state.field_ids,
        feat_mean=state.feat_mean, feat_std=state.feat_std,
        standardize=bool(state.standardize), n_fields=int(state.n_fields),
        X=X,
    )


def apply_state_one(state: "FwFMState", x: np.ndarray) -> float:
    return apply_one(
        w0=float(state.w0),
        w=state.w, V=state.V, r=state.r,
        field_ids=state.field_ids,
        feat_mean=state.feat_mean, feat_std=state.feat_std,
        standardize=bool(state.standardize), n_fields=int(state.n_fields),
        x=x,
    )


# ---------------------------------------------------------------------------
# Chunked standardization helpers (mirror logreg_member for OOM-safety)
# ---------------------------------------------------------------------------


def _chunked_mean_std(
    X: np.ndarray,
    idx: np.ndarray,
    chunk: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ``(mean, std)`` of ``X[idx]`` without a full f64 copy."""
    n = int(idx.shape[0])
    F = int(X.shape[1])
    if n == 0:
        return np.zeros(F, dtype=np.float64), np.ones(F, dtype=np.float64)
    chunk = max(1, int(chunk))
    s_sum = np.zeros(F, dtype=np.float64)
    for s_ in range(0, n, chunk):
        e_ = min(s_ + chunk, n)
        s_sum += X[idx[s_:e_]].sum(axis=0, dtype=np.float64)
    mu = s_sum / float(n)
    s_var = np.zeros(F, dtype=np.float64)
    for s_ in range(0, n, chunk):
        e_ = min(s_ + chunk, n)
        block = X[idx[s_:e_]].astype(np.float64, copy=False) - mu[None, :]
        s_var += (block * block).sum(axis=0)
    var = s_var / float(n)
    return mu, np.sqrt(np.maximum(var, 0.0))


def _chunked_standardize_into(
    X: np.ndarray,
    idx: np.ndarray,
    mu_f32: np.ndarray,
    sigma_f32: np.ndarray,
    chunk: int = 65_536,
) -> np.ndarray:
    n = int(idx.shape[0])
    F = int(X.shape[1])
    out = np.empty((n, F), dtype=np.float32)
    if n == 0:
        return out
    chunk = max(1, int(chunk))
    for s_ in range(0, n, chunk):
        e_ = min(s_ + chunk, n)
        block = X[idx[s_:e_]]
        np.subtract(block, mu_f32, out=out[s_:e_])
        out[s_:e_] /= sigma_f32
    return out


def _chunked_gather_f32(
    X: np.ndarray,
    idx: np.ndarray,
    chunk: int = 65_536,
) -> np.ndarray:
    n = int(idx.shape[0])
    F = int(X.shape[1])
    out = np.empty((n, F), dtype=np.float32)
    if n == 0:
        return out
    chunk = max(1, int(chunk))
    for s_ in range(0, n, chunk):
        e_ = min(s_ + chunk, n)
        out[s_:e_] = X[idx[s_:e_]]
    return out


# ---------------------------------------------------------------------------
# Offline trainer (uses torch; called from the notebook only)
# ---------------------------------------------------------------------------


def fit_fwfm_member(
    *,
    X: np.ndarray,                    # [N, F] float32
    y: np.ndarray,                    # [N] float32 in {0, 1}
    feature_names: Sequence[str],
    field_ids: np.ndarray | None = None,  # [F] int; default zeros (classic FM)
    k: int = 8,
    sample_weights: np.ndarray | None = None,
    weight_decay_w: float = 1.0e-5,
    weight_decay_V: float = 1.0e-4,
    weight_decay_r: float = 1.0e-4,
    learning_rate: float = 1.0e-3,
    epochs: int = 40,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    seed: int = 0,
    early_stopping_patience: int = 5,
    device: str | None = None,
    standardize: bool = True,
    init_V_scale: float | None = None,
    log_every: int = 5,
    holdout_group_id: np.ndarray | None = None,
    init_w0_from_prior: bool = True,
) -> FwFMState:
    """Fit Member 6 (FwFM) with Adam + early stopping.

    Parameters
    ----------
    X, y
        ``[N, F]`` dense feature matrix and ``[N]`` 0/1 labels. ``X`` is
        expected float32; no full f64 copy is ever materialized.
    feature_names
        Length must equal ``X.shape[1]``; used for the saved state's
        ``feature_names`` field (provenance / runtime sanity).
    field_ids
        Optional ``[F]`` int array assigning each feature column to a
        field. Defaults to ``zeros(F)`` (single field -> classic FM
        with a single ``r[0, 0]`` scalar). To use multiple fields,
        pass an int array with values in ``[0, n_fields)``; the trainer
        infers ``n_fields = field_ids.max() + 1``.
    k
        Latent factor dimensionality. Literature suggests small values
        (4-16) for FwFM; the default 8 is a good tradeoff between
        capacity and regularization.
    weight_decay_w, weight_decay_V, weight_decay_r
        Independent L2 penalties on the linear weights, latent factors,
        and field-pair weight matrix respectively. Separate knobs let
        you reproduce the literature pattern of stronger L2 on V than
        on w.
    learning_rate, epochs, batch_size, early_stopping_patience
        Adam optimizer schedule. The model is bilinear, not linear, so
        the learning rate is smaller than ``logreg_member``'s default
        (1e-3 vs 5e-2).
    standardize
        Z-score features at fit time using the TRAIN-slice mean/std
        (no val leakage); the same mean/std are *stored* in the state
        and reapplied at inference. (Unlike logreg_member we cannot
        bake standardization into the bilinear weights cheaply.)
    init_V_scale
        If ``None``, initializes V ~ N(0, 1/sqrt(k)). Otherwise uses
        the given scale.
    init_w0_from_prior
        If True (default), initializes the bias at ``logit(mean(y))``
        on the train slice so Adam doesn't burn epochs lifting the
        bias from 0.
    holdout_group_id
        Optional per-row int array. When provided the internal
        train / val split holds out whole groups (typically item_id)
        instead of random rows; mirrors ``logreg_member``'s flag and
        the cold-item discipline.
    """
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    N, F = int(X.shape[0]), int(X.shape[1])
    if y.shape != (N,):
        raise ValueError(f"y shape {y.shape} != ({N},)")
    if int(len(feature_names)) != F:
        raise ValueError(
            f"feature_names len {len(feature_names)} != X cols {F}"
        )
    if sample_weights is not None and sample_weights.shape != y.shape:
        raise ValueError(
            f"sample_weights shape {sample_weights.shape} != y shape {y.shape}"
        )

    if field_ids is None:
        field_ids_arr = np.zeros(F, dtype=np.int32)
    else:
        field_ids_arr = np.asarray(field_ids, dtype=np.int32).reshape(-1)
        if int(field_ids_arr.shape[0]) != F:
            raise ValueError(
                f"field_ids len {field_ids_arr.shape[0]} != F {F}"
            )
        if int(field_ids_arr.min()) < 0:
            raise ValueError(
                f"field_ids contains negative entries (min={int(field_ids_arr.min())})"
            )
    n_fields = int(field_ids_arr.max()) + 1 if F > 0 else 1
    if n_fields <= 0:
        n_fields = 1

    rng = np.random.default_rng(int(seed))
    if holdout_group_id is None:
        perm = rng.permutation(N)
        n_val = max(64, int(round(val_fraction * N)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        gids = np.asarray(holdout_group_id).reshape(-1)
        if gids.shape != (N,):
            raise ValueError(
                f"holdout_group_id shape {gids.shape} != ({N},)"
            )
        unique_groups = np.unique(gids)
        n_groups = int(unique_groups.shape[0])
        if n_groups < 2:
            raise ValueError(
                f"holdout_group_id has {n_groups} unique groups; need >=2"
            )
        n_val_groups = max(1, int(round(val_fraction * n_groups)))
        held_groups = rng.choice(unique_groups, size=n_val_groups, replace=False)
        held_set = set(int(g) for g in held_groups)
        val_mask = np.fromiter(
            (int(g) in held_set for g in gids), count=N, dtype=bool,
        )
        if int(val_mask.sum()) == 0 or int((~val_mask).sum()) == 0:
            raise RuntimeError(
                "Group-stratified split yielded an empty side; "
                "increase n_groups or val_fraction."
            )
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        LOG.info(
            "fwfm_member: group-stratified split (%d groups -> %d held; "
            "%d train rows / %d val rows)",
            n_groups, n_val_groups,
            int(train_idx.shape[0]), int(val_idx.shape[0]),
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _STD_CHUNK = 65_536
    if standardize:
        feat_mean, feat_std = _chunked_mean_std(X, train_idx, chunk=_STD_CHUNK)
        n_zero_std = int(np.sum(feat_std < 1.0e-9))
        feat_std_safe = np.where(feat_std < 1.0e-9, 1.0, feat_std)
        LOG.info(
            "fwfm_member: standardizing %d features  "
            "mean range=[%.4g, %.4g]  std range=[%.4g, %.4g]  n_zero_std=%d",
            F,
            float(feat_mean.min()), float(feat_mean.max()),
            float(feat_std.min()), float(feat_std.max()),
            n_zero_std,
        )
        mu_f32 = feat_mean.astype(np.float32)
        sigma_f32 = feat_std_safe.astype(np.float32)
    else:
        mu_f32 = np.zeros(F, dtype=np.float32)
        sigma_f32 = np.ones(F, dtype=np.float32)

    # ----- Streaming data path (OOM-safe) -----
    #
    # Earlier revisions materialized the full standardized
    # ``[N_train_slice, F]`` and ``[N_val_slice, F]`` matrices on
    # CPU, then uploaded both to GPU as resident tensors. With the
    # M4 hybrid matrix (``F ~ 1216``) and ``N_train ~ 4.6M`` that
    # path costs ~22 GB of additional CPU RAM and ~22 GB of VRAM
    # for the train slice alone -- enough to OOM both host and
    # device on Colab tiers we routinely target.
    #
    # The streaming path below keeps neither resident. We only
    # cache ``mu/sigma`` (a couple of KB), then per-batch:
    #   1. gather ``X[idx_global]`` (a fresh ``[B, F]`` fp32 copy
    #      on CPU, ~80 MB at B=16384, F=1216)
    #   2. standardize in-place on that copy (subtract mu, divide
    #      sigma)
    #   3. upload to ``device`` as a fp32 tensor
    # The val + final-train-loss evaluators do the same thing
    # under ``torch.no_grad()`` and aggregate per-row losses across
    # mini-batches so no single forward pass holds an [N_val, F]
    # tensor on GPU.
    #
    # The numerical contract is unchanged: this is the same
    # standardize/gather computation, just done lazily per batch.
    # Tests cover bit-for-bit equivalence on a small problem
    # against the previously-resident behavior (see
    # ``tests/test_fwfm_member.py``).

    train_idx_np = np.asarray(train_idx, dtype=np.int64)
    val_idx_np = np.asarray(val_idx, dtype=np.int64)
    y_np = np.asarray(y, dtype=np.float32)
    if sample_weights is not None:
        sw_np = np.asarray(sample_weights, dtype=np.float32)
    else:
        sw_np = None

    def _gather_batch_to_device(
        global_idx: np.ndarray,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor | None"]:
        """Gather one mini-batch (CPU fancy-index + in-place
        standardize) and upload to ``device``. ``global_idx`` are
        row indices into the caller's ``X`` (i.e. already
        composed of train_idx[perm_slice] or val_idx[chunk_slice]).
        Returns ``(xb, yb, wb_or_none)`` on ``device``."""
        rows = X[global_idx]
        if rows.dtype != np.float32:
            rows = rows.astype(np.float32, copy=True)
        elif not rows.flags.writeable or rows.base is X:
            # numpy fancy-indexing returns a writable copy, but
            # guard against future shape/dtype changes that would
            # silently turn this into a view.
            rows = rows.copy()
        if standardize:
            rows -= mu_f32
            rows /= sigma_f32
        xb = torch.from_numpy(rows).to(device, non_blocking=False)
        yb = torch.from_numpy(y_np[global_idx]).to(device, non_blocking=False)
        wb = (
            None if sw_np is None
            else torch.from_numpy(sw_np[global_idx]).to(device, non_blocking=False)
        )
        return xb, yb, wb

    # ---- Torch model ----
    init_scale = float(1.0 / math.sqrt(max(int(k), 1)))
    if init_V_scale is not None:
        init_scale = float(init_V_scale)

    # Per-field column index lists (torch long tensors). Used to gather
    # the (n_in_f) slice of features and V rows per field at every batch.
    field_id_tensor = torch.as_tensor(
        field_ids_arr.astype(np.int64), dtype=torch.long, device=device,
    )
    field_idx_lists: list[torch.Tensor] = []
    for f in range(int(n_fields)):
        idx_f = torch.nonzero(field_id_tensor == int(f), as_tuple=False).reshape(-1)
        field_idx_lists.append(idx_f)

    class _FwFM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w0 = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
            self.w = nn.Parameter(torch.zeros(F, dtype=torch.float32, device=device))
            self.V = nn.Parameter(
                torch.empty(F, int(k), dtype=torch.float32, device=device)
                .normal_(mean=0.0, std=init_scale)
            )
            # Symmetric field-pair weights: store the full matrix and
            # symmetrize via (R + R.T) / 2 on every forward pass so
            # gradient flow respects symmetry exactly.
            self.r_raw = nn.Parameter(
                torch.full(
                    (int(n_fields), int(n_fields)), 0.5,
                    dtype=torch.float32, device=device,
                )
            )

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            # Linear term.
            z = (xb @ self.w) + self.w0  # [B]

            # Per-field s_f and q_f.
            B = int(xb.shape[0])
            s_per_field = torch.zeros(
                B, int(n_fields), int(k), dtype=xb.dtype, device=xb.device
            )
            q_per_field = torch.zeros(
                B, int(n_fields), dtype=xb.dtype, device=xb.device
            )
            for f_, idx_f in enumerate(field_idx_lists):
                if int(idx_f.numel()) == 0:
                    continue
                Xf = xb.index_select(1, idx_f)             # [B, n_in]
                Vf = self.V.index_select(0, idx_f)         # [n_in, k]
                s_per_field[:, f_, :] = Xf @ Vf
                v_sq_row_norms = (Vf * Vf).sum(dim=1)      # [n_in]
                q_per_field[:, f_] = (Xf * Xf) @ v_sq_row_norms

            r_sym = 0.5 * (self.r_raw + self.r_raw.t())    # [nf, nf]

            # Within-field upper-triangular sum.
            s_norm_sq = (s_per_field * s_per_field).sum(dim=2)   # [B, nf]
            within = 0.5 * (s_norm_sq - q_per_field)             # [B, nf]
            r_diag = torch.diagonal(r_sym, 0).contiguous()        # [nf]
            z = z + within @ r_diag

            # Cross-field sum.
            if int(n_fields) > 1:
                # IP[b, f1, f2] = <s[b, f1, :], s[b, f2, :]>
                IP = torch.einsum("bfk,bgk->bfg", s_per_field, s_per_field)
                r_off = r_sym.clone()
                r_off.fill_diagonal_(0.0)
                # Keep only strict upper triangle (each pair counted once).
                tri = torch.triu(torch.ones_like(r_off), diagonal=1)
                r_off = r_off * tri
                z = z + torch.einsum("bfg,fg->b", IP, r_off)

            return z

    model = _FwFM()
    if init_w0_from_prior:
        with torch.no_grad():
            p_init = float(np.clip(y[train_idx].mean(), 1.0e-6, 1.0 - 1.0e-6))
            bias_init = float(math.log(p_init / (1.0 - p_init)))
            model.w0.fill_(bias_init)

    opt = Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    n_train = int(train_idx_np.shape[0])
    _EVAL_CHUNK = max(int(batch_size), 65_536)  # eval can use larger chunks safely

    def _compute_loss_on_batch(
        model_: _FwFM, x_: torch.Tensor, y_: torch.Tensor,
        w_: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-batch loss used during training. Returns a scalar
        tensor on ``device`` so backward can flow."""
        logits = model_(x_)
        per_row = bce(logits, y_)
        if w_ is None:
            data_loss = per_row.mean()
        else:
            data_loss = (per_row * w_).sum() / torch.clamp(w_.sum(), min=1.0e-9)
        reg = (
            0.5 * float(weight_decay_w) * (model_.w * model_.w).sum()
            + 0.5 * float(weight_decay_V) * (model_.V * model_.V).sum()
            + 0.5 * float(weight_decay_r) * (model_.r_raw * model_.r_raw).sum()
        )
        return data_loss + reg

    def _streamed_eval_loss(
        model_: _FwFM,
        idx_pool: np.ndarray,
        chunk: int,
    ) -> float:
        """Compute the *full-slice* loss (data + reg) on the
        rows in ``idx_pool`` by streaming mini-batches through
        the model under ``torch.no_grad()``. Numerically
        equivalent to the resident-tensor path:

            data_loss = sum(per_row * w) / sum(w)         (weighted)
            data_loss = sum(per_row) / N                  (unweighted)
            return    = data_loss + reg

        Aggregates ``sum_pr_w`` and ``sum_w`` across chunks, so
        the final ratio is exactly the same float as a single
        forward pass over the entire slice would produce
        (modulo floating-point reduction order, which is what
        ``BCEWithLogitsLoss(reduction='mean')`` is also
        susceptible to).
        """
        model_.eval()
        n = int(idx_pool.shape[0])
        sum_pr_w = 0.0
        sum_w = 0.0
        with torch.no_grad():
            for s_ in range(0, n, int(chunk)):
                e_ = min(s_ + int(chunk), n)
                global_idx = idx_pool[s_:e_]
                xb, yb, wb = _gather_batch_to_device(global_idx)
                logits = model_(xb)
                per_row = bce(logits, yb)
                if wb is None:
                    sum_pr_w += float(per_row.sum().item())
                    sum_w += float(per_row.numel())
                else:
                    sum_pr_w += float((per_row * wb).sum().item())
                    sum_w += float(wb.sum().item())
                # Free per-chunk GPU buffers eagerly so the next
                # chunk doesn't stack on top of them in the
                # caching allocator.
                xb = None
                yb = None
                wb = None
        data_loss = sum_pr_w / max(sum_w, 1.0e-9)
        with torch.no_grad():
            reg = (
                0.5 * float(weight_decay_w) * float((model_.w * model_.w).sum().item())
                + 0.5 * float(weight_decay_V) * float((model_.V * model_.V).sum().item())
                + 0.5 * float(weight_decay_r) * float((model_.r_raw * model_.r_raw).sum().item())
            )
        return float(data_loss + reg)

    # Per-epoch CPU permutation. We do shuffling on CPU (numpy)
    # because the gather happens against the CPU-resident ``X``
    # anyway; doing it on GPU would force a perm->CPU round-trip
    # per batch.
    perm_rng = np.random.default_rng(int(seed) + 17)

    best_val = float("inf")
    best_state: dict[str, np.ndarray | float] = {}
    epochs_since_improve = 0

    for ep in range(int(epochs)):
        model.train()
        perm_in = perm_rng.permutation(n_train)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_train, int(batch_size)):
            e = min(s + int(batch_size), n_train)
            global_idx = train_idx_np[perm_in[s:e]]
            xb, yb, wb = _gather_batch_to_device(global_idx)
            opt.zero_grad(set_to_none=True)
            loss = _compute_loss_on_batch(model, xb, yb, wb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
            xb = None
            yb = None
            wb = None

        val_loss = _streamed_eval_loss(model, val_idx_np, chunk=_EVAL_CHUNK)
        train_loss_ep = ep_loss / max(n_batches, 1)

        if val_loss < best_val - 1.0e-6:
            best_val = val_loss
            with torch.no_grad():
                r_sym = 0.5 * (model.r_raw + model.r_raw.t())
            best_state = {
                "w0": float(model.w0.detach().cpu().item()),
                "w": model.w.detach().cpu().numpy().astype(np.float32).copy(),
                "V": model.V.detach().cpu().numpy().astype(np.float32).copy(),
                "r": r_sym.detach().cpu().numpy().astype(np.float32).copy(),
            }
            epochs_since_improve = 0
            improved_marker = "*"
        else:
            epochs_since_improve += 1
            improved_marker = " "

        if int(log_every) > 0 and ((ep + 1) % int(log_every) == 0 or ep == 0):
            with torch.no_grad():
                w_norm = float(model.w.detach().norm().item())
                V_norm = float(model.V.detach().norm().item())
            LOG.info(
                "fwfm_member: ep %3d/%d  train=%.5f  val=%.5f  best=%.5f %s "
                "(no-improve=%d/%d)  ||w||=%.3f  ||V||=%.3f",
                ep + 1, int(epochs), train_loss_ep, val_loss, best_val,
                improved_marker, epochs_since_improve, int(early_stopping_patience),
                w_norm, V_norm,
            )

        if epochs_since_improve >= int(early_stopping_patience):
            LOG.info(
                "fwfm_member: early stop at epoch %d/%d (best_val=%.5f)",
                ep + 1, int(epochs), best_val,
            )
            break

    if not best_state:
        # Fallback (model never improved past best_val=inf? guard anyway).
        with torch.no_grad():
            r_sym = 0.5 * (model.r_raw + model.r_raw.t())
        best_state = {
            "w0": float(model.w0.detach().cpu().item()),
            "w": model.w.detach().cpu().numpy().astype(np.float32).copy(),
            "V": model.V.detach().cpu().numpy().astype(np.float32).copy(),
            "r": r_sym.detach().cpu().numpy().astype(np.float32).copy(),
        }

    # Final train_loss on the train slice (reported as the saved state's
    # train_loss; the val_loss is best_val from the early-stopping slice).
    # Use the same streaming evaluator the val pass uses -- avoids any
    # accidental ``[N_train, F]`` GPU residency for callers with a huge
    # train slice.
    final_train_loss = _streamed_eval_loss(
        model, train_idx_np, chunk=_EVAL_CHUNK,
    )

    result = FwFMState(
        w0=float(best_state["w0"]),
        w=best_state["w"],
        V=best_state["V"],
        r=best_state["r"],
        field_ids=field_ids_arr.astype(np.int32),
        feat_mean=mu_f32.astype(np.float32),
        feat_std=sigma_f32.astype(np.float32),
        feature_dim=int(F),
        feature_names=tuple(str(s) for s in feature_names),
        k=int(k),
        n_fields=int(n_fields),
        fit_method="adam_std_fwfm" if standardize else "adam_fwfm",
        n_train=int(n_train),
        n_pos=int(np.sum(y == 1.0)),
        train_loss=float(final_train_loss),
        val_loss=float(best_val),
        standardize=bool(standardize),
        weight_decay_w=float(weight_decay_w),
        weight_decay_V=float(weight_decay_V),
        weight_decay_r=float(weight_decay_r),
    )

    # ---- Explicit teardown before returning ----
    #
    # The streaming data path means we no longer hold large
    # ``[N_*, F]`` tensors on either CPU or GPU at this point --
    # only the model, optimizer, field-index lookup, and per-batch
    # buffers (which the train/eval loops already nulled out).
    # Still null model/opt/field-lookup/best_state and flush the
    # caching allocator: the next stage (e.g. Member 1 retraining
    # inside the OOF loop) should see freed VRAM rather than a
    # "reserved" pool.
    model = None  # noqa: F841
    opt = None  # noqa: F841
    field_id_tensor = None  # noqa: F841
    field_idx_lists = None  # noqa: F841
    best_state = None  # noqa: F841
    import gc as _gc

    _gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return result


__all__ = [
    "FwFMState",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
    "apply_state_one",
    "fit_fwfm_member",
]
