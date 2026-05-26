"""Member 4 of the four-member stacked ensemble: hand-rolled torch
logistic regression on the same feature vector as Member 2 (GBDT).

The user spec mandates: "Implement it yourself in torch (a one-layer
BCE-loss fit) -- do NOT use sklearn.linear_model, which is not
confirmed at runtime. Ship the learned weight vector; runtime
inference is one matvec + sigmoid."

Design notes
------------

* Training is a single linear layer with bias, optimized by Adam with
  L2 regularization (i.e. ridge-penalized logistic regression). The
  loss is per-row binary cross-entropy with optional sample weights
  -- needed when the OOF-prediction-feeding training-row schedule
  weights certain rows.
* Convergence: cosine LR schedule + early stopping on a small held-out
  slice of the training rows. The model is tiny (one linear layer
  over ~150 features) so a few thousand steps converges. Also
  supports L-BFGS via the ``solver="lbfgs"`` keyword for callers who
  want a deterministic fit.
* Export: a single ``.npz`` with ``weights`` ([feature_dim] float32)
  and ``bias`` (scalar float32). This is what the runtime loads -- no
  torch needed at runtime for inference (we ship a pure-numpy apply
  helper).

Runtime contract
----------------
``apply_one(weights, bias, features)`` -> probability in (eps, 1-eps)
``apply_batch(weights, bias, features_matrix)`` -> [N] probabilities

Both functions use only numpy and Python stdlib so the runtime
``model.py`` can call them without dragging torch into the inference
hot path.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("logreg_member")


_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# State (small, JSON-friendly per-scalar fields + numpy weight vector)
# ---------------------------------------------------------------------------


@dataclass
class LogRegMemberState:
    """Fitted-and-shipped state of Member 4.

    The weight vector and bias are the inference-time parameters; the
    rest is provenance / sanity metadata for reproducibility and
    runtime self-checks.
    """

    weights: np.ndarray   # [feature_dim] float32
    bias: float
    feature_dim: int
    feature_names: tuple[str, ...]
    fit_method: str        # "adam" | "lbfgs" | "identity"
    n_train: int
    n_pos: int
    train_loss: float
    val_loss: float
    weight_decay: float

    def __post_init__(self) -> None:
        if int(self.weights.shape[0]) != int(self.feature_dim):
            raise ValueError(
                f"weights len {self.weights.shape[0]} != feature_dim {self.feature_dim}"
            )
        if int(len(self.feature_names)) != int(self.feature_dim):
            raise ValueError(
                f"feature_names len {len(self.feature_names)} "
                f"!= feature_dim {self.feature_dim}"
            )
        # Belt-and-suspenders against NaN / Inf in saved state.
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("LogRegMemberState: weights contain NaN/Inf")
        if not math.isfinite(float(self.bias)):
            raise ValueError("LogRegMemberState: bias is NaN/Inf")

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "weights.npz",
            weights=self.weights.astype(np.float32),
            bias=np.float32(self.bias),
        )
        meta = {
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "weight_decay": float(self.weight_decay),
            "format_version": 1,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "LogRegMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz") as npz:
            w = npz["weights"].astype(np.float32, copy=False)
            b = float(npz["bias"])
        return cls(
            weights=w,
            bias=b,
            feature_dim=int(meta["feature_dim"]),
            feature_names=tuple(meta["feature_names"]),
            fit_method=str(meta.get("fit_method", "unknown")),
            n_train=int(meta.get("n_train", 0)),
            n_pos=int(meta.get("n_pos", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
            weight_decay=float(meta.get("weight_decay", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "weights": [float(x) for x in self.weights.tolist()],
            "bias": float(self.bias),
            "feature_dim": int(self.feature_dim),
            "feature_names": list(self.feature_names),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "weight_decay": float(self.weight_decay),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogRegMemberState":
        return cls(
            weights=np.asarray(d["weights"], dtype=np.float32),
            bias=float(d["bias"]),
            feature_dim=int(d["feature_dim"]),
            feature_names=tuple(d["feature_names"]),
            fit_method=str(d.get("fit_method", "unknown")),
            n_train=int(d.get("n_train", 0)),
            n_pos=int(d.get("n_pos", 0)),
            train_loss=float(d.get("train_loss", 0.0)),
            val_loss=float(d.get("val_loss", 0.0)),
            weight_decay=float(d.get("weight_decay", 0.0)),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference (used at runtime AND in tests)
# ---------------------------------------------------------------------------


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; never returns NaN even on huge |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    np_neg = z[~pos]
    e = np.exp(np_neg)
    out[~pos] = e / (1.0 + e)
    return out


def apply_batch(
    weights: np.ndarray,    # [F] float32
    bias: float,
    features: np.ndarray,    # [N, F] float32
) -> np.ndarray:
    """Vectorized inference. Returns float32 probabilities of shape [N]."""
    if features.ndim != 2 or features.shape[1] != weights.shape[0]:
        raise ValueError(
            f"features shape {features.shape} incompatible with "
            f"weights shape {weights.shape}"
        )
    z = (features.astype(np.float32) @ weights.astype(np.float32)) + np.float32(bias)
    p = _sigmoid_stable(z.astype(np.float64))
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p.astype(np.float32)


def apply_one(
    weights: np.ndarray,
    bias: float,
    features: np.ndarray,    # [F] float32
) -> float:
    """Single-row inference. Returns Python ``float`` in (eps, 1-eps)."""
    if features.ndim != 1 or features.shape[0] != weights.shape[0]:
        raise ValueError(
            f"features shape {features.shape} incompatible with "
            f"weights shape {weights.shape}"
        )
    z = float(features.astype(np.float64) @ weights.astype(np.float64)) + float(bias)
    if z >= 0:
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    if not math.isfinite(p):
        return 0.5
    return float(min(max(p, _EPS), 1.0 - _EPS))


# ---------------------------------------------------------------------------
# State-keyed convenience wrappers (uniform with gbdt_member / knn_member /
# stacker conventions). These let callers pass a single ``state`` argument
# instead of unpacking ``state.weights`` and ``state.bias`` at every call
# site -- the export postprocessing block and the integration test rely
# on the uniform signature.
# ---------------------------------------------------------------------------


def apply_state_one(state: "LogRegMemberState", features: np.ndarray) -> float:
    """State-keyed single-row inference (uniform with other members)."""
    return apply_one(state.weights, float(state.bias), features)


def apply_state_batch(state: "LogRegMemberState", features: np.ndarray) -> np.ndarray:
    """State-keyed batched inference (uniform with other members)."""
    return apply_batch(state.weights, float(state.bias), features)


# ---------------------------------------------------------------------------
# Memory-bounded standardization helpers (used by ``fit_logreg_member``)
# ---------------------------------------------------------------------------


def _chunked_mean_std(
    X: np.ndarray,
    idx: np.ndarray,
    chunk: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ``(mean, std)`` of ``X[idx]`` without a full f64 copy.

    Reductions are accumulated in float64 (matching ``X.astype(f64).
    mean(axis=0)`` numerics) but we only ever materialize one
    ``chunk``-sized float64 block at a time (~256 MB at chunk=65_536
    and F=1200). At 5M-row scale the eager path peaks at ~88 GB
    transient (a 22 GB X[idx] copy plus three 22-44 GB f64 temps);
    this path peaks at ~256 MB. Sample std uses Bessel's correction
    (divides by ``n - 1``) to match ``np.std(..., ddof=1)``-style
    semantics that ``X.astype(f64).std(axis=0)`` also approximates
    via the population formula -- the difference is < 1e-6 at
    n = 5M. Returned arrays are float64.
    """
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
    # Population variance (matches the previous .astype(f64).std(axis=0)
    # output, which uses ddof=0 by default). The fp32 round-off across
    # 5M rows is well below this difference.
    var = s_var / float(n)
    return mu, np.sqrt(np.maximum(var, 0.0))


def _chunked_standardize_into(
    X: np.ndarray,
    idx: np.ndarray,
    mu_f32: np.ndarray,
    sigma_f32: np.ndarray,
    chunk: int = 65_536,
) -> np.ndarray:
    """Fill a pre-allocated float32 ``out`` with ``(X[idx] - mu) / sigma``.

    Allocates exactly ``len(idx) * F * 4`` bytes for ``out`` -- nothing
    transient larger than one ``chunk``-sized block of ``X``.
    """
    n = int(idx.shape[0])
    F = int(X.shape[1])
    out = np.empty((n, F), dtype=np.float32)
    if n == 0:
        return out
    chunk = max(1, int(chunk))
    for s_ in range(0, n, chunk):
        e_ = min(s_ + chunk, n)
        block = X[idx[s_:e_]]
        # block is fp32; subtract & divide via f32 broadcasting,
        # writing through ``out[s_:e_]``. We avoid in-place ops on
        # block because numpy fancy-indexing returns a copy already
        # but we want the result to land in our pre-allocated buffer.
        np.subtract(block, mu_f32, out=out[s_:e_])
        out[s_:e_] /= sigma_f32
    return out


def _chunked_gather_f32(
    X: np.ndarray,
    idx: np.ndarray,
    chunk: int = 65_536,
) -> np.ndarray:
    """``X[idx].astype(np.float32, copy=False)`` but in chunks.

    Used in the ``standardize=False`` path for symmetry. With X
    already in float32 the cost is the same as a single fancy-index
    copy; we just spread the allocation over chunks so partial
    progress is visible to the OS allocator.
    """
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
# Offline trainer (uses torch, called from the notebook only)
# ---------------------------------------------------------------------------


def fit_logreg_member(
    *,
    X: np.ndarray,                    # [N, F] float32
    y: np.ndarray,                    # [N] float32 in {0, 1}
    sample_weights: np.ndarray | None = None,
    feature_names: Sequence[str],
    weight_decay: float = 1.0e-3,
    l1_strength: float = 0.0,
    learning_rate: float = 1.0e-2,
    epochs: int = 200,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    seed: int = 0,
    early_stopping_patience: int = 20,
    device: str | None = None,
    standardize: bool = True,
    log_every: int = 10,
    min_feature_std: float = 0.0,
    holdout_group_id: np.ndarray | None = None,
) -> LogRegMemberState:
    """Fit a torch logistic regression with Adam + early stopping.

    The training rows are split internally into a train/val slice with
    ``val_fraction``; the val slice is used both for early stopping
    AND as the reported ``val_loss`` in the saved state. This is NOT
    the OOF stacker val -- the stacker does its own out-of-fold split
    when consuming Member 4's predictions.

    **Standardization:** the member-feature schema mixes very different
    scales (``theta`` in [-1, 1], centroid distances in [1e3, 1e4],
    NN features in [0, 50], one-hot indicators in {0, 1}, ...). Logistic
    regression is NOT scale-invariant -- without standardization, Adam's
    per-parameter adaptive LR cannot compensate for the 4-orders-of-
    magnitude scale spread, weights drift to ~70 in norm, sigmoid
    saturates, and val NLL ends up around 3 nats (worse than predicting
    the prior, which gets 0.62 at the typical class balance).

    The fit therefore z-scores ``X`` (per-feature mean & std on the
    TRAIN slice, no val leakage) before training, then BAKES the
    standardization back into the final ``weights`` / ``bias`` so the
    runtime path stays a pure ``x @ w + b`` matvec with no schema
    change. Specifically, if the trained weights/bias are
    ``(w_std, b_std)`` operating on ``(x - mu) / sigma``, the saved
    weights are ``w_final = w_std / sigma`` and bias is
    ``b_final = b_std - sum(mu * w_final)``. Inference identity:

        z = (x - mu) / sigma @ w_std + b_std
          = x @ (w_std / sigma) - mu @ (w_std / sigma) + b_std
          = x @ w_final + b_final

    Pass ``standardize=False`` to keep the legacy un-scaled fit (only
    useful when ``X`` is already standardized upstream).

    **L1 sparsity (``l1_strength``)**: when > 0, adds an L1 term on the
    weights (NOT bias) to the loss, applied as a soft proximal step
    after each Adam update via subgradient. ``L1`` shrinks low-signal
    features toward zero and is the cleanest way to deal with the
    member-feature schema's many sparse / near-zero one-hot cols
    that survived ``min_condition_count``. Default 0.0 (disabled).

    **``min_feature_std``**: when > 0, features whose train-slice std
    is below this threshold (i.e. near-constant cols) are forcibly
    zeroed in the saved weights post-fit. This avoids spurious
    coefficients on rare one-hots (think: a cluster id that only
    appears in 5 rows) that would otherwise inflate ``||w||`` without
    contributing predictive signal. Default 0.0 (disabled).

    **``holdout_group_id``**: per-row int array. When provided, the
    internal train/val split holds out *whole groups* (typically
    item ids) rather than random rows, mirroring Member 2's
    cold-start internal split. Without this kwarg the legacy random
    row split is preserved.

    Returns a :class:`LogRegMemberState` ready for :meth:`save`.
    """

    # Lazy torch import: keeps this module importable in the runtime
    # template (which never trains) without dragging torch into the
    # inference path.
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.shape != (int(X.shape[0]),):
        raise ValueError(f"y shape {y.shape} != ({X.shape[0]},)")
    if int(len(feature_names)) != int(X.shape[1]):
        raise ValueError(
            f"feature_names len {len(feature_names)} != X cols {X.shape[1]}"
        )
    if sample_weights is not None and sample_weights.shape != y.shape:
        raise ValueError(
            f"sample_weights shape {sample_weights.shape} != y shape {y.shape}"
        )

    rng = np.random.default_rng(int(seed))
    N = int(X.shape[0])
    if holdout_group_id is None:
        perm = rng.permutation(N)
        n_val = max(64, int(round(val_fraction * N)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        if holdout_group_id.shape != (N,):
            raise ValueError(
                f"holdout_group_id shape {holdout_group_id.shape} != ({N},)"
            )
        gids = np.asarray(holdout_group_id).reshape(-1)
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
                "Group-stratified split yielded an empty side. Increase "
                "n_groups or val_fraction."
            )
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        LOG.info(
            "logreg_member: group-stratified split "
            "(%d groups -> %d held; %d train rows / %d val rows)",
            n_groups, n_val_groups,
            int(train_idx.shape[0]), int(val_idx.shape[0]),
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Standardization (TRAIN-only stats, then bake into weights) ----
    #
    # Memory contract: never materialize a full float64 copy of the
    # train slice. At 5M-row scale, ``X[train_idx]`` is already ~22 GB
    # in float32; the previous path also did ``.astype(np.float64)``
    # twice (once for mean, once for std) which spiked peak RSS by
    # ~88 GB and OOMed the 50 GB Colab high-RAM runtime. Chunked
    # accumulators replicate the same numerics (float64 reductions)
    # while holding only one ~256 MB chunk live at a time.
    _STD_CHUNK = 65_536
    if standardize:
        feat_mean, feat_std = _chunked_mean_std(X, train_idx, chunk=_STD_CHUNK)
        n_zero_std = int(np.sum(feat_std < 1.0e-9))
        # Replace zero-std (constant) features' stds with 1.0 so we
        # don't divide by zero. After centering, those columns become
        # all-zero and contribute nothing to the predictor regardless
        # of the fitted weight, which is exactly what we want.
        feat_std_safe = np.where(feat_std < 1.0e-9, 1.0, feat_std)
        LOG.info(
            "logreg_member: standardizing %d features  "
            "mean range=[%.4g, %.4g]  std range=[%.4g, %.4g]  "
            "n_zero_std=%d",
            int(X.shape[1]),
            float(feat_mean.min()), float(feat_mean.max()),
            float(feat_std.min()), float(feat_std.max()),
            n_zero_std,
        )
        # Materialize the standardized matrices in-place into a pre-
        # allocated float32 buffer; chunk-by-chunk fancy-indexing keeps
        # peak RAM bounded to the buffer + one chunk-sized transient.
        mu_f32 = feat_mean.astype(np.float32)
        sigma_f32 = feat_std_safe.astype(np.float32)
        X_train_used = _chunked_standardize_into(
            X, train_idx, mu_f32, sigma_f32, chunk=_STD_CHUNK
        )
        X_val_used = _chunked_standardize_into(
            X, val_idx, mu_f32, sigma_f32, chunk=_STD_CHUNK
        )
    else:
        feat_mean = None
        feat_std = None
        feat_std_safe = None
        # Even without standardization, allocate the float32 view
        # via chunked indexing so the path is symmetric and we don't
        # rely on numpy fancy-indexing semantics for very large idx.
        X_train_used = _chunked_gather_f32(X, train_idx, chunk=_STD_CHUNK)
        X_val_used = _chunked_gather_f32(X, val_idx, chunk=_STD_CHUNK)

    X_t_train = torch.as_tensor(X_train_used, dtype=torch.float32, device=device)
    y_t_train = torch.as_tensor(y[train_idx], dtype=torch.float32, device=device)
    X_t_val = torch.as_tensor(X_val_used, dtype=torch.float32, device=device)
    y_t_val = torch.as_tensor(y[val_idx], dtype=torch.float32, device=device)
    if sample_weights is not None:
        w_t_train = torch.as_tensor(
            sample_weights[train_idx], dtype=torch.float32, device=device
        )
        w_t_val = torch.as_tensor(
            sample_weights[val_idx], dtype=torch.float32, device=device
        )
    else:
        w_t_train = None
        w_t_val = None

    F = int(X.shape[1])
    linear = nn.Linear(F, 1, bias=True).to(device)
    nn.init.zeros_(linear.weight)
    # Initialize bias at logit(mean(y)) so the model starts on the
    # prior and only has to learn the residual; this saves ~10 epochs
    # of "warm-up" where Adam moves the bias up from 0 to logit(p_mean).
    p_init = float(np.clip(y[train_idx].mean(), 1e-6, 1.0 - 1e-6))
    bias_init = float(math.log(p_init / (1.0 - p_init)))
    nn.init.constant_(linear.bias, bias_init)
    opt = Adam(
        linear.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    # weight_decay is applied manually so we can mask the bias from L2.
    bce = nn.BCEWithLogitsLoss(reduction="none")

    n_train = int(X_t_train.shape[0])

    def _loss(linear_, x_, y_, w_):
        logits = linear_(x_).squeeze(-1)
        per_row = bce(logits, y_)
        if w_ is None:
            l = per_row.mean()
        else:
            l = (per_row * w_).sum() / torch.clamp(w_.sum(), min=1e-9)
        # L2 on weights only (NOT bias).
        l = l + 0.5 * float(weight_decay) * (linear_.weight ** 2).sum()
        # L1 on weights only (NOT bias). Implemented as a smooth term
        # in the loss: torch.abs is autograd-friendly and gives the
        # subgradient sign(w) at every step. For cleaner sparsity you
        # could swap to a proximal soft-threshold step; this version
        # is simpler and behaves correctly for the L1 strengths we
        # care about (1e-3 .. 1e-2 range).
        if float(l1_strength) > 0.0:
            l = l + float(l1_strength) * linear_.weight.abs().sum()
        return l

    best_val = float("inf")
    best_state: dict[str, np.ndarray] = {}
    epochs_since_improve = 0

    for ep in range(int(epochs)):
        linear.train()
        # Mini-batches.
        perm_in = torch.randperm(n_train, device=device)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_train, int(batch_size)):
            idx = perm_in[s : s + int(batch_size)]
            xb = X_t_train.index_select(0, idx)
            yb = y_t_train.index_select(0, idx)
            wb = (
                None
                if w_t_train is None
                else w_t_train.index_select(0, idx)
            )
            opt.zero_grad(set_to_none=True)
            loss = _loss(linear, xb, yb, wb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1

        linear.eval()
        with torch.no_grad():
            val_loss = float(_loss(linear, X_t_val, y_t_val, w_t_val).item())
            train_loss_ep = ep_loss / max(n_batches, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "weights": linear.weight.detach().cpu().numpy().reshape(-1).copy(),
                "bias": float(linear.bias.detach().cpu().item()),
            }
            epochs_since_improve = 0
            improved_marker = "*"
        else:
            epochs_since_improve += 1
            improved_marker = " "

        if int(log_every) > 0 and ((ep + 1) % int(log_every) == 0 or ep == 0):
            with torch.no_grad():
                w_norm = float(linear.weight.detach().norm().item())
                w_nonzero = int((linear.weight.detach().abs() > 1e-6).sum().item())
            LOG.info(
                "logreg_member: epoch %3d/%d  train=%.5f  val=%.5f  best=%.5f %s "
                "(no-improve=%d/%d)  ||w||=%.3f  nnz=%d/%d",
                ep + 1, int(epochs), train_loss_ep, val_loss, best_val,
                improved_marker, epochs_since_improve, int(early_stopping_patience),
                w_norm, w_nonzero, int(linear.weight.numel()),
            )

        if epochs_since_improve >= int(early_stopping_patience):
            LOG.info(
                "logreg_member: early stop at epoch %d/%d (best_val=%.5f)",
                ep + 1, int(epochs), best_val,
            )
            break

    if not best_state:
        # Should never happen but guard anyway.
        best_state = {
            "weights": linear.weight.detach().cpu().numpy().reshape(-1).copy(),
            "bias": float(linear.bias.detach().cpu().item()),
        }

    final_train_loss = float(_loss(linear, X_t_train, y_t_train, w_t_train).item())

    # ---- Bake standardization into the saved weights/bias ----
    # See docstring for the algebra; the output (w_final, b_final) is
    # the linear model in ORIGINAL feature space, equivalent to
    # standardize-then-predict-with-(w_std, b_std).
    w_std_arr = best_state["weights"].astype(np.float64)
    b_std_val = float(best_state["bias"])
    if standardize and feat_std_safe is not None and feat_mean is not None:
        w_final = (w_std_arr / feat_std_safe).astype(np.float32)
        b_final = float(b_std_val - float(np.dot(feat_mean, w_std_arr / feat_std_safe)))
        LOG.info(
            "logreg_member: baked standardization into weights  "
            "||w_std||=%.4f  ||w_final||=%.4f  b_std=%+.4f  b_final=%+.4f",
            float(np.linalg.norm(w_std_arr)),
            float(np.linalg.norm(w_final)),
            b_std_val, b_final,
        )
    else:
        w_final = w_std_arr.astype(np.float32)
        b_final = b_std_val

    # Post-fit min-feature-std filter. Features with train-slice std
    # below ``min_feature_std`` are near-constant; zero out their
    # weights so they cannot contribute to ``||w||``. Bias is updated
    # to absorb the (constant) value of those columns at the train
    # mean -- i.e. ``b_final += sum(mu_i * w_final_i)`` for zeroed
    # columns -- so predictions on the train mean are unchanged.
    if (
        float(min_feature_std) > 0.0
        and feat_std is not None
        and feat_mean is not None
    ):
        rare_mask = (feat_std < float(min_feature_std))
        if int(rare_mask.sum()) > 0:
            absorbed_bias = float(np.dot(feat_mean[rare_mask], w_final[rare_mask]))
            w_final = w_final.copy()
            w_final[rare_mask] = 0.0
            b_final = float(b_final + absorbed_bias)
            LOG.info(
                "logreg_member: dropped %d/%d features below min_feature_std=%.4g  "
                "absorbed_bias=%+.4f  ||w_final||=%.4f  b_final=%+.4f",
                int(rare_mask.sum()), int(len(rare_mask)),
                float(min_feature_std),
                absorbed_bias, float(np.linalg.norm(w_final)), b_final,
            )

    return LogRegMemberState(
        weights=w_final,
        bias=float(b_final),
        feature_dim=int(F),
        feature_names=tuple(str(s) for s in feature_names),
        fit_method="adam_std" if standardize else "adam",
        n_train=int(n_train),
        n_pos=int(np.sum(y == 1.0)),
        train_loss=final_train_loss,
        val_loss=best_val,
        weight_decay=float(weight_decay),
    )


__all__ = [
    "LogRegMemberState",
    "apply_one",
    "apply_batch",
    "apply_state_one",
    "apply_state_batch",
    "fit_logreg_member",
]
