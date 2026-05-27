"""Stacker for the four-member ensemble (Phase 4 of the upgrade).

Replaces the legacy ``coverage_blend`` (two scalar weights gated by
``bench_present``) with an out-of-fold logistic regression on the
four member predictions plus coverage and a small handful of
NN/centroid features. Coverage enters as a feature, not as a hard
gate -- one stacker, no `w_present`/`w_missing` split.

Architecture
------------
Inputs to the stacker (locked column order, same offline + runtime):
  [0]  logit(p_member1)        # IRT-MLP backbone
  [1]  logit(p_member2)        # GBDT
  [2]  logit(p_member3)        # FAISS-free kNN
  [3]  logit(p_member4)        # logistic regression
  [4]  chi_bench_present       # coverage indicator (0 or 1)
  [5]  nn_neighbor_support     # NN feature: log1p(n_labeled_neighbors)
  [6]  nn_mean_similarity      # NN feature: avg cosine over top-K
  [7]  centroid_distance       # min cluster distance from item_features

Output: a single sigmoid -> calibrated-ish blended probability. The
NN-residual calibrator (Phase 4b) is then applied ONCE on top of
this output.

Offline trainer
---------------
The trainer uses out-of-fold predictions for each member and
synthesizes benchmark-cold-start rows so the stacker learns a
coverage-dependent weighting on chi=0 rows. Hand-rolled Adam
optimizer in PyTorch with L2 weight decay (ridge); we don't reach for
``torch.optim`` so we can dump simple weight-vector + bias state.

Runtime inference
-----------------
Pure NumPy: weights * logit_features -> sigmoid -> clamp to (eps, 1-eps).
No torch dependency at runtime -- this means even if the runtime
container has a broken torch install, the stacker still works.

Determinism
-----------
``apply_one(state, feats)`` returns the same value across runs for
the same inputs (no RNG, no hidden state).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("stacker")

_EPS = 1.0e-6


# Locked feature schema ----------------------------------------------------
#
# The legacy 4-member schema (kept as the module-level default for backward
# compatibility with pre-Task-4 callers/bundles). For Task 4 (Member 5 -- the
# difficulty-projected kNN) we allow a 5-member variant, and the builder
# below accepts an arbitrary member count so the only thing the notebook has
# to do is pass an [N, 5] member_probs matrix.

STACKER_FEATURE_NAMES: tuple[str, ...] = (
    "logit_member1",
    "logit_member2",
    "logit_member3",
    "logit_member4",
    "bench_present",
    "nn_neighbor_support",
    "nn_mean_similarity",
    "centroid_distance",
)
STACKER_FEATURE_DIM: int = len(STACKER_FEATURE_NAMES)

_NUM_AUX_FEATURES: int = 4  # bench_present + 3 NN/centroid features


def stacker_feature_names(n_members: int) -> tuple[str, ...]:
    """Return the canonical column names for a stacker with ``n_members`` members.

    The trailing 4 columns (bench_present, nn_neighbor_support,
    nn_mean_similarity, centroid_distance) are fixed; the first
    ``n_members`` columns are ``logit_memberN`` in 1-indexed order.
    """
    if int(n_members) < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    member_cols = tuple(f"logit_member{i + 1}" for i in range(int(n_members)))
    aux_cols = (
        "bench_present",
        "nn_neighbor_support",
        "nn_mean_similarity",
        "centroid_distance",
    )
    return member_cols + aux_cols


def stacker_feature_dim(n_members: int) -> int:
    """Total stacker feature dimensionality for ``n_members`` members."""
    return int(n_members) + _NUM_AUX_FEATURES


def logit_clipped(p: np.ndarray | float, eps: float = _EPS) -> np.ndarray | float:
    """Numerically stable logit. p is clipped to (eps, 1-eps)."""
    if isinstance(p, (int, float)):
        x = max(min(float(p), 1.0 - eps), eps)
        return math.log(x / (1.0 - x))
    arr = np.asarray(p, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class StackerState:
    """Fitted stacker -- shipped to runtime."""

    weights: np.ndarray            # [F] fp32
    bias: float
    feature_names: tuple[str, ...]
    feature_dim: int
    l2: float
    n_train: int
    n_pos: int
    train_loss: float
    val_loss: float
    n_iters: int

    def __post_init__(self) -> None:
        if int(self.weights.shape[0]) != int(self.feature_dim):
            raise ValueError(
                f"weights len {self.weights.shape[0]} != feature_dim "
                f"{self.feature_dim}"
            )
        if int(len(self.feature_names)) != int(self.feature_dim):
            raise ValueError(
                f"feature_names len {len(self.feature_names)} != feature_dim "
                f"{self.feature_dim}"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("StackerState weights contain NaN/Inf")
        if not math.isfinite(float(self.bias)):
            raise ValueError("StackerState bias is NaN/Inf")

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "stacker_state.npz",
            weights=self.weights.astype(np.float32),
        )
        meta = {
            "bias": float(self.bias),
            "feature_names": list(self.feature_names),
            "feature_dim": int(self.feature_dim),
            "l2": float(self.l2),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "n_iters": int(self.n_iters),
            "format_version": 1,
        }
        (out / "stacker_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "StackerState":
        d = Path(in_dir)
        meta = json.loads((d / "stacker_meta.json").read_text(encoding="utf-8"))
        with np.load(d / "stacker_state.npz") as npz:
            weights = npz["weights"].astype(np.float32, copy=False)
        return cls(
            weights=weights,
            bias=float(meta["bias"]),
            feature_names=tuple(meta["feature_names"]),
            feature_dim=int(meta["feature_dim"]),
            l2=float(meta["l2"]),
            n_train=int(meta["n_train"]),
            n_pos=int(meta["n_pos"]),
            train_loss=float(meta["train_loss"]),
            val_loss=float(meta["val_loss"]),
            n_iters=int(meta["n_iters"]),
        )

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.astype(np.float32).tolist(),
            "bias": float(self.bias),
            "feature_names": list(self.feature_names),
            "feature_dim": int(self.feature_dim),
            "l2": float(self.l2),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "n_iters": int(self.n_iters),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StackerState":
        return cls(
            weights=np.asarray(d["weights"], dtype=np.float32),
            bias=float(d["bias"]),
            feature_names=tuple(d["feature_names"]),
            feature_dim=int(d["feature_dim"]),
            l2=float(d["l2"]),
            n_train=int(d["n_train"]),
            n_pos=int(d["n_pos"]),
            train_loss=float(d["train_loss"]),
            val_loss=float(d["val_loss"]),
            n_iters=int(d["n_iters"]),
        )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _sigmoid_stable_one(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _sigmoid_stable_arr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    pos = z >= 0
    out = np.empty_like(z)
    # avoid overflow via piecewise
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def apply_one(state: StackerState, features: np.ndarray) -> float:
    """Single-row inference -> Python float in (eps, 1-eps)."""
    if features.ndim != 1:
        raise ValueError(f"features must be 1D, got {features.shape}")
    if int(features.shape[0]) != int(state.feature_dim):
        raise ValueError(
            f"features dim {features.shape[0]} != state.feature_dim "
            f"{state.feature_dim}"
        )
    x = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        # NaN / Inf in features is recoverable: fill with 0 so the
        # bias dominates. This prevents a single-feature failure from
        # crashing the whole runtime.
        x = np.where(np.isfinite(x), x, 0.0)
    z = float(np.dot(state.weights.astype(np.float64), x) + float(state.bias))
    if not math.isfinite(z):
        return 0.5
    p = _sigmoid_stable_one(z)
    return float(min(max(p, _EPS), 1.0 - _EPS))


def apply_batch(state: StackerState, features_matrix: np.ndarray) -> np.ndarray:
    if features_matrix.ndim != 2:
        raise ValueError("features_matrix must be 2D")
    if int(features_matrix.shape[1]) != int(state.feature_dim):
        raise ValueError(
            f"features_matrix dim {features_matrix.shape[1]} != "
            f"state.feature_dim {state.feature_dim}"
        )
    X = np.asarray(features_matrix, dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    z = X @ state.weights.astype(np.float64) + float(state.bias)
    p = _sigmoid_stable_arr(z)
    return np.clip(p, _EPS, 1.0 - _EPS).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Stacker feature construction
# ---------------------------------------------------------------------------


def build_stacker_features(
    *,
    member_probs: np.ndarray,         # [N, M] in (0, 1); M = #members (4 or 5)
    bench_present: np.ndarray,        # [N] in {0, 1}
    nn_neighbor_support: np.ndarray,  # [N] non-negative
    nn_mean_similarity: np.ndarray,   # [N] in [-1, 1] for cosine
    centroid_distance: np.ndarray,    # [N] non-negative
) -> np.ndarray:
    """Assemble the [N, M+4] stacker feature matrix.

    ``M`` is the number of member columns (4 for the legacy ensemble,
    5 once Task 4's difficulty-projected kNN is enabled). Member probs
    are converted to logit space here (clipped to (eps, 1-eps)) -- the
    user spec mandates logit-space inputs because bce-on-sigmoid is
    well-conditioned there and a stacker fed raw probabilities tends
    to learn a near-linear combination that biases toward the prior
    at the boundaries.
    """
    if member_probs.ndim != 2:
        raise ValueError(
            f"member_probs must be 2D [N, M], got {member_probs.shape}"
        )
    if member_probs.shape[1] < 1:
        raise ValueError(
            f"member_probs must have >= 1 member columns, got {member_probs.shape}"
        )
    N = int(member_probs.shape[0])
    M = int(member_probs.shape[1])
    for arr_name, arr in [
        ("bench_present", bench_present),
        ("nn_neighbor_support", nn_neighbor_support),
        ("nn_mean_similarity", nn_mean_similarity),
        ("centroid_distance", centroid_distance),
    ]:
        a = np.asarray(arr).reshape(-1)
        if int(a.shape[0]) != N:
            raise ValueError(
                f"{arr_name} length {a.shape[0]} != N {N}"
            )

    out = np.empty((N, M + _NUM_AUX_FEATURES), dtype=np.float32)
    out[:, 0:M] = logit_clipped(np.asarray(member_probs, dtype=np.float64)).astype(
        np.float32
    )
    out[:, M + 0] = np.asarray(bench_present, dtype=np.float32).reshape(-1)
    out[:, M + 1] = np.asarray(nn_neighbor_support, dtype=np.float32).reshape(-1)
    out[:, M + 2] = np.asarray(nn_mean_similarity, dtype=np.float32).reshape(-1)
    out[:, M + 3] = np.asarray(centroid_distance, dtype=np.float32).reshape(-1)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    return out


def build_stacker_features_one(
    *,
    member_probs: Sequence[float],
    bench_present: float,
    nn_neighbor_support: float,
    nn_mean_similarity: float,
    centroid_distance: float,
) -> np.ndarray:
    """Single-row builder; mirrors ``build_stacker_features`` exactly.

    ``member_probs`` length is read at call time, so the same function
    handles both the legacy 4-member case and the Task-4 5-member case.
    """
    M = int(len(member_probs))
    if M < 1:
        raise ValueError(
            f"member_probs must have >= 1 entries, got {M}"
        )
    feats = np.empty(M + _NUM_AUX_FEATURES, dtype=np.float32)
    for i, p in enumerate(member_probs):
        feats[i] = float(logit_clipped(float(p)))
    feats[M + 0] = float(bench_present)
    feats[M + 1] = float(nn_neighbor_support)
    feats[M + 2] = float(nn_mean_similarity)
    feats[M + 3] = float(centroid_distance)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32, copy=False)
    return feats


# ---------------------------------------------------------------------------
# Offline trainer (hand-rolled torch logistic regression with ridge)
# ---------------------------------------------------------------------------


def fit_stacker(
    *,
    X: np.ndarray,                    # [N, F] float32, OOF features
    y: np.ndarray,                    # [N] float32 in {0, 1}
    feature_names: Sequence[str] | None = None,
    sample_weights: np.ndarray | None = None,
    val_fraction: float = 0.15,
    n_iters: int = 4000,
    learning_rate: float = 0.05,
    l2: float = 1.0,
    early_stopping_patience: int = 200,
    device: str = "cpu",
    seed: int = 0,
) -> StackerState:
    """Fit a hand-rolled torch logistic regression on stacker features.

    Uses Adam with L2 weight decay applied as a ridge penalty in the
    loss directly (so the weight-decay -> regularization equivalence
    is unambiguous, unlike torch.optim's mixed weight_decay handling).

    The loss is:
        L = mean(BCE(sigmoid(X w + b), y) * sample_weights) + (l2/N) * ||w||^2
    """
    import torch

    if feature_names is None:
        feature_names = STACKER_FEATURE_NAMES
    F = int(X.shape[1])
    if int(len(feature_names)) != F:
        raise ValueError(
            f"feature_names len {len(feature_names)} != X cols {F}"
        )

    rng = np.random.default_rng(int(seed))
    N = int(X.shape[0])
    perm = rng.permutation(N)
    n_val = max(64, int(round(val_fraction * N)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if sample_weights is None:
        sw = np.ones(N, dtype=np.float32)
    else:
        sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if int(sw.shape[0]) != N:
            raise ValueError(
                f"sample_weights len {sw.shape[0]} != N {N}"
            )

    dev = torch.device(device)
    Xt = torch.tensor(X.astype(np.float32, copy=False), device=dev)
    yt = torch.tensor(y.astype(np.float32, copy=False), device=dev)
    swt = torch.tensor(sw, device=dev)
    Xtr = Xt[train_idx]
    ytr = yt[train_idx]
    swtr = swt[train_idx]
    Xva = Xt[val_idx]
    yva = yt[val_idx]
    swva = swt[val_idx]

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.zeros(F, dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, device=dev, requires_grad=True)

    optimizer = torch.optim.Adam([w, b], lr=float(learning_rate))

    def _eval_loss(Xs: "torch.Tensor", ys: "torch.Tensor", sws: "torch.Tensor") -> float:
        with torch.no_grad():
            logits = Xs @ w + b
            p = torch.sigmoid(logits)
            p = torch.clamp(p, _EPS, 1.0 - _EPS)
            bce = -(ys * torch.log(p) + (1.0 - ys) * torch.log(1.0 - p))
            return float((sws * bce).sum().item() / max(float(sws.sum().item()), 1.0))

    best_val = float("inf")
    best_w: np.ndarray = np.zeros(F, dtype=np.float32)
    best_b: float = 0.0
    best_iter = 0
    no_improve = 0

    n_train_eff = float(swtr.sum().item())
    if n_train_eff <= 0:
        raise ValueError("sample_weights sum to <= 0 on training split")

    for it in range(int(n_iters)):
        optimizer.zero_grad()
        logits = Xtr @ w + b
        # BCE with logits (numerically stable):
        # loss = log(1 + exp(-yt*logit)) + (1-yt)*logit
        bce = (
            torch.clamp(logits, min=0.0)
            - logits * ytr
            + torch.log1p(torch.exp(-torch.abs(logits)))
        )
        loss = (swtr * bce).sum() / n_train_eff
        ridge = (float(l2) / max(N, 1)) * (w * w).sum()
        total = loss + ridge
        total.backward()
        optimizer.step()

        if (it + 1) % 25 == 0 or it + 1 == n_iters:
            val_loss = _eval_loss(Xva, yva, swva)
            if val_loss + 1.0e-6 < best_val:
                best_val = val_loss
                best_w = w.detach().cpu().numpy().astype(np.float32)
                best_b = float(b.detach().cpu().numpy())
                best_iter = it + 1
                no_improve = 0
            else:
                no_improve += 25
                if no_improve >= int(early_stopping_patience):
                    LOG.info(
                        "Stacker early-stopped at iter %d (best at iter %d, "
                        "val_loss=%.5f)",
                        it + 1,
                        best_iter,
                        best_val,
                    )
                    break

    train_loss = _eval_loss(Xtr, yt[train_idx], swt[train_idx])
    state = StackerState(
        weights=best_w,
        bias=float(best_b),
        feature_names=tuple(str(s) for s in feature_names),
        feature_dim=F,
        l2=float(l2),
        n_train=int(N),
        n_pos=int(np.sum(y == 1.0)),
        train_loss=float(train_loss),
        val_loss=float(best_val),
        n_iters=int(best_iter),
    )
    LOG.info(
        "Stacker fit OK: F=%d N=%d val_loss=%.5f weights=%s bias=%.4f",
        F,
        N,
        best_val,
        np.array2string(best_w, precision=3, max_line_width=120),
        best_b,
    )
    return state


# ---------------------------------------------------------------------------
# Out-of-fold helpers (no leakage)
# ---------------------------------------------------------------------------


def make_kfold_split(
    *,
    item_keys: Sequence[str],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold split that preserves item cold-start within each fold.

    Splits the UNIQUE item_keys into K folds; for each fold returns
    ``(train_row_idx, val_row_idx)`` over the FULL row index space
    (rows are (subject, item) pairs). Held-out items are entirely
    disjoint from the fold's training items.

    Returns:
        list of (train_idx, val_idx) tuples, one per fold.
    """
    keys = np.asarray([str(k) for k in item_keys])
    unique_items = np.unique(keys)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(unique_items.shape[0])
    items_shuffled = unique_items[perm]
    folds_items = np.array_split(items_shuffled, int(n_folds))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in folds_items:
        val_mask = np.isin(keys, f)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        folds.append((train_idx, val_idx))
    return folds


def assert_no_item_leakage(
    item_keys: Sequence[str],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """RED-TEAM helper for the user spec: prove each fold's val items
    are disjoint from its train items. Raises on leakage."""
    keys = np.asarray([str(k) for k in item_keys])
    for f_idx, (tr, va) in enumerate(folds):
        tr_items = set(keys[tr].tolist())
        va_items = set(keys[va].tolist())
        inter = tr_items & va_items
        if inter:
            raise RuntimeError(
                f"Fold {f_idx}: leakage -- {len(inter)} items appear in both "
                f"train and val sets. Sample: {list(inter)[:5]}"
            )


def assert_oof_covers_all_rows(
    n_rows: int,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """RED-TEAM helper: confirm OOF predictions cover 100% of rows
    exactly once."""
    coverage = np.zeros(int(n_rows), dtype=np.int32)
    for tr, va in folds:
        coverage[va] += 1
    not_covered = int(np.sum(coverage == 0))
    multi_covered = int(np.sum(coverage > 1))
    if not_covered > 0 or multi_covered > 0:
        raise RuntimeError(
            f"OOF coverage check failed: not_covered={not_covered}, "
            f"multi_covered={multi_covered} of {n_rows} rows."
        )


__all__ = [
    "STACKER_FEATURE_NAMES",
    "STACKER_FEATURE_DIM",
    "StackerState",
    "apply_one",
    "apply_batch",
    "build_stacker_features",
    "build_stacker_features_one",
    "fit_stacker",
    "logit_clipped",
    "make_kfold_split",
    "assert_no_item_leakage",
    "assert_oof_covers_all_rows",
]
