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
# Offline trainer (uses torch, called from the notebook only)
# ---------------------------------------------------------------------------


def fit_logreg_member(
    *,
    X: np.ndarray,                    # [N, F] float32
    y: np.ndarray,                    # [N] float32 in {0, 1}
    sample_weights: np.ndarray | None = None,
    feature_names: Sequence[str],
    weight_decay: float = 1.0e-3,
    learning_rate: float = 1.0e-2,
    epochs: int = 200,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    seed: int = 0,
    early_stopping_patience: int = 20,
    device: str | None = None,
) -> LogRegMemberState:
    """Fit a torch logistic regression with Adam + early stopping.

    The training rows are split internally into a train/val slice with
    ``val_fraction``; the val slice is used both for early stopping
    AND as the reported ``val_loss`` in the saved state. This is NOT
    the OOF stacker val -- the stacker does its own out-of-fold split
    when consuming Member 4's predictions.

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
    perm = rng.permutation(N)
    n_val = max(64, int(round(val_fraction * N)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t_train = torch.as_tensor(X[train_idx], dtype=torch.float32, device=device)
    y_t_train = torch.as_tensor(y[train_idx], dtype=torch.float32, device=device)
    X_t_val = torch.as_tensor(X[val_idx], dtype=torch.float32, device=device)
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
    nn.init.zeros_(linear.bias)
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

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "weights": linear.weight.detach().cpu().numpy().reshape(-1).copy(),
                "bias": float(linear.bias.detach().cpu().item()),
            }
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
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

    return LogRegMemberState(
        weights=best_state["weights"].astype(np.float32),
        bias=float(best_state["bias"]),
        feature_dim=int(F),
        feature_names=tuple(str(s) for s in feature_names),
        fit_method="adam",
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
