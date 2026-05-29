"""Member 2 (replacement): metadata-only MLP with GLU activations.

Replaces the previous Member 2 (LightGBM on the dense embedding matrix).
The previous M2 architecture was a structural mismatch: GBDTs split one
axis at a time, but the dense matrix is mostly smooth embedding dims
with no natural cutpoints. Even with PCA the OOF NLL plateaued at ~0.72.

This module is the M2 replacement. It deliberately consumes **no item
embedding columns** -- the inductive bias is "what we know about this
(subject, benchmark, cluster) cell, completely independent of item
content." That makes its errors structurally orthogonal to M1 (deep on
embeddings), M3 (kNN on embeddings), M4 (linear on embeddings), and
M6 (bilinear on embeddings).

Architecture
------------
Per-row inputs:
    subject_id   -> learned embedding [d_subj]   (with categorical dropout)
    bc_id        -> learned embedding [d_bc]     (with categorical dropout)
    cluster_id   -> learned embedding [d_cluster](with categorical dropout)
    marginals    -> z-scored [n_marginals]       (the 14 mean-encoded cols
                                                  already built for M4)

Each categorical embedding table has a trailing UNK slot at index
``n_categories`` that the dataloader maps unknown / dropout-replaced IDs
to. The same UNK index is used at inference for cold-start (unseen
subject / bc / cluster).

Body: two GLU blocks. A "GLU" block is
    h = (W_value @ x + b_value) * sigmoid(W_gate @ x + b_gate)
i.e. the classic Dauphin et al. (2017) gated linear unit. Compared to a
plain Linear -> activation, GLU lets the network learn an input-dependent
gate that selectively masks dims of the value stream. For a small (~200k
parameter) metadata MLP this is the right efficiency knob -- you get
5-10% lower val loss for the same parameter count.

Head: single linear -> raw logit. Probability is sigmoid(logit) and is
clipped to (eps, 1-eps) by the apply functions.

Inference
---------
Pure numpy. ``apply_state_batch(state, ...)`` for offline rescoring;
``apply_state_one(state, ...)`` for the single-row runtime path. The
torch dependency is **fit-time only**; the runtime bundle never imports
torch.

Cold-start contract
-------------------
Categorical dropout during training (5% subject / 10% bc / 10% cluster
by default) teaches the model to fall through to the UNK embedding
without losing too much signal. At inference, any subject_id / bc_id /
cluster_id outside ``[0, n_categories)`` is mapped to the UNK slot.

Why this and not M1's "gated_mlp"
---------------------------------
M1's gating is between an *embedding path* and a *metadata path* (use
embedding when in-distribution, fall back to metadata when not). This
model has no embedding path -- the gate has nothing to switch between.
Categorical dropout is the structurally-correct way to handle cold-start
when all inputs are categorical / shrunken-statistic.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("member2_metadata_mlp")

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Numeric helpers (shared between trainer and pure-numpy inference)
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


def _logit_clipped_scalar(p: float, eps: float = _EPS) -> float:
    x = max(min(float(p), 1.0 - eps), eps)
    return math.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Member2MLPState:
    """Fitted state of the metadata-only Member 2 MLP.

    The categorical embedding tables include a trailing UNK row at the
    final index, used both for cold-start (unseen IDs at inference) and
    for the categorical-dropout training trick.
    """

    # Categorical embedding tables ([n_categories + 1, d]). The trailing
    # row is UNK.
    subject_emb: np.ndarray            # [n_subjects + 1, d_subj]   float32
    bc_emb: np.ndarray                 # [n_bcs + 1,      d_bc]     float32
    cluster_emb: np.ndarray            # [n_clusters + 1, d_cluster]float32

    # GLU block 1: in_dim = d_subj + d_bc + d_cluster + n_marginals -> hid1.
    l1_value_W: np.ndarray             # [in_dim, hid1] float32
    l1_value_b: np.ndarray             # [hid1]         float32
    l1_gate_W: np.ndarray              # [in_dim, hid1] float32
    l1_gate_b: np.ndarray              # [hid1]         float32

    # GLU block 2: hid1 -> hid2.
    l2_value_W: np.ndarray             # [hid1, hid2]   float32
    l2_value_b: np.ndarray             # [hid2]         float32
    l2_gate_W: np.ndarray              # [hid1, hid2]   float32
    l2_gate_b: np.ndarray              # [hid2]         float32

    # Output head: hid2 -> 1.
    head_W: np.ndarray                 # [hid2, 1]      float32
    head_b: float

    # Standardization for the dense marginal features (so the apply path
    # mirrors what the network saw at fit time).
    marg_mean: np.ndarray              # [n_marginals]  float32
    marg_std: np.ndarray               # [n_marginals]  float32

    # Provenance / runtime indexing metadata.
    subject_keys: tuple[str, ...]
    bc_keys: tuple[str, ...]
    n_subjects: int
    n_bcs: int
    n_clusters: int
    d_subj: int
    d_bc: int
    d_cluster: int
    hid1: int
    hid2: int
    marg_feature_names: tuple[str, ...]
    n_marginals: int

    # Fit diagnostics.
    fit_method: str
    n_train: int
    n_pos: int
    train_loss: float
    val_loss: float
    cat_dropout_subject: float
    cat_dropout_bc: float
    cat_dropout_cluster: float
    weight_decay: float
    learning_rate: float
    epochs_run: int

    def __post_init__(self) -> None:
        # Shapes.
        if self.subject_emb.shape != (int(self.n_subjects) + 1, int(self.d_subj)):
            raise ValueError(
                f"subject_emb shape {self.subject_emb.shape} != "
                f"({self.n_subjects + 1}, {self.d_subj})"
            )
        if self.bc_emb.shape != (int(self.n_bcs) + 1, int(self.d_bc)):
            raise ValueError(
                f"bc_emb shape {self.bc_emb.shape} != "
                f"({self.n_bcs + 1}, {self.d_bc})"
            )
        if self.cluster_emb.shape != (int(self.n_clusters) + 1, int(self.d_cluster)):
            raise ValueError(
                f"cluster_emb shape {self.cluster_emb.shape} != "
                f"({self.n_clusters + 1}, {self.d_cluster})"
            )
        in_dim = int(self.d_subj) + int(self.d_bc) + int(self.d_cluster) + int(self.n_marginals)
        if self.l1_value_W.shape != (in_dim, int(self.hid1)):
            raise ValueError(
                f"l1_value_W shape {self.l1_value_W.shape} != "
                f"({in_dim}, {self.hid1})"
            )
        if self.l1_gate_W.shape != self.l1_value_W.shape:
            raise ValueError("l1_gate_W shape != l1_value_W shape")
        if self.l1_value_b.shape != (int(self.hid1),):
            raise ValueError(
                f"l1_value_b shape {self.l1_value_b.shape} != ({self.hid1},)"
            )
        if self.l1_gate_b.shape != self.l1_value_b.shape:
            raise ValueError("l1_gate_b shape != l1_value_b shape")
        if self.l2_value_W.shape != (int(self.hid1), int(self.hid2)):
            raise ValueError(
                f"l2_value_W shape {self.l2_value_W.shape} != "
                f"({self.hid1}, {self.hid2})"
            )
        if self.l2_gate_W.shape != self.l2_value_W.shape:
            raise ValueError("l2_gate_W shape != l2_value_W shape")
        if self.l2_value_b.shape != (int(self.hid2),):
            raise ValueError(
                f"l2_value_b shape {self.l2_value_b.shape} != ({self.hid2},)"
            )
        if self.l2_gate_b.shape != self.l2_value_b.shape:
            raise ValueError("l2_gate_b shape != l2_value_b shape")
        if self.head_W.shape != (int(self.hid2), 1):
            raise ValueError(
                f"head_W shape {self.head_W.shape} != ({self.hid2}, 1)"
            )
        if self.marg_mean.shape != (int(self.n_marginals),):
            raise ValueError(
                f"marg_mean shape {self.marg_mean.shape} != ({self.n_marginals},)"
            )
        if self.marg_std.shape != self.marg_mean.shape:
            raise ValueError("marg_std shape != marg_mean shape")
        # Provenance.
        if int(len(self.subject_keys)) != int(self.n_subjects):
            raise ValueError(
                f"subject_keys len {len(self.subject_keys)} != "
                f"n_subjects {self.n_subjects}"
            )
        if int(len(self.bc_keys)) != int(self.n_bcs):
            raise ValueError(
                f"bc_keys len {len(self.bc_keys)} != n_bcs {self.n_bcs}"
            )
        if int(len(self.marg_feature_names)) != int(self.n_marginals):
            raise ValueError(
                f"marg_feature_names len {len(self.marg_feature_names)} != "
                f"n_marginals {self.n_marginals}"
            )
        # Numerical sanity.
        if not math.isfinite(float(self.head_b)):
            raise ValueError("head_b is NaN/Inf")
        for name, arr in (
            ("subject_emb", self.subject_emb),
            ("bc_emb", self.bc_emb),
            ("cluster_emb", self.cluster_emb),
            ("l1_value_W", self.l1_value_W), ("l1_value_b", self.l1_value_b),
            ("l1_gate_W", self.l1_gate_W), ("l1_gate_b", self.l1_gate_b),
            ("l2_value_W", self.l2_value_W), ("l2_value_b", self.l2_value_b),
            ("l2_gate_W", self.l2_gate_W), ("l2_gate_b", self.l2_gate_b),
            ("head_W", self.head_W),
            ("marg_mean", self.marg_mean), ("marg_std", self.marg_std),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Member2MLPState: {name} contains NaN/Inf")
        if not np.all(self.marg_std > 0):
            raise ValueError(
                "Member2MLPState: marg_std must be strictly positive (replace "
                "zero-std columns with 1.0 at fit time)."
            )

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "weights.npz",
            subject_emb=self.subject_emb.astype(np.float32),
            bc_emb=self.bc_emb.astype(np.float32),
            cluster_emb=self.cluster_emb.astype(np.float32),
            l1_value_W=self.l1_value_W.astype(np.float32),
            l1_value_b=self.l1_value_b.astype(np.float32),
            l1_gate_W=self.l1_gate_W.astype(np.float32),
            l1_gate_b=self.l1_gate_b.astype(np.float32),
            l2_value_W=self.l2_value_W.astype(np.float32),
            l2_value_b=self.l2_value_b.astype(np.float32),
            l2_gate_W=self.l2_gate_W.astype(np.float32),
            l2_gate_b=self.l2_gate_b.astype(np.float32),
            head_W=self.head_W.astype(np.float32),
            marg_mean=self.marg_mean.astype(np.float32),
            marg_std=self.marg_std.astype(np.float32),
        )
        meta = {
            "head_b": float(self.head_b),
            "subject_keys": list(self.subject_keys),
            "bc_keys": list(self.bc_keys),
            "n_subjects": int(self.n_subjects),
            "n_bcs": int(self.n_bcs),
            "n_clusters": int(self.n_clusters),
            "d_subj": int(self.d_subj),
            "d_bc": int(self.d_bc),
            "d_cluster": int(self.d_cluster),
            "hid1": int(self.hid1),
            "hid2": int(self.hid2),
            "marg_feature_names": list(self.marg_feature_names),
            "n_marginals": int(self.n_marginals),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "cat_dropout_subject": float(self.cat_dropout_subject),
            "cat_dropout_bc": float(self.cat_dropout_bc),
            "cat_dropout_cluster": float(self.cat_dropout_cluster),
            "weight_decay": float(self.weight_decay),
            "learning_rate": float(self.learning_rate),
            "epochs_run": int(self.epochs_run),
            "format_version": 1,
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "Member2MLPState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz") as npz:
            return cls(
                subject_emb=npz["subject_emb"].astype(np.float32, copy=False),
                bc_emb=npz["bc_emb"].astype(np.float32, copy=False),
                cluster_emb=npz["cluster_emb"].astype(np.float32, copy=False),
                l1_value_W=npz["l1_value_W"].astype(np.float32, copy=False),
                l1_value_b=npz["l1_value_b"].astype(np.float32, copy=False),
                l1_gate_W=npz["l1_gate_W"].astype(np.float32, copy=False),
                l1_gate_b=npz["l1_gate_b"].astype(np.float32, copy=False),
                l2_value_W=npz["l2_value_W"].astype(np.float32, copy=False),
                l2_value_b=npz["l2_value_b"].astype(np.float32, copy=False),
                l2_gate_W=npz["l2_gate_W"].astype(np.float32, copy=False),
                l2_gate_b=npz["l2_gate_b"].astype(np.float32, copy=False),
                head_W=npz["head_W"].astype(np.float32, copy=False),
                head_b=float(meta["head_b"]),
                marg_mean=npz["marg_mean"].astype(np.float32, copy=False),
                marg_std=npz["marg_std"].astype(np.float32, copy=False),
                subject_keys=tuple(str(s) for s in meta["subject_keys"]),
                bc_keys=tuple(str(s) for s in meta["bc_keys"]),
                n_subjects=int(meta["n_subjects"]),
                n_bcs=int(meta["n_bcs"]),
                n_clusters=int(meta["n_clusters"]),
                d_subj=int(meta["d_subj"]),
                d_bc=int(meta["d_bc"]),
                d_cluster=int(meta["d_cluster"]),
                hid1=int(meta["hid1"]),
                hid2=int(meta["hid2"]),
                marg_feature_names=tuple(
                    str(s) for s in meta["marg_feature_names"]
                ),
                n_marginals=int(meta["n_marginals"]),
                fit_method=str(meta.get("fit_method", "unknown")),
                n_train=int(meta.get("n_train", 0)),
                n_pos=int(meta.get("n_pos", 0)),
                train_loss=float(meta.get("train_loss", 0.0)),
                val_loss=float(meta.get("val_loss", 0.0)),
                cat_dropout_subject=float(meta.get("cat_dropout_subject", 0.0)),
                cat_dropout_bc=float(meta.get("cat_dropout_bc", 0.0)),
                cat_dropout_cluster=float(meta.get("cat_dropout_cluster", 0.0)),
                weight_decay=float(meta.get("weight_decay", 0.0)),
                learning_rate=float(meta.get("learning_rate", 0.0)),
                epochs_run=int(meta.get("epochs_run", 0)),
            )


# ---------------------------------------------------------------------------
# Pure-numpy inference
# ---------------------------------------------------------------------------


def _safe_cat_lookup(
    ids: np.ndarray,
    table: np.ndarray,
    n_known: int,
) -> np.ndarray:
    """Look up ``ids`` in ``table`` (shape ``[n_known + 1, d]``).

    IDs outside ``[0, n_known)`` route to the trailing UNK row at index
    ``n_known``. Returns a ``[N, d]`` float32 array.
    """
    ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    valid = (ids_arr >= 0) & (ids_arr < int(n_known))
    safe_ids = np.where(valid, ids_arr, int(n_known)).astype(np.int64)
    return table[safe_ids].astype(np.float32, copy=False)


def apply_batch(
    *,
    state: "Member2MLPState",
    subject_ids: np.ndarray,
    bc_ids: np.ndarray,
    cluster_ids: np.ndarray,
    marginals: np.ndarray,
) -> np.ndarray:
    """Pure-numpy forward pass for a batch. Returns ``[N]`` float32 probs.

    Parameters
    ----------
    subject_ids, bc_ids, cluster_ids
        ``[N]`` int arrays. Values outside the valid range for each
        table are routed to the UNK slot at index ``n_categories``.
    marginals
        ``[N, n_marginals]`` float array. Z-scored using the stored
        ``marg_mean`` / ``marg_std`` from fit time.
    """
    s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    c = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if s.shape != b.shape or s.shape != c.shape:
        raise ValueError(
            f"subject_ids {s.shape}, bc_ids {b.shape}, "
            f"cluster_ids {c.shape} must be same length"
        )
    M = np.asarray(marginals, dtype=np.float32)
    if M.ndim != 2 or int(M.shape[0]) != int(s.shape[0]):
        raise ValueError(
            f"marginals shape {M.shape} must be (N, n_marginals) "
            f"matching subject_ids length {s.shape[0]}"
        )
    if int(M.shape[1]) != int(state.n_marginals):
        raise ValueError(
            f"marginals has {M.shape[1]} cols but state was fit on "
            f"{state.n_marginals}"
        )

    subj_e = _safe_cat_lookup(s, state.subject_emb, state.n_subjects)
    bc_e = _safe_cat_lookup(b, state.bc_emb, state.n_bcs)
    cl_e = _safe_cat_lookup(c, state.cluster_emb, state.n_clusters)
    Mz = (M - state.marg_mean.astype(np.float32)) / state.marg_std.astype(np.float32)
    Mz = np.where(np.isfinite(Mz), Mz, 0.0).astype(np.float32, copy=False)

    x = np.concatenate([subj_e, bc_e, cl_e, Mz], axis=1).astype(np.float32, copy=False)

    # GLU block 1: (Wv x + bv) * sigmoid(Wg x + bg)
    val1 = x @ state.l1_value_W + state.l1_value_b
    gate1 = x @ state.l1_gate_W + state.l1_gate_b
    h1 = (val1 * _sigmoid_stable(gate1).astype(np.float32)).astype(np.float32, copy=False)

    # GLU block 2.
    val2 = h1 @ state.l2_value_W + state.l2_value_b
    gate2 = h1 @ state.l2_gate_W + state.l2_gate_b
    h2 = (val2 * _sigmoid_stable(gate2).astype(np.float32)).astype(np.float32, copy=False)

    # Linear head -> logit -> prob.
    z = (h2 @ state.head_W).reshape(-1) + float(state.head_b)
    p = _sigmoid_stable(z)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p.astype(np.float32)


def apply_one(
    *,
    state: "Member2MLPState",
    subject_id: int,
    bc_id: int,
    cluster_id: int,
    marginals: np.ndarray,
) -> float:
    """Single-row inference. ``marginals`` is ``[n_marginals]``."""
    M = np.asarray(marginals, dtype=np.float32).reshape(1, -1)
    p = apply_batch(
        state=state,
        subject_ids=np.array([int(subject_id)], dtype=np.int64),
        bc_ids=np.array([int(bc_id)], dtype=np.int64),
        cluster_ids=np.array([int(cluster_id)], dtype=np.int64),
        marginals=M,
    )
    return float(p[0])


def apply_state_batch(state: "Member2MLPState", **kwargs) -> np.ndarray:
    """Uniform with other members' apply_state_batch convention.

    Required kwargs: subject_ids, bc_ids, cluster_ids, marginals.
    """
    return apply_batch(state=state, **kwargs)


def apply_state_one(state: "Member2MLPState", **kwargs) -> float:
    return apply_one(state=state, **kwargs)


# ---------------------------------------------------------------------------
# Trainer (uses torch; called from the notebook / fit-time only)
# ---------------------------------------------------------------------------


def _chunked_mean_std(
    X: np.ndarray, idx: np.ndarray, chunk: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) of X[idx] without a full f64 copy."""
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


def fit_member2_metadata_mlp(
    *,
    # Per-row inputs ([N_rows]):
    subject_ids: np.ndarray,
    bc_ids: np.ndarray,
    cluster_ids: np.ndarray,
    marginals: np.ndarray,           # [N_rows, n_marginals] float32
    y: np.ndarray,                   # [N_rows] float in {0, 1}
    # Provenance:
    subject_keys: Sequence[str],
    bc_keys: Sequence[str],
    marg_feature_names: Sequence[str],
    n_subjects: int,
    n_bcs: int,
    n_clusters: int,
    # Architecture:
    d_subj: int = 32,
    d_bc: int = 32,
    d_cluster: int = 16,
    hid1: int = 256,
    hid2: int = 128,
    # Training:
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    epochs: int = 40,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    early_stopping_patience: int = 5,
    cat_dropout_subject: float = 0.05,
    cat_dropout_bc: float = 0.10,
    cat_dropout_cluster: float = 0.10,
    feat_dropout: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    holdout_group_id: np.ndarray | None = None,
    init_head_b_from_prior: bool = True,
    log_every: int = 5,
    show_progress: bool = True,
) -> Member2MLPState:
    """Fit the metadata-only Member 2 MLP.

    Parameters
    ----------
    subject_ids, bc_ids, cluster_ids, marginals, y
        Aligned per-row arrays of length ``N_rows``. ``marginals`` is
        ``[N_rows, n_marginals]`` and is z-scored using the train-slice
        mean/std (recorded in the state for inference).
    subject_keys, bc_keys
        Length must equal ``n_subjects`` / ``n_bcs`` respectively; used
        for provenance only (no runtime indexing).
    marg_feature_names
        Length must equal ``marginals.shape[1]``.
    d_subj, d_bc, d_cluster, hid1, hid2
        Architecture sizes. Defaults: 32-32-16 embeddings, 256-128 GLU
        hidden dims. Total params ~ 200k at the standard subject/bc/
        cluster cardinalities -- small enough for fast iteration, large
        enough to learn rich (subject, bc, cluster) interactions.
    cat_dropout_subject, cat_dropout_bc, cat_dropout_cluster
        Per-row probability that each categorical input is replaced by
        the UNK index during training. Defaults skew higher on bc
        (0.10) and cluster (0.10) than on subject (0.05) because
        unknown bc is the dominant cold-start case at inference.
    feat_dropout
        Dropout applied to the GLU block outputs.
    val_fraction
        Train/val split for early stopping. When ``holdout_group_id``
        is provided the split holds out whole groups (typically
        ``gbdt_train_item_id`` -- the item-cold split) instead of
        random rows; this matches what Members 2/4 already do and
        avoids inflated val estimates.
    init_head_b_from_prior
        Initialize the bias at ``logit(mean(y))`` on the train slice so
        Adam doesn't burn epochs lifting the bias from zero.
    seed
        Seed for the train/val split, embedding init, and dropout rng.
    show_progress
        Wraps training loops in tqdm progress bars when True.

    Returns
    -------
    Member2MLPState
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW

    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]
    except Exception:
        def tqdm(it, **kwargs):
            return it

    # --- Shape + provenance checks ---
    s_arr = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    b_arr = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    c_arr = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    M_arr = np.asarray(marginals, dtype=np.float32)
    N = int(y_arr.shape[0])
    if s_arr.shape != (N,) or b_arr.shape != (N,) or c_arr.shape != (N,):
        raise ValueError(
            f"subject_ids/bc_ids/cluster_ids must match y length {N}; got "
            f"{s_arr.shape}, {b_arr.shape}, {c_arr.shape}"
        )
    if M_arr.ndim != 2 or int(M_arr.shape[0]) != N:
        raise ValueError(
            f"marginals shape {M_arr.shape} != (N={N}, n_marginals)"
        )
    n_marginals = int(M_arr.shape[1])
    if int(len(marg_feature_names)) != n_marginals:
        raise ValueError(
            f"marg_feature_names len {len(marg_feature_names)} != "
            f"marginals cols {n_marginals}"
        )
    if int(len(subject_keys)) != int(n_subjects):
        raise ValueError(
            f"subject_keys len {len(subject_keys)} != n_subjects {n_subjects}"
        )
    if int(len(bc_keys)) != int(n_bcs):
        raise ValueError(
            f"bc_keys len {len(bc_keys)} != n_bcs {n_bcs}"
        )
    for nm, p in (
        ("cat_dropout_subject", cat_dropout_subject),
        ("cat_dropout_bc", cat_dropout_bc),
        ("cat_dropout_cluster", cat_dropout_cluster),
        ("feat_dropout", feat_dropout),
        ("val_fraction", val_fraction),
    ):
        if not (0.0 <= float(p) < 1.0):
            raise ValueError(f"{nm} must be in [0, 1), got {p}")

    # --- Train / val split ---
    rng = np.random.default_rng(int(seed))
    if holdout_group_id is not None:
        gid = np.asarray(holdout_group_id, dtype=np.int64).reshape(-1)
        if gid.shape != (N,):
            raise ValueError(
                f"holdout_group_id shape {gid.shape} != (N={N},)"
            )
        unique = np.unique(gid[gid >= 0])
        if unique.size > 0:
            n_val_groups = max(1, int(round(unique.size * float(val_fraction))))
            val_groups = set(
                rng.choice(unique, size=n_val_groups, replace=False).tolist()
            )
            is_val = np.isin(gid, list(val_groups))
        else:
            is_val = rng.random(N) < float(val_fraction)
    else:
        is_val = rng.random(N) < float(val_fraction)
    train_idx = np.flatnonzero(~is_val).astype(np.int64)
    val_idx = np.flatnonzero(is_val).astype(np.int64)
    if int(train_idx.size) == 0:
        raise RuntimeError("train slice is empty after split")
    if int(val_idx.size) == 0:
        # Fall back to a random 5% slice so early-stopping has something
        # to monitor -- happens only on tiny inputs (e.g. unit tests).
        rand_val = rng.choice(train_idx, size=max(1, int(0.05 * train_idx.size)),
                              replace=False)
        is_val = np.zeros(N, dtype=bool)
        is_val[rand_val] = True
        train_idx = np.flatnonzero(~is_val).astype(np.int64)
        val_idx = np.flatnonzero(is_val).astype(np.int64)
    n_train = int(train_idx.size)
    n_val = int(val_idx.size)
    n_pos = int(y_arr.sum())

    # --- Standardize marginals on TRAIN slice only (no val leakage) ---
    mu, sigma = _chunked_mean_std(M_arr, train_idx)
    sigma = np.where(sigma > 0, sigma, 1.0).astype(np.float64)
    marg_mean = mu.astype(np.float32)
    marg_std = sigma.astype(np.float32)

    # --- Choose device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Build the torch model ---
    in_dim = int(d_subj) + int(d_bc) + int(d_cluster) + n_marginals
    UNK_S = int(n_subjects)
    UNK_B = int(n_bcs)
    UNK_C = int(n_clusters)

    class _GLUBlock(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.value = nn.Linear(d_in, d_out)
            self.gate = nn.Linear(d_in, d_out)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.value(x) * torch.sigmoid(self.gate(x))

    class _M2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.subject_emb = nn.Embedding(int(n_subjects) + 1, int(d_subj))
            self.bc_emb = nn.Embedding(int(n_bcs) + 1, int(d_bc))
            self.cluster_emb = nn.Embedding(int(n_clusters) + 1, int(d_cluster))
            self.block1 = _GLUBlock(in_dim, int(hid1))
            self.drop1 = nn.Dropout(float(feat_dropout))
            self.block2 = _GLUBlock(int(hid1), int(hid2))
            self.drop2 = nn.Dropout(float(feat_dropout))
            self.head = nn.Linear(int(hid2), 1)
            self._init_weights()

        def _init_weights(self) -> None:
            for emb in (self.subject_emb, self.bc_emb, self.cluster_emb):
                nn.init.normal_(emb.weight, std=0.01)
                # Initialize UNK row to zeros so the prediction collapses
                # to "use only the other inputs" when an id is unknown
                # at inference. This is what the model also learns under
                # categorical dropout, but starting close to it accelerates
                # convergence.
                with torch.no_grad():
                    emb.weight[-1].zero_()
            for lin in (
                self.block1.value, self.block1.gate,
                self.block2.value, self.block2.gate, self.head,
            ):
                nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            # Init gate biases positive so sigmoid(gate) ~ 0.7 at start
            # (not 0.5, which halves the effective scale and slows the
            # first few epochs).
            with torch.no_grad():
                self.block1.gate.bias.fill_(1.0)
                self.block2.gate.bias.fill_(1.0)

        def forward(
            self,
            s: "torch.Tensor", b: "torch.Tensor", c: "torch.Tensor",  # type: ignore[name-defined]
            m: "torch.Tensor",  # type: ignore[name-defined]
        ) -> "torch.Tensor":  # type: ignore[name-defined]
            se = self.subject_emb(s)
            be = self.bc_emb(b)
            ce = self.cluster_emb(c)
            x = torch.cat([se, be, ce, m], dim=1)
            h = self.drop1(self.block1(x))
            h = self.drop2(self.block2(h))
            z = self.head(h).squeeze(-1)
            return z

    torch.manual_seed(int(seed))
    model = _M2MLP().to(dev)
    if bool(init_head_b_from_prior):
        prior = float(np.clip(y_arr[train_idx].mean(), _EPS, 1.0 - _EPS))
        with torch.no_grad():
            model.head.bias.fill_(float(_logit_clipped_scalar(prior)))

    optimizer = AdamW(
        model.parameters(), lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    # --- Standardize marginals on the GPU once per row block ---
    mu_t = torch.from_numpy(marg_mean.astype(np.float32)).to(dev)
    sd_t = torch.from_numpy(marg_std.astype(np.float32)).to(dev)

    # Persistent val tensors -- val is fixed across epochs.
    s_val = torch.from_numpy(s_arr[val_idx].astype(np.int64)).to(dev)
    b_val = torch.from_numpy(b_arr[val_idx].astype(np.int64)).to(dev)
    c_val = torch.from_numpy(c_arr[val_idx].astype(np.int64)).to(dev)
    M_val = torch.from_numpy(M_arr[val_idx].astype(np.float32)).to(dev)
    y_val = torch.from_numpy(y_arr[val_idx].astype(np.float32)).to(dev)
    # Pre-standardize + UNK-clamp val once (val does NOT get categorical
    # dropout). UNK-clamp: any id outside [0, n_categories) goes to UNK.
    M_val_z = ((M_val - mu_t) / sd_t).nan_to_num_(0.0, posinf=0.0, neginf=0.0)
    s_val = torch.where((s_val >= 0) & (s_val < UNK_S), s_val,
                        torch.full_like(s_val, UNK_S))
    b_val = torch.where((b_val >= 0) & (b_val < UNK_B), b_val,
                        torch.full_like(b_val, UNK_B))
    c_val = torch.where((c_val >= 0) & (c_val < UNK_C), c_val,
                        torch.full_like(c_val, UNK_C))

    best_val_nll = float("inf")
    best_state = None
    patience_left = int(early_stopping_patience)
    train_loss_last = float("nan")
    val_loss_last = float("nan")
    epochs_run = 0

    t0 = time.time()
    epoch_iter = range(int(epochs))
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="[Member 2 MLP] epochs", total=int(epochs))
    for epoch in epoch_iter:
        epochs_run = int(epoch) + 1
        model.train()
        perm = rng.permutation(n_train)
        n_batches = (n_train + int(batch_size) - 1) // int(batch_size)
        running_loss = 0.0
        running_n = 0
        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter, desc=f"[ep {epoch + 1}/{int(epochs)}]",
                leave=False, total=n_batches,
            )
        for bi in batch_iter:
            sl = perm[bi * int(batch_size): (bi + 1) * int(batch_size)]
            rows = train_idx[sl]
            s_b = s_arr[rows]
            b_b = b_arr[rows]
            c_b = c_arr[rows]
            M_b = M_arr[rows]
            y_b = y_arr[rows]
            # Categorical dropout + UNK-clamp on the train batch.
            s_keep = (s_b >= 0) & (s_b < UNK_S)
            b_keep = (b_b >= 0) & (b_b < UNK_B)
            c_keep = (c_b >= 0) & (c_b < UNK_C)
            if float(cat_dropout_subject) > 0:
                s_keep &= rng.random(len(s_b)) >= float(cat_dropout_subject)
            if float(cat_dropout_bc) > 0:
                b_keep &= rng.random(len(b_b)) >= float(cat_dropout_bc)
            if float(cat_dropout_cluster) > 0:
                c_keep &= rng.random(len(c_b)) >= float(cat_dropout_cluster)
            s_b = np.where(s_keep, s_b, UNK_S).astype(np.int64)
            b_b = np.where(b_keep, b_b, UNK_B).astype(np.int64)
            c_b = np.where(c_keep, c_b, UNK_C).astype(np.int64)

            s_t = torch.from_numpy(s_b).to(dev, non_blocking=True)
            b_t = torch.from_numpy(b_b).to(dev, non_blocking=True)
            c_t = torch.from_numpy(c_b).to(dev, non_blocking=True)
            M_t = torch.from_numpy(M_b.astype(np.float32)).to(dev, non_blocking=True)
            y_t = torch.from_numpy(y_b.astype(np.float32)).to(dev, non_blocking=True)
            M_t = ((M_t - mu_t) / sd_t).nan_to_num_(0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)
            z = model(s_t, b_t, c_t, M_t)
            loss = bce(z, y_t)
            loss.backward()
            optimizer.step()

            bs_actual = int(y_t.shape[0])
            running_loss += float(loss.item()) * bs_actual
            running_n += bs_actual

        train_loss_last = float(running_loss / max(running_n, 1))

        # Val pass (no dropout, no categorical dropout).
        model.eval()
        with torch.no_grad():
            z_val = model(s_val, b_val, c_val, M_val_z)
            val_loss = float(bce(z_val, y_val).item())
        val_loss_last = val_loss

        if (epoch % max(int(log_every), 1) == 0) or (epoch + 1 == int(epochs)):
            LOG.info(
                "[Member 2 MLP] epoch %d/%d  train_loss=%.5f  val_loss=%.5f  "
                "best=%.5f  patience=%d  elapsed=%.1fs",
                epoch + 1, int(epochs), train_loss_last, val_loss,
                best_val_nll, patience_left, time.time() - t0,
            )

        if val_loss + 1.0e-6 < best_val_nll:
            best_val_nll = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = int(early_stopping_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOG.info(
                    "[Member 2 MLP] early stop at epoch %d "
                    "(best val_loss=%.5f).", epoch + 1, best_val_nll,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # --- Materialize the pure-numpy state ---
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def _w(key: str) -> np.ndarray:
        return sd[key].astype(np.float32, copy=False)

    state = Member2MLPState(
        subject_emb=_w("subject_emb.weight"),
        bc_emb=_w("bc_emb.weight"),
        cluster_emb=_w("cluster_emb.weight"),
        l1_value_W=_w("block1.value.weight").T.copy(),
        l1_value_b=_w("block1.value.bias"),
        l1_gate_W=_w("block1.gate.weight").T.copy(),
        l1_gate_b=_w("block1.gate.bias"),
        l2_value_W=_w("block2.value.weight").T.copy(),
        l2_value_b=_w("block2.value.bias"),
        l2_gate_W=_w("block2.gate.weight").T.copy(),
        l2_gate_b=_w("block2.gate.bias"),
        head_W=_w("head.weight").T.copy(),
        head_b=float(sd["head.bias"][0]),
        marg_mean=marg_mean,
        marg_std=marg_std,
        subject_keys=tuple(str(s) for s in subject_keys),
        bc_keys=tuple(str(s) for s in bc_keys),
        n_subjects=int(n_subjects),
        n_bcs=int(n_bcs),
        n_clusters=int(n_clusters),
        d_subj=int(d_subj),
        d_bc=int(d_bc),
        d_cluster=int(d_cluster),
        hid1=int(hid1),
        hid2=int(hid2),
        marg_feature_names=tuple(str(s) for s in marg_feature_names),
        n_marginals=int(n_marginals),
        fit_method=f"adamw_glu_cat_dropout (dev={device})",
        n_train=int(n_train),
        n_pos=int(n_pos),
        train_loss=float(train_loss_last),
        val_loss=float(best_val_nll if math.isfinite(best_val_nll) else val_loss_last),
        cat_dropout_subject=float(cat_dropout_subject),
        cat_dropout_bc=float(cat_dropout_bc),
        cat_dropout_cluster=float(cat_dropout_cluster),
        weight_decay=float(weight_decay),
        learning_rate=float(learning_rate),
        epochs_run=int(epochs_run),
    )
    LOG.info(
        "[Member 2 MLP] fit done: n_train=%d n_val=%d n_pos=%d  "
        "best_val_loss=%.5f  train_loss=%.5f  epochs_run=%d  device=%s  "
        "elapsed=%.1fs",
        n_train, n_val, n_pos, state.val_loss, state.train_loss,
        epochs_run, device, time.time() - t0,
    )
    return state


__all__ = [
    "Member2MLPState",
    "fit_member2_metadata_mlp",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
    "apply_state_one",
]
