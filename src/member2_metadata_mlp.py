"""Member 2 (v2): dense metadata MLP with DCN-v2 cross network.

This is the *dense* metadata-only member. It deliberately consumes **no
item-embedding columns** -- the inductive bias is "what we know about
this (subject, benchmark, cluster, ...) cell, completely independent
of item content." That makes its errors structurally orthogonal to
M1 (deep on embeddings), M3 (kNN on embeddings), M4 (linear on
embeddings + marginals), and M6 (bilinear on embeddings + marginals).

What changed from v1 (purely cat-id + marginals)
------------------------------------------------
* **All available subject metadata**: subject_id, family_id,
  macro_family_id, organization_id (categorical) plus subject CSV
  numerical fields (log_params + missing-flag, release_date +
  missing-flag).
* **All available benchmark metadata**: bc_id (benchmark::condition)
  and bench_topic_id (categorical), plus benchmark_age + missing-flag
  (numerical) and a benchmark-redacted-this-row flag.
* **DCN-v2 cross network** (Wang et al. 2021, "DCN V2: Improved Deep &
  Cross Network..."): two low-rank cross layers running in parallel
  with the deep GLU tower. Cross layers explicitly model element-wise
  *multiplicative* feature interactions, which the deep tower can only
  approximate. With 7 categorical fields + a dozen numerics the cross
  network captures (subject x cluster x bench) -style triples without
  an explicit triple marginal.
* **Training tricks**: cosine LR with linear warm-up, weight EMA
  (decay=0.999 by default), snapshot ensembling over the top-K best
  val checkpoints, label smoothing, and Mixup on the numerical
  channel only (cats are left untouched). All off-by-default cost
  nothing when disabled.

Forward pass (high-level)
-------------------------
1. Look up every categorical id in its embedding table (with a UNK
   slot at the trailing index for cold-start / dropout).
2. Z-score the numerical channel using train-slice mean/std.
3. Concatenate everything -> ``x_0 in R^d``.
4. Cross tower: 2 low-rank cross layers
        x_{l+1} = x_0 * (U_l @ (V_l^T @ x_l) + b_l) + x_l
   produces ``x_cross in R^d``.
5. Deep tower: two GLU blocks
        h_{l+1} = (W_v h_l + b_v) * sigmoid(W_g h_l + b_g)
   produces ``h_deep in R^{hid2}``.
6. Head: ``logit = head_W^T @ concat(x_cross, h_deep) + head_b``.

Pure-NumPy inference
--------------------
``apply_state_batch(state, ...)`` for offline rescoring;
``apply_state_one(state, ...)`` for the single-row runtime path. The
torch dependency is fit-time only; the runtime bundle never imports
torch.

Cold-start contract
-------------------
Every categorical embedding table has a trailing UNK slot at index
``n_categories``. IDs outside ``[0, n_categories)`` (including the
sentinel -1) are routed there. During training, categorical dropout
randomly replaces each id with UNK so the network learns to fall
through gracefully when only some fields are known.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("member2_metadata_mlp")

_EPS = 1.0e-6

# Names of the categorical fields, in the canonical input/concat order.
# This ordering is part of the runtime contract: ``apply_batch`` expects
# the embeddings to be concatenated in this exact order, and the
# notebook / runtime feature builder must pass the corresponding IDs.
_CAT_FIELD_NAMES: tuple[str, ...] = (
    "subject", "bc", "cluster",
    "family", "macro_family", "organization", "bench_topic",
)


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
    """Fitted state of the dense metadata Member 2 MLP (v2).

    Categorical embedding tables each have shape ``[n_categories + 1,
    d]`` with the trailing row dedicated to UNK (cold-start / dropout).
    """

    # ---- Categorical embedding tables (each [n_cat + 1, d]) ----
    subject_emb: np.ndarray
    bc_emb: np.ndarray
    cluster_emb: np.ndarray
    family_emb: np.ndarray
    macro_emb: np.ndarray
    org_emb: np.ndarray
    topic_emb: np.ndarray

    # ---- DCN-v2 cross layers ----
    # Each cross layer is parameterised by (V, U, b) with:
    #   V : [d_in, cross_rank]
    #   U : [cross_rank, d_in]
    #   b : [d_in]
    # The forward step is
    #   x_{l+1} = x_0 * (x_l @ V @ U + b) + x_l
    # which is the low-rank DCN-v2 cross layer of Wang et al. (2021).
    cross_V: list[np.ndarray] = field(default_factory=list)
    cross_U: list[np.ndarray] = field(default_factory=list)
    cross_b: list[np.ndarray] = field(default_factory=list)

    # ---- Deep tower (two GLU blocks) ----
    # GLU block 1: d_in -> hid1.
    l1_value_W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    l1_value_b: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    l1_gate_W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    l1_gate_b: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    # GLU block 2: hid1 -> hid2.
    l2_value_W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    l2_value_b: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    l2_gate_W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    l2_gate_b: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))

    # ---- Head: concat(cross, deep) -> 1 ----
    # head_W shape is [d_in + hid2, 1] so the head sees both the
    # cross tower's residual representation AND the deep tower's
    # compressed representation -- the parallel-tower DCN-v2 form.
    head_W: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    head_b: float = 0.0

    # ---- Numerical standardization (train-slice mean / std) ----
    # The numerical channel is the concatenation (in order):
    #   subject_num | bench_num | bc_redacted_flag (1 col) | marginals
    # n_num = sum of those widths; num_feature_names records each.
    num_mean: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    num_std: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float32))

    # ---- Provenance / runtime indexing metadata ----
    subject_keys: tuple[str, ...] = ()
    bc_keys: tuple[str, ...] = ()
    n_subjects: int = 0
    n_bcs: int = 0
    n_clusters: int = 0
    n_families: int = 0
    n_macro_families: int = 0
    n_organizations: int = 0
    n_bench_topics: int = 0
    d_subj: int = 0
    d_bc: int = 0
    d_cluster: int = 0
    d_family: int = 0
    d_macro: int = 0
    d_org: int = 0
    d_topic: int = 0
    hid1: int = 0
    hid2: int = 0
    n_cross_layers: int = 0
    cross_rank: int = 0
    num_feature_names: tuple[str, ...] = ()
    n_num: int = 0
    # Sub-counts (so apply / runtime can validate the per-block widths).
    n_subj_num: int = 0
    n_bench_num: int = 0
    n_marginals: int = 0  # kept for legacy assertions; equals last block width.

    # ---- Fit diagnostics ----
    fit_method: str = "unknown"
    n_train: int = 0
    n_pos: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    cat_dropout_subject: float = 0.0
    cat_dropout_bc: float = 0.0
    cat_dropout_cluster: float = 0.0
    cat_dropout_family: float = 0.0
    cat_dropout_macro: float = 0.0
    cat_dropout_org: float = 0.0
    cat_dropout_topic: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 0.0
    epochs_run: int = 0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    ema_decay: float = 0.0
    snapshot_ensemble_k: int = 1
    warmup_epochs: int = 0

    def __post_init__(self) -> None:
        self._validate_shapes()
        self._validate_numerics()

    # ---- Internal validation ----
    @property
    def d_in(self) -> int:
        return (
            int(self.d_subj) + int(self.d_bc) + int(self.d_cluster)
            + int(self.d_family) + int(self.d_macro) + int(self.d_org)
            + int(self.d_topic) + int(self.n_num)
        )

    def _validate_shapes(self) -> None:
        emb_specs = [
            ("subject_emb", self.subject_emb, self.n_subjects, self.d_subj),
            ("bc_emb", self.bc_emb, self.n_bcs, self.d_bc),
            ("cluster_emb", self.cluster_emb, self.n_clusters, self.d_cluster),
            ("family_emb", self.family_emb, self.n_families, self.d_family),
            ("macro_emb", self.macro_emb, self.n_macro_families, self.d_macro),
            ("org_emb", self.org_emb, self.n_organizations, self.d_org),
            ("topic_emb", self.topic_emb, self.n_bench_topics, self.d_topic),
        ]
        for name, arr, n_cat, d in emb_specs:
            expected = (int(n_cat) + 1, int(d))
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(arr.shape)} != {expected} "
                    f"(n_cat={n_cat}, d={d})"
                )
        d_in = self.d_in
        # Cross tower.
        if len(self.cross_V) != int(self.n_cross_layers):
            raise ValueError(
                f"cross_V len {len(self.cross_V)} != n_cross_layers "
                f"{self.n_cross_layers}"
            )
        if len(self.cross_U) != int(self.n_cross_layers):
            raise ValueError(
                f"cross_U len {len(self.cross_U)} != n_cross_layers "
                f"{self.n_cross_layers}"
            )
        if len(self.cross_b) != int(self.n_cross_layers):
            raise ValueError(
                f"cross_b len {len(self.cross_b)} != n_cross_layers "
                f"{self.n_cross_layers}"
            )
        for li in range(int(self.n_cross_layers)):
            if tuple(self.cross_V[li].shape) != (d_in, int(self.cross_rank)):
                raise ValueError(
                    f"cross_V[{li}] shape {tuple(self.cross_V[li].shape)} != "
                    f"({d_in}, {self.cross_rank})"
                )
            if tuple(self.cross_U[li].shape) != (int(self.cross_rank), d_in):
                raise ValueError(
                    f"cross_U[{li}] shape {tuple(self.cross_U[li].shape)} != "
                    f"({self.cross_rank}, {d_in})"
                )
            if tuple(self.cross_b[li].shape) != (d_in,):
                raise ValueError(
                    f"cross_b[{li}] shape {tuple(self.cross_b[li].shape)} != "
                    f"({d_in},)"
                )
        # Deep tower.
        deep_specs = [
            ("l1_value_W", self.l1_value_W, (d_in, int(self.hid1))),
            ("l1_gate_W", self.l1_gate_W, (d_in, int(self.hid1))),
            ("l1_value_b", self.l1_value_b, (int(self.hid1),)),
            ("l1_gate_b", self.l1_gate_b, (int(self.hid1),)),
            ("l2_value_W", self.l2_value_W, (int(self.hid1), int(self.hid2))),
            ("l2_gate_W", self.l2_gate_W, (int(self.hid1), int(self.hid2))),
            ("l2_value_b", self.l2_value_b, (int(self.hid2),)),
            ("l2_gate_b", self.l2_gate_b, (int(self.hid2),)),
        ]
        for name, arr, expected in deep_specs:
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"{name} shape {tuple(arr.shape)} != {expected}"
                )
        # Head.
        expected_head = (d_in + int(self.hid2), 1)
        if tuple(self.head_W.shape) != expected_head:
            raise ValueError(
                f"head_W shape {tuple(self.head_W.shape)} != {expected_head}"
            )
        # Numerical standardization.
        if tuple(self.num_mean.shape) != (int(self.n_num),):
            raise ValueError(
                f"num_mean shape {tuple(self.num_mean.shape)} != "
                f"({self.n_num},)"
            )
        if tuple(self.num_std.shape) != tuple(self.num_mean.shape):
            raise ValueError("num_std shape != num_mean shape")
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
        if int(len(self.num_feature_names)) != int(self.n_num):
            raise ValueError(
                f"num_feature_names len {len(self.num_feature_names)} != "
                f"n_num {self.n_num}"
            )
        block_sum = (
            int(self.n_subj_num) + int(self.n_bench_num) + 1
            + int(self.n_marginals)
        )
        if block_sum != int(self.n_num):
            raise ValueError(
                f"sum of numerical blocks ({self.n_subj_num} subj + "
                f"{self.n_bench_num} bench + 1 redact + {self.n_marginals} "
                f"marginals = {block_sum}) != n_num ({self.n_num})"
            )

    def _validate_numerics(self) -> None:
        if not math.isfinite(float(self.head_b)):
            raise ValueError("head_b is NaN/Inf")
        arrays: list[tuple[str, np.ndarray]] = [
            ("subject_emb", self.subject_emb),
            ("bc_emb", self.bc_emb),
            ("cluster_emb", self.cluster_emb),
            ("family_emb", self.family_emb),
            ("macro_emb", self.macro_emb),
            ("org_emb", self.org_emb),
            ("topic_emb", self.topic_emb),
            ("l1_value_W", self.l1_value_W),
            ("l1_value_b", self.l1_value_b),
            ("l1_gate_W", self.l1_gate_W),
            ("l1_gate_b", self.l1_gate_b),
            ("l2_value_W", self.l2_value_W),
            ("l2_value_b", self.l2_value_b),
            ("l2_gate_W", self.l2_gate_W),
            ("l2_gate_b", self.l2_gate_b),
            ("head_W", self.head_W),
            ("num_mean", self.num_mean),
            ("num_std", self.num_std),
        ]
        for li, (v, u, b) in enumerate(zip(self.cross_V, self.cross_U, self.cross_b)):
            arrays.append((f"cross_V[{li}]", v))
            arrays.append((f"cross_U[{li}]", u))
            arrays.append((f"cross_b[{li}]", b))
        for name, arr in arrays:
            if arr.size and not np.all(np.isfinite(arr)):
                raise ValueError(f"Member2MLPState: {name} contains NaN/Inf")
        if not np.all(self.num_std > 0):
            raise ValueError(
                "Member2MLPState: num_std must be strictly positive (replace "
                "zero-std columns with 1.0 at fit time)."
            )

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights: dict[str, np.ndarray] = {
            "subject_emb": self.subject_emb.astype(np.float32),
            "bc_emb": self.bc_emb.astype(np.float32),
            "cluster_emb": self.cluster_emb.astype(np.float32),
            "family_emb": self.family_emb.astype(np.float32),
            "macro_emb": self.macro_emb.astype(np.float32),
            "org_emb": self.org_emb.astype(np.float32),
            "topic_emb": self.topic_emb.astype(np.float32),
            "l1_value_W": self.l1_value_W.astype(np.float32),
            "l1_value_b": self.l1_value_b.astype(np.float32),
            "l1_gate_W": self.l1_gate_W.astype(np.float32),
            "l1_gate_b": self.l1_gate_b.astype(np.float32),
            "l2_value_W": self.l2_value_W.astype(np.float32),
            "l2_value_b": self.l2_value_b.astype(np.float32),
            "l2_gate_W": self.l2_gate_W.astype(np.float32),
            "l2_gate_b": self.l2_gate_b.astype(np.float32),
            "head_W": self.head_W.astype(np.float32),
            "num_mean": self.num_mean.astype(np.float32),
            "num_std": self.num_std.astype(np.float32),
        }
        for li in range(int(self.n_cross_layers)):
            weights[f"cross_V_{li}"] = self.cross_V[li].astype(np.float32)
            weights[f"cross_U_{li}"] = self.cross_U[li].astype(np.float32)
            weights[f"cross_b_{li}"] = self.cross_b[li].astype(np.float32)
        np.savez_compressed(out / "weights.npz", **weights)
        meta = {
            "head_b": float(self.head_b),
            "subject_keys": list(self.subject_keys),
            "bc_keys": list(self.bc_keys),
            "n_subjects": int(self.n_subjects),
            "n_bcs": int(self.n_bcs),
            "n_clusters": int(self.n_clusters),
            "n_families": int(self.n_families),
            "n_macro_families": int(self.n_macro_families),
            "n_organizations": int(self.n_organizations),
            "n_bench_topics": int(self.n_bench_topics),
            "d_subj": int(self.d_subj),
            "d_bc": int(self.d_bc),
            "d_cluster": int(self.d_cluster),
            "d_family": int(self.d_family),
            "d_macro": int(self.d_macro),
            "d_org": int(self.d_org),
            "d_topic": int(self.d_topic),
            "hid1": int(self.hid1),
            "hid2": int(self.hid2),
            "n_cross_layers": int(self.n_cross_layers),
            "cross_rank": int(self.cross_rank),
            "num_feature_names": list(self.num_feature_names),
            "n_num": int(self.n_num),
            "n_subj_num": int(self.n_subj_num),
            "n_bench_num": int(self.n_bench_num),
            "n_marginals": int(self.n_marginals),
            "fit_method": str(self.fit_method),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "cat_dropout_subject": float(self.cat_dropout_subject),
            "cat_dropout_bc": float(self.cat_dropout_bc),
            "cat_dropout_cluster": float(self.cat_dropout_cluster),
            "cat_dropout_family": float(self.cat_dropout_family),
            "cat_dropout_macro": float(self.cat_dropout_macro),
            "cat_dropout_org": float(self.cat_dropout_org),
            "cat_dropout_topic": float(self.cat_dropout_topic),
            "weight_decay": float(self.weight_decay),
            "learning_rate": float(self.learning_rate),
            "epochs_run": int(self.epochs_run),
            "label_smoothing": float(self.label_smoothing),
            "mixup_alpha": float(self.mixup_alpha),
            "ema_decay": float(self.ema_decay),
            "snapshot_ensemble_k": int(self.snapshot_ensemble_k),
            "warmup_epochs": int(self.warmup_epochs),
            "format_version": 2,
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
            n_cross_layers = int(meta.get("n_cross_layers", 0))
            cross_V = [
                npz[f"cross_V_{li}"].astype(np.float32, copy=False)
                for li in range(n_cross_layers)
            ]
            cross_U = [
                npz[f"cross_U_{li}"].astype(np.float32, copy=False)
                for li in range(n_cross_layers)
            ]
            cross_b = [
                npz[f"cross_b_{li}"].astype(np.float32, copy=False)
                for li in range(n_cross_layers)
            ]
            return cls(
                subject_emb=npz["subject_emb"].astype(np.float32, copy=False),
                bc_emb=npz["bc_emb"].astype(np.float32, copy=False),
                cluster_emb=npz["cluster_emb"].astype(np.float32, copy=False),
                family_emb=npz["family_emb"].astype(np.float32, copy=False),
                macro_emb=npz["macro_emb"].astype(np.float32, copy=False),
                org_emb=npz["org_emb"].astype(np.float32, copy=False),
                topic_emb=npz["topic_emb"].astype(np.float32, copy=False),
                cross_V=cross_V,
                cross_U=cross_U,
                cross_b=cross_b,
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
                num_mean=npz["num_mean"].astype(np.float32, copy=False),
                num_std=npz["num_std"].astype(np.float32, copy=False),
                subject_keys=tuple(str(s) for s in meta["subject_keys"]),
                bc_keys=tuple(str(s) for s in meta["bc_keys"]),
                n_subjects=int(meta["n_subjects"]),
                n_bcs=int(meta["n_bcs"]),
                n_clusters=int(meta["n_clusters"]),
                n_families=int(meta["n_families"]),
                n_macro_families=int(meta["n_macro_families"]),
                n_organizations=int(meta["n_organizations"]),
                n_bench_topics=int(meta["n_bench_topics"]),
                d_subj=int(meta["d_subj"]),
                d_bc=int(meta["d_bc"]),
                d_cluster=int(meta["d_cluster"]),
                d_family=int(meta["d_family"]),
                d_macro=int(meta["d_macro"]),
                d_org=int(meta["d_org"]),
                d_topic=int(meta["d_topic"]),
                hid1=int(meta["hid1"]),
                hid2=int(meta["hid2"]),
                n_cross_layers=int(meta["n_cross_layers"]),
                cross_rank=int(meta["cross_rank"]),
                num_feature_names=tuple(
                    str(s) for s in meta["num_feature_names"]
                ),
                n_num=int(meta["n_num"]),
                n_subj_num=int(meta["n_subj_num"]),
                n_bench_num=int(meta["n_bench_num"]),
                n_marginals=int(meta["n_marginals"]),
                fit_method=str(meta.get("fit_method", "unknown")),
                n_train=int(meta.get("n_train", 0)),
                n_pos=int(meta.get("n_pos", 0)),
                train_loss=float(meta.get("train_loss", 0.0)),
                val_loss=float(meta.get("val_loss", 0.0)),
                cat_dropout_subject=float(meta.get("cat_dropout_subject", 0.0)),
                cat_dropout_bc=float(meta.get("cat_dropout_bc", 0.0)),
                cat_dropout_cluster=float(meta.get("cat_dropout_cluster", 0.0)),
                cat_dropout_family=float(meta.get("cat_dropout_family", 0.0)),
                cat_dropout_macro=float(meta.get("cat_dropout_macro", 0.0)),
                cat_dropout_org=float(meta.get("cat_dropout_org", 0.0)),
                cat_dropout_topic=float(meta.get("cat_dropout_topic", 0.0)),
                weight_decay=float(meta.get("weight_decay", 0.0)),
                learning_rate=float(meta.get("learning_rate", 0.0)),
                epochs_run=int(meta.get("epochs_run", 0)),
                label_smoothing=float(meta.get("label_smoothing", 0.0)),
                mixup_alpha=float(meta.get("mixup_alpha", 0.0)),
                ema_decay=float(meta.get("ema_decay", 0.0)),
                snapshot_ensemble_k=int(meta.get("snapshot_ensemble_k", 1)),
                warmup_epochs=int(meta.get("warmup_epochs", 0)),
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
    ``n_known``. Returns a ``[N, d]`` float32 array. Embedding tables
    with d=0 (i.e. that field disabled) return an ``[N, 0]`` slice.
    """
    if int(table.shape[1]) == 0:
        return np.zeros((int(np.asarray(ids).reshape(-1).shape[0]), 0), dtype=np.float32)
    ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    valid = (ids_arr >= 0) & (ids_arr < int(n_known))
    safe_ids = np.where(valid, ids_arr, int(n_known)).astype(np.int64)
    return table[safe_ids].astype(np.float32, copy=False)


def _build_input_rows(
    state: "Member2MLPState",
    subject_ids: np.ndarray,
    bc_ids: np.ndarray,
    cluster_ids: np.ndarray,
    family_ids: np.ndarray,
    macro_family_ids: np.ndarray,
    organization_ids: np.ndarray,
    bench_topic_ids: np.ndarray,
    numerical: np.ndarray,
) -> np.ndarray:
    """Assemble the ``[N, d_in]`` model input row from the raw fields.

    This is the single source of truth for the categorical-concat
    order; ``apply_batch`` and ``fit_*`` share it.
    """
    s_emb = _safe_cat_lookup(subject_ids, state.subject_emb, state.n_subjects)
    b_emb = _safe_cat_lookup(bc_ids, state.bc_emb, state.n_bcs)
    c_emb = _safe_cat_lookup(cluster_ids, state.cluster_emb, state.n_clusters)
    f_emb = _safe_cat_lookup(family_ids, state.family_emb, state.n_families)
    mf_emb = _safe_cat_lookup(macro_family_ids, state.macro_emb, state.n_macro_families)
    o_emb = _safe_cat_lookup(organization_ids, state.org_emb, state.n_organizations)
    t_emb = _safe_cat_lookup(bench_topic_ids, state.topic_emb, state.n_bench_topics)
    N = int(s_emb.shape[0])
    M = np.asarray(numerical, dtype=np.float32)
    if M.ndim != 2 or int(M.shape[0]) != N:
        raise ValueError(
            f"numerical shape {M.shape} must be (N={N}, n_num={state.n_num})"
        )
    if int(M.shape[1]) != int(state.n_num):
        raise ValueError(
            f"numerical has {M.shape[1]} cols but state was fit on n_num="
            f"{state.n_num}"
        )
    Mz = (M - state.num_mean.astype(np.float32)) / state.num_std.astype(np.float32)
    Mz = np.where(np.isfinite(Mz), Mz, 0.0).astype(np.float32, copy=False)
    return np.concatenate(
        [s_emb, b_emb, c_emb, f_emb, mf_emb, o_emb, t_emb, Mz], axis=1
    ).astype(np.float32, copy=False)


def _cross_forward(
    state: "Member2MLPState", x0: np.ndarray,
) -> np.ndarray:
    """Run the DCN-v2 low-rank cross tower over ``x0`` ``[N, d_in]``.

    Returns ``[N, d_in]``.
    """
    x = x0
    for V, U, b in zip(state.cross_V, state.cross_U, state.cross_b):
        proj = x @ V                  # [N, rank]
        out = proj @ U + b            # [N, d_in]
        x = x0 * out + x              # residual
    return x.astype(np.float32, copy=False)


def _deep_forward(
    state: "Member2MLPState", x0: np.ndarray,
) -> np.ndarray:
    """Run the deep GLU tower over ``x0`` ``[N, d_in]``. Returns ``[N, hid2]``."""
    val1 = x0 @ state.l1_value_W + state.l1_value_b
    gate1 = x0 @ state.l1_gate_W + state.l1_gate_b
    h1 = (val1 * _sigmoid_stable(gate1).astype(np.float32)).astype(np.float32, copy=False)
    val2 = h1 @ state.l2_value_W + state.l2_value_b
    gate2 = h1 @ state.l2_gate_W + state.l2_gate_b
    h2 = (val2 * _sigmoid_stable(gate2).astype(np.float32)).astype(np.float32, copy=False)
    return h2


def apply_batch(
    *,
    state: "Member2MLPState",
    subject_ids: np.ndarray,
    bc_ids: np.ndarray,
    cluster_ids: np.ndarray,
    family_ids: np.ndarray,
    macro_family_ids: np.ndarray,
    organization_ids: np.ndarray,
    bench_topic_ids: np.ndarray,
    numerical: np.ndarray,
) -> np.ndarray:
    """Pure-numpy forward pass for a batch. Returns ``[N]`` float32 probs.

    Parameters
    ----------
    subject_ids, bc_ids, cluster_ids, family_ids, macro_family_ids,
    organization_ids, bench_topic_ids
        ``[N]`` int arrays. Values outside the valid range for each
        embedding table are routed to that table's UNK slot at index
        ``n_categories``.
    numerical
        ``[N, n_num]`` float array. The columns are interpreted in the
        ``state.num_feature_names`` order; values are z-scored using
        the stored ``num_mean`` / ``num_std`` from fit time.
    """
    s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    b = np.asarray(bc_ids, dtype=np.int64).reshape(-1)
    c = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    f = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    mf = np.asarray(macro_family_ids, dtype=np.int64).reshape(-1)
    o = np.asarray(organization_ids, dtype=np.int64).reshape(-1)
    t = np.asarray(bench_topic_ids, dtype=np.int64).reshape(-1)
    shapes = {s.shape, b.shape, c.shape, f.shape, mf.shape, o.shape, t.shape}
    if len(shapes) != 1:
        raise ValueError(
            "all categorical id arrays must have the same length; got "
            f"subject={s.shape}, bc={b.shape}, cluster={c.shape}, "
            f"family={f.shape}, macro={mf.shape}, organization={o.shape}, "
            f"bench_topic={t.shape}"
        )

    x0 = _build_input_rows(
        state, s, b, c, f, mf, o, t, np.asarray(numerical, dtype=np.float32),
    )
    x_cross = _cross_forward(state, x0)
    h_deep = _deep_forward(state, x0)
    z = (np.concatenate([x_cross, h_deep], axis=1) @ state.head_W).reshape(-1)
    z = z + float(state.head_b)
    p = _sigmoid_stable(z)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p.astype(np.float32)


def apply_one(
    *,
    state: "Member2MLPState",
    subject_id: int,
    bc_id: int,
    cluster_id: int,
    family_id: int,
    macro_family_id: int,
    organization_id: int,
    bench_topic_id: int,
    numerical: np.ndarray,
) -> float:
    """Single-row inference. ``numerical`` is ``[n_num]``."""
    M = np.asarray(numerical, dtype=np.float32).reshape(1, -1)
    p = apply_batch(
        state=state,
        subject_ids=np.array([int(subject_id)], dtype=np.int64),
        bc_ids=np.array([int(bc_id)], dtype=np.int64),
        cluster_ids=np.array([int(cluster_id)], dtype=np.int64),
        family_ids=np.array([int(family_id)], dtype=np.int64),
        macro_family_ids=np.array([int(macro_family_id)], dtype=np.int64),
        organization_ids=np.array([int(organization_id)], dtype=np.int64),
        bench_topic_ids=np.array([int(bench_topic_id)], dtype=np.int64),
        numerical=M,
    )
    return float(p[0])


def apply_state_batch(state: "Member2MLPState", **kwargs) -> np.ndarray:
    """Uniform with other members' apply_state_batch convention.

    Required kwargs: subject_ids, bc_ids, cluster_ids, family_ids,
    macro_family_ids, organization_ids, bench_topic_ids, numerical.
    """
    return apply_batch(state=state, **kwargs)


def apply_state_one(state: "Member2MLPState", **kwargs) -> float:
    return apply_one(state=state, **kwargs)


# ---------------------------------------------------------------------------
# Helpers for assembling the numerical channel
# ---------------------------------------------------------------------------


def assemble_numerical(
    *,
    subject_numerical: np.ndarray,    # [N, n_subj_num]
    bench_numerical: np.ndarray,      # [N, n_bench_num]
    bc_redacted_flag: np.ndarray,     # [N]
    marginals: np.ndarray,            # [N, n_marginals]
) -> np.ndarray:
    """Concatenate the four numerical sub-blocks in the canonical order.

    Order: subject_numerical | bench_numerical | bc_redacted_flag |
    marginals.  This order is part of the state's runtime contract.
    """
    s = np.asarray(subject_numerical, dtype=np.float32)
    b = np.asarray(bench_numerical, dtype=np.float32)
    r = np.asarray(bc_redacted_flag, dtype=np.float32).reshape(-1, 1)
    m = np.asarray(marginals, dtype=np.float32)
    if s.ndim != 2 or b.ndim != 2 or m.ndim != 2:
        raise ValueError(
            "subject_numerical, bench_numerical, and marginals must be 2-D"
        )
    N = int(s.shape[0])
    if int(b.shape[0]) != N or int(r.shape[0]) != N or int(m.shape[0]) != N:
        raise ValueError(
            f"assemble_numerical: row counts disagree "
            f"(subj={s.shape[0]}, bench={b.shape[0]}, "
            f"redact={r.shape[0]}, marginals={m.shape[0]})"
        )
    return np.concatenate([s, b, r, m], axis=1).astype(np.float32, copy=False)


def numerical_feature_names(
    *,
    subj_num_names: Sequence[str],
    bench_num_names: Sequence[str],
    marginal_names: Sequence[str],
) -> tuple[str, ...]:
    """Canonical ``num_feature_names`` matching :func:`assemble_numerical`."""
    out: list[str] = []
    out.extend(f"subj_num__{n}" for n in subj_num_names)
    out.extend(f"bench_num__{n}" for n in bench_num_names)
    out.append("bc_redacted_flag")
    out.extend(f"marg__{n}" for n in marginal_names)
    return tuple(out)


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
    # ---- Per-row categorical inputs ([N_rows]) ----
    subject_ids: np.ndarray,
    bc_ids: np.ndarray,
    cluster_ids: np.ndarray,
    family_ids: np.ndarray,
    macro_family_ids: np.ndarray,
    organization_ids: np.ndarray,
    bench_topic_ids: np.ndarray,
    # ---- Per-row numerical input ----
    numerical: np.ndarray,                  # [N_rows, n_num] float32
    # ---- Per-row label ----
    y: np.ndarray,                          # [N_rows] float in {0, 1}
    # ---- Provenance / cardinalities ----
    subject_keys: Sequence[str],
    bc_keys: Sequence[str],
    num_feature_names: Sequence[str],
    n_subjects: int,
    n_bcs: int,
    n_clusters: int,
    n_families: int,
    n_macro_families: int,
    n_organizations: int,
    n_bench_topics: int,
    n_subj_num: int,
    n_bench_num: int,
    n_marginals: int,
    # ---- Architecture ----
    d_subj: int = 32,
    d_bc: int = 32,
    d_cluster: int = 16,
    d_family: int = 16,
    d_macro: int = 8,
    d_org: int = 16,
    d_topic: int = 16,
    hid1: int = 256,
    hid2: int = 128,
    n_cross_layers: int = 2,
    cross_rank: int = 64,
    # ---- Training (core) ----
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    epochs: int = 40,
    batch_size: int = 16_384,
    val_fraction: float = 0.1,
    early_stopping_patience: int = 5,
    # ---- Regularisation ----
    cat_dropout_subject: float = 0.05,
    cat_dropout_bc: float = 0.10,
    cat_dropout_cluster: float = 0.10,
    cat_dropout_family: float = 0.05,
    cat_dropout_macro: float = 0.05,
    cat_dropout_org: float = 0.05,
    cat_dropout_topic: float = 0.10,
    feat_dropout: float = 0.10,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    # ---- LR schedule ----
    warmup_epochs: int = 2,
    use_cosine_schedule: bool = True,
    # ---- EMA / snapshot ensembling ----
    ema_decay: float = 0.999,
    snapshot_ensemble_k: int = 3,
    # ---- Misc ----
    seed: int = 0,
    device: str | None = None,
    holdout_group_id: np.ndarray | None = None,
    init_head_b_from_prior: bool = True,
    log_every: int = 5,
    show_progress: bool = True,
) -> Member2MLPState:
    """Fit the dense metadata Member 2 MLP (v2).

    The training input is split row-wise into a train slice and a val
    slice (item-cold grouping when ``holdout_group_id`` is provided).
    The val slice is used for early stopping, EMA / snapshot model
    selection, and recording the final ``val_loss`` on the state.

    Returns
    -------
    Member2MLPState
        The fitted state. When ``snapshot_ensemble_k > 1`` the returned
        weights are the *average* of the top-K best-val checkpoints
        (uniform weighting; biggest contribution at the single best
        epoch). When EMA is enabled (``ema_decay > 0``), the EMA copy
        replaces the live weights for both selection and the saved
        state.
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
    f_arr = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    mf_arr = np.asarray(macro_family_ids, dtype=np.int64).reshape(-1)
    o_arr = np.asarray(organization_ids, dtype=np.int64).reshape(-1)
    t_arr = np.asarray(bench_topic_ids, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    M_arr = np.asarray(numerical, dtype=np.float32)
    N = int(y_arr.shape[0])
    for name, arr in (
        ("subject_ids", s_arr), ("bc_ids", b_arr), ("cluster_ids", c_arr),
        ("family_ids", f_arr), ("macro_family_ids", mf_arr),
        ("organization_ids", o_arr), ("bench_topic_ids", t_arr),
    ):
        if arr.shape != (N,):
            raise ValueError(
                f"{name} shape {arr.shape} != (N={N},)"
            )
    if M_arr.ndim != 2 or int(M_arr.shape[0]) != N:
        raise ValueError(
            f"numerical shape {M_arr.shape} != (N={N}, n_num)"
        )
    n_num_actual = int(M_arr.shape[1])
    if int(len(num_feature_names)) != n_num_actual:
        raise ValueError(
            f"num_feature_names len {len(num_feature_names)} != "
            f"numerical cols {n_num_actual}"
        )
    block_sum = int(n_subj_num) + int(n_bench_num) + 1 + int(n_marginals)
    if block_sum != n_num_actual:
        raise ValueError(
            f"sum of numerical blocks ({n_subj_num} subj + {n_bench_num} "
            f"bench + 1 redact + {n_marginals} marginals = {block_sum}) "
            f"!= numerical cols {n_num_actual}"
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
        ("cat_dropout_family", cat_dropout_family),
        ("cat_dropout_macro", cat_dropout_macro),
        ("cat_dropout_org", cat_dropout_org),
        ("cat_dropout_topic", cat_dropout_topic),
        ("feat_dropout", feat_dropout),
        ("val_fraction", val_fraction),
        ("label_smoothing", label_smoothing),
    ):
        if not (0.0 <= float(p) < 1.0):
            raise ValueError(f"{nm} must be in [0, 1), got {p}")
    if float(mixup_alpha) < 0.0:
        raise ValueError(f"mixup_alpha must be >= 0, got {mixup_alpha}")
    if int(snapshot_ensemble_k) < 1:
        raise ValueError(f"snapshot_ensemble_k must be >= 1, got {snapshot_ensemble_k}")
    if not (0.0 <= float(ema_decay) < 1.0):
        raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
    if int(warmup_epochs) < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

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
        rand_val = rng.choice(
            train_idx, size=max(1, int(0.05 * train_idx.size)), replace=False,
        )
        is_val = np.zeros(N, dtype=bool)
        is_val[rand_val] = True
        train_idx = np.flatnonzero(~is_val).astype(np.int64)
        val_idx = np.flatnonzero(is_val).astype(np.int64)
    n_train = int(train_idx.size)
    n_val = int(val_idx.size)
    n_pos = int(y_arr.sum())

    # --- Standardize numerical channel on the TRAIN slice only ---
    mu, sigma = _chunked_mean_std(M_arr, train_idx)
    sigma = np.where(sigma > 0, sigma, 1.0).astype(np.float64)
    num_mean = mu.astype(np.float32)
    num_std = sigma.astype(np.float32)

    # --- Choose device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Build the torch model ---
    d_in = (
        int(d_subj) + int(d_bc) + int(d_cluster)
        + int(d_family) + int(d_macro) + int(d_org) + int(d_topic)
        + int(n_num_actual)
    )
    UNK_S = int(n_subjects)
    UNK_B = int(n_bcs)
    UNK_C = int(n_clusters)
    UNK_F = int(n_families)
    UNK_MF = int(n_macro_families)
    UNK_O = int(n_organizations)
    UNK_T = int(n_bench_topics)

    class _GLUBlock(nn.Module):
        def __init__(self, d_in_: int, d_out: int):
            super().__init__()
            self.value = nn.Linear(d_in_, d_out)
            self.gate = nn.Linear(d_in_, d_out)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.value(x) * torch.sigmoid(self.gate(x))

    class _CrossV2Block(nn.Module):
        """Low-rank DCN-v2 cross layer."""

        def __init__(self, d_in_: int, rank: int):
            super().__init__()
            self.V = nn.Linear(d_in_, rank, bias=False)
            self.U = nn.Linear(rank, d_in_, bias=True)

        def forward(self, x0: "torch.Tensor", x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return x0 * self.U(self.V(x)) + x

    class _M2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Metadata-only mode sets d_subj / d_bc / d_cluster to 0.
            # nn.Embedding(n, 0) still registers a [n, 0] weight tensor;
            # AdamW's CUDA foreach path then hits illegal memory access on
            # those zero-numel parameters.  Skip the module entirely and
            # emit an explicit [N, 0] slice in forward (matches the numpy
            # path in _safe_cat_lookup when table.shape[1] == 0).
            self._d_subj = int(d_subj)
            self._d_bc = int(d_bc)
            self._d_cluster = int(d_cluster)
            self.subject_emb = (
                nn.Embedding(int(n_subjects) + 1, int(d_subj))
                if int(d_subj) > 0 else None
            )
            self.bc_emb = (
                nn.Embedding(int(n_bcs) + 1, int(d_bc))
                if int(d_bc) > 0 else None
            )
            self.cluster_emb = (
                nn.Embedding(int(n_clusters) + 1, int(d_cluster))
                if int(d_cluster) > 0 else None
            )
            self.family_emb = nn.Embedding(int(n_families) + 1, int(d_family))
            self.macro_emb = nn.Embedding(int(n_macro_families) + 1, int(d_macro))
            self.org_emb = nn.Embedding(int(n_organizations) + 1, int(d_org))
            self.topic_emb = nn.Embedding(int(n_bench_topics) + 1, int(d_topic))
            self.cross_blocks = nn.ModuleList(
                [_CrossV2Block(d_in, int(cross_rank)) for _ in range(int(n_cross_layers))]
            )
            self.block1 = _GLUBlock(d_in, int(hid1))
            self.drop1 = nn.Dropout(float(feat_dropout))
            self.block2 = _GLUBlock(int(hid1), int(hid2))
            self.drop2 = nn.Dropout(float(feat_dropout))
            self.head = nn.Linear(d_in + int(hid2), 1)
            self._init_weights()

        @staticmethod
        def _cat_emb(
            emb: "nn.Embedding | None",  # type: ignore[name-defined]
            ids: "torch.Tensor",  # type: ignore[name-defined]
            d: int,
        ) -> "torch.Tensor":  # type: ignore[name-defined]
            if int(d) == 0:
                return torch.empty(
                    (int(ids.shape[0]), 0),
                    device=ids.device,
                    dtype=torch.float32,
                )
            assert emb is not None
            return emb(ids)

        def _init_weights(self) -> None:
            for emb in (
                self.subject_emb, self.bc_emb, self.cluster_emb,
                self.family_emb, self.macro_emb, self.org_emb, self.topic_emb,
            ):
                if emb is None:
                    continue
                nn.init.normal_(emb.weight, std=0.01)
                # UNK row -> zero so cold-start equals "ignore this field".
                with torch.no_grad():
                    emb.weight[-1].zero_()
            for lin in (
                self.block1.value, self.block1.gate,
                self.block2.value, self.block2.gate, self.head,
            ):
                nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            with torch.no_grad():
                self.block1.gate.bias.fill_(1.0)
                self.block2.gate.bias.fill_(1.0)
            # DCN-v2 cross blocks: small init so the residual stays close
            # to x_0 at start (prevents early-epoch explosion when d_in
            # is large).
            for cb in self.cross_blocks:
                nn.init.kaiming_uniform_(cb.V.weight, a=math.sqrt(5))
                nn.init.zeros_(cb.U.weight)
                nn.init.zeros_(cb.U.bias)

        def forward(
            self,
            s: "torch.Tensor", b: "torch.Tensor", c: "torch.Tensor",  # type: ignore[name-defined]
            f: "torch.Tensor", mf: "torch.Tensor",  # type: ignore[name-defined]
            o: "torch.Tensor", t: "torch.Tensor",   # type: ignore[name-defined]
            m: "torch.Tensor",                       # type: ignore[name-defined]
        ) -> "torch.Tensor":                        # type: ignore[name-defined]
            x0 = torch.cat([
                self._cat_emb(self.subject_emb, s, self._d_subj),
                self._cat_emb(self.bc_emb, b, self._d_bc),
                self._cat_emb(self.cluster_emb, c, self._d_cluster),
                self.family_emb(f), self.macro_emb(mf), self.org_emb(o),
                self.topic_emb(t), m,
            ], dim=1)
            # Cross tower.
            x = x0
            for cb in self.cross_blocks:
                x = cb(x0, x)
            # Deep tower.
            h = self.drop1(self.block1(x0))
            h = self.drop2(self.block2(h))
            # Head over concat(cross, deep).
            z = self.head(torch.cat([x, h], dim=1)).squeeze(-1)
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

    # --- LR schedule: linear warmup + cosine decay ---
    base_lr = float(learning_rate)
    warmup_e = int(warmup_epochs)

    def _lr_at_epoch(ep: int) -> float:
        ep = int(ep)
        if warmup_e > 0 and ep < warmup_e:
            return base_lr * float(ep + 1) / float(warmup_e + 1)
        if not bool(use_cosine_schedule):
            return base_lr
        # Cosine from epoch warmup_e to the end of training.
        total = max(1, int(epochs) - warmup_e)
        t = float(ep - warmup_e) / float(total)
        t = max(0.0, min(1.0, t))
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))

    # --- EMA shadow state (kept in CPU memory to keep GPU free) ---
    use_ema = float(ema_decay) > 0.0
    if use_ema:
        ema_state: dict[str, "torch.Tensor"] = {  # type: ignore[name-defined]
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
    else:
        ema_state = {}

    # --- Snapshot ensemble: keep the top-K best val checkpoints ---
    K = int(snapshot_ensemble_k)
    # Each entry: (val_loss, epoch, dict[name -> cpu tensor]). Kept
    # sorted ascending by val_loss; worst entry evicted when full.
    snapshots: list[tuple[float, int, dict[str, "torch.Tensor"]]] = []  # type: ignore[name-defined]

    # --- Standardize numerical channel on the GPU once per row block ---
    mu_t = torch.from_numpy(num_mean.astype(np.float32)).to(dev)
    sd_t = torch.from_numpy(num_std.astype(np.float32)).to(dev)

    # Persistent val tensors -- val is fixed across epochs.
    def _clamp(arr: np.ndarray, n_known: int) -> np.ndarray:
        out = np.where((arr >= 0) & (arr < int(n_known)), arr, int(n_known))
        return out.astype(np.int64)

    s_val_np = _clamp(s_arr[val_idx], n_subjects)
    b_val_np = _clamp(b_arr[val_idx], n_bcs)
    c_val_np = _clamp(c_arr[val_idx], n_clusters)
    f_val_np = _clamp(f_arr[val_idx], n_families)
    mf_val_np = _clamp(mf_arr[val_idx], n_macro_families)
    o_val_np = _clamp(o_arr[val_idx], n_organizations)
    t_val_np = _clamp(t_arr[val_idx], n_bench_topics)
    s_val = torch.from_numpy(s_val_np).to(dev)
    b_val = torch.from_numpy(b_val_np).to(dev)
    c_val = torch.from_numpy(c_val_np).to(dev)
    f_val = torch.from_numpy(f_val_np).to(dev)
    mf_val = torch.from_numpy(mf_val_np).to(dev)
    o_val = torch.from_numpy(o_val_np).to(dev)
    t_val = torch.from_numpy(t_val_np).to(dev)
    M_val = torch.from_numpy(M_arr[val_idx].astype(np.float32)).to(dev)
    y_val = torch.from_numpy(y_arr[val_idx].astype(np.float32)).to(dev)
    M_val_z = ((M_val - mu_t) / sd_t).nan_to_num_(0.0, posinf=0.0, neginf=0.0)

    def _set_lr(lr: float) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = float(lr)

    def _ema_step() -> None:
        if not use_ema:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                shadow = ema_state[k]
                if v.dtype.is_floating_point:
                    shadow.mul_(float(ema_decay)).add_(
                        v.detach(), alpha=1.0 - float(ema_decay),
                    )
                else:
                    # Buffer (e.g. num_batches_tracked) -- just copy.
                    shadow.copy_(v.detach())

    def _eval_val_with_state(weights: dict[str, "torch.Tensor"]) -> float:  # type: ignore[name-defined]
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            z_v = model(s_val, b_val, c_val, f_val, mf_val, o_val, t_val, M_val_z)
            loss = float(bce(z_v, y_val).item())
        model.load_state_dict(saved)
        return loss

    best_val_nll = float("inf")
    patience_left = int(early_stopping_patience)
    train_loss_last = float("nan")
    val_loss_last = float("nan")
    epochs_run = 0

    t0 = time.time()
    epoch_iter = range(int(epochs))
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter, desc="[Member 2 MLP] epochs", total=int(epochs),
        )
    for epoch in epoch_iter:
        epochs_run = int(epoch) + 1
        _set_lr(_lr_at_epoch(epoch))
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
            f_b = f_arr[rows]
            mf_b = mf_arr[rows]
            o_b = o_arr[rows]
            t_b = t_arr[rows]
            M_b = M_arr[rows]
            y_b = y_arr[rows]
            # UNK-clamp out-of-range ids, then apply per-field categorical dropout.
            s_keep = (s_b >= 0) & (s_b < UNK_S)
            b_keep = (b_b >= 0) & (b_b < UNK_B)
            c_keep = (c_b >= 0) & (c_b < UNK_C)
            f_keep = (f_b >= 0) & (f_b < UNK_F)
            mf_keep = (mf_b >= 0) & (mf_b < UNK_MF)
            o_keep = (o_b >= 0) & (o_b < UNK_O)
            t_keep = (t_b >= 0) & (t_b < UNK_T)
            if float(cat_dropout_subject) > 0:
                s_keep &= rng.random(len(s_b)) >= float(cat_dropout_subject)
            if float(cat_dropout_bc) > 0:
                b_keep &= rng.random(len(b_b)) >= float(cat_dropout_bc)
            if float(cat_dropout_cluster) > 0:
                c_keep &= rng.random(len(c_b)) >= float(cat_dropout_cluster)
            if float(cat_dropout_family) > 0:
                f_keep &= rng.random(len(f_b)) >= float(cat_dropout_family)
            if float(cat_dropout_macro) > 0:
                mf_keep &= rng.random(len(mf_b)) >= float(cat_dropout_macro)
            if float(cat_dropout_org) > 0:
                o_keep &= rng.random(len(o_b)) >= float(cat_dropout_org)
            if float(cat_dropout_topic) > 0:
                t_keep &= rng.random(len(t_b)) >= float(cat_dropout_topic)
            s_b = np.where(s_keep, s_b, UNK_S).astype(np.int64)
            b_b = np.where(b_keep, b_b, UNK_B).astype(np.int64)
            c_b = np.where(c_keep, c_b, UNK_C).astype(np.int64)
            f_b = np.where(f_keep, f_b, UNK_F).astype(np.int64)
            mf_b = np.where(mf_keep, mf_b, UNK_MF).astype(np.int64)
            o_b = np.where(o_keep, o_b, UNK_O).astype(np.int64)
            t_b = np.where(t_keep, t_b, UNK_T).astype(np.int64)

            s_t = torch.from_numpy(s_b).to(dev, non_blocking=True)
            b_t = torch.from_numpy(b_b).to(dev, non_blocking=True)
            c_t = torch.from_numpy(c_b).to(dev, non_blocking=True)
            f_t = torch.from_numpy(f_b).to(dev, non_blocking=True)
            mf_t = torch.from_numpy(mf_b).to(dev, non_blocking=True)
            o_t = torch.from_numpy(o_b).to(dev, non_blocking=True)
            t_t = torch.from_numpy(t_b).to(dev, non_blocking=True)
            M_t = torch.from_numpy(M_b.astype(np.float32)).to(dev, non_blocking=True)
            y_t = torch.from_numpy(y_b.astype(np.float32)).to(dev, non_blocking=True)
            M_t = ((M_t - mu_t) / sd_t).nan_to_num_(0.0, posinf=0.0, neginf=0.0)

            # Optional Mixup on the numerical channel ONLY. We don't mix
            # categorical ids -- mixing categorical embeddings would
            # only blur the discrete-id signal we just embedded.
            if float(mixup_alpha) > 0.0:
                lam_np = float(
                    np.clip(
                        np.random.default_rng().beta(
                            float(mixup_alpha), float(mixup_alpha),
                        ),
                        0.0, 1.0,
                    )
                )
                perm_t = torch.randperm(int(M_t.shape[0]), device=dev)
                M_t = lam_np * M_t + (1.0 - lam_np) * M_t[perm_t]
                y_t = lam_np * y_t + (1.0 - lam_np) * y_t[perm_t]

            # Label smoothing for binary targets.
            if float(label_smoothing) > 0.0:
                eps = float(label_smoothing)
                y_t = y_t * (1.0 - eps) + 0.5 * eps

            optimizer.zero_grad(set_to_none=True)
            z = model(s_t, b_t, c_t, f_t, mf_t, o_t, t_t, M_t)
            loss = bce(z, y_t)
            loss.backward()
            optimizer.step()
            _ema_step()

            bs_actual = int(y_t.shape[0])
            running_loss += float(loss.item()) * bs_actual
            running_n += bs_actual

        train_loss_last = float(running_loss / max(running_n, 1))

        # Val pass: evaluate EMA weights when EMA is enabled (a
        # well-known regulariser; usually 0.001-0.005 nats better than
        # raw weights on tabular metadata MLPs).
        if use_ema:
            val_loss = _eval_val_with_state(ema_state)
        else:
            model.eval()
            with torch.no_grad():
                z_val = model(
                    s_val, b_val, c_val, f_val, mf_val, o_val, t_val, M_val_z,
                )
                val_loss = float(bce(z_val, y_val).item())
        val_loss_last = val_loss

        if (epoch % max(int(log_every), 1) == 0) or (epoch + 1 == int(epochs)):
            LOG.info(
                "[Member 2 MLP] epoch %d/%d  lr=%.2e  train_loss=%.5f  "
                "val_loss=%.5f  best=%.5f  patience=%d  elapsed=%.1fs",
                epoch + 1, int(epochs), _lr_at_epoch(epoch),
                train_loss_last, val_loss, best_val_nll, patience_left,
                time.time() - t0,
            )

        # Snapshot ensemble bookkeeping: maintain top-K by val_loss.
        snap_weights = ema_state if use_ema else {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        # Make sure stored snapshots live on CPU (keeps GPU memory flat).
        snap_weights_cpu = {
            k: v.detach().cpu().clone() for k, v in snap_weights.items()
        }
        snapshots.append((float(val_loss), int(epoch) + 1, snap_weights_cpu))
        snapshots.sort(key=lambda x: x[0])
        if len(snapshots) > K:
            snapshots = snapshots[:K]

        if val_loss + 1.0e-6 < best_val_nll:
            best_val_nll = val_loss
            patience_left = int(early_stopping_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOG.info(
                    "[Member 2 MLP] early stop at epoch %d "
                    "(best val_loss=%.5f).", epoch + 1, best_val_nll,
                )
                break

    # --- Finalise: average top-K snapshots into a single set of weights ---
    if len(snapshots) == 0:
        # Fall back to the live weights if nothing was recorded (e.g.
        # epochs=0 path).
        final_weights = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    elif len(snapshots) == 1:
        final_weights = copy.deepcopy(snapshots[0][2])
    else:
        first = snapshots[0][2]
        avg: dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
        for k, v in first.items():
            if v.dtype.is_floating_point:
                acc = torch.zeros_like(v, dtype=torch.float64)
                for _, _, w in snapshots:
                    acc.add_(w[k].detach().to(torch.float64))
                acc.div_(float(len(snapshots)))
                avg[k] = acc.to(v.dtype)
            else:
                avg[k] = snapshots[0][2][k].clone()
        final_weights = avg
        if math.isfinite(best_val_nll):
            # Evaluate the snapshot-averaged weights on val; if it's
            # worse than the single best snapshot, fall back to that
            # (snapshot averaging can hurt when the K snapshots come
            # from very different basins).
            avg_val = _eval_val_with_state(final_weights)
            if avg_val > snapshots[0][0] + 1.0e-5:
                LOG.info(
                    "[Member 2 MLP] snapshot avg val=%.5f WORSE than best "
                    "single snapshot val=%.5f -- using single best.",
                    avg_val, snapshots[0][0],
                )
                final_weights = copy.deepcopy(snapshots[0][2])
                best_val_nll = snapshots[0][0]
            else:
                LOG.info(
                    "[Member 2 MLP] snapshot avg (k=%d) val=%.5f vs best "
                    "single val=%.5f -- using avg.",
                    len(snapshots), avg_val, snapshots[0][0],
                )
                best_val_nll = avg_val

    model.load_state_dict(final_weights)
    model.eval()

    # --- Materialize the pure-numpy state ---
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def _w(key: str) -> np.ndarray:
        return sd[key].astype(np.float32, copy=False)

    def _emb_weight(key: str, n_cat: int, d: int) -> np.ndarray:
        """Embedding table for state export; synthesise [n+1, 0] when d=0."""
        if int(d) == 0:
            return np.zeros((int(n_cat) + 1, 0), dtype=np.float32)
        return _w(key)

    cross_V_list: list[np.ndarray] = []
    cross_U_list: list[np.ndarray] = []
    cross_b_list: list[np.ndarray] = []
    for li in range(int(n_cross_layers)):
        # PyTorch Linear stores weight as [out, in]. We store V as
        # [d_in, rank] and U as [rank, d_in] so the numpy forward is
        #   proj = x @ V; out = proj @ U + b
        # which matches the torch forward
        #   self.U(self.V(x)) = (V.weight @ x.T).T @ U.weight.T + U.bias
        # i.e. x @ V.weight.T @ U.weight.T + U.bias.
        cross_V_list.append(_w(f"cross_blocks.{li}.V.weight").T.copy())
        cross_U_list.append(_w(f"cross_blocks.{li}.U.weight").T.copy())
        cross_b_list.append(_w(f"cross_blocks.{li}.U.bias"))

    state = Member2MLPState(
        subject_emb=_emb_weight("subject_emb.weight", n_subjects, d_subj),
        bc_emb=_emb_weight("bc_emb.weight", n_bcs, d_bc),
        cluster_emb=_emb_weight("cluster_emb.weight", n_clusters, d_cluster),
        family_emb=_w("family_emb.weight"),
        macro_emb=_w("macro_emb.weight"),
        org_emb=_w("org_emb.weight"),
        topic_emb=_w("topic_emb.weight"),
        cross_V=cross_V_list,
        cross_U=cross_U_list,
        cross_b=cross_b_list,
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
        num_mean=num_mean,
        num_std=num_std,
        subject_keys=tuple(str(s) for s in subject_keys),
        bc_keys=tuple(str(s) for s in bc_keys),
        n_subjects=int(n_subjects),
        n_bcs=int(n_bcs),
        n_clusters=int(n_clusters),
        n_families=int(n_families),
        n_macro_families=int(n_macro_families),
        n_organizations=int(n_organizations),
        n_bench_topics=int(n_bench_topics),
        d_subj=int(d_subj),
        d_bc=int(d_bc),
        d_cluster=int(d_cluster),
        d_family=int(d_family),
        d_macro=int(d_macro),
        d_org=int(d_org),
        d_topic=int(d_topic),
        hid1=int(hid1),
        hid2=int(hid2),
        n_cross_layers=int(n_cross_layers),
        cross_rank=int(cross_rank),
        num_feature_names=tuple(str(s) for s in num_feature_names),
        n_num=int(n_num_actual),
        n_subj_num=int(n_subj_num),
        n_bench_num=int(n_bench_num),
        n_marginals=int(n_marginals),
        fit_method=(
            f"adamw_dcnv2_glu (dev={device}, ema={ema_decay:.3f}, "
            f"snapk={snapshot_ensemble_k}, ls={label_smoothing:.3f}, "
            f"mixup={mixup_alpha:.2f})"
        ),
        n_train=int(n_train),
        n_pos=int(n_pos),
        train_loss=float(train_loss_last),
        val_loss=float(best_val_nll if math.isfinite(best_val_nll) else val_loss_last),
        cat_dropout_subject=float(cat_dropout_subject),
        cat_dropout_bc=float(cat_dropout_bc),
        cat_dropout_cluster=float(cat_dropout_cluster),
        cat_dropout_family=float(cat_dropout_family),
        cat_dropout_macro=float(cat_dropout_macro),
        cat_dropout_org=float(cat_dropout_org),
        cat_dropout_topic=float(cat_dropout_topic),
        weight_decay=float(weight_decay),
        learning_rate=float(learning_rate),
        epochs_run=int(epochs_run),
        label_smoothing=float(label_smoothing),
        mixup_alpha=float(mixup_alpha),
        ema_decay=float(ema_decay),
        snapshot_ensemble_k=int(snapshot_ensemble_k),
        warmup_epochs=int(warmup_epochs),
    )
    LOG.info(
        "[Member 2 MLP] fit done: n_train=%d n_val=%d n_pos=%d  "
        "best_val_loss=%.5f  train_loss=%.5f  epochs_run=%d  device=%s  "
        "elapsed=%.1fs  d_in=%d  n_num=%d (subj=%d, bench=%d, marg=%d)",
        n_train, n_val, n_pos, state.val_loss, state.train_loss,
        epochs_run, device, time.time() - t0,
        d_in, n_num_actual, n_subj_num, n_bench_num, n_marginals,
    )

    # ---- Explicit teardown before returning ----
    #
    # Training holds: the live torch model on the chosen device, the
    # AdamW optimizer state (one fp32 buffer per parameter), the
    # snapshot-ensemble queue (CPU-side copies of the top-K best
    # weights), the optional EMA shadow state, and the persistent
    # GPU val tensors (``s_val``, ..., ``M_val_z`` -- the largest is
    # ``M_val_z`` at ``[n_val, n_num]`` float32). After we've
    # materialised the pure-numpy ``Member2MLPState`` above we don't
    # need any of them; dropping them now + flushing the CUDA caching
    # allocator means the next stage (M3/M4/M6 fit, OOF M1
    # retraining, ...) sees the VRAM as free instead of reserved.
    #
    # NOTE: rebind to ``None`` rather than ``del`` so static analysis
    # doesn't trip over the closures (``_set_lr``, ``_ema_step``,
    # ``_eval_val_with_state``) that captured these names. The
    # closures themselves go out of scope on ``return``.
    model = None  # noqa: F841
    optimizer = None  # noqa: F841
    s_val = b_val = c_val = f_val = mf_val = o_val = t_val = None  # noqa: F841
    M_val = M_val_z = y_val = None  # noqa: F841
    mu_t = sd_t = None  # noqa: F841
    snapshots = None  # noqa: F841
    if use_ema:
        ema_state = None  # noqa: F841
    import gc as _gc

    _gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return state


__all__ = [
    "Member2MLPState",
    "assemble_numerical",
    "numerical_feature_names",
    "fit_member2_metadata_mlp",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
    "apply_state_one",
]
