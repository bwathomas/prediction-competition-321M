"""Generic GLU-MLP ensemble member.

This module backs the two feature-diversity members added in the
May-2026 diversification pass:

* **Member 7 (pure-marginal MLP)** -- a small GLU-MLP on the 14
  mean-encoded marginal features ONLY (no embeddings, no raw
  metadata). Captures non-linear interactions among the marginals
  that the additive-linear Member 4 and the field-structured-bilinear
  Member 6 cannot express.
* **Member 8 (embeddings MLP)** -- a learned subject-id embedding
  concatenated with the full item embedding, fed to a GLU-MLP. A
  collaborative + content model, structurally distinct from Member 1's
  IRT-MLP head.

The same trainer serves both via three optional input channels that
are concatenated (in this canonical order) to form the MLP input:

    [ subject_embedding(subject_id) | item_embedding | dense_numeric ]

Any channel may be absent:

* M7 uses only ``dense_X`` (the 14 marginals).
* M8 uses ``subject_ids`` (learned embedding) + item embeddings.

Memory contract
---------------
Item embeddings are **never** materialised as a dense ``[N, D]`` array
for the full row set (that would be ~80 GB at 5M rows x 4096). Instead
the trainer takes a ``[n_unique_items, D]`` matrix plus a ``[N]``
``row_to_uniq`` index and gathers each minibatch's item rows on the
fly, exactly like Member 1's ``IndexedEmbeddingView``.

Runtime contract
----------------
``apply_batch(state, subject_ids=..., item_emb=..., dense_X=...)``
returns ``[N]`` probabilities using only numpy. The caller gathers the
per-row item embeddings (``item_emb`` is already ``[N, D]``).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("mlp_member")

_EPS = 1.0e-6


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MlpMemberState:
    """Fitted GLU-MLP member parameters + provenance.

    The model is a two-layer GLU tower over the concatenated input
    channels, followed by a single linear head producing a logit.
    """

    # ---- input channel config ----
    subj_emb: np.ndarray          # [n_subjects + 1, subj_emb_dim] (UNK at last row); [0, 0] if unused
    n_subjects: int
    subj_emb_dim: int
    use_item_emb: bool
    item_emb_dim: int
    dense_dim: int
    dense_mean: np.ndarray        # [dense_dim] float32 (z-score stats; [0] if no dense)
    dense_std: np.ndarray         # [dense_dim] float32
    dense_feature_names: tuple[str, ...]

    # ---- GLU tower ----
    l1_value_W: np.ndarray        # [in_dim, hid1]
    l1_value_b: np.ndarray        # [hid1]
    l1_gate_W: np.ndarray
    l1_gate_b: np.ndarray
    l2_value_W: np.ndarray        # [hid1, hid2]
    l2_value_b: np.ndarray
    l2_gate_W: np.ndarray
    l2_gate_b: np.ndarray
    head_W: np.ndarray            # [hid2]
    head_b: float

    in_dim: int
    hid1: int
    hid2: int

    # ---- provenance ----
    n_train: int
    n_pos: int
    train_loss: float
    val_loss: float

    def __post_init__(self) -> None:
        exp_in = (
            int(self.subj_emb_dim)
            + (int(self.item_emb_dim) if self.use_item_emb else 0)
            + int(self.dense_dim)
        )
        if exp_in != int(self.in_dim):
            raise ValueError(
                f"in_dim {self.in_dim} != subj({self.subj_emb_dim}) + "
                f"item({self.item_emb_dim if self.use_item_emb else 0}) + "
                f"dense({self.dense_dim}) = {exp_in}"
            )
        if int(self.l1_value_W.shape[0]) != int(self.in_dim):
            raise ValueError(
                f"l1_value_W rows {self.l1_value_W.shape[0]} != in_dim {self.in_dim}"
            )
        if int(self.head_W.shape[0]) != int(self.hid2):
            raise ValueError(
                f"head_W len {self.head_W.shape[0]} != hid2 {self.hid2}"
            )
        if self.subj_emb_dim > 0 and int(self.subj_emb.shape[0]) != int(self.n_subjects) + 1:
            raise ValueError(
                f"subj_emb rows {self.subj_emb.shape[0]} != n_subjects+1 "
                f"{self.n_subjects + 1}"
            )
        if int(len(self.dense_feature_names)) != int(self.dense_dim):
            raise ValueError(
                f"dense_feature_names len {len(self.dense_feature_names)} "
                f"!= dense_dim {self.dense_dim}"
            )
        for nm, arr in [
            ("l1_value_W", self.l1_value_W), ("l2_value_W", self.l2_value_W),
            ("head_W", self.head_W),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"MlpMemberState: {nm} contains NaN/Inf")
        if not math.isfinite(float(self.head_b)):
            raise ValueError("MlpMemberState: head_b is NaN/Inf")

    # ---- I/O ----
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "weights.npz",
            subj_emb=self.subj_emb.astype(np.float32),
            dense_mean=self.dense_mean.astype(np.float32),
            dense_std=self.dense_std.astype(np.float32),
            l1_value_W=self.l1_value_W.astype(np.float32),
            l1_value_b=self.l1_value_b.astype(np.float32),
            l1_gate_W=self.l1_gate_W.astype(np.float32),
            l1_gate_b=self.l1_gate_b.astype(np.float32),
            l2_value_W=self.l2_value_W.astype(np.float32),
            l2_value_b=self.l2_value_b.astype(np.float32),
            l2_gate_W=self.l2_gate_W.astype(np.float32),
            l2_gate_b=self.l2_gate_b.astype(np.float32),
            head_W=self.head_W.astype(np.float32),
            head_b=np.float32(self.head_b),
        )
        meta = {
            "n_subjects": int(self.n_subjects),
            "subj_emb_dim": int(self.subj_emb_dim),
            "use_item_emb": bool(self.use_item_emb),
            "item_emb_dim": int(self.item_emb_dim),
            "dense_dim": int(self.dense_dim),
            "dense_feature_names": list(self.dense_feature_names),
            "in_dim": int(self.in_dim),
            "hid1": int(self.hid1),
            "hid2": int(self.hid2),
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "format_version": 1,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "MlpMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz") as npz:
            kw = {k: npz[k] for k in npz.files}
        return cls(
            subj_emb=kw["subj_emb"].astype(np.float32, copy=False),
            n_subjects=int(meta["n_subjects"]),
            subj_emb_dim=int(meta["subj_emb_dim"]),
            use_item_emb=bool(meta["use_item_emb"]),
            item_emb_dim=int(meta["item_emb_dim"]),
            dense_dim=int(meta["dense_dim"]),
            dense_mean=kw["dense_mean"].astype(np.float32, copy=False),
            dense_std=kw["dense_std"].astype(np.float32, copy=False),
            dense_feature_names=tuple(meta["dense_feature_names"]),
            l1_value_W=kw["l1_value_W"].astype(np.float32, copy=False),
            l1_value_b=kw["l1_value_b"].astype(np.float32, copy=False),
            l1_gate_W=kw["l1_gate_W"].astype(np.float32, copy=False),
            l1_gate_b=kw["l1_gate_b"].astype(np.float32, copy=False),
            l2_value_W=kw["l2_value_W"].astype(np.float32, copy=False),
            l2_value_b=kw["l2_value_b"].astype(np.float32, copy=False),
            l2_gate_W=kw["l2_gate_W"].astype(np.float32, copy=False),
            l2_gate_b=kw["l2_gate_b"].astype(np.float32, copy=False),
            head_W=kw["head_W"].astype(np.float32, copy=False),
            head_b=float(kw["head_b"]),
            in_dim=int(meta["in_dim"]),
            hid1=int(meta["hid1"]),
            hid2=int(meta["hid2"]),
            n_train=int(meta["n_train"]),
            n_pos=int(meta["n_pos"]),
            train_loss=float(meta["train_loss"]),
            val_loss=float(meta["val_loss"]),
        )


# ---------------------------------------------------------------------------
# Pure-numpy forward (runtime inference path)
# ---------------------------------------------------------------------------


def _build_input(
    state: MlpMemberState,
    subject_ids: np.ndarray | None,
    item_emb: np.ndarray | None,
    dense_X: np.ndarray | None,
    n_rows: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if state.subj_emb_dim > 0:
        if subject_ids is None:
            raise ValueError("state expects subject_ids but none were given")
        s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
        unk = int(state.n_subjects)
        s = np.where((s >= 0) & (s < state.n_subjects), s, unk)
        parts.append(state.subj_emb[s].astype(np.float32, copy=False))
    if state.use_item_emb:
        if item_emb is None:
            raise ValueError("state expects item_emb but none was given")
        e = np.asarray(item_emb, dtype=np.float32)
        if e.ndim != 2 or int(e.shape[1]) != int(state.item_emb_dim):
            raise ValueError(
                f"item_emb shape {e.shape} must be (N, {state.item_emb_dim})"
            )
        parts.append(e)
    if state.dense_dim > 0:
        if dense_X is None:
            raise ValueError("state expects dense_X but none was given")
        m = np.asarray(dense_X, dtype=np.float32)
        if m.ndim != 2 or int(m.shape[1]) != int(state.dense_dim):
            raise ValueError(
                f"dense_X shape {m.shape} must be (N, {state.dense_dim})"
            )
        mz = (m - state.dense_mean) / state.dense_std
        mz = np.where(np.isfinite(mz), mz, 0.0).astype(np.float32, copy=False)
        parts.append(mz)
    if not parts:
        raise ValueError("MlpMemberState has no active input channel")
    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if int(x.shape[0]) != int(n_rows):
        raise ValueError(f"assembled input has {x.shape[0]} rows, expected {n_rows}")
    return x


def _glu(x: np.ndarray, vW, vb, gW, gb) -> np.ndarray:
    val = x @ vW + vb
    gate = _sigmoid_stable(x @ gW + gb).astype(np.float32)
    return (val * gate).astype(np.float32, copy=False)


def apply_batch(
    state: MlpMemberState,
    *,
    subject_ids: np.ndarray | None = None,
    item_emb: np.ndarray | None = None,
    dense_X: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-numpy forward. Returns ``[N]`` float32 probabilities."""
    # Determine N from whichever channel is present.
    if subject_ids is not None:
        n_rows = int(np.asarray(subject_ids).reshape(-1).shape[0])
    elif item_emb is not None:
        n_rows = int(np.asarray(item_emb).shape[0])
    elif dense_X is not None:
        n_rows = int(np.asarray(dense_X).shape[0])
    else:
        raise ValueError("apply_batch: no input channel provided")
    x = _build_input(state, subject_ids, item_emb, dense_X, n_rows)
    h1 = _glu(x, state.l1_value_W, state.l1_value_b, state.l1_gate_W, state.l1_gate_b)
    h2 = _glu(h1, state.l2_value_W, state.l2_value_b, state.l2_gate_W, state.l2_gate_b)
    z = (h2 @ state.head_W).reshape(-1) + float(state.head_b)
    p = _sigmoid_stable(z)
    return np.clip(p, _EPS, 1.0 - _EPS).astype(np.float32)


def apply_one(
    state: MlpMemberState,
    *,
    subject_id: int | None = None,
    item_emb: np.ndarray | None = None,
    dense_X: np.ndarray | None = None,
) -> float:
    sid = None if subject_id is None else np.array([int(subject_id)], dtype=np.int64)
    ie = None if item_emb is None else np.asarray(item_emb, dtype=np.float32).reshape(1, -1)
    dx = None if dense_X is None else np.asarray(dense_X, dtype=np.float32).reshape(1, -1)
    return float(apply_batch(state, subject_ids=sid, item_emb=ie, dense_X=dx)[0])


def apply_state_batch(state: MlpMemberState, **kwargs) -> np.ndarray:
    return apply_batch(state, **kwargs)


# ---------------------------------------------------------------------------
# Offline trainer (torch; notebook-only)
# ---------------------------------------------------------------------------


def fit_mlp_member(
    *,
    labels: np.ndarray,                       # [N] in {0, 1}
    subject_ids: np.ndarray | None = None,    # [N] int (M8)
    n_subjects: int = 0,
    subj_emb_dim: int = 0,
    item_emb_unique: np.ndarray | None = None,  # [n_uniq, D] float32 (M8)
    row_to_uniq: np.ndarray | None = None,      # [N] int into item_emb_unique
    dense_X: np.ndarray | None = None,          # [N, F] float32 (M7)
    dense_feature_names: Sequence[str] = (),
    hid1: int = 128,
    hid2: int = 64,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    epochs: int = 40,
    batch_size: int = 16384,
    val_fraction: float = 0.1,
    early_stopping_patience: int = 6,
    warmup_epochs: int = 2,
    use_cosine_schedule: bool = True,
    feat_dropout: float = 0.10,
    seed: int = 0,
    device: str | None = None,
    holdout_group_id: np.ndarray | None = None,
    show_progress: bool = True,
    ncl_anchor_preds: np.ndarray | None = None,
    ncl_lambda: float = 0.0,
) -> MlpMemberState:
    """Train a two-layer GLU-MLP via Adam + early stopping.

    Exactly one of the embedding/dense channels must be active.
    Standardisation stats for ``dense_X`` are computed on the internal
    train slice only and baked into the saved state.

    Negative-correlation learning (optional): when ``ncl_anchor_preds`` (a
    ``[N]`` array of frozen anchor probabilities, row-aligned with ``labels``)
    and ``ncl_lambda > 0`` are supplied, the per-batch loss gains a term
    ``ncl_lambda * mean((p - y)(p_anchor - y))``. Minimising it drives this
    member's signed errors to be negatively correlated with the anchors',
    i.e. it learns to be right where the anchors are wrong. The penalty is a
    regulariser on the *training* rows only (it never sees held-out labels),
    so OOF predictions remain honest.
    """
    import torch
    import torch.nn as nn

    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    N = int(y.shape[0])
    if N == 0:
        raise ValueError("fit_mlp_member: empty label array")

    use_subj = int(subj_emb_dim) > 0
    use_item = item_emb_unique is not None
    use_dense = dense_X is not None

    if use_subj and subject_ids is None:
        raise ValueError("subj_emb_dim > 0 requires subject_ids")
    if use_item and row_to_uniq is None:
        raise ValueError("item_emb_unique requires row_to_uniq")
    if not (use_subj or use_item or use_dense):
        raise ValueError("fit_mlp_member: no input channel active")

    rng = np.random.default_rng(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- standardise dense channel on the (group-aware) train slice ----
    if use_dense:
        dense_X = np.asarray(dense_X, dtype=np.float32)
        if dense_X.ndim != 2 or int(dense_X.shape[0]) != N:
            raise ValueError(f"dense_X shape {dense_X.shape} must be (N={N}, F)")
        dense_dim = int(dense_X.shape[1])
        if len(dense_feature_names) != dense_dim:
            dense_feature_names = tuple(f"dense_{i}" for i in range(dense_dim))
    else:
        dense_dim = 0
        dense_feature_names = ()

    # ---- internal train/val split (group-aware if holdout_group_id) ----
    if holdout_group_id is not None and int(np.asarray(holdout_group_id).reshape(-1).shape[0]) == N:
        groups = np.asarray(holdout_group_id).reshape(-1)
        uniq_groups = np.unique(groups[groups >= 0])
        n_val_groups = max(1, int(round(val_fraction * uniq_groups.size)))
        perm_g = rng.permutation(uniq_groups.size)
        val_groups = set(uniq_groups[perm_g[:n_val_groups]].tolist())
        val_mask = np.array([g in val_groups for g in groups], dtype=bool)
    else:
        perm = rng.permutation(N)
        n_val = max(1, int(round(val_fraction * N)))
        val_mask = np.zeros(N, dtype=bool)
        val_mask[perm[:n_val]] = True
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    if tr_idx.size == 0 or va_idx.size == 0:
        tr_idx = np.arange(N)
        va_idx = np.arange(N)

    dense_mean = np.zeros(dense_dim, dtype=np.float32)
    dense_std = np.ones(dense_dim, dtype=np.float32)
    if use_dense and dense_dim > 0:
        dense_mean = dense_X[tr_idx].mean(axis=0).astype(np.float32)
        dense_std = dense_X[tr_idx].std(axis=0).astype(np.float32)
        dense_std = np.where(dense_std < 1e-6, 1.0, dense_std).astype(np.float32)

    item_emb_dim = int(item_emb_unique.shape[1]) if use_item else 0
    in_dim = int(subj_emb_dim) + int(item_emb_dim) + int(dense_dim)

    # ---- torch model ----
    torch.manual_seed(int(seed))

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if use_subj:
                self.subj = nn.Embedding(int(n_subjects) + 1, int(subj_emb_dim))
                nn.init.normal_(self.subj.weight, std=0.05)
            self.l1v = nn.Linear(in_dim, hid1)
            self.l1g = nn.Linear(in_dim, hid1)
            self.l2v = nn.Linear(hid1, hid2)
            self.l2g = nn.Linear(hid1, hid2)
            self.head = nn.Linear(hid2, 1)
            self.drop = nn.Dropout(float(feat_dropout))

        def forward(self, sid, iemb, dz):
            parts = []
            if use_subj:
                parts.append(self.subj(sid))
            if use_item:
                parts.append(iemb)
            if use_dense:
                parts.append(dz)
            x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            x = self.drop(x)
            h1 = self.l1v(x) * torch.sigmoid(self.l1g(x))
            h2 = self.l2v(h1) * torch.sigmoid(self.l2g(h1))
            return self.head(h2).reshape(-1)

    net = _Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    item_t = (
        torch.from_numpy(np.asarray(item_emb_unique, dtype=np.float32)).to(device)
        if use_item else None
    )
    sid_all = (
        torch.from_numpy(np.asarray(subject_ids, dtype=np.int64).reshape(-1)).to(device)
        if use_subj else None
    )
    r2u_all = (
        torch.from_numpy(np.asarray(row_to_uniq, dtype=np.int64).reshape(-1)).to(device)
        if use_item else None
    )
    dz_all = (
        torch.from_numpy(
            ((dense_X - dense_mean) / dense_std).astype(np.float32)
        ).to(device)
        if use_dense else None
    )
    y_t = torch.from_numpy(y).to(device)

    use_ncl = (ncl_anchor_preds is not None) and (float(ncl_lambda) != 0.0)
    if use_ncl:
        _anchor_np = np.asarray(ncl_anchor_preds, dtype=np.float32).reshape(-1)
        if _anchor_np.shape[0] != N:
            raise ValueError(
                f"ncl_anchor_preds length {_anchor_np.shape[0]} != N={N}"
            )
        anchor_t = torch.from_numpy(_anchor_np).to(device)
    else:
        anchor_t = None

    def _run_batch(idx_t: "torch.Tensor"):
        sid_b = sid_all[idx_t] if use_subj else None
        iemb_b = item_t[r2u_all[idx_t]] if use_item else None
        dz_b = dz_all[idx_t] if use_dense else None
        return net(sid_b, iemb_b, dz_b)

    n_steps_per_epoch = max(1, int(math.ceil(tr_idx.size / batch_size)))
    total_steps = n_steps_per_epoch * int(epochs)
    warmup_steps = n_steps_per_epoch * int(warmup_epochs)

    def _lr_at(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return learning_rate * (step + 1) / warmup_steps
        if not use_cosine_schedule:
            return learning_rate
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * learning_rate * (1.0 + math.cos(math.pi * min(1.0, prog)))

    tr_idx_t = torch.from_numpy(tr_idx.astype(np.int64)).to(device)
    va_idx_t = torch.from_numpy(va_idx.astype(np.int64)).to(device)

    best_val = float("inf")
    best_state: dict | None = None
    patience = 0
    step = 0
    for ep in range(int(epochs)):
        net.train()
        perm = torch.randperm(tr_idx.size, device=device)
        tr_shuf = tr_idx_t[perm]
        for bstart in range(0, tr_idx.size, batch_size):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step)
            b = tr_shuf[bstart:bstart + batch_size]
            logits = _run_batch(b)
            loss = bce(logits, y_t[b])
            if use_ncl:
                p_b = torch.sigmoid(logits)
                err_m = p_b - y_t[b]
                err_a = anchor_t[b] - y_t[b]
                loss = loss + float(ncl_lambda) * (err_m * err_a).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        # ---- validate ----
        net.eval()
        with torch.no_grad():
            vloss_sum = 0.0
            vn = 0
            for bstart in range(0, va_idx.size, batch_size):
                b = va_idx_t[bstart:bstart + batch_size]
                logits = _run_batch(b)
                vloss_sum += float(bce(logits, y_t[b])) * int(b.shape[0])
                vn += int(b.shape[0])
            vloss = vloss_sum / max(1, vn)
        if show_progress:
            LOG.info("mlp epoch %d/%d val_loss=%.5f", ep + 1, epochs, vloss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(early_stopping_patience):
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # ---- export numpy params ----
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    subj_emb = (
        sd["subj.weight"].astype(np.float32) if use_subj
        else np.zeros((0, 0), dtype=np.float32)
    )

    # nn.Linear stores weight as [out, in]; our numpy forward uses x @ W with W [in, out].
    def _W(name: str) -> np.ndarray:
        return sd[name].T.astype(np.float32, copy=False)

    def _b(name: str) -> np.ndarray:
        return sd[name].astype(np.float32, copy=False)

    # ---- final train loss (sampled) ----
    with torch.no_grad():
        net_train_loss = float("nan")
        try:
            s_n = min(tr_idx.size, 200_000)
            s_b = tr_idx_t[:s_n]
            logits = _run_batch(s_b)
            net_train_loss = float(bce(logits, y_t[s_b]))
        except Exception:  # pragma: no cover - diagnostics only
            pass

    state = MlpMemberState(
        subj_emb=subj_emb,
        n_subjects=int(n_subjects),
        subj_emb_dim=int(subj_emb_dim),
        use_item_emb=bool(use_item),
        item_emb_dim=int(item_emb_dim),
        dense_dim=int(dense_dim),
        dense_mean=dense_mean,
        dense_std=dense_std,
        dense_feature_names=tuple(dense_feature_names),
        l1_value_W=_W("l1v.weight"), l1_value_b=_b("l1v.bias"),
        l1_gate_W=_W("l1g.weight"), l1_gate_b=_b("l1g.bias"),
        l2_value_W=_W("l2v.weight"), l2_value_b=_b("l2v.bias"),
        l2_gate_W=_W("l2g.weight"), l2_gate_b=_b("l2g.bias"),
        head_W=_W("head.weight").reshape(-1),
        head_b=float(sd["head.bias"].reshape(-1)[0]),
        in_dim=int(in_dim),
        hid1=int(hid1),
        hid2=int(hid2),
        n_train=int(tr_idx.size),
        n_pos=int(y[tr_idx].sum()),
        train_loss=float(net_train_loss),
        val_loss=float(best_val),
    )
    return state


__all__ = [
    "MlpMemberState",
    "fit_mlp_member",
    "apply_batch",
    "apply_one",
    "apply_state_batch",
]
