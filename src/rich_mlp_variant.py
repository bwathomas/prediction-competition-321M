"""Experimental "rich" DCN-V2 MLP trainer for the rich-M8 / MoE probe.

The :mod:`src.mlp_variant` module trains a minimal M8-shape network:
``[subject_emb | item_emb | dense_metadata] -> GLU MLP -> head``. That is
the right baseline for loss-diversity / partition-routing probes that
hold architecture FIXED. This module is the *larger* sibling that adds
the channels the audit said M8 was blind to:

* Multiple categorical embedding tables -- ``benchmark+condition``,
  ``cluster_id``, ``family``, ``macro_family``, ``organization``,
  ``benchmark_topic`` -- plus the existing ``subject_id`` table.
* Wider dense block: the existing pool + item-type + CoT block is the
  caller's responsibility; this module simply concatenates whatever
  ``dense_X`` the caller supplies (so the NN block, centroid distances,
  subject/benchmark numerics, etc. all flow in through that one channel).
* **DCN-V2 low-rank cross layers** sitting between the concat and the
  GLU deep tower. The cross tower learns explicit polynomial
  interactions between all input channels (categorical embeddings,
  item embedding, dense numerics) without paying the full quadratic
  cost. This is the lift any ``(family, NN_support)``-style cross has
  to come from -- a plain MLP can only learn additive interactions of
  the concatenated input.
* **Per-field categorical dropout** at training time. Each batch we
  randomly mask each categorical id to the UNK row with the
  configured probability. This matches the production ~20% bc-redaction
  rate and trains the model to degrade gracefully when subject /
  benchmark metadata is absent at inference.

Design constraints
------------------
* Item embeddings are passed via ``(emb_tensor, row_to_uniq)`` -- never
  materialized as ``[N, D]`` -- same trick :mod:`src.mlp_member` uses
  to fit 5M rows x 4096-D embeddings in 24 GB.
* Any categorical channel can be disabled by setting its dimension to
  ``0`` (the embedding table is then skipped entirely; AdamW's CUDA
  foreach kernel crashes on zero-numel parameters, so this matters).
* The deep tower mirrors :mod:`src.mlp_variant`'s GLU+dropout block so
  the rich net degrades cleanly to the M8 baseline if every extra
  channel is set to ``0`` -- making "rich vs plain" comparisons an
  honest architecture+features ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RichMLPConfig:
    """Hyperparameters and channel knobs for the rich DCN-V2 MLP.

    Set any ``d_*`` to ``0`` to disable that categorical channel
    entirely (no embedding table allocated). Set per-field dropout to
    ``0.0`` to disable random UNK-masking for that field at training
    time. The caller passes the actual cardinalities of each table
    (``n_*``) at build time, separately from the dimension knobs.
    """

    # Categorical embedding dimensions (0 = disabled).
    subj_emb_dim: int = 32
    bc_emb_dim: int = 16
    cluster_emb_dim: int = 16
    family_emb_dim: int = 16
    macro_emb_dim: int = 8
    org_emb_dim: int = 16
    topic_emb_dim: int = 16

    # Deep tower.
    hid1: int = 256
    hid2: int = 128
    feat_dropout: float = 0.10

    # DCN-V2 low-rank cross tower. ``n_cross_layers=0`` disables the
    # cross path entirely (degrades to plain MLP).
    n_cross_layers: int = 2
    cross_rank: int = 64

    # Training-time categorical dropout (UNK-masking probabilities).
    # ~0.20 for benchmark matches the production redact rate so the
    # generalization story isn't fitted to a different distribution
    # than test sees.
    cat_dropout_subject: float = 0.05
    cat_dropout_bc: float = 0.20
    cat_dropout_cluster: float = 0.10
    cat_dropout_family: float = 0.05
    cat_dropout_macro: float = 0.05
    cat_dropout_org: float = 0.05
    cat_dropout_topic: float = 0.10

    # Optimizer.
    lr: float = 1.0e-3
    wd: float = 1.0e-5
    epochs: int = 30
    batch_size: int = 16384
    val_fraction: float = 0.10
    patience: int = 5

    seed: int = 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _build_arch(
    cfg: RichMLPConfig,
    *,
    n_subjects: int,
    n_bcs: int,
    n_clusters: int,
    n_families: int,
    n_macros: int,
    n_orgs: int,
    n_topics: int,
    item_dim: int,
    dense_dim: int,
    device,
):
    """Construct the DCN-V2 net described by ``cfg`` on ``device``."""

    import torch
    import torch.nn as nn

    torch.manual_seed(int(cfg.seed))

    # Resolved channel dimensions; tables with d=0 are skipped entirely
    # so we don't pay for zero-numel parameters.
    d_subj = int(cfg.subj_emb_dim)
    d_bc = int(cfg.bc_emb_dim)
    d_cl = int(cfg.cluster_emb_dim)
    d_fam = int(cfg.family_emb_dim)
    d_mac = int(cfg.macro_emb_dim)
    d_org = int(cfg.org_emb_dim)
    d_top = int(cfg.topic_emb_dim)

    in_dim = (
        d_subj + d_bc + d_cl + d_fam + d_mac + d_org + d_top
        + int(item_dim) + int(dense_dim)
    )
    if in_dim <= 0:
        raise ValueError("rich MLP needs at least one input channel")

    class _GLUBlock(nn.Module):
        def __init__(self, d_in_: int, d_out: int):
            super().__init__()
            self.value = nn.Linear(d_in_, d_out)
            self.gate = nn.Linear(d_in_, d_out)

        def forward(self, x):
            return self.value(x) * torch.sigmoid(self.gate(x))

    class _CrossV2Block(nn.Module):
        """Low-rank DCN-V2 cross layer.

        ``x_{l+1} = x_0 * (V U^T x_l + b) + x_l``. Cost is
        ``O(d_in * rank)`` per layer instead of ``O(d_in^2)``. With
        ``rank=64`` and ``d_in ~ 4300`` this is ~4 GFLOP per forward
        per layer -- negligible next to the dense item embedding.
        """

        def __init__(self, d_in_: int, rank: int):
            super().__init__()
            self.V = nn.Linear(d_in_, rank, bias=False)
            self.U = nn.Linear(rank, d_in_, bias=True)

        def forward(self, x0, x):
            return x0 * self.U(self.V(x)) + x

    class _RichMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # Each table: cardinality + 1 to hold the UNK row at the
            # end. UNK rows are zero-initialized so a missing field
            # contributes nothing to the concat (i.e. "ignore this
            # field" semantics, not "guess this field's prior").
            self.subject_emb = (
                nn.Embedding(int(n_subjects) + 1, d_subj) if d_subj > 0 else None
            )
            self.bc_emb = (
                nn.Embedding(int(n_bcs) + 1, d_bc) if d_bc > 0 else None
            )
            self.cluster_emb = (
                nn.Embedding(int(n_clusters) + 1, d_cl) if d_cl > 0 else None
            )
            self.family_emb = (
                nn.Embedding(int(n_families) + 1, d_fam) if d_fam > 0 else None
            )
            self.macro_emb = (
                nn.Embedding(int(n_macros) + 1, d_mac) if d_mac > 0 else None
            )
            self.org_emb = (
                nn.Embedding(int(n_orgs) + 1, d_org) if d_org > 0 else None
            )
            self.topic_emb = (
                nn.Embedding(int(n_topics) + 1, d_top) if d_top > 0 else None
            )

            self.cross_blocks = nn.ModuleList(
                [
                    _CrossV2Block(in_dim, int(cfg.cross_rank))
                    for _ in range(int(cfg.n_cross_layers))
                ]
            )
            self.block1 = _GLUBlock(in_dim, int(cfg.hid1))
            self.drop1 = nn.Dropout(float(cfg.feat_dropout))
            self.block2 = _GLUBlock(int(cfg.hid1), int(cfg.hid2))
            self.drop2 = nn.Dropout(float(cfg.feat_dropout))
            self.head = nn.Linear(in_dim + int(cfg.hid2), 1)

            self._init_weights()

        def _init_weights(self):
            for emb in (
                self.subject_emb, self.bc_emb, self.cluster_emb,
                self.family_emb, self.macro_emb, self.org_emb, self.topic_emb,
            ):
                if emb is None:
                    continue
                nn.init.normal_(emb.weight, std=0.05)
                with torch.no_grad():
                    emb.weight[-1].zero_()  # UNK row.
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
            # Small init for cross blocks so the residual starts as
            # ~x_0 and the deep tower can find the additive baseline
            # before the cross terms wake up.
            for cb in self.cross_blocks:
                nn.init.kaiming_uniform_(cb.V.weight, a=math.sqrt(5))
                nn.init.zeros_(cb.U.weight)
                nn.init.zeros_(cb.U.bias)

        @staticmethod
        def _maybe_emb(emb, ids, d):
            if emb is None or int(d) == 0:
                return torch.empty(
                    (int(ids.shape[0]), 0),
                    device=ids.device,
                    dtype=torch.float32,
                )
            return emb(ids)

        def forward(
            self,
            s_ids, bc_ids, cl_ids, fam_ids, mac_ids, org_ids, top_ids,
            item_emb, dense,
        ):
            parts = [
                self._maybe_emb(self.subject_emb, s_ids, d_subj),
                self._maybe_emb(self.bc_emb, bc_ids, d_bc),
                self._maybe_emb(self.cluster_emb, cl_ids, d_cl),
                self._maybe_emb(self.family_emb, fam_ids, d_fam),
                self._maybe_emb(self.macro_emb, mac_ids, d_mac),
                self._maybe_emb(self.org_emb, org_ids, d_org),
                self._maybe_emb(self.topic_emb, top_ids, d_top),
            ]
            if int(item_dim) > 0:
                parts.append(item_emb)
            if int(dense_dim) > 0 and dense is not None:
                parts.append(dense)
            x0 = torch.cat(parts, dim=1)
            x = x0
            for cb in self.cross_blocks:
                x = cb(x0, x)
            h = self.drop1(self.block1(x0))
            h = self.drop2(self.block2(h))
            return self.head(torch.cat([x, h], dim=1)).reshape(-1)

    net = _RichMLP().to(device)
    return net, in_dim


# ---------------------------------------------------------------------------
# Per-field UNK dropout (vectorized; applied per batch on the device)
# ---------------------------------------------------------------------------


def _apply_unk_dropout(
    ids, *, unk_id: int, p: float, generator,
):
    """Mask each entry of ``ids`` to ``unk_id`` with probability ``p``.

    ``generator`` is a ``torch.Generator`` on the same device as ``ids``
    so the dropout is deterministic given the seed and reproducible
    across runs.
    """
    if float(p) <= 0.0:
        return ids
    import torch

    rand = torch.rand(ids.shape, device=ids.device, generator=generator)
    return torch.where(rand < float(p), torch.full_like(ids, int(unk_id)), ids)


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


def train_rich_mlp(
    *,
    y,
    subject_ids,        # [N] int64
    bc_ids,             # [N] int64
    cluster_ids,        # [N] int64
    family_ids,         # [N] int64
    macro_ids,          # [N] int64
    org_ids,            # [N] int64
    topic_ids,          # [N] int64
    item_emb_tensor,    # [n_unique, item_dim] torch.Tensor on device
    row_to_uniq,        # [N] int64
    dense_X,            # [N, dense_dim] float32 or None
    n_subjects: int,
    n_bcs: int,
    n_clusters: int,
    n_families: int,
    n_macros: int,
    n_orgs: int,
    n_topics: int,
    cfg: RichMLPConfig,
    device,
    sample_weights=None,
    show_progress: bool = True,
):
    """Train one rich-MLP variant; return the trained ``nn.Module`` in eval mode.

    All categorical id arrays must already use the "UNK row = N + 0"
    convention from the production indexer: an id of ``-1`` or any
    value ``>= N`` is mapped to ``N`` (the UNK row) at the very start
    of training. This keeps the dataloader free of branching and lets
    cold subjects/benches share the same code path as the
    cat-dropout-masked ones.
    """
    import torch

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    N = int(y.shape[0])
    item_dim = int(item_emb_tensor.shape[1])
    dense_dim = 0 if dense_X is None else int(np.asarray(dense_X).shape[1])

    net, in_dim = _build_arch(
        cfg,
        n_subjects=n_subjects, n_bcs=n_bcs, n_clusters=n_clusters,
        n_families=n_families, n_macros=n_macros, n_orgs=n_orgs,
        n_topics=n_topics, item_dim=item_dim, dense_dim=dense_dim,
        device=device,
    )
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.wd))

    # Push all per-row tensors to the device once. Categorical ids are
    # clamped to UNK if out-of-range (so the caller doesn't have to
    # re-do the clamp).
    UNK_S = int(n_subjects)
    UNK_B = int(n_bcs)
    UNK_C = int(n_clusters)
    UNK_F = int(n_families)
    UNK_MF = int(n_macros)
    UNK_O = int(n_orgs)
    UNK_T = int(n_topics)

    def _clamp_to_unk(arr, n_known: int):
        a = np.asarray(arr, dtype=np.int64).reshape(-1)
        return np.where((a >= 0) & (a < int(n_known)), a, int(n_known)).astype(np.int64)

    s_all = torch.from_numpy(_clamp_to_unk(subject_ids, n_subjects)).to(device)
    b_all = torch.from_numpy(_clamp_to_unk(bc_ids, n_bcs)).to(device)
    c_all = torch.from_numpy(_clamp_to_unk(cluster_ids, n_clusters)).to(device)
    f_all = torch.from_numpy(_clamp_to_unk(family_ids, n_families)).to(device)
    mf_all = torch.from_numpy(_clamp_to_unk(macro_ids, n_macros)).to(device)
    o_all = torch.from_numpy(_clamp_to_unk(org_ids, n_orgs)).to(device)
    t_all = torch.from_numpy(_clamp_to_unk(topic_ids, n_topics)).to(device)
    r2u_all = torch.from_numpy(np.asarray(row_to_uniq, dtype=np.int64).reshape(-1)).to(device)
    y_all = torch.from_numpy(y).to(device)
    dz_all = (
        torch.from_numpy(np.asarray(dense_X, dtype=np.float32)).to(device)
        if dense_dim > 0 else None
    )
    w_all = (
        torch.from_numpy(np.asarray(sample_weights, dtype=np.float32).reshape(-1)).to(device)
        if sample_weights is not None else None
    )

    # Item-grouped internal val split (early stopping on held-out
    # items) -- same regime as src.mlp_variant.
    rng = np.random.default_rng(int(cfg.seed))
    groups = np.asarray(row_to_uniq, dtype=np.int64).reshape(-1)
    uniq_g = np.unique(groups)
    n_val_g = max(1, int(round(float(cfg.val_fraction) * uniq_g.size)))
    val_g = set(uniq_g[rng.permutation(uniq_g.size)[:n_val_g]].tolist())
    val_mask = np.fromiter((g in val_g for g in groups), count=N, dtype=bool)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    if tr_idx.size == 0 or va_idx.size == 0:
        tr_idx = np.arange(N)
        va_idx = np.arange(N)

    tr_t = torch.from_numpy(tr_idx.astype(np.int64)).to(device)
    va_t = torch.from_numpy(va_idx.astype(np.int64)).to(device)

    # One torch.Generator per cat field so dropout streams stay
    # decorrelated across fields. All seeded off cfg.seed so the run
    # is reproducible.
    gens = {
        name: torch.Generator(device=device).manual_seed(int(cfg.seed) + h)
        for h, name in enumerate(
            ("s", "b", "c", "f", "mf", "o", "t")
        )
    }

    def _ids_with_dropout(idx_t, training: bool):
        s = s_all[idx_t]
        b = b_all[idx_t]
        c = c_all[idx_t]
        f = f_all[idx_t]
        mf = mf_all[idx_t]
        o = o_all[idx_t]
        t = t_all[idx_t]
        if training:
            s = _apply_unk_dropout(s, unk_id=UNK_S, p=cfg.cat_dropout_subject, generator=gens["s"])
            b = _apply_unk_dropout(b, unk_id=UNK_B, p=cfg.cat_dropout_bc, generator=gens["b"])
            c = _apply_unk_dropout(c, unk_id=UNK_C, p=cfg.cat_dropout_cluster, generator=gens["c"])
            f = _apply_unk_dropout(f, unk_id=UNK_F, p=cfg.cat_dropout_family, generator=gens["f"])
            mf = _apply_unk_dropout(mf, unk_id=UNK_MF, p=cfg.cat_dropout_macro, generator=gens["mf"])
            o = _apply_unk_dropout(o, unk_id=UNK_O, p=cfg.cat_dropout_org, generator=gens["o"])
            t = _apply_unk_dropout(t, unk_id=UNK_T, p=cfg.cat_dropout_topic, generator=gens["t"])
        return s, b, c, f, mf, o, t

    def _logits(idx_t, training: bool):
        s, b, c, f, mf, o, t = _ids_with_dropout(idx_t, training)
        ie = item_emb_tensor[r2u_all[idx_t]]
        dz = dz_all[idx_t] if dz_all is not None else None
        return net(s, b, c, f, mf, o, t, ie, dz)

    n_steps = max(1, int(math.ceil(tr_idx.size / cfg.batch_size)))
    total_steps = n_steps * int(cfg.epochs)
    warmup = n_steps * 2

    def _lr_at(step):
        if step < warmup and warmup > 0:
            return float(cfg.lr) * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * float(cfg.lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _val_loss():
        net.eval()
        with torch.no_grad():
            vs, vn = 0.0, 0
            for bs in range(0, va_idx.size, cfg.batch_size):
                b = va_t[bs : bs + int(cfg.batch_size)]
                # No cat-dropout at val time: this is the "real metadata"
                # path. The training-time dropout's job is to teach the
                # model to degrade cleanly when an id is UNK, which we
                # test separately by zeroing fields explicitly.
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    _logits(b, training=False), y_all[b], reduction="sum",
                )
                vs += float(per)
                vn += int(b.shape[0])
            return vs / max(1, vn)

    best_val = float("inf")
    best_state = None
    bad = 0
    step = 0
    for ep in range(int(cfg.epochs)):
        net.train()
        perm = torch.randperm(tr_idx.size, device=device)
        tr_shuf = tr_t[perm]
        for bs in range(0, tr_idx.size, int(cfg.batch_size)):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step)
            b = tr_shuf[bs : bs + int(cfg.batch_size)]
            logits = _logits(b, training=True)
            target = y_all[b]
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none",
            )
            if w_all is not None:
                w = w_all[b]
                loss = (per * w).sum() / w.sum().clamp(min=1e-8)
            else:
                loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        vl = _val_loss()
        if show_progress:
            print(f"    epoch {ep + 1}/{cfg.epochs}  val={vl:.5f}")
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= int(cfg.patience):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_rich_mlp(
    net,
    *,
    subject_ids,
    bc_ids,
    cluster_ids,
    family_ids,
    macro_ids,
    org_ids,
    topic_ids,
    item_emb_tensor,
    row_to_uniq,
    dense_X,
    n_subjects: int,
    n_bcs: int,
    n_clusters: int,
    n_families: int,
    n_macros: int,
    n_orgs: int,
    n_topics: int,
    device,
    chunk: int = 131_072,
) -> np.ndarray:
    """Chunked forward -> ``[N]`` float32 probabilities (sigmoid of logit)."""
    import torch

    def _clamp(arr, n_known: int):
        a = np.asarray(arr, dtype=np.int64).reshape(-1)
        return np.where((a >= 0) & (a < int(n_known)), a, int(n_known)).astype(np.int64)

    s = _clamp(subject_ids, n_subjects)
    b = _clamp(bc_ids, n_bcs)
    c = _clamp(cluster_ids, n_clusters)
    f = _clamp(family_ids, n_families)
    mf = _clamp(macro_ids, n_macros)
    o = _clamp(org_ids, n_orgs)
    t = _clamp(topic_ids, n_topics)
    r = np.asarray(row_to_uniq, dtype=np.int64).reshape(-1)
    dz = None if dense_X is None else np.asarray(dense_X, dtype=np.float32)
    n = int(s.shape[0])
    out = np.empty(n, dtype=np.float32)

    net.eval()
    with torch.no_grad():
        for st in range(0, n, chunk):
            en = min(st + chunk, n)
            s_t = torch.from_numpy(s[st:en]).to(device)
            b_t = torch.from_numpy(b[st:en]).to(device)
            c_t = torch.from_numpy(c[st:en]).to(device)
            f_t = torch.from_numpy(f[st:en]).to(device)
            mf_t = torch.from_numpy(mf[st:en]).to(device)
            o_t = torch.from_numpy(o[st:en]).to(device)
            t_t = torch.from_numpy(t[st:en]).to(device)
            r_t = torch.from_numpy(r[st:en]).to(device)
            ie = item_emb_tensor[r_t]
            dz_t = torch.from_numpy(dz[st:en]).to(device) if dz is not None else None
            logits = net(s_t, b_t, c_t, f_t, mf_t, o_t, t_t, ie, dz_t)
            out[st:en] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Soft routing helpers (shared with the MoE probe notebooks)
# ---------------------------------------------------------------------------


def soft_routing_weights_categorical(
    bucket_per_row: np.ndarray, *, n_buckets: int, epsilon: float,
) -> np.ndarray:
    """Smoothed one-hot soft routing for categorical / unordered partitions.

    Each row's bucket gets weight ``(1 - eps)`` and the remaining
    ``eps`` is spread uniformly across the other ``K - 1`` buckets.
    ``eps=0`` is hard routing; ``eps=1`` is uniform averaging.
    Recommended sweep: ``[0.0, 0.05, 0.10, 0.25, 0.50, 1.0]``.
    """
    if int(n_buckets) <= 1:
        raise ValueError("n_buckets must be >= 2 for soft routing")
    eps = float(epsilon)
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    N = int(len(bucket_per_row))
    K = int(n_buckets)
    base = np.full((N, K), eps / max(1, K - 1), dtype=np.float32)
    idx = np.arange(N)
    base[idx, np.asarray(bucket_per_row, dtype=np.int64)] = 1.0 - eps
    return base


def soft_routing_weights_kernel(
    score_per_row: np.ndarray,
    *,
    bucket_centroids: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Gaussian-kernel soft routing for ordinal / continuous partitions.

    Used for NN-support buckets where each bucket has a mean
    "supportiness" score (e.g. mean cosine sim to top-K train items).
    Row weights are ``softmax(-(s - mu_k)^2 / tau^2)``, so adjacent
    buckets blend more than distant ones. Smaller ``tau`` = sharper
    (approaches hard routing); larger = smoother (approaches uniform).

    Pick ``tau`` on the order of the std of ``score_per_row`` to get
    meaningful blending without collapsing to uniform.
    """
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    s = np.asarray(score_per_row, dtype=np.float32).reshape(-1, 1)
    mu = np.asarray(bucket_centroids, dtype=np.float32).reshape(1, -1)
    # Logits: -(s - mu)^2 / tau^2. Subtract row-max for numerical
    # stability, then exponentiate and renormalize.
    lg = -((s - mu) ** 2) / (float(tau) ** 2)
    lg = lg - lg.max(axis=1, keepdims=True)
    w = np.exp(lg)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_soft_routing(
    per_expert_val: dict, *, expert_names, weights: np.ndarray,
) -> np.ndarray:
    """Linear blend of per-expert calibrated probabilities.

    ``per_expert_val[name]`` is a ``[N_val]`` float array. ``weights``
    is the ``[N_val, K]`` matrix from one of the helpers above (rows
    sum to 1.0). Returns the row-mixed probability vector.
    """
    if int(weights.shape[1]) != len(expert_names):
        raise ValueError(
            f"weights have {weights.shape[1]} cols but {len(expert_names)} "
            "expert names provided"
        )
    N = int(weights.shape[0])
    out = np.zeros(N, dtype=np.float32)
    for k, name in enumerate(expert_names):
        out += weights[:, k] * np.asarray(per_expert_val[name], dtype=np.float32)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


__all__ = [
    "RichMLPConfig",
    "train_rich_mlp",
    "predict_rich_mlp",
    "soft_routing_weights_categorical",
    "soft_routing_weights_kernel",
    "apply_soft_routing",
]
