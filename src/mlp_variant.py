"""Shared experimental GLU-MLP trainer for diversity probes.

Production members 7/8 live in :mod:`src.mlp_member`, which is hardened
for the pure-NumPy inference path of the submitted bundle. This module is
the *experimental* twin used by probe notebooks where the inference path
stays in torch and the trainer needs more flexibility than the production
one allows:

* :mod:`notebooks.loss_diversity_probe` -- one MLP per training objective
  (NLL / Brier / focal / label-smooth / ranking / distillation / cold
  reweighting / residual specialist).
* :mod:`notebooks.moe_poc` -- one MLP per item-type "expert", upweighted
  by item type to test the mixture-of-experts hypothesis on the M8
  architecture.

The architecture is byte-for-byte the production Member 8 design
(subject embedding | item embedding | dense metadata -> two GLU layers ->
linear head). Only the **loss** and the **sample weighting** are exposed
as knobs, so any diversity gain measured downstream is attributable to
the training signal change, not to an architectural difference.

Memory contract
---------------
Item embeddings are passed as a ``[n_unique_items, D]`` torch tensor +
per-row ``row_to_uniq`` int64 array, NOT a dense ``[N, D]`` materialization.
Each minibatch gathers from the unique table on the device. This is the
same trick :mod:`src.mlp_member` uses to fit 5M rows x 4096-D embeddings
in 24 GB without OOM.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Architecture: subject embedding | item embedding | dense metadata -> GLU MLP
# ---------------------------------------------------------------------------


def build_net(
    n_subjects: int,
    *,
    subj_emb_dim: int,
    item_dim: int,
    dense_dim: int,
    hid1: int,
    hid2: int,
    feat_dropout: float,
    seed: int,
    device,
):
    """Construct the M8-style GLU-MLP and move it to ``device``.

    Any input channel may be absent: pass ``dense_dim=0`` for an
    embedding-only run (matches the M8 ablation knob in the probe
    notebooks). The subject channel is always present -- pass a small
    ``subj_emb_dim`` (e.g. ``1``) to neutralize it.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    in_dim = int(subj_emb_dim) + int(item_dim) + int(dense_dim)
    use_dense = int(dense_dim) > 0

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.subj = nn.Embedding(int(n_subjects) + 1, int(subj_emb_dim))
            nn.init.normal_(self.subj.weight, std=0.05)
            self.l1v = nn.Linear(in_dim, hid1)
            self.l1g = nn.Linear(in_dim, hid1)
            self.l2v = nn.Linear(hid1, hid2)
            self.l2g = nn.Linear(hid1, hid2)
            self.head = nn.Linear(hid2, 1)
            self.drop = nn.Dropout(float(feat_dropout))

        def forward(self, sid, iemb, dz):
            parts = [self.subj(sid), iemb]
            if use_dense and dz is not None:
                parts.append(dz)
            x = torch.cat(parts, dim=1)
            x = self.drop(x)
            h1 = self.l1v(x) * torch.sigmoid(self.l1g(x))
            h2 = self.l2v(h1) * torch.sigmoid(self.l2g(h1))
            return self.head(h2).reshape(-1)

    return _Net().to(device)


# ---------------------------------------------------------------------------
# Objective dispatch (everything except `ranking`; that one needs a different
# data path in the train/eval loops)
# ---------------------------------------------------------------------------


def objective_loss(
    logits,
    target,
    weights,
    *,
    objective: str,
    focal_gamma: float,
    label_smooth_eps: float,
):
    """Per-objective scalar loss. ``target`` may be soft (distillation)."""
    import torch
    import torch.nn.functional as F

    if objective in ("bce", "class_weighted", "cold", "specialist", "distill"):
        eps = float(label_smooth_eps) if objective == "label_smooth" else 0.0
        t = target * (1.0 - eps) + 0.5 * eps
        per = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    elif objective == "label_smooth":
        eps = float(label_smooth_eps)
        t = target * (1.0 - eps) + 0.5 * eps
        per = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    elif objective == "brier":
        p = torch.sigmoid(logits)
        per = (p - target) ** 2
    elif objective == "focal":
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = target * p + (1.0 - target) * (1.0 - p)
        per = (1.0 - pt).clamp(min=0.0, max=1.0) ** float(focal_gamma) * ce
    else:
        raise ValueError(f"unknown objective {objective!r}")

    if weights is not None:
        return (per * weights).sum() / weights.sum().clamp(min=1e-8)
    return per.mean()


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


def train_m8_variant(
    *,
    y,
    subj_ids,
    r2u,
    emb_tensor,
    n_subjects: int,
    device,
    dense_X=None,
    objective: str = "bce",
    focal_gamma: float = 2.0,
    label_smooth_eps: float = 0.0,
    sample_weights=None,
    soft_targets=None,
    ranking: bool = False,
    subj_emb_dim: int = 32,
    hid1: int = 256,
    hid2: int = 128,
    lr: float = 1e-3,
    wd: float = 1e-5,
    epochs: int = 20,
    batch_size: int = 16384,
    val_fraction: float = 0.10,
    patience: int = 4,
    feat_dropout: float = 0.10,
    seed: int = 0,
    show_progress: bool = True,
):
    """Train one objective/weight variant of the M8 GLU-MLP.

    Returns the trained ``torch.nn.Module`` in eval mode on ``device``.
    The caller scores it via :func:`predict_probs` (chunked to keep
    memory bounded).

    Item-grouped internal early-stopping split (val_fraction of unique
    item groups held out) is the same regime used by every probe
    notebook so the early-stop signal is item-cold w.r.t. the per-fold
    training set, just like the OOF folds themselves.
    """
    import torch

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    N = int(y.shape[0])
    dense_dim = 0 if dense_X is None else int(np.asarray(dense_X).shape[1])
    rng = np.random.default_rng(int(seed))

    # Item-grouped internal val split (early stopping on held-out items).
    groups = np.asarray(r2u, dtype=np.int64).reshape(-1)
    uniq_g = np.unique(groups)
    n_val_g = max(1, int(round(float(val_fraction) * uniq_g.size)))
    val_g = set(uniq_g[rng.permutation(uniq_g.size)[:n_val_g]].tolist())
    val_mask = np.fromiter((g in val_g for g in groups), count=N, dtype=bool)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    if tr_idx.size == 0 or va_idx.size == 0:
        tr_idx = np.arange(N)
        va_idx = np.arange(N)

    net = build_net(
        n_subjects,
        subj_emb_dim=subj_emb_dim,
        item_dim=int(emb_tensor.shape[1]),
        dense_dim=dense_dim,
        hid1=hid1,
        hid2=hid2,
        feat_dropout=feat_dropout,
        seed=seed,
        device=device,
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    sid_all = torch.from_numpy(
        np.asarray(subj_ids, dtype=np.int64).reshape(-1)
    ).to(device)
    r2u_all = torch.from_numpy(groups).to(device)
    y_all = torch.from_numpy(y).to(device)
    dz_all = (
        torch.from_numpy(np.asarray(dense_X, dtype=np.float32)).to(device)
        if dense_dim > 0
        else None
    )
    tgt_all = (
        torch.from_numpy(
            np.asarray(soft_targets, dtype=np.float32).reshape(-1)
        ).to(device)
        if soft_targets is not None
        else y_all
    )
    w_all = (
        torch.from_numpy(
            np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        ).to(device)
        if sample_weights is not None
        else None
    )

    def _logits(idx_t):
        dz = dz_all[idx_t] if dz_all is not None else None
        return net(sid_all[idx_t], emb_tensor[r2u_all[idx_t]], dz)

    tr_idx_t = torch.from_numpy(tr_idx.astype(np.int64)).to(device)
    va_idx_t = torch.from_numpy(va_idx.astype(np.int64)).to(device)

    n_steps = max(1, int(math.ceil(tr_idx.size / batch_size)))
    total_steps = n_steps * int(epochs)
    warmup = n_steps * 2

    def _lr_at(step):
        if step < warmup and warmup > 0:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _val_metric():
        net.eval()
        with torch.no_grad():
            if ranking:
                lg = []
                for bs in range(0, va_idx.size, batch_size):
                    b = va_idx_t[bs : bs + batch_size]
                    lg.append(_logits(b))
                lg = torch.cat(lg)
                yv = y_all[va_idx_t]
                pos = lg[yv >= 0.5]
                neg = lg[yv < 0.5]
                m = int(min(pos.numel(), neg.numel()))
                if m < 8:
                    return float("inf")
                gp = torch.Generator(device=device).manual_seed(0)
                pi = torch.randperm(pos.numel(), generator=gp, device=device)[:m]
                ni = torch.randperm(neg.numel(), generator=gp, device=device)[:m]
                return float(
                    torch.nn.functional.softplus(-(pos[pi] - neg[ni])).mean()
                )
            vs, vn = 0.0, 0
            for bs in range(0, va_idx.size, batch_size):
                b = va_idx_t[bs : bs + batch_size]
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    _logits(b), y_all[b], reduction="sum"
                )
                vs += float(per)
                vn += int(b.shape[0])
            return vs / max(1, vn)

    best_val = float("inf")
    best_state = None
    bad = 0
    step = 0
    for ep in range(int(epochs)):
        net.train()
        perm = torch.randperm(tr_idx.size, device=device)
        tr_shuf = tr_idx_t[perm]
        for bs in range(0, tr_idx.size, batch_size):
            for g in opt.param_groups:
                g["lr"] = _lr_at(step)
            b = tr_shuf[bs : bs + batch_size]
            logits = _logits(b)
            if ranking:
                yb = y_all[b]
                pos = logits[yb >= 0.5]
                neg = logits[yb < 0.5]
                m = int(min(pos.numel(), neg.numel()))
                if m >= 1:
                    pi = torch.randperm(pos.numel(), device=device)[:m]
                    ni = torch.randperm(neg.numel(), device=device)[:m]
                    loss = torch.nn.functional.softplus(
                        -(pos[pi] - neg[ni])
                    ).mean()
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, yb
                    )
            else:
                loss = objective_loss(
                    logits,
                    tgt_all[b],
                    (w_all[b] if w_all is not None else None),
                    objective=objective,
                    focal_gamma=focal_gamma,
                    label_smooth_eps=label_smooth_eps,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        vloss = _val_metric()
        if show_progress:
            print(f"    epoch {ep + 1}/{epochs}  val={vloss:.5f}")
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_probs(
    net,
    subj_ids,
    r2u,
    emb_tensor,
    device,
    chunk: int = 131_072,
    dense_X=None,
) -> np.ndarray:
    """Chunked forward -> ``[N]`` float32 probabilities (sigmoid of logit)."""
    import torch

    sid = np.asarray(subj_ids, dtype=np.int64).reshape(-1)
    r2 = np.asarray(r2u, dtype=np.int64).reshape(-1)
    dz = None if dense_X is None else np.asarray(dense_X, dtype=np.float32)
    n = int(sid.shape[0])
    out = np.empty(n, dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sid_t = torch.from_numpy(sid[s:e]).to(device)
            r2_t = torch.from_numpy(r2[s:e]).to(device)
            dz_t = (
                torch.from_numpy(dz[s:e]).to(device) if dz is not None else None
            )
            logits = net(sid_t, emb_tensor[r2_t], dz_t)
            out[s:e] = (
                torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            )
    return np.clip(out, 1e-6, 1.0 - 1e-6)


__all__ = [
    "build_net",
    "objective_loss",
    "predict_probs",
    "train_m8_variant",
]
