"""Model definitions for the k-factor / neural-IRT ablation.

The original three model variants share the same backbone:

    logit(eta) = mu + alpha_s + beta_bc - d_i + <u_s, v_i> / sqrt(k)

where

- ``mu`` is a global bias,
- ``alpha_s`` is a learned subject intercept,
- ``beta_bc`` is a learned benchmark-condition intercept,
- ``d_i`` and ``v_i`` are predicted from the item embedding,
- ``u_s`` is a learned subject ability vector.

The three baseline variants are:

- ``KFactorModel``               -- exactly the factor model above.
- ``KFactorMLPResidual``         -- factor model + a small dense MLP residual.
- ``KFactorGatedMLPResidual``    -- factor model + a SwiGLU-style gated MLP.

This module also adds three Item-IRT variants:

- ``IRTItemKFactor``             -- parallel IRT channel:
                                    ``logit = alpha(item) * (theta_subj - beta(item))
                                              + mu + beta_bc``
- ``IRTItemKFactorMLP``          -- IRT channel + dense MLP residual.
- ``IRTItemKFactorGatedMLP``     -- IRT channel + gated MLP residual.

The IRT heads live *outside* the MLP (the multiplicative subject-item
interaction is exactly what classical IRT exists to express), while
pool features and cluster embeddings are fed *into* the MLP (noisy scalars
whose interactions with the embedding are not specified a priori).

Residuals are scaled by a (configurable, optionally trainable) ``lambda_resid``
so the residual cannot dominate the factor / IRT channel early in training.

We never fine-tune the encoder here: item / subject embeddings are pre-computed
and looked up by integer id at training time. A TODO hook is left in place for
later LoRA-based encoder fine-tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyperparameters shared by all model variants.

    The new IRT / pool / cluster fields default to "off" so existing
    checkpoints and existing variants behave identically to before this
    change unless explicitly enabled.
    """

    k: int = 16                          # factor dimension
    item_embed_dim: int = 768            # raw transformer embedding dim
    item_map_hidden_dim: int = 512       # hidden dim of the item-parameter MLP
    residual_hidden_dim: int = 256       # hidden dim of the residual MLP
    dropout: float = 0.1
    n_subjects: int = 1                  # populated at fit time
    n_benchmark_conditions: int = 1      # populated at fit time
    use_subject_text_embedding: bool = False
    subject_embed_dim: int = 0
    lambda_resid_init: float = 0.1
    lambda_resid_trainable: bool = True

    # --- pool features (z-scored hand-engineered scalars, fed into MLP) ---
    use_pool_features: bool = False
    pool_feature_dim: int = 0

    # --- cluster embedding (k-means cluster id, fed into MLP) ---
    use_cluster_features: bool = False
    n_clusters: int = 0
    cluster_embed_dim: int = 0

    # --- IRT heads regularization weights (used by trainer, not in forward) ---
    irt_lambda_beta: float = 1.0e-4
    irt_lambda_alpha: float = 1.0e-4

    @property
    def use_subject_embed_features(self) -> bool:
        return bool(self.use_subject_text_embedding and self.subject_embed_dim > 0)

    @property
    def effective_pool_dim(self) -> int:
        return int(self.pool_feature_dim) if self.use_pool_features else 0

    @property
    def effective_cluster_emb_dim(self) -> int:
        return int(self.cluster_embed_dim) if self.use_cluster_features else 0

    @property
    def has_cluster_embedding(self) -> bool:
        return (
            self.use_cluster_features
            and self.n_clusters > 0
            and self.cluster_embed_dim > 0
        )


# ---------------------------------------------------------------------------
# Item parameter heads
# ---------------------------------------------------------------------------


class ItemParameterMap(nn.Module):
    """Maps a raw item embedding to (item factor vector, difficulty scalar).

    Architecture: LN -> Linear -> SiLU -> Dropout -> Linear -> split into
    [k] item factor + [1] difficulty.
    """

    def __init__(
        self, item_embed_dim: int, k: int, hidden: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(item_embed_dim)
        self.fc1 = nn.Linear(item_embed_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.head_factor = nn.Linear(hidden, k)
        self.head_diff = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head_factor.bias)
        nn.init.zeros_(self.head_diff.bias)

    def forward(self, item_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(item_emb)
        h = self.fc1(h)
        h = F.silu(h)
        h = self.drop(h)
        v = self.head_factor(h)
        d = self.head_diff(h).squeeze(-1)
        return v, d


class ItemIRTHeads(nn.Module):
    """Predicts ``(beta_i, alpha_i)`` from the item embedding.

    - ``beta_i``  is a free scalar (item difficulty).
    - ``alpha_i`` is positive via softplus, initialized so the pre-softplus
      logit is ``softplus(0.54) ~= 1.0`` so alpha starts at ~1.0 (the
      classical 1PL initialization). The bias of the last linear is zero;
      we add the +0.54 in forward so hidden weights init small but alpha
      starts at 1.

    The IRT logit is then ``alpha_i * (theta_subj - beta_i)``.
    """

    def __init__(self, item_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.beta_head = nn.Sequential(
            nn.LayerNorm(item_dim),
            nn.Linear(item_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(item_dim),
            nn.Linear(item_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.beta_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].bias)
        self._alpha_pre_bias: float = 0.54  # softplus(0.54) ~= 1.0

    def forward(
        self, item_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta_head(item_embed).squeeze(-1)
        alpha_pre = self.alpha_head(item_embed).squeeze(-1) + self._alpha_pre_bias
        alpha = F.softplus(alpha_pre)
        return beta, alpha


# ---------------------------------------------------------------------------
# Residual-feature builder (used by both KFactor* and IRTItemKFactor* MLP
# residuals)
# ---------------------------------------------------------------------------


def _residual_feature_dim_kfactor(cfg: ModelConfig) -> int:
    """Width of the residual-MLP input for the KFactor-family residuals."""
    base = (
        cfg.k                            # u_s
        + cfg.k                          # v_i
        + cfg.k                          # u_s * v_i
        + cfg.item_embed_dim
        + 1                              # beta_bc scalar
    )
    if cfg.use_subject_embed_features:
        base += cfg.subject_embed_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    return base


def _residual_feature_dim_irt_item(cfg: ModelConfig) -> int:
    """Width of the residual-MLP input for the IRTItemKFactor-family residuals.

    The IRT variants don't have the k-factor interaction; the residual MLP
    is allowed to see the IRT channel's components (theta, beta_i, alpha_i)
    plus the raw item embedding, optional subject embedding, pool features,
    and cluster embedding.
    """
    base = (
        cfg.item_embed_dim
        + 3                              # theta, beta_i, alpha_i scalars
    )
    if cfg.use_subject_embed_features:
        base += cfg.subject_embed_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    return base


def _maybe_append(parts: list[torch.Tensor], cfg: ModelConfig, *, pool_z, cluster_emb):
    if cfg.use_pool_features and pool_z is not None and pool_z.shape[-1] > 0:
        parts.append(pool_z)
    if cfg.has_cluster_embedding and cluster_emb is not None and cluster_emb.shape[-1] > 0:
        parts.append(cluster_emb)


def _build_residual_features_kfactor(
    cfg: ModelConfig,
    u_s: torch.Tensor,
    v_i: torch.Tensor,
    item_emb: torch.Tensor,
    bc_idx_embed: torch.Tensor,
    subject_emb: torch.Tensor | None,
    pool_z: torch.Tensor | None,
    cluster_emb: torch.Tensor | None,
) -> torch.Tensor:
    parts = [u_s, v_i, u_s * v_i, item_emb, bc_idx_embed]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(parts, cfg, pool_z=pool_z, cluster_emb=cluster_emb)
    return torch.cat(parts, dim=-1)


def _build_residual_features_irt(
    cfg: ModelConfig,
    theta: torch.Tensor,
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    item_emb: torch.Tensor,
    subject_emb: torch.Tensor | None,
    pool_z: torch.Tensor | None,
    cluster_emb: torch.Tensor | None,
) -> torch.Tensor:
    parts = [
        item_emb,
        theta.unsqueeze(-1),
        beta_i.unsqueeze(-1),
        alpha_i.unsqueeze(-1),
    ]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(parts, cfg, pool_z=pool_z, cluster_emb=cluster_emb)
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Base k-factor model (Model A)
# ---------------------------------------------------------------------------


class KFactorModel(nn.Module):
    """Pure k-factor / neural IRT.

    logit = mu + alpha_s + beta_bc - d_i + <u_s, v_i> / sqrt(k)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Embedding(cfg.n_subjects, 1)
        self.beta = nn.Embedding(cfg.n_benchmark_conditions, 1)
        self.u = nn.Embedding(cfg.n_subjects, cfg.k)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.normal_(self.u.weight, std=0.05)
        self.item_map = ItemParameterMap(
            item_embed_dim=cfg.item_embed_dim,
            k=cfg.k,
            hidden=cfg.item_map_hidden_dim,
            dropout=cfg.dropout,
        )
        # New variants may register a cluster embedding; the base model
        # doesn't use it but we keep the attribute so we have a single place
        # to look it up.
        self.cluster_embedding: nn.Embedding | None = None
        if cfg.has_cluster_embedding:
            self.cluster_embedding = nn.Embedding(
                cfg.n_clusters + 1,  # +1 for UNK at index 0
                cfg.cluster_embed_dim,
                padding_idx=0,
            )
            nn.init.normal_(self.cluster_embedding.weight, std=0.05)

    @property
    def has_residual(self) -> bool:
        return False

    @property
    def has_irt_heads(self) -> bool:
        return False

    def _cluster_emb(self, cluster_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self.cluster_embedding is None or cluster_ids is None:
            return None
        if cluster_ids.numel() == 0 or cluster_ids.dim() < 1:
            return None
        # Long ids -> embedding vectors.
        return self.cluster_embedding(cluster_ids.long())

    def factor_logit(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (eta_factor, u_s, v_i, item_emb)."""
        u_s = self.u(subject_idx)
        alpha_s = self.alpha(subject_idx).squeeze(-1)
        beta_bc = self.beta(bc_idx).squeeze(-1)
        v_i, d_i = self.item_map(item_emb)
        k = max(1, self.cfg.k)
        interaction = (u_s * v_i).sum(dim=-1) / math.sqrt(k)
        eta = self.mu + alpha_s + beta_bc - d_i + interaction
        return eta, u_s, v_i, item_emb

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eta, _, _, _ = self.factor_logit(subject_idx, bc_idx, item_emb)
        return eta

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-row additive components of the final logit.

        For the kfactor family we expose (factor, mlp). For IRT variants
        the override below expands this into (irt, offset, mlp). The trainer
        / diagnostic uses whatever keys are present, so adding more is safe.
        """
        eta, u_s, v_i, _ = self.factor_logit(subject_idx, bc_idx, item_emb)
        bsz = item_emb.shape[0]
        zero = torch.zeros(bsz, device=item_emb.device, dtype=eta.dtype)
        return {"factor": eta, "mlp": zero}


# ---------------------------------------------------------------------------
# Residual heads (Models B and C)
# ---------------------------------------------------------------------------


class DenseMLPResidual(nn.Module):
    """LN -> Linear -> SiLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.silu(h)
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


class GatedSwiGLUResidual(nn.Module):
    """SwiGLU-style gated MLP residual.

    h = SiLU(W_gate x) * (W_up x)
    r = W_down(h)
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gate = nn.Linear(in_dim, hidden)
        self.up = nn.Linear(in_dim, hidden)
        self.down = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        h = F.silu(self.gate(x)) * self.up(x)
        h = self.drop(h)
        return self.down(h).squeeze(-1)


# ---------------------------------------------------------------------------
# K-factor + residual (existing variants B and C)
# ---------------------------------------------------------------------------


class _ResidualKFactor(KFactorModel):
    """Common implementation for the MLP / gated-MLP residual models."""

    residual_cls: type[nn.Module] = DenseMLPResidual

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        in_dim = _residual_feature_dim_kfactor(cfg)
        self.residual = self.residual_cls(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lambda_resid = nn.Parameter(
            torch.tensor(float(cfg.lambda_resid_init)),
            requires_grad=bool(cfg.lambda_resid_trainable),
        )

    @property
    def has_residual(self) -> bool:
        return True

    def _residual_input(
        self,
        u_s: torch.Tensor,
        v_i: torch.Tensor,
        item_emb_out: torch.Tensor,
        beta_bc: torch.Tensor,
        subject_emb: torch.Tensor | None,
        pool_feats: torch.Tensor | None,
        cluster_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        cluster_emb = self._cluster_emb(cluster_ids)
        return _build_residual_features_kfactor(
            self.cfg,
            u_s,
            v_i,
            item_emb_out,
            beta_bc,
            subject_emb,
            pool_feats,
            cluster_emb,
        )

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        *,
        override_mlp_zero: bool = False,
    ) -> torch.Tensor:
        eta_factor, u_s, v_i, item_emb_out = self.factor_logit(
            subject_idx, bc_idx, item_emb
        )
        if override_mlp_zero:
            return eta_factor
        beta_bc = self.beta(bc_idx)
        x = self._residual_input(
            u_s, v_i, item_emb_out, beta_bc, subject_emb, pool_feats, cluster_ids
        )
        r = self.residual(x)
        return eta_factor + self.lambda_resid * r

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        eta_factor, u_s, v_i, item_emb_out = self.factor_logit(
            subject_idx, bc_idx, item_emb
        )
        beta_bc = self.beta(bc_idx)
        x = self._residual_input(
            u_s, v_i, item_emb_out, beta_bc, subject_emb, pool_feats, cluster_ids
        )
        r = self.lambda_resid * self.residual(x)
        return {"factor": eta_factor, "mlp": r}


class KFactorMLPResidual(_ResidualKFactor):
    residual_cls = DenseMLPResidual


class KFactorGatedMLPResidual(_ResidualKFactor):
    residual_cls = GatedSwiGLUResidual


# ---------------------------------------------------------------------------
# Item-IRT variants (new)
# ---------------------------------------------------------------------------


class IRTItemKFactor(nn.Module):
    """Pure Item-IRT model: IRT channel + offset.

    logit = mu + beta_bc + alpha(item) * (theta_subj - beta(item))

    ``theta_subj`` is a learned per-subject scalar (reusing the IRT name for
    the scalar subject ability, distinct from the multi-dimensional u_s in
    the kfactor family). Index 0 is always reserved for UNK.

    The IRT heads are predicted from the item embedding and live OUTSIDE the
    MLP -- the multiplicative subject-item interaction is exactly what
    classical IRT is for, and feeding alpha/beta into the MLP as plain
    features would discard that inductive bias.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.zeros(1))
        self.theta = nn.Embedding(cfg.n_subjects, 1)         # subject ability
        self.beta = nn.Embedding(cfg.n_benchmark_conditions, 1)
        nn.init.zeros_(self.theta.weight)
        nn.init.zeros_(self.beta.weight)
        self.irt_heads = ItemIRTHeads(
            item_dim=cfg.item_embed_dim,
            hidden=cfg.item_map_hidden_dim,
            dropout=cfg.dropout,
        )
        self.cluster_embedding: nn.Embedding | None = None
        if cfg.has_cluster_embedding:
            self.cluster_embedding = nn.Embedding(
                cfg.n_clusters + 1, cfg.cluster_embed_dim, padding_idx=0
            )
            nn.init.normal_(self.cluster_embedding.weight, std=0.05)

    @property
    def has_residual(self) -> bool:
        return False

    @property
    def has_irt_heads(self) -> bool:
        return True

    def _cluster_emb(self, cluster_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self.cluster_embedding is None or cluster_ids is None:
            return None
        if cluster_ids.numel() == 0 or cluster_ids.dim() < 1:
            return None
        return self.cluster_embedding(cluster_ids.long())

    def _irt_components(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        theta = self.theta(subject_idx).squeeze(-1)
        bc_off = self.beta(bc_idx).squeeze(-1)
        beta_i, alpha_i = self.irt_heads(item_emb)
        c_irt = alpha_i * (theta - beta_i)
        # broadcast the global bias across the batch
        c_offset = bc_off + self.mu
        return {
            "irt": c_irt,
            "offset": c_offset,
            "theta": theta,
            "beta_i": beta_i,
            "alpha_i": alpha_i,
        }

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta"] - beta_i)
        return c_irt + comps["offset"]

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        bsz = item_emb.shape[0]
        zero_mlp = torch.zeros(bsz, device=item_emb.device, dtype=comps["irt"].dtype)
        return {
            "irt": comps["irt"],
            "offset": comps["offset"],
            "mlp": zero_mlp,
            "theta": comps["theta"],
            "beta_i": comps["beta_i"],
            "alpha_i": comps["alpha_i"],
        }


class _ResidualIRTItem(IRTItemKFactor):
    """Common implementation for IRT-item + residual variants."""

    residual_cls: type[nn.Module] = DenseMLPResidual

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        in_dim = _residual_feature_dim_irt_item(cfg)
        self.residual = self.residual_cls(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lambda_resid = nn.Parameter(
            torch.tensor(float(cfg.lambda_resid_init)),
            requires_grad=bool(cfg.lambda_resid_trainable),
        )

    @property
    def has_residual(self) -> bool:
        return True

    def _residual_input(
        self,
        comps: dict[str, torch.Tensor],
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None,
        pool_feats: torch.Tensor | None,
        cluster_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        cluster_emb = self._cluster_emb(cluster_ids)
        return _build_residual_features_irt(
            self.cfg,
            theta=comps["theta"],
            beta_i=comps["beta_i"],
            alpha_i=comps["alpha_i"],
            item_emb=item_emb,
            subject_emb=subject_emb,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
        )

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
        override_mlp_zero: bool = False,
    ) -> torch.Tensor:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta"] - beta_i)
        c_off = comps["offset"]
        if override_mlp_zero:
            return c_irt + c_off
        x = self._residual_input(
            comps, item_emb, subject_emb, pool_feats, cluster_ids
        )
        r = self.lambda_resid * self.residual(x)
        return c_irt + c_off + r

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        x = self._residual_input(
            comps, item_emb, subject_emb, pool_feats, cluster_ids
        )
        r = self.lambda_resid * self.residual(x)
        return {
            "irt": comps["irt"],
            "offset": comps["offset"],
            "mlp": r,
            "theta": comps["theta"],
            "beta_i": comps["beta_i"],
            "alpha_i": comps["alpha_i"],
        }


class IRTItemKFactorMLP(_ResidualIRTItem):
    residual_cls = DenseMLPResidual


class IRTItemKFactorGatedMLP(_ResidualIRTItem):
    residual_cls = GatedSwiGLUResidual


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "kfactor": KFactorModel,
    "kfactor_mlp": KFactorMLPResidual,
    "kfactor_gated_mlp": KFactorGatedMLPResidual,
    "kfactor_irt_item": IRTItemKFactor,
    "kfactor_irt_item_mlp": IRTItemKFactorMLP,
    "kfactor_irt_item_gated_mlp": IRTItemKFactorGatedMLP,
}


def build_model(name: str, cfg: ModelConfig) -> nn.Module:
    """Instantiate a model variant by registry name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg)


def model_has_irt_heads(model: nn.Module) -> bool:
    """Return True iff the model exposes Item-IRT heads (beta_i, alpha_i)."""
    return bool(getattr(model, "has_irt_heads", False))


def irt_regularization(
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    *,
    lambda_beta: float,
    lambda_alpha: float,
) -> torch.Tensor:
    """Soft regularization on the Item-IRT heads.

    Without these the heads and the residual MLP collude in degenerate
    ways (beta explodes, MLP cancels it).
    """
    if lambda_beta == 0 and lambda_alpha == 0:
        return torch.zeros((), device=beta_i.device, dtype=beta_i.dtype)
    reg = torch.zeros((), device=beta_i.device, dtype=beta_i.dtype)
    if lambda_beta != 0:
        reg = reg + lambda_beta * (beta_i.float() ** 2).mean()
    if lambda_alpha != 0:
        reg = reg + lambda_alpha * (torch.log(alpha_i.float().clamp_min(1e-6)) ** 2).mean()
    return reg


# ---------------------------------------------------------------------------
# Indexer: maps subject_key / benchmark_condition_key to integer ids and the
# UNK rows that test-time `predict()` will fall back to.
# ---------------------------------------------------------------------------


@dataclass
class Indexer:
    """Bijective mapping ``{subject_key -> int}`` and ``{bc_key -> int}``.

    Index 0 is always reserved for UNK in each space.
    """

    subject_to_id: dict[str, int] = field(default_factory=dict)
    bc_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        subject_keys: Sequence[str],
        bc_keys: Sequence[str],
    ) -> "Indexer":
        sub = {"<unk>": 0}
        for k in subject_keys:
            if k not in sub:
                sub[k] = len(sub)
        bc = {"<unk>": 0}
        for k in bc_keys:
            if k not in bc:
                bc[k] = len(bc)
        return cls(subject_to_id=sub, bc_to_id=bc)

    def subject_id(self, key: str) -> int:
        return self.subject_to_id.get(key, 0)

    def bc_id(self, key: str) -> int:
        return self.bc_to_id.get(key, 0)

    @property
    def n_subjects(self) -> int:
        return len(self.subject_to_id)

    @property
    def n_bc(self) -> int:
        return len(self.bc_to_id)

    def to_dict(self) -> dict:
        return {
            "subject_to_id": self.subject_to_id,
            "bc_to_id": self.bc_to_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Indexer":
        return cls(
            subject_to_id=dict(d["subject_to_id"]),
            bc_to_id=dict(d["bc_to_id"]),
        )


# ---------------------------------------------------------------------------
# Lookup-style dataset (no transformer at training time)
# ---------------------------------------------------------------------------


class LookupDataset(torch.utils.data.Dataset):
    """Yields ``(subject_id, bc_id, item_emb, subject_emb, pool_feats,
    cluster_id, label)`` per row.

    Pool features and cluster ids are optional: if missing, zero-sized
    tensors are returned. The trainer + eval code always unpacks the
    7-tuple; downstream models silently ignore zero-sized channels.
    """

    def __init__(
        self,
        subject_ids: np.ndarray,
        bc_ids: np.ndarray,
        item_emb: np.ndarray,
        labels: np.ndarray,
        subject_emb: np.ndarray | None = None,
        pool_feats: np.ndarray | None = None,
        cluster_ids: np.ndarray | None = None,
    ):
        self.subject_ids = torch.from_numpy(np.asarray(subject_ids, dtype=np.int64))
        self.bc_ids = torch.from_numpy(np.asarray(bc_ids, dtype=np.int64))
        self.item_emb = torch.from_numpy(np.asarray(item_emb, dtype=np.float32))
        if subject_emb is None:
            self.subject_emb = torch.zeros((len(labels), 0), dtype=torch.float32)
        else:
            self.subject_emb = torch.from_numpy(
                np.asarray(subject_emb, dtype=np.float32)
            )
        if pool_feats is None:
            self.pool_feats = torch.zeros((len(labels), 0), dtype=torch.float32)
        else:
            self.pool_feats = torch.from_numpy(
                np.asarray(pool_feats, dtype=np.float32)
            )
        if cluster_ids is None:
            self.cluster_ids = torch.zeros(len(labels), dtype=torch.long)
        else:
            self.cluster_ids = torch.from_numpy(
                np.asarray(cluster_ids, dtype=np.int64)
            )
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.subject_ids[idx],
            self.bc_ids[idx],
            self.item_emb[idx],
            self.subject_emb[idx],
            self.pool_feats[idx],
            self.cluster_ids[idx],
            self.labels[idx],
        )


__all__ = [
    "DenseMLPResidual",
    "GatedSwiGLUResidual",
    "IRTItemKFactor",
    "IRTItemKFactorGatedMLP",
    "IRTItemKFactorMLP",
    "Indexer",
    "ItemIRTHeads",
    "ItemParameterMap",
    "KFactorGatedMLPResidual",
    "KFactorMLPResidual",
    "KFactorModel",
    "LookupDataset",
    "MODEL_REGISTRY",
    "ModelConfig",
    "build_model",
    "irt_regularization",
    "model_has_irt_heads",
]
