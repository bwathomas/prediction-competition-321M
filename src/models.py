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
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default cluster-embedding width applied by ``ModelConfig.__post_init__``
# when the cluster channel is enabled but the embedding dim is missing or
# non-positive. Matches ``configs/default.yaml`` so checkpoints loaded with
# the old (cluster_embed_dim=0) dataclass default no longer silently
# degrade into a zero-width ``nn.Embedding(n_clusters, 0)`` no-op.
DEFAULT_CLUSTER_EMBED_DIM: int = 16


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
    # --- Pattern-2 subject-text -> subject-id soft tying ---
    # When ``use_subject_tie`` is on AND a raw subject text embedding is
    # available (``use_subject_text_embedding`` + ``subject_embed_dim > 0``),
    # we project the raw text embedding through a learned LayerNorm+Linear
    # into ``subject_proj_dim`` channels and (a) feed the *projected* vector
    # into the residual MLP in place of the raw text embedding and (b) tie
    # the projected text embedding to the model's k-dim subject id embedding
    # (``self.u`` for KFactor-family models, ``self.theta_vec`` for the
    # hierarchical-MIRT variant). The trainer adds
    # ``lambda_tie * MSE(id_emb, proj_text)`` to the BCE loss; the per-step
    # contribution is computed by :func:`compute_subject_tie_loss`.
    #
    # Backward compat: when ``use_subject_tie=False`` (the default) the
    # residual-input width and weight layout are bit-identical to the
    # pre-Pattern-2 code path -- so existing checkpoints load unchanged.
    use_subject_tie: bool = False
    subject_proj_dim: int = 0
    lambda_tie: float = 0.0
    tie_direction: str = "id_toward_text"
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

    # --- LLM-as-judge features (fed into the residual MLP) ---
    # When True, the residual MLP receives [lp_yes, lp_no, lp_diff, p_yes]
    # as additional inputs concatenated with the item embedding / pool / etc.
    # The head learns *where to trust the judge* rather than applying a
    # global blend weight. See `src.judge` for how these are produced.
    use_judge_features: bool = False
    judge_feature_dim: int = 4

    # --- Nearest-neighbor features (fed into the residual MLP) ---
    # When True the residual MLP receives an additional 8-scalar feature
    # vector summarizing the subject's performance on the top-K nearest
    # training items. ``nn_feature_dim`` is locked to the schema in
    # ``src.nn_features``; the field is exposed for introspection only and
    # not user-tunable. See ``src.nn_features._aggregate_nn_features`` for
    # the exact ordering.
    use_nn_features: bool = False
    nn_feature_dim: int = 8

    def __post_init__(self) -> None:
        # Repair the cluster channel when the caller (or an old checkpoint
        # whose ``model_cfg`` dict predates the ``cluster_embed_dim`` field)
        # left the dataclass in the contradictory ``use_cluster_features=True,
        # n_clusters > 0, cluster_embed_dim == 0`` state.
        #
        # Without this, ``has_cluster_embedding`` evaluates to False, the
        # cluster ``nn.Embedding`` is never built, and the cluster ids flow
        # through forward() contributing nothing -- a silent no-op that
        # shows up as ``cluster_embed_dim: 0`` in the printed config while
        # ``use_cluster_features: True`` and ``n_clusters: 64``. We restore
        # the configured default (matches ``configs/default.yaml``) and
        # warn loudly so the run still trains a real cluster channel.
        if (
            self.use_cluster_features
            and int(self.n_clusters) > 0
            and int(self.cluster_embed_dim) <= 0
        ):
            warnings.warn(
                "ModelConfig: use_cluster_features=True with n_clusters="
                f"{int(self.n_clusters)} but cluster_embed_dim="
                f"{int(self.cluster_embed_dim)}; coercing cluster_embed_dim "
                f"to {DEFAULT_CLUSTER_EMBED_DIM} so the cluster channel is "
                "actually live (was a silent no-op before).",
                stacklevel=2,
            )
            self.cluster_embed_dim = int(DEFAULT_CLUSTER_EMBED_DIM)

    @property
    def use_subject_embed_features(self) -> bool:
        return bool(self.use_subject_text_embedding and self.subject_embed_dim > 0)

    @property
    def use_pattern2_subject_tie(self) -> bool:
        """Pattern-2 is *on* iff text embeddings are present AND a tie is requested."""
        return bool(
            self.use_subject_embed_features
            and self.use_subject_tie
            and int(self.subject_proj_dim) > 0
        )

    @property
    def effective_subject_feature_dim(self) -> int:
        """Width of the subject-side feature actually appended to the residual MLP.

        - Pattern 2 ON   -> low-dim projected embedding (``subject_proj_dim``).
        - Pattern 2 OFF  -> raw text embedding (``subject_embed_dim``) [legacy].
        - Subject text   -> 0 when no subject text embedding is configured.
          embedding OFF
        """
        if not self.use_subject_embed_features:
            return 0
        if self.use_pattern2_subject_tie:
            return int(self.subject_proj_dim)
        return int(self.subject_embed_dim)

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

    @property
    def effective_judge_dim(self) -> int:
        return int(self.judge_feature_dim) if self.use_judge_features else 0

    @property
    def effective_nn_dim(self) -> int:
        return int(self.nn_feature_dim) if self.use_nn_features else 0


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
# Pattern-2 subject-text projector + tying helpers
#
# The projector turns a raw subject text embedding (``subject_embed_dim`` dims,
# typically 1024-8192 depending on the encoder) into a low-dim vector aligned
# with the model's k-dim subject id embedding. The same projection is used for
# (a) feeding the residual MLP a cheap subject-text channel and (b) the soft
# tying loss that pulls the id embedding toward the (semantically grounded)
# text-derived embedding -- the marginal generalization across subjects this
# is meant to buy us comes from forcing the id table to inherit some of the
# text-derived structure rather than learning a free embedding per subject.
# ---------------------------------------------------------------------------


class SubjectTextProjector(nn.Module):
    """LayerNorm -> Linear projection from ``in_dim`` to ``out_dim``.

    Two-layer projector was tried in early prototyping and it overfit the
    sparse subject space; a single linear with LN is enough to align the
    geometry, and the soft tying loss keeps the id table close to the
    projected text geometry without collapsing it.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if int(in_dim) <= 0 or int(out_dim) <= 0:
            raise ValueError(
                f"SubjectTextProjector: in_dim={in_dim} and out_dim={out_dim} "
                "must both be positive."
            )
        self.norm = nn.LayerNorm(int(in_dim))
        self.proj = nn.Linear(int(in_dim), int(out_dim))
        # Kaiming-ish small init; the tying loss does the rest of the work.
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, subject_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(subject_emb))


def _register_subject_text_proj(model: nn.Module, cfg: ModelConfig) -> None:
    """Attach (or skip) ``self.subject_text_proj`` based on ``cfg``.

    Idempotent: callers should call this exactly once from each top-level
    model's ``__init__``. Subclasses inherit the attribute through normal
    Python attribute lookup so we don't need to call it again.
    """
    if cfg.use_pattern2_subject_tie:
        model.subject_text_proj = SubjectTextProjector(
            in_dim=int(cfg.subject_embed_dim),
            out_dim=int(cfg.subject_proj_dim),
        )
    else:
        # Always set the attribute (``None`` sentinel) so downstream
        # ``getattr(model, "subject_text_proj", None)`` is unambiguous.
        model.subject_text_proj = None


def _maybe_project_subject_emb(
    model: nn.Module, subject_emb: torch.Tensor | None
) -> torch.Tensor | None:
    """Return projected subject emb when Pattern-2 is active, else passthrough."""
    proj = getattr(model, "subject_text_proj", None)
    if proj is None or subject_emb is None or subject_emb.shape[-1] == 0:
        return subject_emb
    return proj(subject_emb)


def _subject_id_embedding_table(model: nn.Module) -> nn.Embedding | None:
    """Pick the k-dim subject-id embedding table to tie against.

    KFactor-family models use ``self.u``; HierarchicalMIRT uses
    ``self.theta_vec``. Models without a multi-dim subject table (the
    scalar-only IRT variants) return ``None`` -- there is nothing to tie.
    """
    for name in ("u", "theta_vec"):
        table = getattr(model, name, None)
        if isinstance(table, nn.Embedding) and table.embedding_dim > 0:
            return table
    return None


def compute_subject_tie_loss(
    model: nn.Module,
    subject_idx: torch.Tensor,
    subject_emb: torch.Tensor | None,
) -> torch.Tensor:
    """Pattern-2 soft tying loss: ``MSE(subject_id_emb, proj(text_emb))``.

    Returns a zero scalar when tying is disabled, the projector is not
    registered, or the model has no multi-dim subject id embedding (e.g.
    the scalar-only IRT variants). The trainer multiplies this by
    ``cfg.lambda_tie`` before adding it to the BCE loss.

    ``tie_direction``:
        * ``"id_toward_text"`` (default): gradient flows into ``id_emb``
          only; text projection is detached. Use this when you trust the
          text encoder more than the id table (the usual case for sparse
          subjects).
        * ``"text_toward_id"``: gradient flows into the projector only.
        * ``"both"`` / ``"bidirectional"``: gradient flows both ways.
    """
    cfg: ModelConfig = model.cfg
    proj = getattr(model, "subject_text_proj", None)
    if (
        proj is None
        or subject_emb is None
        or subject_emb.shape[-1] == 0
        or not bool(cfg.use_subject_tie)
    ):
        return torch.zeros(
            (), device=subject_idx.device, dtype=torch.float32
        )

    id_table = _subject_id_embedding_table(model)
    if id_table is None:
        return torch.zeros(
            (), device=subject_idx.device, dtype=torch.float32
        )

    id_emb = id_table(subject_idx)
    proj_text = proj(subject_emb)

    if id_emb.shape[-1] != proj_text.shape[-1]:
        raise ValueError(
            "compute_subject_tie_loss: subject id-emb dim "
            f"{id_emb.shape[-1]} != projected text dim {proj_text.shape[-1]}. "
            "Set ModelConfig.subject_proj_dim == k for KFactor-family models "
            "(or == k for HierarchicalMIRT)."
        )

    direction = (cfg.tie_direction or "id_toward_text").lower()
    if direction == "id_toward_text":
        return F.mse_loss(id_emb, proj_text.detach())
    if direction == "text_toward_id":
        return F.mse_loss(proj_text, id_emb.detach())
    if direction in {"both", "bidirectional", "sym", "symmetric"}:
        return F.mse_loss(id_emb, proj_text)
    raise ValueError(
        "compute_subject_tie_loss: unknown tie_direction "
        f"{cfg.tie_direction!r}; expected one of "
        "'id_toward_text', 'text_toward_id', 'both'."
    )


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
        base += cfg.effective_subject_feature_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    if cfg.use_judge_features:
        base += cfg.effective_judge_dim
    if cfg.use_nn_features:
        base += cfg.effective_nn_dim
    return base


def _residual_feature_dim_irt_item(cfg: ModelConfig) -> int:
    """Width of the residual-MLP input for the IRTItemKFactor-family residuals.

    The IRT variants don't have the k-factor interaction; the residual MLP
    is allowed to see the IRT channel's components (theta, beta_i, alpha_i)
    plus the raw item embedding, optional subject embedding, pool features,
    cluster embedding, (optionally) judge features, and the 8-scalar NN
    feature vector when enabled.
    """
    base = (
        cfg.item_embed_dim
        + 3                              # theta, beta_i, alpha_i scalars
    )
    if cfg.use_subject_embed_features:
        base += cfg.effective_subject_feature_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    if cfg.use_judge_features:
        base += cfg.effective_judge_dim
    if cfg.use_nn_features:
        base += cfg.effective_nn_dim
    return base


def _maybe_append(
    parts: list[torch.Tensor],
    cfg: ModelConfig,
    *,
    pool_z,
    cluster_emb,
    judge_feats=None,
    nn_feats=None,
):
    if cfg.use_pool_features and pool_z is not None and pool_z.shape[-1] > 0:
        parts.append(pool_z)
    if cfg.has_cluster_embedding and cluster_emb is not None and cluster_emb.shape[-1] > 0:
        parts.append(cluster_emb)
    if (
        cfg.use_judge_features
        and judge_feats is not None
        and judge_feats.shape[-1] > 0
    ):
        parts.append(judge_feats)
    if (
        cfg.use_nn_features
        and nn_feats is not None
        and nn_feats.shape[-1] > 0
    ):
        parts.append(nn_feats)


def _build_residual_features_kfactor(
    cfg: ModelConfig,
    u_s: torch.Tensor,
    v_i: torch.Tensor,
    item_emb: torch.Tensor,
    bc_idx_embed: torch.Tensor,
    subject_emb: torch.Tensor | None,
    pool_z: torch.Tensor | None,
    cluster_emb: torch.Tensor | None,
    judge_feats: torch.Tensor | None = None,
    nn_feats: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [u_s, v_i, u_s * v_i, item_emb, bc_idx_embed]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(
        parts,
        cfg,
        pool_z=pool_z,
        cluster_emb=cluster_emb,
        judge_feats=judge_feats,
        nn_feats=nn_feats,
    )
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
    judge_feats: torch.Tensor | None = None,
    nn_feats: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [
        item_emb,
        theta.unsqueeze(-1),
        beta_i.unsqueeze(-1),
        alpha_i.unsqueeze(-1),
    ]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(
        parts,
        cfg,
        pool_z=pool_z,
        cluster_emb=cluster_emb,
        judge_feats=judge_feats,
        nn_feats=nn_feats,
    )
    return torch.cat(parts, dim=-1)


def _residual_feature_dim_hybrid_irt_kfactor(cfg: ModelConfig) -> int:
    """Width of the residual-MLP input for the hybrid IRT + k-factor variant.

    The hybrid variant exposes the IRT scalars (theta, beta_i, alpha_i) AND
    the multidimensional kfactor components (u_s, v_i, u_s * v_i) plus a
    scalar for the raw u_s . v_i interaction. This is a superset of both
    the IRT-only and kfactor-only residual inputs.
    """
    base = (
        cfg.item_embed_dim
        + 3                              # theta, beta_i, alpha_i scalars
        + cfg.k                          # u_s
        + cfg.k                          # v_i
        + cfg.k                          # u_s * v_i
        + 1                              # raw factor interaction scalar
    )
    if cfg.use_subject_embed_features:
        base += cfg.effective_subject_feature_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    if cfg.use_judge_features:
        base += cfg.effective_judge_dim
    if cfg.use_nn_features:
        base += cfg.effective_nn_dim
    return base


def _build_residual_features_hybrid_irt_kfactor(
    cfg: ModelConfig,
    *,
    theta: torch.Tensor,
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    u_s: torch.Tensor,
    v_i: torch.Tensor,
    raw_factor: torch.Tensor,
    item_emb: torch.Tensor,
    subject_emb: torch.Tensor | None,
    pool_z: torch.Tensor | None,
    cluster_emb: torch.Tensor | None,
    judge_feats: torch.Tensor | None = None,
    nn_feats: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [
        item_emb,
        theta.unsqueeze(-1),
        beta_i.unsqueeze(-1),
        alpha_i.unsqueeze(-1),
        u_s,
        v_i,
        u_s * v_i,
        raw_factor.unsqueeze(-1),
    ]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(
        parts,
        cfg,
        pool_z=pool_z,
        cluster_emb=cluster_emb,
        judge_feats=judge_feats,
        nn_feats=nn_feats,
    )
    return torch.cat(parts, dim=-1)


def _maybe_zero_judge(
    cfg: ModelConfig,
    judge_feats: torch.Tensor | None,
    *,
    force_judge_zero: bool,
) -> torch.Tensor | None:
    """Helper for ablation: return zeros (same shape) when caller requests it."""
    if not cfg.use_judge_features or judge_feats is None or judge_feats.shape[-1] == 0:
        return judge_feats
    if force_judge_zero:
        return torch.zeros_like(judge_feats)
    return judge_feats


def _maybe_zero_nn(
    cfg: ModelConfig,
    nn_feats: torch.Tensor | None,
    *,
    force_nn_zero: bool,
) -> torch.Tensor | None:
    """Helper for ablation: return zeros (same shape) when caller requests it."""
    if not cfg.use_nn_features or nn_feats is None or nn_feats.shape[-1] == 0:
        return nn_feats
    if force_nn_zero:
        return torch.zeros_like(nn_feats)
    return nn_feats


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
        _register_subject_text_proj(self, cfg)

    @property
    def has_residual(self) -> bool:
        return False

    @property
    def has_irt_heads(self) -> bool:
        return False

    @property
    def has_judge_features(self) -> bool:
        return bool(self.cfg.use_judge_features and self.cfg.effective_judge_dim > 0)

    @property
    def has_nn_features(self) -> bool:
        return bool(self.cfg.use_nn_features and self.cfg.effective_nn_dim > 0)

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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-row additive components of the final logit.

        For the kfactor family we expose (factor, mlp). For IRT variants
        the override below expands this into (irt, offset, mlp). The trainer
        / diagnostic uses whatever keys are present, so adding more is safe.

        NN features ride inside the residual ``mlp`` component (they are an
        input to the MLP, not a separate additive logit). The diagnostic in
        cell 14b ablates them via ``force_nn_zero=True`` rather than by
        splitting out an independent component term.
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
        judge_feats: torch.Tensor | None,
        nn_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
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
            judge_feats=judge_feats,
            nn_feats=nn_feats,
        )

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        override_mlp_zero: bool = False,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
    ) -> torch.Tensor:
        eta_factor, u_s, v_i, item_emb_out = self.factor_logit(
            subject_idx, bc_idx, item_emb
        )
        if override_mlp_zero:
            return eta_factor
        beta_bc = self.beta(bc_idx)
        jf = _maybe_zero_judge(self.cfg, judge_feats, force_judge_zero=force_judge_zero)
        nf = _maybe_zero_nn(self.cfg, nn_feats, force_nn_zero=force_nn_zero)
        x = self._residual_input(
            u_s, v_i, item_emb_out, beta_bc, subject_emb, pool_feats, cluster_ids, jf, nf
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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        eta_factor, u_s, v_i, item_emb_out = self.factor_logit(
            subject_idx, bc_idx, item_emb
        )
        beta_bc = self.beta(bc_idx)
        x = self._residual_input(
            u_s, v_i, item_emb_out, beta_bc, subject_emb, pool_feats, cluster_ids, judge_feats, nn_feats
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
        _register_subject_text_proj(self, cfg)

    @property
    def has_residual(self) -> bool:
        return False

    @property
    def has_irt_heads(self) -> bool:
        return True

    @property
    def has_judge_features(self) -> bool:
        return bool(self.cfg.use_judge_features and self.cfg.effective_judge_dim > 0)

    @property
    def has_nn_features(self) -> bool:
        return bool(self.cfg.use_nn_features and self.cfg.effective_nn_dim > 0)

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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
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
        judge_feats: torch.Tensor | None,
        nn_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
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
            judge_feats=judge_feats,
            nn_feats=nn_feats,
        )

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
        override_mlp_zero: bool = False,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
    ) -> torch.Tensor:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta"] - beta_i)
        c_off = comps["offset"]
        if override_mlp_zero:
            return c_irt + c_off
        jf = _maybe_zero_judge(self.cfg, judge_feats, force_judge_zero=force_judge_zero)
        nf = _maybe_zero_nn(self.cfg, nn_feats, force_nn_zero=force_nn_zero)
        x = self._residual_input(
            comps, item_emb, subject_emb, pool_feats, cluster_ids, jf, nf
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
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._irt_components(subject_idx, bc_idx, item_emb)
        x = self._residual_input(
            comps, item_emb, subject_emb, pool_feats, cluster_ids, judge_feats, nn_feats
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
# Hybrid IRT + k-factor + gated residual MLP (new)
# ---------------------------------------------------------------------------


class HybridIRTItemKFactorGatedMLP(nn.Module):
    """Hybrid Item-IRT + multidimensional k-factor + gated residual MLP.

    The final logit is:

        logit = mu
              + beta_bc
              + alpha_i * (theta_s - beta_i)
              + rho_factor * ((u_s . v_i) / sqrt(k))
              + lambda_resid * gated_residual(F)

    where:
      - ``theta_s`` is a learned per-subject scalar ability (IRT),
      - ``beta_i, alpha_i`` come from the item-IRT heads (IRT difficulty /
        discrimination predicted from the item embedding),
      - ``u_s`` is a learned ``k``-dim subject vector,
      - ``v_i`` is a ``k``-dim item factor predicted from the item embedding,
      - ``rho_factor`` is a trainable scalar mixing the k-factor channel.

    This deliberately does *not* re-use the full ``KFactorModel`` logit,
    because that would double-count the global ``mu``, the benchmark-condition
    offset ``beta_bc``, and would introduce a second scalar item difficulty
    (``-d_i`` from ``ItemParameterMap``). The IRT ``beta_i`` is already the
    scalar item difficulty; the only thing missing from the IRT channel is
    the multidimensional subject-item interaction ``u_s . v_i / sqrt(k)``,
    which is precisely what the factor branch contributes here.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.zeros(1))
        self.theta = nn.Embedding(cfg.n_subjects, 1)          # subject ability
        self.beta = nn.Embedding(cfg.n_benchmark_conditions, 1)
        nn.init.zeros_(self.theta.weight)
        nn.init.zeros_(self.beta.weight)

        self.irt_heads = ItemIRTHeads(
            item_dim=cfg.item_embed_dim,
            hidden=cfg.item_map_hidden_dim,
            dropout=cfg.dropout,
        )

        self.u = nn.Embedding(cfg.n_subjects, cfg.k)
        nn.init.normal_(self.u.weight, std=0.05)

        self.item_map = ItemParameterMap(
            item_embed_dim=cfg.item_embed_dim,
            k=cfg.k,
            hidden=cfg.item_map_hidden_dim,
            dropout=cfg.dropout,
        )

        self.rho_factor = nn.Parameter(torch.tensor(0.1))

        in_dim = _residual_feature_dim_hybrid_irt_kfactor(cfg)
        self.residual = GatedSwiGLUResidual(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lambda_resid = nn.Parameter(
            torch.tensor(float(cfg.lambda_resid_init)),
            requires_grad=bool(cfg.lambda_resid_trainable),
        )

        self.cluster_embedding: nn.Embedding | None = None
        if cfg.has_cluster_embedding:
            self.cluster_embedding = nn.Embedding(
                cfg.n_clusters + 1, cfg.cluster_embed_dim, padding_idx=0
            )
            nn.init.normal_(self.cluster_embedding.weight, std=0.05)
        _register_subject_text_proj(self, cfg)

    @property
    def has_residual(self) -> bool:
        return True

    @property
    def has_irt_heads(self) -> bool:
        return True

    @property
    def has_judge_features(self) -> bool:
        return bool(self.cfg.use_judge_features and self.cfg.effective_judge_dim > 0)

    @property
    def has_nn_features(self) -> bool:
        return bool(self.cfg.use_nn_features and self.cfg.effective_nn_dim > 0)

    def _cluster_emb(self, cluster_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self.cluster_embedding is None or cluster_ids is None:
            return None
        if cluster_ids.numel() == 0 or cluster_ids.dim() < 1:
            return None
        return self.cluster_embedding(cluster_ids.long())

    def _components(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        theta = self.theta(subject_idx).squeeze(-1)
        bc_off = self.beta(bc_idx).squeeze(-1)
        beta_i, alpha_i = self.irt_heads(item_emb)
        u_s = self.u(subject_idx)
        v_i, _unused_d_i = self.item_map(item_emb)
        k = max(1, self.cfg.k)
        raw_factor = (u_s * v_i).sum(dim=-1) / math.sqrt(k)
        c_irt = alpha_i * (theta - beta_i)
        c_offset = bc_off + self.mu
        c_factor = self.rho_factor * raw_factor
        return {
            "theta": theta,
            "beta_i": beta_i,
            "alpha_i": alpha_i,
            "u_s": u_s,
            "v_i": v_i,
            "raw_factor": raw_factor,
            "irt": c_irt,
            "offset": c_offset,
            "factor": c_factor,
        }

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
        override_mlp_zero: bool = False,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
    ) -> torch.Tensor:
        comps = self._components(subject_idx, bc_idx, item_emb)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta"] - beta_i)
        c_offset = comps["offset"]
        c_factor = comps["factor"]
        if override_mlp_zero:
            return c_irt + c_offset + c_factor
        jf = _maybe_zero_judge(self.cfg, judge_feats, force_judge_zero=force_judge_zero)
        nf = _maybe_zero_nn(self.cfg, nn_feats, force_nn_zero=force_nn_zero)
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
        cluster_emb = self._cluster_emb(cluster_ids)
        x = _build_residual_features_hybrid_irt_kfactor(
            self.cfg,
            theta=comps["theta"],
            beta_i=comps["beta_i"],
            alpha_i=comps["alpha_i"],
            u_s=comps["u_s"],
            v_i=comps["v_i"],
            raw_factor=comps["raw_factor"],
            item_emb=item_emb,
            subject_emb=subject_emb,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
            judge_feats=jf,
            nn_feats=nf,
        )
        r = self.lambda_resid * self.residual(x)
        return c_irt + c_offset + c_factor + r

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._components(subject_idx, bc_idx, item_emb)
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
        cluster_emb = self._cluster_emb(cluster_ids)
        x = _build_residual_features_hybrid_irt_kfactor(
            self.cfg,
            theta=comps["theta"],
            beta_i=comps["beta_i"],
            alpha_i=comps["alpha_i"],
            u_s=comps["u_s"],
            v_i=comps["v_i"],
            raw_factor=comps["raw_factor"],
            item_emb=item_emb,
            subject_emb=subject_emb,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
            judge_feats=judge_feats,
            nn_feats=nn_feats,
        )
        r = self.lambda_resid * self.residual(x)
        return {
            "irt": comps["irt"],
            "offset": comps["offset"],
            "factor": comps["factor"],
            "mlp": r,
            "theta": comps["theta"],
            "beta_i": comps["beta_i"],
            "alpha_i": comps["alpha_i"],
        }


# ---------------------------------------------------------------------------
# Hierarchical MIRT (scalar IRT inductive bias + zero-init multi-dim MIRT
# residual + gated MLP). New variant; see scripts/_sim_cold_start_hierarchical.py
# for the design rationale and ablation results.
# ---------------------------------------------------------------------------


def _residual_feature_dim_hierarchical_mirt(cfg: ModelConfig) -> int:
    """Width of the residual-MLP input for the hierarchical-MIRT variant.

    Exposes the scalar-IRT components (theta, beta_i, alpha_i) and the
    multi-dim MIRT components (theta_vec, alpha_vec, theta_vec * alpha_vec,
    raw dot scalar) so the residual MLP can correct anything the structured
    channels miss. ``cfg.k`` is reused as the MIRT dimension d.
    """
    d = max(1, int(cfg.k))
    base = (
        cfg.item_embed_dim
        + 3                              # theta, beta_i, alpha_i scalars
        + d                              # theta_vec
        + d                              # alpha_vec
        + d                              # theta_vec * alpha_vec
        + 1                              # raw MIRT dot scalar
    )
    if cfg.use_subject_embed_features:
        base += cfg.effective_subject_feature_dim
    if cfg.use_pool_features:
        base += cfg.effective_pool_dim
    if cfg.has_cluster_embedding:
        base += cfg.cluster_embed_dim
    if cfg.use_judge_features:
        base += cfg.effective_judge_dim
    if cfg.use_nn_features:
        base += cfg.effective_nn_dim
    return base


def _build_residual_features_hierarchical_mirt(
    cfg: ModelConfig,
    *,
    theta: torch.Tensor,
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    theta_vec: torch.Tensor,
    alpha_vec: torch.Tensor,
    raw_mirt: torch.Tensor,
    item_emb: torch.Tensor,
    subject_emb: torch.Tensor | None,
    pool_z: torch.Tensor | None,
    cluster_emb: torch.Tensor | None,
    judge_feats: torch.Tensor | None = None,
    nn_feats: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [
        item_emb,
        theta.unsqueeze(-1),
        beta_i.unsqueeze(-1),
        alpha_i.unsqueeze(-1),
        theta_vec,
        alpha_vec,
        theta_vec * alpha_vec,
        raw_mirt.unsqueeze(-1),
    ]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    _maybe_append(
        parts,
        cfg,
        pool_z=pool_z,
        cluster_emb=cluster_emb,
        judge_feats=judge_feats,
        nn_feats=nn_feats,
    )
    return torch.cat(parts, dim=-1)


class HierarchicalMIRT(nn.Module):
    """Hierarchical MIRT head: scalar IRT prior + zero-init multi-dim residual.

    The final logit is:

        logit = mu
              + beta_bc
              + alpha_i(emb) * (theta_s - beta_i(emb))     # scalar 2PL
              + < alpha_vec(emb), theta_vec[s] >            # MIRT-d (zero at init)
              + lambda_resid * gated_residual(F)            # gated MLP correction

    Init recipe (the whole reason this exists):

      * ``theta_s`` zero, ``alpha_i`` softplus pre-biased to start near 1.0
        (reuses :class:`ItemIRTHeads`), so the scalar-IRT channel is the
        active learner from step 0.
      * The output layer of ``alpha_vec_head`` is zero-initialized so the
        multi-dim contribution is exactly 0 at step 0 and grows only as
        gradients flow.  ``theta_vec`` carries small N(0, 0.05) noise so
        the gradient is non-degenerate from step 1.
      * The gated MLP residual has zero-initialized output (matches the
        rest of the family) and is scaled by a small ``lambda_resid``.

    The point: classical 2PL inductive bias when data is scarce (the cold-
    start regime), with the multi-dim MIRT and the residual MLP picking
    up structure that the scalar prior misses as data accumulates.
    Simulation results in ``scripts/_sim_cold_start_hierarchical.py``
    confirmed roughly +0.07 nats over the shipped hybrid at d=16 in the
    small-data cold-start regime, tying or matching plain MIRT_MLP in
    higher-data regimes.

    ``cfg.k`` is reused as the MIRT dimension ``d``; the existing notebook
    plumbing that passes ``k`` through to ``ModelConfig`` works unchanged.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.zeros(1))
        self.theta = nn.Embedding(cfg.n_subjects, 1)          # scalar ability
        self.beta = nn.Embedding(cfg.n_benchmark_conditions, 1)
        nn.init.zeros_(self.theta.weight)
        nn.init.zeros_(self.beta.weight)

        # Scalar IRT channel (2PL): predicts (beta_i, alpha_i>0) from the
        # item embedding.  Reuses the existing ItemIRTHeads so checkpoints
        # share the same head layout the older IRT variants use.
        self.irt_heads = ItemIRTHeads(
            item_dim=cfg.item_embed_dim,
            hidden=cfg.item_map_hidden_dim,
            dropout=cfg.dropout,
        )
        # Match the hierarchical-MIRT simulation: the scalar difficulty head
        # starts at exactly zero so the model begins as mu + beta_bc, then
        # learns the 2PL channel before the zero-init MIRT residual grows in.
        # Reusing ItemIRTHeads without this would leave random beta_i logits
        # active at step 0, which breaks the intended inductive bias.
        nn.init.zeros_(self.irt_heads.beta_head[-1].weight)
        nn.init.zeros_(self.irt_heads.beta_head[-1].bias)

        # MIRT-d residual channel (multi-dim discrimination + subject vector).
        d = max(1, int(cfg.k))
        self.theta_vec = nn.Embedding(cfg.n_subjects, d)
        nn.init.normal_(self.theta_vec.weight, std=0.05)

        # alpha_vec_head: LN -> Linear -> GELU -> Dropout -> Linear[hidden -> d].
        # Output layer is zero-initialized; this is the critical bit that
        # makes the multi-dim contribution start at exactly 0.
        self.alpha_vec_head = nn.Sequential(
            nn.LayerNorm(cfg.item_embed_dim),
            nn.Linear(cfg.item_embed_dim, cfg.item_map_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.item_map_hidden_dim, d),
        )
        nn.init.zeros_(self.alpha_vec_head[-1].weight)
        nn.init.zeros_(self.alpha_vec_head[-1].bias)

        in_dim = _residual_feature_dim_hierarchical_mirt(cfg)
        self.residual = GatedSwiGLUResidual(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lambda_resid = nn.Parameter(
            torch.tensor(float(cfg.lambda_resid_init)),
            requires_grad=bool(cfg.lambda_resid_trainable),
        )

        self.cluster_embedding: nn.Embedding | None = None
        if cfg.has_cluster_embedding:
            self.cluster_embedding = nn.Embedding(
                cfg.n_clusters + 1, cfg.cluster_embed_dim, padding_idx=0
            )
            nn.init.normal_(self.cluster_embedding.weight, std=0.05)
        _register_subject_text_proj(self, cfg)

    @property
    def has_residual(self) -> bool:
        return True

    @property
    def has_irt_heads(self) -> bool:
        return True

    @property
    def has_judge_features(self) -> bool:
        return bool(self.cfg.use_judge_features and self.cfg.effective_judge_dim > 0)

    @property
    def has_nn_features(self) -> bool:
        return bool(self.cfg.use_nn_features and self.cfg.effective_nn_dim > 0)

    def _cluster_emb(self, cluster_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self.cluster_embedding is None or cluster_ids is None:
            return None
        if cluster_ids.numel() == 0 or cluster_ids.dim() < 1:
            return None
        return self.cluster_embedding(cluster_ids.long())

    def _components(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        theta = self.theta(subject_idx).squeeze(-1)
        bc_off = self.beta(bc_idx).squeeze(-1)
        beta_i, alpha_i = self.irt_heads(item_emb)
        theta_vec = self.theta_vec(subject_idx)
        alpha_vec = self.alpha_vec_head(item_emb)
        raw_mirt = (theta_vec * alpha_vec).sum(dim=-1)
        c_irt = alpha_i * (theta - beta_i)
        c_offset = bc_off + self.mu
        c_mirt = raw_mirt
        return {
            "theta": theta,
            "beta_i": beta_i,
            "alpha_i": alpha_i,
            "theta_vec": theta_vec,
            "alpha_vec": alpha_vec,
            "raw_mirt": raw_mirt,
            "irt": c_irt,
            "offset": c_offset,
            "mirt": c_mirt,
        }

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
        *,
        override_alpha: torch.Tensor | None = None,
        override_beta: torch.Tensor | None = None,
        override_mlp_zero: bool = False,
        force_judge_zero: bool = False,
        force_nn_zero: bool = False,
    ) -> torch.Tensor:
        comps = self._components(subject_idx, bc_idx, item_emb)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta"] - beta_i)
        c_offset = comps["offset"]
        c_mirt = comps["mirt"]
        if override_mlp_zero:
            return c_irt + c_offset + c_mirt
        jf = _maybe_zero_judge(self.cfg, judge_feats, force_judge_zero=force_judge_zero)
        nf = _maybe_zero_nn(self.cfg, nn_feats, force_nn_zero=force_nn_zero)
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
        cluster_emb = self._cluster_emb(cluster_ids)
        x = _build_residual_features_hierarchical_mirt(
            self.cfg,
            theta=comps["theta"],
            beta_i=comps["beta_i"],
            alpha_i=comps["alpha_i"],
            theta_vec=comps["theta_vec"],
            alpha_vec=comps["alpha_vec"],
            raw_mirt=comps["raw_mirt"],
            item_emb=item_emb,
            subject_emb=subject_emb,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
            judge_feats=jf,
            nn_feats=nf,
        )
        r = self.lambda_resid * self.residual(x)
        return c_irt + c_offset + c_mirt + r

    def decompose(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
        pool_feats: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        judge_feats: torch.Tensor | None = None,
        nn_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._components(subject_idx, bc_idx, item_emb)
        subject_emb = _maybe_project_subject_emb(self, subject_emb)
        cluster_emb = self._cluster_emb(cluster_ids)
        x = _build_residual_features_hierarchical_mirt(
            self.cfg,
            theta=comps["theta"],
            beta_i=comps["beta_i"],
            alpha_i=comps["alpha_i"],
            theta_vec=comps["theta_vec"],
            alpha_vec=comps["alpha_vec"],
            raw_mirt=comps["raw_mirt"],
            item_emb=item_emb,
            subject_emb=subject_emb,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
            judge_feats=judge_feats,
            nn_feats=nn_feats,
        )
        r = self.lambda_resid * self.residual(x)
        return {
            "irt": comps["irt"],
            "offset": comps["offset"],
            "mirt": comps["mirt"],
            "mlp": r,
            "theta": comps["theta"],
            "beta_i": comps["beta_i"],
            "alpha_i": comps["alpha_i"],
        }


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
    "hybrid_irt_kfactor_gated_mlp": HybridIRTItemKFactorGatedMLP,
    "hierarchical_mirt": HierarchicalMIRT,
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
    cluster_id, judge_feats, nn_feats, label, sample_weight)`` per row.

    Pool features, cluster ids, judge features, and NN features are
    optional: if missing, zero-sized tensors are returned. ``sample_weight``
    is always present and defaults to a per-row scalar of ``1.0`` when the
    caller does not supply weights, so existing unweighted training is
    bit-for-bit equivalent to the pre-weights behavior. The trainer +
    eval code unpacks the 10-tuple; downstream models silently ignore
    zero-sized channels.

    The training and val datasets each receive their own ``nn`` matrix
    (shape ``[N, 8]`` when enabled, ``[N, 0]`` otherwise), pre-computed
    once at dataset construction time. This keeps the per-batch step
    free of FAISS / sparse-matrix lookups.
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
        judge_feats: np.ndarray | None = None,
        nn_feats: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
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
        if judge_feats is None:
            self.judge_feats = torch.zeros((len(labels), 0), dtype=torch.float32)
        else:
            self.judge_feats = torch.from_numpy(
                np.asarray(judge_feats, dtype=np.float32)
            )
        if nn_feats is None:
            self.nn_feats = torch.zeros((len(labels), 0), dtype=torch.float32)
        else:
            self.nn_feats = torch.from_numpy(
                np.asarray(nn_feats, dtype=np.float32)
            )
        if sample_weights is None:
            self.sample_weights = torch.ones(len(labels), dtype=torch.float32)
        else:
            sw = np.asarray(sample_weights, dtype=np.float32)
            if sw.shape[0] != len(labels):
                raise ValueError(
                    f"sample_weights length {sw.shape[0]} != labels length "
                    f"{len(labels)}"
                )
            self.sample_weights = torch.from_numpy(sw)

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
            self.judge_feats[idx],
            self.nn_feats[idx],
            self.labels[idx],
            self.sample_weights[idx],
        )


__all__ = [
    "DenseMLPResidual",
    "GatedSwiGLUResidual",
    "HierarchicalMIRT",
    "HybridIRTItemKFactorGatedMLP",
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
    "SubjectTextProjector",
    "build_model",
    "compute_subject_tie_loss",
    "irt_regularization",
    "model_has_irt_heads",
]
