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
    nn_feature_dim: int = 15

    # --- Structured-metadata features (subject + benchmark-condition) ---
    # When True, the model gains two metadata MLP "towers", a
    # factorization-machine cross head, and a small explicit-cross head
    # that all consume the columns from ``model_info.csv`` (subject
    # side: organization, family, macro_family, log_params,
    # release_date) and ``benchmark_info.csv`` (benchmark side: topic,
    # benchmark_age). Towers' outputs are added as cold-start priors to
    # the structured channels (``theta_eff = theta + a_meta``,
    # ``beta_eff = beta + b_meta``, ``u_eff = u + u_meta``); the FM and
    # explicit-cross heads add directly to the logit; the raw per-field
    # embeddings are also fed into the gated residual MLP for higher-
    # order nonlinear crosses. All output heads are zero-initialized so
    # ``use_metadata_features=True`` boots up bit-identical to the
    # non-metadata baseline (parity) and grows in only from gradients.
    use_metadata_features: bool = False
    meta_subject_categorical: tuple = ("organization", "family", "macro_family")
    meta_subject_numeric: tuple = ("log_params", "release_date")
    meta_benchmark_categorical: tuple = ("topic",)
    meta_benchmark_numeric: tuple = ("benchmark_age",)
    meta_explicit_crosses: tuple = ("family__topic", "macro_family__topic")
    meta_tower_hidden_dim: int = 128
    meta_tower_num_layers: int = 2
    meta_emb_max_dim: int = 16
    meta_fm_dim: int = 16
    meta_explicit_cross_emb_dim: int = 8
    # Channel mask -- lets ablations turn individual pathways off without
    # rebuilding the model. ``include_in_residual`` controls whether the
    # raw per-field metadata embeddings are concatenated into the gated
    # residual MLP input (the higher-order nonlinear channel).
    meta_include_tower_priors: bool = True
    meta_include_fm_cross: bool = True
    meta_include_explicit_crosses: bool = True
    meta_include_in_residual: bool = True
    # Pattern-2-style soft tie between the subject id embeddings and the
    # subject-tower metadata-derived embeddings. ``lambda_meta_tie > 0``
    # adds ``lambda * (MSE(theta, a_meta) + MSE(u, u_meta))`` to the
    # training loss. Off by default; the trainer calls a hook on the
    # model when this is non-zero.
    lambda_meta_tie: float = 0.0

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
# Metadata-aware hybrid: IRT + k-factor + gated MLP + structured metadata
# (towers / FM cross / explicit crosses). The training-time class. The
# runtime mirror lives in src/export_submission.py inside the
# _RUNTIME_MODEL_PY string.
# ---------------------------------------------------------------------------


def _residual_feature_dim_meta_hybrid(cfg: ModelConfig, *, meta_subj_emb_dim: int, meta_bc_emb_dim: int) -> int:
    """Width of the residual-MLP input for the metadata hybrid variant.

    Superset of :func:`_residual_feature_dim_hybrid_irt_kfactor`, plus
    (when ``cfg.meta_include_in_residual`` is set):
      - the concatenated per-field subject categorical embeddings,
      - the per-field benchmark categorical embeddings,
      - the subject numeric channels (scaled + missingness, 2*N),
      - the benchmark numeric channels,
      - an elementwise interaction ``proj(subj_meta) * proj(bench_meta)``
        materialized by the model (matches the existing ``u_s, v_i,
        u_s * v_i`` pattern so the MLP can directly read the cross).
    """
    base = _residual_feature_dim_hybrid_irt_kfactor(cfg)
    if not cfg.use_metadata_features or not cfg.meta_include_in_residual:
        return base
    n_sub_num = 2 * len(cfg.meta_subject_numeric)
    n_bc_num = 2 * len(cfg.meta_benchmark_numeric)
    inter_dim = min(meta_subj_emb_dim, meta_bc_emb_dim) if (meta_subj_emb_dim and meta_bc_emb_dim) else 0
    base += int(meta_subj_emb_dim) + int(meta_bc_emb_dim) + int(n_sub_num) + int(n_bc_num) + int(inter_dim)
    return base


class MetaHybridIRTKFactorGatedMLP(nn.Module):
    """Hybrid IRT + k-factor + gated MLP + structured metadata channels.

    Final logit:

        logit = mu
              + beta_bc_id + b_meta_bc
              + alpha_i * ((theta_id + a_meta_m) - beta_i)
              + rho_factor * ((u_id + u_meta_m) . v_i) / sqrt(k)
              + eta_fm_cross
              + eta_explicit_crosses
              + lambda_resid * gated_residual(F + meta_raw)

    Where ``a_meta_m, u_meta_m, b_meta_bc`` come from the per-side
    metadata MLP towers, and the FM + explicit-cross heads compose
    nonlinear interactions like ``family=Mistral x topic=Medicine``.

    Per-id metadata tensors live on the model as buffers (registered via
    :meth:`attach_metadata_tables`), so the existing ``LookupDataset``
    and trainer plumbing do not need to thread metadata through the
    batch. The buffers ship in the state_dict, so save / load is
    automatic.

    All metadata output heads (towers, FM, explicit cross) are
    zero-initialized -- a freshly built model with metadata enabled is
    bit-identical to its non-metadata sibling, and learns the metadata
    channel only as gradients flow.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Late import to avoid a circular import (data.py imports models;
        # metadata_features is standalone but kept untouched at the top
        # of this module so the existing import graph stays unchanged).
        from .metadata_features import (
            ExplicitCrossEmbeddings,
            FactorizationMachineCross,
            MetaTower,
            MetadataSchema,
            _PerFieldCategoricalEmbeddings,
        )

        self.cfg = cfg

        # --- Hybrid IRT + k-factor core (mirrors HybridIRTItemKFactorGatedMLP) ---
        self.mu = nn.Parameter(torch.zeros(1))
        self.theta = nn.Embedding(cfg.n_subjects, 1)
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

        # --- Metadata channels (built only when use_metadata_features=True) ---
        schema = MetadataSchema(
            subject_categorical=tuple(cfg.meta_subject_categorical),
            subject_numeric=tuple(cfg.meta_subject_numeric),
            benchmark_categorical=tuple(cfg.meta_benchmark_categorical),
            benchmark_numeric=tuple(cfg.meta_benchmark_numeric),
            explicit_crosses=tuple(cfg.meta_explicit_crosses),
        )
        self._meta_schema = schema
        # Cardinalities are populated when ``attach_metadata_tables`` is
        # called; we register placeholder modules with the right
        # *width* but zero rows so checkpoints save/load consistently
        # regardless of attachment order.
        self.subject_cat_embs = _PerFieldCategoricalEmbeddings(
            cardinalities=(2,) * len(schema.subject_categorical),
            max_emb_dim=cfg.meta_emb_max_dim,
        )
        self.benchmark_cat_embs = _PerFieldCategoricalEmbeddings(
            cardinalities=(2,) * len(schema.benchmark_categorical),
            max_emb_dim=cfg.meta_emb_max_dim,
        )
        n_sub_num = 2 * len(schema.subject_numeric)
        n_bc_num = 2 * len(schema.benchmark_numeric)
        subj_in = self.subject_cat_embs.total_dim + n_sub_num
        bench_in = self.benchmark_cat_embs.total_dim + n_bc_num

        self.subject_meta_tower = MetaTower(
            in_dim=subj_in,
            hidden_dim=cfg.meta_tower_hidden_dim,
            k=cfg.k,
            dropout=cfg.dropout,
            num_layers=cfg.meta_tower_num_layers,
        )
        # Benchmark tower only emits the scalar prior (b_meta_bc). The
        # k-vec on the bench side would have to combine with v_i and
        # is already covered by the bench_cat_embs flowing through the
        # residual MLP -- making it a tower output would duplicate that
        # path and risk double-counting against ``v_i``. So we set
        # ``k=1`` (a single dummy channel) and ignore the vector output.
        self.bench_meta_tower = MetaTower(
            in_dim=bench_in,
            hidden_dim=cfg.meta_tower_hidden_dim,
            k=1,
            dropout=cfg.dropout,
            num_layers=cfg.meta_tower_num_layers,
        )

        # Factorization-machine cross head over every metadata trait
        # (categorical + numeric on both sides). Each numeric field is
        # encoded as the ``(value, mask)`` pair via a per-field
        # ``Linear(2 -> d_fm)`` projection, so the FM bag covers every
        # pairwise interaction including subject_cat x bench_num,
        # subject_num x bench_cat, and subject_num x bench_num.
        self.fm_cross = FactorizationMachineCross(
            subject_field_dims=self.subject_cat_embs.dims,
            benchmark_field_dims=self.benchmark_cat_embs.dims,
            d_fm=cfg.meta_fm_dim,
            subject_num_field_count=len(schema.subject_numeric),
            bench_num_field_count=len(schema.benchmark_numeric),
        )

        # Explicit per-pair cross embeddings (one table per named cross).
        self.explicit_cross = ExplicitCrossEmbeddings(
            crosses=tuple(cfg.meta_explicit_crosses),
            schema=schema,
            subject_cardinalities=self.subject_cat_embs.cardinalities,
            benchmark_cardinalities=self.benchmark_cat_embs.cardinalities,
            emb_dim=cfg.meta_explicit_cross_emb_dim,
        )

        # Projection used to align the subject- and benchmark-side
        # categorical bags before elementwise interaction in the
        # residual-MLP feed. Matches the (u_s, v_i, u_s * v_i) trick
        # the hybrid already uses.
        inter_dim = (
            min(self.subject_cat_embs.total_dim, self.benchmark_cat_embs.total_dim)
            if (self.subject_cat_embs.total_dim and self.benchmark_cat_embs.total_dim)
            else 0
        )
        self.meta_inter_dim = int(inter_dim)
        if inter_dim > 0:
            self.meta_subj_inter_proj = nn.Linear(self.subject_cat_embs.total_dim, inter_dim)
            self.meta_bench_inter_proj = nn.Linear(self.benchmark_cat_embs.total_dim, inter_dim)
            nn.init.normal_(self.meta_subj_inter_proj.weight, std=0.05)
            nn.init.zeros_(self.meta_subj_inter_proj.bias)
            nn.init.normal_(self.meta_bench_inter_proj.weight, std=0.05)
            nn.init.zeros_(self.meta_bench_inter_proj.bias)
        else:
            self.meta_subj_inter_proj = None
            self.meta_bench_inter_proj = None

        # --- Per-id metadata buffer tables (filled by ``attach_metadata_tables``) ---
        # We register zero-sized buffers up front so ``state_dict`` /
        # ``load_state_dict`` always include the buffers; the attach
        # call resizes them in place. ``persistent=True`` is the
        # default and what we want for save/load.
        self.register_buffer(
            "subject_meta_cat_ids",
            torch.zeros((cfg.n_subjects, max(1, len(schema.subject_categorical))), dtype=torch.long),
        )
        self.register_buffer(
            "subject_meta_num",
            torch.zeros((cfg.n_subjects, max(2, n_sub_num)), dtype=torch.float32),
        )
        self.register_buffer(
            "bc_meta_cat_ids",
            torch.zeros(
                (cfg.n_benchmark_conditions, max(1, len(schema.benchmark_categorical))),
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "bc_meta_num",
            torch.zeros(
                (cfg.n_benchmark_conditions, max(2, n_bc_num)),
                dtype=torch.float32,
            ),
        )

        # --- Residual gated MLP (sized to the new total) ---
        meta_subj_emb_dim = self.subject_cat_embs.total_dim
        meta_bc_emb_dim = self.benchmark_cat_embs.total_dim
        in_dim = _residual_feature_dim_meta_hybrid(
            cfg,
            meta_subj_emb_dim=meta_subj_emb_dim if cfg.use_metadata_features else 0,
            meta_bc_emb_dim=meta_bc_emb_dim if cfg.use_metadata_features else 0,
        )
        self.residual = GatedSwiGLUResidual(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lambda_resid = nn.Parameter(
            torch.tensor(float(cfg.lambda_resid_init)),
            requires_grad=bool(cfg.lambda_resid_trainable),
        )

        # Pattern-1 cluster channel + Pattern-2 subject tie (carry over
        # the existing hybrid's channels so toggling metadata on top of
        # an established run is a clean superset).
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
    def has_metadata(self) -> bool:
        return bool(self.cfg.use_metadata_features)

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

    def attach_metadata_tables(self, tables) -> None:
        """Rebuild per-field embeddings + buffers from a MetadataIdTables.

        Called once after construction and before training so the embedding
        cardinalities, buffer shapes, and FM / explicit-cross modules
        agree with the actual fitted preprocessor. Idempotent: calling
        twice with the same tables is safe and produces no diff.

        This is the only place the model touches the
        :class:`MetadataIdTables` type, so the import stays local.
        """
        from .metadata_features import (
            ExplicitCrossEmbeddings,
            FactorizationMachineCross,
            MetadataIdTables,
            _PerFieldCategoricalEmbeddings,
        )

        if not isinstance(tables, MetadataIdTables):
            raise TypeError(
                f"attach_metadata_tables expects MetadataIdTables, got {type(tables).__name__}"
            )
        cfg = self.cfg
        schema = self._meta_schema
        n_sub_cat = len(schema.subject_categorical)
        n_bc_cat = len(schema.benchmark_categorical)
        n_sub_num = 2 * len(schema.subject_numeric)
        n_bc_num = 2 * len(schema.benchmark_numeric)

        # Rebuild per-field embedding tables with the *real* cardinalities.
        self.subject_cat_embs = _PerFieldCategoricalEmbeddings(
            cardinalities=tuple(tables.subject_cat_cardinalities) or (2,) * n_sub_cat,
            max_emb_dim=cfg.meta_emb_max_dim,
        )
        self.benchmark_cat_embs = _PerFieldCategoricalEmbeddings(
            cardinalities=tuple(tables.benchmark_cat_cardinalities) or (2,) * n_bc_cat,
            max_emb_dim=cfg.meta_emb_max_dim,
        )
        subj_in = self.subject_cat_embs.total_dim + n_sub_num
        bench_in = self.benchmark_cat_embs.total_dim + n_bc_num

        # Rebuild the towers so their input dims match the new emb widths.
        from .metadata_features import MetaTower

        self.subject_meta_tower = MetaTower(
            in_dim=subj_in,
            hidden_dim=cfg.meta_tower_hidden_dim,
            k=cfg.k,
            dropout=cfg.dropout,
            num_layers=cfg.meta_tower_num_layers,
        )
        self.bench_meta_tower = MetaTower(
            in_dim=bench_in,
            hidden_dim=cfg.meta_tower_hidden_dim,
            k=1,
            dropout=cfg.dropout,
            num_layers=cfg.meta_tower_num_layers,
        )
        self.fm_cross = FactorizationMachineCross(
            subject_field_dims=self.subject_cat_embs.dims,
            benchmark_field_dims=self.benchmark_cat_embs.dims,
            d_fm=cfg.meta_fm_dim,
            subject_num_field_count=len(schema.subject_numeric),
            bench_num_field_count=len(schema.benchmark_numeric),
        )
        self.explicit_cross = ExplicitCrossEmbeddings(
            crosses=tuple(cfg.meta_explicit_crosses),
            schema=schema,
            subject_cardinalities=self.subject_cat_embs.cardinalities,
            benchmark_cardinalities=self.benchmark_cat_embs.cardinalities,
            emb_dim=cfg.meta_explicit_cross_emb_dim,
        )

        # Refresh the elementwise-interaction projections.
        inter_dim = (
            min(self.subject_cat_embs.total_dim, self.benchmark_cat_embs.total_dim)
            if (self.subject_cat_embs.total_dim and self.benchmark_cat_embs.total_dim)
            else 0
        )
        self.meta_inter_dim = int(inter_dim)
        if inter_dim > 0:
            self.meta_subj_inter_proj = nn.Linear(
                self.subject_cat_embs.total_dim, inter_dim
            )
            self.meta_bench_inter_proj = nn.Linear(
                self.benchmark_cat_embs.total_dim, inter_dim
            )
            nn.init.normal_(self.meta_subj_inter_proj.weight, std=0.05)
            nn.init.zeros_(self.meta_subj_inter_proj.bias)
            nn.init.normal_(self.meta_bench_inter_proj.weight, std=0.05)
            nn.init.zeros_(self.meta_bench_inter_proj.bias)
        else:
            self.meta_subj_inter_proj = None
            self.meta_bench_inter_proj = None

        # Resize / refill the buffers with the attached tables. Keep the
        # tensors on whatever device the model lives on.
        device = self.subject_meta_cat_ids.device
        self.subject_meta_cat_ids = tables.subject_cat_ids.to(device).long()
        self.subject_meta_num = tables.subject_num.to(device).float()
        self.bc_meta_cat_ids = tables.bc_cat_ids.to(device).long()
        self.bc_meta_num = tables.bc_num.to(device).float()

        # Re-register so PyTorch's state_dict / device-move plumbing
        # tracks the new tensors. The simple ``self.attr = tensor`` swap
        # would not register them as persistent buffers; we have to call
        # ``register_buffer`` to maintain the persistent flag.
        self.register_buffer("subject_meta_cat_ids", self.subject_meta_cat_ids)
        self.register_buffer("subject_meta_num", self.subject_meta_num)
        self.register_buffer("bc_meta_cat_ids", self.bc_meta_cat_ids)
        self.register_buffer("bc_meta_num", self.bc_meta_num)

        # Rebuild the residual MLP with the right input width.
        meta_subj_emb_dim = self.subject_cat_embs.total_dim
        meta_bc_emb_dim = self.benchmark_cat_embs.total_dim
        in_dim = _residual_feature_dim_meta_hybrid(
            cfg,
            meta_subj_emb_dim=meta_subj_emb_dim if cfg.use_metadata_features else 0,
            meta_bc_emb_dim=meta_bc_emb_dim if cfg.use_metadata_features else 0,
        )
        self.residual = GatedSwiGLUResidual(
            in_dim=in_dim,
            hidden=cfg.residual_hidden_dim,
            dropout=cfg.dropout,
        )

    # ------------------------------------------------------------------
    # Forward / decompose
    # ------------------------------------------------------------------

    def _gather_metadata(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        meta_override: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Look up per-row metadata tensors via the buffers (or override).

        At inference time the runtime can pass ``meta_override`` to
        substitute the buffer lookup for true cold-start subjects/bcs.
        """
        if meta_override is not None:
            return {
                "subj_cat": meta_override["subj_cat"].to(subject_idx.device).long(),
                "subj_num": meta_override["subj_num"].to(subject_idx.device).float(),
                "bc_cat": meta_override["bc_cat"].to(bc_idx.device).long(),
                "bc_num": meta_override["bc_num"].to(bc_idx.device).float(),
            }
        return {
            "subj_cat": self.subject_meta_cat_ids[subject_idx],
            "subj_num": self.subject_meta_num[subject_idx],
            "bc_cat": self.bc_meta_cat_ids[bc_idx],
            "bc_num": self.bc_meta_num[bc_idx],
        }

    def _meta_channels(
        self,
        meta: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the metadata stack: per-field embeddings + towers + FM + crosses.

        Returns a dict with the additive contributions and the raw
        per-field embedding bags so the residual MLP can read them.
        """
        cfg = self.cfg

        # Per-field categorical embedding bags. Each side returns a
        # (B, total_emb_dim) tensor; the *list* of per-field tensors is
        # what the FM head wants.
        subj_emb_bag = self.subject_cat_embs(meta["subj_cat"])
        bench_emb_bag = self.benchmark_cat_embs(meta["bc_cat"])
        subj_field_embs = [
            self.subject_cat_embs.embs[i](meta["subj_cat"][:, i])
            for i in range(len(self.subject_cat_embs.embs))
        ]
        bench_field_embs = [
            self.benchmark_cat_embs.embs[i](meta["bc_cat"][:, i])
            for i in range(len(self.benchmark_cat_embs.embs))
        ]

        # Subject + benchmark concatenated (categorical + numeric) for the towers.
        subj_tower_in = torch.cat([subj_emb_bag, meta["subj_num"]], dim=-1)
        bench_tower_in = torch.cat([bench_emb_bag, meta["bc_num"]], dim=-1)
        a_meta, u_meta = self.subject_meta_tower(subj_tower_in)
        b_meta, _bench_vec_unused = self.bench_meta_tower(bench_tower_in)

        # FM cross over every metadata trait: per-field categorical
        # embeddings on both sides PLUS the (value, mask) numeric pairs
        # on both sides. Passing ``subj_num`` / ``bc_num`` lets the FM
        # head compute pairwise interactions that involve numerics
        # (cat x num, num x cat, num x num) -- not just cat x cat.
        if cfg.meta_include_fm_cross:
            eta_fm = self.fm_cross(
                subj_field_embs,
                bench_field_embs,
                subj_num_features=meta["subj_num"],
                bench_num_features=meta["bc_num"],
            )
        else:
            eta_fm = torch.zeros_like(a_meta)
        # Explicit crosses
        if cfg.meta_include_explicit_crosses and self.explicit_cross.has_any:
            eta_explicit = self.explicit_cross(meta["subj_cat"], meta["bc_cat"])
        else:
            eta_explicit = torch.zeros_like(a_meta)

        return {
            "a_meta": a_meta,
            "u_meta": u_meta,
            "b_meta": b_meta,
            "eta_fm": eta_fm,
            "eta_explicit": eta_explicit,
            "subj_emb_bag": subj_emb_bag,
            "bench_emb_bag": bench_emb_bag,
        }

    def _components(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        meta_override: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        theta_id = self.theta(subject_idx).squeeze(-1)
        bc_off = self.beta(bc_idx).squeeze(-1)
        beta_i, alpha_i = self.irt_heads(item_emb)
        u_id = self.u(subject_idx)
        v_i, _unused_d_i = self.item_map(item_emb)
        k = max(1, self.cfg.k)

        if self.cfg.use_metadata_features:
            meta = self._gather_metadata(subject_idx, bc_idx, meta_override)
            mc = self._meta_channels(meta)
            if self.cfg.meta_include_tower_priors:
                theta_eff = theta_id + mc["a_meta"]
                u_eff = u_id + mc["u_meta"]
                b_meta_for_offset = mc["b_meta"]
            else:
                theta_eff = theta_id
                u_eff = u_id
                b_meta_for_offset = torch.zeros_like(theta_id)
            eta_fm = mc["eta_fm"] if self.cfg.meta_include_fm_cross else torch.zeros_like(theta_id)
            eta_explicit = (
                mc["eta_explicit"]
                if self.cfg.meta_include_explicit_crosses
                else torch.zeros_like(theta_id)
            )
        else:
            mc = None
            theta_eff = theta_id
            u_eff = u_id
            b_meta_for_offset = torch.zeros_like(theta_id)
            eta_fm = torch.zeros_like(theta_id)
            eta_explicit = torch.zeros_like(theta_id)

        raw_factor = (u_eff * v_i).sum(dim=-1) / math.sqrt(k)
        c_irt = alpha_i * (theta_eff - beta_i)
        c_offset = bc_off + self.mu + b_meta_for_offset
        c_factor = self.rho_factor * raw_factor

        return {
            "theta_id": theta_id,
            "theta_eff": theta_eff,
            "beta_i": beta_i,
            "alpha_i": alpha_i,
            "u_id": u_id,
            "u_eff": u_eff,
            "v_i": v_i,
            "raw_factor": raw_factor,
            "irt": c_irt,
            "offset": c_offset,
            "factor": c_factor,
            "eta_fm": eta_fm,
            "eta_explicit": eta_explicit,
            "meta_channels": mc,
        }

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
        cfg = self.cfg
        subject_emb_p = _maybe_project_subject_emb(self, subject_emb)
        cluster_emb = self._cluster_emb(cluster_ids)
        parts = [
            item_emb,
            comps["theta_eff"].unsqueeze(-1),
            comps["beta_i"].unsqueeze(-1),
            comps["alpha_i"].unsqueeze(-1),
            comps["u_eff"],
            comps["v_i"],
            comps["u_eff"] * comps["v_i"],
            comps["raw_factor"].unsqueeze(-1),
        ]
        if cfg.use_subject_embed_features and subject_emb_p is not None:
            parts.append(subject_emb_p)
        _maybe_append(
            parts,
            cfg,
            pool_z=pool_feats,
            cluster_emb=cluster_emb,
            judge_feats=judge_feats,
            nn_feats=nn_feats,
        )
        if cfg.use_metadata_features and cfg.meta_include_in_residual and comps["meta_channels"] is not None:
            mc = comps["meta_channels"]
            # The two cat-embedding bags + the two numeric channels go in
            # raw so the gated MLP can read per-field embeddings and
            # missingness directly. The elementwise interaction term
            # ``proj(subj_emb) * proj(bench_emb)`` is the "the MLP can
            # discover Mistral x Medicine" hard prior, matching the
            # existing ``u_s, v_i, u_s * v_i`` trick the hybrid already
            # uses for the k-factor channel.
            parts.append(mc["subj_emb_bag"])
            parts.append(mc["bench_emb_bag"])
            parts.append(mc["subj_num"])
            parts.append(mc["bc_num"])
            if (
                self.meta_subj_inter_proj is not None
                and self.meta_bench_inter_proj is not None
                and self.meta_inter_dim > 0
            ):
                inter = self.meta_subj_inter_proj(mc["subj_emb_bag"]) * self.meta_bench_inter_proj(mc["bench_emb_bag"])
                parts.append(inter)
        return torch.cat(parts, dim=-1)

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
        meta_override: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        comps = self._components(subject_idx, bc_idx, item_emb, meta_override=meta_override)
        alpha_i = override_alpha if override_alpha is not None else comps["alpha_i"]
        beta_i = override_beta if override_beta is not None else comps["beta_i"]
        c_irt = alpha_i * (comps["theta_eff"] - beta_i)
        c_offset = comps["offset"]
        c_factor = comps["factor"]
        eta_struct = c_irt + c_offset + c_factor + comps["eta_fm"] + comps["eta_explicit"]
        if override_mlp_zero:
            return eta_struct
        jf = _maybe_zero_judge(self.cfg, judge_feats, force_judge_zero=force_judge_zero)
        nf = _maybe_zero_nn(self.cfg, nn_feats, force_nn_zero=force_nn_zero)
        # Attach the raw numerics to the meta_channels dict so
        # ``_residual_input`` can include them in the residual feed.
        if comps["meta_channels"] is not None:
            meta = self._gather_metadata(subject_idx, bc_idx, meta_override)
            comps["meta_channels"]["subj_num"] = meta["subj_num"]
            comps["meta_channels"]["bc_num"] = meta["bc_num"]
        x = self._residual_input(comps, item_emb, subject_emb, pool_feats, cluster_ids, jf, nf)
        r = self.lambda_resid * self.residual(x)
        return eta_struct + r

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
        meta_override: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        comps = self._components(subject_idx, bc_idx, item_emb, meta_override=meta_override)
        if comps["meta_channels"] is not None:
            meta = self._gather_metadata(subject_idx, bc_idx, meta_override)
            comps["meta_channels"]["subj_num"] = meta["subj_num"]
            comps["meta_channels"]["bc_num"] = meta["bc_num"]
        x = self._residual_input(
            comps, item_emb, subject_emb, pool_feats, cluster_ids, judge_feats, nn_feats
        )
        r = self.lambda_resid * self.residual(x)
        return {
            "irt": comps["irt"],
            "offset": comps["offset"],
            "factor": comps["factor"],
            "fm": comps["eta_fm"],
            "explicit_cross": comps["eta_explicit"],
            "mlp": r,
            "theta": comps["theta_eff"],
            "beta_i": comps["beta_i"],
            "alpha_i": comps["alpha_i"],
        }

    def compute_meta_tie_loss(
        self, subject_idx: torch.Tensor
    ) -> torch.Tensor:
        """Pattern-1-style soft tie between subject id and metadata-tower outputs.

        Trainer multiplies the result by ``cfg.lambda_meta_tie`` and
        adds it to the BCE loss. Returns zero when metadata is disabled
        or when no subjects in the batch have any metadata (the all-
        UNK case at module-init time before ``attach_metadata_tables``).
        """
        if not self.cfg.use_metadata_features or self.cfg.lambda_meta_tie == 0.0:
            return torch.zeros((), device=subject_idx.device, dtype=torch.float32)
        meta = self._gather_metadata(subject_idx, subject_idx.new_zeros(subject_idx.shape))
        mc = self._meta_channels(meta)
        theta_id = self.theta(subject_idx).squeeze(-1)
        u_id = self.u(subject_idx)
        # Detach the metadata side to send gradients into the id table
        # only (we trust the metadata tower as a stable prior and pull
        # the id embedding toward it -- "id_toward_meta" direction).
        loss_theta = F.mse_loss(theta_id, mc["a_meta"].detach())
        loss_u = F.mse_loss(u_id, mc["u_meta"].detach())
        return loss_theta + loss_u


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
    "meta_hybrid_irt_kfactor_gated_mlp": MetaHybridIRTKFactorGatedMLP,
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


class IndexedEmbeddingView:
    """Compact row-indexable view over a unique embedding matrix
    plus a per-row pointer array.

    Behaviour contract: behaves like a ``[N, D]`` torch tensor for
    the *indexing* operations that ``LookupDataset`` and the
    chunked ``_score_dataset`` actually use --

    * ``len(view) == N``
    * ``view.shape == (N, D)``
    * ``view[i]`` (scalar int) returns the i-th row as a 1-D
      ``[D]`` tensor (zero-copy: a view into the unique stack).
    * ``view[a:b]`` (slice) returns the contiguous range as a
      stacked ``[b-a, D]`` tensor (materialised on demand via
      ``index_select`` -- one allocation per *batch*, not per
      *row*).
    * ``view.to(device, dtype=...)`` moves only the unique
      stack + pointers (which are tiny).

    Memory comparison on the 5M-row x 4096-dim x 296k-unique-items
    setting (the M-train dataset for the qwen8b notebook):

    * Stacked dense ``[N, D]`` float32 = ~80 GB.
    * IndexedEmbeddingView = ``[U, D]`` float32 + ``[N]`` int64
      = ~4.85 GB. Approximately 16x less RAM at no
      throughput cost (the per-batch ``index_select`` is
      negligible vs the model forward pass).

    Construction does *not* copy the unique stack. The caller is
    responsible for keeping the underlying tensors alive for the
    lifetime of the view; in practice ``IndexedEmbeddingView``
    holds Python references to both, so they are released
    together when the view is dropped.
    """

    def __init__(
        self,
        unique_emb: "torch.Tensor",  # type: ignore[name-defined]
        row_to_uniq: "torch.Tensor",  # type: ignore[name-defined]
    ) -> None:
        import torch as _torch

        if not isinstance(unique_emb, _torch.Tensor):
            unique_emb = _torch.from_numpy(
                np.asarray(unique_emb, dtype=np.float32)
            )
        if not isinstance(row_to_uniq, _torch.Tensor):
            row_to_uniq = _torch.from_numpy(
                np.asarray(row_to_uniq, dtype=np.int64)
            )
        if unique_emb.ndim != 2:
            raise ValueError(
                f"unique_emb must be 2-D, got shape {tuple(unique_emb.shape)}"
            )
        if row_to_uniq.ndim != 1:
            raise ValueError(
                f"row_to_uniq must be 1-D, got shape {tuple(row_to_uniq.shape)}"
            )
        if int(row_to_uniq.numel()) > 0:
            if int(row_to_uniq.max().item()) >= int(unique_emb.shape[0]):
                raise IndexError(
                    f"row_to_uniq max={int(row_to_uniq.max().item())} "
                    f">= unique_emb.shape[0]={int(unique_emb.shape[0])}"
                )
            if int(row_to_uniq.min().item()) < 0:
                raise IndexError(
                    f"row_to_uniq min={int(row_to_uniq.min().item())} < 0"
                )
        self._uniq = unique_emb.contiguous()
        self._idx = row_to_uniq.contiguous().to(_torch.long)
        self._n = int(self._idx.numel())
        self._d = int(self._uniq.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, self._d)

    @property
    def dtype(self):
        return self._uniq.dtype

    @property
    def device(self):
        return self._uniq.device

    @property
    def nbytes(self) -> int:
        return int(
            self._uniq.element_size() * self._uniq.numel()
            + self._idx.element_size() * self._idx.numel()
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        import torch as _torch

        if isinstance(idx, slice):
            sel = self._idx[idx]
            if int(sel.numel()) == 0:
                return self._uniq.new_empty((0, self._d))
            return self._uniq.index_select(0, sel)
        if isinstance(idx, (list, tuple, np.ndarray)):
            # ``idx`` is a list of LOGICAL row indices into the
            # ``[N, D]`` view, NOT into the underlying unique
            # stack. Map row -> unique pointer first, then gather.
            row_sel = _torch.as_tensor(idx, dtype=_torch.long)
            return self._uniq.index_select(0, self._idx.index_select(0, row_sel))
        if isinstance(idx, _torch.Tensor):
            if idx.dtype == _torch.bool:
                sel = self._idx[idx]
            else:
                sel = self._idx.index_select(0, idx.long())
            return self._uniq.index_select(0, sel.long())
        return self._uniq[int(self._idx[int(idx)].item())]

    def to(self, *args, **kwargs) -> "IndexedEmbeddingView":
        moved_uniq = self._uniq.to(*args, **kwargs)
        # The pointer array stays on CPU by default; downstream
        # consumers only index into ``self._uniq`` from a single
        # device at a time so we follow the unique-stack's device.
        if "device" in kwargs:
            moved_idx = self._idx.to(kwargs["device"])
        elif args:
            first = args[0]
            try:
                moved_idx = self._idx.to(first)
            except (TypeError, RuntimeError):
                moved_idx = self._idx
        else:
            moved_idx = self._idx
        return IndexedEmbeddingView(moved_uniq, moved_idx)


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

    ``item_emb`` may be either a stacked ``[N, D]`` array/tensor (the
    legacy form, which materialises one row per observation and so
    needs ~``N*D*4`` bytes) **or** an :class:`IndexedEmbeddingView`,
    which holds ``[U, D]`` unique-item embeddings + a ``[N]``
    pointer and lazily yields per-row vectors on indexing. The
    view is the right choice whenever many of the ``N`` rows share
    the same item (e.g. the M-train split with ~17x average
    item-row multiplicity); it cuts ~80 GB of dataset memory on
    the qwen8b notebook to ~5 GB at zero accuracy cost.
    """

    def __init__(
        self,
        subject_ids: np.ndarray,
        bc_ids: np.ndarray,
        item_emb,
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
        if isinstance(item_emb, IndexedEmbeddingView):
            if int(item_emb.shape[0]) != int(len(labels)):
                raise ValueError(
                    f"item_emb (IndexedEmbeddingView) length "
                    f"{int(item_emb.shape[0])} != labels length "
                    f"{int(len(labels))}"
                )
            self.item_emb = item_emb
        else:
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
    "IndexedEmbeddingView",
    "Indexer",
    "ItemIRTHeads",
    "ItemParameterMap",
    "KFactorGatedMLPResidual",
    "KFactorMLPResidual",
    "KFactorModel",
    "LookupDataset",
    "MODEL_REGISTRY",
    "MetaHybridIRTKFactorGatedMLP",
    "ModelConfig",
    "SubjectTextProjector",
    "build_model",
    "compute_subject_tie_loss",
    "irt_regularization",
    "model_has_irt_heads",
]
