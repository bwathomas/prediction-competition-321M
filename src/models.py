"""Model definitions for the k-factor / neural-IRT ablation.

All three models share the same backbone:

    logit(eta) = mu + alpha_s + beta_bc - d_i + <u_s, v_i> / sqrt(k)

where

- ``mu`` is a global bias,
- ``alpha_s`` is a learned subject intercept,
- ``beta_bc`` is a learned benchmark-condition intercept,
- ``d_i`` and ``v_i`` are predicted from the item embedding,
- ``u_s`` is a learned subject ability vector.

The three model variants are:

- ``KFactorModel``               -- exactly the factor model above.
- ``KFactorMLPResidual``         -- factor model + a small dense MLP residual.
- ``KFactorGatedMLPResidual``    -- factor model + a SwiGLU-style gated MLP residual.

Residuals are scaled by a (configurable, optionally trainable) ``lambda_resid``
so the residual cannot dominate the factor model early in training.

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
    """Hyperparameters shared by all three model variants."""

    k: int = 16                          # factor dimension
    item_embed_dim: int = 768            # raw transformer embedding dim
    item_map_hidden_dim: int = 512       # hidden dim of the item-parameter MLP
    residual_hidden_dim: int = 256       # hidden dim of the residual MLP (B/C)
    dropout: float = 0.1
    n_subjects: int = 1                  # populated at fit time
    n_benchmark_conditions: int = 1      # populated at fit time
    use_subject_text_embedding: bool = False  # use subject text vec as feature?
    subject_embed_dim: int = 0           # raw subject text embedding dim
    lambda_resid_init: float = 0.1
    lambda_resid_trainable: bool = True

    @property
    def use_subject_embed_features(self) -> bool:
        return bool(self.use_subject_text_embedding and self.subject_embed_dim > 0)


# ---------------------------------------------------------------------------
# Item parameter head (predicts difficulty d_i and item vector v_i)
# ---------------------------------------------------------------------------


class ItemParameterMap(nn.Module):
    """Maps a raw item embedding to (difficulty scalar, item factor vector).

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

    @property
    def has_residual(self) -> bool:
        return False

    def factor_logit(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (eta_factor, u_s, v_i, item_emb_norm).

        Returning the factor parts separately keeps the residual classes thin.
        """
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
        subject_emb: torch.Tensor | None = None,  # accepted for symmetry; unused
    ) -> torch.Tensor:
        eta, _, _, _ = self.factor_logit(subject_idx, bc_idx, item_emb)
        return eta


# ---------------------------------------------------------------------------
# MLP residual heads (Models B and C)
# ---------------------------------------------------------------------------


def _residual_feature_dim(cfg: ModelConfig) -> int:
    """Width of the feature vector handed to the residual MLP."""
    base = cfg.k + cfg.k + cfg.k + cfg.item_embed_dim + 1  # u_s, v_i, u_s*v_i, item_emb, beta
    if cfg.use_subject_embed_features:
        base += cfg.subject_embed_dim
    return base


def _build_residual_features(
    cfg: ModelConfig,
    u_s: torch.Tensor,
    v_i: torch.Tensor,
    item_emb: torch.Tensor,
    bc_idx_embed: torch.Tensor,
    subject_emb: torch.Tensor | None,
) -> torch.Tensor:
    parts = [u_s, v_i, u_s * v_i, item_emb, bc_idx_embed]
    if cfg.use_subject_embed_features and subject_emb is not None:
        parts.append(subject_emb)
    return torch.cat(parts, dim=-1)


class DenseMLPResidual(nn.Module):
    """Plain 2-layer MLP residual: LN -> Linear -> SiLU -> Dropout -> Linear."""

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
# Residual-augmented models (B and C)
# ---------------------------------------------------------------------------


class _ResidualKFactor(KFactorModel):
    """Common implementation for the MLP / gated-MLP residual models."""

    residual_cls: type[nn.Module] = DenseMLPResidual

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        in_dim = _residual_feature_dim(cfg)
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

    def forward(
        self,
        subject_idx: torch.Tensor,
        bc_idx: torch.Tensor,
        item_emb: torch.Tensor,
        subject_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eta_factor, u_s, v_i, item_emb_out = self.factor_logit(
            subject_idx, bc_idx, item_emb
        )
        beta_bc = self.beta(bc_idx)  # [B, 1]
        x = _build_residual_features(
            self.cfg, u_s, v_i, item_emb_out, beta_bc, subject_emb
        )
        r = self.residual(x)
        return eta_factor + self.lambda_resid * r


class KFactorMLPResidual(_ResidualKFactor):
    residual_cls = DenseMLPResidual


class KFactorGatedMLPResidual(_ResidualKFactor):
    residual_cls = GatedSwiGLUResidual


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "kfactor": KFactorModel,
    "kfactor_mlp": KFactorMLPResidual,
    "kfactor_gated_mlp": KFactorGatedMLPResidual,
}


def build_model(name: str, cfg: ModelConfig) -> nn.Module:
    """Instantiate a model variant by registry name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg)


# ---------------------------------------------------------------------------
# Indexer: maps subject_key / benchmark_condition_key to integer ids and the
# UNK rows that test-time `predict()` will fall back to.
# ---------------------------------------------------------------------------


@dataclass
class Indexer:
    """Bijective mapping {subject_key -> int} and {bc_key -> int}.

    Index 0 is always reserved for UNK in each space, so test-time
    `predict()` can route unseen subjects / benchmark-conditions to a safe
    fallback that contributes the learned UNK intercept (zero by initialization)
    and no usable u_s direction.

    Train-time keys are appended starting at index 1.
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
    """Yields (subject_id, bc_id, item_emb, subject_emb_or_zeros, label).

    Embeddings are passed in as float32 numpy arrays indexed by item_key /
    subject_key. We materialize them at __init__ time as float32 tensors
    aligned with the dataframe row order.
    """

    def __init__(
        self,
        subject_ids: np.ndarray,
        bc_ids: np.ndarray,
        item_emb: np.ndarray,
        labels: np.ndarray,
        subject_emb: np.ndarray | None = None,
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
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.subject_ids[idx],
            self.bc_ids[idx],
            self.item_emb[idx],
            self.subject_emb[idx],
            self.labels[idx],
        )


__all__ = [
    "DenseMLPResidual",
    "GatedSwiGLUResidual",
    "Indexer",
    "ItemParameterMap",
    "KFactorGatedMLPResidual",
    "KFactorMLPResidual",
    "KFactorModel",
    "LookupDataset",
    "MODEL_REGISTRY",
    "ModelConfig",
    "build_model",
]
