"""Metadata-only latent-factor model for the Predictive AI Evaluation Challenge.

Model formula
-------------
    eta_{m,b,c} = mu                                # global bias
                  + a_m                             # model ability (scalar)
                  + b_{b,c}                         # benchmark/condition easiness (scalar)
                  + dot(u_m, v_{b,c}) / sqrt(k)     # bilinear skill x requirement

    p_{m,b,c}   = sigmoid(eta_{m,b,c})

Where each "tower" produces (scalar, k-vector) from metadata embeddings + an
explicit residual ID embedding (regularized):

    a_m,    u_m    = ModelTower    (model_metadata, model_id)
    b_{bc}, v_{bc} = BenchmarkTower(benchmark_metadata, condition, benchmark_id)

Item content is intentionally NOT used. The model is purely a function of
the four-string runtime input minus item_content.

This module also handles:
- categorical vocabulary fitting (`__UNK__` / `__MISSING__` buckets,
  fitted strictly on the train split),
- numeric imputation/scaling (train-set median impute + mean/std scale,
  optional log1p transform for parameter count, missingness indicators),
- aggregated cell-level training (success rate per (model, benchmark,
  condition) cell), keeping row-level scoring for validation,
- mixed-precision training with gradient clipping and early stopping,
- a runtime inference wrapper (`LatentFactorInference`) that accepts the
  four-string official input and returns a probability,
- save/load of the full bundle (state_dict + preprocessor + lookups).
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EPS_PROB = 1e-6
LOGGER = logging.getLogger("latent_factor")
NAME_RE = re.compile(r"(?im)^\s*Name:\s*(.*?)\s*$")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def normalize_condition(value: object) -> str:
    """Mirror harness/utils.normalize_condition (literal "none" sentinel)."""
    if value is None:
        return "none"
    s = str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "none"
    return s


def extract_name_from_subject_content(subject_content: object) -> str:
    """Pull the model display name out of the rendered `subject_content`."""
    if not isinstance(subject_content, str):
        return ""
    m = NAME_RE.search(subject_content)
    return m.group(1).strip() if m else ""


def auto_embedding_dim(cardinality: int, max_dim: int = 32) -> int:
    """min(max_dim, round(1.6 * cardinality**0.56)), floor 4."""
    raw = max(4, int(round(1.6 * (max(2, cardinality) ** 0.56))))
    return int(min(max_dim, raw))


def detect_label_column(df: pd.DataFrame) -> str:
    """Find a binary-correctness column under any of the common names."""
    for c in ("label", "response", "correct", "is_correct", "y"):
        if c in df.columns:
            return c
    raise KeyError(
        f"No label column found in df. Looked for label/response/correct. "
        f"Columns present: {list(df.columns)[:20]}..."
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LFConfig:
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 65536
    epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0

    aggregate: bool = True
    use_amp: bool = True
    id_emb_l2: float = 1e-3

    max_embedding_dim: int = 32
    min_token_count: int = 1

    seed: int = 0

    model_categorical: tuple = ("organization", "family", "macro_family")
    benchmark_categorical: tuple = ("topic",)

    use_log_params: bool = True
    use_release: bool = True
    use_benchmark_age: bool = True


# ---------------------------------------------------------------------------
# Vocab + scaler
# ---------------------------------------------------------------------------


class CategoricalVocab:
    UNK = "__UNK__"
    MISSING = "__MISSING__"

    def __init__(self, name: str) -> None:
        self.name = name
        self.token_to_id: dict[str, int] = {self.MISSING: 0, self.UNK: 1}
        self.frozen = False

    def fit(
        self,
        values: Iterable[Any],
        min_count: int = 1,
    ) -> "CategoricalVocab":
        counter: Counter[str] = Counter()
        for v in values:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                key = self.MISSING
            else:
                s = str(v).strip()
                key = self.MISSING if s == "" else s
            counter[key] += 1
        for tok, cnt in sorted(counter.items()):
            if tok in (self.MISSING, self.UNK):
                continue
            if cnt >= min_count:
                self.token_to_id.setdefault(tok, len(self.token_to_id))
        self.frozen = True
        return self

    def encode(self, values: Iterable[Any]) -> np.ndarray:
        miss = self.token_to_id[self.MISSING]
        unk = self.token_to_id[self.UNK]
        out = np.empty(len(values) if hasattr(values, "__len__") else 0, dtype=np.int64)
        if not hasattr(values, "__len__"):
            values = list(values)
            out = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[i] = miss
                continue
            s = str(v).strip()
            if s == "":
                out[i] = miss
            else:
                out[i] = self.token_to_id.get(s, unk)
        return out

    def __len__(self) -> int:
        return len(self.token_to_id)


class NumericScaler:
    def __init__(self, name: str, log_transform: bool = False) -> None:
        self.name = name
        self.log_transform = log_transform
        self.median = 0.0
        self.mean = 0.0
        self.std = 1.0

    def _maybe_log(self, x: np.ndarray) -> np.ndarray:
        if self.log_transform:
            return np.log1p(np.maximum(x, 0.0))
        return x

    def fit(self, values: pd.Series) -> "NumericScaler":
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return self
        x_t = self._maybe_log(x)
        self.median = float(np.median(x))
        self.mean = float(np.mean(x_t))
        self.std = float(np.std(x_t)) or 1.0
        return self

    def transform(self, values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
        x = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
        missing = (~np.isfinite(x)).astype(np.float32)
        x_filled = np.where(np.isfinite(x), x, self.median)
        x_t = self._maybe_log(x_filled)
        x_t = (x_t - self.mean) / self.std
        return x_t.astype(np.float32), missing


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


@dataclass
class PreprocessOutput:
    """All tensors needed by the model for a batch."""
    model_id: torch.Tensor                        # (N,)
    benchmark_id: torch.Tensor                    # (N,)
    condition_id: torch.Tensor                    # (N,)
    benchmark_condition_id: torch.Tensor          # (N,)

    model_cat_ids: dict[str, torch.Tensor]        # each (N,)
    benchmark_cat_ids: dict[str, torch.Tensor]    # each (N,)

    model_numeric: torch.Tensor                   # (N, n_model_num)
    benchmark_numeric: torch.Tensor               # (N, n_bench_num)


class MetadataPreprocessor:
    """Vocabularies + scalers fitted on the training split.

    Stores model_info / benchmark_info as small in-memory lookup tables so
    inference-time can join by display_name / benchmark.
    """

    def __init__(
        self,
        model_info_df: pd.DataFrame,
        benchmark_info_df: pd.DataFrame,
        config: LFConfig,
    ) -> None:
        self.config = config
        self.model_info = self._normalize_model_info(model_info_df)
        self.benchmark_info = self._normalize_benchmark_info(benchmark_info_df)

        self.model_id_vocab = CategoricalVocab("model_id")
        self.benchmark_vocab = CategoricalVocab("benchmark")
        self.condition_vocab = CategoricalVocab("condition")
        self.benchmark_condition_vocab = CategoricalVocab("benchmark::condition")

        self.model_cat_vocabs: dict[str, CategoricalVocab] = {}
        self.benchmark_cat_vocabs: dict[str, CategoricalVocab] = {}

        self.model_num_scalers: dict[str, NumericScaler] = {}
        self.benchmark_num_scalers: dict[str, NumericScaler] = {}

        self.cols_used: dict[str, list[str]] = {
            "model_categorical": [],
            "benchmark_categorical": [],
            "model_numeric": [],
            "benchmark_numeric": [],
        }

    @staticmethod
    def _normalize_model_info(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "name" not in out.columns:
            for cand in ("model_id", "model", "subject_id", "display_name"):
                if cand in out.columns:
                    out = out.rename(columns={cand: "name"})
                    break
        if "name" not in out.columns:
            raise KeyError(f"model_info must have a 'name' column; got {list(df.columns)}")
        out["name"] = out["name"].astype(str)
        if "macro_family" not in out.columns and "macro-family" in out.columns:
            out = out.rename(columns={"macro-family": "macro_family"})
        rename_map = {"params": "parameters"}
        for src, dst in rename_map.items():
            if src in out.columns and dst not in out.columns:
                out = out.rename(columns={src: dst})
        for col in ("organization", "family", "macro_family"):
            if col in out.columns:
                out[col] = out[col].astype(str)
                out.loc[out[col].str.lower().isin(["unknown", "nan", ""]), col] = np.nan
        for col in ("parameters", "release_date"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def _normalize_benchmark_info(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "benchmark" not in out.columns:
            for cand in ("benchmark_id", "name"):
                if cand in out.columns:
                    out = out.rename(columns={cand: "benchmark"})
                    break
        if "benchmark" not in out.columns:
            raise KeyError(f"benchmark_info must have a 'benchmark' column; got {list(df.columns)}")
        out["benchmark"] = out["benchmark"].astype(str)
        if "topic" in out.columns:
            out["topic"] = out["topic"].astype(str)
            out.loc[out["topic"].str.lower().isin(["unknown", "nan", ""]), "topic"] = np.nan
        for col in ("age", "has_conditions"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    # -- fit ---------------------------------------------------------------

    def _join_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "model_name" not in d.columns:
            d["model_name"] = d["subject_content"].map(extract_name_from_subject_content)
        d["benchmark"] = d["benchmark"].astype(str)
        d["condition_norm"] = d.get("condition", "none").map(normalize_condition)
        d["benchmark_condition"] = d["benchmark"] + "::" + d["condition_norm"]
        d = d.merge(self.model_info, left_on="model_name", right_on="name", how="left", suffixes=("", "_modelinfo"))
        d = d.merge(self.benchmark_info, on="benchmark", how="left", suffixes=("", "_benchinfo"))
        return d

    def fit(self, train_df: pd.DataFrame) -> "MetadataPreprocessor":
        joined = self._join_metadata(train_df)
        self.model_id_vocab.fit(joined["model_name"], min_count=self.config.min_token_count)
        self.benchmark_vocab.fit(joined["benchmark"], min_count=self.config.min_token_count)
        self.condition_vocab.fit(joined["condition_norm"], min_count=self.config.min_token_count)
        self.benchmark_condition_vocab.fit(
            joined["benchmark_condition"], min_count=self.config.min_token_count
        )

        for col in self.config.model_categorical:
            if col in joined.columns:
                v = CategoricalVocab(f"model_{col}")
                v.fit(joined[col], min_count=self.config.min_token_count)
                self.model_cat_vocabs[col] = v
                self.cols_used["model_categorical"].append(col)
        for col in self.config.benchmark_categorical:
            if col in joined.columns:
                v = CategoricalVocab(f"benchmark_{col}")
                v.fit(joined[col], min_count=self.config.min_token_count)
                self.benchmark_cat_vocabs[col] = v
                self.cols_used["benchmark_categorical"].append(col)

        if self.config.use_log_params and "parameters" in joined.columns:
            s = NumericScaler("log_params", log_transform=True).fit(joined["parameters"])
            self.model_num_scalers["log_params"] = s
            self.cols_used["model_numeric"].append("log_params")
        if self.config.use_release and "release_date" in joined.columns:
            s = NumericScaler("release_date", log_transform=False).fit(joined["release_date"])
            self.model_num_scalers["release_date"] = s
            self.cols_used["model_numeric"].append("release_date")
        if self.config.use_benchmark_age and "age" in joined.columns:
            s = NumericScaler("benchmark_age", log_transform=False).fit(joined["age"])
            self.benchmark_num_scalers["benchmark_age"] = s
            self.cols_used["benchmark_numeric"].append("benchmark_age")

        LOGGER.info(
            "Preprocessor fit: model_ids=%d  benchmarks=%d  conditions=%d  bench_cond=%d",
            len(self.model_id_vocab), len(self.benchmark_vocab),
            len(self.condition_vocab), len(self.benchmark_condition_vocab),
        )
        for col, v in self.model_cat_vocabs.items():
            LOGGER.info("  model.%s vocab size = %d", col, len(v))
        for col, v in self.benchmark_cat_vocabs.items():
            LOGGER.info("  bench.%s vocab size = %d", col, len(v))
        return self

    # -- transform ---------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> PreprocessOutput:
        joined = self._join_metadata(df)
        n = len(joined)

        model_id = torch.from_numpy(self.model_id_vocab.encode(joined["model_name"].tolist()))
        bench_id = torch.from_numpy(self.benchmark_vocab.encode(joined["benchmark"].tolist()))
        cond_id = torch.from_numpy(self.condition_vocab.encode(joined["condition_norm"].tolist()))
        bc_id = torch.from_numpy(self.benchmark_condition_vocab.encode(joined["benchmark_condition"].tolist()))

        model_cat_ids = {
            col: torch.from_numpy(v.encode(joined[col].tolist()))
            for col, v in self.model_cat_vocabs.items()
        }
        bench_cat_ids = {
            col: torch.from_numpy(v.encode(joined[col].tolist()))
            for col, v in self.benchmark_cat_vocabs.items()
        }

        model_num_cols, bench_num_cols = [], []
        for col, sc in self.model_num_scalers.items():
            src = joined["parameters"] if col == "log_params" else joined[col]
            x, miss = sc.transform(src.tolist())
            model_num_cols.append(x)
            model_num_cols.append(miss)
        for col, sc in self.benchmark_num_scalers.items():
            src = joined["age"] if col == "benchmark_age" else joined[col]
            x, miss = sc.transform(src.tolist())
            bench_num_cols.append(x)
            bench_num_cols.append(miss)

        if model_num_cols:
            model_numeric = torch.from_numpy(np.stack(model_num_cols, axis=1)).float()
        else:
            model_numeric = torch.zeros((n, 0), dtype=torch.float32)
        if bench_num_cols:
            benchmark_numeric = torch.from_numpy(np.stack(bench_num_cols, axis=1)).float()
        else:
            benchmark_numeric = torch.zeros((n, 0), dtype=torch.float32)

        return PreprocessOutput(
            model_id=model_id,
            benchmark_id=bench_id,
            condition_id=cond_id,
            benchmark_condition_id=bc_id,
            model_cat_ids=model_cat_ids,
            benchmark_cat_ids=bench_cat_ids,
            model_numeric=model_numeric,
            benchmark_numeric=benchmark_numeric,
        )

    # -- introspection ----------------------------------------------------

    def model_numeric_dim(self) -> int:
        return 2 * len(self.model_num_scalers)

    def benchmark_numeric_dim(self) -> int:
        return 2 * len(self.benchmark_num_scalers)


# ---------------------------------------------------------------------------
# Cell aggregation (success/count) for fast training
# ---------------------------------------------------------------------------


def aggregate_cells(
    df: pd.DataFrame,
    label_col: str,
    extra_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """Group rows by (model_name, benchmark, condition_norm) and return
    a frame with successes / count / target. Predictions of this model
    are constant inside each cell, so this is loss-equivalent to row
    training but ~rows/cells smaller.
    """
    work = df.copy()
    work["model_name"] = work["subject_content"].map(extract_name_from_subject_content)
    work["condition_norm"] = work.get("condition", "none").map(normalize_condition)
    label = pd.to_numeric(work[label_col], errors="coerce").clip(0.0, 1.0)
    work["_label"] = label.fillna(0.0)
    work["_count"] = label.notna().astype(np.int64)
    keys = ["model_name", "benchmark", "condition_norm", *extra_keys]
    keys = [k for k in keys if k in work.columns]
    grouped = work.groupby(keys, sort=False, observed=True)
    agg = grouped.agg(
        successes=("_label", "sum"),
        count=("_count", "sum"),
        subject_content=("subject_content", "first"),
        condition=("condition", "first"),
    ).reset_index()
    agg = agg[agg["count"] > 0]
    agg["target"] = agg["successes"] / agg["count"]
    return agg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MetadataDataset(Dataset):
    """Holds preprocessed tensors + targets/weights, indexed by integer row."""

    def __init__(
        self,
        pp: PreprocessOutput,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        self.pp = pp
        self.targets = targets.float()
        self.weights = weights.float() if weights is not None else torch.ones_like(self.targets)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return idx


def make_collate(pp: PreprocessOutput, targets: torch.Tensor, weights: torch.Tensor):
    def collate(idxs):
        idx = torch.as_tensor(idxs, dtype=torch.long)
        batch = {
            "model_id": pp.model_id[idx],
            "benchmark_id": pp.benchmark_id[idx],
            "condition_id": pp.condition_id[idx],
            "benchmark_condition_id": pp.benchmark_condition_id[idx],
            "model_cat_ids": {k: v[idx] for k, v in pp.model_cat_ids.items()},
            "benchmark_cat_ids": {k: v[idx] for k, v in pp.benchmark_cat_ids.items()},
            "model_numeric": pp.model_numeric[idx],
            "benchmark_numeric": pp.benchmark_numeric[idx],
            "target": targets[idx],
            "weight": weights[idx],
        }
        return batch
    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLPTower(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentFactorModel(nn.Module):
    """eta = mu + a_m + b_bc + (u_m . v_bc) / sqrt(k)."""

    def __init__(self, pp: MetadataPreprocessor, config: LFConfig) -> None:
        super().__init__()
        self.config = config
        self.k = int(config.latent_dim)

        # Residual ID embeddings (regularized via id_emb_l2)
        self.model_id_bias = nn.Embedding(len(pp.model_id_vocab), 1)
        self.model_id_skill = nn.Embedding(len(pp.model_id_vocab), self.k)
        self.bench_cond_bias = nn.Embedding(len(pp.benchmark_condition_vocab), 1)
        self.bench_cond_skill = nn.Embedding(len(pp.benchmark_condition_vocab), self.k)
        nn.init.zeros_(self.model_id_bias.weight)
        nn.init.normal_(self.model_id_skill.weight, std=0.05)
        nn.init.zeros_(self.bench_cond_bias.weight)
        nn.init.normal_(self.bench_cond_skill.weight, std=0.05)

        # Per-feature categorical embeddings (model side + benchmark side)
        self.model_cat_embs = nn.ModuleDict()
        model_cat_dim_total = 0
        for col, vocab in pp.model_cat_vocabs.items():
            d = auto_embedding_dim(len(vocab), max_dim=config.max_embedding_dim)
            self.model_cat_embs[col] = nn.Embedding(len(vocab), d)
            nn.init.normal_(self.model_cat_embs[col].weight, std=0.05)
            model_cat_dim_total += d

        self.bench_cat_embs = nn.ModuleDict()
        bench_cat_dim_total = 0
        for col, vocab in pp.benchmark_cat_vocabs.items():
            d = auto_embedding_dim(len(vocab), max_dim=config.max_embedding_dim)
            self.bench_cat_embs[col] = nn.Embedding(len(vocab), d)
            nn.init.normal_(self.bench_cat_embs[col].weight, std=0.05)
            bench_cat_dim_total += d

        # Also embed benchmark + condition standalone so the bench tower can
        # see them even if benchmark_categorical is empty.
        self.benchmark_emb = nn.Embedding(
            len(pp.benchmark_vocab),
            auto_embedding_dim(len(pp.benchmark_vocab), max_dim=config.max_embedding_dim),
        )
        nn.init.normal_(self.benchmark_emb.weight, std=0.05)
        self.condition_emb = nn.Embedding(
            len(pp.condition_vocab),
            auto_embedding_dim(len(pp.condition_vocab), max_dim=config.max_embedding_dim),
        )
        nn.init.normal_(self.condition_emb.weight, std=0.05)

        model_in = model_cat_dim_total + pp.model_numeric_dim()
        bench_in = (
            bench_cat_dim_total
            + self.benchmark_emb.embedding_dim
            + self.condition_emb.embedding_dim
            + pp.benchmark_numeric_dim()
        )

        # +1 for the per-side scalar (a_m / b_bc), +k for the side vector.
        self.model_tower = MLPTower(
            in_dim=max(model_in, 1),
            hidden_dim=config.hidden_dim,
            out_dim=1 + self.k,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.bench_tower = MLPTower(
            in_dim=max(bench_in, 1),
            hidden_dim=config.hidden_dim,
            out_dim=1 + self.k,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.global_bias = nn.Parameter(torch.zeros(1))

        self._model_in = model_in
        self._bench_in = bench_in

    def _model_features(self, batch: dict) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for col, emb in self.model_cat_embs.items():
            parts.append(emb(batch["model_cat_ids"][col]))
        if batch["model_numeric"].shape[1] > 0:
            parts.append(batch["model_numeric"])
        if not parts:
            n = batch["model_id"].shape[0]
            return torch.zeros((n, 1), device=batch["model_id"].device)
        return torch.cat(parts, dim=-1)

    def _bench_features(self, batch: dict) -> torch.Tensor:
        parts: list[torch.Tensor] = [
            self.benchmark_emb(batch["benchmark_id"]),
            self.condition_emb(batch["condition_id"]),
        ]
        for col, emb in self.bench_cat_embs.items():
            parts.append(emb(batch["benchmark_cat_ids"][col]))
        if batch["benchmark_numeric"].shape[1] > 0:
            parts.append(batch["benchmark_numeric"])
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        n = batch["model_id"].shape[0]

        m_feats = self._model_features(batch)
        b_feats = self._bench_features(batch)

        m_out = self.model_tower(m_feats)
        b_out = self.bench_tower(b_feats)
        assert m_out.shape == (n, 1 + self.k), m_out.shape
        assert b_out.shape == (n, 1 + self.k), b_out.shape

        a_m_prior = m_out[:, 0]
        u_m_prior = m_out[:, 1:]
        b_bc_prior = b_out[:, 0]
        v_bc_prior = b_out[:, 1:]

        a_m = a_m_prior + self.model_id_bias(batch["model_id"]).squeeze(-1)
        u_m = u_m_prior + self.model_id_skill(batch["model_id"])

        b_bc = b_bc_prior + self.bench_cond_bias(batch["benchmark_condition_id"]).squeeze(-1)
        v_bc = v_bc_prior + self.bench_cond_skill(batch["benchmark_condition_id"])

        bilinear = (u_m * v_bc).sum(dim=-1) / math.sqrt(self.k)

        eta = self.global_bias + a_m + b_bc + bilinear
        assert eta.shape == (n,), eta.shape
        return eta

    def regularization(self) -> torch.Tensor:
        """L2 over the residual ID embeddings only (rest is via AdamW)."""
        reg = (
            self.model_id_bias.weight.pow(2).mean()
            + self.model_id_skill.weight.pow(2).mean()
            + self.bench_cond_bias.weight.pow(2).mean()
            + self.bench_cond_skill.weight.pow(2).mean()
        )
        return self.config.id_emb_l2 * reg


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _move_to(batch: dict, device: torch.device) -> dict:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _compute_loss(model: LatentFactorModel, batch: dict) -> torch.Tensor:
    eta = model(batch)
    bce = F.binary_cross_entropy_with_logits(
        eta, batch["target"], reduction="none"
    )
    weighted = (bce * batch["weight"]).sum() / batch["weight"].sum().clamp_min(1.0)
    return weighted + model.regularization()


@torch.no_grad()
def evaluate(
    model: LatentFactorModel,
    dataset: MetadataDataset,
    device: torch.device,
    batch_size: int = 65536,
) -> dict[str, float]:
    model.eval()
    targets, preds, weights = [], [], []
    n = len(dataset)
    collate = make_collate(dataset.pp, dataset.targets, dataset.weights)
    for start in range(0, n, batch_size):
        batch = collate(list(range(start, min(start + batch_size, n))))
        batch = _move_to(batch, device)
        eta = model(batch)
        p = torch.sigmoid(eta).clamp(EPS_PROB, 1 - EPS_PROB)
        preds.append(p.detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
        weights.append(batch["weight"].detach().cpu().numpy())
    p_all = np.concatenate(preds)
    y_all = np.clip(np.concatenate(targets), 0.0, 1.0)
    w_all = np.concatenate(weights)
    ll = -(w_all * (y_all * np.log(p_all) + (1 - y_all) * np.log(1 - p_all))).sum() / w_all.sum()
    brier = (w_all * (y_all - p_all) ** 2).sum() / w_all.sum()
    out = {
        "log_likelihood": float(-ll),
        "brier": float(brier),
        "mean_pred": float((w_all * p_all).sum() / w_all.sum()),
        "mean_label": float((w_all * y_all).sum() / w_all.sum()),
    }
    yb = (y_all >= 0.5).astype(int)
    if yb.min() != yb.max():
        try:
            from sklearn.metrics import roc_auc_score
            out["auc_roc"] = float(roc_auc_score(yb, p_all, sample_weight=w_all))
        except Exception:
            out["auc_roc"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
    return out


def train_model(
    model: LatentFactorModel,
    train_dataset: MetadataDataset,
    val_dataset: MetadataDataset | None,
    config: LFConfig,
    device: torch.device,
    *,
    run_label: str = "",
    print_every_epoch: bool = True,
) -> dict[str, Any]:
    """Training loop with progress, ETA, AMP, gradient clipping, early stop.

    `run_label` is prepended to every log line so that interleaved logs
    from a parallel sweep stay attributable to their run. When called from
    a sweep, the worker also bumps the root logging format with its run
    index, so this is just an extra tag for downstream parsing.
    """
    tag = f"{run_label} " if run_label else ""

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_kwargs = dict(device_type=device.type, enabled=use_amp, dtype=torch.float16)

    n = len(train_dataset)
    bs = max(1, int(config.batch_size))
    steps_per_epoch = (n + bs - 1) // bs

    history: list[dict[str, float]] = []
    best_ll = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_no_improve = 0
    train_collate = make_collate(train_dataset.pp, train_dataset.targets, train_dataset.weights)

    started = time.perf_counter()
    LOGGER.info(
        "%sTrain start: rows=%d  steps/epoch=%d  bs=%d  amp=%s  device=%s  "
        "epochs=%d  patience=%d  lr=%.1e  wd=%.1e  id_l2=%.1e",
        tag, n, steps_per_epoch, bs, use_amp, device,
        config.epochs, config.patience, config.lr, config.weight_decay, config.id_emb_l2,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n)
        total_loss, total_w = 0.0, 0.0

        for step, start in enumerate(range(0, n, bs), start=1):
            idx = perm[start:start + bs].tolist()
            batch = train_collate(idx)
            batch = _move_to(batch, device)

            with torch.amp.autocast(**autocast_kwargs):
                loss = _compute_loss(model, batch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()

            wsum = float(batch["weight"].sum().item())
            total_loss += float(loss.detach().item()) * wsum
            total_w += wsum

        train_loss = total_loss / max(1.0, total_w)
        epoch_dt = time.perf_counter() - epoch_t0

        val_ll = float("nan")
        val_brier = float("nan")
        val_auc = float("nan")
        if val_dataset is not None:
            val_metrics = evaluate(model, val_dataset, device, batch_size=bs)
            val_ll = val_metrics["log_likelihood"]
            val_brier = val_metrics["brier"]
            val_auc = val_metrics.get("auc_roc", float("nan"))

        elapsed = time.perf_counter() - started
        avg_per_epoch = elapsed / epoch
        eta_s = avg_per_epoch * (config.epochs - epoch)
        improved = (val_dataset is not None) and (val_ll > best_ll + 1e-5)
        marker = " *" if improved else ""
        if print_every_epoch:
            LOGGER.info(
                "%sepoch %3d/%d  train_loss=%.4f  val_ll=%+.4f  val_brier=%.4f  "
                "val_auc=%.4f  epoch=%.2fs  elapsed=%.1fs  eta~%.1fs  best=%+.4f@%d%s",
                tag, epoch, config.epochs, train_loss, val_ll, val_brier, val_auc,
                epoch_dt, elapsed, eta_s,
                best_ll if best_ll != -float("inf") else val_ll,
                best_epoch if best_epoch > 0 else epoch,
                marker,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_log_likelihood": val_ll,
                "val_brier": val_brier,
                "val_auc": val_auc,
                "epoch_seconds": epoch_dt,
            }
        )

        if val_dataset is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            continue

        if improved:
            best_ll = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                LOGGER.info(
                    "%sEarly stopping at epoch %d (best epoch %d, best val_ll=%+.4f)",
                    tag, epoch, best_epoch, best_ll,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_log_likelihood": best_ll,
        "best_epoch": best_epoch,
        "wall_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_artifacts(
    out_dir: Path,
    model: LatentFactorModel,
    preprocessor: MetadataPreprocessor,
    config: LFConfig,
    metrics: Mapping[str, Any] | None = None,
    *,
    save_raw_weights: bool = True,
) -> None:
    """Persist the full reproducible bundle for one trained model.

    Files written
    -------------
    best_model.pt       full bundle: state_dict + config (load via load_artifacts)
    weights.pt          raw state_dict only (smaller, loadable into a fresh
                        LatentFactorModel built from the same preprocessor)
    preprocessor.pkl    fitted preprocessor (vocabularies, scalers, lookups)
    metrics.json        per-run scalar metrics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "config": asdict(config),
        },
        out_dir / "best_model.pt",
    )
    if save_raw_weights:
        torch.save(state, out_dir / "weights.pt")
    with open(out_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "cols_used": preprocessor.cols_used,
                "config": asdict(config),
            },
            f,
        )
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def load_artifacts(in_dir: Path, device: torch.device | None = None) -> tuple[LatentFactorModel, MetadataPreprocessor, LFConfig]:
    in_dir = Path(in_dir)
    with open(in_dir / "preprocessor.pkl", "rb") as f:
        bundle = pickle.load(f)
    preprocessor: MetadataPreprocessor = bundle["preprocessor"]
    config = LFConfig(**bundle["config"])

    ckpt = torch.load(in_dir / "best_model.pt", map_location=device or "cpu", weights_only=False)
    model = LatentFactorModel(preprocessor, config)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, preprocessor, config


# ---------------------------------------------------------------------------
# Inference wrapper used by submission/model.py
# ---------------------------------------------------------------------------


class LatentFactorInference:
    """Per-call wrapper for the four-string runtime contract.

    Caches by (benchmark, condition, model_name) since the model is
    item-content-independent.
    """

    def __init__(
        self,
        model: LatentFactorModel,
        preprocessor: MetadataPreprocessor,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        self._cache: dict[tuple[str, str, str], float] = {}

    @torch.no_grad()
    def predict_one(self, benchmark: str, condition: str, subject_content: str) -> float:
        name = extract_name_from_subject_content(subject_content)
        cond_n = normalize_condition(condition)
        key = (str(benchmark), cond_n, name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        df = pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "condition": condition,
                    "subject_content": subject_content,
                    "item_content": "",
                }
            ]
        )
        pp = self.preprocessor.transform(df)
        batch = {
            "model_id": pp.model_id.to(self.device),
            "benchmark_id": pp.benchmark_id.to(self.device),
            "condition_id": pp.condition_id.to(self.device),
            "benchmark_condition_id": pp.benchmark_condition_id.to(self.device),
            "model_cat_ids": {k: v.to(self.device) for k, v in pp.model_cat_ids.items()},
            "benchmark_cat_ids": {k: v.to(self.device) for k, v in pp.benchmark_cat_ids.items()},
            "model_numeric": pp.model_numeric.to(self.device),
            "benchmark_numeric": pp.benchmark_numeric.to(self.device),
        }
        eta = self.model(batch)
        p = float(torch.sigmoid(eta).clamp(EPS_PROB, 1 - EPS_PROB).item())
        self._cache[key] = p
        return p


# ---------------------------------------------------------------------------
# Helper for submission-time metadata lookup export
# ---------------------------------------------------------------------------


def export_metadata_lookup(
    preprocessor: MetadataPreprocessor,
    out_dir: Path,
) -> None:
    """Save model_info / benchmark_info next to the checkpoint so the
    runtime model.py can join (subject_content -> Name -> metadata)
    without depending on the original source paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.model_info.to_csv(out_dir / "model_info.csv", index=False)
    preprocessor.benchmark_info.to_csv(out_dir / "benchmark_info.csv", index=False)
