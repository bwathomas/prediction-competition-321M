"""Structured-metadata feature stack for the Predictive AI Evaluation Challenge.

This module adds a model-side and benchmark/condition-side **structured
metadata pathway** on top of the existing hybrid IRT + k-factor + gated
residual MLP. The motivation, in two lines:

- The id embeddings in the current hybrid model (`theta`, `beta`, `u`)
  are purely id-indexed -- a brand-new subject or benchmark-condition
  collapses to a single random-init UNK row even though we know the
  subject's organization / family / parameters and the benchmark's
  topic / age. The old metadata-only ``latent_factor_pytorch`` proved
  those structured columns carry the bulk of the cold-start signal.
- A pure additive metadata tower (the bilinear ``u_meta . v_meta`` trick
  from that LF model) captures *cold-start priors* but can only model
  rank-k bilinear cross-features. We additionally want to learn
  nonlinear crosses like ``family=Mistral x topic=Medicine`` -- those
  require a dedicated cross channel.

Architecture (this file builds the three pieces; ``models.py`` plugs them
in as parallel additive heads on top of the hybrid logit):

1. ``MetadataPreprocessor`` -- fits categorical vocabularies + numeric
   scalers on the training split, joins ``model_info.csv`` (subject
   metadata) and ``benchmark_info.csv`` (benchmark metadata). Saves /
   loads as a primitive-types dict (JSON-serializable) so the runtime
   ``model.py`` can rebuild it without depending on this module.

2. ``SubjectMetaTower`` / ``BenchCondMetaTower`` -- per-side MLPs that
   produce ``(scalar bias, k-vec)`` additive priors. Their output heads
   are **zero-initialized** so the model boots up bit-identical to the
   unmodified hybrid and learns the metadata channel from gradients
   only as data accumulates.

3. ``FactorizationMachineCross`` -- a pairwise-cross head that scores
   every ``(subject_categorical, bench_categorical)`` pair via a
   shared latent space. Sample-efficient at exactly the bilinear
   cross-feature task. Output is a single additive scalar.

4. ``ExplicitCrossEmbeddings`` -- a tiny lookup table per named cross
   (e.g. ``(family, topic)``). The cross id is computed from the
   per-side categorical ids and looked up directly. Maximally
   sample-efficient for crosses you know matter.

5. ``MetadataIdTables`` -- per-(subject_idx, bench_condition_idx)
   integer / numeric tensor tables, built once at fit time from the
   ``Indexer`` and the training dataframe. The model stores these as
   buffers and indexes them by ``subject_idx`` / ``bc_idx`` at forward
   time, so the existing ``LookupDataset`` and trainer plumbing don't
   need to change. Inference-time UNK subjects/benchmarks fall through
   the index-0 (MISSING) row by construction; the export runtime can
   optionally override with on-the-fly CSV lookup for true cold-start
   subjects (see ``runtime_lookup_for_one``).

All new channels are zero-init at the output and additive at the logit
level, so ``ModelConfig.use_metadata_features=False`` (the default) is
guaranteed parity with pre-metadata checkpoints.

Save / load convention
----------------------

``MetadataPreprocessor.to_dict()`` and ``MetadataIdTables.to_dict()``
serialize to plain Python dicts of lists / numbers. The runtime
``model.py`` does NOT import this module; it re-implements the few
encoding primitives it needs (~25 lines) and reads the serialized
dicts straight from JSON. This keeps the submission self-contained
without forcing the runtime to pickle module references.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sentinel tokens for categorical vocabularies. Index 0 is always
# ``__MISSING__`` (the "I have no value for this field" bucket) and
# index 1 is always ``__UNK__`` (the "value is present but was not
# seen during training" bucket). Keeping these in lockstep with the
# old ``latent_factor_pytorch.CategoricalVocab`` means we can lift the
# fitted vocab over without re-fitting if we ever want to.
TOKEN_MISSING = "__MISSING__"
TOKEN_UNK = "__UNK__"

# Regex used to extract the model display name from the rendered
# ``subject_content`` string at training / runtime. The hosted runtime
# constructs subject_content from ``subjects.parquet`` by emitting a
# ``Name: <display_name>`` first line (see
# ``src.data.render_subject_content``), so a single anchored regex
# recovers the name reliably.
_NAME_RE = re.compile(r"(?im)^\s*Name:\s*(.*?)\s*$")


# ---------------------------------------------------------------------------
# Small helpers (kept module-local to avoid circular imports with src.data
# -- this module is loaded by src.models which is loaded by src.data).
# ---------------------------------------------------------------------------


def extract_display_name(subject_content: object) -> str:
    """Pull the display name out of the rendered subject card."""
    if not isinstance(subject_content, str):
        return ""
    m = _NAME_RE.search(subject_content)
    return m.group(1).strip() if m else ""


def normalize_condition_token(value: object) -> str:
    """Mirror the harness's normalize_condition (literal "none" sentinel)."""
    if value is None:
        return "none"
    s = str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "none"
    return s


# ---------------------------------------------------------------------------
# Categorical vocab and numeric scaler
# ---------------------------------------------------------------------------


@dataclass
class CategoricalVocab:
    """Fit-once token -> int map with MISSING and UNK reserved at 0 / 1.

    Designed so that ``encode([])`` works for cold-start (returns an
    empty array) and ``encode([None])`` -> ``[0]`` (the MISSING bucket).
    """

    name: str
    token_to_id: dict[str, int] = field(default_factory=dict)
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {TOKEN_MISSING: 0, TOKEN_UNK: 1}

    @property
    def n_tokens(self) -> int:
        return len(self.token_to_id)

    def _key_for(self, v: object) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return TOKEN_MISSING
        s = str(v).strip()
        if not s:
            return TOKEN_MISSING
        if s.lower() in {"nan", "none", "null", "unknown"}:
            return TOKEN_MISSING
        return s

    def fit(self, values: Iterable[Any], min_count: int = 1) -> "CategoricalVocab":
        counter: Counter[str] = Counter()
        for v in values:
            k = self._key_for(v)
            if k in (TOKEN_MISSING, TOKEN_UNK):
                continue
            counter[k] += 1
        # Sorted for determinism: token_to_id depends only on the values.
        for tok, cnt in sorted(counter.items()):
            if cnt >= int(min_count):
                self.token_to_id.setdefault(tok, len(self.token_to_id))
        self.frozen = True
        return self

    def encode_one(self, value: object) -> int:
        k = self._key_for(value)
        if k == TOKEN_MISSING:
            return 0
        return self.token_to_id.get(k, 1)

    def encode(self, values: Iterable[Any]) -> np.ndarray:
        vals = list(values)
        out = np.empty(len(vals), dtype=np.int64)
        for i, v in enumerate(vals):
            out[i] = self.encode_one(v)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "token_to_id": dict(self.token_to_id),
            "frozen": bool(self.frozen),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CategoricalVocab":
        v = cls(name=str(d["name"]))
        # Replace the default {MISSING: 0, UNK: 1} with the saved map.
        v.token_to_id = {str(k): int(i) for k, i in dict(d["token_to_id"]).items()}
        v.frozen = bool(d.get("frozen", True))
        return v


@dataclass
class NumericScaler:
    """Median-impute + mean/std standardize, with optional log1p.

    ``transform`` returns ``(x_scaled, missingness_indicator)`` so the
    consumer can carry the "was this missing?" bit alongside the imputed
    value. That two-channel encoding matches the old LF preprocessor
    and is what the trainer expects.
    """

    name: str
    log_transform: bool = False
    median: float = 0.0
    mean: float = 0.0
    std: float = 1.0

    def _maybe_log(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(x, 0.0)) if self.log_transform else x

    def fit(self, values: Iterable[Any]) -> "NumericScaler":
        x = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            # Nothing to fit on: keep defaults so transform() always
            # returns finite output even on a fully-missing column.
            return self
        x_t = self._maybe_log(finite)
        self.median = float(np.median(finite))
        self.mean = float(np.mean(x_t))
        # std == 0 happens on a single-valued column; guard so we don't
        # divide by zero in transform.
        self.std = float(np.std(x_t)) or 1.0
        return self

    def transform(self, values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
        x = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        missing = (~np.isfinite(x)).astype(np.float32)
        filled = np.where(np.isfinite(x), x, self.median)
        scaled = (self._maybe_log(filled) - self.mean) / self.std
        return scaled.astype(np.float32), missing

    def transform_one(self, value: object) -> tuple[float, float]:
        s, m = self.transform([value])
        return float(s[0]), float(m[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NumericScaler":
        return cls(
            name=str(d["name"]),
            log_transform=bool(d.get("log_transform", False)),
            median=float(d.get("median", 0.0)),
            mean=float(d.get("mean", 0.0)),
            std=float(d.get("std", 1.0)) or 1.0,
        )


# ---------------------------------------------------------------------------
# Schema -- declares which columns each side uses (mirrors the YAML config)
# ---------------------------------------------------------------------------


@dataclass
class MetadataSchema:
    """Which columns the preprocessor pulls from the metadata CSVs.

    The defaults mirror the columns the old ``latent_factor`` model used
    in production. ``release_date`` is a numeric (years since epoch);
    ``log_params`` is a derived log1p(parameters) -- the preprocessor
    builds it from the ``parameters`` source column automatically.
    """

    subject_categorical: tuple[str, ...] = ("organization", "family", "macro_family")
    subject_numeric: tuple[str, ...] = ("log_params", "release_date")
    benchmark_categorical: tuple[str, ...] = ("topic",)
    benchmark_numeric: tuple[str, ...] = ("benchmark_age",)
    # Named crosses given to ``ExplicitCrossEmbeddings``. Each entry is
    # ``"subject_field__benchmark_field"`` -- the per-side fields must
    # appear in the subject / benchmark categorical lists above.
    explicit_crosses: tuple[str, ...] = ("family__topic", "macro_family__topic")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_categorical": list(self.subject_categorical),
            "subject_numeric": list(self.subject_numeric),
            "benchmark_categorical": list(self.benchmark_categorical),
            "benchmark_numeric": list(self.benchmark_numeric),
            "explicit_crosses": list(self.explicit_crosses),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MetadataSchema":
        def _tup(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(str(x) for x in d.get(key, default))

        return cls(
            subject_categorical=_tup("subject_categorical", ("organization", "family", "macro_family")),
            subject_numeric=_tup("subject_numeric", ("log_params", "release_date")),
            benchmark_categorical=_tup("benchmark_categorical", ("topic",)),
            benchmark_numeric=_tup("benchmark_numeric", ("benchmark_age",)),
            explicit_crosses=_tup("explicit_crosses", ("family__topic", "macro_family__topic")),
        )


# ---------------------------------------------------------------------------
# MetadataPreprocessor: fits vocabs + scalers, holds the joined lookup tables
# ---------------------------------------------------------------------------


def _normalize_model_info(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw model_info dataframe into the columns we expect.

    Recognized source column aliases (training-set CSV from the old
    codabench submission uses ``name`` and ``parameters``):

        name           -> name           (display name; primary join key)
        params         -> parameters
        macro-family   -> macro_family

    Missing values in categorical columns are replaced with NaN so the
    vocab's MISSING bucket fires.
    """
    out = df.copy()
    if "name" not in out.columns:
        for cand in ("model_id", "model", "subject_id", "display_name"):
            if cand in out.columns:
                out = out.rename(columns={cand: "name"})
                break
    if "name" not in out.columns:
        raise KeyError(
            f"model_info must have a 'name' column; got {list(df.columns)}"
        )
    out["name"] = out["name"].astype(str)
    rename = {"params": "parameters", "macro-family": "macro_family"}
    for src, dst in rename.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    for col in ("organization", "family", "macro_family"):
        if col in out.columns:
            out[col] = out[col].astype(str)
            out.loc[out[col].str.lower().isin({"unknown", "nan", "none", ""}), col] = np.nan
    for col in ("parameters", "release_date"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_benchmark_info(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw benchmark_info dataframe into the columns we expect."""
    out = df.copy()
    if "benchmark" not in out.columns:
        for cand in ("benchmark_id", "name"):
            if cand in out.columns:
                out = out.rename(columns={cand: "benchmark"})
                break
    if "benchmark" not in out.columns:
        raise KeyError(
            f"benchmark_info must have a 'benchmark' column; got {list(df.columns)}"
        )
    out["benchmark"] = out["benchmark"].astype(str)
    if "topic" in out.columns:
        out["topic"] = out["topic"].astype(str)
        out.loc[out["topic"].str.lower().isin({"unknown", "nan", "none", ""}), "topic"] = np.nan
    # The old CSV calls the numeric column ``age`` (days since release);
    # we standardize on ``benchmark_age`` to avoid name-shadowing with the
    # subject-side ``release_date``.
    if "age" in out.columns and "benchmark_age" not in out.columns:
        out = out.rename(columns={"age": "benchmark_age"})
    for col in ("benchmark_age", "has_conditions"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class MetadataPreprocessor:
    """Fitted vocabularies + scalers over the metadata CSVs.

    Construct with ``MetadataPreprocessor.fit(model_info_df, benchmark_info_df,
    schema=...)``. The lookup tables (``_model_info``, ``_benchmark_info``)
    are kept on the object so :class:`MetadataIdTables` can join against
    them by display name / benchmark id without re-loading the CSVs.
    """

    schema: MetadataSchema = field(default_factory=MetadataSchema)
    subject_cat_vocabs: dict[str, CategoricalVocab] = field(default_factory=dict)
    subject_num_scalers: dict[str, NumericScaler] = field(default_factory=dict)
    benchmark_cat_vocabs: dict[str, CategoricalVocab] = field(default_factory=dict)
    benchmark_num_scalers: dict[str, NumericScaler] = field(default_factory=dict)

    # Normalized metadata frames. Kept lightweight (no parquet); the
    # join keys are ``name`` (subject side) and ``benchmark`` (bench side).
    _model_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    _benchmark_info: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Per-(name, benchmark) joined frame caches, populated lazily so
    # :meth:`encode_subject_row` and :meth:`encode_benchmark_row` stay
    # O(1) after the first lookup. ``_subject_by_name`` maps a model
    # display name to the normalized model_info row dict;
    # ``_benchmark_by_id`` maps a benchmark id to the normalized
    # benchmark_info row dict.
    _subject_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    _benchmark_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        model_info_df: pd.DataFrame,
        benchmark_info_df: pd.DataFrame,
        *,
        schema: MetadataSchema | None = None,
        min_count: int = 1,
    ) -> "MetadataPreprocessor":
        schema = schema or MetadataSchema()
        mp = cls(schema=schema)

        mi = _normalize_model_info(model_info_df)
        bi = _normalize_benchmark_info(benchmark_info_df)
        mp._model_info = mi
        mp._benchmark_info = bi

        # Subject side
        for col in schema.subject_categorical:
            if col in mi.columns:
                vocab = CategoricalVocab(name=f"subject.{col}")
                vocab.fit(mi[col].tolist(), min_count=min_count)
                mp.subject_cat_vocabs[col] = vocab
        for col in schema.subject_numeric:
            src = "parameters" if col == "log_params" else col
            if src in mi.columns:
                sc = NumericScaler(
                    name=f"subject.{col}",
                    log_transform=(col == "log_params"),
                )
                sc.fit(mi[src].tolist())
                mp.subject_num_scalers[col] = sc

        # Benchmark side
        for col in schema.benchmark_categorical:
            if col in bi.columns:
                vocab = CategoricalVocab(name=f"benchmark.{col}")
                vocab.fit(bi[col].tolist(), min_count=min_count)
                mp.benchmark_cat_vocabs[col] = vocab
        for col in schema.benchmark_numeric:
            if col in bi.columns:
                sc = NumericScaler(name=f"benchmark.{col}", log_transform=False)
                sc.fit(bi[col].tolist())
                mp.benchmark_num_scalers[col] = sc

        # Lookups
        mp._subject_by_name = {
            str(row["name"]): row.to_dict() for _, row in mi.iterrows()
        }
        mp._benchmark_by_id = {
            str(row["benchmark"]): row.to_dict() for _, row in bi.iterrows()
        }
        return mp

    # ------------------------------------------------------------------
    # Single-row encoding (used by the runtime cold-start path)
    # ------------------------------------------------------------------

    def encode_subject(
        self, display_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode one subject by display name.

        Returns
        -------
        cat_ids : int64 array, shape (n_subject_cat_fields,)
        num     : float32 array, shape (n_subject_num_fields,)
        num_missing : float32 array, shape (n_subject_num_fields,)
        """
        info = self._subject_by_name.get(str(display_name).strip(), {})
        cat_ids = np.zeros(len(self.schema.subject_categorical), dtype=np.int64)
        for j, col in enumerate(self.schema.subject_categorical):
            vocab = self.subject_cat_vocabs.get(col)
            if vocab is None:
                continue
            cat_ids[j] = vocab.encode_one(info.get(col))
        n_num = len(self.schema.subject_numeric)
        num = np.zeros(n_num, dtype=np.float32)
        miss = np.ones(n_num, dtype=np.float32)
        for j, col in enumerate(self.schema.subject_numeric):
            sc = self.subject_num_scalers.get(col)
            if sc is None:
                continue
            src = info.get("parameters") if col == "log_params" else info.get(col)
            x, m = sc.transform_one(src)
            num[j] = x
            miss[j] = m
        return cat_ids, num, miss

    def encode_benchmark(
        self, benchmark: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode one benchmark by id. Mirrors :meth:`encode_subject`."""
        info = self._benchmark_by_id.get(str(benchmark).strip(), {})
        cat_ids = np.zeros(len(self.schema.benchmark_categorical), dtype=np.int64)
        for j, col in enumerate(self.schema.benchmark_categorical):
            vocab = self.benchmark_cat_vocabs.get(col)
            if vocab is None:
                continue
            cat_ids[j] = vocab.encode_one(info.get(col))
        n_num = len(self.schema.benchmark_numeric)
        num = np.zeros(n_num, dtype=np.float32)
        miss = np.ones(n_num, dtype=np.float32)
        for j, col in enumerate(self.schema.benchmark_numeric):
            sc = self.benchmark_num_scalers.get(col)
            if sc is None:
                continue
            x, m = sc.transform_one(info.get(col))
            num[j] = x
            miss[j] = m
        return cat_ids, num, miss

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "subject_cat_vocabs": {
                k: v.to_dict() for k, v in self.subject_cat_vocabs.items()
            },
            "subject_num_scalers": {
                k: v.to_dict() for k, v in self.subject_num_scalers.items()
            },
            "benchmark_cat_vocabs": {
                k: v.to_dict() for k, v in self.benchmark_cat_vocabs.items()
            },
            "benchmark_num_scalers": {
                k: v.to_dict() for k, v in self.benchmark_num_scalers.items()
            },
            # We ship the normalized lookup tables as records so the
            # runtime can rebuild the join without re-loading the CSV.
            # Each table is small (953 rows model_info, ~17 rows
            # benchmark_info) so the on-disk size is negligible.
            "model_info_records": [
                {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
                for row in self._model_info.to_dict(orient="records")
            ],
            "benchmark_info_records": [
                {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
                for row in self._benchmark_info.to_dict(orient="records")
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MetadataPreprocessor":
        mp = cls(schema=MetadataSchema.from_dict(d.get("schema", {})))
        mp.subject_cat_vocabs = {
            str(k): CategoricalVocab.from_dict(v)
            for k, v in dict(d.get("subject_cat_vocabs", {})).items()
        }
        mp.subject_num_scalers = {
            str(k): NumericScaler.from_dict(v)
            for k, v in dict(d.get("subject_num_scalers", {})).items()
        }
        mp.benchmark_cat_vocabs = {
            str(k): CategoricalVocab.from_dict(v)
            for k, v in dict(d.get("benchmark_cat_vocabs", {})).items()
        }
        mp.benchmark_num_scalers = {
            str(k): NumericScaler.from_dict(v)
            for k, v in dict(d.get("benchmark_num_scalers", {})).items()
        }
        mi_records = list(d.get("model_info_records", []))
        bi_records = list(d.get("benchmark_info_records", []))
        mp._model_info = pd.DataFrame.from_records(mi_records) if mi_records else pd.DataFrame()
        mp._benchmark_info = pd.DataFrame.from_records(bi_records) if bi_records else pd.DataFrame()
        if not mp._model_info.empty:
            mp._subject_by_name = {
                str(row["name"]): row.to_dict() for _, row in mp._model_info.iterrows()
            }
        if not mp._benchmark_info.empty:
            mp._benchmark_by_id = {
                str(row["benchmark"]): row.to_dict()
                for _, row in mp._benchmark_info.iterrows()
            }
        return mp


# ---------------------------------------------------------------------------
# Per-id tensor tables (built once at fit time; live on the model as buffers)
# ---------------------------------------------------------------------------


@dataclass
class MetadataIdTables:
    """Per-id metadata tensors indexed by ``subject_idx`` / ``bc_idx``.

    Shapes:
        subject_cat_ids : LongTensor[n_subjects, n_subject_cat_fields]
        subject_num     : FloatTensor[n_subjects, 2 * n_subject_num_fields]
                          (scaled + missingness indicator interleaved per field)
        bc_cat_ids      : LongTensor[n_bc, n_bench_cat_fields]
        bc_num          : FloatTensor[n_bc, 2 * n_bench_num_fields]
        cross_ids       : LongTensor[n_subjects, n_bc, n_explicit_crosses]
                          OR materialized lazily as needed
                          (not stored; computed per-row from the two
                          ``*_cat_ids`` tables at forward time)

    Index 0 is always reserved for the UNK subject / UNK bc. We fill row
    0 with MISSING tokens / zero numerics so an UNK lookup is the same
    as a "I have no metadata for this entity" lookup.
    """

    subject_cat_ids: torch.Tensor
    subject_num: torch.Tensor
    bc_cat_ids: torch.Tensor
    bc_num: torch.Tensor

    # Per-field cardinalities (used by the model to size embedding tables)
    subject_cat_cardinalities: tuple[int, ...] = ()
    benchmark_cat_cardinalities: tuple[int, ...] = ()
    subject_num_dim: int = 0       # = 2 * n_subject_num_fields
    benchmark_num_dim: int = 0     # = 2 * n_bench_num_fields

    @property
    def n_subjects(self) -> int:
        return int(self.subject_cat_ids.shape[0])

    @property
    def n_bc(self) -> int:
        return int(self.bc_cat_ids.shape[0])


def build_metadata_id_tables(
    *,
    preprocessor: MetadataPreprocessor,
    subject_to_id: Mapping[str, int],
    bc_to_id: Mapping[str, int],
    subject_content_by_key: Mapping[str, str],
) -> MetadataIdTables:
    """Materialize per-(subject_idx, bc_idx) metadata tensors.

    Parameters
    ----------
    preprocessor : the fitted MetadataPreprocessor.
    subject_to_id : ``Indexer.subject_to_id`` from ``src.models``. Maps
        ``subject_key = sha256(subject_content)`` -> integer idx. Index 0
        is the UNK subject.
    bc_to_id : ``Indexer.bc_to_id`` (``"{benchmark}::{condition}"`` ->
        idx). Index 0 is the UNK bc.
    subject_content_by_key : maps ``subject_key`` -> raw
        ``subject_content`` string (from the training dataframe). We
        extract the display name to join against ``model_info``.

    Returns
    -------
    MetadataIdTables with row 0 = UNK = MISSING for every field.
    """
    schema = preprocessor.schema
    n_sub = len(subject_to_id)
    n_bc = len(bc_to_id)
    n_sub_cat = len(schema.subject_categorical)
    n_bc_cat = len(schema.benchmark_categorical)
    n_sub_num = len(schema.subject_numeric)
    n_bc_num = len(schema.benchmark_numeric)

    sub_cat = np.zeros((n_sub, max(1, n_sub_cat)), dtype=np.int64)
    sub_num = np.zeros((n_sub, 2 * max(1, n_sub_num)), dtype=np.float32)
    bc_cat = np.zeros((n_bc, max(1, n_bc_cat)), dtype=np.int64)
    bc_num = np.zeros((n_bc, 2 * max(1, n_bc_num)), dtype=np.float32)

    # Row 0 (UNK) defaults: zeros = MISSING categorical token, zero
    # numeric, missingness=1. We initialize the missingness channels to
    # 1 across the board first, then overwrite where we actually have
    # data. The categorical zero values are already MISSING ids by
    # construction of CategoricalVocab.
    if n_sub_num > 0:
        for j in range(n_sub_num):
            sub_num[:, 2 * j + 1] = 1.0
    if n_bc_num > 0:
        for j in range(n_bc_num):
            bc_num[:, 2 * j + 1] = 1.0

    # Subject side: iterate the subject_to_id map.
    for key, idx in subject_to_id.items():
        if idx == 0 or key == "<unk>":
            continue
        content = subject_content_by_key.get(key, "")
        name = extract_display_name(content)
        cat_ids, num_x, num_m = preprocessor.encode_subject(name)
        if n_sub_cat > 0:
            sub_cat[idx, :n_sub_cat] = cat_ids[:n_sub_cat]
        if n_sub_num > 0:
            for j in range(n_sub_num):
                sub_num[idx, 2 * j] = num_x[j]
                sub_num[idx, 2 * j + 1] = num_m[j]

    # Benchmark-condition side: parse bc_key = "{benchmark}::{condition}".
    for key, idx in bc_to_id.items():
        if idx == 0 or key == "<unk>":
            continue
        if "::" in key:
            benchmark, _condition = key.split("::", 1)
        else:
            benchmark = key
        cat_ids, num_x, num_m = preprocessor.encode_benchmark(benchmark)
        if n_bc_cat > 0:
            bc_cat[idx, :n_bc_cat] = cat_ids[:n_bc_cat]
        if n_bc_num > 0:
            for j in range(n_bc_num):
                bc_num[idx, 2 * j] = num_x[j]
                bc_num[idx, 2 * j + 1] = num_m[j]

    subject_cat_cardinalities = tuple(
        preprocessor.subject_cat_vocabs[col].n_tokens
        if col in preprocessor.subject_cat_vocabs
        else 2
        for col in schema.subject_categorical
    )
    benchmark_cat_cardinalities = tuple(
        preprocessor.benchmark_cat_vocabs[col].n_tokens
        if col in preprocessor.benchmark_cat_vocabs
        else 2
        for col in schema.benchmark_categorical
    )

    return MetadataIdTables(
        subject_cat_ids=torch.from_numpy(sub_cat),
        subject_num=torch.from_numpy(sub_num),
        bc_cat_ids=torch.from_numpy(bc_cat),
        bc_num=torch.from_numpy(bc_num),
        subject_cat_cardinalities=subject_cat_cardinalities,
        benchmark_cat_cardinalities=benchmark_cat_cardinalities,
        subject_num_dim=2 * n_sub_num,
        benchmark_num_dim=2 * n_bc_num,
    )


# ---------------------------------------------------------------------------
# Towers / FM / Explicit cross modules
# ---------------------------------------------------------------------------


def _auto_emb_dim(card: int, max_dim: int = 16) -> int:
    """min(max_dim, ceil(1.6 * card^0.56)), floor 4 -- old LF heuristic."""
    raw = max(4, int(round(1.6 * max(2, int(card)) ** 0.56)))
    return int(min(int(max_dim), raw))


class _PerFieldCategoricalEmbeddings(nn.Module):
    """Concatenates per-field categorical embeddings into a single vector.

    Each field gets its own ``nn.Embedding`` sized via the standard
    cardinality-based heuristic. ``forward(cat_ids)`` takes a
    ``LongTensor[B, n_fields]`` and returns ``FloatTensor[B,
    sum_of_embedding_dims]``.
    """

    def __init__(self, cardinalities: Sequence[int], max_emb_dim: int = 16):
        super().__init__()
        self.cardinalities = tuple(int(c) for c in cardinalities)
        dims: list[int] = []
        embs: list[nn.Embedding] = []
        for c in self.cardinalities:
            d = _auto_emb_dim(c, max_dim=max_emb_dim)
            dims.append(d)
            e = nn.Embedding(max(1, c), d, padding_idx=0)
            # MISSING (idx 0) initialized to zero so a fully-missing
            # row contributes nothing on the linear path.
            nn.init.normal_(e.weight, std=0.05)
            with torch.no_grad():
                e.weight[0].zero_()
            embs.append(e)
        self.embs = nn.ModuleList(embs)
        self.dims = tuple(dims)
        self.total_dim = int(sum(dims))

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        if self.total_dim == 0 or cat_ids.shape[-1] == 0:
            return torch.zeros(
                (cat_ids.shape[0], 0), device=cat_ids.device, dtype=torch.float32
            )
        parts = [self.embs[i](cat_ids[:, i]) for i in range(len(self.embs))]
        return torch.cat(parts, dim=-1)


class MetaTower(nn.Module):
    """MLP that maps a concatenated metadata vector to ``(scalar, k-vec)``.

    Output heads are zero-initialized: at construction time the tower
    contributes exactly 0 to the logit, so a freshly built metadata
    model is bit-identical to its non-metadata sibling.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        k: int,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        if in_dim <= 0:
            # Degenerate case: no metadata fields configured. Build a
            # no-op tower that always outputs zero (k+1) so the model
            # plumbing stays valid.
            self.is_noop = True
            self.k = int(k)
            self.scalar_bias = nn.Parameter(torch.zeros(1))
            return
        self.is_noop = False
        self.k = int(k)
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        d = in_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_scalar = nn.Linear(d, 1)
        self.head_vec = nn.Linear(d, max(1, int(k)))
        # Zero-init the output heads so the tower starts as a no-op.
        nn.init.zeros_(self.head_scalar.weight)
        nn.init.zeros_(self.head_scalar.bias)
        nn.init.zeros_(self.head_vec.weight)
        nn.init.zeros_(self.head_vec.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_noop or x.shape[-1] == 0:
            b = x.shape[0]
            return (
                torch.zeros(b, device=x.device, dtype=torch.float32),
                torch.zeros(b, max(1, self.k), device=x.device, dtype=torch.float32),
            )
        h = self.trunk(x)
        return self.head_scalar(h).squeeze(-1), self.head_vec(h)


class FactorizationMachineCross(nn.Module):
    """Pairwise-interaction head over (subject, bench-condition) embeddings.

    Implementation:
        Concatenate the per-field categorical embeddings on both sides
        into a single ``(B, n_fields_total)`` *bag* by stacking each
        field embedding as a separate latent vector in a shared FM
        space. Then the standard FM pairwise sum identity gives

            score = 0.5 * ( sum_i v_i )^2 - sum_i v_i^2

        applied along the field axis, summed over the FM latent dim.

    The trick: every per-field embedding is first projected through a
    small ``Linear(d_field -> d_fm)`` into a shared FM space. The
    pairwise interactions are then between all per-field FM-space
    vectors, including subject<->subject and bench<->bench (cheap
    parameter-wise but architecturally subsumed by the cross terms we
    actually care about: subject_field <-> bench_field).

    The output is a **single additive scalar logit**. Output head is
    zero-initialized so the channel boots up inert.
    """

    def __init__(
        self,
        subject_field_dims: Sequence[int],
        benchmark_field_dims: Sequence[int],
        d_fm: int,
    ):
        super().__init__()
        self.d_fm = int(d_fm)
        sd = list(int(x) for x in subject_field_dims)
        bd = list(int(x) for x in benchmark_field_dims)
        self.subj_projs = nn.ModuleList([nn.Linear(d, self.d_fm) for d in sd])
        self.bench_projs = nn.ModuleList([nn.Linear(d, self.d_fm) for d in bd])
        for m in self.subj_projs:
            nn.init.normal_(m.weight, std=0.05)
            nn.init.zeros_(m.bias)
        for m in self.bench_projs:
            nn.init.normal_(m.weight, std=0.05)
            nn.init.zeros_(m.bias)
        # Output projection from FM latent dim -> scalar logit. Zero-init
        # so the head starts at exactly 0 and learns from gradients only.
        self.head = nn.Linear(self.d_fm, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        subj_field_embs: Sequence[torch.Tensor],
        bench_field_embs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not subj_field_embs and not bench_field_embs:
            return torch.zeros((0,), dtype=torch.float32)
        bsz = (
            subj_field_embs[0].shape[0]
            if subj_field_embs
            else bench_field_embs[0].shape[0]
        )
        # Project each field into the FM space and stack:
        #   v: (B, n_fields_total, d_fm).
        vs: list[torch.Tensor] = []
        for proj, e in zip(self.subj_projs, subj_field_embs):
            vs.append(proj(e))
        for proj, e in zip(self.bench_projs, bench_field_embs):
            vs.append(proj(e))
        if not vs:
            return torch.zeros(bsz, dtype=torch.float32, device=self.head.weight.device)
        v = torch.stack(vs, dim=1)               # (B, F, d_fm)
        sum_v = v.sum(dim=1)                     # (B, d_fm)
        sum_v_sq = (v * v).sum(dim=1)            # (B, d_fm)
        # Standard FM identity: sum_{i<j} <v_i, v_j>
        #   = 0.5 * ( (sum v)^2 - sum (v^2) )
        interactions = 0.5 * (sum_v * sum_v - sum_v_sq)   # (B, d_fm)
        return self.head(interactions).squeeze(-1)


class ExplicitCrossEmbeddings(nn.Module):
    """Hand-picked categorical cross embeddings.

    Each cross is configured as ``"subject_field__benchmark_field"``. The
    cross id at runtime is computed as

        cross_id = (subj_cat_id_for_field * card_bench_field) + bench_cat_id_for_field

    so the lookup table has size ``card_subj * card_bench``. Output is a
    single zero-initialized scalar logit summed over crosses.
    """

    def __init__(
        self,
        crosses: Sequence[str],
        schema: MetadataSchema,
        subject_cardinalities: Sequence[int],
        benchmark_cardinalities: Sequence[int],
        emb_dim: int = 8,
    ):
        super().__init__()
        self.crosses: list[tuple[str, str, int, int, int, int]] = []
        # Each tuple: (subj_field_name, bench_field_name,
        #              subj_field_idx, bench_field_idx,
        #              subj_card, bench_card)
        sub_index = {c: i for i, c in enumerate(schema.subject_categorical)}
        bench_index = {c: i for i, c in enumerate(schema.benchmark_categorical)}
        for spec in crosses:
            if "__" not in spec:
                continue
            sf, bf = spec.split("__", 1)
            if sf not in sub_index or bf not in bench_index:
                continue
            si = sub_index[sf]
            bi = bench_index[bf]
            sc = int(subject_cardinalities[si])
            bc = int(benchmark_cardinalities[bi])
            self.crosses.append((sf, bf, si, bi, sc, bc))
        self.emb_dim = int(emb_dim)
        tables = []
        for _, _, _, _, sc, bc in self.crosses:
            # +1 for an explicit OOV/UNK row at index 0 so any 0 in
            # either side maps to a benign "no info" embedding.
            n = max(1, sc * bc) + 1
            t = nn.Embedding(n, self.emb_dim, padding_idx=0)
            nn.init.normal_(t.weight, std=0.05)
            with torch.no_grad():
                t.weight[0].zero_()
            tables.append(t)
        self.tables = nn.ModuleList(tables)
        if self.tables:
            self.head = nn.Linear(self.emb_dim, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = None

    @property
    def has_any(self) -> bool:
        return bool(self.tables)

    def forward(
        self, subj_cat_ids: torch.Tensor, bench_cat_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.tables or self.head is None:
            bsz = subj_cat_ids.shape[0] if subj_cat_ids.dim() > 0 else 0
            return torch.zeros(bsz, dtype=torch.float32, device=subj_cat_ids.device)
        per_cross: list[torch.Tensor] = []
        for (_, _, si, bi, sc, bc), table in zip(self.crosses, self.tables):
            s = subj_cat_ids[:, si].long()
            b = bench_cat_ids[:, bi].long()
            # MISSING (0) on either side -> route to OOV row 0.
            is_oov = (s == 0) | (b == 0)
            cross_id = s * bc + b + 1     # +1 leaves row 0 as the OOV row
            cross_id = torch.where(
                is_oov,
                torch.zeros_like(cross_id),
                cross_id.clamp(min=1, max=table.num_embeddings - 1),
            )
            per_cross.append(table(cross_id))
        # Sum the per-cross embeddings, then project to scalar. Summing
        # (rather than concatenating + wide linear) keeps the output
        # parameter count constant w.r.t. the number of crosses, which
        # is what we want -- each cross gets to express itself through
        # its own embedding table, not its own output row.
        merged = torch.stack(per_cross, dim=1).sum(dim=1)   # (B, emb_dim)
        return self.head(merged).squeeze(-1)


__all__ = [
    "CategoricalVocab",
    "ExplicitCrossEmbeddings",
    "FactorizationMachineCross",
    "MetadataIdTables",
    "MetadataPreprocessor",
    "MetadataSchema",
    "MetaTower",
    "NumericScaler",
    "TOKEN_MISSING",
    "TOKEN_UNK",
    "_PerFieldCategoricalEmbeddings",
    "_auto_emb_dim",
    "build_metadata_id_tables",
    "extract_display_name",
    "normalize_condition_token",
]
