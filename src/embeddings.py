"""Transformer embeddings with caching, length bucketing, and Flash Attention 2.

Design goals:

- Cache **unique** texts. The dataset has many subjects answering many items,
  so the unique-text set is much smaller than the row set. Dedup is verified
  loudly at the top of every encoding pass.

- Length bucketing. Texts are sorted by tokenized length before forming
  batches so each batch pads to its own batch max (not the global max),
  cutting wasted compute on short sequences dramatically.

- Flash Attention 2 with a graceful fallback to SDPA if the wheel isn't
  installed or the GPU doesn't support it.

- Data-driven ``max_length``. We tokenize once on CPU to compute the 99th
  percentile token length, round up to the next multiple of 64, and use
  that as the encoder cap. The chosen value is persisted to ``meta.json``
  so subsequent runs reuse it (and so the runtime knows what to expect).

- OOM-resilient batch size. The encoder forward pass is wrapped in a
  try/except on ``torch.cuda.OutOfMemoryError``; on OOM we halve the
  batch size, clear the cache, and retry. The final successful batch size
  is persisted to ``meta.json``.

- Qwen3-Embedding instruction prefix. When the encoder id matches
  ``Qwen3-Embedding`` we automatically wrap inputs in the recommended
  instruction format. The exact prefix is persisted so the runtime
  ``model.py`` can verify a match.

Cache layout (one directory per encoder, slug-safe):

    {cache_dir}/{encoder_slug}/
        meta.json                # encoder id, dim, dtype, max_length, prefix, batch_size, content_hash
        items.parquet            # columns: item_key, embedding (list[float16])
        subjects.parquet         # columns: subject_key, embedding (list[float16])
        encoding_log.json        # timing per phase, OOM events, counts
"""

from __future__ import annotations

import dataclasses
import getpass
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, *args, **kwargs):
        return x

LOG = logging.getLogger("embeddings")


# ---------------------------------------------------------------------------
# HF token plumbing -- never print, never persist on disk
# ---------------------------------------------------------------------------


def _try_colab_userdata() -> str | None:
    """Best-effort fetch of HF_TOKEN from the Colab userdata secret store."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        token = userdata.get("HF_TOKEN")
    except Exception:
        return None
    if not token:
        return None
    token = str(token).strip()
    return token or None


def _try_secret_manager() -> str | None:
    """Best-effort fetch of HF_TOKEN from Google Secret Manager."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project:
        return None
    try:
        from google.cloud import secretmanager
    except Exception:
        return None
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/HF_TOKEN/versions/latest"
        resp = client.access_secret_version(request={"name": name})
        token = resp.payload.data.decode("utf-8").strip()
        return token or None
    except Exception:
        return None


def resolve_hf_token(*, interactive: bool = True) -> str | None:
    """Resolve the HF token without ever printing or persisting it."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token = _try_colab_userdata()
    if token:
        return token
    token = _try_secret_manager()
    if token:
        return token
    if not interactive:
        return None
    try:
        token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    except Exception:
        return None
    return token or None


def login_huggingface(token: str | None) -> bool:
    """Call huggingface_hub.login() if we have a token. Never logs the token."""
    if not token:
        LOG.warning("No HF_TOKEN available; private repos will fail to load.")
        return False
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        LOG.info("huggingface_hub.login: OK")
        return True
    except Exception as exc:  # noqa: BLE001
        LOG.warning("huggingface_hub.login failed: %s", type(exc).__name__)
        return False


# ---------------------------------------------------------------------------
# Flash Attention availability
# ---------------------------------------------------------------------------


def verify_flash_attention(requested: bool = True) -> tuple[bool, str]:
    """Verify that Flash Attention 2 is actually usable.

    The PyPI ``flash-attn`` wheel ships a Python shim plus a CUDA kernel
    extension (``flash_attn_2_cuda``). The Python shim can import even when
    the CUDA extension is missing or ABI-mismatched against the running
    torch build; the kernel binding only blows up later, inside the first
    forward pass. We probe both eagerly here so callers can downgrade to
    SDPA before constructing the encoder.

    Returns a ``(active, message)`` pair so callers can log a clear status
    line without re-deriving the failure mode.
    """
    if not requested:
        return False, "disabled in config"
    try:
        import flash_attn  # type: ignore
        import flash_attn_2_cuda  # type: ignore  # noqa: F401  (kernel binding)
        return True, f"flash_attn=={flash_attn.__version__}"
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"unavailable ({type(exc).__name__}: {exc}); falling back to SDPA",
        )


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def encoder_slug(model_id: str) -> str:
    """Stable, filesystem-safe directory name for an encoder."""
    safe = model_id.replace("/", "__")
    return safe[:200]


def _encoder_dir(cache_dir: str | os.PathLike, model_id: str) -> Path:
    base = Path(cache_dir) / encoder_slug(model_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Text templates
# ---------------------------------------------------------------------------


def item_only_text(item_content: str, *, prefix: str = "") -> str:
    return f"{prefix}{item_content}" if prefix else str(item_content)


def item_contextual_text(
    benchmark: str, condition: str, item_content: str, *, prefix: str = ""
) -> str:
    body = (
        f"Benchmark: {benchmark}\n"
        f"Condition: {condition}\n"
        f"Item: {item_content}"
    )
    return f"{prefix}{body}" if prefix else body


def subject_text(subject_content: str, *, prefix: str = "") -> str:
    return f"{prefix}{subject_content}" if prefix else str(subject_content)


def qwen3_prefix(instruction: str) -> str:
    """The standard Qwen3-Embedding ``Instruct: ... Query: `` envelope."""
    if not instruction:
        return ""
    return f"Instruct: {instruction}\nQuery: "


def is_qwen3_embedding(model_id: str) -> bool:
    return "Qwen3-Embedding" in (model_id or "")


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _last_token_pool(last_hidden, attention_mask):
    """Last non-pad token pool. Used by Qwen3-Embedding and e5-mistral."""
    import torch

    seq_lens = attention_mask.sum(dim=1) - 1
    seq_lens = seq_lens.clamp(min=0)
    batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch, seq_lens]


# ---------------------------------------------------------------------------
# Config / stats dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStats:
    """Diagnostics produced during an encoding pass."""

    n_texts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    token_lengths: list[int] = field(default_factory=list)
    truncated: int = 0
    nan_or_inf: int = 0
    zero_norm: int = 0
    per_benchmark_truncation: dict[str, int] = field(default_factory=dict)
    per_benchmark_total: dict[str, int] = field(default_factory=dict)

    def report(self) -> dict:
        if not self.token_lengths:
            return {
                "n_texts": self.n_texts,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "truncation_rate": 0.0,
                "nan_or_inf": self.nan_or_inf,
                "zero_norm": self.zero_norm,
            }
        arr = np.asarray(self.token_lengths)
        rate = self.truncated / max(1, len(arr))
        per_bench_rate = {
            b: (self.per_benchmark_truncation[b] / max(1, self.per_benchmark_total[b]))
            for b in self.per_benchmark_total
        }
        return {
            "n_texts": self.n_texts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "p50_tokens": float(np.quantile(arr, 0.5)),
            "p90_tokens": float(np.quantile(arr, 0.9)),
            "p95_tokens": float(np.quantile(arr, 0.95)),
            "p99_tokens": float(np.quantile(arr, 0.99)),
            "max_tokens": int(arr.max()),
            "truncation_rate": float(rate),
            "nan_or_inf": self.nan_or_inf,
            "zero_norm": self.zero_norm,
            "per_benchmark_truncation_rate": per_bench_rate,
        }


@dataclass
class EncoderConfig:
    model_id: str
    max_length: int | None = None              # None -> auto from data
    max_length_floor: int = 256
    max_length_ceiling: int = 4096
    batch_size: int = 64
    batch_size_fallback: int = 16
    pooling: str = "mean"
    bf16: bool = True
    use_flash_attention: bool = True
    qwen3_instruction: str = (
        "Represent this AI evaluation context for difficulty prediction"
    )
    query_prefix: str = ""
    passage_prefix: str = ""
    use_contextual_item_text: bool = True
    cache_dir: str = "artifacts/embeddings"
    trust_remote_code: bool = False
    use_random_embeddings: bool = False
    random_embedding_dim: int = 256
    # Sample size for the data-driven max_length percentile scan. Smaller =
    # faster startup; the 99th percentile is stable on a few-thousand sample.
    max_length_sample_size: int = 2000


# ---------------------------------------------------------------------------
# Dedup verification
# ---------------------------------------------------------------------------


def assert_deduplicated(
    keys: list[str], *, kind: str, log: logging.Logger | None = None
) -> dict:
    """Assert the key list is deduplicated. Raises AssertionError if not.

    Dedup before encoding is non-negotiable: forwarding duplicate texts is
    typically the largest source of wasted encoder time. The check is cheap
    (a set comparison) and the error is loud.
    """
    log = log or LOG
    n_total = len(keys)
    n_unique = len(set(keys))
    ratio = n_unique / max(1, n_total)
    log.info(
        "dedup[%s]: n_total=%d n_unique=%d ratio=%.4f",
        kind,
        n_total,
        n_unique,
        ratio,
    )
    if n_unique != n_total:
        raise AssertionError(
            f"{kind} input is not deduplicated: "
            f"{n_total} rows but only {n_unique} unique keys. "
            f"Dedup BEFORE calling the encoder."
        )
    return {"n_total": n_total, "n_unique": n_unique, "dedup_ratio": ratio}


# ---------------------------------------------------------------------------
# Cache + meta helpers
# ---------------------------------------------------------------------------


def content_hash_for_items(items: list[tuple[str, str]]) -> str:
    """Stable SHA256 of a list of (key, text) pairs.

    Used to detect when underlying text changes between runs so we can
    invalidate the cache (and Drive cache) automatically.
    """
    h = hashlib.sha256()
    for k, t in sorted(items, key=lambda x: x[0]):
        h.update(k.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    return h.hexdigest()


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _write_log(log_path: Path, payload: dict) -> None:
    log_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_parquet_cache(path: Path) -> dict[str, np.ndarray]:
    """Read a {key, embedding} parquet into a {key: float32 vector} dict."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    out: dict[str, np.ndarray] = {}
    for k, v in zip(df.iloc[:, 0].astype(str).tolist(), df["embedding"].tolist()):
        arr = np.asarray(v, dtype=np.float32)
        out[k] = arr
    return out


def _write_parquet_cache(
    path: Path, lookup: dict[str, np.ndarray], *, key_col: str
) -> None:
    """Write a {key: vector} dict as a parquet (vectors are stored as fp16)."""
    if not lookup:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame({key_col: pd.Series(dtype=str), "embedding": pd.Series(dtype=object)})
        empty.to_parquet(path, index=False)
        return
    keys = sorted(lookup.keys())
    embs = [np.asarray(lookup[k], dtype=np.float16).tolist() for k in keys]
    df = pd.DataFrame({key_col: keys, "embedding": embs})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------


def _tokenize_lengths(
    tokenizer, texts: list[str], max_length: int
) -> np.ndarray:
    """Tokenize once on CPU to get the per-text token length (clipped).

    Uses a single batched call so the fast tokenizer can parallelize across
    Rust threads. Falls back to a per-text loop only if the batched path
    raises (which can happen with some custom slow tokenizers).
    """
    if not texts:
        return np.zeros((0,), dtype=np.int64)
    try:
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return np.asarray([len(x) for x in enc["input_ids"]], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "Batched length-tokenization failed (%s: %s); falling back to per-text loop",
            type(exc).__name__,
            exc,
        )
        out = np.zeros((len(texts),), dtype=np.int64)
        for i, t in enumerate(tqdm(texts, desc="length-scan", unit="txt", leave=False)):
            ids = tokenizer.encode(
                t, add_special_tokens=True, truncation=True, max_length=max_length
            )
            out[i] = len(ids)
        return out


def choose_max_length(
    raw_lengths: np.ndarray,
    *,
    floor: int,
    ceiling: int,
    quantile: float = 0.99,
    multiple_of: int = 64,
) -> int:
    """Pick a data-driven max_length: round-up of the requested quantile."""
    if raw_lengths.size == 0:
        return max(floor, multiple_of)
    q = int(np.ceil(float(np.quantile(raw_lengths, quantile))))
    q = max(floor, q)
    q = min(ceiling, q)
    # round up to the next multiple of `multiple_of` for kernel-friendly padding
    rounded = int(math.ceil(q / multiple_of) * multiple_of)
    return max(rounded, multiple_of)


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------


class TransformerEmbedder:
    """Length-bucketed HF encoder with parquet cache, FA2, and OOM retry."""

    def __init__(self, cfg: EncoderConfig, *, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._tok = None
        self._model = None
        self._hidden_size: int | None = None
        self._attn_impl: str = "sdpa"
        self._oom_events: list[dict] = []
        self.stats = EmbeddingStats()
        self.base = _encoder_dir(cfg.cache_dir, cfg.model_id)
        self.items_path = self.base / "items.parquet"
        self.subjects_path = self.base / "subjects.parquet"
        self.meta_path = self.base / "meta.json"
        self.log_path = self.base / "encoding_log.json"
        # Pre-loaded caches (filled in lazily; key -> np.float32 vector).
        self._item_lookup: dict[str, np.ndarray] = {}
        self._subject_lookup: dict[str, np.ndarray] = {}
        # Effective max_length / batch_size are decided just-in-time on the
        # first encoding pass and persisted to meta.json.
        self._effective_max_length: int | None = None
        self._effective_batch_size: int = int(cfg.batch_size)

    # ------------------------------------------------------------------ init

    def _resolve_query_prefix(self) -> str:
        if is_qwen3_embedding(self.cfg.model_id) and self.cfg.qwen3_instruction:
            return qwen3_prefix(self.cfg.qwen3_instruction)
        return self.cfg.query_prefix or ""

    def _resolve_passage_prefix(self) -> str:
        if is_qwen3_embedding(self.cfg.model_id) and self.cfg.qwen3_instruction:
            return qwen3_prefix(self.cfg.qwen3_instruction)
        return self.cfg.passage_prefix or ""

    def _write_meta_snapshot(
        self,
        *,
        content_hash: str | None = None,
        n_items: int | None = None,
        n_subjects: int | None = None,
    ) -> None:
        existing = _read_meta(self.meta_path)
        meta = {
            "model_id": self.cfg.model_id,
            "dim": int(self.embedding_dim),
            "dtype": "float16",
            "max_length": int(
                self._effective_max_length or (self.cfg.max_length or 0) or 0
            ),
            "batch_size": int(self._effective_batch_size),
            "pooling": self.cfg.pooling,
            "bf16": self.cfg.bf16,
            "use_flash_attention": self.cfg.use_flash_attention,
            "attn_implementation": self._attn_impl,
            "qwen3_instruction": (
                self.cfg.qwen3_instruction
                if is_qwen3_embedding(self.cfg.model_id)
                else ""
            ),
            "query_prefix": self._resolve_query_prefix(),
            "passage_prefix": self._resolve_passage_prefix(),
            "use_contextual_item_text": self.cfg.use_contextual_item_text,
            "use_random_embeddings": self.cfg.use_random_embeddings,
        }
        if content_hash is not None:
            meta["content_hash"] = content_hash
        elif "content_hash" in existing:
            meta["content_hash"] = existing["content_hash"]
        if n_items is not None:
            meta["n_items"] = int(n_items)
        elif "n_items" in existing:
            meta["n_items"] = existing["n_items"]
        if n_subjects is not None:
            meta["n_subjects"] = int(n_subjects)
        elif "n_subjects" in existing:
            meta["n_subjects"] = existing["n_subjects"]
        _write_meta(self.meta_path, meta)

    def _load(self):
        if self._model is not None or self.cfg.use_random_embeddings:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        LOG.info("Loading tokenizer %s (use_fast=True)", self.cfg.model_id)
        self._tok = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
            use_fast=True,
        )
        tok_kind = type(self._tok).__name__
        if "Fast" not in tok_kind:
            LOG.warning(
                "Tokenizer is the slow Python variant (%s). The data-driven "
                "max_length scan and per-batch tokenization will be much slower. "
                "Consider `pip install -U tokenizers` or pinning max_length via "
                "the config to skip the scan.",
                tok_kind,
            )
        else:
            LOG.info("Tokenizer loaded: %s", tok_kind)

        dtype = self._pick_dtype()
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        loaded = False
        if self.cfg.use_flash_attention and self.device.startswith("cuda"):
            try:
                LOG.info("Loading encoder %s with Flash Attention 2", self.cfg.model_id)
                self._model = AutoModel.from_pretrained(
                    self.cfg.model_id,
                    attn_implementation="flash_attention_2",
                    **kwargs,
                )
                self._attn_impl = "flash_attention_2"
                loaded = True
                LOG.info("Encoder loaded with Flash Attention 2")
            except (ImportError, ValueError, RuntimeError) as exc:
                LOG.warning(
                    "Flash Attention 2 unavailable (%s: %s); falling back to SDPA",
                    type(exc).__name__,
                    exc,
                )
        if not loaded:
            LOG.info("Loading encoder %s with SDPA", self.cfg.model_id)
            try:
                self._model = AutoModel.from_pretrained(
                    self.cfg.model_id, attn_implementation="sdpa", **kwargs
                )
                self._attn_impl = "sdpa"
            except (ValueError, TypeError):
                # Older transformers might not accept attn_implementation kwarg.
                self._model = AutoModel.from_pretrained(self.cfg.model_id, **kwargs)
                self._attn_impl = "default"
        self._model.eval()
        LOG.info("Moving encoder to %s ...", self.device)
        t0 = time.time()
        self._model.to(self.device)
        LOG.info("Encoder ready on %s (%.1fs)", self.device, time.time() - t0)
        self._hidden_size = int(getattr(self._model.config, "hidden_size", 0)) or None

    def _pick_dtype(self):
        import torch

        if not self.cfg.bf16:
            return torch.float32
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    @property
    def embedding_dim(self) -> int:
        if self.cfg.use_random_embeddings:
            return int(self.cfg.random_embedding_dim)
        self._load()
        return int(self._hidden_size or 0)

    # ----------------------------------------------------------- cache I/O

    def warm_caches_from_disk(self) -> None:
        """Load existing items.parquet / subjects.parquet into memory.

        Called once at startup so subsequent ``embed_unique_*`` calls can
        skip already-cached keys without re-tokenizing them.
        """
        self._item_lookup = _load_parquet_cache(self.items_path)
        self._subject_lookup = _load_parquet_cache(self.subjects_path)
        if self._item_lookup or self._subject_lookup:
            LOG.info(
                "Warmed embedding cache from disk: items=%d subjects=%d",
                len(self._item_lookup),
                len(self._subject_lookup),
            )

    def install_lookups(
        self,
        *,
        items: Mapping[str, np.ndarray] | None = None,
        subjects: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Inject lookups pulled from Drive cache before encoding runs."""
        if items:
            self._item_lookup.update(
                {k: np.asarray(v, dtype=np.float32) for k, v in items.items()}
            )
        if subjects:
            self._subject_lookup.update(
                {k: np.asarray(v, dtype=np.float32) for k, v in subjects.items()}
            )

    def flush_to_disk(self) -> None:
        """Persist in-memory caches as parquet files."""
        if self._item_lookup:
            _write_parquet_cache(
                self.items_path, self._item_lookup, key_col="item_key"
            )
        if self._subject_lookup:
            _write_parquet_cache(
                self.subjects_path, self._subject_lookup, key_col="subject_key"
            )

    # ------------------------------------------------------------ encoding

    def _encode_with_length_bucketing(
        self,
        texts: list[str],
        *,
        max_length: int,
        benchmarks: list[str] | None = None,
        progress_desc: str = "encode",
    ) -> tuple[np.ndarray, list[int]]:
        """Encode a list of texts in length-sorted order.

        Returns ``(embeddings_unsorted_back_to_input_order, token_lengths)``.
        """
        import torch

        n = len(texts)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        # Step 1: tokenize once to get per-text lengths (CPU bound).
        LOG.info("Tokenizing %d texts to compute per-text length...", n)
        t0 = time.time()
        lengths = _tokenize_lengths(self._tok, texts, max_length)
        LOG.info(
            "Length scan done in %.1fs (p50=%d p99=%d max=%d)",
            time.time() - t0,
            int(np.quantile(lengths, 0.5)) if lengths.size else 0,
            int(np.quantile(lengths, 0.99)) if lengths.size else 0,
            int(lengths.max()) if lengths.size else 0,
        )
        order = np.argsort(lengths, kind="stable")
        inv_order = np.argsort(order, kind="stable")
        sorted_texts = [texts[i] for i in order]

        # Step 2: encode in length-sorted order with OOM-resilient batching.
        embeds_sorted: list[np.ndarray] = []
        i = 0
        bs = max(1, int(self._effective_batch_size))
        pbar = tqdm(
            total=n,
            desc=progress_desc,
            unit="txt",
            dynamic_ncols=True,
            leave=False,
        )
        try:
            while i < n:
                batch = sorted_texts[i : i + bs]
                try:
                    vecs = self._forward_batch(batch, max_length=max_length)
                    embeds_sorted.append(vecs)
                    i += len(batch)
                    pbar.update(len(batch))
                    # Show effective batch size & longest seq in this batch
                    pbar.set_postfix(
                        bs=bs,
                        seq_pad=int(lengths[order[max(0, i - 1)]]) if lengths.size else 0,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    # Halve, clear cache, and retry from the same offset.
                    old = bs
                    bs = max(1, bs // 2)
                    self._oom_events.append(
                        {
                            "at_index": int(i),
                            "old_batch_size": int(old),
                            "new_batch_size": int(bs),
                            "max_length": int(max_length),
                            "error": str(exc)[:200],
                        }
                    )
                    LOG.warning(
                        "OOM at index=%d bs=%d -> retrying with bs=%d (max_len=%d)",
                        i,
                        old,
                        bs,
                        max_length,
                    )
                    self._effective_batch_size = bs
                    torch.cuda.empty_cache()
                    if bs <= 0:
                        raise
        finally:
            pbar.close()

        embeds_sorted_arr = np.concatenate(embeds_sorted, axis=0)
        return embeds_sorted_arr[inv_order], lengths.tolist()

    def _forward_batch(self, batch_texts: list[str], *, max_length: int) -> np.ndarray:
        import torch

        enc = self._tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            outputs = self._model(input_ids=input_ids, attention_mask=attn)
        last_hidden = outputs.last_hidden_state
        if self.cfg.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.cfg.pooling == "last_token":
            pooled = _last_token_pool(last_hidden, attn)
        else:
            pooled = _mean_pool(last_hidden, attn)
        return pooled.float().cpu().numpy()

    # ---------------------------------------------- public batch endpoints

    def embed_unique(
        self,
        *,
        kind: str,
        keys: list[str],
        texts: list[str],
        benchmarks: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Embed (and cache) a deduplicated list of (key, text) pairs.

        ``kind`` is "item" or "subject"; controls which in-memory cache and
        parquet file we read/write. Returns ``(lookup, log_payload)``.
        """
        if kind not in {"item", "subject"}:
            raise ValueError("kind must be 'item' or 'subject'")
        if len(keys) != len(texts):
            raise ValueError("keys and texts must have the same length")

        dedup_info = assert_deduplicated(keys, kind=kind)
        if benchmarks is not None and len(benchmarks) != len(keys):
            raise ValueError("benchmarks must align with keys/texts")

        cache = self._item_lookup if kind == "item" else self._subject_lookup
        out: dict[str, np.ndarray] = {}
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        missing_bench: list[str] = []
        for i, k in enumerate(keys):
            cached = cache.get(k)
            if cached is not None and cached.shape[0] > 0:
                out[k] = cached
                self.stats.cache_hits += 1
            else:
                missing_keys.append(k)
                missing_texts.append(texts[i])
                if benchmarks is not None:
                    missing_bench.append(benchmarks[i])
                self.stats.cache_misses += 1
        self.stats.n_texts += len(keys)

        LOG.info(
            "embed_unique[%s]: total=%d cached=%d to_encode=%d",
            kind,
            len(keys),
            len(keys) - len(missing_keys),
            len(missing_keys),
        )

        t0 = time.time()
        if missing_texts:
            if self.cfg.use_random_embeddings:
                vecs = _random_embeddings(missing_keys, dim=int(self.cfg.random_embedding_dim))
            else:
                self._load()
                if self._effective_max_length is None:
                    self._effective_max_length = self._resolve_max_length(missing_texts)
                vecs, tok_lengths = self._encode_with_length_bucketing(
                    missing_texts,
                    max_length=self._effective_max_length,
                    benchmarks=missing_bench if missing_bench else None,
                    progress_desc=f"encode {kind}",
                )
                self._record_token_diagnostics(
                    tok_lengths,
                    benchmarks=missing_bench if missing_bench else None,
                )
            for k, v in zip(missing_keys, vecs):
                vec = np.asarray(v, dtype=np.float32)
                if not np.all(np.isfinite(vec)):
                    self.stats.nan_or_inf += 1
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                if np.linalg.norm(vec) < 1e-8:
                    self.stats.zero_norm += 1
                cache[k] = vec
                out[k] = vec

        elapsed = time.time() - t0

        log_payload = {
            "kind": kind,
            "elapsed_seconds": float(elapsed),
            "n_total": int(len(keys)),
            "n_cache_hits": int(dedup_info["n_total"] - len(missing_keys)),
            "n_encoded": int(len(missing_keys)),
            "n_unique": int(dedup_info["n_unique"]),
            "max_length": int(self._effective_max_length or 0),
            "batch_size": int(self._effective_batch_size),
            "attn_implementation": self._attn_impl,
            "oom_events": list(self._oom_events),
        }
        return out, log_payload

    def _record_token_diagnostics(
        self, lengths: list[int], *, benchmarks: list[str] | None = None
    ) -> None:
        ml = int(self._effective_max_length or 0)
        for i, length in enumerate(lengths):
            self.stats.token_lengths.append(int(length))
            trunc = int(length) >= ml
            self.stats.truncated += int(trunc)
            if benchmarks is not None and i < len(benchmarks):
                b = benchmarks[i]
                self.stats.per_benchmark_total[b] = (
                    self.stats.per_benchmark_total.get(b, 0) + 1
                )
                if trunc:
                    self.stats.per_benchmark_truncation[b] = (
                        self.stats.per_benchmark_truncation.get(b, 0) + 1
                    )

    def _resolve_max_length(self, sample_texts: list[str]) -> int:
        """Pick the encoder ``max_length`` from data (or honor the override)."""
        if self.cfg.max_length is not None and int(self.cfg.max_length) > 0:
            chosen = int(self.cfg.max_length)
            LOG.info("max_length: honoring config override = %d", chosen)
            return chosen
        # Use the encoder's own configured upper bound as a ceiling fallback.
        model_max = int(getattr(self._tok, "model_max_length", 0) or 0)
        ceiling = (
            min(self.cfg.max_length_ceiling, model_max)
            if model_max
            else self.cfg.max_length_ceiling
        )
        # Sample at most ``max_length_sample_size`` texts for the length
        # quantile. The 99th percentile is stable at a few thousand even on
        # corpora with hundreds of thousands of items, and this keeps the
        # silent startup window short. Cap the measurement max_length at the
        # config ceiling instead of overshooting -- we only need to detect
        # that something exceeds the ceiling, not measure exactly how far.
        sample_cap = max(64, int(self.cfg.max_length_sample_size))
        if len(sample_texts) > sample_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(sample_texts), size=sample_cap, replace=False)
            sample = [sample_texts[i] for i in idx]
        else:
            sample = sample_texts
        LOG.info(
            "max_length: scanning %d sample texts (cap=%d) for percentile...",
            len(sample),
            ceiling,
        )
        t0 = time.time()
        lengths = _tokenize_lengths(self._tok, sample, ceiling)
        LOG.info(
            "max_length: scan done in %.1fs (p50=%d p99=%d max=%d)",
            time.time() - t0,
            int(np.quantile(lengths, 0.5)) if lengths.size else 0,
            int(np.quantile(lengths, 0.99)) if lengths.size else 0,
            int(lengths.max()) if lengths.size else 0,
        )
        chosen = choose_max_length(
            lengths,
            floor=int(self.cfg.max_length_floor),
            ceiling=int(ceiling),
        )
        LOG.info(
            "max_length: chose %d from data (p99=%.1f, ceiling=%d, floor=%d)",
            chosen,
            float(np.quantile(lengths, 0.99)) if lengths.size else 0.0,
            int(ceiling),
            int(self.cfg.max_length_floor),
        )
        return chosen

    # ------------------------------------------------------------- summary

    def finalize(
        self,
        *,
        content_hash: str,
        n_items: int,
        n_subjects: int,
        extra_log: Mapping | None = None,
    ) -> None:
        """Flush parquet caches, write meta.json + encoding_log.json."""
        self.flush_to_disk()
        self._write_meta_snapshot(
            content_hash=content_hash, n_items=n_items, n_subjects=n_subjects
        )
        payload = {
            "model_id": self.cfg.model_id,
            "attn_implementation": self._attn_impl,
            "effective_max_length": int(self._effective_max_length or 0),
            "effective_batch_size": int(self._effective_batch_size),
            "n_items": int(n_items),
            "n_subjects": int(n_subjects),
            "oom_events": list(self._oom_events),
            "stats": self.stats.report(),
        }
        if extra_log:
            payload["phases"] = dict(extra_log)
        _write_log(self.log_path, payload)


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def build_unique_items(
    df: pd.DataFrame,
    *,
    contextual: bool,
    passage_prefix: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return (keys, texts, benchmarks) for the unique item rows."""
    sub = df[
        ["item_key", "benchmark", "condition", "item_content"]
    ].drop_duplicates(subset=["item_key"]).reset_index(drop=True)
    keys = sub["item_key"].astype(str).tolist()
    benches = sub["benchmark"].astype(str).tolist()
    if contextual:
        texts = [
            item_contextual_text(b, c, t, prefix=passage_prefix)
            for b, c, t in zip(
                sub["benchmark"].astype(str),
                sub["condition"].astype(str),
                sub["item_content"].astype(str),
            )
        ]
    else:
        texts = [
            item_only_text(t, prefix=passage_prefix)
            for t in sub["item_content"].astype(str)
        ]
    return keys, texts, benches


def build_unique_subjects(
    df: pd.DataFrame, *, query_prefix: str
) -> tuple[list[str], list[str]]:
    sub = (
        df[["subject_key", "subject_content"]]
        .drop_duplicates(subset=["subject_key"])
        .reset_index(drop=True)
    )
    keys = sub["subject_key"].astype(str).tolist()
    texts = [
        subject_text(t, prefix=query_prefix)
        for t in sub["subject_content"].astype(str)
    ]
    return keys, texts


def embed_unique_items(
    df: pd.DataFrame, embedder: TransformerEmbedder
) -> dict[str, np.ndarray]:
    """Backward-compatible convenience wrapper (legacy callers)."""
    keys, texts, benches = build_unique_items(
        df,
        contextual=embedder.cfg.use_contextual_item_text,
        passage_prefix=embedder._resolve_passage_prefix(),
    )
    out, _ = embedder.embed_unique(
        kind="item", keys=keys, texts=texts, benchmarks=benches
    )
    return out


def embed_unique_subjects(
    df: pd.DataFrame, embedder: TransformerEmbedder
) -> dict[str, np.ndarray]:
    """Backward-compatible convenience wrapper (legacy callers)."""
    keys, texts = build_unique_subjects(
        df, query_prefix=embedder._resolve_query_prefix()
    )
    out, _ = embedder.embed_unique(kind="subject", keys=keys, texts=texts)
    return out


def stack_lookup(
    keys: Iterable[str], lookup: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Stack the per-row embedding matrix in dataframe order."""
    return np.stack([lookup[k] for k in keys], axis=0).astype(np.float32, copy=False)


def _random_embeddings(keys: list[str], *, dim: int) -> np.ndarray:
    """Deterministic random unit vectors keyed by the item/subject key.

    Used when ``use_random_embeddings`` is set for fast pipeline debugging.
    """
    out = np.zeros((len(keys), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        seed = int(k[:16], 16)
        r = np.random.default_rng(seed)
        v = r.standard_normal(dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            v /= n
        out[i] = v
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def print_gpu_banner(*, allow_cpu: bool = False) -> None:
    import torch

    print("Torch version :", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available: True")
        print("CUDA version  :", torch.version.cuda)
        idx = torch.cuda.current_device()
        print(f"GPU name      : {torch.cuda.get_device_name(idx)}")
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"GPU VRAM (GiB): {vram:.1f}")
        print("bf16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("CUDA available: False")
        if not allow_cpu and not os.environ.get("ALLOW_CPU"):
            raise RuntimeError(
                "No GPU available. Set ALLOW_CPU=1 to run on CPU explicitly."
            )


__all__ = [
    "EncoderConfig",
    "TransformerEmbedder",
    "EmbeddingStats",
    "assert_deduplicated",
    "build_unique_items",
    "build_unique_subjects",
    "choose_max_length",
    "content_hash_for_items",
    "embed_unique_items",
    "embed_unique_subjects",
    "encoder_slug",
    "is_qwen3_embedding",
    "item_contextual_text",
    "item_only_text",
    "login_huggingface",
    "print_gpu_banner",
    "qwen3_prefix",
    "resolve_hf_token",
    "stack_lookup",
    "subject_text",
    "verify_flash_attention",
]
