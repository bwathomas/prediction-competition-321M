"""Transformer embeddings with caching, mean pooling, and truncation diagnostics.

Design goals:

- Cache **unique** texts. The dataset has many subjects answering many items,
  so the unique-text set is much smaller than the row set. The disk cache key
  is the stable sha256 of the source string(s), so the same text always gets
  the same embedding regardless of which dataframe row referenced it.

- Report truncation honestly. Hidden truncation silently degrades downstream
  models. We log p50 / p90 / p95 / p99 token lengths, plus per-benchmark
  truncation rates, so the user can change `max_length` or the encoder
  before training.

- Support a "contextual" item text variant
  ``Benchmark: ... Condition: ... Item: ...`` separately from item-only text.
  The downstream model gets to choose which variant it uses.

- bf16 forward where supported, with a fallback to fp16 / fp32 if the GPU
  doesn't support bf16. The forward pass is `torch.inference_mode`; we do
  NOT fine-tune the encoder here (LoRA hooks are TODO).

The cache layout is:

    {cache_dir}/{encoder_slug}/item_only/{sha256}.npy
    {cache_dir}/{encoder_slug}/item_ctx/{sha256}.npy
    {cache_dir}/{encoder_slug}/subject/{sha256}.npy
    {cache_dir}/{encoder_slug}/metadata.json
"""

from __future__ import annotations

import dataclasses
import getpass
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("embeddings")


# ---------------------------------------------------------------------------
# HF token plumbing -- never print, never persist on disk
# ---------------------------------------------------------------------------


def _try_colab_userdata() -> str | None:
    """Best-effort fetch of HF_TOKEN from the Colab userdata secret store.

    Requires running inside Google Colab with an `HF_TOKEN` secret created
    via the left-rail "Secrets" panel. Silently returns None outside Colab
    or if the secret is absent / not authorized for the current notebook.
    """
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
    """Best-effort fetch of HF_TOKEN from Google Secret Manager.

    Requires `google-cloud-secret-manager` and ADC. Silently returns None
    on any failure -- the caller will fall through to interactive prompt.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GCP_PROJECT"
    )
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
    """Resolve the HF token without ever printing or persisting it.

    Order:
      1. ``os.environ['HF_TOKEN']``
      2. Google Colab ``userdata.get('HF_TOKEN')`` secret (no-op outside Colab)
      3. Google Secret Manager ``HF_TOKEN`` (no-op outside GCP)
      4. ``getpass`` prompt (only if ``interactive=True``)

    Returns None if all sources fail (caller must handle that).
    """
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
# Cache layout
# ---------------------------------------------------------------------------


def encoder_slug(model_id: str) -> str:
    """Stable, filesystem-safe directory name for an encoder."""
    safe = model_id.replace("/", "__")
    return safe[:200]


def _cache_paths(cache_dir: Path, encoder_id: str) -> tuple[Path, Path, Path, Path]:
    base = Path(cache_dir) / encoder_slug(encoder_id)
    item_only = base / "item_only"
    item_ctx = base / "item_ctx"
    subject = base / "subject"
    for d in (item_only, item_ctx, subject):
        d.mkdir(parents=True, exist_ok=True)
    return base, item_only, item_ctx, subject


def _key_to_path(folder: Path, key: str) -> Path:
    """Sharded by the first two hex chars to keep directory sizes sane."""
    sub = folder / key[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{key}.npy"


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


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden, attention_mask):
    """Mean pool the last hidden state over the attention mask.

    Works for any HF encoder that returns ``last_hidden_state`` of shape
    [B, T, H]. Returns [B, H] in the same dtype as the input.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _last_token_pool(last_hidden, attention_mask):
    """Last non-pad token pool. Used by some e5 / instruction-tuned encoders."""
    import torch

    seq_lens = attention_mask.sum(dim=1) - 1
    seq_lens = seq_lens.clamp(min=0)
    batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch, seq_lens]


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStats:
    """Diagnostics produced during a batch of `embed_texts` calls."""

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
    max_length: int = 512
    batch_size: int = 32
    pooling: str = "mean"  # mean | cls | last_token
    bf16: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    use_contextual_item_text: bool = True
    cache_dir: str = "artifacts/embeddings"
    trust_remote_code: bool = False
    use_random_embeddings: bool = False
    random_embedding_dim: int = 256


class TransformerEmbedder:
    """Mean-pooled HF encoder with on-disk cache and truncation diagnostics."""

    def __init__(self, cfg: EncoderConfig, *, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._tok = None
        self._model = None
        self._hidden_size: int | None = None
        self.stats = EmbeddingStats()
        self.base, self.dir_item, self.dir_ctx, self.dir_subject = _cache_paths(
            Path(cfg.cache_dir), cfg.model_id
        )
        self._write_metadata()

    # ------------------------------------------------------------------ init

    def _write_metadata(self) -> None:
        meta_path = self.base / "metadata.json"
        meta = {
            "model_id": self.cfg.model_id,
            "max_length": self.cfg.max_length,
            "pooling": self.cfg.pooling,
            "bf16": self.cfg.bf16,
            "query_prefix": self.cfg.query_prefix,
            "passage_prefix": self.cfg.passage_prefix,
            "use_contextual_item_text": self.cfg.use_contextual_item_text,
            "use_random_embeddings": self.cfg.use_random_embeddings,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def _load(self):
        if self._model is not None or self.cfg.use_random_embeddings:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        LOG.info("Loading encoder %s", self.cfg.model_id)
        self._tok = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=self.cfg.trust_remote_code
        )
        dtype = self._pick_dtype()
        self._model = AutoModel.from_pretrained(
            self.cfg.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        self._model.eval()
        self._model.to(self.device)
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

    # --------------------------------------------------------------- caching

    @staticmethod
    def _cache_key_for_text(text: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def _read_cached(self, folder: Path, key: str) -> np.ndarray | None:
        path = _key_to_path(folder, key)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def _write_cached(self, folder: Path, key: str, vec: np.ndarray) -> None:
        path = _key_to_path(folder, key)
        np.save(path, vec.astype(np.float32, copy=False))

    # ----------------------------------------------------------------- core

    def embed_texts(
        self,
        texts: list[str],
        *,
        folder: Path,
        keys: list[str] | None = None,
        benchmarks: list[str] | None = None,
    ) -> np.ndarray:
        """Embed (and cache) a list of texts. Returns [N, H] float32 numpy array.

        We don't deduplicate `texts` here; if you pass duplicates you'll get
        duplicate forward passes (but they will still hit the on-disk cache
        after the first one). For production, call once per unique text via
        `embed_unique_texts`.
        """
        if keys is None:
            keys = [self._cache_key_for_text(t) for t in texts]
        if len(keys) != len(texts):
            raise ValueError("keys length must equal texts length")

        self.stats.n_texts += len(texts)
        out: list[np.ndarray | None] = [None] * len(texts)
        need_idx: list[int] = []
        for i, k in enumerate(keys):
            cached = self._read_cached(folder, k)
            if cached is not None:
                out[i] = cached
                self.stats.cache_hits += 1
            else:
                need_idx.append(i)
                self.stats.cache_misses += 1

        if need_idx:
            if self.cfg.use_random_embeddings:
                self._fill_random(out, need_idx, keys, folder)
            else:
                self._encode_indices(out, need_idx, texts, keys, folder, benchmarks)

        arr = np.stack([o for o in out], axis=0)
        return arr.astype(np.float32, copy=False)

    def _fill_random(
        self,
        out: list[np.ndarray | None],
        need_idx: list[int],
        keys: list[str],
        folder: Path,
    ) -> None:
        dim = int(self.cfg.random_embedding_dim)
        rng = np.random.default_rng(0)
        for i in need_idx:
            seed = int(keys[i][:16], 16)
            r = np.random.default_rng(seed)
            v = r.standard_normal(dim).astype(np.float32)
            v /= max(1e-6, np.linalg.norm(v))
            out[i] = v
            self._write_cached(folder, keys[i], v)

    def _encode_indices(
        self,
        out: list[np.ndarray | None],
        need_idx: list[int],
        texts: list[str],
        keys: list[str],
        folder: Path,
        benchmarks: list[str] | None,
    ) -> None:
        import torch

        self._load()
        bs = max(1, int(self.cfg.batch_size))
        max_len = int(self.cfg.max_length)
        with torch.inference_mode():
            for start in range(0, len(need_idx), bs):
                batch_pos = need_idx[start : start + bs]
                batch_texts = [texts[i] for i in batch_pos]
                enc = self._tok(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                    return_length=True,
                )
                lengths = enc.get("length")
                if lengths is None:
                    lengths = enc["attention_mask"].sum(dim=1)
                lengths = lengths.tolist()
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                outputs = self._model(input_ids=input_ids, attention_mask=attn)
                last_hidden = outputs.last_hidden_state
                if self.cfg.pooling == "cls":
                    pooled = last_hidden[:, 0]
                elif self.cfg.pooling == "last_token":
                    pooled = _last_token_pool(last_hidden, attn)
                else:
                    pooled = _mean_pool(last_hidden, attn)
                pooled = pooled.float().cpu().numpy()
                for pos, vec, length, raw_text in zip(
                    batch_pos, pooled, lengths, batch_texts
                ):
                    if not np.all(np.isfinite(vec)):
                        self.stats.nan_or_inf += 1
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.linalg.norm(vec) < 1e-8:
                        self.stats.zero_norm += 1
                    self.stats.token_lengths.append(int(length))
                    truncated = int(length) >= max_len
                    self.stats.truncated += int(truncated)
                    bench = (
                        benchmarks[pos]
                        if benchmarks is not None and pos < len(benchmarks)
                        else None
                    )
                    if bench is not None:
                        self.stats.per_benchmark_total[bench] = (
                            self.stats.per_benchmark_total.get(bench, 0) + 1
                        )
                        if truncated:
                            self.stats.per_benchmark_truncation[bench] = (
                                self.stats.per_benchmark_truncation.get(bench, 0) + 1
                            )
                    out[pos] = vec
                    self._write_cached(folder, keys[pos], vec)


# ---------------------------------------------------------------------------
# High-level batch helpers
# ---------------------------------------------------------------------------


def embed_unique_items(
    df: pd.DataFrame,
    embedder: TransformerEmbedder,
    *,
    contextual: bool | None = None,
) -> dict[str, np.ndarray]:
    """Embed each unique item_key once. Returns {item_key: vector}.

    Adds the `Benchmark/Condition/Item` template iff `contextual` (defaults to
    the encoder's configured value).
    """
    contextual = (
        contextual if contextual is not None else embedder.cfg.use_contextual_item_text
    )
    sub = df[["item_key", "benchmark", "condition", "item_content"]].drop_duplicates(
        subset=["item_key"]
    )
    if contextual:
        texts = [
            item_contextual_text(b, c, t, prefix=embedder.cfg.passage_prefix)
            for b, c, t in zip(sub["benchmark"], sub["condition"], sub["item_content"])
        ]
        folder = embedder.dir_ctx
    else:
        texts = [
            item_only_text(t, prefix=embedder.cfg.passage_prefix)
            for t in sub["item_content"]
        ]
        folder = embedder.dir_item
    keys = sub["item_key"].tolist()
    benches = sub["benchmark"].astype(str).tolist()
    vecs = embedder.embed_texts(texts, folder=folder, keys=keys, benchmarks=benches)
    return dict(zip(keys, [v for v in vecs]))


def embed_unique_subjects(
    df: pd.DataFrame, embedder: TransformerEmbedder
) -> dict[str, np.ndarray]:
    """Embed each unique subject_key once. Returns {subject_key: vector}."""
    sub = df[["subject_key", "subject_content"]].drop_duplicates(subset=["subject_key"])
    texts = [
        subject_text(t, prefix=embedder.cfg.query_prefix) for t in sub["subject_content"]
    ]
    keys = sub["subject_key"].tolist()
    vecs = embedder.embed_texts(texts, folder=embedder.dir_subject, keys=keys)
    return dict(zip(keys, [v for v in vecs]))


def stack_lookup(
    keys: Iterable[str], lookup: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Stack the per-row embedding matrix in dataframe order."""
    return np.stack([lookup[k] for k in keys], axis=0).astype(np.float32, copy=False)


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
    """Print GPU name / VRAM / CUDA / torch version. Fail loudly if no GPU."""
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
    "embed_unique_items",
    "embed_unique_subjects",
    "encoder_slug",
    "item_contextual_text",
    "item_only_text",
    "login_huggingface",
    "print_gpu_banner",
    "resolve_hf_token",
    "stack_lookup",
    "subject_text",
]
