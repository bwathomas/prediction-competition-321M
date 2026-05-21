"""Pre-tokenized item cache for the LoRA fine-tuning loop.

LoRA training cannot reuse the frozen-encoder embedding cache: every batch
must forward raw item token ids through the adapter-augmented encoder so the
output reflects the current adapter weights. Pre-tokenizing each unique
item once and persisting ``input_ids`` / ``attention_mask`` as a parquet
keyed by ``item_key`` removes the per-step CPU tokenization cost (which
would dominate a fast LoRA step at bf16 on an A100).

Cache layout (mirrors the embedding cache, one directory per encoder):

    {cache_dir}/tokenized_items/{encoder_slug}/
        meta.json           # encoder id, tokenizer kind, max_length, content_hash, n_items
        tokenized.parquet   # columns: item_key, input_ids (list[int32]), n_tokens (int32)

Notes:

- The cache writes UNPADDED ``input_ids``. Padding (and the matching
  attention mask) happens at batch construction time in ``src.lora_train``
  so each batch only pads to its own length max. Storing padded tensors
  for ~300k items at max_length=1024 would otherwise waste ~1.2 GB of disk.
- We reuse the *exact* tokenizer + prefix + contextual-item-text template
  the embedder uses, so the LoRA forward path sees byte-identical text
  to what was originally encoded into the frozen embedding cache.
- The cache is mirrored to Google Drive via ``drive_cache`` using the same
  staging+rename atomic upload pattern as the embedding cache.

Subjects are deliberately NOT tokenized here: the LoRA loop keeps subject
embeddings frozen (subjects are not cold-start; the subject side is a
lookup table). Only item text flows through the LoRA encoder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .colab_tqdm import get_tqdm

tqdm = get_tqdm()

from .embeddings import (
    EncoderConfig,
    TransformerEmbedder,
    build_unique_items,
    encoder_slug as _encoder_slug,
)

LOG = logging.getLogger("tokenized_items")


# ---------------------------------------------------------------------------
# Layout + meta
# ---------------------------------------------------------------------------


_CACHE_FILES = ("meta.json", "tokenized.parquet")


def tokenized_items_dir(cache_root: str | os.PathLike, model_id: str) -> Path:
    """Return the local on-disk directory for the tokenized-item cache."""
    base = Path(cache_root) / "tokenized_items" / _encoder_slug(model_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class TokenizedItemCache:
    """Loaded tokenized-item cache.

    ``input_ids`` is a list of variable-length int32 arrays, one per unique
    item key in ``item_keys`` order. Per-item token lengths are precomputed
    so length-bucketed sampling does not need to inspect the arrays.
    """

    item_keys: list[str]
    input_ids: list[np.ndarray]
    token_lens: np.ndarray
    meta: dict

    @property
    def n_items(self) -> int:
        return len(self.item_keys)

    @property
    def max_length(self) -> int:
        return int(self.meta.get("max_length", 0) or 0)

    def index_map(self) -> dict[str, int]:
        """Map ``item_key -> row index`` for O(1) per-row lookup."""
        return {k: i for i, k in enumerate(self.item_keys)}


def _content_hash(
    *,
    model_id: str,
    max_length: int,
    pairs: Iterable[tuple[str, str]],
) -> str:
    """Stable hash over (model id, max_length, sorted (key, text) pairs).

    Used as a cache-invalidation signal: any change to the encoder id, the
    chosen max_length, or the underlying item text rolls the hash. The
    Drive layer treats a hash mismatch as a partial-hit and will rebuild.
    """
    h = hashlib.sha256()
    h.update(b"tokenized_items_v1\x00")
    h.update(str(model_id).encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(str(int(max_length)).encode("ascii"))
    h.update(b"\x00")
    for k, t in sorted(pairs, key=lambda x: x[0]):
        h.update(str(k).encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(str(t).encode("utf-8", errors="replace"))
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
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------


def build_tokenized_item_cache(
    item_df: pd.DataFrame,
    embedder: TransformerEmbedder,
    *,
    out_path: str | os.PathLike,
    max_length: int | None = None,
    force_rebuild: bool = False,
) -> TokenizedItemCache:
    """Tokenize every unique item once and persist to ``out_path``.

    ``item_df`` must have columns ``item_key, benchmark, condition,
    item_content`` (the same shape ``build_unique_items`` consumes). We
    reuse the embedder's prefix resolvers + contextual-item-text template
    so train-time tokenization is identical to the frozen-encoder path.

    ``max_length`` defaults to the embedder's effective ``max_length``
    (or the encoder config's override). LoRA training typically uses a
    smaller cap (e.g. 1024) than the frozen cache (which can run up to
    the encoder ceiling) to keep activation memory under control.

    Returns a loaded :class:`TokenizedItemCache`. Side effect: writes
    ``out_path/{meta.json, tokenized.parquet}``.
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    parquet_path = out_dir / "tokenized.parquet"

    cfg: EncoderConfig = embedder.cfg
    passage_prefix = embedder._resolve_passage_prefix()
    keys, texts, _benches = build_unique_items(
        item_df,
        contextual=cfg.use_contextual_item_text,
        passage_prefix=passage_prefix,
    )

    if not keys:
        raise ValueError("build_tokenized_item_cache: item_df is empty")

    eff_max_length = int(
        max_length
        or embedder._effective_max_length
        or cfg.max_length
        or 1024
    )
    if eff_max_length <= 0:
        eff_max_length = 1024

    pairs = list(zip(keys, texts))
    expected_hash = _content_hash(
        model_id=cfg.model_id, max_length=eff_max_length, pairs=pairs
    )

    if not force_rebuild and parquet_path.exists() and meta_path.exists():
        meta = _read_meta(meta_path)
        if (
            str(meta.get("content_hash") or "") == expected_hash
            and int(meta.get("n_items") or 0) == len(keys)
            and int(meta.get("max_length") or 0) == eff_max_length
        ):
            LOG.info(
                "tokenized-item cache HIT (%s, n=%d, max_length=%d)",
                meta_path,
                len(keys),
                eff_max_length,
            )
            return load_tokenized_item_cache(out_dir)
        LOG.info(
            "tokenized-item cache present but stale (%s); rebuilding "
            "(expected_hash[:8]=%s cached_hash[:8]=%s)",
            meta_path,
            expected_hash[:8],
            str(meta.get("content_hash") or "")[:8],
        )

    # Lazy-load the tokenizer via the embedder's resolver. We never need to
    # load the encoder weights here -- tokenization is CPU work.
    embedder._load()  # noqa: SLF001 - shared module, intentional
    tokenizer = embedder._tok  # noqa: SLF001
    if tokenizer is None:
        raise RuntimeError(
            "tokenized-item cache build requires a usable tokenizer; "
            "embedder._tok is None (random embeddings mode?)."
        )

    tokenizer_kind = type(tokenizer).__name__
    # ``"Fast" in tokenizer_kind`` used to be the detection rule, but newer
    # Hugging Face tokenizers (notably ``Qwen2Tokenizer`` in recent
    # Transformers) are backed by the Rust ``tokenizers`` library yet do
    # not carry "Fast" in their class name. The class-name check then
    # incorrectly took the slow Python fallback path -- a 5-10x slowdown
    # on the ~300k unique item corpus. Use attribute / backend detection
    # instead, which is what HF's own utilities do internally.
    is_fast = bool(
        getattr(tokenizer, "is_fast", False)
        or getattr(tokenizer, "backend", None) == "tokenizers"
        or hasattr(tokenizer, "backend_tokenizer")
    )
    LOG.info(
        "Tokenizing %d unique items (max_length=%d, tokenizer=%s, fast=%s, backend=%s)",
        len(keys),
        eff_max_length,
        tokenizer_kind,
        is_fast,
        getattr(tokenizer, "backend", None),
    )

    t0 = time.time()
    if is_fast:
        # Fast tokenizers release the GIL and parallelize across Rust
        # threads. We chunk into batches of ``TOKENIZE_BATCH_SIZE`` (env
        # override) instead of one enormous batched call so we never
        # materialize a single ``BatchEncoding`` of all ~300k items at
        # once and the progress bar updates several times per second.
        ids_lists = []
        batch_size = int(os.environ.get("TOKENIZE_BATCH_SIZE", "4096"))
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="tokenize item batches",
            unit="batch",
            dynamic_ncols=True,
        ):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=eff_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids_lists.extend(enc["input_ids"])
    else:
        ids_lists = []
        for t in tqdm(
            texts, desc="tokenize items", unit="item", dynamic_ncols=True
        ):
            ids_lists.append(
                tokenizer.encode(
                    t,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=eff_max_length,
                )
            )
    elapsed = time.time() - t0

    input_ids: list[np.ndarray] = []
    lens: list[int] = []
    truncated = 0
    for ids in ids_lists:
        arr = np.asarray(ids, dtype=np.int32)
        n = int(arr.size)
        if n >= eff_max_length:
            truncated += 1
        lens.append(n)
        input_ids.append(arr)

    LOG.info(
        "Tokenization done in %.1fs: n=%d p50=%d p99=%d max=%d truncated=%d (%.2f%%)",
        elapsed,
        len(input_ids),
        int(np.quantile(lens, 0.5)) if lens else 0,
        int(np.quantile(lens, 0.99)) if lens else 0,
        int(max(lens) if lens else 0),
        truncated,
        100.0 * truncated / max(1, len(lens)),
    )

    # Persist parquet (each row's input_ids list[int32] becomes a parquet
    # list-of-int column). We keep ``n_tokens`` separately so the LoRA
    # loop can length-bucket without materializing the arrays.
    df = pd.DataFrame(
        {
            "item_key": keys,
            "input_ids": [ids.tolist() for ids in input_ids],
            "n_tokens": np.asarray(lens, dtype=np.int32),
        }
    )
    df.to_parquet(parquet_path, index=False)

    meta = {
        "encoder_model_id": cfg.model_id,
        "tokenizer_kind": tokenizer_kind,
        "is_fast_tokenizer": bool(is_fast),
        "max_length": int(eff_max_length),
        "n_items": int(len(keys)),
        "content_hash": expected_hash,
        "passage_prefix": passage_prefix,
        "qwen3_instruction": cfg.qwen3_instruction
        if "Qwen3-Embedding" in (cfg.model_id or "")
        else "",
        "use_contextual_item_text": bool(cfg.use_contextual_item_text),
        "p50_tokens": int(np.quantile(lens, 0.5)) if lens else 0,
        "p99_tokens": int(np.quantile(lens, 0.99)) if lens else 0,
        "max_tokens_seen": int(max(lens) if lens else 0),
        "truncation_rate": float(truncated / max(1, len(lens))),
        "elapsed_seconds": float(elapsed),
    }
    _write_meta(meta_path, meta)
    LOG.info(
        "Wrote tokenized-item cache: %s (%d items, %.1fs)",
        out_dir,
        len(keys),
        elapsed,
    )

    return TokenizedItemCache(
        item_keys=list(keys),
        input_ids=input_ids,
        token_lens=np.asarray(lens, dtype=np.int32),
        meta=meta,
    )


def load_tokenized_item_cache(
    cache_dir: str | os.PathLike,
) -> TokenizedItemCache:
    """Read an existing tokenized-item parquet + meta into memory."""
    base = Path(cache_dir)
    meta_path = base / "meta.json"
    parquet_path = base / "tokenized.parquet"
    if not parquet_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"tokenized-item cache missing at {base} "
            f"(parquet={parquet_path.exists()} meta={meta_path.exists()})"
        )
    meta = _read_meta(meta_path)
    df = pd.read_parquet(parquet_path)
    keys = df["item_key"].astype(str).tolist()
    input_ids = [np.asarray(v, dtype=np.int32) for v in df["input_ids"].tolist()]
    if "n_tokens" in df.columns:
        lens = df["n_tokens"].to_numpy(dtype=np.int32, copy=False)
    else:
        lens = np.asarray([a.size for a in input_ids], dtype=np.int32)
    return TokenizedItemCache(
        item_keys=keys, input_ids=input_ids, token_lens=lens, meta=meta
    )


# ---------------------------------------------------------------------------
# Drive sync (symmetric to drive_cache.upload_from_local for the embedding
# cache, but with the tokenized files instead).
# ---------------------------------------------------------------------------


def drive_folder_for(
    *, drive_root: str | os.PathLike, model_id: str
) -> Path:
    """``{drive_root}/tokenized_items/{encoder_slug}``."""
    return Path(drive_root) / "tokenized_items" / _encoder_slug(model_id)


def resolve_drive_cache(
    *,
    cfg: Mapping,
    model_id: str,
    local_cache_root: str | os.PathLike,
    expected_hash: str | None = None,
) -> dict:
    """Best-effort populate the local tokenized-item cache from Drive.

    Mirrors ``drive_cache.resolve_cache`` for the embedding cache. ``cfg``
    is the full project config dict; we read ``cfg["drive_cache"]`` and
    derive the tokenized-items subfolder from its ``folder`` root.

    Outside Colab (or with ``drive_cache.enabled = false``) this is a
    no-op and the caller proceeds to build the cache locally. Returns a
    small status dict for logging.
    """
    from . import drive_cache as drive_cache_mod  # local import to avoid cycles

    local_folder = tokenized_items_dir(local_cache_root, model_id)
    dc = (cfg or {}).get("drive_cache") or {}
    enabled = bool(dc.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "mounted": False,
            "cache_hit": False,
            "drive_folder": None,
            "local_folder": str(local_folder),
            "reason": "drive_cache.enabled = false",
        }

    mounted = drive_cache_mod.mount_drive_if_needed()
    if not mounted:
        return {
            "enabled": True,
            "mounted": False,
            "cache_hit": False,
            "drive_folder": None,
            "local_folder": str(local_folder),
            "reason": "drive mount unavailable (not in Colab or mount failed)",
        }

    drive_root = dc.get("folder", "")
    if not drive_root:
        return {
            "enabled": True,
            "mounted": True,
            "cache_hit": False,
            "drive_folder": None,
            "local_folder": str(local_folder),
            "reason": "drive_cache.folder is empty",
        }

    drive_folder = drive_folder_for(drive_root=drive_root, model_id=model_id)
    drive_meta = drive_folder / "meta.json"
    if not drive_meta.exists():
        return {
            "enabled": True,
            "mounted": True,
            "cache_hit": False,
            "drive_folder": str(drive_folder),
            "local_folder": str(local_folder),
            "reason": "drive cache empty (no meta.json)",
        }

    try:
        meta = json.loads(drive_meta.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    cached_hash = str(meta.get("content_hash") or "")
    hit = (
        expected_hash is not None
        and cached_hash == expected_hash
        and (drive_folder / "tokenized.parquet").exists()
    )

    # Copy whatever's there so a partial-hit can be re-validated locally.
    for fname in _CACHE_FILES:
        src = drive_folder / fname
        if src.exists():
            local_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_folder / fname)

    return {
        "enabled": True,
        "mounted": True,
        "cache_hit": bool(hit),
        "drive_folder": str(drive_folder),
        "local_folder": str(local_folder),
        "cached_hash": cached_hash or None,
        "expected_hash": expected_hash,
        "reason": (
            "drive cache HIT (content_hash match)"
            if hit
            else "drive cache present but content_hash differs / unknown"
        ),
    }


def upload_to_drive(
    *,
    local_folder: str | os.PathLike,
    drive_folder: str | os.PathLike,
) -> dict:
    """Atomically publish the local tokenized cache to Drive.

    Writes into ``{drive_folder}.tmp/`` first and renames into place so
    consumers never read a half-uploaded parquet.
    """
    from . import drive_cache as drive_cache_mod  # local import to avoid cycles

    return drive_cache_mod.upload_from_local(
        local_folder=Path(local_folder),
        drive_folder=Path(drive_folder),
        files=list(_CACHE_FILES),
    )


__all__ = [
    "TokenizedItemCache",
    "build_tokenized_item_cache",
    "drive_folder_for",
    "load_tokenized_item_cache",
    "resolve_drive_cache",
    "tokenized_items_dir",
    "upload_to_drive",
]
