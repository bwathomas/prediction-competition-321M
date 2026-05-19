"""Local LLM-as-judge: per-(subject, item) features for the residual MLP.

A small HF causal LM is loaded once at module scope and asked, for each
(subject, item) pair: "would this subject answer this item correctly? Reply
yes or no." We then read the next-token logits at the answer position and
extract four scalar features:

    [lp_yes, lp_no, lp_yes - lp_no, p_yes_renormalized]

These are concatenated alongside the item embedding, pool features, subject
embedding (if used), and cluster embedding inside the residual head input
(see ``src.models`` for the wiring). The head learns *where to trust the
judge* rather than a global blend weight.

Caching
-------
The judge cache is a parquet file at
``{cache_dir}/{judge_slug}/scores.parquet`` with columns
``[subject_key, item_key, lp_yes, lp_no, lp_diff, p_yes_renorm,
judge_model_id, prompt_version]``. The slug is the model id + the prompt
version hash; changing the prompt template or token list invalidates the
slug and forces a re-score. **Do not edit the prompt template casually**:
every change costs GPU-hours.

We never re-score a ``(subject_key, item_key)`` pair already present for
the current ``judge_model_id`` + ``prompt_version``.

Implementation notes
--------------------
- The tokenizer and model are loaded at ``__init__`` time, not per call.
- Flash Attention 2 is attempted first with a graceful fallback to SDPA.
- Right-padding is forced. Left-padding silently breaks next-token logprob
  extraction; we override ``tokenizer.padding_side = 'right'`` at load and
  fail loudly if the override doesn't stick.
- Yes/no token variants are summed in probability space before renormalizing
  to handle BPE tokenizers that split " yes" vs "yes" differently across
  contexts.
- ``model.eval()`` + ``torch.inference_mode()`` everywhere; the judge never
  participates in gradient computation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, *args, **kwargs):
        return x

LOG = logging.getLogger("judge")

# Columns persisted in the on-disk cache.
JUDGE_CACHE_COLUMNS: tuple[str, ...] = (
    "subject_key",
    "item_key",
    "lp_yes",
    "lp_no",
    "lp_diff",
    "p_yes_renorm",
    "judge_model_id",
    "prompt_version",
)

# Number of scalar features exposed to the residual MLP.
JUDGE_FEATURE_DIM: int = 4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """Hyperparameters for the local LLM-as-judge."""

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 1                       # we only read next-token logits
    batch_size: int = 16
    max_prompt_tokens: int = 1024                 # truncate item/subject if needed
    bf16: bool = True
    use_flash_attention: bool = True
    yes_tokens: tuple[str, ...] = (" yes", "yes", " Yes", "Yes")
    no_tokens: tuple[str, ...] = (" no", "no", " No", "No")
    prompt_template: str = (
        "You will see a description of an AI subject and an evaluation item. "
        "Decide whether the subject would answer the item correctly. "
        "Reply with a single token: yes or no.\n\n"
        "Benchmark: {benchmark}\n"
        "Condition: {condition}\n"
        "Subject: {subject_content}\n"
        "Item: {item_content}\n"
        "Answer:"
    )
    cache_dir: str = "artifacts/judge"
    trust_remote_code: bool = False
    # Tokens reserved for the prompt tail "...\nAnswer:" -- the truncation
    # logic guarantees at least this many tokens are kept for the suffix
    # after item / subject content has been truncated from the start.
    suffix_reserve_tokens: int = 256


# ---------------------------------------------------------------------------
# Slug / hash helpers
# ---------------------------------------------------------------------------


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")[:200]


def compute_prompt_version(cfg: JudgeConfig) -> str:
    """Stable hash of the prompt template + yes/no token lists.

    Changing any of these invalidates the on-disk cache so future runs
    re-score (which is expensive). The model_id is *not* included here -- it
    lives in the slug.
    """
    h = hashlib.sha256()
    h.update(cfg.prompt_template.encode("utf-8", errors="replace"))
    h.update(b"\x00yes\x00")
    for t in cfg.yes_tokens:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    h.update(b"\x00no\x00")
    for t in cfg.no_tokens:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def judge_slug(cfg: JudgeConfig) -> str:
    """Filesystem-safe directory name = model + prompt-version hash."""
    return f"{_slugify_model_id(cfg.model_id)}__{compute_prompt_version(cfg)}"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _cache_path(cfg: JudgeConfig) -> Path:
    base = Path(cfg.cache_dir) / judge_slug(cfg)
    base.mkdir(parents=True, exist_ok=True)
    return base / "scores.parquet"


def load_cache(cfg: JudgeConfig) -> pd.DataFrame:
    """Read the cache parquet for ``cfg``. Returns an empty df if missing.

    Filters to rows matching the current ``model_id`` + ``prompt_version``
    so accidental cross-contamination (two slugs sharing a path) is impossible.
    """
    path = _cache_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=list(JUDGE_CACHE_COLUMNS))
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to read judge cache at %s: %s", path, exc)
        return pd.DataFrame(columns=list(JUDGE_CACHE_COLUMNS))
    for c in JUDGE_CACHE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series(dtype=float if c.startswith(("lp_", "p_")) else str)
    prompt_version = compute_prompt_version(cfg)
    mask = (df["judge_model_id"] == cfg.model_id) & (
        df["prompt_version"] == prompt_version
    )
    return df.loc[mask, list(JUDGE_CACHE_COLUMNS)].reset_index(drop=True)


def append_cache(cfg: JudgeConfig, new_rows: pd.DataFrame) -> Path:
    """Atomically append ``new_rows`` to the on-disk parquet.

    Reads existing rows (filtered by model + prompt version), concatenates,
    drops duplicate (subject_key, item_key) pairs (keeping the new value),
    and writes the merged frame back. Returns the cache path.
    """
    if new_rows.empty:
        return _cache_path(cfg)
    path = _cache_path(cfg)
    existing = load_cache(cfg)
    merged = pd.concat([existing, new_rows[list(JUDGE_CACHE_COLUMNS)]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["subject_key", "item_key"], keep="last")
    merged = merged[list(JUDGE_CACHE_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


# ---------------------------------------------------------------------------
# Tokenizer / encoding helpers
# ---------------------------------------------------------------------------


def _first_token_id(tokenizer, variant: str) -> int | None:
    """First token id of ``variant`` (without BOS / specials).

    Returns ``None`` if the variant tokenizes to zero pieces (rare; we just
    skip it then).
    """
    ids = tokenizer.encode(variant, add_special_tokens=False)
    if not ids:
        return None
    return int(ids[0])


def _resolve_token_ids(tokenizer, variants: Iterable[str]) -> list[int]:
    """De-duplicated list of first-token ids for the given variants."""
    seen: dict[int, None] = {}
    for v in variants:
        tid = _first_token_id(tokenizer, v)
        if tid is None:
            continue
        seen.setdefault(tid, None)
    return list(seen.keys())


def _truncate_prompt(
    tokenizer,
    cfg: JudgeConfig,
    *,
    benchmark: str,
    condition: str,
    subject_content: str,
    item_content: str,
) -> str:
    """Build a prompt that fits in ``cfg.max_prompt_tokens``.

    Truncates ``item_content`` first (it's typically the longest), then
    ``subject_content``. Always reserves ``cfg.suffix_reserve_tokens`` for
    the trailing "...\nAnswer:" portion so the answer position never falls
    off the end of the context.
    """
    template_full = cfg.prompt_template.format(
        benchmark=benchmark,
        condition=condition,
        subject_content=subject_content,
        item_content=item_content,
    )
    if cfg.max_prompt_tokens <= 0:
        return template_full

    # Fast path: no truncation needed.
    full_ids = tokenizer.encode(template_full, add_special_tokens=False)
    if len(full_ids) <= cfg.max_prompt_tokens:
        return template_full

    # Build with shrinking item_content first, then subject_content. We use
    # a simple character-based shrink because token-based shrink requires a
    # full re-tokenize per iteration; for our content lengths the char-based
    # heuristic converges in <=3 iterations.
    suffix_reserve = max(64, int(cfg.suffix_reserve_tokens))
    budget = max(64, int(cfg.max_prompt_tokens) - suffix_reserve)

    def _attempt(it_text: str, sb_text: str) -> tuple[str, int]:
        text = cfg.prompt_template.format(
            benchmark=benchmark,
            condition=condition,
            subject_content=sb_text,
            item_content=it_text,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, len(ids)

    it_text = item_content
    sb_text = subject_content
    text, n_tokens = _attempt(it_text, sb_text)

    # Phase 1: shrink item until it fits or we hit a floor.
    while n_tokens > budget and len(it_text) > 64:
        # Estimate the overshoot in chars: rough rule-of-thumb 4 chars/token.
        overshoot_tokens = n_tokens - budget
        shrink_chars = max(64, overshoot_tokens * 4)
        it_text = it_text[: max(64, len(it_text) - shrink_chars)]
        text, n_tokens = _attempt(it_text, sb_text)

    # Phase 2: shrink subject if we still don't fit.
    while n_tokens > budget and len(sb_text) > 64:
        overshoot_tokens = n_tokens - budget
        shrink_chars = max(64, overshoot_tokens * 4)
        sb_text = sb_text[: max(64, len(sb_text) - shrink_chars)]
        text, n_tokens = _attempt(it_text, sb_text)

    # Last resort: hard-truncate the rendered prompt by token id.
    if n_tokens > cfg.max_prompt_tokens:
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = int(cfg.max_prompt_tokens) - 8
        ids = ids[:keep]
        text = tokenizer.decode(ids, skip_special_tokens=False) + "\nAnswer:"

    return text


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Local LLM-as-judge. Loaded once at module scope, batched aggressively.

    Reads next-token logits at the answer position and returns
    ``(lp_yes, lp_no, lp_diff, p_yes_renormalized)`` per (subject, item).
    Token variants are summed in probability space before renormalizing.
    """

    def __init__(self, cfg: JudgeConfig, *, device: str | None = None):
        self.cfg = cfg
        self.device = device or _default_device()
        self._tok = None
        self._model = None
        self._attn_impl: str = "sdpa"
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []
        self._loaded: bool = False
        self._wall_time_seconds: float = 0.0
        self._n_scored: int = 0

    # ---------------------------------------------------------------- init

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def slug(self) -> str:
        return judge_slug(self.cfg)

    @property
    def prompt_version(self) -> str:
        return compute_prompt_version(self.cfg)

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        LOG.info(
            "Loading judge tokenizer %s (use_fast=True, trust_remote_code=%s)",
            self.cfg.model_id,
            self.cfg.trust_remote_code,
        )
        tok = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
            use_fast=True,
        )
        # Right-padding is mandatory: we extract the answer-position via
        # attention_mask.sum(dim=1) - 1, which only works under right-padding.
        if tok.padding_side != "right":
            LOG.warning(
                "Judge tokenizer defaulted to padding_side=%r; forcing 'right'",
                tok.padding_side,
            )
            tok.padding_side = "right"
        if tok.pad_token_id is None:
            # Use EOS as PAD when the tokenizer has no dedicated pad token.
            tok.pad_token = tok.eos_token
        self._tok = tok

        dtype = (
            torch.bfloat16
            if (self.cfg.bf16 and self.device.startswith("cuda") and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        kwargs: dict = {
            "torch_dtype": dtype,
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        loaded = False
        if self.cfg.use_flash_attention and self.device.startswith("cuda"):
            try:
                LOG.info("Loading judge %s with Flash Attention 2", self.cfg.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id,
                    attn_implementation="flash_attention_2",
                    **kwargs,
                )
                self._attn_impl = "flash_attention_2"
                loaded = True
            except (ImportError, ValueError, RuntimeError) as exc:
                LOG.warning(
                    "Flash Attention 2 unavailable for judge (%s: %s); falling back to SDPA",
                    type(exc).__name__,
                    exc,
                )
        if not loaded:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id, attn_implementation="sdpa", **kwargs
                )
                self._attn_impl = "sdpa"
            except (TypeError, ValueError):
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id, **kwargs
                )
                self._attn_impl = "default"
        self._model.eval()
        self._model.to(self.device)

        self._yes_ids = _resolve_token_ids(self._tok, self.cfg.yes_tokens)
        self._no_ids = _resolve_token_ids(self._tok, self.cfg.no_tokens)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(
                f"Judge yes/no tokens did not resolve: yes={self._yes_ids} no={self._no_ids}. "
                "Check JudgeConfig.yes_tokens / no_tokens or the tokenizer."
            )
        self._loaded = True
        LOG.info(
            "Judge loaded on %s in %.1fs (attn=%s, yes_ids=%s, no_ids=%s)",
            self.device,
            time.time() - t0,
            self._attn_impl,
            self._yes_ids,
            self._no_ids,
        )

    # --------------------------------------------------------------- score

    def score(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        """Score a list of dicts in batches. Returns ``[N, 4]`` float32.

        Each row must contain ``benchmark``, ``condition``, ``subject_content``,
        ``item_content``. This bypasses the cache: ``score_dataframe`` is
        the public entry point that consults the cache before calling us.
        """
        if not rows:
            return np.zeros((0, JUDGE_FEATURE_DIM), dtype=np.float32)
        self._load()
        import torch

        n = len(rows)
        out = np.zeros((n, JUDGE_FEATURE_DIM), dtype=np.float32)
        bs = max(1, int(self.cfg.batch_size))

        t0 = time.time()

        for i in range(0, n, bs):
            chunk = rows[i : i + bs]
            prompts = [
                _truncate_prompt(
                    self._tok,
                    self.cfg,
                    benchmark=str(r.get("benchmark", "") or ""),
                    condition=str(r.get("condition", "none") or "none"),
                    subject_content=str(r.get("subject_content", "") or ""),
                    item_content=str(r.get("item_content", "") or ""),
                )
                for r in chunk
            ]
            enc = self._tok(
                prompts,
                padding="longest",
                truncation=True,
                max_length=self.cfg.max_prompt_tokens,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            with torch.inference_mode():
                outputs = self._model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # [B, T, V]

            # Right-padded: the answer position is the last real token. We
            # read logits[:, last] = next-token distribution at the answer.
            seq_lens = attn.sum(dim=1) - 1
            seq_lens = seq_lens.clamp(min=0)
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            next_logits = logits[batch_idx, seq_lens]   # [B, V]
            logprobs = torch.log_softmax(next_logits.float(), dim=-1)

            yes_ids = torch.tensor(self._yes_ids, dtype=torch.long, device=logprobs.device)
            no_ids = torch.tensor(self._no_ids, dtype=torch.long, device=logprobs.device)
            # logsumexp across variants -- this is exactly "sum exp(lp), take log"
            log_sum_yes = torch.logsumexp(logprobs.index_select(1, yes_ids), dim=-1)
            log_sum_no = torch.logsumexp(logprobs.index_select(1, no_ids), dim=-1)
            # Renormalize over the two-class restricted distribution.
            # p_yes = exp(log_sum_yes) / (exp(log_sum_yes) + exp(log_sum_no))
            #       = sigmoid(log_sum_yes - log_sum_no)
            lp_diff = log_sum_yes - log_sum_no
            p_yes = torch.sigmoid(lp_diff)

            out[i : i + len(chunk), 0] = log_sum_yes.float().cpu().numpy()
            out[i : i + len(chunk), 1] = log_sum_no.float().cpu().numpy()
            out[i : i + len(chunk), 2] = lp_diff.float().cpu().numpy()
            out[i : i + len(chunk), 3] = p_yes.float().cpu().numpy()

        self._wall_time_seconds += time.time() - t0
        self._n_scored += n
        return out

    # -------------------------------------------------------- score_dataframe

    def score_dataframe(
        self,
        df: pd.DataFrame,
        *,
        write_cache: bool = True,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Score every (subject_key, item_key) in ``df`` exactly once.

        ``df`` must contain ``subject_key``, ``item_key``, ``benchmark``,
        ``condition``, ``subject_content``, ``item_content``. Returns the
        original frame with four ``judge_*`` columns appended; uses an
        on-disk parquet cache so re-running is cheap.

        If the cache already contains every needed pair the judge model is
        never loaded (this matters when ``judge.enabled = false`` -- there's
        nothing left to score).
        """
        required = (
            "subject_key",
            "item_key",
            "benchmark",
            "condition",
            "subject_content",
            "item_content",
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"score_dataframe: missing required columns {missing}")

        # 1) Resolve uniques.
        uniques = df[["subject_key", "item_key"]].drop_duplicates().reset_index(drop=True)
        unique_pairs = list(
            zip(uniques["subject_key"].astype(str), uniques["item_key"].astype(str))
        )
        # 2) Look up the cache.
        cached = load_cache(self.cfg)
        cached_pairs = set()
        if not cached.empty:
            cached_pairs = set(
                zip(
                    cached["subject_key"].astype(str).tolist(),
                    cached["item_key"].astype(str).tolist(),
                )
            )

        # 3) Determine what's still missing.
        to_score_mask = [pair not in cached_pairs for pair in unique_pairs]
        missing_uniques = uniques.loc[to_score_mask].reset_index(drop=True)

        n_cached = len(unique_pairs) - len(missing_uniques)
        n_to_score = len(missing_uniques)

        LOG.info(
            "judge.score_dataframe: %d unique pairs (cached=%d, to_score=%d) slug=%s",
            len(unique_pairs),
            n_cached,
            n_to_score,
            self.slug,
        )

        new_rows_df: pd.DataFrame | None = None
        if n_to_score > 0:
            # Build the input rows for the missing pairs. We pull the *first*
            # occurrence of each (subject_key, item_key) since the four text
            # fields are deterministic given the pair (validated upstream).
            first_idx = (
                df.drop_duplicates(subset=["subject_key", "item_key"], keep="first")
                .set_index(["subject_key", "item_key"])
            )
            keep = first_idx.loc[
                pd.MultiIndex.from_frame(missing_uniques)
            ].reset_index()
            rows = keep[
                ["benchmark", "condition", "subject_content", "item_content"]
            ].to_dict(orient="records")

            # Score in batches with tqdm wrapping the chunks.
            self._load()
            chunks = self._batched_score_with_progress(rows, progress=progress)
            scored = np.concatenate(chunks, axis=0) if chunks else np.zeros(
                (0, JUDGE_FEATURE_DIM), dtype=np.float32
            )
            new_rows_df = pd.DataFrame(
                {
                    "subject_key": missing_uniques["subject_key"].astype(str).values,
                    "item_key": missing_uniques["item_key"].astype(str).values,
                    "lp_yes": scored[:, 0].astype(np.float32),
                    "lp_no": scored[:, 1].astype(np.float32),
                    "lp_diff": scored[:, 2].astype(np.float32),
                    "p_yes_renorm": scored[:, 3].astype(np.float32),
                    "judge_model_id": self.cfg.model_id,
                    "prompt_version": self.prompt_version,
                }
            )
            if write_cache:
                append_cache(self.cfg, new_rows_df)

        # 4) Build the per-row feature frame by joining cache+new on (s,i).
        if new_rows_df is None or new_rows_df.empty:
            combined = cached
        elif cached.empty:
            combined = new_rows_df
        else:
            combined = pd.concat([cached, new_rows_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["subject_key", "item_key"], keep="last"
            )

        # Filter to current model+version to avoid contamination.
        prompt_version = self.prompt_version
        mask = (combined["judge_model_id"] == self.cfg.model_id) & (
            combined["prompt_version"] == prompt_version
        )
        combined = combined.loc[mask, list(JUDGE_CACHE_COLUMNS)].reset_index(drop=True)

        out = df.merge(
            combined[
                ["subject_key", "item_key", "lp_yes", "lp_no", "lp_diff", "p_yes_renorm"]
            ],
            on=["subject_key", "item_key"],
            how="left",
        )
        return out

    def _batched_score_with_progress(
        self, rows: list[dict], *, progress: bool
    ) -> list[np.ndarray]:
        """Run ``score`` in batches, optionally wrapping with tqdm."""
        bs = max(1, int(self.cfg.batch_size))
        n = len(rows)
        chunks: list[np.ndarray] = []
        starts = list(range(0, n, bs))
        iterator = tqdm(
            starts,
            desc=f"judge {self.cfg.model_id.split('/')[-1]}",
            unit="batch",
            leave=False,
            disable=not progress,
        )
        for s in iterator:
            chunk = rows[s : s + bs]
            scored = self.score(chunk)
            chunks.append(scored)
        return chunks

    # --------------------------------------------------------------- stats

    def stats(self) -> dict:
        """Cheap diagnostic snapshot for the notebook to print."""
        return {
            "model_id": self.cfg.model_id,
            "slug": self.slug,
            "prompt_version": self.prompt_version,
            "loaded": bool(self._loaded),
            "attn_implementation": self._attn_impl,
            "n_scored_this_session": int(self._n_scored),
            "wall_time_seconds": float(self._wall_time_seconds),
        }


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Convenience: build a lookup dict from the cache for the notebook
# ---------------------------------------------------------------------------


def features_lookup_from_dataframe(df: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Turn a ``score_dataframe`` output into a ``(s_key, i_key) -> [4]`` dict.

    Used by the notebook to project judge features onto the training matrix
    keyed by ``(subject_key, item_key)``. Unknown pairs at val time fall
    back to zeros (and the residual MLP's input LayerNorm absorbs the shift).
    """
    if df.empty or "lp_yes" not in df.columns:
        return {}
    sub = df[["subject_key", "item_key", "lp_yes", "lp_no", "lp_diff", "p_yes_renorm"]]
    sub = sub.drop_duplicates(subset=["subject_key", "item_key"], keep="last")
    out: dict[tuple[str, str], np.ndarray] = {}
    for sk, ik, lp_y, lp_n, lp_d, p_y in zip(
        sub["subject_key"].astype(str),
        sub["item_key"].astype(str),
        sub["lp_yes"].astype(float),
        sub["lp_no"].astype(float),
        sub["lp_diff"].astype(float),
        sub["p_yes_renorm"].astype(float),
    ):
        out[(sk, ik)] = np.asarray([lp_y, lp_n, lp_d, p_y], dtype=np.float32)
    return out


def build_judge_matrix(
    subject_keys: Sequence[str],
    item_keys: Sequence[str],
    lookup: Mapping[tuple[str, str], np.ndarray],
) -> np.ndarray:
    """Stack ``[N, 4]`` per-row judge features in dataframe order.

    Pairs missing from ``lookup`` get the zero vector. The residual MLP's
    input LayerNorm absorbs the shift, and the model can learn to ignore
    rows with that fingerprint.
    """
    n = len(subject_keys)
    out = np.zeros((n, JUDGE_FEATURE_DIM), dtype=np.float32)
    for i, (sk, ik) in enumerate(zip(subject_keys, item_keys)):
        v = lookup.get((str(sk), str(ik)))
        if v is not None:
            out[i] = np.asarray(v, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# JSON snapshot for the cache directory (useful for drive cache hash check)
# ---------------------------------------------------------------------------


def write_meta_snapshot(cfg: JudgeConfig, *, n_rows: int) -> Path:
    """Persist a ``meta.json`` next to ``scores.parquet`` describing the slug.

    The Drive cache helper uses ``content_hash`` (here, the
    ``model_id + prompt_version``) to decide whether a cached parquet from
    Drive matches the current config.
    """
    path = _cache_path(cfg).parent / "meta.json"
    payload = {
        "judge_model_id": cfg.model_id,
        "prompt_version": compute_prompt_version(cfg),
        "n_rows": int(n_rows),
        "yes_tokens": list(cfg.yes_tokens),
        "no_tokens": list(cfg.no_tokens),
        "max_prompt_tokens": int(cfg.max_prompt_tokens),
        "bf16": bool(cfg.bf16),
        "use_flash_attention": bool(cfg.use_flash_attention),
        "content_hash": compute_prompt_version(cfg),
        "slug": judge_slug(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_meta_snapshot(cfg: JudgeConfig) -> dict:
    path = _cache_path(cfg).parent / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


__all__ = [
    "JUDGE_CACHE_COLUMNS",
    "JUDGE_FEATURE_DIM",
    "JudgeConfig",
    "LLMJudge",
    "append_cache",
    "build_judge_matrix",
    "compute_prompt_version",
    "features_lookup_from_dataframe",
    "judge_slug",
    "load_cache",
    "read_meta_snapshot",
    "write_meta_snapshot",
]
