"""Solver-based *item-side* difficulty proxies (the orthogonal-signal probe).

The production stack reads items with a frozen encoder; it never *solves* them.
This module has a small local model actually attempt each item and distils the
attempt into a handful of item-level scalars that the encoder structurally
cannot represent:

    self_consistency   modal-answer vote fraction over n sampled CoT solutions
    answer_entropy     Shannon entropy of the answer-vote distribution (norm.)
    fsd                first-second distance: top-1 minus top-2 vote share
    n_distinct         distinct parsed answers / n_samples
    mean_trace_len     mean CoT length in tokens (a difficulty proxy)
    refusal_rate       fraction of samples with no parseable answer
    p_true             P(model says its own modal answer is correct)  [optional]

These are *solvability* signals (Tier-1 self-consistency + Tier-2 P(True) in
the literature), which the evidence says separate correct from incorrect and
transfer across models -- unlike single-pass token-likelihood (the old judge).

Multi-model use: run two/three *different-family* solvers and combine their
modal answers into a cross-model *disagreement* score, which estimates item
**discrimination** (a_i) rather than difficulty (see ``cross_model_disagreement``).

Caching mirrors ``src.judge``: per (model_id, config_hash, item_key) rows in a
parquet, so re-running is free and only the round's unseen items cost a pass.

The GPU class is intentionally thin and modelled on the proven batching path in
``src.judge``; the scientific content lives in the pure helpers below, which
are fully unit-tested without a GPU.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("solver_proxy")

PROXY_FEATURE_NAMES: tuple[str, ...] = (
    "self_consistency",
    "answer_entropy",
    "fsd",
    "n_distinct",
    "mean_trace_len",
    "refusal_rate",
    "p_true",
)


# ---------------------------------------------------------------------------
# Pure helpers: answer parsing + vote statistics (GPU-free, unit-tested)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|the\s+answer\s+is)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)
_MC_RE = re.compile(r"\b([A-E])\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NO_ANSWER = "<none>"


def normalize_answer(text: str) -> str:
    """Best-effort extraction of a comparable final-answer token from a CoT.

    Priority: ``\\boxed{...}`` > an "answer:"-style tail > a lone MC letter on
    the last line > the last number > the last non-empty line. Returns a
    lowercased, whitespace-collapsed token, or ``"<none>"`` when nothing
    parseable is found (a refusal / runaway generation).
    """
    if not text:
        return _NO_ANSWER
    s = str(text).strip()
    if not s:
        return _NO_ANSWER

    boxed = _BOXED_RE.findall(s)
    if boxed:
        return _clean_token(boxed[-1])

    # Look at the tail first: final answers live near the end of a CoT.
    tail = "\n".join(s.splitlines()[-4:])
    m = _ANSWER_RE.search(tail) or _ANSWER_RE.search(s)
    if m:
        cand = m.group(1).strip()
        # Prefer a boxed/number/MC inside the captured tail.
        b2 = _BOXED_RE.findall(cand)
        if b2:
            return _clean_token(b2[-1])
        nums = _NUM_RE.findall(cand)
        if nums:
            return _clean_token(nums[-1])
        mc = _MC_RE.findall(cand[:8])  # MC letter at the very start of answer
        if mc:
            return _clean_token(mc[0])
        return _clean_token(cand)

    last_line = next(
        (ln for ln in reversed(s.splitlines()) if ln.strip()), ""
    )
    mc = _MC_RE.findall(last_line.strip())
    if mc and len(last_line.strip()) <= 3:
        return _clean_token(mc[0])
    nums = _NUM_RE.findall(s)
    if nums:
        return _clean_token(nums[-1])
    return _clean_token(last_line) if last_line else _NO_ANSWER


def _clean_token(tok: str) -> str:
    t = re.sub(r"\s+", " ", str(tok).strip().lower())
    t = t.strip(" .,:;$\\")
    # Canonicalize trivially-equal numerics (e.g. "4.0" -> "4").
    if re.fullmatch(r"-?\d+\.0+", t):
        t = t.split(".")[0]
    return t if t else _NO_ANSWER


def vote_statistics(answers: Sequence[str]) -> dict[str, float]:
    """Vote-distribution stats over parsed answers (one CoT sample each).

    Returns ``self_consistency`` (modal share), ``answer_entropy`` (Shannon
    entropy normalized to [0,1] by log(n_samples)), ``fsd`` (top1-top2 share),
    ``n_distinct`` (distinct/n), and ``refusal_rate`` (share of ``"<none>"``).
    Entropy/consistency are computed over the *non-refusal* answers so a model
    that refuses half the time isn't scored as "consistent" on the rest by
    accident -- but refusal_rate is reported separately so the head can use it.
    """
    n = len(answers)
    if n == 0:
        return {
            "self_consistency": 0.0,
            "answer_entropy": 1.0,
            "fsd": 0.0,
            "n_distinct": 1.0,
            "refusal_rate": 1.0,
        }
    refusals = sum(1 for a in answers if a == _NO_ANSWER)
    valid = [a for a in answers if a != _NO_ANSWER]
    refusal_rate = refusals / n
    if not valid:
        return {
            "self_consistency": 0.0,
            "answer_entropy": 1.0,
            "fsd": 0.0,
            "n_distinct": 1.0,
            "refusal_rate": refusal_rate,
        }
    counts = Counter(valid)
    m = len(valid)
    shares = sorted((c / m for c in counts.values()), reverse=True)
    top1 = shares[0]
    top2 = shares[1] if len(shares) > 1 else 0.0
    ent = -sum(p * math.log(p) for p in shares)
    ent_norm = ent / math.log(m) if m > 1 else 0.0
    return {
        "self_consistency": float(top1),
        "answer_entropy": float(min(1.0, max(0.0, ent_norm))),
        "fsd": float(top1 - top2),
        "n_distinct": float(len(counts) / n),
        "refusal_rate": float(refusal_rate),
    }


def modal_answer(answers: Sequence[str]) -> str:
    """Most-common non-refusal parsed answer (``"<none>"`` if all refuse)."""
    valid = [a for a in answers if a != _NO_ANSWER]
    if not valid:
        return _NO_ANSWER
    return Counter(valid).most_common(1)[0][0]


def cross_model_disagreement(
    modal_by_model: Mapping[str, Mapping[str, str]],
) -> dict[str, float]:
    """Per-item cross-model disagreement in ``[0,1]`` (estimates a_i).

    ``modal_by_model[model_id][item_key] = modal_answer``. For each item we
    compute ``1 - (largest agreeing fraction across the models that produced a
    non-refusal answer)``. 0 == all models agree (easy / low-discrimination),
    1 == every model gives a different answer (hard / high-discrimination).
    Items with <2 non-refusal models get ``nan`` (uninformative).
    """
    models = list(modal_by_model.keys())
    all_items: set[str] = set()
    for m in models:
        all_items.update(modal_by_model[m].keys())
    out: dict[str, float] = {}
    for ik in all_items:
        votes = [
            modal_by_model[m][ik]
            for m in models
            if ik in modal_by_model[m] and modal_by_model[m][ik] != _NO_ANSWER
        ]
        if len(votes) < 2:
            out[ik] = float("nan")
            continue
        top = Counter(votes).most_common(1)[0][1]
        out[ik] = float(1.0 - top / len(votes))
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SolverProxyConfig:
    """Hyperparameters for one solver model's proxy extraction."""

    model_id: str = "Qwen/Qwen3-8B"
    n_samples: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    max_prompt_tokens: int = 1024
    batch_size: int = 16
    seed: int = 0
    bf16: bool = True
    use_flash_attention: bool = True
    trust_remote_code: bool = False
    compute_p_true: bool = True
    cache_dir: str = "artifacts/solver_proxy"
    solve_template: str = (
        "Solve the following evaluation item. Reason step by step, then state "
        "your final answer on the last line as 'Answer: <answer>'.\n\n"
        "Benchmark: {benchmark}\n"
        "Item: {item_content}\n\n"
        "Solution:"
    )
    verify_template: str = (
        "Item: {item_content}\n"
        "Proposed answer: {answer}\n"
        "Is the proposed answer correct? Reply with a single token: yes or no.\n"
        "Answer:"
    )
    yes_tokens: tuple[str, ...] = (" yes", "yes", " Yes", "Yes")
    no_tokens: tuple[str, ...] = (" no", "no", " No", "No")


def config_hash(cfg: SolverProxyConfig) -> str:
    """Stable hash of everything that changes the produced scores."""
    h = hashlib.sha256()
    for part in (
        cfg.solve_template, cfg.verify_template,
        str(cfg.n_samples), str(round(cfg.temperature, 4)), str(round(cfg.top_p, 4)),
        str(cfg.max_new_tokens), str(cfg.max_prompt_tokens), str(cfg.seed),
        str(cfg.compute_p_true),
        "|".join(cfg.yes_tokens), "|".join(cfg.no_tokens),
    ):
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _slugify(model_id: str) -> str:
    return model_id.replace("/", "__")[:200]


def proxy_slug(cfg: SolverProxyConfig) -> str:
    return f"{_slugify(cfg.model_id)}__{config_hash(cfg)}"


def _cache_path(cfg: SolverProxyConfig) -> Path:
    base = Path(cfg.cache_dir) / proxy_slug(cfg)
    base.mkdir(parents=True, exist_ok=True)
    return base / "proxy.parquet"


PROXY_CACHE_COLUMNS: tuple[str, ...] = (
    ("item_key",) + PROXY_FEATURE_NAMES + ("modal_answer", "model_id", "config_hash")
)


def load_cache(cfg: SolverProxyConfig) -> pd.DataFrame:
    path = _cache_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=list(PROXY_CACHE_COLUMNS))
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to read solver-proxy cache %s: %s", path, exc)
        return pd.DataFrame(columns=list(PROXY_CACHE_COLUMNS))
    ch = config_hash(cfg)
    if "config_hash" in df.columns:
        df = df[(df["model_id"] == cfg.model_id) & (df["config_hash"] == ch)]
    return df.reset_index(drop=True)


def append_cache(cfg: SolverProxyConfig, new_rows: pd.DataFrame) -> Path:
    path = _cache_path(cfg)
    if new_rows.empty:
        return path
    existing = load_cache(cfg)
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["item_key"], keep="last")
    tmp = path.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# GPU solver (thin; mirrors src.judge load/batch patterns)
# ---------------------------------------------------------------------------


class SolverProxy:
    """Load one HF causal LM, sample n CoT solutions per item, distil scalars.

    Only the round's *unseen* items cost a forward pass; everything is keyed by
    ``item_key`` in a parquet cache. The heavy generation path is deliberately
    simple HF ``.generate`` batching -- swap in vLLM later if throughput bites.
    """

    def __init__(self, cfg: SolverProxyConfig, *, device: str | None = None):
        self.cfg = cfg
        self.device = device or _default_device()
        self._tok = None
        self._model = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []
        self._loaded = False

    @property
    def slug(self) -> str:
        return proxy_slug(self.cfg)

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=self.cfg.trust_remote_code, use_fast=True
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # Left-pad for batched generation so completions align at the right edge.
        tok.padding_side = "left"
        self._tok = tok
        dtype = (
            torch.bfloat16
            if (self.cfg.bf16 and self.device.startswith("cuda") and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        kwargs = {"torch_dtype": dtype, "trust_remote_code": self.cfg.trust_remote_code}
        loaded = False
        if self.cfg.use_flash_attention and self.device.startswith("cuda"):
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id, attn_implementation="flash_attention_2", **kwargs
                )
                loaded = True
            except (ImportError, ValueError, RuntimeError) as exc:
                LOG.warning("FlashAttention2 unavailable (%s); using SDPA", exc)
        if not loaded:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id, attn_implementation="sdpa", **kwargs
            )
        self._model.eval()
        self._model.to(self.device)
        self._yes_ids = _resolve_token_ids(self._tok, self.cfg.yes_tokens)
        self._no_ids = _resolve_token_ids(self._tok, self.cfg.no_tokens)
        self._loaded = True
        LOG.info("Solver %s loaded on %s (slug=%s)", self.cfg.model_id, self.device, self.slug)

    def _build_solve_prompt(self, row: Mapping[str, object]) -> str:
        text = self.cfg.solve_template.format(
            benchmark=str(row.get("benchmark", "") or ""),
            item_content=str(row.get("item_content", "") or ""),
        )
        return text

    def _generate_samples(self, prompts: Sequence[str]) -> list[list[str]]:
        """Return ``n_samples`` decoded completions for each prompt."""
        import torch

        n = len(prompts)
        out: list[list[str]] = [[] for _ in range(n)]
        bs = max(1, int(self.cfg.batch_size))
        for s in range(0, n, bs):
            chunk = list(prompts[s : s + bs])
            enc = self._tok(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=int(self.cfg.max_prompt_tokens), add_special_tokens=True,
            ).to(self.device)
            with torch.inference_mode():
                gen = self._model.generate(
                    **enc,
                    do_sample=True,
                    temperature=float(self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    max_new_tokens=int(self.cfg.max_new_tokens),
                    num_return_sequences=int(self.cfg.n_samples),
                    pad_token_id=self._tok.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            completions = gen[:, prompt_len:]
            decoded = self._tok.batch_decode(completions, skip_special_tokens=True)
            # decoded is [chunk * n_samples] grouped per prompt.
            ns = int(self.cfg.n_samples)
            for j in range(len(chunk)):
                out[s + j] = decoded[j * ns : (j + 1) * ns]
        return out

    def _p_true(self, rows: Sequence[Mapping[str, object]], modal: Sequence[str]) -> np.ndarray:
        """One verification forward pass; read yes/no logits at the answer slot."""
        import torch

        n = len(rows)
        res = np.full(n, 0.5, dtype=np.float64)
        if not self._yes_ids or not self._no_ids:
            return res
        # Right-pad for next-token logit reads (mirror judge.py).
        prev_side = self._tok.padding_side
        self._tok.padding_side = "right"
        try:
            bs = max(1, int(self.cfg.batch_size))
            for s in range(0, n, bs):
                chunk_rows = rows[s : s + bs]
                prompts = [
                    self.cfg.verify_template.format(
                        item_content=str(r.get("item_content", "") or ""),
                        answer=str(modal[s + j]),
                    )
                    for j, r in enumerate(chunk_rows)
                ]
                enc = self._tok(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=int(self.cfg.max_prompt_tokens), add_special_tokens=True,
                ).to(self.device)
                with torch.inference_mode():
                    logits = self._model(**enc, use_cache=False).logits
                seq_lens = enc["attention_mask"].sum(dim=1) - 1
                bidx = torch.arange(logits.size(0), device=logits.device)
                nxt = torch.log_softmax(logits[bidx, seq_lens].float(), dim=-1)
                yi = torch.tensor(self._yes_ids, device=logits.device)
                ni = torch.tensor(self._no_ids, device=logits.device)
                lp_yes = torch.logsumexp(nxt.index_select(1, yi), dim=-1)
                lp_no = torch.logsumexp(nxt.index_select(1, ni), dim=-1)
                res[s : s + len(chunk_rows)] = torch.sigmoid(lp_yes - lp_no).cpu().numpy()
        finally:
            self._tok.padding_side = prev_side
        return res

    def score_items(
        self, item_rows: pd.DataFrame, *, write_cache: bool = True, progress: bool = True
    ) -> pd.DataFrame:
        """Score unique items in ``item_rows`` (needs ``item_key``, ``benchmark``,
        ``item_content``). Returns a per-item frame with ``PROXY_FEATURE_NAMES``
        + ``modal_answer``. Cached rows are never re-generated.
        """
        req = {"item_key", "benchmark", "item_content"}
        missing = req - set(item_rows.columns)
        if missing:
            raise ValueError(f"score_items: missing columns {sorted(missing)}")
        uniq = item_rows.drop_duplicates(subset=["item_key"]).reset_index(drop=True)
        cached = load_cache(self.cfg)
        cached_keys = set(cached["item_key"].astype(str)) if not cached.empty else set()
        todo = uniq[~uniq["item_key"].astype(str).isin(cached_keys)].reset_index(drop=True)
        LOG.info(
            "solver_proxy[%s]: %d unique items (cached=%d, to_score=%d)",
            self.cfg.model_id, len(uniq), len(uniq) - len(todo), len(todo),
        )
        if len(todo) > 0:
            self._load()
            rows = todo.to_dict(orient="records")
            prompts = [self._build_solve_prompt(r) for r in rows]
            try:
                from tqdm.auto import tqdm
                rng = tqdm(range(0, len(rows), self.cfg.batch_size), disable=not progress,
                           desc=f"solve {self.cfg.model_id.split('/')[-1]}")
            except Exception:
                rng = range(0, len(rows), self.cfg.batch_size)
            feats: list[dict] = []
            modal_list: list[str] = []
            samples_all = self._generate_samples(prompts)
            tok = self._tok
            for i, samples in enumerate(samples_all):
                parsed = [normalize_answer(t) for t in samples]
                vs = vote_statistics(parsed)
                lens = [len(tok.encode(t, add_special_tokens=False)) for t in samples]
                vs["mean_trace_len"] = float(np.mean(lens)) if lens else 0.0
                feats.append(vs)
                modal_list.append(modal_answer(parsed))
            if self.cfg.compute_p_true:
                p_true = self._p_true(rows, modal_list)
            else:
                p_true = np.full(len(rows), 0.5, dtype=np.float64)
            new_df = pd.DataFrame({
                "item_key": todo["item_key"].astype(str).values,
                "self_consistency": [f["self_consistency"] for f in feats],
                "answer_entropy": [f["answer_entropy"] for f in feats],
                "fsd": [f["fsd"] for f in feats],
                "n_distinct": [f["n_distinct"] for f in feats],
                "mean_trace_len": [f["mean_trace_len"] for f in feats],
                "refusal_rate": [f["refusal_rate"] for f in feats],
                "p_true": p_true,
                "modal_answer": modal_list,
                "model_id": self.cfg.model_id,
                "config_hash": config_hash(self.cfg),
            })
            if write_cache:
                append_cache(self.cfg, new_df)
            cached = load_cache(self.cfg)
        # Return rows for the requested items, in input order.
        want = uniq[["item_key"]].astype(str)
        out = want.merge(cached, on="item_key", how="left")
        return out

    def release(self) -> None:
        if not self._loaded:
            return
        try:
            import gc
            import torch
            if self._model is not None:
                self._model.to("cpu")
            self._model = None
            self._tok = None
            self._loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "SolverProxy":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_token_ids(tokenizer, variants: Sequence[str]) -> list[int]:
    seen: dict[int, None] = {}
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if ids:
            seen.setdefault(int(ids[0]), None)
    return list(seen.keys())


def build_proxy_row_vector(
    item_key_per_row: Sequence[str],
    per_item: Mapping[str, float],
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """Broadcast a per-item proxy scalar onto a per-row vector (missing->fill)."""
    return np.fromiter(
        (float(per_item.get(str(k), fill)) for k in item_key_per_row),
        count=len(item_key_per_row), dtype=np.float64,
    )


__all__ = [
    "PROXY_CACHE_COLUMNS",
    "PROXY_FEATURE_NAMES",
    "SolverProxy",
    "SolverProxyConfig",
    "append_cache",
    "build_proxy_row_vector",
    "config_hash",
    "cross_model_disagreement",
    "load_cache",
    "modal_answer",
    "normalize_answer",
    "proxy_slug",
    "vote_statistics",
]
