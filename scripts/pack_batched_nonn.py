"""Pack a batched-flush bundle that does NOT use NN features.

Surgical add-on to ``submission_cacheless.zip``: keep that bundle's
``model.py`` byte-for-byte (it's the same code as the working slim
template at the bs=1 path) and inject the minimal batching plumbing
strictly via additions -- no modifications to the existing functions.

The five injections, in order:

  1. ``score_batch`` method into ``_LLMJudgeRuntime`` (right before
     ``def score_cached``) -- batched judge forward in chunks of
     ``JUDGE_RUNTIME_BATCH_SIZE``.

  2. Module-level batched plumbing after the ``JUDGE: ... | None = None``
     line:
       - ``ENCODER_RUNTIME_BATCH_SIZE`` (default 4 for L4 co-resident)
       - ``JUDGE_RUNTIME_BATCH_SIZE``   (default 8 for L4 co-resident)
       - three pending queues + three pending-key dedup sets
       - ``_FLUSHED`` sentinel
       - ``_enqueue_for_batch``  (called by labeling.py per candidate)
       - ``_flush_pending_batches`` (called by predict on first call)

  3. ``_embed_batch`` function right after ``_embed_one``.

  4. A single line at the top of ``predict()``:
     ``    _flush_pending_batches()``
     so the first ``predict()`` call drains the queue before any
     per-pair lookup.

  5. Replace ``labeling.py`` with an enqueue-only acquisition function
     that calls ``_enqueue_for_batch`` and returns 0.0 (constant rank
     -> platform's random tie-break, identical to not shipping
     labeling.py at all from the platform's selection-of-K perspective).

Critical simplifications relative to the previous batched bundles
(the ones that all failed PAIEC-UNKNOWN-001 in 1-5 minutes):

  - NO ``_write_progress`` calls.  Zero disk I/O on the hot path.
  - NO ``_free_encoder_vram`` / encoder eviction.  Both 4B-param bf16
    models stay co-resident on the L4; batch sizes are sized for that.
  - NO new ``attn_implementation`` paths (we inherit the cacheless
    bundle's existing ``flash_attention_2 -> default-eager`` fallback,
    which is the same code path the working slim bundle used).
  - NO FAISS / LoRA / cache-loader changes.  ``ship_training_cache``
    defaults to False, ``submission/cache/`` is not shipped, and
    ``TRAINING_CACHE`` stays ``None`` -- NN features are zero by
    construction.

Net: the batched ``model.py`` differs from the cacheless ``model.py``
by exactly ~150 lines of pure additions, with one one-line edit inside
``predict()``.  Easy to bisect further if this still fails.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_batched_nonn.zip")


SCORE_BATCH_METHOD = '''\
    def score_batch(self, rows: list) -> list:
        """Score many ``(benchmark, condition, subject, item)`` rows in
        chunks of ``JUDGE_RUNTIME_BATCH_SIZE``.  Returns one length-4
        float32 vector per row, in the same order, matching the per-row
        ``score`` output (lp_yes, lp_no, lp_diff, p_yes).
        """
        if not rows:
            return []
        B = max(1, int(JUDGE_RUNTIME_BATCH_SIZE))
        out: list = [None] * len(rows)
        for start in range(0, len(rows), B):
            chunk = rows[start : start + B]
            prompts = [
                _judge_truncate(
                    self.tokenizer,
                    benchmark=str(b),
                    condition=str(c),
                    subject_content=str(s),
                    item_content=str(i),
                )
                for (b, c, s, i) in chunk
            ]
            enc = self.tokenizer(
                prompts,
                padding="longest",
                truncation=True,
                max_length=JUDGE_MAX_PROMPT_TOKENS,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(_DEVICE)
            attn = enc["attention_mask"].to(_DEVICE)
            with torch.inference_mode():
                res = self.model(input_ids=input_ids, attention_mask=attn)
            logits = res.logits
            seq_lens = attn.sum(dim=1) - 1
            seq_lens = seq_lens.clamp(min=0)
            next_logits = logits[
                torch.arange(logits.size(0), device=logits.device), seq_lens
            ]
            logprobs = torch.log_softmax(next_logits.float(), dim=-1)
            yes_t = torch.tensor(self.yes_ids, dtype=torch.long, device=logprobs.device)
            no_t = torch.tensor(self.no_ids, dtype=torch.long, device=logprobs.device)
            lp_yes = torch.logsumexp(logprobs.index_select(1, yes_t), dim=-1)
            lp_no = torch.logsumexp(logprobs.index_select(1, no_t), dim=-1)
            lp_diff = lp_yes - lp_no
            p_yes = torch.sigmoid(lp_diff)
            ly = lp_yes.float().cpu().numpy()
            ln = lp_no.float().cpu().numpy()
            ld = lp_diff.float().cpu().numpy()
            py = p_yes.float().cpu().numpy()
            for j in range(len(chunk)):
                out[start + j] = np.asarray(
                    [float(ly[j]), float(ln[j]), float(ld[j]), float(py[j])],
                    dtype=np.float32,
                )
        return out

'''


MODULE_SCOPE_BATCHING = '''

ENCODER_RUNTIME_BATCH_SIZE: int = int(META.get("encoder_runtime_batch_size", 4))
JUDGE_RUNTIME_BATCH_SIZE: int = int(JUDGE_META.get("runtime_batch_size", 8))

_ITEM_PENDING: list[tuple[str, str, str]] = []
_SUBJECT_PENDING: list[str] = []
_JUDGE_PENDING: list[tuple[str, str, str, str]] = []
_ITEM_PENDING_KEYS: set[str] = set()
_SUBJECT_PENDING_KEYS: set[str] = set()
_JUDGE_PENDING_KEYS: set[tuple[str, str]] = set()
_FLUSHED: bool = True


def _enqueue_for_batch(
    *,
    benchmark: str,
    condition: str,
    subject_content: str,
    item_content: str,
) -> None:
    """Queue per-content compute for this candidate.

    Called from ``acquisition_function`` for every candidate the platform
    surfaces.  Deduped against the per-round pending sets *and* the
    persistent caches (``_ITEM_EMB_CACHE``, ``_SUBJECT_EMB_CACHE``,
    ``JUDGE.score_cache``) so we never enqueue work that is already done.
    """
    global _FLUSHED
    benchmark = str(benchmark or "")
    condition = normalize_condition(condition)
    subject_content = str(subject_content or "")
    item_content = str(item_content or "")

    item_key = stable_sha256(benchmark, condition, item_content)
    if item_key not in _ITEM_PENDING_KEYS and item_key not in _ITEM_EMB_CACHE:
        _ITEM_PENDING_KEYS.add(item_key)
        _ITEM_PENDING.append((benchmark, condition, item_content))
        _FLUSHED = False

    if _MODEL_CFG.get("use_subject_text_embedding"):
        s_key = stable_sha256(subject_content)
        if s_key not in _SUBJECT_PENDING_KEYS and s_key not in _SUBJECT_EMB_CACHE:
            _SUBJECT_PENDING_KEYS.add(s_key)
            _SUBJECT_PENDING.append(subject_content)
            _FLUSHED = False

    if JUDGE is not None and _MODEL_CFG.get("use_judge_features"):
        j_key = (stable_sha256(subject_content), item_key)
        if j_key not in _JUDGE_PENDING_KEYS and j_key not in JUDGE.score_cache:
            _JUDGE_PENDING_KEYS.add(j_key)
            _JUDGE_PENDING.append(
                (benchmark, condition, subject_content, item_content)
            )
            _FLUSHED = False


def _flush_pending_batches() -> None:
    """Drain queues into the existing per-content caches.

    Idempotent: returns immediately if nothing is pending. The single
    expensive call is the first ``predict()`` of the round, which
    absorbs the batched compute up front so every subsequent
    ``predict()`` is a pure cache lookup against ``_ITEM_EMB_CACHE`` /
    ``_SUBJECT_EMB_CACHE`` / ``JUDGE.score_cache``.
    """
    global _FLUSHED
    if _FLUSHED:
        return
    if _ITEM_PENDING:
        texts = [item_text_for(b, c, i) for (b, c, i) in _ITEM_PENDING]
        vecs = _embed_batch(texts)
        for (b, c, i), vec in zip(_ITEM_PENDING, vecs):
            _ITEM_EMB_CACHE[stable_sha256(b, c, i)] = vec
        _ITEM_PENDING.clear()
        _ITEM_PENDING_KEYS.clear()
    if _SUBJECT_PENDING:
        texts = [subject_text_for(s) for s in _SUBJECT_PENDING]
        vecs = _embed_batch(texts)
        for s, vec in zip(_SUBJECT_PENDING, vecs):
            _SUBJECT_EMB_CACHE[stable_sha256(s)] = vec
        _SUBJECT_PENDING.clear()
        _SUBJECT_PENDING_KEYS.clear()
    if _JUDGE_PENDING and JUDGE is not None:
        vecs = JUDGE.score_batch(_JUDGE_PENDING)
        for (b, c, s, i), vec in zip(_JUDGE_PENDING, vecs):
            key = (stable_sha256(s), stable_sha256(b, c, i))
            JUDGE.score_cache[key] = vec
        _JUDGE_PENDING.clear()
        _JUDGE_PENDING_KEYS.clear()
    _FLUSHED = True
'''


EMBED_BATCH_FUNCTION = '''

def _embed_batch(texts: list) -> list:
    """Encode many texts in chunks of ``ENCODER_RUNTIME_BATCH_SIZE``.

    Returns one float32 vector per input string, in the same order.
    Uses ``padding="longest"`` within each chunk so wasted compute
    scales with the per-chunk maximum length rather than the global
    one.
    """
    if not texts:
        return []
    out: list = [None] * len(texts)
    B = max(1, int(ENCODER_RUNTIME_BATCH_SIZE))
    for start in range(0, len(texts), B):
        chunk = texts[start : start + B]
        enc = _TOKENIZER(
            chunk,
            padding="longest",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(_DEVICE)
        attn = enc["attention_mask"].to(_DEVICE)
        with torch.inference_mode():
            res = _ENCODER(input_ids=input_ids, attention_mask=attn)
        last_hidden = res.last_hidden_state
        if POOLING == "cls":
            pooled = last_hidden[:, 0]
        elif POOLING == "last_token":
            pooled = _last_token_pool(last_hidden, attn)
        else:
            pooled = _mean_pool(last_hidden, attn)
        vecs = pooled.float().cpu().numpy()
        for i in range(vecs.shape[0]):
            v = vecs[i]
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            out[start + i] = v.astype(np.float32, copy=False)
    return out
'''


ENQUEUE_LABELING_PY = '''"""Adaptive labeling: enqueue-only acquisition.

The platform calls ``acquisition_function`` once per hidden (subject,
item) pair before any ``predict()`` call.  We use that opportunity to
queue all per-content compute through ``model._enqueue_for_batch`` so
the first ``predict()`` of the round can drain the queue via batched
encoder + judge forwards.

Returning 0.0 for every candidate yields a constant ranking; the
platform breaks ties randomly, which is equivalent (in expectation) to
the default random-sample fallback applied when no ``labeling.py`` is
shipped.  The selection-of-K labels the platform reveals are therefore
unchanged in distribution; the only effect of this file is to populate
the batching queues.
"""

from __future__ import annotations


def acquisition_function(input: dict) -> float:  # noqa: A002
    try:
        from model import _enqueue_for_batch  # type: ignore
    except Exception:
        return 0.0
    try:
        _enqueue_for_batch(
            benchmark=str(input.get("benchmark", "") or ""),
            condition=str(input.get("condition", "none") or "none"),
            subject_content=str(input.get("subject_content", "") or ""),
            item_content=str(input.get("item_content", "") or ""),
        )
    except Exception:
        pass
    return 0.0
'''


def transform_model_py(src: str) -> str:
    """Apply the five surgical injections to the cacheless model.py."""

    # ------------------------------------------------------------------
    # 1. Insert ``score_batch`` into ``_LLMJudgeRuntime``, right before
    #    ``def score_cached``.  The cacheless template uses 4-space
    #    indentation for class methods.
    # ------------------------------------------------------------------
    anchor_score_cached = (
        "    def score_cached(self, *, benchmark: str, condition: str, "
        "subject_content: str, item_content: str) -> np.ndarray:"
    )
    if anchor_score_cached not in src:
        raise RuntimeError("anchor for score_batch insertion not found in model.py")
    src = src.replace(
        anchor_score_cached, SCORE_BATCH_METHOD + anchor_score_cached, 1
    )

    # ------------------------------------------------------------------
    # 2. Insert module-scope queues + ``_enqueue_for_batch`` /
    #    ``_flush_pending_batches`` immediately after the
    #    ``JUDGE: _LLMJudgeRuntime | None = None`` declaration.
    # ------------------------------------------------------------------
    anchor_judge_decl = (
        "JUDGE: _LLMJudgeRuntime | None = None"
        "  # populated in the module-init section below"
    )
    if anchor_judge_decl not in src:
        raise RuntimeError("anchor for module-scope batching not found in model.py")
    src = src.replace(
        anchor_judge_decl, anchor_judge_decl + MODULE_SCOPE_BATCHING, 1
    )

    # ------------------------------------------------------------------
    # 3. Insert ``_embed_batch`` right after ``_embed_one`` ends.  We
    #    anchor on the final line of ``_embed_one``'s body
    #    (``return vec.astype(np.float32, copy=False)``) and inject
    #    after the implicit blank line that follows.
    # ------------------------------------------------------------------
    anchor_embed_one_end = "    return vec.astype(np.float32, copy=False)\n"
    occurrences = src.count(anchor_embed_one_end)
    if occurrences < 1:
        raise RuntimeError("anchor for _embed_batch insertion not found in model.py")
    # Replace only the FIRST occurrence (which is inside _embed_one).
    src = src.replace(
        anchor_embed_one_end,
        anchor_embed_one_end + EMBED_BATCH_FUNCTION,
        1,
    )

    # ------------------------------------------------------------------
    # 4. Inject ``_flush_pending_batches()`` at the top of ``predict()``
    #    immediately after the docstring.
    # ------------------------------------------------------------------
    anchor_predict_top = (
        'def predict(input: dict, labeled: list[dict] | None = None) -> float:\n'
        '    """Return a native Python float in (0, 1)."""\n'
    )
    if anchor_predict_top not in src:
        raise RuntimeError("anchor for predict flush call not found in model.py")
    src = src.replace(
        anchor_predict_top,
        anchor_predict_top + "    _flush_pending_batches()\n",
        1,
    )

    return src


def _verify_transformed(src: str) -> None:
    """Sanity checks on the transformed model.py text."""
    must_contain = [
        "def score_batch(self, rows: list) -> list:",
        "def _enqueue_for_batch(",
        "def _flush_pending_batches() -> None:",
        "def _embed_batch(texts: list) -> list:",
        "_FLUSHED: bool = True",
        "_ITEM_PENDING: list[tuple[str, str, str]] = []",
        "ENCODER_RUNTIME_BATCH_SIZE: int = int(META.get",
        "JUDGE_RUNTIME_BATCH_SIZE: int = int(JUDGE_META.get",
    ]
    for needle in must_contain:
        if needle not in src:
            raise RuntimeError(f"verification failed: {needle!r} not in transformed model.py")

    forbidden = [
        # We deliberately exclude these "complications" that previous
        # batched bundles carried.  If a future refactor reintroduces
        # them here, this check fires and forces a deliberate decision.
        "_write_progress(",
        "_free_encoder_vram(",
        "FREE_ENCODER_AFTER_FLUSH",
        'meta.get("lora"',
    ]
    for needle in forbidden:
        if needle in src:
            raise RuntimeError(
                f"verification failed: forbidden anti-pattern {needle!r} present in transformed model.py"
            )

    # Sanity on syntax: compile-check the transformed source.
    try:
        compile(src, "<transformed model.py>", "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"transformed model.py has a SyntaxError: {exc}") from exc


def _patch_meta_batched(raw: bytes) -> bytes:
    """Mark the runtime as batched + give it the small L4 batch sizes."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "batched_nonn"
    meta["encoder_runtime_batch_size"] = 4
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["runtime_batch_size"] = 8
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = False
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    with zipfile.ZipFile(SRC, "r") as zin:
        original_model_py = zin.read("model.py").decode("utf-8")
        names = list(zin.namelist())

    transformed_model_py = transform_model_py(original_model_py)
    _verify_transformed(transformed_model_py)

    # Confirm the transform genuinely added new lines and didn't remove
    # anything we cared about by accident.
    n_added = len(transformed_model_py.splitlines()) - len(original_model_py.splitlines())
    print(f"  model.py grew by {n_added} lines (expected ~150-180)")

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if info.filename == "model.py":
                zout.writestr(info.filename, transformed_model_py.encode("utf-8"))
            elif info.filename == "labeling.py":
                zout.writestr(info.filename, ENQUEUE_LABELING_PY.encode("utf-8"))
            elif info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, _patch_meta_batched(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")

    with zipfile.ZipFile(DST, "r") as zf:
        zf_names = set(zf.namelist())
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
        labeling = zf.read("labeling.py").decode("utf-8")

    assert not any(n.startswith("cache/") for n in zf_names), "cache/ leaked into bundle"
    assert "requirements.txt" not in zf_names, "requirements.txt must be absent"
    assert "model.py" in zf_names
    assert "labeling.py" in zf_names
    assert "artifacts/checkpoint.pt" in zf_names
    assert meta["runtime_architecture"] == "batched_nonn"
    assert meta["encoder_runtime_batch_size"] == 4
    assert meta["judge"]["runtime_batch_size"] == 8
    assert meta["judge"]["enabled"] is True
    assert meta["nn_features"]["enabled"] is False
    assert "_enqueue_for_batch" in labeling
    assert "return 0.0" in labeling
    assert size_mb < 70.0, f"bundle is {size_mb:.2f} MB, over the 70 MB ceiling"
    print("  [OK] no cache/* paths")
    print("  [OK] no requirements.txt")
    print("  [OK] batched runtime_meta (encoder_bs=4, judge_bs=8, no eviction)")
    print("  [OK] judge.enabled = True / nn_features.enabled = False")
    print("  [OK] labeling.py is enqueue-only")
    print(f"  [OK] {size_mb:.2f} MB < 70 MB ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
