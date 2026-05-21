"""Pack a streamed-batching bundle WITH the judge.

Same architecture as ``pack_streamed_encoder.py`` (work happens inside
``acquisition_function`` so ``predict()`` never has to flush a large
queue under the 10-second per-call timeout) but the judge is kept
enabled.  Each acquisition call does at most one small batch:

  - priority 1: if pending judge >= JUDGE_RUNTIME_BATCH_SIZE (default 8),
    fire one judge bs=8 forward (~200 ms on L4).
  - priority 2: if pending items >= ENCODER_RUNTIME_BATCH_SIZE (default 16),
    fire one encoder bs=16 forward (~100 ms on L4).
  - priority 3: if pending subjects >= ENCODER_RUNTIME_BATCH_SIZE, same
    for the subject encoder pass.

Both the encoder (~8 GB bf16) and the judge (~8 GB bf16) stay co-resident
on a 24 GB L4 -- the small batch sizes are sized for that footprint.

Expected wall-clock work to drain all queues (assuming ~256k pairs,
~5k unique items, ~50 unique subjects):

  - judge: 256k / 8 = 32k batches x ~200 ms = ~107 minutes
  - encoder items: 5k / 16 = 313 batches x ~100 ms = ~31 seconds
  - encoder subjects: 50 / 16 = 4 batches x ~100 ms = ~0.4 seconds

Total ~110 minutes, spread evenly across 256k acquisition calls
(~25 ms / call amortized, ~200 ms peak when a judge batch fires).

Per-call peak is well under 10 s, so neither acquisition_function nor
predict() should ever trip a per-call timeout.  predict()'s first
call drains residuals < 1 batch per queue (< 1.5 s total), then each
predict() call is a pure cache lookup + IRT head forward (~5-10 ms).

Source bundle: ``submission_cacheless.zip`` (NN off, judge ON, no
cache, no requirements.txt).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_judge.zip")


OFFLINE_ENV_INJECTION = '''import os as _os_for_hf_offline
_os_for_hf_offline.environ.setdefault("HF_HUB_OFFLINE", "1")
_os_for_hf_offline.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os_for_hf_offline.environ.setdefault("HF_DATASETS_OFFLINE", "1")
del _os_for_hf_offline

'''


SCORE_BATCH_METHOD = '''\
    def score_batch(self, rows: list) -> list:
        """Score ``(benchmark, condition, subject, item)`` rows in chunks
        of ``JUDGE_RUNTIME_BATCH_SIZE``.  Returns one length-4 float32
        vector per row, matching the per-row ``score`` output."""
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

# ---------------------------------------------------------------------------
# Streamed batching with judge.
#
# Every ``_enqueue_for_batch`` call appends to the pending queues and, when
# the deepest queue (judge) has enough pairs to fire a small batch, drains
# one batch inline.  Bounded per-call work (~100-200 ms peak when a batch
# fires; ~30 us otherwise).  Total queued judge work for ~256k pairs at
# bs=8 is ~107 minutes spread evenly across acquisition calls -- well
# inside any reasonable run-time budget, and never blocks a single call
# for more than ~200 ms.
# ---------------------------------------------------------------------------

ENCODER_RUNTIME_BATCH_SIZE: int = int(META.get("encoder_runtime_batch_size", 16))
JUDGE_RUNTIME_BATCH_SIZE: int = int(JUDGE_META.get("runtime_batch_size", 8))

_ITEM_PENDING: list[tuple[str, str, str]] = []
_SUBJECT_PENDING: list[str] = []
_JUDGE_PENDING: list[tuple[str, str, str, str]] = []
_ITEM_PENDING_KEYS: set[str] = set()
_SUBJECT_PENDING_KEYS: set[str] = set()
_JUDGE_PENDING_KEYS: set[tuple[str, str]] = set()


def _flush_one_item_batch(n: int) -> None:
    if not _ITEM_PENDING:
        return
    n = max(1, min(n, len(_ITEM_PENDING)))
    chunk = _ITEM_PENDING[:n]
    del _ITEM_PENDING[:n]
    texts = [item_text_for(b, c, i) for (b, c, i) in chunk]
    vecs = _embed_batch(texts)
    for (b, c, i), vec in zip(chunk, vecs):
        key = stable_sha256(b, c, i)
        _ITEM_EMB_CACHE[key] = vec
        _ITEM_PENDING_KEYS.discard(key)


def _flush_one_subject_batch(n: int) -> None:
    if not _SUBJECT_PENDING:
        return
    n = max(1, min(n, len(_SUBJECT_PENDING)))
    chunk = _SUBJECT_PENDING[:n]
    del _SUBJECT_PENDING[:n]
    texts = [subject_text_for(s) for s in chunk]
    vecs = _embed_batch(texts)
    for s, vec in zip(chunk, vecs):
        key = stable_sha256(s)
        _SUBJECT_EMB_CACHE[key] = vec
        _SUBJECT_PENDING_KEYS.discard(key)


def _flush_one_judge_batch(n: int) -> None:
    if not _JUDGE_PENDING or JUDGE is None:
        return
    n = max(1, min(n, len(_JUDGE_PENDING)))
    chunk = _JUDGE_PENDING[:n]
    del _JUDGE_PENDING[:n]
    vecs = JUDGE.score_batch(chunk)
    for (b, c, s, i), vec in zip(chunk, vecs):
        j_key = (stable_sha256(s), stable_sha256(b, c, i))
        JUDGE.score_cache[j_key] = vec
        _JUDGE_PENDING_KEYS.discard(j_key)


def _enqueue_for_batch(
    *,
    benchmark: str,
    condition: str,
    subject_content: str,
    item_content: str,
) -> None:
    """Enqueue + opportunistically fire at most one small batch."""
    benchmark = str(benchmark or "")
    condition = normalize_condition(condition)
    subject_content = str(subject_content or "")
    item_content = str(item_content or "")

    item_key = stable_sha256(benchmark, condition, item_content)
    if item_key not in _ITEM_PENDING_KEYS and item_key not in _ITEM_EMB_CACHE:
        _ITEM_PENDING_KEYS.add(item_key)
        _ITEM_PENDING.append((benchmark, condition, item_content))

    if _MODEL_CFG.get("use_subject_text_embedding"):
        s_key = stable_sha256(subject_content)
        if s_key not in _SUBJECT_PENDING_KEYS and s_key not in _SUBJECT_EMB_CACHE:
            _SUBJECT_PENDING_KEYS.add(s_key)
            _SUBJECT_PENDING.append(subject_content)

    if JUDGE is not None and _MODEL_CFG.get("use_judge_features"):
        j_key = (stable_sha256(subject_content), item_key)
        if j_key not in _JUDGE_PENDING_KEYS and j_key not in JUDGE.score_cache:
            _JUDGE_PENDING_KEYS.add(j_key)
            _JUDGE_PENDING.append(
                (benchmark, condition, subject_content, item_content)
            )

    # One opportunistic flush per call.  Judge is highest priority because
    # it is the slowest queue per pair and would otherwise grow without bound.
    if JUDGE is not None and len(_JUDGE_PENDING) >= JUDGE_RUNTIME_BATCH_SIZE:
        _flush_one_judge_batch(JUDGE_RUNTIME_BATCH_SIZE)
        return
    if len(_ITEM_PENDING) >= ENCODER_RUNTIME_BATCH_SIZE:
        _flush_one_item_batch(ENCODER_RUNTIME_BATCH_SIZE)
        return
    if len(_SUBJECT_PENDING) >= ENCODER_RUNTIME_BATCH_SIZE:
        _flush_one_subject_batch(ENCODER_RUNTIME_BATCH_SIZE)
        return


def _flush_pending_batches() -> None:
    """Drain residual pending items / subjects / judge pairs.

    Called by ``predict()`` on first invocation.  Each queue's residual
    is < its batch size (because every time the queue hits the batch
    size, ``_enqueue_for_batch`` drained one batch), so each loop fires
    at most one final small batch.  Worst-case wall time: <1.5 seconds.
    """
    while _ITEM_PENDING:
        _flush_one_item_batch(ENCODER_RUNTIME_BATCH_SIZE)
    while _SUBJECT_PENDING:
        _flush_one_subject_batch(ENCODER_RUNTIME_BATCH_SIZE)
    while _JUDGE_PENDING:
        _flush_one_judge_batch(JUDGE_RUNTIME_BATCH_SIZE)
'''


EMBED_BATCH_FUNCTION = '''

def _embed_batch(texts: list) -> list:
    """Encode ``texts`` in chunks of ``ENCODER_RUNTIME_BATCH_SIZE``.

    Returns a list of float32 vectors, in input order.  Uses
    ``padding="longest"`` per chunk so wasted compute scales with the
    per-chunk max length rather than the global one.
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


ENQUEUE_LABELING_PY = '''"""Adaptive labeling: enqueue-only acquisition (streamed-judge bundle).

The platform calls ``acquisition_function`` once per hidden
(subject, item) pair BEFORE any ``predict()`` call.  Each call here
forwards to ``model._enqueue_for_batch``, which appends to the
encoder + judge pending queues and opportunistically fires one small
batch (judge bs=8 if available, else encoder bs=16) before returning.

Returning 0.0 yields a constant ranking; the platform breaks ties
randomly, which is equivalent in expectation to the default
random-sample fallback applied when no ``labeling.py`` is shipped.
The only effect of this file is to drive the streamed flush.
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
    """Apply five surgical injections to the cacheless model.py."""

    # 1. Offline env vars at the very top, right after the
    #    ``from __future__ import annotations`` line.
    anchor_future = "from __future__ import annotations\n"
    if anchor_future not in src:
        raise RuntimeError("anchor for offline env var insertion not found")
    src = src.replace(
        anchor_future,
        anchor_future + "\n" + OFFLINE_ENV_INJECTION,
        1,
    )

    # 2. ``score_batch`` into ``_LLMJudgeRuntime`` (before ``score_cached``).
    anchor_score_cached = (
        "    def score_cached(self, *, benchmark: str, condition: str, "
        "subject_content: str, item_content: str) -> np.ndarray:"
    )
    if anchor_score_cached not in src:
        raise RuntimeError("anchor for score_batch insertion not found")
    src = src.replace(
        anchor_score_cached,
        SCORE_BATCH_METHOD + anchor_score_cached,
        1,
    )

    # 3. Module-scope queues + helpers, after the ``JUDGE`` declaration.
    anchor_judge_decl = (
        "JUDGE: _LLMJudgeRuntime | None = None"
        "  # populated in the module-init section below"
    )
    if anchor_judge_decl not in src:
        raise RuntimeError("anchor for module-scope batching not found")
    src = src.replace(
        anchor_judge_decl,
        anchor_judge_decl + MODULE_SCOPE_BATCHING,
        1,
    )

    # 4. ``_embed_batch`` after ``_embed_one`` (first occurrence of its
    #    closing line in the file).
    anchor_embed_one_end = "    return vec.astype(np.float32, copy=False)\n"
    if anchor_embed_one_end not in src:
        raise RuntimeError("anchor for _embed_batch insertion not found")
    src = src.replace(
        anchor_embed_one_end,
        anchor_embed_one_end + EMBED_BATCH_FUNCTION,
        1,
    )

    # 5. ``_flush_pending_batches()`` call at the top of ``predict``.
    anchor_predict_top = (
        'def predict(input: dict, labeled: list[dict] | None = None) -> float:\n'
        '    """Return a native Python float in (0, 1)."""\n'
    )
    if anchor_predict_top not in src:
        raise RuntimeError("anchor for predict() flush call not found")
    src = src.replace(
        anchor_predict_top,
        anchor_predict_top + "    _flush_pending_batches()\n",
        1,
    )

    return src


def _verify_transformed(src: str) -> None:
    must_contain = [
        '_os_for_hf_offline.environ.setdefault("HF_HUB_OFFLINE"',
        '_os_for_hf_offline.environ.setdefault("TRANSFORMERS_OFFLINE"',
        "def score_batch(self, rows: list) -> list:",
        "def _enqueue_for_batch(",
        "def _flush_pending_batches() -> None:",
        "def _flush_one_item_batch(",
        "def _flush_one_subject_batch(",
        "def _flush_one_judge_batch(",
        "def _embed_batch(texts: list) -> list:",
        "ENCODER_RUNTIME_BATCH_SIZE: int",
        "JUDGE_RUNTIME_BATCH_SIZE: int",
        "_JUDGE_PENDING:",
    ]
    for needle in must_contain:
        if needle not in src:
            raise RuntimeError(f"verification failed: missing {needle!r}")

    forbidden = [
        "_write_progress(",
        "_free_encoder_vram(",
        "FREE_ENCODER_AFTER_FLUSH",
    ]
    for needle in forbidden:
        if needle in src:
            raise RuntimeError(
                f"verification failed: forbidden anti-pattern {needle!r} present"
            )

    try:
        compile(src, "<transformed model.py>", "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"transformed model.py has a SyntaxError: {exc}") from exc


def _patch_meta(raw: bytes) -> bytes:
    """Mark this as the streamed-batching runtime with judge ON."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "streamed_batched_judge"
    meta["encoder_runtime_batch_size"] = 16
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["enabled"] = True
    j["ship_at_runtime"] = True
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

    transformed_model_py = transform_model_py(original_model_py)
    _verify_transformed(transformed_model_py)
    n_added = len(transformed_model_py.splitlines()) - len(original_model_py.splitlines())
    print(f"  model.py grew by {n_added} lines (expected ~220-280)")

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if info.filename == "model.py":
                zout.writestr(info.filename, transformed_model_py.encode("utf-8"))
            elif info.filename == "labeling.py":
                zout.writestr(info.filename, ENQUEUE_LABELING_PY.encode("utf-8"))
            elif info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, _patch_meta(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")

    with zipfile.ZipFile(DST, "r") as zf:
        zf_names = set(zf.namelist())
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
        labeling = zf.read("labeling.py").decode("utf-8")
        model_py_extracted = zf.read("model.py").decode("utf-8")
        m_txt = zf.read("models.txt").decode("utf-8")

    assert not any(n.startswith("cache/") for n in zf_names), "cache/ leaked into bundle"
    assert "requirements.txt" not in zf_names, "requirements.txt must be absent"
    assert "model.py" in zf_names
    assert "labeling.py" in zf_names
    assert "artifacts/checkpoint.pt" in zf_names
    assert meta["runtime_architecture"] == "streamed_batched_judge"
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["judge"]["enabled"] is True
    assert meta["judge"]["runtime_batch_size"] == 8
    assert meta["nn_features"]["enabled"] is False
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    assert "Qwen/Qwen3-4B-Instruct" in m_txt, "judge MUST stay in models.txt"
    assert "_enqueue_for_batch" in labeling and "return 0.0" in labeling
    head = "\n".join(model_py_extracted.splitlines()[:25])
    assert "HF_HUB_OFFLINE" in head, "HF_HUB_OFFLINE must be set near top of model.py"
    assert "TRANSFORMERS_OFFLINE" in head, "TRANSFORMERS_OFFLINE must be set near top of model.py"
    assert size_mb < 70.0, f"bundle is {size_mb:.2f} MB, over the 70 MB ceiling"
    print("  [OK] cache/, requirements.txt absent")
    print("  [OK] models.txt declares encoder + judge")
    print("  [OK] judge.enabled = True / nn_features.enabled = False")
    print("  [OK] HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE at top of model.py")
    print("  [OK] streamed_batched_judge runtime_meta (encoder_bs=16, judge_bs=8)")
    print(f"  [OK] {size_mb:.2f} MB < 70 MB ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
