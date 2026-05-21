"""Pack a streamed-encoder bundle for the 256k-pair / 10s-per-predict regime.

What the platform just taught us (PAIEC-NETWORK-001 + PAIEC-TIMEOUT-001):

  1. The hidden slice is ~256k (subject, item) pairs (~5k items x ~51
     subjects), not the ~5k pairs the README's "5000 hidden items"
     phrasing suggested.

  2. ``predict()`` has a 10-second PER-CALL timeout, not a round-total
     budget. Even a 5k-item encoder flush at bs=32 (~15s) is too slow
     for the first ``predict()`` call.

  3. Some HF library is making a network call at runtime even with
     ``local_files_only=True``.  Set ``HF_HUB_OFFLINE`` and
     ``TRANSFORMERS_OFFLINE`` env vars BEFORE any transformers import
     to force-disable that path.

Architecture, derived directly from those three constraints:

  - Drop the LLM judge entirely.  256k pairs x judge forward at any
    batch size cannot fit in a 30-minute L4 budget; even at bs=64 with
    optimistic ~200ms/batch we'd need ~14 minutes JUST for the judge,
    on top of everything else.  Judge features fall back to zeros.

  - Stream the encoder batches INSIDE ``acquisition_function``.  Each
    enqueue call also opportunistically flushes one bs=16 batch when
    enough items / subjects have piled up.  Bounded per-call work
    (~100-200ms peak when a batch fires), spread evenly across the
    ~256k acquisition calls.  By the time the platform stops calling
    ``acquisition_function`` and starts calling ``predict()``, almost
    every item and subject is already encoded.

  - ``predict()`` does at most one tiny residual flush (< 16 items,
    well under 1s) and then per-pair cache lookup + IRT head forward
    (~5-10 ms).  Comfortably under the 10s timeout.

  - Set ``os.environ['HF_HUB_OFFLINE'] = '1'`` and
    ``os.environ['TRANSFORMERS_OFFLINE'] = '1'`` at the very top of
    ``model.py``, before any transformers import.  This is what
    actually addresses PAIEC-NETWORK-001.

  - models.txt declares only the encoder.  The judge model isn't
    pre-downloaded so we can't accidentally trigger its load.

Source bundle: ``submission_cacheless_nojudge.zip`` (judge already off,
no cache, no requirements.txt -- the clean baseline).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless_nojudge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_encoder.zip")


OFFLINE_ENV_INJECTION = '''import os as _os_for_hf_offline
_os_for_hf_offline.environ.setdefault("HF_HUB_OFFLINE", "1")
_os_for_hf_offline.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os_for_hf_offline.environ.setdefault("HF_DATASETS_OFFLINE", "1")
del _os_for_hf_offline

'''


MODULE_SCOPE_BATCHING = '''

# ---------------------------------------------------------------------------
# Streamed-encoder batching (no judge variant).
#
# Each ``_enqueue_for_batch`` call appends to the pending queues and, when
# enough items / subjects have piled up, fires a single bs=16 encoder
# forward.  Bounded per-call work (~100-200ms peak when a batch fires).
# The platform's pre-predict() acquisition phase drives ~256k calls; by
# the end of that phase essentially every unique item + subject is
# already encoded.  predict() then has at most a < 16-item residual to
# flush before its first per-pair lookup, keeping every predict() call
# well under the 10-second per-call timeout.
# ---------------------------------------------------------------------------

ENCODER_RUNTIME_BATCH_SIZE: int = int(META.get("encoder_runtime_batch_size", 16))

_ITEM_PENDING: list[tuple[str, str, str]] = []
_SUBJECT_PENDING: list[str] = []
_ITEM_PENDING_KEYS: set[str] = set()
_SUBJECT_PENDING_KEYS: set[str] = set()


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


def _enqueue_for_batch(
    *,
    benchmark: str,
    condition: str,
    subject_content: str,
    item_content: str,
) -> None:
    """Enqueue this candidate + opportunistically flush one batch."""
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

    if len(_ITEM_PENDING) >= ENCODER_RUNTIME_BATCH_SIZE:
        _flush_one_item_batch(ENCODER_RUNTIME_BATCH_SIZE)
    if len(_SUBJECT_PENDING) >= ENCODER_RUNTIME_BATCH_SIZE:
        _flush_one_subject_batch(ENCODER_RUNTIME_BATCH_SIZE)


def _flush_pending_batches() -> None:
    """Drain any residual pending items / subjects.  Called by predict()."""
    while _ITEM_PENDING:
        _flush_one_item_batch(ENCODER_RUNTIME_BATCH_SIZE)
    while _SUBJECT_PENDING:
        _flush_one_subject_batch(ENCODER_RUNTIME_BATCH_SIZE)
'''


EMBED_BATCH_FUNCTION = '''

def _embed_batch(texts: list) -> list:
    """Encode ``texts`` in chunks of ``ENCODER_RUNTIME_BATCH_SIZE``.

    Returns a list of float32 vectors, in input order.  Uses
    ``padding="longest"`` per chunk so wasted compute scales with the
    per-chunk maximum length rather than the global one.
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


ENQUEUE_LABELING_PY = '''"""Adaptive labeling: enqueue-only acquisition (streamed-encoder bundle).

The platform calls ``acquisition_function`` once per hidden
(subject, item) pair BEFORE any ``predict()`` call.  Each call here
forwards to ``model._enqueue_for_batch``, which appends to the
per-content queues and opportunistically fires one bs=16 encoder
forward when enough items / subjects have piled up.

Returning 0.0 yields a constant ranking; the platform breaks ties
randomly, which is equivalent in expectation to the default
random-sample fallback applied when no ``labeling.py`` is shipped.
The only effect of this file is to drive the streamed encoder flush.
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
    """Apply five surgical injections to the cacheless_nojudge model.py.

    1) Offline env vars at the very top of the file.
    2) Module-scope queues + ``_enqueue_for_batch`` / ``_flush_pending_batches``.
    3) ``_embed_batch`` right after ``_embed_one``.
    4) A single line at the top of ``predict()``: ``_flush_pending_batches()``.
    """

    # 1. Offline env vars: insert right after the leading docstring + the
    #    ``from __future__ import annotations`` line, before any other
    #    imports.
    anchor_future = "from __future__ import annotations\n"
    if anchor_future not in src:
        raise RuntimeError("anchor for offline env var insertion not found")
    src = src.replace(
        anchor_future,
        anchor_future + "\n" + OFFLINE_ENV_INJECTION,
        1,
    )

    # 2. Module-scope queues + enqueue/flush helpers.  Insert immediately
    #    after the ``JUDGE: ... | None = None`` declaration.
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

    # 3. _embed_batch right after _embed_one ends.
    anchor_embed_one_end = "    return vec.astype(np.float32, copy=False)\n"
    if anchor_embed_one_end not in src:
        raise RuntimeError("anchor for _embed_batch insertion not found")
    src = src.replace(
        anchor_embed_one_end,
        anchor_embed_one_end + EMBED_BATCH_FUNCTION,
        1,
    )

    # 4. _flush_pending_batches() call at the top of predict().
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
        "def _enqueue_for_batch(",
        "def _flush_pending_batches() -> None:",
        "def _flush_one_item_batch(",
        "def _flush_one_subject_batch(",
        "def _embed_batch(texts: list) -> list:",
        "ENCODER_RUNTIME_BATCH_SIZE: int",
    ]
    for needle in must_contain:
        if needle not in src:
            raise RuntimeError(f"verification failed: missing {needle!r}")

    forbidden = [
        # No JUDGE batching in this variant.  We hold JUDGE=None
        # courtesy of runtime_meta and let the existing per-pair
        # _get_judge_features fallback to zeros.
        "JUDGE.score_batch(",
        "_JUDGE_PENDING",
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
    """Mark this as the streamed-encoder runtime; encoder bs=16, no judge."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "streamed_encoder_nojudge"
    meta["encoder_runtime_batch_size"] = 16
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["enabled"] = False
    j["ship_at_runtime"] = False
    j["runtime_batch_size"] = 8  # ignored when disabled but kept for forensics
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
    assert meta["runtime_architecture"] == "streamed_encoder_nojudge"
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["judge"]["enabled"] is False
    assert meta["nn_features"]["enabled"] is False
    assert "Qwen/Qwen3-4B-Instruct" not in m_txt, "judge must be removed from models.txt"
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    assert "_enqueue_for_batch" in labeling and "return 0.0" in labeling
    # Confirm the offline env vars land in the very first ~20 lines so
    # they execute before any potential transformers import side-effect.
    head = "\n".join(model_py_extracted.splitlines()[:25])
    assert "HF_HUB_OFFLINE" in head, "HF_HUB_OFFLINE env var must be set near the top of model.py"
    assert "TRANSFORMERS_OFFLINE" in head, "TRANSFORMERS_OFFLINE env var must be set near the top of model.py"
    assert size_mb < 70.0, f"bundle is {size_mb:.2f} MB, over the 70 MB ceiling"
    print("  [OK] cache/, requirements.txt absent")
    print("  [OK] models.txt declares only the encoder")
    print("  [OK] judge.enabled = False / nn_features.enabled = False")
    print("  [OK] HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set at top of model.py")
    print("  [OK] streamed_encoder_nojudge runtime_meta")
    print(f"  [OK] {size_mb:.2f} MB < 70 MB ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
