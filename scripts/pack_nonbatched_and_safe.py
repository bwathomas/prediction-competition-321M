"""Pack a non-batched and a batched-safe bundle for the next platform test.

Every batched bundle so far (variants A, preregression, fix1) has failed
with PAIEC-UNKNOWN-001 ("no details safe to show") within 1-5 minutes.
That code is the platform's catch-all for OS-level kills (OOM/SIGKILL)
when there's no Python exception to surface. 1-5 minutes is consistent
with: module init (~1-2 min loading 16 GB of model weights) followed
by an immediate kill on the first flush batch.

Common factor across the failing bundles: ``free_encoder_after_flush =
False``. That keeps BOTH 4B-param bf16 models (encoder ~ 8 GB + judge
~ 8 GB = 16 GB of weights) resident in VRAM while the encoder phase
runs its first forward at bs=16. Encoder activations at seq=512, bs=16
on a 4B model are ~ 6-8 GB. Total: 22-24 GB on a 24 GB L4, plus
~ 1-2 GB of CUDA driver / framework overhead -> OOM on the first batch.

This script produces two bundles to isolate the issue:

  1. submission_nonbatched.zip
        - same model.py + artifacts as the failing bundles
        - labeling.py replaced by a no-op that returns 0.0 without
          calling _enqueue_for_batch
        - _FLUSHED stays True (its initial value), so _flush_pending_batches
          is a no-op when the first predict() lands
        - predict() falls through to the per-pair (bs=1) path, which
          never co-resident-fwds two models at scale
        - if THIS passes, the bundle artifacts + everything except
          batching work; if it fails, the failure isn't about batching
          at all

  2. submission_batched_safe.zip
        - same model.py + artifacts + current batched labeling.py
        - meta: encoder_runtime_batch_size=8, judge.runtime_batch_size=8,
                free_encoder_after_flush=true
        - bs=8 encoder while judge is co-resident: 8 + 8 + ~3 = ~ 19 GB.
          Safe.
        - judge after eviction: 8 + ~3 = ~ 11 GB. Safe with lots of
          headroom.
        - if THIS passes, batching works with eviction + small bs;
          we then incrementally crank settings up
        - if it FAILS too, batching has a deeper issue independent of
          memory
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export_submission import _RUNTIME_MODEL_PY, _RUNTIME_LABELING_PY  # noqa: E402

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip")
DST_DIR = Path(r"C:\Users\benja\Downloads\submission")

NONBATCHED_DST = DST_DIR / "submission_nonbatched.zip"
SAFE_DST = DST_DIR / "submission_batched_safe.zip"

NOOP_LABELING_PY = '''"""No-op acquisition function: returns 0.0 for every candidate.

This file is the *non-batched* control bundle. The platform still calls
acquisition_function once per candidate, but we deliberately do nothing
-- no enqueueing, no module-state mutation, no work at all. The
module-level _FLUSHED in model.py stays at its initial True value, so
the first predict()'s ``if not _FLUSHED`` check is False and the flush
is skipped. predict() then runs the per-pair (bs=1) inference path,
which is what every non-batched submission has always done.
"""

from __future__ import annotations


def acquisition_function(input: dict) -> float:  # noqa: A002
    return 0.0
'''


def patch_meta_nonbatched(raw: bytes) -> bytes:
    """For the non-batched bundle, meta knobs don't matter (no flush)
    but we still set them to safe values for consistency."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "nonbatched"
    meta["encoder_runtime_batch_size"] = 8
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["runtime_batch_size"] = 8
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def patch_meta_safe(raw: bytes) -> bytes:
    """Safest practical batched config: bs=8 both, eviction on."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "batched_flush_v1"
    meta["encoder_runtime_batch_size"] = 8
    meta["free_encoder_after_flush"] = True
    j = meta.setdefault("judge", {})
    j["runtime_batch_size"] = 8
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def pack(dst: Path, model_py: bytes, labeling_py: bytes, meta_patch) -> None:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename == "model.py":
                zout.writestr(info.filename, model_py)
            elif info.filename == "labeling.py":
                zout.writestr(info.filename, labeling_py)
            elif info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, meta_patch(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1

    model_bytes = _RUNTIME_MODEL_PY.encode("utf-8")
    batched_labeling_bytes = _RUNTIME_LABELING_PY.encode("utf-8")
    noop_labeling_bytes = NOOP_LABELING_PY.encode("utf-8")

    pack(NONBATCHED_DST, model_bytes, noop_labeling_bytes, patch_meta_nonbatched)
    nb_mb = NONBATCHED_DST.stat().st_size / (1024 * 1024)
    print(f"  [OK] {NONBATCHED_DST.name:40s}  {nb_mb:.2f} MB  (non-batched control)")

    pack(SAFE_DST, model_bytes, batched_labeling_bytes, patch_meta_safe)
    sf_mb = SAFE_DST.stat().st_size / (1024 * 1024)
    print(f"  [OK] {SAFE_DST.name:40s}  {sf_mb:.2f} MB  (batched: bs=8 + eviction)")

    # Verification.
    with zipfile.ZipFile(NONBATCHED_DST, "r") as zf:
        L = zf.read("labeling.py").decode("utf-8")
        m = json.loads(zf.read("artifacts/runtime_meta.json"))
        # The no-op labeling.py must NOT touch _enqueue_for_batch.
        assert "_enqueue_for_batch" not in L, (
            "non-batched labeling.py still imports _enqueue_for_batch"
        )
        assert "return 0.0" in L
        assert m["runtime_architecture"] == "nonbatched"
        print(f"       nonbatched: labeling.py is a true no-op, "
              f"runtime_architecture={m['runtime_architecture']!r}")

    with zipfile.ZipFile(SAFE_DST, "r") as zf:
        L = zf.read("labeling.py").decode("utf-8")
        m = json.loads(zf.read("artifacts/runtime_meta.json"))
        # The batched labeling.py must call _enqueue_for_batch.
        assert "_enqueue_for_batch" in L
        assert m["encoder_runtime_batch_size"] == 8
        assert m["judge"]["runtime_batch_size"] == 8
        assert m["free_encoder_after_flush"] is True
        print(f"       batched_safe: enc_bs=8, jdg_bs=8, evict=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
