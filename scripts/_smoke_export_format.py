"""Smoke-check that export_run produces the streamed-flush format end to end."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import export_submission as ES  # noqa: E402

# 1. Templates parse.
ast.parse(ES._RUNTIME_MODEL_PY)
ast.parse(ES._RUNTIME_LABELING_PY)
print("[OK] templates parse")

# 2. Runtime template reads the new fields.
for needle in (
    "JUDGE_RUNTIME_BATCH_SIZE",
    'JUDGE_META.get("runtime_batch_size"',
    'META.get("encoder_runtime_batch_size"',
    "ENCODER_RUNTIME_BATCH_SIZE",
    "def _enqueue_for_batch",
    "def _flush_pending_batches",
    "def _flush_one_item_batch",
    "def _flush_one_subject_batch",
    "def _flush_one_judge_batch",
    "def score_batch",
    "def _embed_batch",
    # Offline mode must be forced before the transformers import; without
    # it, PAIEC-NETWORK-001 kills the submission even with a fully cached
    # ``from_pretrained(..., local_files_only=True)`` call.
    'HF_HUB_OFFLINE',
    'TRANSFORMERS_OFFLINE',
    'HF_DATASETS_OFFLINE',
):
    assert needle in ES._RUNTIME_MODEL_PY, f"missing in model.py template: {needle!r}"
    print(f"[OK] runtime model.py reads/declares: {needle}")

# 2b. The obsolete batched-flush plumbing must be gone -- it tripped the
#     10 s per-call predict() timeout on the documented ~256 k workload.
for stale in (
    "_write_progress",
    "_free_encoder_vram",
    "FREE_ENCODER_AFTER_FLUSH",
    "RUNTIME_PROGRESS_PATH",
    "_PROGRESS_T0",
):
    assert stale not in ES._RUNTIME_MODEL_PY, (
        f"stale obsolete-arch symbol still in model.py template: {stale!r}"
    )
print("[OK] obsolete batched-flush plumbing fully removed from model.py")

# 3. labeling.py is the enqueue-only variant.
for needle in ("_enqueue_for_batch", "return 0.0"):
    assert needle in ES._RUNTIME_LABELING_PY, f"missing in labeling.py: {needle!r}"
print("[OK] runtime labeling.py is enqueue-only")

# 4. export_run writes the right fields and the broken repair is gone.
src = inspect.getsource(ES.export_run)
for needle in (
    '"encoder_runtime_batch_size":',
    '"runtime_batch_size": runtime_judge_bs',
    '"runtime_architecture": "streamed_flush_v1"',
):
    assert needle in src, f"missing in export_run: {needle!r}"
    print(f"[OK] export_run writes: {needle}")
assert "coercing cluster_embed_dim=16" not in src, "broken repair STILL in export_run"
print("[OK] broken cluster repair removed from export_run")
# The obsolete batched-flush tag must not leak through any other code path.
assert '"runtime_architecture": "batched_flush_v1"' not in src, (
    "export_run still tags bundles as batched_flush_v1"
)
print("[OK] export_run no longer emits the obsolete batched_flush_v1 tag")

# 5. include_labeling=False warning is wired up.
assert "if not include_labeling:" in src and "LOG.warning(" in src, (
    "include_labeling=False warning not wired"
)
print("[OK] include_labeling=False emits a runtime warning")

# 6. configs/default.yaml has the correct streamed-flush runtime batch
#    sizes for both encoder (bs=16) and judge (bs=8).
cfg_yaml = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
assert cfg_yaml.count("runtime_batch_size: 16") >= 1, (
    "default.yaml is missing 'runtime_batch_size: 16' (encoder)"
)
print("[OK] default.yaml has encoder.runtime_batch_size: 16")
assert cfg_yaml.count("runtime_batch_size: 8") >= 1, (
    "default.yaml is missing 'runtime_batch_size: 8' (judge -- streamed-flush "
    "co-resident default; raise only after verifying no OOM on target tier)"
)
print("[OK] default.yaml has judge.runtime_batch_size: 8")

# 7. Quick check that the runtime template handles missing fields gracefully.
#    Defaults are sized for an L4 with the encoder + judge co-resident in
#    the streamed-flush architecture: encoder bs=16, judge bs=8.
import re

m = re.search(
    r'JUDGE_RUNTIME_BATCH_SIZE: int = int\(JUDGE_META\.get\("runtime_batch_size", (\d+)\)\)',
    ES._RUNTIME_MODEL_PY,
)
assert m and int(m.group(1)) == 8, "judge runtime batch default not 8"
print(f"[OK] JUDGE_RUNTIME_BATCH_SIZE defaults to {m.group(1)}")

m = re.search(
    r'ENCODER_RUNTIME_BATCH_SIZE: int = int\(\s*META\.get\("encoder_runtime_batch_size", (\d+)\)\s*\)',
    ES._RUNTIME_MODEL_PY,
)
assert m and int(m.group(1)) == 16, "encoder runtime batch default not 16"
print(f"[OK] ENCODER_RUNTIME_BATCH_SIZE defaults to {m.group(1)}")

# 8. Critical invariant: both ``_enqueue_for_batch`` and
#    ``_predict_uncalibrated`` must apply ``normalize_condition`` *before*
#    computing any cache key. Otherwise the flush populates raw-condition
#    keys while predict() looks up normalized-condition keys, every
#    lookup misses, and the runtime silently falls back to the bs=1
#    forward path -- the speedup does not materialize. This is the bug
#    that caused submission_judge_batched.zip to run as slowly as the
#    unbatched version on Codabench.
for fn_name in ("_enqueue_for_batch", "_predict_uncalibrated"):
    body_match = re.search(
        rf"def {re.escape(fn_name)}\([^)]*\)[^:]*:\n(.+?)\n(?:def |class )",
        ES._RUNTIME_MODEL_PY,
        flags=re.DOTALL,
    )
    assert body_match, f"could not locate {fn_name} body in runtime template"
    body = body_match.group(1)
    normalize_idx = body.find("normalize_condition(")
    sha_idx = body.find("stable_sha256(")
    assert normalize_idx >= 0, (
        f"{fn_name} does not call normalize_condition -- cache keys will "
        f"diverge from the enqueue side. See "
        f"'the speedup did not actualize' regression."
    )
    assert 0 <= normalize_idx < sha_idx or sha_idx < 0, (
        f"{fn_name} calls stable_sha256 BEFORE normalize_condition -- "
        f"cache key is built from raw condition. See "
        f"'the speedup did not actualize' regression."
    )
    print(f"[OK] {fn_name} normalizes condition before any stable_sha256 call")

print("\nAll checks passed.")
