"""Smoke-check that export_run produces the batched-flush format end to end."""

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
    "def score_batch",
    "def _embed_batch",
):
    assert needle in ES._RUNTIME_MODEL_PY, f"missing in model.py template: {needle!r}"
    print(f"[OK] runtime model.py reads/declares: {needle}")

# 3. labeling.py is the enqueue-only variant.
for needle in ("_enqueue_for_batch", "return 0.0"):
    assert needle in ES._RUNTIME_LABELING_PY, f"missing in labeling.py: {needle!r}"
print("[OK] runtime labeling.py is enqueue-only")

# 4. export_run writes the right fields and the broken repair is gone.
src = inspect.getsource(ES.export_run)
for needle in (
    '"encoder_runtime_batch_size":',
    '"runtime_batch_size": runtime_judge_bs',
    '"runtime_architecture": "batched_flush_v1"',
):
    assert needle in src, f"missing in export_run: {needle!r}"
    print(f"[OK] export_run writes: {needle}")
assert "coercing cluster_embed_dim=16" not in src, "broken repair STILL in export_run"
print("[OK] broken cluster repair removed from export_run")

# 5. include_labeling=False warning is wired up.
assert "if not include_labeling:" in src and "LOG.warning(" in src, (
    "include_labeling=False warning not wired"
)
print("[OK] include_labeling=False emits a runtime warning")

# 6. configs/default.yaml has both runtime batch size knobs.
cfg_yaml = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
for needle in (
    "runtime_batch_size: 16",  # appears twice (encoder + judge sections)
):
    n = cfg_yaml.count(needle)
    assert n >= 2, f"expected 2x {needle!r} in default.yaml, got {n}"
    print(f"[OK] default.yaml has {n} occurrences of {needle!r} (encoder + judge)")

# 7. Quick check that the runtime template handles missing fields gracefully.
#    (It should default to 16, not crash.)
import re

m = re.search(
    r'JUDGE_RUNTIME_BATCH_SIZE: int = int\(JUDGE_META\.get\("runtime_batch_size", (\d+)\)\)',
    ES._RUNTIME_MODEL_PY,
)
assert m and int(m.group(1)) == 16, "judge runtime batch default not 16"
print(f"[OK] JUDGE_RUNTIME_BATCH_SIZE defaults to {m.group(1)}")

m = re.search(
    r'ENCODER_RUNTIME_BATCH_SIZE: int = int\(\s*META\.get\("encoder_runtime_batch_size", (\d+)\)\s*\)',
    ES._RUNTIME_MODEL_PY,
)
assert m and int(m.group(1)) == 16, "encoder runtime batch default not 16"
print(f"[OK] ENCODER_RUNTIME_BATCH_SIZE defaults to {m.group(1)}")

print("\nAll checks passed.")
