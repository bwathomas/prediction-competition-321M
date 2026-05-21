"""Pack a no-judge non-batched bundle.

Non-batched ALSO failed at PAIEC-UNKNOWN-001 within 1-5 min. Since the
non-batched bundle does no batching at all (no-op labeling.py, flush is
a no-op, predict falls through to per-pair), the failure must be in
module init or in code that runs regardless of batching.

The largest single thing module init does is load the LLM-as-judge
(Qwen3-4B-Instruct-2507, ~ 8 GB in bf16). If we aren't on an L4 24 GB
GPU but a smaller tier (T4 16 GB? a smaller Modal default?), 8 GB
encoder + 8 GB judge alone already saturates VRAM, before any forward
pass.

This bundle removes the judge entirely so module init only loads the
encoder. If it PASSES, the judge load is the OOM cause and we know
the GPU tier is smaller than expected. We can then either: (a) skip
the judge and accept the prediction-quality hit, or (b) quantize the
judge to 4-bit (cuts to ~ 2 GB) and add it back.

Changes vs the failing non-batched bundle:
  - models.txt: drop Qwen/Qwen3-4B-Instruct-2507 (platform won't
    pre-fetch the judge weights)
  - runtime_meta.json: judge.enabled = false (model.py won't try to
    instantiate _LLMJudgeRuntime)
  - labeling.py: same no-op as the non-batched control
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export_submission import _RUNTIME_MODEL_PY  # noqa: E402

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_nojudge.zip")

NOOP_LABELING_PY = '''"""No-op acquisition function: returns 0.0 for every candidate.

The judge is disabled in this bundle; we also stay non-batched so the
runtime has minimum complexity. acquisition_function() is called once
per candidate but does nothing; predict() falls through to the per-pair
(bs=1) inference path with zero judge features.
"""

from __future__ import annotations


def acquisition_function(input: dict) -> float:  # noqa: A002
    return 0.0
'''


def strip_judge_from_models_txt(raw: bytes) -> bytes:
    """Drop any line that mentions the judge model id from models.txt."""
    judge_id = "Qwen/Qwen3-4B-Instruct-2507"
    lines = raw.decode("utf-8").splitlines()
    kept = [ln for ln in lines if judge_id not in ln]
    return ("\n".join(kept) + "\n").encode("utf-8")


def patch_meta_disable_judge(raw: bytes) -> bytes:
    """Disable the judge in runtime_meta.json; keep everything else."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "nonbatched_nojudge"
    meta["encoder_runtime_batch_size"] = 8
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["enabled"] = False
    j["ship_at_runtime"] = False
    j["runtime_batch_size"] = 8
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename == "model.py":
                zout.writestr(info.filename, _RUNTIME_MODEL_PY.encode("utf-8"))
            elif info.filename == "labeling.py":
                zout.writestr(info.filename, NOOP_LABELING_PY.encode("utf-8"))
            elif info.filename == "models.txt":
                zout.writestr(info.filename, strip_judge_from_models_txt(zin.read(info)))
            elif info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, patch_meta_disable_judge(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")

    # Verification.
    with zipfile.ZipFile(DST, "r") as zf:
        m_txt = zf.read("models.txt").decode("utf-8")
        L = zf.read("labeling.py").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))

    assert "Qwen3-4B-Instruct" not in m_txt, "judge still in models.txt!"
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    assert "_enqueue_for_batch" not in L, "labeling.py should be a no-op"
    assert "return 0.0" in L
    assert meta["judge"]["enabled"] is False
    assert meta["judge"]["ship_at_runtime"] is False
    print("  [OK] models.txt declares only the encoder")
    print("  [OK] labeling.py is a no-op (non-batched)")
    print("  [OK] meta judge.enabled = False / ship_at_runtime = False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
