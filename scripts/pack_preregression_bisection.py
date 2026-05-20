"""Pack the pre-regression batched bundle (variant O) for bisection.

Variant A failed -> the regression is in my non-meta-controlled
model.py code. To prove that, we need a bundle that uses the
SAME artifacts as the current batched bundle but the model.py /
labeling.py from BEFORE the regression commit (0853e5f -- 'Speed up
batched-judge runtime with encoder eviction + add progress beacons').

The pre-regression commit is e86cd20 ('Normalize condition in
_enqueue_for_batch so flush keys match predict()'), which is the
batched-flush bundle WITHOUT eviction / progress writes / attn
fallback chain. That commit produced the slow-but-completing version
the user observed running for >1h previously.

If variant O passes where variant A failed, the regression is
*definitively* in my changes between e86cd20 and HEAD; we then
revert those specific changes in src/export_submission.py and
ship a clean fixed bundle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_BUNDLE = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge_batched.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_bisection_real_O_preregression.zip")

PREREG_COMMIT = "e86cd20"


def extract_runtime_templates(commit: str) -> tuple[str, str]:
    """Pull the _RUNTIME_MODEL_PY and _RUNTIME_LABELING_PY string
    literals from the export_submission.py at `commit`."""
    src = subprocess.check_output(
        ["git", "show", f"{commit}:src/export_submission.py"],
        cwd=str(ROOT),
        text=False,
    ).decode("utf-8", errors="replace")
    # Both templates use the raw-triple-quoted form
    #     _RUNTIME_MODEL_PY = r'''...'''
    #     _RUNTIME_LABELING_PY = r'''...'''
    pat_model = re.compile(r"_RUNTIME_MODEL_PY\s*=\s*r'''(.*?)'''", re.DOTALL)
    pat_label = re.compile(r"_RUNTIME_LABELING_PY\s*=\s*r'''(.*?)'''", re.DOTALL)
    m = pat_model.search(src)
    L = pat_label.search(src)
    if not m:
        raise RuntimeError(f"_RUNTIME_MODEL_PY not found in {commit}")
    if not L:
        raise RuntimeError(f"_RUNTIME_LABELING_PY not found in {commit}")
    return m.group(1), L.group(1)


def main() -> int:
    if not SRC_BUNDLE.exists():
        print(f"ERROR: source bundle missing: {SRC_BUNDLE}", file=sys.stderr)
        return 1

    model_py, labeling_py = extract_runtime_templates(PREREG_COMMIT)
    print(
        f"extracted pre-regression templates from {PREREG_COMMIT}: "
        f"model.py={len(model_py)} B, labeling.py={len(labeling_py)} B"
    )

    # Keep meta knobs at the conservative (16, 16, false) so any
    # observed pass/fail is attributable to *code only*, not to meta.
    overrides: dict[str, bytes] = {
        "model.py": model_py.encode("utf-8"),
        "labeling.py": labeling_py.encode("utf-8"),
    }

    if DST.exists():
        DST.unlink()

    n_replaced = 0
    n_copied = 0
    with zipfile.ZipFile(SRC_BUNDLE, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename in overrides:
                zout.writestr(info.filename, overrides[info.filename])
                n_replaced += 1
            elif info.filename == "artifacts/runtime_meta.json":
                meta = json.loads(zin.read(info))
                # Mirror variant A's settings so meta is constant
                # across the comparison: 16/16/false. The
                # pre-regression model.py reads these knobs too --
                # it just doesn't know about FREE_ENCODER_AFTER_FLUSH
                # (older code path), which is fine -- the older
                # runtime simply never calls _free_encoder_vram.
                meta["encoder_runtime_batch_size"] = 16
                if "judge" in meta:
                    meta["judge"]["runtime_batch_size"] = 16
                meta["runtime_architecture"] = "batched_flush_v1"
                zout.writestr(
                    info.filename,
                    json.dumps(meta, indent=2, default=str).encode("utf-8"),
                )
                n_replaced += 1
            else:
                zout.writestr(info, zin.read(info))
                n_copied += 1

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(
        f"  wrote {DST.name}  ({size_mb:.2f} MB)  "
        f"[{n_replaced} replaced, {n_copied} copied]"
    )

    # Quick post-pack verification.
    with zipfile.ZipFile(DST, "r") as zf:
        m = zf.read("model.py").decode("utf-8")
        L = zf.read("labeling.py").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))

    # Pre-regression invariants: these MUST NOT be in the older runtime.
    forbidden_in_model = (
        "def _write_progress",
        "def _free_encoder_vram",
        "FREE_ENCODER_AFTER_FLUSH",
        "attn_candidates",
        "runtime_progress.json",
    )
    for needle in forbidden_in_model:
        if needle in m:
            raise RuntimeError(
                f"variant O looks like it pulled HEAD code, not "
                f"{PREREG_COMMIT}: {needle!r} should NOT be in the "
                f"pre-regression model.py"
            )

    # Pre-regression invariants: these MUST be in the older runtime.
    required_in_model = (
        "def _enqueue_for_batch",
        "def _flush_pending_batches",
        "def score_batch",
        "JUDGE_RUNTIME_BATCH_SIZE",
        "ENCODER_RUNTIME_BATCH_SIZE",
    )
    for needle in required_in_model:
        if needle not in m:
            raise RuntimeError(
                f"variant O missing batched-flush plumbing: {needle!r}"
            )

    assert "_enqueue_for_batch" in L
    assert "return 0.0" in L
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["judge"]["runtime_batch_size"] == 16

    print("verification passed: variant O uses pre-regression runtime "
          "without eviction / progress / attn-fallback code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
