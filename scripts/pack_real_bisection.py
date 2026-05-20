"""Produce real-bundle bisection variants for the turbo_judge submission.

The current batched bundle fails with no surfaced diagnostics. With
only a pass/fail signal from the platform we need a bisection where
each variant differs by exactly ONE knob, so the first variant that
flips from PASS to FAIL identifies the cause uniquely.

The four meta combinations we ship (all built from the SAME model.py
and labeling.py as the current bundle, so behaviour differences are
strictly attributable to the meta flags):

   variant   encoder_bs   judge_bs   free_encoder_after_flush   purpose
   -------   ----------   --------   ------------------------   -----------------------------------
   A_safe         16           16          false                  most conservative;
                                                                  matches previous slow-working
                                                                  behaviour. If THIS fails, the
                                                                  regression is in non-meta-controlled
                                                                  code (progress writes / attn
                                                                  fallback / new embed paths) and we
                                                                  need to rebuild model.py.
   B_eviction     16           16          true                   eviction alone -- tests whether
                                                                  ``_free_encoder_vram`` is broken.
   C_bs32         16           32          false                  judge bs=32 alone -- tests whether
                                                                  the bigger judge batch is what OOMs.
   D_full         16           32          true                   the current failing combination.
                                                                  Re-emitted for symmetry; you can
                                                                  also just resubmit the existing
                                                                  ``submission_turbo_judge_batched.zip``.

Submission order:
   1. submission_bisection_01_no_torch.zip      (does the platform run us at all?)
   2. submission_bisection_real_A_safe.zip      (does my code work at safe settings?)
   3. submission_bisection_real_B_eviction.zip  (does eviction alone work?)
   4. submission_bisection_real_C_bs32.zip      (does bs=32 alone work?)

The first one that fails -- compared to the one before it -- isolates
the breaking change.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge_batched.zip")
DST_DIR = Path(r"C:\Users\benja\Downloads\submission")

VARIANTS = [
    # (name, encoder_bs, judge_bs, free_encoder_after_flush)
    ("real_A_safe",     16, 16, False),
    ("real_B_eviction", 16, 16, True),
    ("real_C_bs32",     16, 32, False),
    ("real_D_full",     16, 32, True),
]


def patch_meta(raw: bytes, encoder_bs: int, judge_bs: int, free_after_flush: bool) -> bytes:
    meta = json.loads(raw)
    meta["runtime_architecture"] = "batched_flush_v1"
    meta["encoder_runtime_batch_size"] = int(encoder_bs)
    meta["free_encoder_after_flush"] = bool(free_after_flush)
    j = meta.setdefault("judge", {})
    j["runtime_batch_size"] = int(judge_bs)
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def pack_variant(name: str, encoder_bs: int, judge_bs: int, free: bool) -> Path:
    dst = DST_DIR / f"submission_bisection_{name}.zip"
    if dst.exists():
        dst.unlink()

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename == "artifacts/runtime_meta.json":
                patched = patch_meta(zin.read(info), encoder_bs, judge_bs, free)
                zout.writestr(info.filename, patched)
            else:
                zout.writestr(info, zin.read(info))

    # Verify the produced bundle's meta is exactly what we asked for.
    with zipfile.ZipFile(dst, "r") as zf:
        m = json.loads(zf.read("artifacts/runtime_meta.json"))
    assert m["encoder_runtime_batch_size"] == encoder_bs
    assert m["judge"]["runtime_batch_size"] == judge_bs
    assert m["free_encoder_after_flush"] is free
    return dst


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1

    print(
        f"{'variant':24s} {'enc_bs':>6s} {'jdg_bs':>6s} {'free':>6s}  "
        f"{'size_MB':>9s}  path"
    )
    print("-" * 110)
    for name, ebs, jbs, free in VARIANTS:
        dst = pack_variant(name, ebs, jbs, free)
        size_mb = dst.stat().st_size / (1024 * 1024)
        free_str = "true" if free else "false"
        print(
            f"{name:24s} {ebs:>6d} {jbs:>6d} {free_str:>6s}  "
            f"{size_mb:>9.2f}  {dst.name}"
        )

    print()
    print("Suggested submission order (one at a time; stop at first FAIL):")
    print("  1) submission_bisection_01_no_torch.zip")
    print("  2) submission_bisection_real_A_safe.zip")
    print("  3) submission_bisection_real_B_eviction.zip")
    print("  4) submission_bisection_real_C_bs32.zip")
    print()
    print("Decision table:")
    print("  01 fails               -> platform issue, escalate to organizers")
    print("  01 ok, A fails         -> regression in my new model.py code")
    print("                            (not meta-controlled). Roll back code.")
    print("  01,A ok; B fails       -> encoder eviction (_free_encoder_vram) broken")
    print("  01,A,B ok; C fails     -> bs=32 judge OOMs at the actual workload")
    print("  01,A,B,C all pass      -> intermittent / something else; resubmit D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
