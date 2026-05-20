"""Pack the SDPA-reverted bundle for the next platform submission.

Variant A (meta=16,16,false) failed with PAIEC-UNKNOWN-001 ("no details
safe to show") -- the platform's catch-all for OS-level kills (OOM /
SIGKILL), not a Python exception. Module init succeeds, so the failure
happens during execution.

The most plausible cause: my new ``flash_attention_2 -> sdpa -> default``
attn-impl fallback chain. If FA2 isn't installed in the platform's L4
image (very plausible -- the handbook explicitly says additive packages
are organizer-controlled and disabled by default), the new chain lands
on SDPA, which has ~ 3-4 GB more activation memory than the eager
fallback the old code used. With both 4B-param bf16 models co-resident
(~ 16 GB weights) on a 24 GB L4 at bs=16, SDPA tips us over into OOM
where eager wouldn't.

The fix in src/export_submission.py reverts to the original
``FA2 -> default`` chain, which lands on *eager* (low memory) when FA2
isn't available -- the slow-but-completing path. This bundle is shipped
at the conservative meta (encoder_bs=16, judge_bs=16,
free_encoder_after_flush=false) -- matching variant A's meta exactly,
so any pass/fail difference is strictly attributable to the SDPA
removal.
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
DST = Path(r"C:\Users\benja\Downloads\submission\submission_fix1_no_sdpa.zip")


def patch_meta(raw: bytes) -> bytes:
    meta = json.loads(raw)
    meta["runtime_architecture"] = "batched_flush_v1"
    meta["encoder_runtime_batch_size"] = 16
    meta["free_encoder_after_flush"] = False
    j = meta.setdefault("judge", {})
    j["runtime_batch_size"] = 16
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    overrides: dict[str, bytes] = {
        "model.py": _RUNTIME_MODEL_PY.encode("utf-8"),
        "labeling.py": _RUNTIME_LABELING_PY.encode("utf-8"),
    }

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename in overrides:
                zout.writestr(info.filename, overrides[info.filename])
            elif info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, patch_meta(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")

    # Verify the SDPA fallback was actually removed from the shipped model.py.
    with zipfile.ZipFile(DST, "r") as zf:
        m = zf.read("model.py").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
    assert 'attn_candidates' not in m, "attn_candidates loop still present"
    assert '"sdpa"' not in m, '"sdpa" still in attn fallback'
    assert 'attn_implementation"] = "flash_attention_2"' in m, (
        "FA2 fallback should still be tried first"
    )
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["judge"]["runtime_batch_size"] == 16
    assert meta["free_encoder_after_flush"] is False
    print("  [OK] SDPA fallback removed from model.py")
    print("  [OK] FA2 fallback still tried first")
    print("  [OK] meta = (encoder=16, judge=16, free=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
