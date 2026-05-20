"""Repack submission_turbo_judge.zip with the batched-flush runtime.

Strategy: preserve every artifact (trained checkpoint, NN cache, cluster
centroids, pool stats, models.txt, requirements.txt) byte-for-byte, and
overwrite only the inference plumbing -- ``model.py`` and
``labeling.py`` -- with the current templates from
``src/export_submission.py``. Also augments ``runtime_meta.json`` with
the three new fields the batched-flush runtime expects
(``runtime_architecture``, ``encoder_runtime_batch_size``,
``judge.runtime_batch_size``), defaulting both batch sizes to 16 so
the bundle is L4-safe out of the box.

The trained head, the NN cache, and all other artifacts are unchanged
-- only the runtime / labeling logic is replaced. This is the same
strategy used for ``submission_judge_batched.zip``.
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
DST = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge_batched.zip")


def patch_runtime_meta(raw: bytes) -> bytes:
    """Bring the meta forward to the encoder-eviction batched-flush runtime.

    Overwrites the batched-flush knobs in place (using ``=``, not
    ``setdefault``) so an older bundle that was packed with bs=16 picks
    up the new defaults (bs=32 + encoder eviction) on re-pack.
    """
    meta = json.loads(raw)
    meta["runtime_architecture"] = "batched_flush_v1"
    # Encoder runs co-resident with the judge -> bs=16 ceiling on L4.
    # Judge runs after encoder eviction -> bs=32 is safe and gives the
    # actual speedup (judge phase is the dominant cost).
    meta["encoder_runtime_batch_size"] = 16
    meta["free_encoder_after_flush"] = True
    judge_block = meta.setdefault("judge", {})
    judge_block["runtime_batch_size"] = 32
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source zip missing: {SRC}")
    if DST.exists():
        DST.unlink()

    overrides_text: dict[str, bytes] = {
        "model.py": _RUNTIME_MODEL_PY.encode("utf-8"),
        "labeling.py": _RUNTIME_LABELING_PY.encode("utf-8"),
    }

    written: list[tuple[str, int, str]] = []  # (name, size, marker)
    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            name = info.filename
            if "__pycache__" in name.split("/"):
                continue
            if name in overrides_text:
                data = overrides_text[name]
                zout.writestr(name, data)
                written.append((name, len(data), "REPLACE"))
            elif name == "artifacts/runtime_meta.json":
                data = patch_runtime_meta(zin.read(name))
                zout.writestr(name, data)
                written.append((name, len(data), "PATCH"))
            else:
                data = zin.read(name)
                zout.writestr(info, data)
                written.append((name, len(data), "COPY"))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"Wrote {DST}  ({size_mb:.2f} MB)\n")
    print("Entries:")
    for name, sz, marker in written:
        flag = {"REPLACE": "*", "PATCH": "~", "COPY": " "}[marker]
        print(f"  {flag} {name:55s} {sz:>14,d} B  [{marker}]")

    # Sanity check the produced zip.
    with zipfile.ZipFile(DST, "r") as zf:
        names = {i.filename for i in zf.infolist()}
        assert "model.py" in names and "labeling.py" in names
        assert "artifacts/checkpoint.pt" in names
        assert "artifacts/runtime_meta.json" in names

        m = zf.read("model.py").decode("utf-8")
        L = zf.read("labeling.py").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))

    print("\n-- post-pack verification --")
    for needle in (
        "JUDGE_RUNTIME_BATCH_SIZE",
        "ENCODER_RUNTIME_BATCH_SIZE",
        "def _embed_batch",
        "def _enqueue_for_batch",
        "def _flush_pending_batches",
        "def score_batch",
        "if not _FLUSHED:",
        # New in encoder-eviction + observability runtime:
        "def _write_progress",
        "def _free_encoder_vram",
        "FREE_ENCODER_AFTER_FLUSH",
        "runtime_progress.json",
        # Explicit attention fallback chain:
        "attn_candidates",
        '"sdpa"',
    ):
        assert needle in m, f"model.py missing: {needle}"
        print(f"  [OK] model.py contains: {needle}")
    assert "JUDGE_PROMPT_TEMPLATE.format(" not in m, "judge .format() crash still present"
    print("  [OK] judge prompt render is brace-safe")
    assert "_enqueue_for_batch" in L and "return 0.0" in L
    print("  [OK] labeling.py is enqueue-only")
    assert meta["runtime_architecture"] == "batched_flush_v1"
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["judge"]["runtime_batch_size"] == 32
    assert meta["free_encoder_after_flush"] is True
    print(f"  [OK] runtime_meta.runtime_architecture = {meta['runtime_architecture']!r}")
    print(f"  [OK] runtime_meta.encoder_runtime_batch_size = {meta['encoder_runtime_batch_size']}")
    print(f"  [OK] runtime_meta.judge.runtime_batch_size = {meta['judge']['runtime_batch_size']}")
    print(f"  [OK] runtime_meta.free_encoder_after_flush = {meta['free_encoder_after_flush']}")


if __name__ == "__main__":
    main()
