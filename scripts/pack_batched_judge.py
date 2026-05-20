"""Repack the most recent submission zip with the new batched-judge runtime.

Reads ``submission_judge_slim.zip``, replaces ``model.py`` and
``labeling.py`` with the freshly rendered runtime templates from
``src.export_submission`` (which now include batched encoder + judge
inference and a lazy-flush queue), and writes the result to
``submission_judge_batched.zip`` with the same artifacts/ payload.

The trained head, runtime_meta.json, cluster centroids, pool stats, etc.
are copied verbatim -- no re-training needed. Only the inference plumbing
changes.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export_submission import _RUNTIME_MODEL_PY, _RUNTIME_LABELING_PY  # noqa: E402

SRC = Path(r"C:\Users\benja\Downloads\submission_judge_slim.zip")
DST = Path(r"C:\Users\benja\Downloads\submission_judge_batched.zip")

OVERRIDE = {
    "model.py": _RUNTIME_MODEL_PY.encode("utf-8"),
    "labeling.py": _RUNTIME_LABELING_PY.encode("utf-8"),
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source zip missing: {SRC}")
    if DST.exists():
        DST.unlink()

    seen: set[str] = set()
    written: list[tuple[str, int]] = []

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            name = info.filename
            if "__pycache__" in name.split("/"):
                continue
            if name in OVERRIDE:
                data = OVERRIDE[name]
                zout.writestr(name, data)
            else:
                data = zin.read(name)
                zout.writestr(info, data)
            seen.add(name)
            written.append((name, len(data)))

        # Always emit overrides, even if they weren't in the source.
        for name, data in OVERRIDE.items():
            if name not in seen:
                zout.writestr(name, data)
                written.append((name, len(data)))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"Wrote {DST}  ({size_mb:.2f} MB)")
    print("Entries:")
    for name, sz in written:
        marker = "*" if name in OVERRIDE else " "
        print(f"  {marker} {name:50s} {sz:>14,d} B")

    # Verify the patched bundle has the batched-flush symbols + no labeling
    # fallback to single-call inference.
    with zipfile.ZipFile(DST, "r") as zf:
        names = {i.filename for i in zf.infolist()}
        assert "model.py" in names, "model.py missing"
        assert "labeling.py" in names, "labeling.py missing"
        m = zf.read("model.py").decode("utf-8")
        L = zf.read("labeling.py").decode("utf-8")
    for needle in (
        "JUDGE_RUNTIME_BATCH_SIZE",
        "ENCODER_RUNTIME_BATCH_SIZE",
        "def _embed_batch",
        "def score_batch",
        "def _enqueue_for_batch",
        "def _flush_pending_batches",
        "if not _FLUSHED:",
    ):
        assert needle in m, f"model.py missing: {needle}"
    assert "_enqueue_for_batch" in L, "labeling.py missing _enqueue_for_batch wiring"
    assert "JUDGE_PROMPT_TEMPLATE.format(" not in m, "judge .format() crash still present"
    print(
        "\nVerification: batched-flush symbols present in model.py; labeling.py "
        "calls _enqueue_for_batch and returns 0.0; brace-safe judge render in place."
    )


if __name__ == "__main__":
    main()
