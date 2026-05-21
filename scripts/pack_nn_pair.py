"""Pack the two NN-enabled bundles (NN + no judge, NN + judge).

Sister script to ``pack_cacheless.py`` / ``pack_cacheless_nojudge.py``.
Together the four scripts form a 2x2 design over the two knobs the user
is currently varying:

                            judge ON              judge OFF
    cache ON   (NN ON)      submission_nn_judge   submission_nn_nojudge
    cache OFF  (NN OFF)     submission_cacheless  submission_cacheless_nojudge

Cache "ON" here means the bundle ships the full quantized training-item
cache that ``submission_turbo_judge.zip`` was exported with -- the
int8+PCA-256 embeddings, the FAISS HNSW index, the sparse subject
pass-rate matrices, the subject indexer JSON, and the NN-feature config.
The runtime ``_TrainingItemCache`` loader mmaps embeddings_int8.npy and
materializes the fp32 dequant on first use; ``_get_nn_features`` then
returns the locked 8-scalar feature vector per (subject, item) instead
of the all-zeros fallback.

Both bundles drop ``requirements.txt`` because the platform README at
``starting_kit/README.md`` lines 239-243 confirms additive
``requirements.txt`` support is *organizer-controlled and defaults to
disabled in the example deployment*. Including one is dead weight at
best; it's not what's keeping us from the 70 MB budget.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip")
DST_NOJUDGE = Path(r"C:\Users\benja\Downloads\submission\submission_nn_nojudge.zip")
DST_JUDGE = Path(r"C:\Users\benja\Downloads\submission\submission_nn_judge.zip")

JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def strip_judge_from_models_txt(raw: bytes) -> bytes:
    """Drop any line that names the judge model."""
    lines = raw.decode("utf-8").splitlines()
    kept = [ln for ln in lines if JUDGE_MODEL_ID not in ln]
    return ("\n".join(kept).rstrip() + "\n").encode("utf-8")


def patch_meta(raw: bytes, *, judge_enabled: bool, arch_tag: str) -> bytes:
    """Patch runtime_meta.json: NN stays enabled; judge toggled per knob."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = arch_tag
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = True
    j = meta.setdefault("judge", {})
    j["enabled"] = bool(judge_enabled)
    j["ship_at_runtime"] = bool(judge_enabled)
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def _pack(dst: Path, *, judge_enabled: bool, arch_tag: str) -> None:
    if dst.exists():
        dst.unlink()

    n_total = 0
    n_kept = 0
    n_dropped_reqs = 0
    n_dropped_pycache = 0
    cache_files_kept = 0

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            n_total += 1
            parts = info.filename.split("/")
            if "__pycache__" in parts:
                n_dropped_pycache += 1
                continue
            if info.filename == "requirements.txt":
                n_dropped_reqs += 1
                continue
            if info.filename == "models.txt":
                if judge_enabled:
                    zout.writestr(info, zin.read(info))
                else:
                    zout.writestr(info.filename, strip_judge_from_models_txt(zin.read(info)))
                n_kept += 1
                continue
            if info.filename == "artifacts/runtime_meta.json":
                zout.writestr(
                    info.filename,
                    patch_meta(zin.read(info), judge_enabled=judge_enabled, arch_tag=arch_tag),
                )
                n_kept += 1
                continue
            if parts and parts[0] == "cache":
                cache_files_kept += 1
            zout.writestr(info, zin.read(info))
            n_kept += 1

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"wrote {dst.name}  ({size_mb:.2f} MB)")
    print(f"  files in source bundle:    {n_total}")
    print(f"  files in target bundle:    {n_kept}")
    print(f"  cache/* files retained:    {cache_files_kept}")
    print(f"  dropped requirements.txt:  {n_dropped_reqs}")
    print(f"  dropped __pycache__:       {n_dropped_pycache}")

    with zipfile.ZipFile(dst, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
        m_txt = zf.read("models.txt").decode("utf-8")

    assert "model.py" in names, "model.py missing"
    assert "labeling.py" in names, "labeling.py missing"
    assert "artifacts/checkpoint.pt" in names, "checkpoint.pt missing"
    assert "requirements.txt" not in names, "requirements.txt should be stripped"
    assert any(n.startswith("cache/") for n in names), "cache/ missing -- NN cannot work"
    assert "cache/cache_meta.json" in names, "cache_meta.json missing -- loader will skip"
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    if judge_enabled:
        assert JUDGE_MODEL_ID in m_txt, "judge must stay in models.txt when enabled"
        assert meta["judge"]["enabled"] is True
        assert meta["judge"]["ship_at_runtime"] is True
    else:
        assert JUDGE_MODEL_ID not in m_txt, "judge must NOT be in models.txt when disabled"
        assert meta["judge"]["enabled"] is False
        assert meta["judge"]["ship_at_runtime"] is False
    assert meta["nn_features"]["enabled"] is True
    assert meta["runtime_architecture"] == arch_tag
    print(f"  [OK] cache/* retained ({cache_files_kept} files)")
    print("  [OK] no requirements.txt")
    if judge_enabled:
        print("  [OK] models.txt declares encoder + judge")
        print("  [OK] judge.enabled = True / ship_at_runtime = True")
    else:
        print("  [OK] models.txt declares only the encoder")
        print("  [OK] judge.enabled = False / ship_at_runtime = False")
    print("  [OK] nn_features.enabled = True")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1
    print("==== submission_nn_nojudge.zip (NN ON, judge OFF) ====")
    _pack(DST_NOJUDGE, judge_enabled=False, arch_tag="nn_nojudge")
    print()
    print("==== submission_nn_judge.zip   (NN ON, judge ON)  ====")
    _pack(DST_JUDGE, judge_enabled=True, arch_tag="nn_judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
