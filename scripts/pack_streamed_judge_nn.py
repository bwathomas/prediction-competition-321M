"""Pack a streamed batched-judge bundle WITH nearest-neighbor features.

Takes the working streamed-judge bundle (HF offline env vars + per-call
opportunistic flush of judge bs=8 / encoder bs=16, drops ``cache/`` and
``requirements.txt``) and adds back the ``cache/`` directory from the
NN-enabled source so the runtime can populate the 8-scalar NN feature
vector per (subject, item) pair.

NN feature computation at runtime is cheap per pair:

  - lookup precomputed (PCA + int8) item embedding by item_cache_key (mmap)
  - brute-force or FAISS top-K against the shipped index (~1 ms for 5k)
  - aggregate K rows of the sparse subject pass-rate matrix (~1 ms)
  - return locked 8-scalar vector

It adds NO judge or encoder work and runs entirely on CPU.  Per-call
overhead is well under 1 ms once the cache is warm, so it does not
threaten the 10-second per-predict() budget.  Cache load happens once
at module init.

Final bundle: ~109 MB (the cache is the bulk).  Per the user's note,
size is not the platform-side blocker.

Source bundles:
  - submission_streamed_judge.zip   (judge+encoder streamed batching,
                                     HF offline env vars, no cache,
                                     no requirements)
  - submission_nn_judge.zip         (contains the cache/ directory we
                                     want -- 11 files, FAISS index,
                                     int8 embeddings, sparse passrate)
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC_RUNTIME = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_judge.zip")
SRC_CACHE = Path(r"C:\Users\benja\Downloads\submission\submission_nn_judge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_judge_nn.zip")


def _patch_meta_enable_nn(raw: bytes) -> bytes:
    """Take the streamed-judge meta and flip NN features on."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "streamed_batched_judge_nn"
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = True
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC_RUNTIME.exists():
        print(f"ERROR: streamed-judge source missing: {SRC_RUNTIME}", file=sys.stderr)
        return 1
    if not SRC_CACHE.exists():
        print(f"ERROR: NN-cache source missing: {SRC_CACHE}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    # Pull cache file blobs from the NN-judge source bundle ONCE so we
    # can write them straight into the destination without re-reading
    # the 100 MB-class zip on every iteration.
    cache_blobs: dict[str, bytes] = {}
    with zipfile.ZipFile(SRC_CACHE, "r") as zin_cache:
        for info in zin_cache.infolist():
            parts = info.filename.split("/")
            if parts and parts[0] == "cache" and not info.is_dir():
                cache_blobs[info.filename] = zin_cache.read(info)
    if not cache_blobs:
        print(
            f"ERROR: SRC_CACHE has no cache/* entries; cannot enable NN features.",
            file=sys.stderr,
        )
        return 1

    n_runtime_kept = 0
    n_cache_copied = 0

    with zipfile.ZipFile(SRC_RUNTIME, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            if "__pycache__" in info.filename.split("/"):
                continue
            if info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, _patch_meta_enable_nn(zin.read(info)))
            else:
                zout.writestr(info, zin.read(info))
            n_runtime_kept += 1

        # Append the cache/ directory verbatim (paths use forward slashes
        # already, which is what Codabench's Linux unzip expects).
        for filename, blob in cache_blobs.items():
            zout.writestr(filename, blob)
            n_cache_copied += 1

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")
    print(f"  files inherited from streamed-judge: {n_runtime_kept}")
    print(f"  cache/* files copied from NN source: {n_cache_copied}")

    with zipfile.ZipFile(DST, "r") as zf:
        zf_names = set(zf.namelist())
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
        m_txt = zf.read("models.txt").decode("utf-8")
        model_py_head = "\n".join(
            zf.read("model.py").decode("utf-8").splitlines()[:25]
        )

    assert "requirements.txt" not in zf_names, "requirements.txt must remain absent"
    assert "model.py" in zf_names
    assert "labeling.py" in zf_names
    assert "artifacts/checkpoint.pt" in zf_names
    assert "cache/cache_meta.json" in zf_names, (
        "cache_meta.json missing -- TRAINING_CACHE loader will skip"
    )
    cache_files_in_dst = [n for n in zf_names if n.startswith("cache/")]
    assert len(cache_files_in_dst) >= 8, (
        f"too few cache/* entries ({len(cache_files_in_dst)}); NN may be incomplete"
    )
    assert meta["runtime_architecture"] == "streamed_batched_judge_nn"
    assert meta["judge"]["enabled"] is True
    assert meta["judge"]["runtime_batch_size"] == 8
    assert meta["encoder_runtime_batch_size"] == 16
    assert meta["nn_features"]["enabled"] is True
    assert "Qwen/Qwen3-Embedding-4B" in m_txt
    assert "Qwen/Qwen3-4B-Instruct" in m_txt
    assert "HF_HUB_OFFLINE" in model_py_head
    assert "TRANSFORMERS_OFFLINE" in model_py_head
    print(f"  [OK] cache/ has {len(cache_files_in_dst)} files")
    print("  [OK] judge ON, encoder_bs=16, judge_bs=8")
    print("  [OK] nn_features.enabled = True")
    print("  [OK] HF offline env vars preserved at top of model.py")
    print("  [OK] no requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
