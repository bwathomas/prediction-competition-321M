"""Pack a streamed-encoder bundle WITH nearest-neighbor features (no judge).

Fills the fourth cell of the streamed 2x2 design (judge x NN):

                            judge ON                     judge OFF
    NN ON    submission_streamed_judge_nn.zip   submission_streamed_encoder_nn.zip
    NN OFF   submission_streamed_judge.zip      submission_streamed_encoder.zip

Recipe: take the working streamed-encoder no-judge bundle (HF offline
env vars at top of model.py, streamed encoder batching inside
``acquisition_function``, judge disabled) and copy in the ``cache/``
directory from ``submission_nn_judge.zip``, flipping
``nn_features.enabled`` to True so the runtime's NN code path
actually consults the cache.

What this exercises that ``submission_streamed_encoder.zip`` does not:

  - ``_TrainingItemCache`` constructor at module init (~ 100 MB cache
    load: mmap embeddings_int8.npy, parquet read of item_keys, sparse
    NPZ load of subject_passrate / subject_passrate_mask, JSON load of
    subject_key_to_id, optional FAISS index read).
  - ``_get_nn_features`` per ``predict()`` call: PCA-project the item
    embedding, top-K cosine search (FAISS or brute-force numpy),
    aggregate K rows of the sparse subject pass-rate matrix into the
    locked 8-scalar vector.

What it does NOT exercise:

  - the judge model (``Qwen/Qwen3-4B-Instruct-2507`` is absent from
    ``models.txt`` so the platform does not pre-download it, and
    ``judge.enabled=false`` keeps ``JUDGE=None`` so the per-pair
    judge-feature call returns zeros).

Final bundle: ~109 MB.  Per the user's note, size is not the
platform-side blocker.

Source bundles:
  - submission_streamed_encoder.zip   (working baseline, log loss 0.64)
  - submission_nn_judge.zip           (donor of the cache/ directory)
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC_RUNTIME = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_encoder.zip")
SRC_CACHE = Path(r"C:\Users\benja\Downloads\submission\submission_nn_judge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_streamed_encoder_nn.zip")


def _patch_meta_enable_nn(raw: bytes) -> bytes:
    """Take the streamed-encoder meta and flip NN features on."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "streamed_encoder_nojudge_nn"
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = True
    j = meta.setdefault("judge", {})
    assert j.get("enabled") is False, "source bundle must have judge disabled"
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC_RUNTIME.exists():
        print(f"ERROR: runtime source missing: {SRC_RUNTIME}", file=sys.stderr)
        return 1
    if not SRC_CACHE.exists():
        print(f"ERROR: cache donor missing: {SRC_CACHE}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    cache_blobs: dict[str, bytes] = {}
    with zipfile.ZipFile(SRC_CACHE, "r") as zin_cache:
        for info in zin_cache.infolist():
            parts = info.filename.split("/")
            if parts and parts[0] == "cache" and not info.is_dir():
                cache_blobs[info.filename] = zin_cache.read(info)
    if not cache_blobs:
        print("ERROR: SRC_CACHE has no cache/* entries", file=sys.stderr)
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
        for filename, blob in cache_blobs.items():
            zout.writestr(filename, blob)
            n_cache_copied += 1

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")
    print(f"  files inherited from streamed-encoder: {n_runtime_kept}")
    print(f"  cache/* files copied from NN source:   {n_cache_copied}")

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
        f"too few cache/* entries ({len(cache_files_in_dst)})"
    )
    assert meta["runtime_architecture"] == "streamed_encoder_nojudge_nn"
    assert meta["judge"]["enabled"] is False
    assert meta["judge"]["ship_at_runtime"] is False
    assert meta["nn_features"]["enabled"] is True
    assert meta["encoder_runtime_batch_size"] == 16
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    assert "Qwen/Qwen3-4B-Instruct" not in m_txt, (
        "judge must NOT be in models.txt when disabled"
    )
    assert "HF_HUB_OFFLINE" in model_py_head, "HF_HUB_OFFLINE must be at top of model.py"
    assert "TRANSFORMERS_OFFLINE" in model_py_head, (
        "TRANSFORMERS_OFFLINE must be at top of model.py"
    )
    print(f"  [OK] cache/ has {len(cache_files_in_dst)} files")
    print("  [OK] judge OFF, nn_features ON, encoder_bs=16 (streamed)")
    print("  [OK] models.txt declares only the encoder")
    print("  [OK] HF offline env vars preserved at top of model.py")
    print("  [OK] no requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
