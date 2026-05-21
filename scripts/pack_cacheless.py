"""Pack a cache-less, requirements-less bundle for Codabench.

Diagnosis lead: every 114 MB bundle we have shipped has failed at
PAIEC-UNKNOWN-001 within 1-5 minutes (non-batched, no-judge, fix1 no SDPA,
preregression -- all of them). The ONE bundle that worked,
``submission_judge_slim.zip``, was 15 MB and shipped:

  - no ``cache/`` directory          (no FAISS / embeddings load at init)
  - no ``requirements.txt``          (platform doesn't try a pip install)
  - ``nn_features.enabled = false``  (smaller, simpler model.py path)

We can't reproduce the slim trained model (it was a k=16 checkpoint
trained before NN features were enabled), but we CAN strip everything
the slim bundle didn't have from the current ``submission_turbo_judge``
bundle and rely on the runtime's graceful fallback in ``_get_nn_features``
(returns zeros when ``TRAINING_CACHE is None``).

If this PASSES:
  -> bundle size / cache loading was the killer.  Future bundles must
     skip ``cache/`` and ``requirements.txt``.  We lose the NN-feature
     signal but everything else (item-IRT, encoder, judge) is intact.

If this FAILS the same way:
  -> the regression lives deeper in model.py.  We then bisect by
     swapping the cache-less ``model.py`` for the slim model.py with
     the new checkpoint.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless.zip")


def patch_meta_disable_nn_features(raw: bytes) -> bytes:
    """Flip ``nn_features.enabled`` to ``false`` and tag the bundle.

    Cosmetic -- the runtime's cache load is already gated by the
    ``cache/cache_meta.json`` file's existence -- but it makes the
    intent explicit in the bundle metadata.
    """
    meta = json.loads(raw)
    meta["runtime_architecture"] = "cacheless"
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = False
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source bundle missing: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        DST.unlink()

    n_total = 0
    n_kept = 0
    n_dropped_cache = 0
    n_dropped_reqs = 0
    n_dropped_pycache = 0

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        DST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            n_total += 1
            parts = info.filename.split("/")
            if "__pycache__" in parts:
                n_dropped_pycache += 1
                continue
            if parts and parts[0] == "cache":
                n_dropped_cache += 1
                continue
            if info.filename == "requirements.txt":
                n_dropped_reqs += 1
                continue
            if info.filename == "artifacts/runtime_meta.json":
                zout.writestr(info.filename, patch_meta_disable_nn_features(zin.read(info)))
                n_kept += 1
                continue
            zout.writestr(info, zin.read(info))
            n_kept += 1

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")
    print(f"  files in source bundle:   {n_total}")
    print(f"  files in cacheless bundle: {n_kept}")
    print(f"  dropped cache/*:          {n_dropped_cache}")
    print(f"  dropped requirements.txt: {n_dropped_reqs}")
    print(f"  dropped __pycache__:      {n_dropped_pycache}")

    with zipfile.ZipFile(DST, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))
    assert not any(n.startswith("cache/") for n in names), "cache/ leaked into bundle"
    assert "requirements.txt" not in names, "requirements.txt should be stripped"
    assert "model.py" in names, "model.py missing"
    assert "labeling.py" in names, "labeling.py missing"
    assert "artifacts/checkpoint.pt" in names, "checkpoint.pt missing"
    assert "artifacts/runtime_meta.json" in names, "runtime_meta.json missing"
    assert meta["nn_features"]["enabled"] is False
    assert meta["runtime_architecture"] == "cacheless"
    assert size_mb < 70.0, f"bundle is {size_mb:.2f} MB, over the 70 MB ceiling"
    print("  [OK] no cache/* paths in zip")
    print("  [OK] no requirements.txt")
    print("  [OK] nn_features.enabled = False")
    print(f"  [OK] {size_mb:.2f} MB < 70 MB ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
