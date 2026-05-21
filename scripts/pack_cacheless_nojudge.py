"""Pack a cacheless bundle with the LLM judge disabled.

Sister script to ``pack_cacheless.py``. Same surgery (drop ``cache/``,
drop ``requirements.txt``, set ``nn_features.enabled=false``) plus:

  - ``models.txt``                  drop ``Qwen/Qwen3-4B-Instruct-2507``
                                    so the platform doesn't pre-fetch
                                    the 8 GB judge weights
  - ``runtime_meta.json``           ``judge.enabled = false`` and
                                    ``judge.ship_at_runtime = false``
                                    so the runtime never tries to
                                    instantiate ``_LLMJudgeRuntime``
                                    (``_get_judge_features`` then
                                    returns zeros, matching how the
                                    trained head's input LayerNorm
                                    absorbs the no-judge fingerprint)

Use this as a runtime A/B against ``submission_cacheless.zip``:

  - Same checkpoint, same encoder, same NN-zero fallback.
  - Only difference: judge on vs off.

Expected runtime impact: removing the judge cuts the per-pair forward
from ~250-450 ms down to the encoder cost (~80-150 ms) on L4, i.e.
roughly 3x faster end-to-end. Predictions degrade slightly because
the head's judge-feature input collapses to zero, but the trained
LayerNorm at the head's input was fit with both populated and zero
judge features, so degradation should be modest.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip")
DST = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless_nojudge.zip")

JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def strip_judge_from_models_txt(raw: bytes) -> bytes:
    """Remove any line that names the judge model from models.txt.

    The remaining declared model(s) -- the Qwen3 encoder -- are what
    the platform pre-fetches before the container starts.
    """
    lines = raw.decode("utf-8").splitlines()
    kept = [ln for ln in lines if JUDGE_MODEL_ID not in ln]
    out = "\n".join(kept).rstrip() + "\n"
    return out.encode("utf-8")


def patch_meta_disable_judge_and_nn(raw: bytes) -> bytes:
    """Flip nn_features.enabled and judge.enabled to false; tag the bundle."""
    meta = json.loads(raw)
    meta["runtime_architecture"] = "cacheless_nojudge"
    nn = meta.setdefault("nn_features", {})
    nn["enabled"] = False
    j = meta.setdefault("judge", {})
    j["enabled"] = False
    j["ship_at_runtime"] = False
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
            if info.filename == "models.txt":
                zout.writestr(info.filename, strip_judge_from_models_txt(zin.read(info)))
                n_kept += 1
                continue
            if info.filename == "artifacts/runtime_meta.json":
                zout.writestr(
                    info.filename,
                    patch_meta_disable_judge_and_nn(zin.read(info)),
                )
                n_kept += 1
                continue
            zout.writestr(info, zin.read(info))
            n_kept += 1

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"wrote {DST.name}  ({size_mb:.2f} MB)")
    print(f"  files in source bundle:    {n_total}")
    print(f"  files in nojudge bundle:   {n_kept}")
    print(f"  dropped cache/*:           {n_dropped_cache}")
    print(f"  dropped requirements.txt:  {n_dropped_reqs}")
    print(f"  dropped __pycache__:       {n_dropped_pycache}")

    with zipfile.ZipFile(DST, "r") as zf:
        names = set(zf.namelist())
        m_txt = zf.read("models.txt").decode("utf-8")
        meta = json.loads(zf.read("artifacts/runtime_meta.json"))

    assert not any(n.startswith("cache/") for n in names), "cache/ leaked into bundle"
    assert "requirements.txt" not in names, "requirements.txt should be stripped"
    assert "model.py" in names, "model.py missing"
    assert "labeling.py" in names, "labeling.py missing"
    assert "artifacts/checkpoint.pt" in names, "checkpoint.pt missing"
    assert JUDGE_MODEL_ID not in m_txt, f"models.txt still names {JUDGE_MODEL_ID}"
    assert "Qwen/Qwen3-Embedding-4B" in m_txt, "encoder MUST stay in models.txt"
    assert meta["nn_features"]["enabled"] is False
    assert meta["judge"]["enabled"] is False
    assert meta["judge"]["ship_at_runtime"] is False
    assert meta["runtime_architecture"] == "cacheless_nojudge"
    assert size_mb < 70.0, f"bundle is {size_mb:.2f} MB, over the 70 MB ceiling"
    print("  [OK] no cache/* paths in zip")
    print("  [OK] no requirements.txt")
    print("  [OK] models.txt declares only the encoder")
    print("  [OK] judge.enabled = False  /  ship_at_runtime = False")
    print("  [OK] nn_features.enabled = False")
    print(f"  [OK] {size_mb:.2f} MB < 70 MB ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
