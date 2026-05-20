"""Inspect submission_turbo_judge.zip end to end.

Reports:
  - Manifest of entries + sizes
  - runtime_meta.json content
  - models.txt content
  - Whether model.py already has the batched-flush architecture
  - Whether labeling.py is enqueue-only or the legacy uncertainty variant
  - Whether the runtime template carries the broken cluster_embed_dim
    "repair" we just removed
  - state_dict load_state_dict against the constructed model class
    (catches the residual / embedding shape-mismatch we saw last time)
  - Cache-key invariant: do _enqueue_for_batch and _predict_uncalibrated
    normalize condition before any stable_sha256 call?
  - Whether predict()'s flush is wired correctly
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

import torch
import torch.nn as nn  # noqa: F401  (used by exec'd source)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ZIP = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\Users\benja\Downloads\submission\submission_turbo_judge.zip"
)


def _extract_pre_init_source(source: str) -> str:
    marker = "_t0 = time.time()"
    idx = source.find(marker)
    return source if idx == -1 else source[:idx]


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(f"missing zip: {ZIP}")

    size_mb = ZIP.stat().st_size / (1024 * 1024)
    print(f"=== {ZIP.name}  ({size_mb:.2f} MB) ===\n")

    td = Path(tempfile.mkdtemp(prefix="inspect_turbo_"))
    with zipfile.ZipFile(ZIP) as zf:
        infos = zf.infolist()
        print(f"Entries ({len(infos)}):")
        for info in infos:
            mb = info.file_size / (1024 * 1024)
            cmb = info.compress_size / (1024 * 1024)
            print(f"  {info.filename:55s} {info.file_size:>14,d} B  ({mb:>6.2f} MB, "
                  f"{cmb:>6.2f} MB on disk)")
        names = {info.filename for info in infos}
        for needed in ("model.py", "labeling.py", "artifacts/checkpoint.pt",
                       "artifacts/runtime_meta.json"):
            tag = "OK" if needed in names else "MISSING"
            print(f"  [{tag}] {needed}")

        for n in (
            "model.py",
            "labeling.py",
            "models.txt",
            "artifacts/checkpoint.pt",
            "artifacts/runtime_meta.json",
            "artifacts/cluster_centroids.npy",
            "artifacts/pool_features_stats.json",
        ):
            if n in names:
                zf.extract(n, td)

    has_labeling = (td / "labeling.py").exists()
    has_model = (td / "model.py").exists()
    has_ckpt = (td / "artifacts" / "checkpoint.pt").exists()
    has_meta = (td / "artifacts" / "runtime_meta.json").exists()
    if not (has_model and has_ckpt and has_meta):
        raise SystemExit("FATAL: bundle is missing required files")

    model_py = (td / "model.py").read_text(encoding="utf-8")
    meta = json.loads((td / "artifacts" / "runtime_meta.json").read_text())
    labeling_py = (td / "labeling.py").read_text(encoding="utf-8") if has_labeling else ""
    models_txt = (
        (td / "models.txt").read_text(encoding="utf-8").strip()
        if (td / "models.txt").exists()
        else "<missing>"
    )

    print("\n--- models.txt ---")
    print(textwrap.indent(models_txt, "  "))

    print("\n--- runtime_meta.json ---")
    print(textwrap.indent(json.dumps(meta, indent=2, default=str), "  "))

    print("\n--- runtime architecture audit ---")
    audit = {
        "ships_labeling_py": has_labeling,
        "runtime_architecture_tag": meta.get("runtime_architecture", "(absent)"),
        "encoder_runtime_batch_size_field": meta.get("encoder_runtime_batch_size"),
        "judge.runtime_batch_size_field": (meta.get("judge") or {}).get("runtime_batch_size"),
        "has_enqueue_for_batch": "def _enqueue_for_batch" in model_py,
        "has_flush_pending_batches": "def _flush_pending_batches" in model_py,
        "has_embed_batch": "def _embed_batch" in model_py,
        "has_score_batch": "def score_batch" in model_py,
        "predict_flushes_on_entry": "_flush_pending_batches()" in model_py
            and ("if not _FLUSHED:" in model_py),
        "labeling_calls_enqueue": ("_enqueue_for_batch" in labeling_py) if has_labeling else False,
        "labeling_calls_predict_uncalibrated": (
            "_predict_uncalibrated" in labeling_py if has_labeling else False
        ),
        "has_broken_cluster_repair": "coercing cluster_embed_dim=16" in model_py,
        "has_judge_format_crash": "JUDGE_PROMPT_TEMPLATE.format(" in model_py,
    }
    for k, v in audit.items():
        tag = "OK" if (v if isinstance(v, bool) else v not in (None, "(absent)")) else "----"
        if k.startswith("has_broken") or k.startswith("has_judge_format"):
            tag = "BUG" if v else "OK"
        print(f"  [{tag}] {k}: {v!r}")

    print("\n--- cache-key invariant check ---")
    for fn in ("_enqueue_for_batch", "_predict_uncalibrated"):
        if f"def {fn}(" not in model_py:
            print(f"  [SKIP] {fn}: not present in this bundle")
            continue
        body_m = re.search(
            rf"def {re.escape(fn)}\([^)]*\)[^:]*:\n(.+?)\n(?:def |class )",
            model_py,
            flags=re.DOTALL,
        )
        if not body_m:
            print(f"  [WARN] {fn}: body not located")
            continue
        body = body_m.group(1)
        n_idx = body.find("normalize_condition(")
        s_idx = body.find("stable_sha256(")
        if n_idx < 0 and s_idx < 0:
            print(f"  [OK]   {fn}: no key built, no normalization needed")
        elif n_idx < 0 and s_idx >= 0:
            print(f"  [BUG]  {fn}: builds key without normalize_condition")
        elif s_idx < 0:
            print(f"  [OK]   {fn}: normalizes condition (no key built here)")
        elif n_idx < s_idx:
            print(f"  [OK]   {fn}: normalize_condition before stable_sha256")
        else:
            print(f"  [BUG]  {fn}: stable_sha256 BEFORE normalize_condition")

    print("\n--- state_dict load check ---")
    ckpt = torch.load(td / "artifacts" / "checkpoint.pt", map_location="cpu", weights_only=False)
    model_cfg = dict(ckpt["model_cfg"])
    print(f"  model_name from runtime_meta: {meta.get('model_name')}")
    print(f"  checkpoint model_cfg:")
    for k in sorted(model_cfg):
        print(f"    {k!r}: {model_cfg[k]!r}")
    state = ckpt["model_state"]
    print(f"\n  saved state_dict has {len(state)} tensors")

    pruned = _extract_pre_init_source(model_py)
    ns: dict = {
        "nn": __import__("torch.nn", fromlist=["nn"]),
        "torch": torch,
        "math": __import__("math"),
        "__file__": str((td / "model.py").resolve()),
        "__name__": "extracted_model",
    }
    exec(compile(pruned, "<turbo_model_classes>", "exec"), ns)
    REGISTRY = ns.get("_REGISTRY")
    if REGISTRY is None:
        print("  [BUG] _REGISTRY not found in pre-init section -- runtime template diverges")
        return

    model = REGISTRY[meta["model_name"]](model_cfg)
    model_state = model.state_dict()

    mismatches = []
    for k in sorted(set(model_state.keys()) | set(state.keys())):
        m_shape = tuple(model_state[k].shape) if k in model_state else None
        c_shape = tuple(state[k].shape) if k in state else None
        if m_shape is None:
            print(f"  [UNEXPECTED_IN_CKPT] {k}: ckpt shape={c_shape}")
        elif c_shape is None:
            print(f"  [MISSING_FROM_CKPT] {k}: model shape={m_shape}")
        elif m_shape != c_shape:
            mismatches.append((k, m_shape, c_shape))
            print(f"  [SHAPE_MISMATCH]    {k}: model={m_shape} ckpt={c_shape}")

    try:
        result = model.load_state_dict(state, strict=False)
        print(f"\n  load_state_dict(strict=False) OK")
        print(f"    missing_keys      : {len(result.missing_keys)}")
        print(f"    unexpected_keys   : {len(result.unexpected_keys)}")
        print(f"    shape mismatches  : {len(mismatches)}")
    except RuntimeError as exc:
        print(f"\n  load_state_dict(strict=False) FAILED:")
        print(textwrap.indent(str(exc), "    "))


if __name__ == "__main__":
    main()
