"""Reproduce the state_dict mismatch from submission_judge_batched.zip locally.

Extracts model.py from the zip, parses out just its torch.nn classes +
the _REGISTRY dict (no module-init code), instantiates the model, and
prints the exact list of shape mismatches.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import zipfile
from pathlib import Path

import torch
import torch.nn as nn  # noqa: F401  (used by exec'd source)

ZIP = Path(r"C:\Users\benja\Downloads\submission_judge_batched.zip")


def _extract_model_classes(source: str) -> str:
    """Truncate model.py at the module-init boundary.

    Keeps everything before ``_t0 = time.time()`` -- that's the line that
    marks the start of the heavy module-init block (loads encoder, judge,
    checkpoint, etc.). Everything above it is pure class / function defs
    plus light constant setup, which is safe to exec in isolation.
    """
    marker = "_t0 = time.time()"
    idx = source.find(marker)
    if idx == -1:
        return source
    head = source[:idx]
    return head


def main() -> None:
    td = tempfile.mkdtemp(prefix="diag_state_dict_")
    td_path = Path(td)
    with zipfile.ZipFile(ZIP) as zf:
        zf.extract("model.py", td_path)
        zf.extract("artifacts/checkpoint.pt", td_path)
        zf.extract("artifacts/runtime_meta.json", td_path)

    model_py_src = (td_path / "model.py").read_text(encoding="utf-8")
    ckpt = torch.load(
        td_path / "artifacts" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    meta = json.loads(
        (td_path / "artifacts" / "runtime_meta.json").read_text()
    )

    pruned = _extract_model_classes(model_py_src)
    print("Pruned model.py source (first 400 chars):")
    print(textwrap.shorten(pruned, width=400))
    print()

    import torch.nn as torch_nn
    ns: dict = {
        "nn": torch_nn,
        "torch": torch,
        "math": __import__("math"),
        "__file__": str((td_path / "model.py").resolve()),
        "__name__": "extracted_model",
    }
    exec(compile(pruned, "<extracted_model_classes>", "exec"), ns)
    REGISTRY = ns["_REGISTRY"]

    model_cfg = dict(ckpt["model_cfg"])
    model_name = meta["model_name"]
    state = ckpt["model_state"]

    print(f"model_name from runtime_meta: {model_name}")
    print(f"\nmodel_cfg from checkpoint:")
    for k in sorted(model_cfg):
        print(f"  {k!r}: {model_cfg[k]!r}")

    print(f"\nstate_dict keys (n={len(state)}):")
    for k in sorted(state.keys()):
        v = state[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:60s} {shape}")

    print("\n-- (No repair applied; using checkpoint's model_cfg verbatim) --")

    print("\n-- Constructing model --")
    model = REGISTRY[model_name](model_cfg)
    model_state = model.state_dict()

    print("\n-- Comparison --")
    print(f"  {'KEY':<55} {'MODEL':<30} {'CKPT':<30} STATUS")
    mismatches = []
    for k in sorted(set(model_state.keys()) | set(state.keys())):
        m_shape = tuple(model_state[k].shape) if k in model_state else None
        c_shape = tuple(state[k].shape) if k in state else None
        if m_shape is None:
            status = "UNEXPECTED_IN_CKPT"
        elif c_shape is None:
            status = "MISSING_FROM_CKPT"
        elif m_shape != c_shape:
            status = "SHAPE_MISMATCH"
            mismatches.append((k, m_shape, c_shape))
        else:
            status = "OK"
        if status != "OK":
            print(f"  {k:<55} {str(m_shape):<30} {str(c_shape):<30} {status}")

    print("\n-- load_state_dict(strict=False) --")
    try:
        result = model.load_state_dict(state, strict=False)
        print("OK (no shape errors)")
        print(f"  missing_keys (n={len(result.missing_keys)}): {result.missing_keys[:8]}")
        print(f"  unexpected_keys (n={len(result.unexpected_keys)}): {result.unexpected_keys[:8]}")
    except RuntimeError as exc:
        print("FAILED:")
        print(textwrap.indent(str(exc), "  "))

    print(f"\nNet shape mismatches: {len(mismatches)}")
    for k, m, c in mismatches:
        print(f"  {k}: model expects {m}, ckpt has {c}")


if __name__ == "__main__":
    main()
