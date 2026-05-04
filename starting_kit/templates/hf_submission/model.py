"""Template: local HuggingFace model submission.

Use this template when your method needs one or more HuggingFace models
listed in `models.txt`. The worker pre-downloads those repos before the
container starts; loading them at module init hits the local HF cache only.

Implement your scoring method in `predict()`.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# ---------------------------------------------------------------------------


def _declared_models() -> list[str]:
    models_path = Path(__file__).with_name("models.txt")
    if not models_path.exists():
        return []
    return [
        line.strip()
        for line in models_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(Path(__file__).with_name(".hf_cache")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


MODEL_LOADED = False
TOKENIZER = None
MODEL = None
REPO_ID = ""

_declared = _declared_models()
print(f"[hf_submission] Declared models: {_declared}", flush=True)

if _declared:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        REPO_ID = _declared[0]
        cache_dir = _resolve_cache_dir()
        TOKENIZER = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=cache_dir)
        MODEL = AutoModel.from_pretrained(REPO_ID, cache_dir=cache_dir)
        if torch.cuda.is_available():
            MODEL = MODEL.to("cuda")
            print("[hf_submission] Loaded model on CUDA.", flush=True)
        else:
            print("[hf_submission] Loaded model on CPU.", flush=True)
        MODEL_LOADED = True
    except Exception as exc:
        print(f"[hf_submission] Could not load model: {exc}", flush=True)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if not MODEL_LOADED:
        return 0.5

    # Replace this with your actual scoring logic. `input` exposes the
    # curated keys: benchmark, condition, subject_content, item_content.
    return 0.5
