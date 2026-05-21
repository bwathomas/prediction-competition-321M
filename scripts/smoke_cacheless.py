"""Local smoke test for submission_cacheless.zip.

Extracts the bundle to a temp dir, imports ``model.py`` as a module, and
verifies it gets through module init without raising.  This catches the
obvious "model.py crashes immediately" class of bugs that would otherwise
manifest as PAIEC-UNKNOWN-001 on Codabench.

We do NOT call ``predict()`` here -- that requires the encoder + judge
weights, which we don't want to download locally just to validate the
bundle.  Module init alone exercises:

  - runtime_meta.json parsing
  - the cache-loading guard (must skip cleanly when ``cache/`` is absent)
  - checkpoint.pt loading and state_dict shape compatibility
  - the LoRA-mode branch (must skip when mode == "none")
  - the encoder / judge *constructor* code paths
  ... up to but not including the network downloads themselves.

Since the model.py downloads weights at module import (which would be
expensive locally), we monkeypatch the encoder / judge model loaders to
return a stub.  This isolates the bundle-specific code from the model
weights.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback
import types
import zipfile
from pathlib import Path

BUNDLE = Path(r"C:\Users\benja\Downloads\submission\submission_cacheless.zip")


def _install_stubs() -> None:
    """Stub heavyweight HF loaders so we don't download weights."""
    try:
        import transformers
    except ImportError:
        print("[smoke] transformers not installed locally; install or skip")
        return

    def _stub_from_pretrained(*args, **kwargs):
        m = types.SimpleNamespace()
        m.eval = lambda: m
        m.to = lambda *a, **k: m
        m.half = lambda: m
        m.bfloat16 = lambda: m
        m.parameters = lambda: iter([])
        return m

    transformers.AutoModel.from_pretrained = staticmethod(_stub_from_pretrained)
    transformers.AutoModelForCausalLM.from_pretrained = staticmethod(_stub_from_pretrained)

    class _StubTokenizer:
        pad_token = "[PAD]"
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, *a, **k):
            import torch

            return {"input_ids": torch.zeros((1, 8), dtype=torch.long), "attention_mask": torch.ones((1, 8), dtype=torch.long)}

        def encode(self, *a, **k):
            return [0]

        def decode(self, *a, **k):
            return ""

        def convert_tokens_to_ids(self, t):
            return 0 if isinstance(t, str) else [0 for _ in t]

    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: _StubTokenizer())


def main() -> int:
    if not BUNDLE.exists():
        print(f"ERROR: bundle missing: {BUNDLE}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "bundle"
        root.mkdir()
        with zipfile.ZipFile(BUNDLE, "r") as zf:
            zf.extractall(root)
        print(f"[smoke] extracted to {root}")
        print("[smoke] files:")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(root)}  ({p.stat().st_size} B)")

        cache_dir = root / "cache"
        assert not cache_dir.exists(), "cache/ unexpectedly present in cacheless bundle"

        _install_stubs()

        sys.path.insert(0, str(root))
        try:
            spec = importlib.util.spec_from_file_location("submission_model", root / "model.py")
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                print("[smoke] module init RAISED:")
                traceback.print_exc()
                return 2
            print("[smoke] module init OK")

            for name in ("predict", "_get_nn_features", "TRAINING_CACHE"):
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    if name == "TRAINING_CACHE":
                        print(f"[smoke] TRAINING_CACHE = {val!r}  (expected: None)")
                        assert val is None, "TRAINING_CACHE should be None when cache/ is absent"
                    else:
                        print(f"[smoke] has {name}: {val}")
                else:
                    print(f"[smoke] WARNING: missing {name}")
            print("[smoke] PASS")
            return 0
        finally:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(main())
