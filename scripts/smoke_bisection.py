"""Smoke-run each bisection bundle locally to confirm import + basic flow.

Mimics the platform's protocol: import labeling -> call acquisition_function
a few times -> call predict a few times -> print the progress file. If
any of these crash locally, the bundle has a syntactic / dependency bug
and a Codabench submission is guaranteed to fail too.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST_DIR = Path(r"C:\Users\benja\Downloads\submission")
BUNDLES = [
    "submission_bisection_01_no_torch.zip",
    "submission_bisection_02_torch_cuda.zip",
    "submission_bisection_03_state_share.zip",
]

FAKE_CANDIDATES = [
    {
        "benchmark": "MMLU",
        "condition": "zero-shot",
        "subject_content": "GPT-4 (OpenAI)",
        "item_content": "What is 2+2?",
    },
    {
        "benchmark": "GSM8K",
        "condition": "none",
        "subject_content": "Claude (Anthropic)",
        "item_content": "Solve for x: 2x = 4",
    },
    {
        "benchmark": "HumanEval",
        "condition": "",
        "subject_content": "Llama-3 (Meta)",
        "item_content": "def reverse(s): ...",
    },
]


def run_one(zip_path: Path) -> bool:
    print(f"\n==== {zip_path.name} ====")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td_path)
        sys.path.insert(0, str(td_path))
        try:
            # Force reimport so each bundle starts from a clean module cache.
            for mod in ("model", "labeling"):
                sys.modules.pop(mod, None)
            labeling = importlib.import_module("labeling")
            model = importlib.import_module("model")
            for inp in FAKE_CANDIDATES:
                s = labeling.acquisition_function(inp)
                assert isinstance(s, (int, float)), f"acq returned {type(s)}"
            for inp in FAKE_CANDIDATES:
                p = model.predict(inp, labeled=[])
                assert 0.0 <= p <= 1.0
            # Read back the progress file.
            prog = td_path / "artifacts" / "runtime_progress.json"
            if not prog.exists():
                prog = td_path / "runtime_progress.json"
            if prog.exists():
                doc = json.loads(prog.read_text())
                print(f"  events ({len(doc.get('events', []))}):")
                for ev in doc.get("events", []):
                    keys = {
                        k: v for k, v in ev.items()
                        if k not in ("t_since_start_s",)
                    }
                    print(f"    - {keys}")
                latest = doc.get("latest", {})
                print(f"  latest.stage = {latest.get('stage')!r}")
            else:
                print("  WARNING: no progress file written")
            return True
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            return False
        finally:
            sys.path.remove(str(td_path))


def main() -> int:
    ok = True
    for name in BUNDLES:
        ok = run_one(DST_DIR / name) and ok
    print("\n==== summary ====")
    print("ALL PASSED" if ok else "SOME BUNDLES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
