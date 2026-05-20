"""Bisection bundle 01: minimal model.py with no torch / transformers.

Purpose: prove that the platform can run our submission at all and
that our progress-file writes are surfaced via the artifacts view.

Returns 0.5 for every prediction. Records every call in
``artifacts/runtime_progress.json`` AND ``./runtime_progress.json``
AND ``/tmp/runtime_progress.json`` (whichever paths are writable) so
at least one surface is available no matter how the platform mounts
the container's filesystem. Mirrors all milestones to ``print(...,
flush=True)`` so platforms that surface stdout get the same signal.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

_T0 = time.time()
_STATE: dict = {
    "bundle": "bisection_01_no_torch",
    "started_at": _T0,
    "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_T0)),
    "events": [],
}

# Paths we try, in priority order. The first writable one wins; the
# others are populated on best-effort if accessible.
_PROGRESS_PATHS = [
    ARTIFACTS / "runtime_progress.json",
    HERE / "runtime_progress.json",
    Path("/tmp/runtime_progress.json"),
    Path(os.environ.get("TMPDIR", "/tmp")) / "runtime_progress.json",
]


def _write_progress(stage: str, **info) -> None:
    try:
        ev = {
            "stage": stage,
            "t_since_start_s": round(time.time() - _T0, 3),
            **info,
        }
        _STATE["events"].append(ev)
        _STATE["latest"] = ev
        try:
            print(f"[runtime-01] {stage} {info}", flush=True)
        except Exception:
            pass
        try:
            sys.stdout.flush()
        except Exception:
            pass
        # Write to every accessible path.
        body = json.dumps(_STATE, indent=2, default=str).encode("utf-8")
        for p in _PROGRESS_PATHS:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
            except Exception:
                continue
    except Exception:
        pass


_write_progress(
    "module_init",
    python_version=sys.version.split()[0],
    cwd=str(Path.cwd()),
    here=str(HERE),
    artifacts_exists=ARTIFACTS.exists(),
    listdir_here=sorted(os.listdir(HERE))[:30],
)

_PREDICT_COUNT = 0
_FIRST_PREDICT_LOGGED = False


def predict(input: dict, labeled=None) -> float:  # noqa: A002
    global _PREDICT_COUNT, _FIRST_PREDICT_LOGGED
    _PREDICT_COUNT += 1
    if not _FIRST_PREDICT_LOGGED:
        _FIRST_PREDICT_LOGGED = True
        _write_progress(
            "predict_first_call",
            input_keys=sorted(list(input.keys())),
            n_labeled=(len(labeled) if labeled else 0),
            sample_benchmark=str(input.get("benchmark", ""))[:60],
            sample_condition=str(input.get("condition", ""))[:60],
        )
    if _PREDICT_COUNT % 500 == 0:
        _write_progress("predict_progress", predict_count=int(_PREDICT_COUNT))
    return 0.5
