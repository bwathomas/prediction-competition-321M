"""Bisection bundle 03: cross-module state sharing.

Tests the single architectural assumption that batching depends on:
when ``labeling.py`` imports ``model`` and mutates a module-level
variable, that mutation must be visible to ``predict()`` within the
same round.

The acquisition function writes to ``model._SHARED_COUNTER``.
``predict()`` reads it back into the progress file as
``shared_counter_at_predict``. If acquisition incremented the
counter N>0 times but ``predict()`` sees 0, module-level state is
isolated between the two modules and our batching architecture is
fundamentally incompatible with the platform's import model.

Returns 0.5 either way. No torch / transformers imports so we
isolate the state-sharing question from every other failure mode.
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
    "bundle": "bisection_03_state_share",
    "started_at": _T0,
    "events": [],
}

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
            print(f"[runtime-03] {stage} {info}", flush=True)
        except Exception:
            pass
        body = json.dumps(_STATE, indent=2, default=str).encode("utf-8")
        for p in _PROGRESS_PATHS:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
            except Exception:
                continue
    except Exception:
        pass


# The shared counter the acquisition function will mutate. Initialised
# to 0; every acquisition call should bump it by 1 (the mechanism is
# the SAME pattern the real bundle uses: ``model._ENQUEUED_KEYS.add(...)``
# from inside the labeling module).
_SHARED_COUNTER: int = 0
_LAST_INPUT_AT_ACQ: dict | None = None

_write_progress("module_init", python_version=sys.version.split()[0])

_PREDICT_COUNT = 0
_FIRST_PREDICT_LOGGED = False


def predict(input: dict, labeled=None) -> float:  # noqa: A002
    global _PREDICT_COUNT, _FIRST_PREDICT_LOGGED
    _PREDICT_COUNT += 1
    if not _FIRST_PREDICT_LOGGED:
        _FIRST_PREDICT_LOGGED = True
        _write_progress(
            "predict_first_call",
            shared_counter_at_predict=int(_SHARED_COUNTER),
            last_input_at_acq_keys=(
                sorted(list(_LAST_INPUT_AT_ACQ.keys()))
                if _LAST_INPUT_AT_ACQ
                else None
            ),
            this_input_keys=sorted(list(input.keys())),
            n_labeled=(len(labeled) if labeled else 0),
            python_id_self=id(sys.modules[__name__]),
        )
    if _PREDICT_COUNT in (1, 100, 500, 1000, 5000, 10000):
        _write_progress(
            "predict_milestone",
            predict_count=int(_PREDICT_COUNT),
            shared_counter=int(_SHARED_COUNTER),
        )
    return 0.5
