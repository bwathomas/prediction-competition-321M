"""Bisection bundle 01: minimal labeling.py.

Returns 0.0 (a finite scalar) for every candidate. Records first and
periodic acquisition calls via the shared progress writer in
``model``. Importing ``model`` here also serves as a basic sanity
check that the platform's import order works -- if labeling.py loads
before model.py and Python can't find ``model``, we'd fail fast.
"""

from __future__ import annotations

import model  # noqa: F401 -- triggers model's module init + progress write


def acquisition_function(input: dict) -> float:  # noqa: A002
    model._ACQ_COUNT = getattr(model, "_ACQ_COUNT", 0) + 1
    if not getattr(model, "_FIRST_ACQ_LOGGED", False):
        model._FIRST_ACQ_LOGGED = True
        model._write_progress(
            "acq_first_call",
            input_keys=sorted(list(input.keys())),
            sample_benchmark=str(input.get("benchmark", ""))[:60],
            sample_condition=str(input.get("condition", ""))[:60],
        )
    if model._ACQ_COUNT % 1000 == 0:
        model._write_progress("acq_progress", acq_count=int(model._ACQ_COUNT))
    return 0.0
