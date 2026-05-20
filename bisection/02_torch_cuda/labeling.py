"""Bisection bundle 02 labeling.py: same enqueue-only pattern as bundle 01."""

from __future__ import annotations

import model  # noqa: F401


def acquisition_function(input: dict) -> float:  # noqa: A002
    model._ACQ_COUNT = getattr(model, "_ACQ_COUNT", 0) + 1
    if not getattr(model, "_FIRST_ACQ_LOGGED", False):
        model._FIRST_ACQ_LOGGED = True
        model._write_progress(
            "acq_first_call",
            n_input_keys=len(input),
        )
    if model._ACQ_COUNT % 1000 == 0:
        model._write_progress("acq_progress", acq_count=int(model._ACQ_COUNT))
    return 0.0
