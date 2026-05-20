"""Bisection bundle 03 labeling.py: mutates a counter inside ``model``.

This is the EXACT pattern the real batched bundle uses: import the
sibling module, mutate one of its module-level variables, and trust
that ``predict()`` reading the same variable later in the same round
will see the mutation. If this bundle's progress shows that
``predict()`` reads 0 even after acquisition has been called many
times, our entire batching architecture is unworkable on the platform
and we need to redesign.
"""

from __future__ import annotations

import sys

import model


def acquisition_function(input: dict) -> float:  # noqa: A002
    model._SHARED_COUNTER = getattr(model, "_SHARED_COUNTER", 0) + 1
    model._LAST_INPUT_AT_ACQ = dict(input)
    if model._SHARED_COUNTER == 1:
        model._write_progress(
            "acq_first_call",
            shared_counter=int(model._SHARED_COUNTER),
            python_id_model=id(model),
            python_id_model_via_sys=id(sys.modules["model"]),
            python_id_self=id(sys.modules[__name__]),
        )
    if model._SHARED_COUNTER % 1000 == 0:
        model._write_progress(
            "acq_progress", shared_counter=int(model._SHARED_COUNTER)
        )
    return 0.0
