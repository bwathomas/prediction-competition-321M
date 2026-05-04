"""Tiny synthetic dataset shared across tests.

Designed so that the harness's invariants are easy to check:
- 4 benchmarks (== 4 data categories)
- 8 subjects, all observed in every benchmark
- per benchmark: 10 unique items, half emitted with condition "none",
  half with condition "skill=foo"
- labels are deterministic from (subject_idx, item_idx) so log-loss is
  meaningful but not degenerate

Total rows: 4 benchmarks * 10 items * 8 subjects = 320 rows.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def make_synthetic_df() -> pd.DataFrame:
    rows = []
    for b_idx, benchmark in enumerate(["bench_a", "bench_b", "bench_c", "bench_d"]):
        for i in range(10):
            cond = "none" if i < 5 else "skill=foo"
            item_id = f"{benchmark}_item_{i}"
            content = f"What is {benchmark} item {i}?"
            for s in range(8):
                subj_id = f"subj_{s}"
                rows.append(
                    {
                        "subject_id": subj_id,
                        "item_id": item_id,
                        "benchmark": benchmark,
                        "condition": cond,
                        "subject_content": f"Name: model_{s}",
                        "item_content": content,
                        "label": float((b_idx + i + s) % 2),
                    }
                )
    df = pd.DataFrame(rows)
    df["data_category"] = df["benchmark"]
    return df


def make_spy_model():
    """Return a module-like object that records every predict() call."""
    state = {"calls": []}

    def predict(input, labeled=None):
        state["calls"].append(
            {
                "input": dict(input),
                "input_keys": tuple(sorted(input.keys())),
                "input_types": {k: type(v).__name__ for k, v in input.items()},
                "labeled_len": 0 if labeled is None else len(labeled),
                "labeled_keys": (
                    None if not labeled else tuple(sorted(labeled[0].keys()))
                ),
            }
        )
        return 0.5

    return SimpleNamespace(predict=predict, _spy_state=state)


def make_spy_labeling(score_fn=None, raise_on=None, return_nonfinite_on=None):
    """Return a module-like object that records every acquisition_function call.

    score_fn(input, idx) -> float : custom scoring (default: 0.0)
    raise_on(idx) -> bool : when True, raise RuntimeError instead of returning
    return_nonfinite_on(idx) -> bool : when True, return float('nan')
    """
    state = {"calls": []}

    def acquisition_function(input):
        idx = len(state["calls"])
        state["calls"].append(
            {
                "input": dict(input),
                "input_keys": tuple(sorted(input.keys())),
                "input_types": {k: type(v).__name__ for k, v in input.items()},
            }
        )
        if raise_on is not None and raise_on(idx):
            raise RuntimeError("intentional acquisition error for test")
        if return_nonfinite_on is not None and return_nonfinite_on(idx):
            return float("nan")
        if score_fn is None:
            return 0.0
        return float(score_fn(input, idx))

    return SimpleNamespace(acquisition_function=acquisition_function, _spy_state=state)
