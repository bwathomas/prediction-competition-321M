"""Aggressive, proxy-aware subject/benchmark dropout.

Entity-keyed: a randomly chosen subset of subjects (and of benchmarks) is "dropped"
per call, and EVERY row of a dropped entity has ALL of that identity's proxy columns
(node + descendants, via proxy_tree) zeroed. This teaches the model to predict on
subjects/benchmarks it has not seen, and — because proxies are masked atomically —
identity cannot leak back through metadata, conditions, judge scores, or aggregated
features.

Roots are iterated from PROXY_TREE (M4), so adding a new identity axis to the tree +
the entity map below is all it takes; nothing is hardcoded to exactly two axes. The
returned ``info["drop_rows"][root]`` boolean masks let the row-aware leakage probe
(``assert_no_proxy_leak``) check exactly the rows that were supposed to be masked.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import PROXY_TREE, all_masked_columns


def _drop_set(entities, rate: float, rng) -> set:
    uniq = sorted(set(str(e) for e in entities))
    return {e for e in uniq if rng.random() < rate}


def apply_proxy_dropout(X, feature_columns, *, subjects, benchmarks,
                        rng, subject_rate: float, benchmark_rate: float):
    """Return (X_masked, info). Does not mutate X.

    info = {
      "dropped_subjects": set, "dropped_benchmarks": set,
      "drop_rows": {root: bool ndarray of masked rows},
    }
    """
    X = np.asarray(X, dtype=np.float32).copy()
    cols = list(feature_columns)
    if len(set(cols)) != len(cols):
        raise ValueError("duplicate feature column names — masking would be ambiguous (m1)")
    col_idx = {c: i for i, c in enumerate(cols)}

    # entity array + rate per identity root; iterate roots from the tree, not literals
    entity_map = {
        "subject": (subjects, subject_rate),
        "benchmark": (benchmarks, benchmark_rate),
    }
    info = {"dropped_subjects": set(), "dropped_benchmarks": set(), "drop_rows": {}}

    for root in PROXY_TREE:
        if root not in entity_map:
            continue
        entities, rate = entity_map[root]
        ent_arr = np.array([str(e) for e in entities])
        dropped = _drop_set(entities, rate, rng)
        rows = (np.isin(ent_arr, list(dropped)) if dropped
                else np.zeros(len(ent_arr), dtype=bool))
        idx = [col_idx[c] for c in all_masked_columns([root], cols) if c in col_idx]
        if dropped and idx:
            X[np.ix_(rows, idx)] = 0.0
        info["drop_rows"][root] = rows
        info["dropped_subjects" if root == "subject" else "dropped_benchmarks"] = dropped

    return X, info
