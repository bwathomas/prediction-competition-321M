"""Aggressive, proxy-aware subject/benchmark dropout.

Entity-keyed: a randomly chosen subset of subjects (and of benchmarks) is "dropped"
per call, and EVERY row of a dropped entity has ALL of that identity's proxy columns
(node + descendants, via proxy_tree) zeroed. This teaches the model to predict on
subjects/benchmarks it has not seen, and — because proxies are masked atomically —
identity cannot leak back through metadata, conditions, or aggregated features.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import all_masked_columns


def _drop_set(entities, rate: float, rng) -> set:
    uniq = sorted(set(str(e) for e in entities))
    return {e for e in uniq if rng.random() < rate}


def apply_proxy_dropout(X, feature_columns, *, subjects, benchmarks,
                        rng, subject_rate: float, benchmark_rate: float):
    """Return (X_masked, info). Does not mutate X."""
    X = np.asarray(X, dtype=np.float32).copy()
    cols = list(feature_columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    dropped_subj = _drop_set(subjects, subject_rate, rng)
    dropped_bench = _drop_set(benchmarks, benchmark_rate, rng)

    subj_cols = all_masked_columns(["subject"], cols)
    bench_cols = all_masked_columns(["benchmark"], cols)

    subj_arr = np.array([str(s) for s in subjects])
    bench_arr = np.array([str(b) for b in benchmarks])

    if dropped_subj:
        rows = np.isin(subj_arr, list(dropped_subj))
        idx = [col_idx[c] for c in subj_cols if c in col_idx]
        if idx:
            X[np.ix_(rows, idx)] = 0.0
    if dropped_bench:
        rows = np.isin(bench_arr, list(dropped_bench))
        idx = [col_idx[c] for c in bench_cols if c in col_idx]
        if idx:
            X[np.ix_(rows, idx)] = 0.0

    return X, {"dropped_subjects": dropped_subj, "dropped_benchmarks": dropped_bench}
