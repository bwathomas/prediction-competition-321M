"""Scoring metrics. ``log_loss`` (mean binary cross-entropy, lower=better) is the
objective the agents minimize; ``auc_roc`` is the secondary debug metric.
"""
from __future__ import annotations

import numpy as np


def log_loss(y, p, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _rankdata_average(a) -> np.ndarray:
    """Ranks 1..n with average rank for ties (scipy-free)."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    pos = np.arange(1, n + 1, dtype=float)
    r = np.empty(n, dtype=float)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and sorted_a[j + 1] == sorted_a[k]:
            j += 1
        r[k:j + 1] = pos[k:j + 1].mean()
        k = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = r
    return ranks


def auc_roc(y, p):
    """Rank (Mann-Whitney) AUC. Returns None if labels are not binary or single-class."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    uniq = np.unique(y)
    if len(uniq) < 2 or not np.all(np.isin(uniq, [0, 1])):
        return None
    ranks = _rankdata_average(p)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    sum_pos = float(np.sum(ranks[y == 1]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
