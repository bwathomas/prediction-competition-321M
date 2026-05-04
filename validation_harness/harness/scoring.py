"""Scoring metrics.

Primary metric: mean log-likelihood (Bernoulli, soft-label safe), higher is
better. We use mean log p (NOT cross-entropy) because the spec literally
asks for "negative log-loss" = mean log probability of the true labels.

We also report AUC-ROC when labels are effectively binary (>=2 classes after
rounding to {0, 1}). AUC is computed via the rank formula -- no scikit-learn
dependency.

Two reporting variants per round:
- score_excluding_adaptively_labeled_rows  -- the main score
- score_including_all_rows                 -- a sanity check; reported but
  not promoted unless the platform proves it scores labeled rows too
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .rounds import RoundResult


def mean_log_likelihood(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    eps: float = 1e-7,
) -> float:
    """Mean Bernoulli log-likelihood. Higher (closer to 0) is better.

    Soft-label-safe: works for y_true in [0, 1]. Labels outside [0, 1] are
    clipped to [0, 1] before scoring so the metric is always non-positive and
    finite -- some response tables in this dataset are continuous /
    out-of-range (the README warns about this) and the user is responsible
    for any meaningful binarization upstream.
    """
    y = np.clip(np.asarray(list(y_true), dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(list(y_pred), dtype=float), eps, 1.0 - eps)
    if y.size == 0:
        return float("nan")
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return float(np.mean(ll))


def fraction_labels_outside_unit_interval(y_true: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    if y.size == 0:
        return 0.0
    out_of_range = (y < 0.0) | (y > 1.0)
    return float(out_of_range.mean())


def auc_roc(y_true: Iterable[float], y_pred: Iterable[float]) -> float | None:
    """Rank-based AUC. Returns None if labels are not effectively binary
    or if only one class is present.
    """
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    if y.size == 0:
        return None
    yb = np.where(y >= 0.5, 1, 0)
    pos = (yb == 1).sum()
    neg = (yb == 0).sum()
    if pos == 0 or neg == 0:
        return None
    if not np.allclose(y, yb, atol=1e-6):
        return None  # treat soft labels as not strictly binary
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_rank_sum = ranks[yb == 1].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


@dataclass
class ScoreSlice:
    n: int
    log_likelihood: float
    auc_roc: float | None
    frac_labels_clipped: float


@dataclass
class RoundScore:
    excluding_labeled: ScoreSlice
    including_all: ScoreSlice
    n_categories: int
    n_labeled: int
    used_random_acquisition: bool
    fallback_reason: str | None

    def main(self) -> float:
        """The headline score (excluding adaptively-labeled rows)."""
        return self.excluding_labeled.log_likelihood


def _score_slice(df: pd.DataFrame) -> ScoreSlice:
    if len(df) == 0:
        return ScoreSlice(
            n=0, log_likelihood=float("nan"), auc_roc=None, frac_labels_clipped=0.0
        )
    y = df["label"].astype(float).values
    p = df["_pred"].astype(float).values
    return ScoreSlice(
        n=len(df),
        log_likelihood=mean_log_likelihood(y, p),
        auc_roc=auc_roc(y, p),
        frac_labels_clipped=fraction_labels_outside_unit_interval(y),
    )


def score_round(result: RoundResult) -> RoundScore:
    cand = result.candidates
    if "_pred" not in cand.columns or "_is_labeled" not in cand.columns:
        raise ValueError("score_round expects a RoundResult from run_official_like_round")
    excl = cand[~cand["_is_labeled"]]
    return RoundScore(
        excluding_labeled=_score_slice(excl),
        including_all=_score_slice(cand),
        n_categories=result.n_categories,
        n_labeled=result.n_labeled,
        used_random_acquisition=result.used_random_acquisition,
        fallback_reason=result.fallback_reason,
    )
