"""Tests for scoring metrics and round scoring."""

from __future__ import annotations

import math

import numpy as np

from harness.rounds import run_official_like_round
from harness.scoring import auc_roc, mean_log_likelihood, score_round
from harness.splits import add_item_variant_id, make_item_cold_start_split
from tests.synthetic import (
    make_spy_labeling,
    make_spy_model,
    make_synthetic_df,
)


def test_log_likelihood_perfect_predictions():
    y = [0, 1, 0, 1, 1]
    p = [0.0, 1.0, 0.0, 1.0, 1.0]
    ll = mean_log_likelihood(y, p, eps=1e-7)
    assert ll > math.log(1 - 1e-7) - 1e-3


def test_log_likelihood_constant_half_is_log_half():
    y = [0, 1, 0, 1, 0, 1]
    p = [0.5] * 6
    assert abs(mean_log_likelihood(y, p) - math.log(0.5)) < 1e-9


def test_log_likelihood_clips_predictions():
    y = [1]
    p = [0.0]  # would be -inf without prediction clipping
    assert math.isfinite(mean_log_likelihood(y, p, eps=1e-7))


def test_log_likelihood_clips_out_of_range_labels():
    """Some response tables are continuous / out-of-range; we clip y to [0, 1]
    so the metric is always non-positive."""
    y = [8.5, -2.0, 0.5]
    p = [0.5, 0.5, 0.5]
    ll = mean_log_likelihood(y, p)
    assert ll <= 0.0
    assert math.isfinite(ll)


def test_auc_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auc_roc(y, p) == 1.0


def test_auc_inverse_separation():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert auc_roc(y, p) == 0.0


def test_auc_returns_none_for_single_class():
    assert auc_roc([1, 1, 1], [0.1, 0.2, 0.9]) is None


def test_auc_returns_none_for_soft_labels():
    assert auc_roc([0.3, 0.7], [0.4, 0.6]) is None


def test_score_round_separates_labeled_and_unlabeled():
    df = add_item_variant_id(make_synthetic_df())
    train, val, _, _ = make_item_cold_start_split(df, val_fraction=0.5, seed=0)
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    s = score_round(result)
    n_labeled = result.n_labeled
    n_total = result.n_candidates
    assert s.excluding_labeled.n + n_labeled == n_total
    assert s.including_all.n == n_total
    assert s.n_labeled == n_labeled
    assert s.n_categories == result.n_categories


def test_score_round_main_returns_excluding_labeled():
    df = add_item_variant_id(make_synthetic_df())
    train, val, _, _ = make_item_cold_start_split(df, val_fraction=0.5, seed=0)
    model = make_spy_model()
    labeling = make_spy_labeling()
    result = run_official_like_round(train, val, model, labeling, N=20, K=2, seed=0)
    s = score_round(result)
    assert s.main() == s.excluding_labeled.log_likelihood
