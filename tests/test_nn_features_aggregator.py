"""Unit tests for ``src.nn_features._aggregate_nn_features``.

These tests cover three properties:

1. The output shape is locked to ``(B, NN_FEATURE_DIM)`` and ``NN_FEATURE_DIM
   == len(NN_FEATURE_NAMES)`` (schema integrity).
2. The first 8 columns reproduce the legacy values bit-identically (so we
   don't silently break consumers that index by integer column).
3. The 7 new self-derived columns satisfy hand-derived formulas on
   carefully constructed synthetic neighborhoods.

Tests are CPU-only and small. They never call FAISS, never load
checkpoints; they purely exercise the aggregator function.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nn_features import (
    NN_FEATURE_DIM,
    NN_FEATURE_NAMES,
    _aggregate_nn_features,
)


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_schema_dim_matches_names_len() -> None:
    assert len(NN_FEATURE_NAMES) == NN_FEATURE_DIM == 23


def test_schema_includes_legacy_first_eight_in_locked_order() -> None:
    """The first 8 names MUST match the legacy schema; downstream code
    indexes by position, so reordering or inserting in the middle would
    silently corrupt every shipped Member 1 / Member 2 / Member 4 model.
    """
    legacy = (
        "passrate_mean",
        "passrate_weighted_mean",
        "passrate_std",
        "coverage",
        "top1_label",
        "top1_similarity",
        "mean_similarity",
        "n_labeled_neighbors_log1p",
    )
    assert NN_FEATURE_NAMES[:8] == legacy


def test_schema_self_derived_features_appear_at_positions_8_through_14() -> None:
    """The 7 self-derived features must occupy columns [8..14]."""
    self_derived = NN_FEATURE_NAMES[8:15]
    assert self_derived == (
        "effective_neighbor_count",
        "top1_minus_topk_similarity",
        "bootstrap_se_passrate",
        "neighbor_label_entropy",
        "top1_label_match",
        "sim_distribution_skew",
        "distance_to_kth_neighbor",
    )


def test_schema_conditional_features_appear_at_positions_15_through_22() -> None:
    """The 8 conditional / context features must occupy columns [15..22] in
    the documented order. Reordering or insertion would break the shipped
    runtime cache contract.
    """
    cond = NN_FEATURE_NAMES[15:23]
    assert cond == (
        "passrate_subject_conditional",
        "passrate_family_conditional",
        "passrate_macro_family_conditional",
        "passrate_organization_conditional",
        "passrate_benchmark_conditional",
        "neighbor_freshness_diff",
        "n_distinct_subjects_in_neighborhood",
        "cluster_passrate_subject_query",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg(passrates, masks, sims, *, fb=0.0, sentinel=-1.0):
    return _aggregate_nn_features(
        np.asarray(passrates, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(sims, dtype=np.float32),
        fallback_value=fb,
        top1_missing_sentinel=sentinel,
    )


# ---------------------------------------------------------------------------
# Output shape + finiteness invariants
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype() -> None:
    B, K = 3, 4
    pr = np.random.default_rng(0).uniform(0, 1, (B, K)).astype(np.float32)
    m = np.ones((B, K), dtype=np.float32)
    s = np.linspace(1.0, 0.0, K, dtype=np.float32)[None, :].repeat(B, axis=0)
    out = _agg(pr, m, s)
    assert out.shape == (B, NN_FEATURE_DIM)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_handles_1d_input_by_treating_as_b1() -> None:
    pr = np.array([0.5, 0.7, 0.3], dtype=np.float32)
    m = np.ones(3, dtype=np.float32)
    s = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    out = _agg(pr, m, s)
    assert out.shape == (1, NN_FEATURE_DIM)


def test_all_missing_neighbors_falls_back_safely() -> None:
    pr = np.zeros((1, 3), dtype=np.float32)
    m = np.zeros((1, 3), dtype=np.float32)
    s = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
    out = _agg(pr, m, s, fb=0.123, sentinel=-1.0)
    # passrate_mean / weighted / std / bootstrap_se / top1_match all use fb
    # when has_any=False; top1_label uses the sentinel.
    names = list(NN_FEATURE_NAMES)
    row = out[0]
    assert row[names.index("passrate_mean")] == pytest.approx(0.123)
    assert row[names.index("passrate_weighted_mean")] == pytest.approx(0.123)
    assert row[names.index("passrate_std")] == pytest.approx(0.123)
    assert row[names.index("top1_label")] == pytest.approx(-1.0)
    assert row[names.index("bootstrap_se_passrate")] == pytest.approx(0.123)
    assert row[names.index("top1_label_match")] == pytest.approx(0.123)
    # entropy of a constant fallback (~0.123) is well-defined and finite,
    # but the aggregator returns 0.0 when has_any is False.
    assert row[names.index("neighbor_label_entropy")] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# New self-derived features
# ---------------------------------------------------------------------------


def test_effective_neighbor_count_peaked_vs_flat() -> None:
    """One dominant neighbor gives eff_count ~ 1; uniform weights give K."""
    K = 4
    # Peaked: one neighbor with sim=1.0, three with sim=0.0 (clipped to 0).
    pr_peaked = np.zeros((1, K), dtype=np.float32)
    m_peaked = np.ones((1, K), dtype=np.float32)
    s_peaked = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    eff_peaked = _agg(pr_peaked, m_peaked, s_peaked)[0, NN_FEATURE_NAMES.index("effective_neighbor_count")]
    assert eff_peaked == pytest.approx(1.0, abs=1e-3)

    # Flat: four neighbors with equal sim.
    s_flat = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    eff_flat = _agg(pr_peaked, m_peaked, s_flat)[0, NN_FEATURE_NAMES.index("effective_neighbor_count")]
    assert eff_flat == pytest.approx(K, abs=1e-3)


def test_top1_minus_topk_similarity_definition() -> None:
    pr = np.zeros((1, 4), dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    s = np.array([[0.9, 0.5, 0.3, 0.1]], dtype=np.float32)  # mean=0.45
    out = _agg(pr, m, s)
    expected = 0.9 - 0.45
    assert out[0, NN_FEATURE_NAMES.index("top1_minus_topk_similarity")] == pytest.approx(expected, abs=1e-5)


def test_bootstrap_se_passrate_matches_std_over_sqrt_n() -> None:
    pr = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    s = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    out = _agg(pr, m, s)
    pr_std = out[0, NN_FEATURE_NAMES.index("passrate_std")]
    se = out[0, NN_FEATURE_NAMES.index("bootstrap_se_passrate")]
    # n_labeled = 4 -> SE = std / sqrt(4) = std / 2
    assert se == pytest.approx(pr_std / 2.0, abs=1e-6)


def test_neighbor_label_entropy_max_at_half() -> None:
    # 50/50 labels should give entropy near ln(2).
    pr = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    s = np.array([[1.0, 0.5, 0.5, 0.5]], dtype=np.float32)
    h = _agg(pr, m, s)[0, NN_FEATURE_NAMES.index("neighbor_label_entropy")]
    assert h == pytest.approx(np.log(2), abs=1e-3)

    # All-1 labels -> entropy near 0.
    pr_all = np.ones((1, 4), dtype=np.float32)
    h_all = _agg(pr_all, m, s)[0, NN_FEATURE_NAMES.index("neighbor_label_entropy")]
    assert h_all < 1e-4


def test_top1_label_match_binarizes_top1() -> None:
    pr_pos = np.array([[0.9, 0.4, 0.5, 0.5]], dtype=np.float32)
    pr_neg = np.array([[0.1, 0.4, 0.5, 0.5]], dtype=np.float32)
    pr_tie = np.array([[0.5, 0.4, 0.5, 0.5]], dtype=np.float32)  # > 0.5 only
    m = np.ones((1, 4), dtype=np.float32)
    s = np.array([[1.0, 0.5, 0.5, 0.5]], dtype=np.float32)
    idx = NN_FEATURE_NAMES.index("top1_label_match")
    assert _agg(pr_pos, m, s)[0, idx] == pytest.approx(1.0)
    assert _agg(pr_neg, m, s)[0, idx] == pytest.approx(0.0)
    assert _agg(pr_tie, m, s)[0, idx] == pytest.approx(0.0)  # strict >


def test_sim_distribution_skew_well_defined() -> None:
    # Symmetric: top1=1.0, median=0.5, min=0.0 -> skew = (1-0.5)/(1-0) = 0.5
    s_sym = np.array([[1.0, 0.5, 0.5, 0.0]], dtype=np.float32)
    pr = np.zeros((1, 4), dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    skew = _agg(pr, m, s_sym)[0, NN_FEATURE_NAMES.index("sim_distribution_skew")]
    assert skew == pytest.approx(0.5, abs=1e-5)


def test_sim_distribution_skew_constant_neighborhood_zero() -> None:
    # All-equal sims: span is 0 -> skew=0 (no division-by-zero blow-up).
    s_const = np.full((1, 4), 0.5, dtype=np.float32)
    pr = np.zeros((1, 4), dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    out = _agg(pr, m, s_const)
    assert out[0, NN_FEATURE_NAMES.index("sim_distribution_skew")] == pytest.approx(0.0)
    assert np.all(np.isfinite(out))


def test_distance_to_kth_neighbor_is_last_column_of_sims() -> None:
    s = np.array([[1.0, 0.7, 0.4, 0.1]], dtype=np.float32)
    pr = np.zeros((1, 4), dtype=np.float32)
    m = np.ones((1, 4), dtype=np.float32)
    out = _agg(pr, m, s)
    assert out[0, NN_FEATURE_NAMES.index("distance_to_kth_neighbor")] == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Regression: legacy columns are unchanged
# ---------------------------------------------------------------------------


def test_legacy_columns_unchanged_against_hand_computed_values() -> None:
    """Sanity-check the existing 8 features against simple hand math so any
    accidental rewrite of the legacy aggregation gets caught."""
    pr = np.array([[1.0, 0.0, 1.0, 0.5]], dtype=np.float32)
    m = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    s = np.array([[1.0, 0.5, 0.5, 0.0]], dtype=np.float32)
    out = _agg(pr, m, s)[0]
    # mean = 0.625, weighted_mean = (1*1 + 0*0.5 + 1*0.5 + 0.5*0)/(1+0.5+0.5+0) = 1.5/2.0 = 0.75
    assert out[NN_FEATURE_NAMES.index("passrate_mean")] == pytest.approx(0.625, abs=1e-5)
    assert out[NN_FEATURE_NAMES.index("passrate_weighted_mean")] == pytest.approx(0.75, abs=1e-5)
    # coverage = 4/4 = 1.0
    assert out[NN_FEATURE_NAMES.index("coverage")] == pytest.approx(1.0)
    # top1_label = 1.0 (top is column 0)
    assert out[NN_FEATURE_NAMES.index("top1_label")] == pytest.approx(1.0)
    # top1_similarity = 1.0
    assert out[NN_FEATURE_NAMES.index("top1_similarity")] == pytest.approx(1.0)
    # mean_similarity = 0.5
    assert out[NN_FEATURE_NAMES.index("mean_similarity")] == pytest.approx(0.5)
    # n_labeled_log1p = log(1+4) = log(5)
    assert out[NN_FEATURE_NAMES.index("n_labeled_neighbors_log1p")] == pytest.approx(float(np.log(5)), abs=1e-5)


def test_aggregator_is_pure_function_of_inputs() -> None:
    """Running twice on the same inputs yields bit-identical outputs."""
    rng = np.random.default_rng(42)
    pr = rng.uniform(0, 1, (5, 6)).astype(np.float32)
    m = rng.uniform(0, 1, (5, 6)).astype(np.float32) > 0.3
    s = rng.uniform(-1, 1, (5, 6)).astype(np.float32)
    a = _agg(pr, m.astype(np.float32), s)
    b = _agg(pr, m.astype(np.float32), s)
    assert np.array_equal(a, b)
