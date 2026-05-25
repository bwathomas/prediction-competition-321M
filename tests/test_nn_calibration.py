"""Unit tests for the Netflix-Prize-style NN calibrator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.nn_calibration import (
    NNCalibrator,
    NNCalibratorState,
    SubjectResidualTable,
)


# ---------------------------------------------------------------------------
# SubjectResidualTable
# ---------------------------------------------------------------------------


def test_residual_table_from_rows_basic():
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 0, 1, 1, 2],
        training_item_rows=[5, 7, 5, 3, 8],
        labels=[0.0, 1.0, 1.0, 0.0, 1.0],
        uncal_probs=[0.4, 0.6, 0.5, 0.7, 0.2],
        n_subjects=3,
        n_training_items=10,
    )
    assert table.n_subjects == 3
    assert table.n_training_items == 10
    # 5 unique (subject, item) pairs -> 5 entries.
    assert int(table.passrate_data.shape[0]) == 5
    # CSR indptr is monotone.
    assert (np.diff(table.passrate_indptr) >= 0).all()


def test_residual_table_aggregates_duplicates():
    """Multiple rows for the same (subject, item) get mean-aggregated."""
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 0, 0],
        training_item_rows=[5, 5, 5],
        labels=[0.0, 1.0, 1.0],
        uncal_probs=[0.2, 0.4, 0.6],
        n_subjects=1,
        n_training_items=10,
    )
    assert int(table.passrate_data.shape[0]) == 1
    np.testing.assert_allclose(table.passrate_data[0], 2.0 / 3.0, rtol=1e-4)
    np.testing.assert_allclose(table.uncal_prob_data[0], 0.4, rtol=1e-4)


def test_residual_table_lookup_hit_and_miss():
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 0, 1],
        training_item_rows=[2, 5, 5],
        labels=[1.0, 0.5, 0.7],
        uncal_probs=[0.3, 0.6, 0.8],
        n_subjects=2,
        n_training_items=10,
    )
    # Subject 0 has rows 2 and 5 recorded.
    y, p, m = table.lookup(0, np.array([5, 2, 7], dtype=np.int64))
    np.testing.assert_array_equal(m, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(y[m > 0], [0.5, 1.0])
    np.testing.assert_allclose(p[m > 0], [0.6, 0.3])
    # Subject out of range -> all miss.
    y2, p2, m2 = table.lookup(99, np.array([2, 5], dtype=np.int64))
    assert (m2 == 0.0).all()


def test_residual_table_save_load_roundtrip(tmp_path: Path):
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 1, 1, 2, 2, 2],
        training_item_rows=[3, 1, 4, 2, 5, 9],
        labels=[1.0, 0.0, 0.5, 0.7, 0.2, 0.9],
        uncal_probs=[0.6, 0.4, 0.3, 0.5, 0.1, 0.8],
        n_subjects=4,
        n_training_items=10,
    )
    table.save(tmp_path)
    loaded = SubjectResidualTable.load(tmp_path)
    np.testing.assert_array_equal(table.passrate_indptr, loaded.passrate_indptr)
    np.testing.assert_array_equal(table.passrate_indices, loaded.passrate_indices)
    np.testing.assert_array_equal(table.passrate_data, loaded.passrate_data)
    np.testing.assert_array_equal(table.uncal_prob_data, loaded.uncal_prob_data)
    assert loaded.n_subjects == table.n_subjects
    assert loaded.n_training_items == table.n_training_items
    # Meta sidecar is well-formed JSON.
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["nnz"] == int(table.passrate_data.shape[0])


def test_residual_table_validates_ranges():
    with pytest.raises(ValueError):
        SubjectResidualTable.from_rows(
            subject_ids=[5],  # out of range
            training_item_rows=[0],
            labels=[1.0],
            uncal_probs=[0.5],
            n_subjects=2,
            n_training_items=10,
        )
    with pytest.raises(ValueError):
        SubjectResidualTable.from_rows(
            subject_ids=[0],
            training_item_rows=[15],  # out of range
            labels=[1.0],
            uncal_probs=[0.5],
            n_subjects=2,
            n_training_items=10,
        )


def test_residual_table_empty_input():
    table = SubjectResidualTable.from_rows(
        subject_ids=[],
        training_item_rows=[],
        labels=[],
        uncal_probs=[],
        n_subjects=4,
        n_training_items=8,
    )
    y, p, m = table.lookup(0, np.array([0, 1], dtype=np.int64))
    assert (m == 0.0).all()


# ---------------------------------------------------------------------------
# NNCalibrator state serialization
# ---------------------------------------------------------------------------


def test_calibrator_state_roundtrip():
    state = NNCalibratorState(
        alpha=0.42,
        temperature=2.0,
        k=8,
        similarity="cosine",
        min_weight_sum=0.01,
        similarity_floor=0.05,
        apply_in_logit_space=True,
        fit_method="alpha_grid_val_nll",
        fit_n_val=999,
    )
    d = state.to_dict()
    state2 = NNCalibratorState.from_dict(d)
    assert state2 == state


# ---------------------------------------------------------------------------
# NNCalibrator fit + apply on synthetic biased data
# ---------------------------------------------------------------------------


def _synthetic_biased_population(seed=0, N_subj=10, N_train=200):
    """Each subject has a per-subject logit bias; labels are sampled
    accordingly. The kNN calibrator should be able to recover most of
    the bias from a *random* set of K neighbors because each subject's
    rows are drawn from the same biased distribution."""
    rng = np.random.default_rng(seed)
    bias = rng.uniform(-0.6, 0.6, size=N_subj).astype(np.float32)
    s_ids = np.repeat(np.arange(N_subj), N_train)
    t_rows = np.tile(np.arange(N_train), N_subj)
    p_uncal = rng.uniform(0.05, 0.95, size=s_ids.size).astype(np.float32)
    logit = np.log(p_uncal / (1 - p_uncal)) + bias[s_ids]
    p_true = 1.0 / (1.0 + np.exp(-logit))
    labels = (p_true > rng.uniform(size=s_ids.size)).astype(np.float32)
    return rng, bias, s_ids, t_rows, p_uncal, labels, N_subj, N_train


def _bce(y, p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def test_calibrator_alpha_zero_is_identity():
    rng = np.random.default_rng(0)
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 1, 0, 1],
        training_item_rows=[0, 0, 1, 1],
        labels=[1.0, 0.0, 0.5, 1.0],
        uncal_probs=[0.5, 0.5, 0.5, 0.5],
        n_subjects=2,
        n_training_items=4,
    )
    cal = NNCalibrator(NNCalibratorState(alpha=0.0))
    p = rng.uniform(0.1, 0.9, size=8).astype(np.float32)
    out = cal.apply(
        residual_table=table,
        subject_ids=np.array([0, 1] * 4, dtype=np.int64),
        neighbor_rows=np.tile(np.array([[0, 1]]), (8, 1)),
        neighbor_sims=np.ones((8, 2), dtype=np.float32),
        p_uncal=p,
    )
    np.testing.assert_array_equal(out, p)


def test_calibrator_recovers_per_subject_bias():
    rng, bias, s_ids, t_rows, p_uncal, labels, N_subj, N_train = (
        _synthetic_biased_population(seed=0)
    )
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids, training_item_rows=t_rows,
        labels=labels, uncal_probs=p_uncal,
        n_subjects=N_subj, n_training_items=N_train,
    )

    K = 16
    N_val = 1000
    val_s = rng.integers(0, N_subj, size=N_val)
    val_p = rng.uniform(0.05, 0.95, size=N_val).astype(np.float32)
    val_logit = np.log(val_p / (1 - val_p)) + bias[val_s]
    val_y = (1.0 / (1 + np.exp(-val_logit)) > rng.uniform(size=N_val)).astype(
        np.float32
    )
    val_nbrs = rng.integers(0, N_train, size=(N_val, K)).astype(np.int64)
    val_sims = rng.uniform(0.5, 1.0, size=(N_val, K)).astype(np.float32)

    cal = NNCalibrator.fit_alpha_on_val(
        residual_table=table, val_subject_ids=val_s,
        val_neighbor_rows=val_nbrs, val_neighbor_sims=val_sims,
        val_uncal_probs=val_p, val_labels=val_y, k=K,
    )
    assert cal.state.alpha > 0
    # The fitter now sweeps (alpha, shrinkage_tau) jointly, so the
    # method tag reflects the 2-D grid. The legacy "alpha_grid_val_nll"
    # tag is only emitted when shrinkage_taus=[0.0] is explicitly
    # passed, which matches the legacy behavior bit-exactly.
    assert cal.state.fit_method == "alpha_tau_grid_val_nll"

    # Test holdout: calibration should reduce log-loss.
    N_test = 1500
    test_s = rng.integers(0, N_subj, size=N_test)
    test_p = rng.uniform(0.05, 0.95, size=N_test).astype(np.float32)
    test_logit = np.log(test_p / (1 - test_p)) + bias[test_s]
    test_y = (1.0 / (1 + np.exp(-test_logit)) > rng.uniform(size=N_test)).astype(
        np.float32
    )
    test_nbrs = rng.integers(0, N_train, size=(N_test, K)).astype(np.int64)
    test_sims = rng.uniform(0.5, 1.0, size=(N_test, K)).astype(np.float32)
    out = cal.apply(
        residual_table=table, subject_ids=test_s,
        neighbor_rows=test_nbrs, neighbor_sims=test_sims, p_uncal=test_p,
    )
    assert _bce(test_y, out) < _bce(test_y, test_p)


def test_calibrator_empty_table_is_identity():
    rng, *_ = _synthetic_biased_population(seed=0)
    empty = SubjectResidualTable.from_rows(
        subject_ids=[], training_item_rows=[], labels=[], uncal_probs=[],
        n_subjects=4, n_training_items=10,
    )
    cal = NNCalibrator(NNCalibratorState(alpha=0.5))
    p = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    out = cal.apply(
        residual_table=empty,
        subject_ids=np.array([0, 1, 2], dtype=np.int64),
        neighbor_rows=np.zeros((3, 4), dtype=np.int64),
        neighbor_sims=np.ones((3, 4), dtype=np.float32),
        p_uncal=p,
    )
    np.testing.assert_allclose(out, p, rtol=1e-5)


def test_calibrator_out_of_range_subject_is_identity():
    rng, _, s_ids, t_rows, p_uncal, labels, N_subj, N_train = (
        _synthetic_biased_population(seed=0)
    )
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids, training_item_rows=t_rows,
        labels=labels, uncal_probs=p_uncal,
        n_subjects=N_subj, n_training_items=N_train,
    )
    cal = NNCalibrator(NNCalibratorState(alpha=0.7, k=4))
    out = cal.apply(
        residual_table=table,
        subject_ids=np.array([N_subj], dtype=np.int64),  # out of range
        neighbor_rows=np.array([[0, 1, 2, 3]], dtype=np.int64),
        neighbor_sims=np.ones((1, 4), dtype=np.float32),
        p_uncal=np.array([0.4], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [0.4], rtol=1e-5)


def test_calibrator_logit_space_outputs_bounded():
    rng, _, s_ids, t_rows, p_uncal, labels, N_subj, N_train = (
        _synthetic_biased_population(seed=2)
    )
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids, training_item_rows=t_rows,
        labels=labels, uncal_probs=p_uncal,
        n_subjects=N_subj, n_training_items=N_train,
    )
    K = 8
    N_val = 200
    val_s = rng.integers(0, N_subj, size=N_val)
    val_p = rng.uniform(0.05, 0.95, size=N_val).astype(np.float32)
    val_y = rng.integers(0, 2, size=N_val).astype(np.float32)
    val_nbrs = rng.integers(0, N_train, size=(N_val, K)).astype(np.int64)
    val_sims = rng.uniform(0.5, 1.0, size=(N_val, K)).astype(np.float32)
    cal = NNCalibrator.fit_alpha_on_val(
        residual_table=table, val_subject_ids=val_s,
        val_neighbor_rows=val_nbrs, val_neighbor_sims=val_sims,
        val_uncal_probs=val_p, val_labels=val_y, k=K,
        apply_in_logit_space=True,
    )
    out = cal.apply(
        residual_table=table, subject_ids=val_s,
        neighbor_rows=val_nbrs, neighbor_sims=val_sims, p_uncal=val_p,
    )
    assert np.isfinite(out).all()
    assert ((out > 0) & (out < 1)).all()


def test_calibrator_min_weight_sum_blocks_low_coverage():
    """When every neighbor's similarity is below the floor, no shift fires."""
    rng = np.random.default_rng(0)
    table = SubjectResidualTable.from_rows(
        subject_ids=[0, 0], training_item_rows=[0, 1],
        labels=[1.0, 0.0], uncal_probs=[0.3, 0.7],
        n_subjects=1, n_training_items=4,
    )
    cal = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=2.0,  # impossibly high floor
        min_weight_sum=1e-3,
    ))
    out = cal.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=np.array([[0, 1]], dtype=np.int64),
        neighbor_sims=np.array([[0.9, 0.8]], dtype=np.float32),
        p_uncal=np.array([0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [0.5], rtol=1e-5)


# ---------------------------------------------------------------------------
# Phase 4 RED-TEAM: continuous shrinkage
# ---------------------------------------------------------------------------


def _two_subject_residual_table_with_known_residual():
    """Build a 1-subject 4-item table where:
      neighbor 0 has y=1.0, p=0.4 (residual = +0.6)
      neighbor 1 has y=1.0, p=0.5 (residual = +0.5)
      neighbor 2 has y=1.0, p=0.6 (residual = +0.4)
      neighbor 3 has y=1.0, p=0.7 (residual = +0.3)
    Mean residual = +0.45. Uniform-similarity weights -> any
    neighbor combo gives this mean. Useful for checking the
    shrinkage multiplier is applied correctly.
    """
    return SubjectResidualTable.from_rows(
        subject_ids=[0, 0, 0, 0],
        training_item_rows=[0, 1, 2, 3],
        labels=[1.0, 1.0, 1.0, 1.0],
        uncal_probs=[0.4, 0.5, 0.6, 0.7],
        n_subjects=1,
        n_training_items=4,
    )


def test_shrinkage_tau_zero_matches_legacy_correction():
    """tau=0 must reproduce the legacy alpha * weighted_residual exactly."""
    table = _two_subject_residual_table_with_known_residual()
    cal_legacy = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=0.0, min_weight_sum=1e-9,
        shrinkage_tau=0.0,
    ))
    sims = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    nbrs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    out = cal_legacy.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=nbrs,
        neighbor_sims=sims,
        p_uncal=np.array([0.5], dtype=np.float32),
    )
    # weighted_residual = (1*0.6 + 1*0.5 + 1*0.4 + 1*0.3) / 4 = 0.45
    # p_cal = 0.5 + 0.5 * 0.45 = 0.725
    np.testing.assert_allclose(out, [0.725], atol=1e-5)


def test_shrinkage_tau_dampens_strong_correction_when_w_sum_small():
    """When effective weight sum is small relative to tau, the
    correction is heavily attenuated (shrinks toward p_uncal)."""
    table = _two_subject_residual_table_with_known_residual()
    # Use very small similarities so w_sum ~ 0.4 (4 * 0.1).
    sims = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    nbrs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    p_uncal = np.array([0.5], dtype=np.float32)

    # tau=0 (no shrinkage): correction = alpha * 0.45 = 0.225
    cal_no_shrink = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=0.0, min_weight_sum=1e-9, shrinkage_tau=0.0,
    ))
    out_no = cal_no_shrink.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=nbrs, neighbor_sims=sims, p_uncal=p_uncal,
    )

    # tau=4 (heavy shrinkage): w_sum=0.4, shrink=0.4/(0.4+4)=0.0909
    # correction = 0.5 * 0.0909 * 0.45 = 0.0205
    cal_heavy = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=0.0, min_weight_sum=1e-9, shrinkage_tau=4.0,
    ))
    out_heavy = cal_heavy.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=nbrs, neighbor_sims=sims, p_uncal=p_uncal,
    )

    # Both should be > p_uncal but the heavy-tau output should be MUCH closer.
    assert float(out_no[0]) > 0.6
    assert float(out_heavy[0]) < float(out_no[0])
    # Quantitative check on heavy-tau: 0.5 + 0.0205 ~ 0.520
    np.testing.assert_allclose(out_heavy, [0.5 + 0.5 * (0.4 / 4.4) * 0.45], atol=1e-4)


def test_shrinkage_tau_preserves_strong_correction_when_w_sum_large():
    """When effective weight sum >> tau, shrinkage multiplier ~ 1
    and the correction is essentially unchanged."""
    table = _two_subject_residual_table_with_known_residual()
    # w_sum = 4 * 1 = 4
    sims = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    nbrs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    p_uncal = np.array([0.5], dtype=np.float32)

    cal_no_shrink = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=0.0, min_weight_sum=1e-9, shrinkage_tau=0.0,
    ))
    cal_mild_shrink = NNCalibrator(NNCalibratorState(
        alpha=0.5, similarity_floor=0.0, min_weight_sum=1e-9, shrinkage_tau=0.1,
    ))
    out_no = cal_no_shrink.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=nbrs, neighbor_sims=sims, p_uncal=p_uncal,
    )
    out_mild = cal_mild_shrink.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=nbrs, neighbor_sims=sims, p_uncal=p_uncal,
    )
    # tau=0.1, w_sum=4 -> shrink=4/4.1=0.976 -> correction barely changed
    np.testing.assert_allclose(out_no, [0.725], atol=1e-5)
    np.testing.assert_allclose(out_mild, [0.5 + 0.5 * (4.0 / 4.1) * 0.45], atol=1e-4)


def test_fit_alpha_on_val_includes_shrinkage_tau():
    """The new fitter must search over shrinkage_tau and report which
    value won."""
    rng = np.random.default_rng(0)
    # Manufacture data where shrinkage helps: weak neighborhoods are
    # noisier than strong ones.
    N_subj = 4
    N_train = 200
    s_ids = rng.integers(0, N_subj, size=2000)
    t_rows = rng.integers(0, N_train, size=2000)
    bias = rng.normal(0, 0.5, size=N_subj)
    p_uncal = rng.uniform(0.2, 0.8, size=2000)
    logit_p = np.log(p_uncal / (1 - p_uncal)) + bias[s_ids]
    p_true = 1 / (1 + np.exp(-logit_p))
    labels = (rng.uniform(size=2000) < p_true).astype(np.float32)
    table = SubjectResidualTable.from_rows(
        subject_ids=s_ids, training_item_rows=t_rows,
        labels=labels, uncal_probs=p_uncal.astype(np.float32),
        n_subjects=N_subj, n_training_items=N_train,
    )
    K = 8
    N_val = 400
    val_s = rng.integers(0, N_subj, size=N_val)
    val_p = rng.uniform(0.1, 0.9, size=N_val).astype(np.float32)
    val_logit = np.log(val_p / (1 - val_p)) + bias[val_s]
    val_y = (1 / (1 + np.exp(-val_logit)) > rng.uniform(size=N_val)).astype(np.float32)
    val_nbrs = rng.integers(0, N_train, size=(N_val, K)).astype(np.int64)
    val_sims = rng.uniform(0.3, 1.0, size=(N_val, K)).astype(np.float32)

    cal = NNCalibrator.fit_alpha_on_val(
        residual_table=table, val_subject_ids=val_s,
        val_neighbor_rows=val_nbrs, val_neighbor_sims=val_sims,
        val_uncal_probs=val_p, val_labels=val_y, k=K,
        shrinkage_taus=[0.0, 0.5, 1.0, 2.0, 5.0],
    )
    assert cal.state.fit_method == "alpha_tau_grid_val_nll"
    assert cal.state.shrinkage_tau in [0.0, 0.5, 1.0, 2.0, 5.0]
    # And the calibrated output is finite + bounded.
    out = cal.apply(
        residual_table=table, subject_ids=val_s,
        neighbor_rows=val_nbrs, neighbor_sims=val_sims, p_uncal=val_p,
    )
    assert np.isfinite(out).all()
    assert ((out > 0) & (out < 1)).all()


def test_calibrator_state_serialization_roundtrips_shrinkage():
    state = NNCalibratorState(alpha=0.4, shrinkage_tau=2.5)
    d = state.to_dict()
    s2 = NNCalibratorState.from_dict(d)
    assert s2.shrinkage_tau == 2.5
    assert s2.alpha == 0.4

    # And round-trip via JSON serialization.
    j = json.dumps(d)
    s3 = NNCalibratorState.from_dict(json.loads(j))
    assert s3.shrinkage_tau == 2.5


def test_calibrator_with_empty_labeled_is_noop():
    """RED-TEAM (Calibrator c): with an empty labeled list (== empty
    residual table), the calibrator must be a no-op, never a crash."""
    table = SubjectResidualTable.from_rows(
        subject_ids=[],
        training_item_rows=[],
        labels=[],
        uncal_probs=[],
        n_subjects=1,
        n_training_items=4,
    )
    cal = NNCalibrator(NNCalibratorState(
        alpha=0.5, shrinkage_tau=2.0, similarity_floor=0.0,
    ))
    out = cal.apply(
        residual_table=table,
        subject_ids=np.array([0], dtype=np.int64),
        neighbor_rows=np.array([[0, 1, 2, 3]], dtype=np.int64),
        neighbor_sims=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        p_uncal=np.array([0.42], dtype=np.float32),
    )
    np.testing.assert_allclose(out, [0.42], rtol=1e-5)
    assert math.isfinite(float(out[0]))


def test_calibrator_runtime_is_faiss_free():
    """RED-TEAM (Calibrator a): the calibrator's runtime path
    must not import faiss."""
    import re
    src_text = open("src/nn_calibration.py", encoding="utf-8").read()
    for pattern in (r"^\s*import\s+faiss", r"^\s*from\s+faiss"):
        matches = list(re.finditer(pattern, src_text, flags=re.MULTILINE))
        assert len(matches) == 0, (
            f"src/nn_calibration.py must not import faiss; matched {pattern}"
        )
