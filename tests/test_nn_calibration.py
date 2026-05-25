"""Unit tests for the Netflix-Prize-style NN calibrator."""

from __future__ import annotations

import json
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
    assert cal.state.fit_method == "alpha_grid_val_nll"

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
