"""Tests for src/subject_mean.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.subject_mean import (
    SubjectMeanTable,
    apply_subject_mean,
    apply_subject_obs_count,
    assert_oof_subject_mean,
    fit_subject_mean_table,
)


def test_fit_basic_shapes():
    subject_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    t = fit_subject_mean_table(
        subject_ids=subject_ids, labels=labels, n_subjects=3, smoothing=0.0
    )
    assert t.subject_mean.shape == (3,)
    assert t.subject_obs_count.shape == (3,)
    # Subject means clipped into the open (0,1) interval so logit() is safe;
    # the clip epsilon is ~1e-6 so use atol=1e-5 rather than rtol=1e-12.
    np.testing.assert_allclose(t.subject_mean, [0.5, 1.0, 0.0], atol=1e-5)
    np.testing.assert_array_equal(t.subject_obs_count, [2, 2, 2])
    assert abs(t.global_mean - 0.5) < 1e-12


def test_fit_with_bayesian_shrinkage():
    """smoothing=30 with global_mean=0.5: a subject with 0 obs gets 0.5,
    a subject with N obs gets (sum_y + 30*0.5)/(N+30)."""
    subject_ids = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    labels = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)  # all 1s
    t = fit_subject_mean_table(
        subject_ids=subject_ids, labels=labels, n_subjects=3, smoothing=30.0
    )
    gm = t.global_mean
    assert abs(gm - 1.0) < 1e-12
    # subj 0: 4 obs at 1.0, shrunk to (4 + 30*1.0) / 34 = 1.0 (clipped to 1-1e-6)
    np.testing.assert_allclose(t.subject_mean[0], 1.0, atol=1e-5)
    # subj 2: 0 obs -> global mean (also clipped to 1-1e-6 here)
    np.testing.assert_allclose(t.subject_mean[2], gm, atol=1e-5)


def test_fit_handles_unk_subject_id():
    subject_ids = np.array([-1, 0, 0, -1, 1], dtype=np.int64)
    labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    t = fit_subject_mean_table(
        subject_ids=subject_ids, labels=labels, n_subjects=2, smoothing=0.0
    )
    # UNK rows dropped: subj 0 sees [1, 0] -> 0.5, subj 1 sees [1] -> 1.0 (clipped)
    # global mean is over the 3 non-UNK rows: (1+0+1)/3 = 2/3
    np.testing.assert_allclose(t.subject_mean[0], 0.5, atol=1e-5)
    np.testing.assert_allclose(t.subject_mean[1], 1.0, atol=1e-5)
    np.testing.assert_allclose(t.global_mean, 2.0 / 3.0, rtol=1e-12)


def test_fit_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        fit_subject_mean_table(
            subject_ids=np.zeros(5, dtype=np.int64),
            labels=np.zeros(7, dtype=np.float64),
            n_subjects=1,
        )


def test_fit_rejects_negative_smoothing():
    with pytest.raises(ValueError, match="smoothing"):
        fit_subject_mean_table(
            subject_ids=np.zeros(2, dtype=np.int64),
            labels=np.zeros(2, dtype=np.float64),
            n_subjects=1,
            smoothing=-1.0,
        )


def test_apply_subject_mean_basic():
    t = SubjectMeanTable(
        subject_mean=np.array([0.2, 0.5, 0.8]),
        subject_obs_count=np.array([10, 5, 100]),
        global_mean=0.4,
        smoothing=0.0,
    )
    ids = np.array([0, 2, 1, -1, 5, 0], dtype=np.int64)  # -1 = UNK, 5 = out-of-range
    out = apply_subject_mean(t, ids)
    np.testing.assert_allclose(out, [0.2, 0.8, 0.5, 0.4, 0.4, 0.2], rtol=1e-12)


def test_apply_subject_mean_clips_to_open_interval():
    """Edge case: subject_mean stored as 0.0 or 1.0 must come back clipped."""
    t = SubjectMeanTable(
        subject_mean=np.array([0.0, 1.0]),
        subject_obs_count=np.array([10, 10]),
        global_mean=0.5,
        smoothing=0.0,
    )
    out = apply_subject_mean(t, np.array([0, 1], dtype=np.int64))
    assert out[0] > 0.0 and out[0] < 1e-5
    assert out[1] < 1.0 and out[1] > 1.0 - 1e-5


def test_apply_subject_obs_count_log1p():
    t = SubjectMeanTable(
        subject_mean=np.array([0.5, 0.5]),
        subject_obs_count=np.array([0, 99]),
        global_mean=0.5,
        smoothing=0.0,
    )
    out = apply_subject_obs_count(t, np.array([0, 1, -1], dtype=np.int64))
    # log1p(0)=0, log1p(99)=log(100)=4.605..., UNK->0
    np.testing.assert_allclose(out, [0.0, np.log(100.0), 0.0], rtol=1e-12)


def test_apply_subject_obs_count_raw():
    t = SubjectMeanTable(
        subject_mean=np.array([0.5, 0.5]),
        subject_obs_count=np.array([7, 22]),
        global_mean=0.5,
        smoothing=0.0,
    )
    out = apply_subject_obs_count(t, np.array([0, 1], dtype=np.int64), log1p=False)
    np.testing.assert_array_equal(out, [7, 22])


# ---------------------------------------------------------------------------
# Gate 3a OOF anchor probe
# ---------------------------------------------------------------------------


def test_gate3a_passes_when_oof_anchor_matches_fold_table():
    """Synthetic case: fold f's subject_mean for an OOF row should equal
    the fold-train table's subject_mean for that row's subject."""
    # Fold-train: 3 subjects, varying obs counts
    fold_train_subj = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    fold_train_lab = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    ftt = fit_subject_mean_table(
        subject_ids=fold_train_subj, labels=fold_train_lab,
        n_subjects=3, smoothing=0.0,
    )
    # Fold-OOF: 4 rows, mix of subjects
    fold_oof_subj = np.array([0, 1, 2, 1], dtype=np.int64)
    # Anchor: what we'd actually use (looked up against fold-train table)
    sm_oof = apply_subject_mean(ftt, fold_oof_subj)
    # Gate 3a should pass since we built sm_oof from ftt.
    result = assert_oof_subject_mean(
        subject_mean_oof_for_fold=sm_oof,
        fold_subject_ids=fold_oof_subj,
        fold_train_subject_mean_table=ftt,
    )
    assert result["n_violations"] == 0


def test_gate3a_catches_global_anchor_leak():
    """If someone mistakenly used the GLOBAL subject_mean (which includes
    fold-OOF labels) as the anchor, Gate 3a should fail."""
    # Subject 0 has different distribution in fold-train vs full-train
    fold_train_subj = np.array([0, 0, 0], dtype=np.int64)
    fold_train_lab = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # 100% pass
    ftt = fit_subject_mean_table(
        subject_ids=fold_train_subj, labels=fold_train_lab,
        n_subjects=1, smoothing=0.0,
    )
    full_subj = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    full_lab = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 50% pass
    full_t = fit_subject_mean_table(
        subject_ids=full_subj, labels=full_lab, n_subjects=1, smoothing=0.0,
    )
    fold_oof_subj = np.array([0, 0, 0], dtype=np.int64)
    # WRONG: using the global table for fold's OOF anchor
    sm_oof_bad = apply_subject_mean(full_t, fold_oof_subj)
    with pytest.raises(AssertionError, match="GATE 3a"):
        assert_oof_subject_mean(
            subject_mean_oof_for_fold=sm_oof_bad,
            fold_subject_ids=fold_oof_subj,
            fold_train_subject_mean_table=ftt,
        )
