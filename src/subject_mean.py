"""Out-of-fold subject-mean anchor for Member 2.

For each subject *s*, the "subject mean" is the empirical pass-rate of
that subject across its training observations, optionally shrunk
toward the global mean (Bayesian smoothing) for low-count subjects.

Member 2's new training target (Task 3 of the diversification plan)
is the LOGIT-RESIDUAL against this anchor:

    target_row = logit(label_row) - logit(subject_mean_row)

which is equivalent to fitting LightGBM in ``binary`` objective with
``init_score = logit(subject_mean_row)`` per row.

For OOF training the anchor MUST be computed OUT OF FOLD: the
subject_mean used for a row in fold *f*'s OOF set must aggregate
labels only from the OTHER K-1 folds' rows. Otherwise the anchor
itself encodes the row's label and the tree learns a trivial
identity mapping.

The runtime anchor (for val / test inference) is the full-train
subject_mean (val rows are cold-start items but the SUBJECT is
seen, so the full-train mean is the appropriate anchor).

Both arrays carry the same Bayesian-shrinkage construction so the
inference distribution matches training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOG = logging.getLogger("subject_mean")

# Clip range for logit transforms inside the residual pipeline.
_PCLIP_LO = 1e-6
_PCLIP_HI = 1.0 - _PCLIP_LO


@dataclass(frozen=True)
class SubjectMeanTable:
    """Per-subject mean-of-label statistics with Bayesian shrinkage.

    Attributes
    ----------
    subject_mean
        Shape ``[n_subjects]``. ``subject_mean[s]`` is the smoothed
        per-row pass-rate estimate for subject *s*. For subjects with
        zero training observations this is the global label mean
        (cold-start subjects are out of scope here; the val set
        guarantees seen subjects).
    subject_obs_count
        Shape ``[n_subjects]``. Raw observation count per subject. A
        useful Member 2 feature in its own right ("trust the per-
        subject anchor more for high-count subjects").
    global_mean
        Scalar prior for the shrinkage. ``mean(labels)``.
    smoothing
        Bayesian prior strength: ``shrunk_mean = (sum_y + smoothing *
        global_mean) / (n + smoothing)``. Default 30 means a subject
        with 0 observations gets the global mean exactly; a subject
        with 30 observations is half-toward its empirical mean; a
        subject with 300 observations is 91% toward empirical.
    """

    subject_mean: np.ndarray
    subject_obs_count: np.ndarray
    global_mean: float
    smoothing: float


def fit_subject_mean_table(
    *,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    n_subjects: int,
    smoothing: float = 30.0,
) -> SubjectMeanTable:
    """Build a :class:`SubjectMeanTable` from labelled rows.

    ``subject_ids`` may contain ``-1`` sentinels (UNK / unknown
    subject); those rows are dropped from the aggregation but the
    output array still has ``n_subjects`` entries (so per-row lookup
    by id stays a simple table indexer).
    """
    subj = np.asarray(subject_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    if subj.shape != y.shape:
        raise ValueError(
            f"subject_ids and labels must have same shape, got "
            f"{subj.shape} vs {y.shape}"
        )
    mask = subj >= 0
    subj_m, y_m = subj[mask], y[mask]
    if subj_m.size == 0:
        global_mean = 0.5
    else:
        global_mean = float(np.mean(y_m))

    sum_y = np.bincount(subj_m, weights=y_m, minlength=int(n_subjects)).astype(np.float64)
    n = np.bincount(subj_m, minlength=int(n_subjects)).astype(np.float64)
    sm = float(smoothing)
    if sm < 0.0:
        raise ValueError(f"smoothing must be >= 0, got {sm}")

    shrunk = (sum_y + sm * global_mean) / (n + sm)
    shrunk = np.clip(shrunk, _PCLIP_LO, _PCLIP_HI).astype(np.float64)
    LOG.info(
        "fit_subject_mean_table: n_subjects=%d  n_rows_used=%d  "
        "global_mean=%.4f  smoothing=%.1f  n_subjects_with_zero_obs=%d",
        int(n_subjects), int(subj_m.size), global_mean, sm,
        int(np.sum(n == 0)),
    )
    return SubjectMeanTable(
        subject_mean=shrunk,
        subject_obs_count=n.astype(np.float64),
        global_mean=global_mean,
        smoothing=sm,
    )


def apply_subject_mean(
    table: SubjectMeanTable,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Look up per-row subject_mean from a fitted table.

    Unknown subject ids (``-1`` sentinel) get the global mean.
    Output is shape ``[N]`` in ``[_PCLIP_LO, _PCLIP_HI]``, ready to
    pass as an ``anchor_p`` to :func:`gbdt_compose_residual_*`.
    """
    subj = np.asarray(subject_ids, dtype=np.int64)
    out = np.full(subj.shape, float(table.global_mean), dtype=np.float64)
    in_range = (subj >= 0) & (subj < int(table.subject_mean.shape[0]))
    out[in_range] = table.subject_mean[subj[in_range]]
    return np.clip(out, _PCLIP_LO, _PCLIP_HI)


def apply_subject_obs_count(
    table: SubjectMeanTable,
    subject_ids: np.ndarray,
    *,
    log1p: bool = True,
) -> np.ndarray:
    """Look up per-row subject observation count, optionally log1p'd.

    Unknown subject ids return 0 (the "cold-start subject" signal).
    This is a useful Member 2 feature: rows whose subject has a small
    training footprint should trust the per-subject anchor less and
    let the tree do more lifting.
    """
    subj = np.asarray(subject_ids, dtype=np.int64)
    out = np.zeros(subj.shape, dtype=np.float64)
    in_range = (subj >= 0) & (subj < int(table.subject_obs_count.shape[0]))
    out[in_range] = table.subject_obs_count[subj[in_range]]
    if log1p:
        out = np.log1p(out)
    return out


# ---------------------------------------------------------------------------
# OOF assertion helper (Gate 3a)
# ---------------------------------------------------------------------------


def assert_oof_subject_mean(
    *,
    subject_mean_oof_for_fold: np.ndarray,
    fold_subject_ids: np.ndarray,
    fold_train_subject_mean_table: SubjectMeanTable,
) -> dict:
    """RED-TEAM GATE 3a: assert each row's subject_mean was computed from
    the FOLD'S TRAIN labels only, NOT from the full train labels.

    Checks: for every fold-OOF row, the supplied ``subject_mean_oof_for_fold``
    value must equal ``fold_train_subject_mean_table.subject_mean[subj_id]``
    (within float tolerance). If the supplied value matches a global
    subject_mean instead -- which would mean it includes this row's own
    label -- the gate fails.

    Returns
    -------
    dict with ``n_checked``, ``n_violations``, ``max_abs_delta`` keys.
    Raises AssertionError on any mismatch (within numerical tolerance).
    """
    sm = np.asarray(subject_mean_oof_for_fold, dtype=np.float64)
    expected = apply_subject_mean(fold_train_subject_mean_table, fold_subject_ids)
    delta = np.abs(sm - expected)
    n_violations = int((delta > 1e-9).sum())
    max_abs = float(delta.max()) if delta.size > 0 else 0.0
    if n_violations > 0:
        bad = int(np.argmax(delta))
        raise AssertionError(
            f"GATE 3a violation: {n_violations}/{sm.size} fold-OOF rows have "
            f"subject_mean values that don't match the fold-train table. "
            f"Worst delta={max_abs:.6f} at row {bad} (subj={int(fold_subject_ids[bad])}, "
            f"got={sm[bad]:.6f}, expected={expected[bad]:.6f}). "
            "Subject_mean anchor must be fit on fold-train labels only."
        )
    return {
        "n_checked": int(sm.size),
        "n_violations": int(n_violations),
        "max_abs_delta": float(max_abs),
    }


__all__ = [
    "SubjectMeanTable",
    "fit_subject_mean_table",
    "apply_subject_mean",
    "apply_subject_obs_count",
    "assert_oof_subject_mean",
]
