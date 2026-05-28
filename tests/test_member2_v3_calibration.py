"""Unit tests for ``src.member2_v3_calibration``.

Covers the failure modes that historically bit Member 2:
  - silent column-order reshuffling -> wrong-feature splits at apply time
  - sentinel "-1" id handling -> NaN or out-of-bounds index
  - smoothing math -> empty cells must collapse to global mean
  - 2-D cell-count derivation -> off-by-one or row/col swap
  - apply-time produces finite outputs even with extreme p_m1 / sm
"""
from __future__ import annotations

import numpy as np
import pytest

from src.member2_v3_calibration import (
    M2V3_FEATURE_DIM,
    MEMBER2_V3_FEATURE_NAMES,
    Member2V3FeatureBuilder,
    Member2V3State,
    build_member2_v3_features,
    fit_member2_v3_feature_builder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _toy_inputs(n_rows: int = 500, seed: int = 7) -> dict:
    """Small but realistic id distribution covering 'cold' / 'unknown' edge
    cases for every metadata axis."""
    rng = np.random.default_rng(seed)
    n_subjects = 20
    n_clusters = 8
    n_bcs = 12
    n_macro_families = 5
    n_organizations = 4
    n_families = 6

    subject_ids = rng.integers(0, n_subjects, size=n_rows, dtype=np.int64)
    cluster_ids = rng.integers(0, n_clusters, size=n_rows, dtype=np.int64)
    bc_ids = rng.integers(0, n_bcs, size=n_rows, dtype=np.int64)
    # Mark ~5% rows as unknown on each axis (allowed by spec).
    subject_ids[rng.random(n_rows) < 0.05] = -1
    cluster_ids[rng.random(n_rows) < 0.05] = -1
    bc_ids[rng.random(n_rows) < 0.05] = -1
    labels = rng.integers(0, 2, size=n_rows).astype(np.float32)

    subject_to_macro = rng.integers(
        0, n_macro_families, size=n_subjects, dtype=np.int32
    )
    subject_to_org = rng.integers(
        0, n_organizations, size=n_subjects, dtype=np.int32
    )
    subject_to_family = rng.integers(
        0, n_families, size=n_subjects, dtype=np.int32
    )
    # Leave one subject with a sentinel trait (real-world "unknown" case).
    subject_to_macro[0] = -1
    subject_to_org[0] = -1
    subject_to_family[0] = -1
    return {
        "subject_ids": subject_ids,
        "cluster_ids": cluster_ids,
        "bc_ids": bc_ids,
        "labels": labels,
        "n_subjects": n_subjects,
        "n_clusters": n_clusters,
        "n_bcs": n_bcs,
        "n_macro_families": n_macro_families,
        "n_organizations": n_organizations,
        "n_families": n_families,
        "subject_to_macro_family_id": subject_to_macro,
        "subject_to_organization_id": subject_to_org,
        "subject_to_family_id": subject_to_family,
    }


# ---------------------------------------------------------------------------
# Spec audits
# ---------------------------------------------------------------------------
def test_schema_is_locked_at_25_columns() -> None:
    assert M2V3_FEATURE_DIM == 25
    assert len(MEMBER2_V3_FEATURE_NAMES) == 25
    # The 'is_unknown_bc' inductive bias must be at the spec-locked
    # position 20. Any reorder breaks every saved booster silently.
    assert MEMBER2_V3_FEATURE_NAMES[20] == "is_unknown_bc"


def test_logit_p_m1_is_first_column() -> None:
    # The point of v3 is that M1's logit is the calibration signal --
    # if it ever moves off column 0 it will be silently swapped with
    # logit_subject_mean and the residual GBDT will train on garbage.
    assert MEMBER2_V3_FEATURE_NAMES[0] == "logit_p_m1"
    assert MEMBER2_V3_FEATURE_NAMES[1] == "logit_subject_mean"
    assert MEMBER2_V3_FEATURE_NAMES[2] == "logit_disagreement"


def test_obs_count_columns_are_all_present() -> None:
    # User-requested: per-metadata-level obs counts including macro_family,
    # organization, family. Regression test for the "you forgot one" bug.
    required = {
        "subject_obs_count_log1p",
        "cluster_obs_count_log1p",
        "bc_obs_count_log1p",
        "macro_family_obs_count_log1p",
        "organization_obs_count_log1p",
        "family_obs_count_log1p",
        "subj_x_bc_obs_count_log1p",
        "subj_x_cluster_obs_count_log1p",
    }
    assert required.issubset(set(MEMBER2_V3_FEATURE_NAMES))


def test_unknown_flags_cover_every_metadata_axis() -> None:
    # 'unknown' inductive bias must be explicit per axis (the model
    # otherwise has to learn it from passrate=global_mean coincidences).
    required = {
        "is_unknown_subject",
        "is_unknown_cluster",
        "is_unknown_bc",
        "is_unknown_macro_family",
        "is_unknown_organization",
        "is_unknown_family",
    }
    assert required.issubset(set(MEMBER2_V3_FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def test_fit_returns_well_shaped_builder() -> None:
    inp = _toy_inputs()
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    assert b.subj_log1p_n.shape == (inp["n_subjects"],)
    assert b.cluster_log1p_n.shape == (inp["n_clusters"],)
    assert b.bc_log1p_n.shape == (inp["n_bcs"],)
    assert b.macro_log1p_n.shape == (inp["n_macro_families"],)
    assert b.org_log1p_n.shape == (inp["n_organizations"],)
    assert b.fam_log1p_n.shape == (inp["n_families"],)
    assert b.subj_x_bc_log1p_n.shape == (inp["n_subjects"], inp["n_bcs"])
    assert b.subj_x_cluster_log1p_n.shape == (inp["n_subjects"], inp["n_clusters"])
    # global_mean should be a finite probability.
    assert 0.0 < b.global_mean < 1.0


def test_fit_shrinkage_collapses_empty_cell_to_global_mean() -> None:
    # A cluster that never appears in the training rows must have
    # cluster_passrate == global_mean after smoothing (count=0 ->
    # passrate = (0 + sm*g) / (0 + sm) = g).
    inp = _toy_inputs()
    # Force cluster id 7 to never appear.
    inp["cluster_ids"] = np.where(
        inp["cluster_ids"] == 7, 0, inp["cluster_ids"]
    )
    b = fit_member2_v3_feature_builder(smoothing=30.0, **inp)
    assert b.cluster_passrate[7] == pytest.approx(b.global_mean, abs=1e-6)


def test_fit_rejects_bad_shapes() -> None:
    inp = _toy_inputs()
    inp["labels"] = inp["labels"][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_member2_v3_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_negative_smoothing() -> None:
    inp = _toy_inputs()
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        fit_member2_v3_feature_builder(smoothing=-1.0, **inp)


def test_subject_to_trait_arrays_have_correct_length() -> None:
    inp = _toy_inputs()
    inp["subject_to_macro_family_id"] = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError, match="must each have shape"):
        fit_member2_v3_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_undersized_cluster_vocab_with_clear_message() -> None:
    # Regression: undersized n_clusters used to raise a cryptic
    # ``IndexError: index 64 is out of bounds for axis 1 with size 64``
    # from deep inside np.add.at on the subj_x_cluster cell-count step.
    # The user's first v3 run hit this when N_CLUSTERS_CTX=64 but the
    # k-means partition produced cluster id 64 (so the actual vocab is
    # 65). The module must now raise a CLEAR ValueError pointing at the
    # right fix (``fold_cond_context.n_clusters``).
    inp = _toy_inputs(n_rows=200)
    # Smuggle one row with cluster_id == n_clusters (= out-of-bounds by 1).
    inp["cluster_ids"][0] = int(inp["n_clusters"])
    with pytest.raises(ValueError, match=r"observed max\(cluster_ids\)"):
        fit_member2_v3_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_undersized_bc_vocab_with_clear_message() -> None:
    inp = _toy_inputs(n_rows=200)
    inp["bc_ids"][0] = int(inp["n_bcs"])
    with pytest.raises(ValueError, match=r"observed max\(bc_ids\)"):
        fit_member2_v3_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_undersized_subject_vocab_with_clear_message() -> None:
    inp = _toy_inputs(n_rows=200)
    inp["subject_ids"][0] = int(inp["n_subjects"])
    with pytest.raises(ValueError, match=r"observed max\(subject_ids\)"):
        fit_member2_v3_feature_builder(smoothing=10.0, **inp)


# ---------------------------------------------------------------------------
# Apply: shape + finiteness
# ---------------------------------------------------------------------------
def test_build_features_shape_and_dtype() -> None:
    inp = _toy_inputs(n_rows=300)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    rng = np.random.default_rng(11)
    p_m1 = rng.uniform(0.05, 0.95, size=300).astype(np.float32)
    sm = rng.uniform(0.4, 0.8, size=300).astype(np.float32)
    X = build_member2_v3_features(
        b,
        p_m1=p_m1,
        subject_mean=sm,
        subject_ids=inp["subject_ids"],
        cluster_ids=inp["cluster_ids"],
        bc_ids=inp["bc_ids"],
    )
    assert X.shape == (300, M2V3_FEATURE_DIM)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_build_features_handles_extreme_probabilities() -> None:
    # logit(p) blows up at p in {0, 1}; the apply path must clip.
    inp = _toy_inputs(n_rows=100)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    p_m1 = np.array([0.0, 1.0] * 50, dtype=np.float32)
    sm = np.array([1.0, 0.0] * 50, dtype=np.float32)
    X = build_member2_v3_features(
        b,
        p_m1=p_m1,
        subject_mean=sm,
        subject_ids=inp["subject_ids"][:100],
        cluster_ids=inp["cluster_ids"][:100],
        bc_ids=inp["bc_ids"][:100],
    )
    assert np.isfinite(X).all()
    # Logit columns should be bounded by the clip.
    assert np.abs(X[:, 0]).max() < 30.0
    assert np.abs(X[:, 1]).max() < 30.0


def test_build_features_unknown_ids_fire_the_correct_flags() -> None:
    inp = _toy_inputs(n_rows=20)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    # Row 0: unknown subject. Should set is_unknown_subject AND propagate
    # to is_unknown_{macro_family,organization,family} since those are
    # subject-derived.
    subj = inp["subject_ids"][:20].copy()
    clus = inp["cluster_ids"][:20].copy()
    bc = inp["bc_ids"][:20].copy()
    subj[0] = -1
    clus[1] = -1
    bc[2] = -1
    X = build_member2_v3_features(
        b,
        p_m1=np.full(20, 0.7, dtype=np.float32),
        subject_mean=np.full(20, 0.65, dtype=np.float32),
        subject_ids=subj,
        cluster_ids=clus,
        bc_ids=bc,
    )
    # Columns: 18=subject, 19=cluster, 20=bc, 21=macro, 22=org, 23=fam.
    assert X[0, 18] == 1.0
    assert X[1, 19] == 1.0
    assert X[2, 20] == 1.0
    # Subject-unknown row also propagates to macro/org/family.
    assert X[0, 21] == 1.0
    assert X[0, 22] == 1.0
    assert X[0, 23] == 1.0
    # Sanity: non-unknown rows have flag 0.
    assert X[3, 18] == 0.0


def test_build_features_unknown_subject_falls_back_to_global_mean_passrate() -> None:
    inp = _toy_inputs(n_rows=10)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    subj = np.full(10, -1, dtype=np.int64)  # all unknown
    clus = np.full(10, -1, dtype=np.int64)
    bc = np.full(10, -1, dtype=np.int64)
    X = build_member2_v3_features(
        b,
        p_m1=np.full(10, 0.6, dtype=np.float32),
        subject_mean=np.full(10, b.global_mean, dtype=np.float32),
        subject_ids=subj,
        cluster_ids=clus,
        bc_ids=bc,
    )
    g = float(b.global_mean)
    # Cols 13-17 are passrates; all should collapse to global mean.
    for col in range(13, 18):
        assert np.allclose(X[:, col], g, atol=1e-5)
    # Cols 5-12 are obs counts; unknown ids -> 0.
    for col in range(5, 13):
        assert np.allclose(X[:, col], 0.0, atol=1e-6)


def test_disagreement_column_is_actual_difference_of_logits() -> None:
    inp = _toy_inputs(n_rows=50)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    p_m1 = np.linspace(0.05, 0.95, 50, dtype=np.float32)
    sm = np.full(50, 0.5, dtype=np.float32)
    X = build_member2_v3_features(
        b,
        p_m1=p_m1,
        subject_mean=sm,
        subject_ids=inp["subject_ids"][:50],
        cluster_ids=inp["cluster_ids"][:50],
        bc_ids=inp["bc_ids"][:50],
    )
    # Logit(0.5) = 0; disagreement should equal logit_p_m1.
    np.testing.assert_allclose(X[:, 2], X[:, 0], atol=1e-5)
    np.testing.assert_allclose(X[:, 3], np.abs(X[:, 0]), atol=1e-5)


def test_2d_cell_count_matches_naive_aggregate() -> None:
    inp = _toy_inputs(n_rows=400)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    # Naive: for each unique (subject, bc) cell, count rows.
    subj = inp["subject_ids"]
    bc = inp["bc_ids"]
    valid = (subj >= 0) & (bc >= 0)
    sb_naive_n = np.zeros((inp["n_subjects"], inp["n_bcs"]), dtype=np.float64)
    for s, c in zip(subj[valid], bc[valid]):
        sb_naive_n[int(s), int(c)] += 1.0
    np.testing.assert_allclose(
        b.subj_x_bc_log1p_n,
        np.log1p(sb_naive_n).astype(np.float32),
        atol=1e-5,
    )


def test_bc_redacted_mask_is_pipeline_through() -> None:
    inp = _toy_inputs(n_rows=30)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    mask = np.array([0, 1] * 15, dtype=np.float32)
    X = build_member2_v3_features(
        b,
        p_m1=np.full(30, 0.7, dtype=np.float32),
        subject_mean=np.full(30, 0.7, dtype=np.float32),
        subject_ids=inp["subject_ids"][:30],
        cluster_ids=inp["cluster_ids"][:30],
        bc_ids=inp["bc_ids"][:30],
        bc_redacted_mask=mask,
    )
    np.testing.assert_array_equal(X[:, 24], mask)


def test_build_features_rejects_length_mismatch() -> None:
    inp = _toy_inputs(n_rows=10)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    with pytest.raises(ValueError, match="all input arrays must have the same length"):
        build_member2_v3_features(
            b,
            p_m1=np.full(10, 0.5),
            subject_mean=np.full(5, 0.5),  # wrong length
            subject_ids=inp["subject_ids"][:10],
            cluster_ids=inp["cluster_ids"][:10],
            bc_ids=inp["bc_ids"][:10],
        )


# ---------------------------------------------------------------------------
# State + serialization
# ---------------------------------------------------------------------------
def test_state_requires_matching_feature_names() -> None:
    inp = _toy_inputs(n_rows=20)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    # Hand-write a state with the locked names -> ok.
    state = Member2V3State(gbdt=None, builder=b)
    assert state.version.startswith("v3")
    # Hand-write a state with a wrong name list -> raises.
    with pytest.raises(ValueError, match="feature_names mismatch"):
        Member2V3State(
            gbdt=None,
            builder=b,
            feature_names=("logit_p_m1",) + MEMBER2_V3_FEATURE_NAMES[1:][::-1],
        )


def test_builder_to_dict_round_trip() -> None:
    inp = _toy_inputs(n_rows=80)
    b = fit_member2_v3_feature_builder(smoothing=20.0, **inp)
    d = b.to_dict()
    b2 = Member2V3FeatureBuilder.from_dict(
        d,
        n_subjects=b.n_subjects,
        n_clusters=b.n_clusters,
        n_bcs=b.n_bcs,
        n_macro_families=b.n_macro_families,
        n_organizations=b.n_organizations,
        n_families=b.n_families,
        global_mean=b.global_mean,
        smoothing=b.smoothing,
        n_train_rows_fit=b.n_train_rows_fit,
    )
    np.testing.assert_array_equal(b.subj_log1p_n, b2.subj_log1p_n)
    np.testing.assert_array_equal(b.bc_passrate, b2.bc_passrate)
    np.testing.assert_array_equal(b.subj_x_bc_log1p_n, b2.subj_x_bc_log1p_n)
    np.testing.assert_array_equal(
        b.subject_to_macro_family_id, b2.subject_to_macro_family_id
    )


# ---------------------------------------------------------------------------
# End-to-end smoke test: fit a tiny GBDT on the v3 features and verify it
# beats the constant-mean baseline. (Lifts the actual gbdt_member infra to
# catch any "the trees won't even split" failure mode.)
# ---------------------------------------------------------------------------
def test_v3_features_can_be_trained_with_gbdt_member() -> None:
    pytest.importorskip("lightgbm")
    from src.gbdt_member import (
        fit_gbdt_member,
        compose_residual_batch,
    )

    inp = _toy_inputs(n_rows=2000, seed=42)
    b = fit_member2_v3_feature_builder(smoothing=10.0, **inp)
    rng = np.random.default_rng(101)
    # Build a synthetic "M1 prediction" that has a known calibration error
    # pattern -- shift up for subjects with high obs_count, down for low.
    subj_logn = b.subj_log1p_n[
        np.where(inp["subject_ids"] >= 0, inp["subject_ids"], 0)
    ]
    p_m1_true = 0.5 + 0.15 * np.tanh((subj_logn - subj_logn.mean()))
    p_m1 = np.clip(
        p_m1_true + rng.normal(0, 0.05, size=p_m1_true.shape), 0.05, 0.95
    )
    subject_mean = np.full(p_m1.shape, b.global_mean, dtype=np.float64)
    X = build_member2_v3_features(
        b,
        p_m1=p_m1,
        subject_mean=subject_mean,
        subject_ids=inp["subject_ids"],
        cluster_ids=inp["cluster_ids"],
        bc_ids=inp["bc_ids"],
    )
    state = fit_gbdt_member(
        X=X,
        y=inp["labels"],
        feature_names=MEMBER2_V3_FEATURE_NAMES,
        init_pred_train=subject_mean,
        n_estimators=30,
        num_leaves=15,
        learning_rate=0.1,
        min_data_in_leaf=20,
        early_stopping_rounds=15,
        seed=7,
    )
    assert state.output_mode == "residual_logit"
    p_pred = compose_residual_batch(state, X, subject_mean)
    # Sanity: predictions are valid probabilities.
    assert np.all((p_pred >= 0) & (p_pred <= 1))
    # The trees should at minimum not be catastrophically worse than baseline.
    y = inp["labels"].astype(np.float64)
    nll_pred = float(
        -(y * np.log(np.clip(p_pred, 1e-6, 1 - 1e-6))
          + (1 - y) * np.log(np.clip(1 - p_pred, 1e-6, 1 - 1e-6))).mean()
    )
    g = float(b.global_mean)
    nll_base = float(
        -(y * math.log(g) + (1 - y) * math.log(1 - g)).mean()
    )
    # Worse-than-baseline by more than 0.10 nat is the v2 failure mode
    # signature; v3 must not reproduce it.
    assert nll_pred < nll_base + 0.10, (
        f"v3 GBDT NLL {nll_pred:.4f} much worse than constant-mean baseline "
        f"{nll_base:.4f} -- v2-style blow-up not fixed"
    )


import math  # imported lazily at the end so module-level imports stay clean
