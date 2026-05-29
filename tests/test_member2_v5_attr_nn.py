"""Unit tests for ``src.member2_v5_attr_nn``.

Covers the failure modes that bit v2/v3/v4:
  - silent column-order reshuffling -> wrong-feature splits at apply time
  - sentinel "-1" id handling -> NaN or out-of-bounds index
  - smoothing math -> empty cells collapse to global mean
  - 2-D cell-count / cell-mean derivation -> off-by-one or row/col swap
  - apply-time produces finite outputs even for missing pool / NN inputs
  - the schema must NOT contain any M1-derived input (the v4 saturation bug)
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from src.member2_v5_attr_nn import (
    M2V5_BUCKET_B_END,
    M2V5_BUCKET_C_END,
    M2V5_BUCKET_D_END,
    M2V5_BUCKET_TE_END,
    M2V5_BUCKET_U_END,
    M2V5_FEATURE_DIM,
    M2V5_NN_SOURCE_INDICES,
    MEMBER2_V5_FEATURE_NAMES,
    Member2V5FeatureBuilder,
    Member2V5State,
    Member2V5Warning,
    build_member2_v5_features,
    fit_member2_v5_feature_builder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _toy_inputs(n_rows: int = 500, seed: int = 7) -> dict:
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


def _toy_pool(N: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, 17)).astype(np.float32)


def _toy_nn_matrix(N: int, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(N, 23)).astype(np.float32)


def _toy_age(N: int, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.0, 5.0, size=N).astype(np.float64)
    # Sprinkle NaNs to exercise the mask channel.
    age[rng.random(N) < 0.05] = np.nan
    return age


# ---------------------------------------------------------------------------
# Spec audits (the v4-saturation regression guards)
# ---------------------------------------------------------------------------
def test_schema_is_locked_at_49_columns() -> None:
    assert M2V5_FEATURE_DIM == 49
    assert len(MEMBER2_V5_FEATURE_NAMES) == 49


def test_schema_contains_no_m1_inputs() -> None:
    # The whole point of v5 over v4: structural orthogonality with M1.
    # Any name containing "p_m1" / "logit_p_m1" / "m1" leaks the anchor.
    banned = ("p_m1", "logit_p_m1", "p1", "model_a", "member1")
    for name in MEMBER2_V5_FEATURE_NAMES:
        for token in banned:
            assert token not in name.lower(), (
                f"v5 schema contains forbidden M1-derived feature: {name}"
            )
    # ALSO: 'logit_subject_mean' is banned -- it's the v3/v4 anchor's
    # second-largest source of M1 mimicry (subject_mean is M1's
    # implicit denominator).
    assert "logit_subject_mean" not in MEMBER2_V5_FEATURE_NAMES
    assert "subject_mean" not in MEMBER2_V5_FEATURE_NAMES


def test_bucket_boundaries_match_schema() -> None:
    assert M2V5_BUCKET_B_END == 19
    assert M2V5_BUCKET_C_END == 28
    assert M2V5_BUCKET_TE_END == 36
    assert M2V5_BUCKET_U_END == 42
    assert M2V5_BUCKET_D_END == 49
    # First Bucket-C col must be subject_obs_count_log1p (cold-start
    # primary "how well do we know this subject").
    assert MEMBER2_V5_FEATURE_NAMES[M2V5_BUCKET_B_END] == "subject_obs_count_log1p"
    # First Bucket-TE col must be subject_passrate_meanenc.
    assert MEMBER2_V5_FEATURE_NAMES[M2V5_BUCKET_C_END] == "subject_passrate_meanenc"
    # is_unknown_bc must be at index 38 (the user's explicit ask).
    assert MEMBER2_V5_FEATURE_NAMES[38] == "is_unknown_bc"


def test_pool_feature_block_matches_canonical_pool_names() -> None:
    # The 9 pool cols must come from item_features.POOL_FEATURE_NAMES
    # in the canonical order so the notebook can call reindex() with no
    # remapping.
    from src.item_features import POOL_FEATURE_NAMES

    for k, name in enumerate(POOL_FEATURE_NAMES):
        assert MEMBER2_V5_FEATURE_NAMES[k] == f"pool_{name}", (
            f"Pool feature {k} must be 'pool_{name}' to match item_features"
        )


def test_centroid_distance_block_is_contiguous_after_pool() -> None:
    # 8 centroid cols immediately after the 9 pool cols.
    for k in range(8):
        assert MEMBER2_V5_FEATURE_NAMES[9 + k] == f"centroid_dist_{k}"


def test_nn_source_indices_within_nn_features_schema() -> None:
    from src.nn_features import NN_FEATURE_NAMES

    assert max(M2V5_NN_SOURCE_INDICES) < len(NN_FEATURE_NAMES)
    # Spot-check the names round-trip.
    assert NN_FEATURE_NAMES[1] == "passrate_weighted_mean"
    assert NN_FEATURE_NAMES[15] == "passrate_subject_conditional"


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def test_fit_returns_well_shaped_builder() -> None:
    inp = _toy_inputs()
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    assert b.subj_log1p_n.shape == (inp["n_subjects"],)
    assert b.cluster_log1p_n.shape == (inp["n_clusters"],)
    assert b.bc_log1p_n.shape == (inp["n_bcs"],)
    assert b.macro_log1p_n.shape == (inp["n_macro_families"],)
    assert b.org_log1p_n.shape == (inp["n_organizations"],)
    assert b.fam_log1p_n.shape == (inp["n_families"],)
    assert b.subj_passrate.shape == (inp["n_subjects"],)
    assert b.subj_x_bc_log1p_n.shape == (inp["n_subjects"], inp["n_bcs"])
    assert b.subj_x_bc_passrate.shape == (inp["n_subjects"], inp["n_bcs"])
    assert b.subj_x_cluster_log1p_n.shape == (
        inp["n_subjects"],
        inp["n_clusters"],
    )
    assert b.subj_x_cluster_passrate.shape == (
        inp["n_subjects"],
        inp["n_clusters"],
    )
    assert 0.0 < b.global_mean < 1.0


def test_fit_shrinkage_collapses_empty_cell_to_global_mean() -> None:
    inp = _toy_inputs()
    inp["cluster_ids"] = np.where(
        inp["cluster_ids"] == 7, 0, inp["cluster_ids"]
    )
    b = fit_member2_v5_feature_builder(smoothing=30.0, **inp)
    assert b.cluster_passrate[7] == pytest.approx(b.global_mean, abs=1e-6)


def test_fit_rejects_bad_shapes() -> None:
    inp = _toy_inputs()
    inp["labels"] = inp["labels"][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_member2_v5_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_negative_smoothing() -> None:
    inp = _toy_inputs()
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        fit_member2_v5_feature_builder(smoothing=-1.0, **inp)


def test_subject_to_trait_arrays_have_correct_length() -> None:
    inp = _toy_inputs()
    inp["subject_to_macro_family_id"] = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError, match="must each have shape"):
        fit_member2_v5_feature_builder(smoothing=10.0, **inp)


def test_fit_rejects_undersized_cluster_vocab_with_clear_message() -> None:
    # Regression: undersized n_clusters used to raise a cryptic
    # ``IndexError: index 64 is out of bounds for axis 1 with size 64``.
    inp = _toy_inputs(n_rows=200)
    inp["cluster_ids"][0] = int(inp["n_clusters"])
    with pytest.raises(ValueError, match=r"observed max\(cluster_ids\)"):
        fit_member2_v5_feature_builder(smoothing=10.0, **inp)


# ---------------------------------------------------------------------------
# Apply: shape, finiteness, NN/pool wiring
# ---------------------------------------------------------------------------
def test_build_features_shape_and_dtype() -> None:
    inp = _toy_inputs(n_rows=300)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    X = build_member2_v5_features(
        b,
        subject_ids=inp["subject_ids"],
        cluster_ids=inp["cluster_ids"],
        bc_ids=inp["bc_ids"],
        pool_features=_toy_pool(300),
        benchmark_age=_toy_age(300),
        nn_features_matrix=_toy_nn_matrix(300),
    )
    assert X.shape == (300, M2V5_FEATURE_DIM)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_build_features_with_missing_pool_warns_but_finite() -> None:
    inp = _toy_inputs(n_rows=50)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = build_member2_v5_features(
            b,
            subject_ids=inp["subject_ids"][:50],
            cluster_ids=inp["cluster_ids"][:50],
            bc_ids=inp["bc_ids"][:50],
            pool_features=None,
            nn_features_matrix=_toy_nn_matrix(50),
        )
    assert np.isfinite(X).all()
    # The 17 pool cols should be exactly zero.
    assert np.all(X[:, 0:17] == 0.0)
    # The warning category should be Member2V5Warning.
    assert any(issubclass(_.category, Member2V5Warning) for _ in w)


def test_build_features_with_missing_nn_warns_but_finite() -> None:
    inp = _toy_inputs(n_rows=40)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = build_member2_v5_features(
            b,
            subject_ids=inp["subject_ids"][:40],
            cluster_ids=inp["cluster_ids"][:40],
            bc_ids=inp["bc_ids"][:40],
            pool_features=_toy_pool(40),
            nn_features_matrix=None,
        )
    assert np.isfinite(X).all()
    # The 7 NN cols at the end should be exactly zero.
    assert np.all(X[:, 42:49] == 0.0)
    assert any(issubclass(_.category, Member2V5Warning) for _ in w)


def test_build_features_unknown_ids_fire_the_correct_flags() -> None:
    inp = _toy_inputs(n_rows=20)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    subj = inp["subject_ids"][:20].copy()
    clus = inp["cluster_ids"][:20].copy()
    bc = inp["bc_ids"][:20].copy()
    subj[0] = -1
    clus[1] = -1
    bc[2] = -1
    X = build_member2_v5_features(
        b,
        subject_ids=subj,
        cluster_ids=clus,
        bc_ids=bc,
        pool_features=_toy_pool(20),
        nn_features_matrix=_toy_nn_matrix(20),
    )
    # Cols 36..41 = unknown flags in subject/cluster/bc/macro/org/fam order.
    assert X[0, 36] == 1.0
    assert X[1, 37] == 1.0
    assert X[2, 38] == 1.0
    # Subject-unknown row also propagates to macro/org/family flags.
    assert X[0, 39] == 1.0
    assert X[0, 40] == 1.0
    assert X[0, 41] == 1.0
    # Sanity: a row with valid subject keeps the subject flag at 0.
    assert X[3, 36] == 0.0


def test_build_features_unknown_subject_falls_back_to_global_mean_passrate() -> None:
    inp = _toy_inputs(n_rows=10)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    subj = np.full(10, -1, dtype=np.int64)
    clus = np.full(10, -1, dtype=np.int64)
    bc = np.full(10, -1, dtype=np.int64)
    X = build_member2_v5_features(
        b,
        subject_ids=subj,
        cluster_ids=clus,
        bc_ids=bc,
        pool_features=_toy_pool(10),
        nn_features_matrix=_toy_nn_matrix(10),
    )
    g = float(b.global_mean)
    # Cols 28..35 are TE passrates; all should collapse to g for unknown ids.
    for col in range(28, 36):
        assert np.allclose(X[:, col], g, atol=1e-5)
    # Cols 19..26 are 1-D / 2-D log1p counts; unknown ids -> 0.
    for col in range(19, 27):
        assert np.allclose(X[:, col], 0.0, atol=1e-6)


def test_2d_cell_count_matches_naive_aggregate() -> None:
    inp = _toy_inputs(n_rows=400)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
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


def test_2d_cell_mean_matches_naive_aggregate() -> None:
    inp = _toy_inputs(n_rows=400)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    subj = inp["subject_ids"]
    bc = inp["bc_ids"]
    y = inp["labels"].astype(np.float64)
    valid = (subj >= 0) & (bc >= 0)
    n_cell = np.zeros((inp["n_subjects"], inp["n_bcs"]), dtype=np.float64)
    sum_cell = np.zeros((inp["n_subjects"], inp["n_bcs"]), dtype=np.float64)
    for s, c, yi in zip(subj[valid], bc[valid], y[valid]):
        n_cell[int(s), int(c)] += 1.0
        sum_cell[int(s), int(c)] += float(yi)
    # Bayes-shrunk mean.
    naive_passrate = (
        sum_cell + 10.0 * b.global_mean
    ) / (n_cell + 10.0)
    np.testing.assert_allclose(
        b.subj_x_bc_passrate,
        naive_passrate.astype(np.float32),
        atol=1e-5,
    )


def test_bc_redacted_mask_is_pipeline_through() -> None:
    inp = _toy_inputs(n_rows=30)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    mask = np.array([0, 1] * 15, dtype=np.float32)
    X = build_member2_v5_features(
        b,
        subject_ids=inp["subject_ids"][:30],
        cluster_ids=inp["cluster_ids"][:30],
        bc_ids=inp["bc_ids"][:30],
        bc_redacted_mask=mask,
        pool_features=_toy_pool(30),
        nn_features_matrix=_toy_nn_matrix(30),
    )
    np.testing.assert_array_equal(X[:, 27], mask)


def test_benchmark_age_value_and_mask_columns() -> None:
    inp = _toy_inputs(n_rows=20)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    age = np.array([1.0, 2.0, np.nan, 4.0] * 5, dtype=np.float64)
    X = build_member2_v5_features(
        b,
        subject_ids=inp["subject_ids"][:20],
        cluster_ids=inp["cluster_ids"][:20],
        bc_ids=inp["bc_ids"][:20],
        pool_features=_toy_pool(20),
        benchmark_age=age,
        nn_features_matrix=_toy_nn_matrix(20),
    )
    # Col 17 = age value (NaN -> 0), col 18 = mask (NaN -> 0, finite -> 1).
    for i, a in enumerate(age):
        if np.isnan(a):
            assert X[i, 17] == 0.0
            assert X[i, 18] == 0.0
        else:
            assert X[i, 17] == pytest.approx(float(a), abs=1e-6)
            assert X[i, 18] == 1.0


def test_nn_column_selection_picks_correct_source_columns() -> None:
    # If a row has a recognisable per-column signature in nn_features_matrix,
    # the v5 NN block should reflect THAT row's signature on the chosen
    # source indices.
    inp = _toy_inputs(n_rows=5)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    nn = np.zeros((5, 23), dtype=np.float32)
    # Mark each of the 7 source indices with a unique sentinel value.
    for k, src in enumerate(M2V5_NN_SOURCE_INDICES):
        nn[0, src] = float(k + 1) * 0.1  # 0.1, 0.2, ..., 0.7
    X = build_member2_v5_features(
        b,
        subject_ids=inp["subject_ids"][:5],
        cluster_ids=inp["cluster_ids"][:5],
        bc_ids=inp["bc_ids"][:5],
        pool_features=_toy_pool(5),
        nn_features_matrix=nn,
    )
    for k in range(7):
        assert X[0, 42 + k] == pytest.approx((k + 1) * 0.1, abs=1e-5)


def test_build_features_rejects_length_mismatch() -> None:
    inp = _toy_inputs(n_rows=10)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    with pytest.raises(ValueError, match="must all have the same length"):
        build_member2_v5_features(
            b,
            subject_ids=inp["subject_ids"][:10],
            cluster_ids=inp["cluster_ids"][:5],  # wrong
            bc_ids=inp["bc_ids"][:10],
        )


# ---------------------------------------------------------------------------
# State + serialization
# ---------------------------------------------------------------------------
def test_state_warns_on_mismatched_feature_names() -> None:
    inp = _toy_inputs(n_rows=20)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    state = Member2V5State(gbdt=None, builder=b)
    assert state.version.startswith("v5")
    # Mismatched names emit a warning, not an exception (v5 policy:
    # soft data-quality issues are warnings).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Member2V5State(
            gbdt=None,
            builder=b,
            feature_names=("foo",) + MEMBER2_V5_FEATURE_NAMES[1:],
        )
    assert any(issubclass(_.category, Member2V5Warning) for _ in w)


def test_builder_to_dict_round_trip() -> None:
    inp = _toy_inputs(n_rows=80)
    b = fit_member2_v5_feature_builder(smoothing=20.0, **inp)
    d = b.to_dict()
    b2 = Member2V5FeatureBuilder.from_dict(
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
    np.testing.assert_array_equal(b.subj_passrate, b2.subj_passrate)
    np.testing.assert_array_equal(b.subj_x_bc_passrate, b2.subj_x_bc_passrate)
    np.testing.assert_array_equal(b.bc_passrate, b2.bc_passrate)
    np.testing.assert_array_equal(b.subj_x_bc_log1p_n, b2.subj_x_bc_log1p_n)
    np.testing.assert_array_equal(
        b.subject_to_macro_family_id, b2.subject_to_macro_family_id
    )


# ---------------------------------------------------------------------------
# End-to-end smoke test: fit a tiny GBDT on the v5 features and verify it
# beats the constant-mean baseline. The point is to catch any "the trees
# refuse to split" failure mode (e.g. all features collapsed to constants)
# before it shows up at notebook scale.
# ---------------------------------------------------------------------------
def test_v5_features_can_be_trained_with_gbdt_member_direct_binary() -> None:
    pytest.importorskip("lightgbm")
    from src.gbdt_member import apply_batch, fit_gbdt_member

    inp = _toy_inputs(n_rows=2000, seed=42)
    b = fit_member2_v5_feature_builder(smoothing=10.0, **inp)
    pool = _toy_pool(2000)
    nn = _toy_nn_matrix(2000)
    age = _toy_age(2000)
    X = build_member2_v5_features(
        b,
        subject_ids=inp["subject_ids"],
        cluster_ids=inp["cluster_ids"],
        bc_ids=inp["bc_ids"],
        pool_features=pool,
        benchmark_age=age,
        nn_features_matrix=nn,
    )
    # Direct binary mode: no init_pred_train.
    state = fit_gbdt_member(
        X=X,
        y=inp["labels"],
        feature_names=MEMBER2_V5_FEATURE_NAMES,
        init_pred_train=None,
        n_estimators=30,
        num_leaves=15,
        learning_rate=0.1,
        min_data_in_leaf=20,
        early_stopping_rounds=15,
        seed=7,
    )
    assert state.output_mode == "probability"
    p_pred = apply_batch(state, X)
    assert np.all((p_pred >= 0) & (p_pred <= 1))
    y = inp["labels"].astype(np.float64)
    nll_pred = float(
        -(y * np.log(np.clip(p_pred, 1e-6, 1 - 1e-6))
          + (1 - y) * np.log(np.clip(1 - p_pred, 1e-6, 1 - 1e-6))).mean()
    )
    g = float(b.global_mean)
    nll_base = float(
        -(y * math.log(g) + (1 - y) * math.log(1 - g)).mean()
    )
    # Worse-than-baseline by more than 0.10 nat is the v4 failure mode
    # signature. v5 must not reproduce it on toy data.
    assert nll_pred < nll_base + 0.10, (
        f"v5 GBDT NLL {nll_pred:.4f} much worse than constant-mean baseline "
        f"{nll_base:.4f} -- v4-style blow-up not fixed"
    )
