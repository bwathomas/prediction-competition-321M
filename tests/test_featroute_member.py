"""Tests for src/featroute_member.py.

FeatRoute = 8 GBDT sub-models (one per feature-group slice) blended by
logit-mean. The load-bearing checks:

* ``apply_one`` == ``apply_batch`` (< 1e-6) -- the single-row and
  vectorized paths must agree.
* save / load round-trips (each sub-model is its own GBDTMemberState).
* the logit-mean equals a HAND-computed value on a tiny case (built from
  hand-crafted single-leaf GBDTs whose raw score is a known constant).
* missing / empty groups are skipped, not crashed on.

The ``fit_featroute`` integration tests need lightgbm (offline only);
the hand-built tests do not and run anywhere.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.featroute_member import (
    FEATROUTE_GROUP_NAMES,
    FeatRouteState,
    apply_batch,
    apply_one,
    fit_featroute,
)
from src.gbdt_member import GBDTMemberState


# ---------------------------------------------------------------------------
# Hand-built helpers (no lightgbm needed)
# ---------------------------------------------------------------------------


def _const_gbdt(feature_dim: int, leaf_value: float, name: str) -> GBDTMemberState:
    """A 1-tree, 1-leaf GBDT whose raw score (logit) is ``leaf_value``
    for EVERY input -- bias 0, single leaf carrying ``leaf_value``.

    ``predict_raw`` on this state returns ``leaf_value`` for all rows, so
    it is the perfect probe for the logit-mean arithmetic.
    """
    return GBDTMemberState(
        feature_concat=np.array([-1], dtype=np.int32),       # single leaf
        threshold_concat=np.array([leaf_value], dtype=np.float64),
        left_concat=np.array([-1], dtype=np.int32),
        right_concat=np.array([-1], dtype=np.int32),
        default_left_concat=np.array([False], dtype=np.bool_),
        tree_offsets=np.array([0, 1], dtype=np.int32),
        feature_dim=int(feature_dim),
        feature_names=tuple(f"{name}__{j}" for j in range(feature_dim)),
        bias=0.0,
        fit_method="hand",
        n_train=0,
        n_pos=0,
        n_trees=1,
        train_loss=0.0,
        val_loss=0.0,
    )


def _equal_group_slices(n_groups: int, width: int) -> tuple[dict, list, int]:
    """Contiguous equal-width slices for ``n_groups`` groups."""
    names = list(FEATROUTE_GROUP_NAMES[:n_groups])
    slices = {g: slice(i * width, (i + 1) * width) for i, g in enumerate(names)}
    feature_dim = n_groups * width
    return slices, names, feature_dim


# ---------------------------------------------------------------------------
# Hand-computed logit-mean
# ---------------------------------------------------------------------------


def test_logit_mean_matches_hand_computed_value():
    """Three constant-logit sub-models -> the blended probability is
    sigmoid(mean of the three constants), exactly."""
    width = 2
    n_groups = 3
    slices, names, feature_dim = _equal_group_slices(n_groups, width)
    logits = [1.0, -0.5, 2.0]
    subs = [_const_gbdt(width, lv, nm) for lv, nm in zip(logits, names)]
    state = FeatRouteState(
        sub_states=subs,
        group_slices=slices,
        group_names=names,
        feature_dim=feature_dim,
    )

    mean_logit = sum(logits) / len(logits)            # (1.0 - 0.5 + 2.0)/3
    expected_p = 1.0 / (1.0 + math.exp(-mean_logit))

    x = np.zeros(feature_dim, dtype=np.float32)        # value irrelevant (const leaves)
    p_one = apply_one(state, x)
    assert math.isclose(p_one, expected_p, rel_tol=1e-9, abs_tol=1e-9)

    X = np.zeros((5, feature_dim), dtype=np.float32)
    p_batch = apply_batch(state, X)
    np.testing.assert_allclose(p_batch, expected_p, atol=1e-7)


def test_apply_one_equals_apply_batch_handbuilt():
    """apply_one == apply_batch (<1e-6) on a hand-built 8-group state,
    even when the sub-models actually read their feature slices."""
    rng = np.random.default_rng(0)
    width = 3
    n_groups = 8
    slices, names, feature_dim = _equal_group_slices(n_groups, width)
    # Give each group a 1-split tree on its first column so different
    # rows route to different leaves (not a degenerate constant).
    subs = []
    for i, nm in enumerate(names):
        feat = np.array([0, -1, -1], dtype=np.int32)
        thr = np.array([0.0, 0.5 + 0.1 * i, -0.5 - 0.1 * i], dtype=np.float64)
        left = np.array([1, -1, -1], dtype=np.int32)
        right = np.array([2, -1, -1], dtype=np.int32)
        dleft = np.array([True, False, False], dtype=np.bool_)
        subs.append(
            GBDTMemberState(
                feature_concat=feat,
                threshold_concat=thr,
                left_concat=left,
                right_concat=right,
                default_left_concat=dleft,
                tree_offsets=np.array([0, 3], dtype=np.int32),
                feature_dim=width,
                feature_names=tuple(f"{nm}__{j}" for j in range(width)),
                bias=0.0,
                fit_method="hand",
                n_train=0, n_pos=0, n_trees=1, train_loss=0.0, val_loss=0.0,
            )
        )
    state = FeatRouteState(
        sub_states=subs, group_slices=slices, group_names=names,
        feature_dim=feature_dim,
    )

    X = rng.normal(size=(64, feature_dim)).astype(np.float32)
    # Sprinkle NaNs to exercise the default_left path through the slices.
    X[rng.uniform(size=X.shape) < 0.05] = np.nan
    p_batch = apply_batch(state, X)
    p_one = np.array(
        [apply_one(state, X[i]) for i in range(X.shape[0])], dtype=np.float32
    )
    np.testing.assert_allclose(p_batch, p_one, atol=1e-6)
    assert np.all(np.isfinite(p_batch))
    assert float(p_batch.min()) > 0.0 and float(p_batch.max()) < 1.0


# ---------------------------------------------------------------------------
# Missing / empty groups
# ---------------------------------------------------------------------------


def test_missing_group_skipped_in_mean():
    """A ``None`` sub-model is skipped; the mean is over present groups."""
    width = 2
    names = list(FEATROUTE_GROUP_NAMES[:3])
    slices = {g: slice(i * width, (i + 1) * width) for i, g in enumerate(names)}
    feature_dim = 3 * width
    subs = [
        _const_gbdt(width, 1.0, names[0]),
        None,                                  # missing group
        _const_gbdt(width, 3.0, names[2]),
    ]
    state = FeatRouteState(
        sub_states=subs, group_slices=slices, group_names=names,
        feature_dim=feature_dim,
    )
    assert state.n_present_groups == 2
    # Mean over the two present logits = (1 + 3) / 2 = 2.0.
    expected_p = 1.0 / (1.0 + math.exp(-2.0))
    p = apply_one(state, np.zeros(feature_dim, dtype=np.float32))
    assert math.isclose(p, expected_p, rel_tol=1e-9, abs_tol=1e-9)


def test_all_groups_missing_returns_half():
    """If NO group is present the member is neutral (p = 0.5)."""
    width = 2
    names = list(FEATROUTE_GROUP_NAMES[:2])
    slices = {g: slice(i * width, (i + 1) * width) for i, g in enumerate(names)}
    state = FeatRouteState(
        sub_states=[None, None], group_slices=slices, group_names=names,
        feature_dim=2 * width,
    )
    p = apply_one(state, np.zeros(2 * width, dtype=np.float32))
    assert math.isclose(p, 0.5, abs_tol=1e-9)
    pb = apply_batch(state, np.zeros((4, 2 * width), dtype=np.float32))
    np.testing.assert_allclose(pb, 0.5, atol=1e-7)


def test_empty_slice_treated_as_missing_at_fit(tmp_path):
    """A group whose slice selects zero columns is recorded absent."""
    # 2 groups, but the second has a zero-width slice.
    names = ["nn_label_derivatives", "cluster_passrate"]
    slices = {
        "nn_label_derivatives": slice(0, 4),
        "cluster_passrate": slice(4, 4),       # zero columns
    }
    state = FeatRouteState(
        sub_states=[_const_gbdt(4, 0.7, names[0]), None],
        group_slices=slices, group_names=names, feature_dim=4,
    )
    assert state.n_present_groups == 1
    expected_p = 1.0 / (1.0 + math.exp(-0.7))
    p = apply_one(state, np.zeros(4, dtype=np.float32))
    assert math.isclose(p, expected_p, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_handbuilt(tmp_path):
    """Round-trip a state with a missing group; predictions must match."""
    width = 3
    names = list(FEATROUTE_GROUP_NAMES[:4])
    slices = {g: slice(i * width, (i + 1) * width) for i, g in enumerate(names)}
    feature_dim = 4 * width
    subs = [
        _const_gbdt(width, 0.5, names[0]),
        None,
        _const_gbdt(width, -1.0, names[2]),
        _const_gbdt(width, 2.0, names[3]),
    ]
    state = FeatRouteState(
        sub_states=subs, group_slices=slices, group_names=names,
        feature_dim=feature_dim,
    )
    state.save(tmp_path)
    s2 = FeatRouteState.load(tmp_path)

    assert s2.group_names == state.group_names
    assert s2.feature_dim == state.feature_dim
    assert s2.n_present_groups == state.n_present_groups
    for g in names:
        assert s2.group_slices[g] == state.group_slices[g]

    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, feature_dim)).astype(np.float32)
    np.testing.assert_allclose(apply_batch(state, X), apply_batch(s2, X), atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_apply_one_rejects_dim_mismatch():
    state = FeatRouteState(
        sub_states=[_const_gbdt(2, 0.0, "g")],
        group_slices={FEATROUTE_GROUP_NAMES[0]: slice(0, 2)},
        group_names=[FEATROUTE_GROUP_NAMES[0]],
        feature_dim=2,
    )
    with pytest.raises(ValueError, match="features dim"):
        apply_one(state, np.zeros(3, dtype=np.float32))


def test_apply_batch_empty_input():
    state = FeatRouteState(
        sub_states=[_const_gbdt(2, 0.0, "g")],
        group_slices={FEATROUTE_GROUP_NAMES[0]: slice(0, 2)},
        group_names=[FEATROUTE_GROUP_NAMES[0]],
        feature_dim=2,
    )
    out = apply_batch(state, np.empty((0, 2), dtype=np.float32))
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_state_rejects_subdim_mismatch():
    """A sub-model whose feature_dim != its slice width is rejected."""
    with pytest.raises(ValueError, match="slice width"):
        FeatRouteState(
            sub_states=[_const_gbdt(3, 0.0, "g")],   # 3-wide sub-model...
            group_slices={FEATROUTE_GROUP_NAMES[0]: slice(0, 2)},  # ...2-wide slice
            group_names=[FEATROUTE_GROUP_NAMES[0]],
            feature_dim=2,
        )


# ---------------------------------------------------------------------------
# fit_featroute integration (needs lightgbm; offline only)
# ---------------------------------------------------------------------------


def _make_synthetic_8group(N=1500, width=4, seed=0):
    """8 contiguous feature groups; the label depends on a couple of
    columns within several groups so each sub-GBDT learns a little."""
    rng = np.random.default_rng(seed)
    n_groups = 8
    feature_dim = n_groups * width
    X = rng.normal(size=(N, feature_dim)).astype(np.float32)
    # Signal spread across groups 0, 2, 5 (first column of each).
    z = (
        1.2 * X[:, 0 * width]
        - 0.8 * X[:, 2 * width]
        + 0.6 * X[:, 5 * width] * (X[:, 5 * width + 1] > 0)
    )
    p = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    y = (rng.random(N) < p).astype(np.float32)
    slices = {
        FEATROUTE_GROUP_NAMES[i]: slice(i * width, (i + 1) * width)
        for i in range(n_groups)
    }
    return X, y, slices, feature_dim


def test_fit_featroute_trains_eight_groups_and_parity():
    pytest.importorskip("lightgbm")
    X, y, slices, feature_dim = _make_synthetic_8group(N=1200, width=4, seed=1)
    state = fit_featroute(
        X=X,
        y=y,
        group_slices=slices,
        fold_train_idx=None,
        seed=0,
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        early_stopping_rounds=5,
        parity_atol=1e-5,
    )
    assert state.n_groups == 8
    assert state.n_present_groups == 8
    assert state.feature_dim == feature_dim

    # apply_one == apply_batch on a held-out batch.
    rng = np.random.default_rng(7)
    X_test = rng.normal(size=(100, feature_dim)).astype(np.float32)
    p_batch = apply_batch(state, X_test)
    p_one = np.array(
        [apply_one(state, X_test[i]) for i in range(X_test.shape[0])],
        dtype=np.float32,
    )
    np.testing.assert_allclose(p_batch, p_one, atol=1e-6)
    assert np.all((p_batch > 0) & (p_batch < 1))


def test_fit_featroute_save_load_predict_match(tmp_path):
    pytest.importorskip("lightgbm")
    X, y, slices, feature_dim = _make_synthetic_8group(N=900, width=3, seed=2)
    state = fit_featroute(
        X=X, y=y, group_slices=slices, seed=3,
        n_estimators=20, learning_rate=0.1, num_leaves=8,
        min_data_in_leaf=10, early_stopping_rounds=5, parity_atol=1e-5,
    )
    state.save(tmp_path)
    s2 = FeatRouteState.load(tmp_path)
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(40, feature_dim)).astype(np.float32)
    np.testing.assert_allclose(
        apply_batch(state, X_test), apply_batch(s2, X_test), atol=1e-6
    )


def test_fit_featroute_fold_train_idx_subsets_rows():
    pytest.importorskip("lightgbm")
    X, y, slices, feature_dim = _make_synthetic_8group(N=1000, width=3, seed=4)
    # Train only on the first 600 rows; member must still predict on all.
    fold_train_idx = np.arange(600)
    state = fit_featroute(
        X=X, y=y, group_slices=slices, fold_train_idx=fold_train_idx, seed=0,
        n_estimators=15, learning_rate=0.1, num_leaves=8,
        min_data_in_leaf=10, early_stopping_rounds=5, parity_atol=1e-5,
    )
    # Predict on the held-out rows (rows 600:).
    p = apply_batch(state, X[600:])
    assert p.shape == (400,)
    assert np.all((p > 0) & (p < 1))
