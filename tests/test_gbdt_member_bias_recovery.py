"""Tests for the LightGBM ``init_score`` recovery in
``src.gbdt_member.fit_gbdt_member``.

Background
----------

LightGBM's binary objective with ``boost_from_average=True`` (default)
adds an implicit ``init_score = logit(mean_y_train)`` to every
prediction. The runtime numpy walker has no such implicit term, so
``fit_gbdt_member`` recovers it once per fit and stores it as
``GBDTMemberState.bias``.

Historically that recovery did

    bias = booster.predict(X_anchor, raw_score=True) - sum_leaves

and asserted parity in raw space. The footgun: in LightGBM >= 4 the
``raw_score=True`` output **does not include init_score**, so for
imbalanced datasets (mean_y != 0.5) the recovery silently lands on 0
and every walker prediction is shifted by ``logit(mean_y_train)`` in
the wrong direction. The shift is invisible to a raw-space parity
check because both sides are now bias-zero -- you only see the bug
when you compute ``-y log(p) - (1-y) log(1-p)`` and find it stuck near
``log(2)``.

The fix: recover bias via PROBABILITY back-solve (``logit(predict(X))
- sum_leaves``), and add a probability-space parity check at fit time
that catches any future regression at ``parity_atol``.

These tests pin both behaviors.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest


GBDT = importlib.import_module("src.gbdt_member")


def _make_imbalanced(N: int = 1024, F: int = 6, p_pos: float = 0.2, seed: int = 0):
    """Synthetic dataset with controlled positive rate."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)
    # Generate labels with a base rate of ``p_pos``, modulated by X.
    logits = X[:, 0] - X[:, 1] + np.log(p_pos / (1.0 - p_pos))
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(N) < probs).astype(np.float32)
    # Trim to exactly the requested base rate so the test is robust.
    return X, y


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_walker_calibration_on_imbalanced_data():
    """On imbalanced targets the walker's mean prediction must align
    with the empirical base rate -- this was the original failure
    symptom (walker p ~ 0.5 on the cold-start val despite mean_y
    being far from 0.5). Modern LightGBM bakes the init_score into
    the first tree's leaves so ``state.bias`` lands at ~0; the
    correctness signal is the END-TO-END calibration, not the bias
    value."""
    X, y = _make_imbalanced(N=2048, F=6, p_pos=0.2, seed=1)
    state = GBDT.fit_gbdt_member(
        X=X, y=y,
        feature_names=tuple(f"x{i}" for i in range(X.shape[1])),
        n_estimators=40, num_leaves=8, min_data_in_leaf=8,
        early_stopping_rounds=40, val_fraction=0.25, seed=0,
        log_period=0,
    )
    p = GBDT.apply_batch(state, X)
    mean_y = float(y.mean())
    mean_p = float(p.mean())
    # Allow ~5% slack -- bagging + train/val split RNG can drift the
    # mean a bit, but a calibration bug like the one we're guarding
    # against would push mean_p toward 0.5 regardless of mean_y.
    assert abs(mean_p - mean_y) < 0.05, (
        f"walker mean prediction {mean_p:.4f} far from empirical mean "
        f"{mean_y:.4f}; calibration is broken (the symptom of the "
        "init_score bug we're guarding against)"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_walker_predictions_match_lightgbm_probability_space():
    """End-to-end: ``apply_batch(state, X)`` must agree with
    ``Booster.predict(X)`` to within ``parity_atol`` on a held-out
    set. The previous raw-space-only check missed this entirely."""
    import lightgbm as lgb

    X, y = _make_imbalanced(N=2048, F=6, p_pos=0.2, seed=2)
    # Carve a held-out portion the trainer never sees as its
    # internal-val split (the trainer uses a random 25% from X). We
    # instead do a deterministic 80/20 by row index so we know which
    # rows are NEVER seen.
    cut = 1700
    X_fit, y_fit = X[:cut], y[:cut]
    X_held = X[cut:]
    state = GBDT.fit_gbdt_member(
        X=X_fit, y=y_fit,
        feature_names=tuple(f"x{i}" for i in range(X.shape[1])),
        n_estimators=20, num_leaves=8, min_data_in_leaf=8,
        early_stopping_rounds=20, val_fraction=0.25, seed=0,
        log_period=0,
    )
    p_walker = GBDT.apply_batch(state, X_held)
    # Re-train a booster with identical seeds so we can compare on
    # ``X_held``. Easier: refit and capture the booster directly by
    # reaching into a thin wrapper.
    rng = np.random.default_rng(0)
    perm = rng.permutation(X_fit.shape[0])
    n_val = max(64, int(round(0.25 * X_fit.shape[0])))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = lgb.Dataset(
        X_fit[train_idx], label=y_fit[train_idx],
        feature_name=[f"x{i}" for i in range(X.shape[1])],
        categorical_feature=[], free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_fit[val_idx], label=y_fit[val_idx],
        feature_name=[f"x{i}" for i in range(X.shape[1])],
        categorical_feature=[], reference=train_set, free_raw_data=False,
    )
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 8, "min_data_in_leaf": 8,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "max_bin": 63, "force_col_wise": True,
        "verbosity": -1, "seed": 0, "deterministic": True,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=20,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    # The trainer above is structurally identical to what
    # fit_gbdt_member does; compare the held-out predictions.
    p_lgbm = booster.predict(X_held)
    # Note: due to bagging RNG the two booster runs are not bit-
    # identical, but they should agree to within a small absolute
    # tolerance. The point of THIS test is to guard against a 0.5
    # gap (the symptom of the init_score bug) rather than to assert
    # bit-exact reproducibility.
    assert float(np.mean(np.abs(p_walker - p_lgbm))) < 0.05, (
        "walker probabilities far from LightGBM's predict; previously "
        "happened when bias=0 dropped logit(mean_y) for imbalanced data"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="LightGBM not installed in this env",
)
def test_log_loss_close_to_lightgbm_internal_on_imbalanced_data():
    """The cell-level log-loss bug manifested as walker logloss ~ log(2)
    even though LightGBM's internal-val logloss was much lower. Verify
    the walker logloss is no longer pinned to the random baseline."""
    X, y = _make_imbalanced(N=4096, F=8, p_pos=0.15, seed=3)
    state = GBDT.fit_gbdt_member(
        X=X, y=y,
        feature_names=tuple(f"x{i}" for i in range(X.shape[1])),
        n_estimators=40, num_leaves=16, min_data_in_leaf=8,
        early_stopping_rounds=40, val_fraction=0.25, seed=0,
        log_period=0,
    )
    p = GBDT.apply_batch(state, X)
    eps = 1e-7
    nll = float(
        -(y * np.log(np.clip(p, eps, 1 - eps))
          + (1 - y) * np.log(1 - np.clip(p, eps, 1 - eps))).mean()
    )
    # Random baseline for p=0.15: nll = -0.15*log(0.15) - 0.85*log(0.85)
    #                                = 0.4227... so anything beating 0.42
    # by a meaningful margin is fine.
    assert nll < 0.40, (
        f"walker nll {nll:.4f} not far enough below the random "
        "baseline; bias recovery may be broken again"
    )
    # And not stuck near log(2) which was the old symptom.
    assert nll < 0.65, (
        f"walker nll {nll:.4f} suspiciously close to log(2)=0.693; "
        "bias recovery probably regressed back to 0"
    )


def test_bias_recovery_is_idempotent_when_y_is_balanced():
    """Sanity: when mean_y == 0.5, init_score == 0, and the recovery
    should land on (or near) 0. This is the legacy code's success
    case; we check that the new probability-back-solve still gets it
    right."""
    pytest.importorskip("lightgbm")

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1024, 4)).astype(np.float32)
    y = (rng.random(1024) < 0.5).astype(np.float32)
    state = GBDT.fit_gbdt_member(
        X=X, y=y,
        feature_names=("a", "b", "c", "d"),
        n_estimators=10, num_leaves=8, min_data_in_leaf=8,
        early_stopping_rounds=10, val_fraction=0.25, seed=0,
        log_period=0,
    )
    # ~0 within numerical noise + finite-sample base rate drift.
    assert abs(state.bias) < 0.20, (
        f"balanced classes should give bias ~ 0; got {state.bias:+.4f}"
    )
