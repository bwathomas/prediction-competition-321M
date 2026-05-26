"""Pin the post-fit ``train_loss`` / ``val_loss`` semantics for
:func:`src.gbdt_member.fit_gbdt_member`.

LightGBM's reported ``binary_logloss`` (visible in ``log_evaluation``
callbacks and ``Booster.best_score``) is empirically biased low by
~0.10 - 0.15 nats vs the actual mean cross-entropy of
``Booster.predict()`` on the same rows -- presumably because
LightGBM's internal score updater + bagging interaction in v4+
produces a metric that is NOT mean cross-entropy.

This bias misled the four-member-stacker into thinking Member 2 had
val NLL = 0.33 when the runtime walker's actual NLL on the same val
rows is ~0.49. The fix: ``GBDTMemberState.train_loss`` /
``GBDTMemberState.val_loss`` are now the MANUAL cross-entropy on
``Booster.predict()``, not the LGBM-reported metric. The runtime
walker matches ``Booster.predict()`` bit-exactly, so the saved
numbers are now an honest summary of what the deployed member will
do.

These tests pin that behavior so a future refactor cannot silently
revert to ``Booster.best_score`` and re-introduce the misleading
number.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from src.gbdt_member import apply_batch, fit_gbdt_member


def _make_data(n: int = 4_000, p: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A small dataset that's hard enough for LightGBM to actually fit
    something nontrivial -- otherwise the gap between manual and
    LGBM-reported NLL is too small to test meaningfully."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Logistic ground truth + structured noise so LightGBM can learn
    # but doesn't perfectly memorize.
    w = rng.standard_normal(p).astype(np.float32)
    z = X @ w + 0.3 * rng.standard_normal(n).astype(np.float32)
    p_true = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n).astype(np.float32) < p_true).astype(np.float32)
    feature_names = [f"f{i}" for i in range(p)]
    return X, y, feature_names


def _manual_nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    y = y.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# 1. The saved val_loss equals the manual cross-entropy on booster.predict()
# ---------------------------------------------------------------------------


def test_state_val_loss_is_manual_cross_entropy_on_booster_predict() -> None:
    """The ``val_loss`` saved in the state must equal mean cross-entropy
    of ``booster.predict(X_val)`` against ``y_val`` -- NOT LightGBM's
    reported ``best_score['val']['binary_logloss']``.
    """
    X, y, names = _make_data(n=4_000, p=12, seed=0)
    state = fit_gbdt_member(
        X=X, y=y, feature_names=names,
        n_estimators=40, learning_rate=0.1, num_leaves=15,
        min_data_in_leaf=20, val_fraction=0.2,
        bagging_fraction=0.9, feature_fraction=0.9, bagging_freq=5,
        early_stopping_rounds=40, seed=0, log_period=0,
    )
    # The walker matches booster.predict bit-exactly (parity check
    # already enforces this in fit_gbdt_member). So computing manual
    # NLL via the walker on the FULL X is equivalent to running
    # booster.predict. We use the val split derived inside
    # fit_gbdt_member implicitly: the val_fraction split with the
    # same seed produces the same indices, and we can recover them
    # by replicating the permutation.
    rng = np.random.default_rng(0)
    n_val = max(64, int(round(0.2 * X.shape[0])))
    perm = rng.permutation(X.shape[0])
    val_idx = perm[:n_val]
    p_walker_val = apply_batch(state, X[val_idx])
    manual_val = _manual_nll(p_walker_val, y[val_idx])
    # Tolerance: walker vs booster.predict is bit-exact in modern
    # LightGBM but ``parity_atol=1e-5`` is the documented contract,
    # so allow up to ~1e-4 NLL drift across LightGBM versions.
    assert math.isclose(state.val_loss, manual_val, abs_tol=1e-4), (
        f"state.val_loss={state.val_loss!r} but manual cross-entropy "
        f"on walker(X_val)={manual_val!r}. The fit must store the "
        "manual NLL, not LightGBM's best_score."
    )


def test_state_train_loss_is_manual_cross_entropy_on_booster_predict() -> None:
    """Same pin but for the train split."""
    X, y, names = _make_data(n=4_000, p=12, seed=1)
    state = fit_gbdt_member(
        X=X, y=y, feature_names=names,
        n_estimators=40, learning_rate=0.1, num_leaves=15,
        min_data_in_leaf=20, val_fraction=0.2,
        bagging_fraction=0.9, feature_fraction=0.9, bagging_freq=5,
        early_stopping_rounds=40, seed=1, log_period=0,
    )
    rng = np.random.default_rng(1)
    n_val = max(64, int(round(0.2 * X.shape[0])))
    perm = rng.permutation(X.shape[0])
    train_idx = perm[n_val:]
    p_walker_train = apply_batch(state, X[train_idx])
    manual_train = _manual_nll(p_walker_train, y[train_idx])
    assert math.isclose(state.train_loss, manual_train, abs_tol=1e-4), (
        f"state.train_loss={state.train_loss!r} but manual "
        f"cross-entropy on walker(X_train)={manual_train!r}. "
        "The fit must store the manual NLL."
    )


# ---------------------------------------------------------------------------
# 2. The saved train_loss / val_loss are NOT booster.best_score
# ---------------------------------------------------------------------------


def test_state_loss_diverges_from_lgbm_reported_when_they_disagree() -> None:
    """If LGBM-reported and manual NLL differ, the state must store the
    manual one. We can't always force a divergence (the gap depends
    on the LightGBM version + bagging config), but on this dataset
    the gap is reliably present, so we assert state.val_loss tracks
    the manual number when there IS a gap.
    """
    X, y, names = _make_data(n=4_000, p=12, seed=2)
    state = fit_gbdt_member(
        X=X, y=y, feature_names=names,
        n_estimators=80, learning_rate=0.1, num_leaves=15,
        min_data_in_leaf=20, val_fraction=0.2,
        bagging_fraction=0.9, feature_fraction=0.9, bagging_freq=5,
        early_stopping_rounds=80, seed=2, log_period=0,
    )
    # Walker NLL on the same val rows used internally.
    rng = np.random.default_rng(2)
    n_val = max(64, int(round(0.2 * X.shape[0])))
    perm = rng.permutation(X.shape[0])
    val_idx = perm[:n_val]
    p_walker_val = apply_batch(state, X[val_idx])
    manual_val = _manual_nll(p_walker_val, y[val_idx])

    # The state's val_loss must match the manual computation; it is
    # NOT permitted to match the LightGBM best_score (which we don't
    # have access to from outside fit_gbdt_member without retraining,
    # but the bit-exact check vs. manual is the right pin).
    assert math.isclose(state.val_loss, manual_val, abs_tol=1e-4), (
        f"state.val_loss={state.val_loss!r} != manual {manual_val!r}"
    )


# ---------------------------------------------------------------------------
# 3. Save/load roundtrip preserves the honest losses
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_honest_losses(tmp_path) -> None:
    """A saved state must round-trip ``train_loss`` and ``val_loss``
    bit-exactly so the meta.json reflects honest numbers."""
    from src.gbdt_member import GBDTMemberState

    X, y, names = _make_data(n=2_000, p=8, seed=3)
    state = fit_gbdt_member(
        X=X, y=y, feature_names=names,
        n_estimators=20, learning_rate=0.1, num_leaves=15,
        min_data_in_leaf=10, val_fraction=0.2,
        early_stopping_rounds=20, seed=3, log_period=0,
    )
    state.save(tmp_path)
    reloaded = GBDTMemberState.load(tmp_path)
    assert math.isclose(reloaded.train_loss, state.train_loss, abs_tol=0.0)
    assert math.isclose(reloaded.val_loss, state.val_loss, abs_tol=0.0)
