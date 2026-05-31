"""Negative-correlation-learning (NCL) hooks across the gradient trainers.

Each test trains the same model with and without the NCL penalty against a
fixed anchor and checks that the penalty pushes the member's signed errors to
be *less* positively correlated with the anchor's signed errors -- i.e. NCL
does what it says on the tin.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.fwfm_member import apply_state_batch, fit_fwfm_member
from src.logreg_member import apply_state_batch as apply_logreg_state, fit_logreg_member
from src.mlp_member import apply_batch as apply_mlp_batch, fit_mlp_member


def _err_corr(p_member, anchor, y):
    em = np.asarray(p_member, dtype=np.float64) - y
    ea = np.asarray(anchor, dtype=np.float64) - y
    if em.std() < 1e-9 or ea.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(em, ea)[0, 1])


def _make_signal(n=5000, f=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    logit = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.4
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    # Anchor: tracks a *different* part of the signal, so an unconstrained
    # member naturally shares error structure with it.
    a_logit = 1.2 * X[:, 0] + 0.5 * X[:, 2]
    anchor = (1.0 / (1.0 + np.exp(-a_logit))).astype(np.float32)
    return X, y, anchor


def test_mlp_ncl_reduces_error_correlation():
    X, y, anchor = _make_signal()
    common = dict(
        labels=y, dense_X=X, hid1=32, hid2=16, epochs=15, batch_size=512,
        seed=3, show_progress=False, learning_rate=3e-3,
    )
    base = fit_mlp_member(**common)
    ncl = fit_mlp_member(**common, ncl_anchor_preds=anchor, ncl_lambda=3.0)
    c_base = _err_corr(apply_mlp_batch(base, dense_X=X), anchor, y)
    c_ncl = _err_corr(apply_mlp_batch(ncl, dense_X=X), anchor, y)
    assert np.isfinite(c_ncl)
    assert c_ncl < c_base - 0.02, f"NCL did not decorrelate: base={c_base:.3f} ncl={c_ncl:.3f}"


def test_logreg_ncl_reduces_error_correlation():
    X, y, anchor = _make_signal(seed=1)
    names = tuple(f"f_{i}" for i in range(X.shape[1]))
    common = dict(
        X=X, y=y, feature_names=names, epochs=80, batch_size=512,
        learning_rate=5e-2, seed=4, log_every=0,
    )
    base = fit_logreg_member(**common)
    ncl = fit_logreg_member(**common, ncl_anchor_preds=anchor, ncl_lambda=3.0)
    c_base = _err_corr(apply_logreg_state(base, X), anchor, y)
    c_ncl = _err_corr(apply_logreg_state(ncl, X), anchor, y)
    assert np.isfinite(c_ncl)
    assert c_ncl < c_base - 0.02, f"NCL did not decorrelate: base={c_base:.3f} ncl={c_ncl:.3f}"


def test_fwfm_ncl_reduces_error_correlation():
    X, y, anchor = _make_signal(seed=2)
    names = tuple(f"f_{i}" for i in range(X.shape[1]))
    common = dict(
        X=X, y=y, feature_names=names, k=4, epochs=25, batch_size=512,
        learning_rate=5e-2, seed=5, log_every=0, standardize=True,
    )
    base = fit_fwfm_member(**common)
    ncl = fit_fwfm_member(**common, ncl_anchor_preds=anchor, ncl_lambda=3.0)
    c_base = _err_corr(apply_state_batch(base, X), anchor, y)
    c_ncl = _err_corr(apply_state_batch(ncl, X), anchor, y)
    assert np.isfinite(c_ncl)
    assert c_ncl < c_base - 0.02, f"NCL did not decorrelate: base={c_base:.3f} ncl={c_ncl:.3f}"


def test_ncl_lambda_zero_is_noop():
    X, y, anchor = _make_signal(seed=7)
    common = dict(
        labels=y, dense_X=X, hid1=16, hid2=8, epochs=6, batch_size=512,
        seed=9, show_progress=False,
    )
    a = fit_mlp_member(**common)
    b = fit_mlp_member(**common, ncl_anchor_preds=anchor, ncl_lambda=0.0)
    pa = apply_mlp_batch(a, dense_X=X)
    pb = apply_mlp_batch(b, dense_X=X)
    np.testing.assert_allclose(pa, pb, atol=1e-6)
