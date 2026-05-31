"""Tests for the PCA tail-subspace utility (Members 4 & 5)."""

from __future__ import annotations

import numpy as np

from src.pca_tail import PcaTailBasis, fit_pca_tail


def _make_structured(n=2000, d=64, n_strong=4, seed=0):
    """Embeddings = a few high-variance axes + low-variance residual."""
    rng = np.random.default_rng(seed)
    strong = rng.normal(scale=10.0, size=(n, n_strong))
    strong_dirs = np.linalg.qr(rng.normal(size=(d, n_strong)))[0]
    weak = rng.normal(scale=0.5, size=(n, d))
    X = strong @ strong_dirs.T + weak
    return X.astype(np.float32), strong_dirs


def test_shapes_and_projection():
    X, _ = _make_structured()
    tail = fit_pca_tail(X, n_components=48, head_drop=8, tail_take=16, seed=1)
    assert tail.basis.shape == (X.shape[1], 16)
    proj = tail.project(X)
    assert proj.shape == (X.shape[0], 16)
    assert np.all(np.isfinite(proj))


def test_tail_drops_dominant_variance():
    # The tail subspace should carry far less variance per-dim than the
    # head (it deliberately excludes the dominant directions).
    X, _ = _make_structured(n_strong=4)
    tail = fit_pca_tail(X, n_components=40, head_drop=4, tail_take=16, seed=2)
    proj = tail.project(X)
    tail_var = proj.var(axis=0).mean()
    # Variance along the dropped strong directions is ~100; tail must be far smaller.
    assert tail_var < 5.0


def test_orthonormal_basis_columns():
    X, _ = _make_structured()
    tail = fit_pca_tail(X, n_components=48, head_drop=8, tail_take=16, seed=3)
    gram = tail.basis.T @ tail.basis
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-3)


def test_save_load_roundtrip(tmp_path):
    X, _ = _make_structured()
    tail = fit_pca_tail(X, n_components=48, head_drop=8, tail_take=16, seed=4)
    p0 = tail.project(X)
    tail.save(tmp_path / "tail")
    loaded = PcaTailBasis.load(tmp_path / "tail")
    p1 = loaded.project(X)
    assert np.max(np.abs(p0 - p1)) < 1e-5


def test_projection_matches_manual():
    X, _ = _make_structured(n=500, d=32)
    tail = fit_pca_tail(X, n_components=24, head_drop=4, tail_take=8, seed=5)
    manual = (X - tail.mean) @ tail.basis
    assert np.allclose(manual, tail.project(X), atol=1e-4)
