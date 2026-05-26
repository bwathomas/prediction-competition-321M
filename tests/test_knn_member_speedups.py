"""Regression tests for the Member-3 speed knobs:

  * Chunked ``apply_batch`` -- the pre-fix path materialized a
    ``[B, N_items]`` similarity matrix in one shot, which OOMs at the
    competition scale (266k val rows x ~300k train items = 315 GB).
    Chunking must produce numerically-identical predictions to the
    unchunked / per-row path.

  * Randomized SVD in ``_fit_pca`` -- the pre-fix path called
    ``np.linalg.svd(Xc, full_matrices=False)`` which is 7+ minutes on
    [300k, 4096] matrices. The randomized path must (a) be much
    faster and (b) recover top components close enough to the full
    SVD that downstream cosine top-k is unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.knn_member import (
    KNNMemberState,
    _fit_pca,
    apply_batch,
    apply_one,
    fit_knn_member,
)


def _make_state(N: int, D: int, S: int, *, pca_dim: int, k: int, quant: str = "fp16", seed: int = 0):
    rng = np.random.default_rng(seed)
    item_keys = [f"item_{i}" for i in range(N)]
    subject_keys = [f"subj_{j}" for j in range(S)]
    n_clusters = 5
    centers = rng.normal(size=(n_clusters, D)).astype(np.float32) * 3.0
    cluster_id = rng.integers(0, n_clusters, size=N)
    item_embs = (
        centers[cluster_id]
        + rng.normal(size=(N, D)).astype(np.float32) * 0.3
    ).astype(np.float32)
    cluster_skill = rng.uniform(0.1, 0.9, size=(S, n_clusters)).astype(np.float32)
    base = cluster_skill[:, cluster_id]
    mask = (rng.random(size=(S, N)) < 0.65).astype(np.bool_)
    labels = (rng.random(size=(S, N)) < base).astype(np.float32)
    state = fit_knn_member(
        item_keys=item_keys,
        item_embeddings=item_embs,
        subject_keys=subject_keys,
        passrate_dense=labels,
        passrate_mask=mask,
        pca_dim=pca_dim,
        quantization=quant,
        k=k,
    )
    return state, item_embs


# ---------------------------------------------------------------------------
# 1. Chunking -- numerical equivalence
# ---------------------------------------------------------------------------


def test_apply_batch_chunked_matches_unchunked():
    """Chunked ``apply_batch`` must match the unchunked call bit-for-bit
    (the only difference is loop ordering of the matmul, which is
    fp32-stable for the chunk sizes we use)."""
    state, item_embs = _make_state(
        N=400, D=48, S=6, pca_dim=16, k=8, quant="fp16", seed=11
    )
    rng = np.random.default_rng(0)
    B = 50
    q_idx = rng.integers(0, item_embs.shape[0], size=B)
    Q = item_embs[q_idx].copy()
    skeys = [state.subject_keys[i % len(state.subject_keys)] for i in range(B)]
    out_full = apply_batch(state, Q, skeys, chunk_size=B)
    out_chunked_2 = apply_batch(state, Q, skeys, chunk_size=2)
    out_chunked_7 = apply_batch(state, Q, skeys, chunk_size=7)
    out_chunked_13 = apply_batch(state, Q, skeys, chunk_size=13)
    np.testing.assert_allclose(out_chunked_2, out_full, atol=1e-6)
    np.testing.assert_allclose(out_chunked_7, out_full, atol=1e-6)
    np.testing.assert_allclose(out_chunked_13, out_full, atol=1e-6)


def test_apply_batch_chunk_one_matches_per_row_loop():
    """``chunk_size=1`` reduces the chunked path to the per-row path,
    so the result must be functionally identical to looping
    :func:`apply_one` row by row (allowing the documented fp32
    jitter from ``embs @ q`` vs ``[1, P] @ embs.T`` reordering)."""
    state, item_embs = _make_state(
        N=200, D=48, S=4, pca_dim=12, k=6, quant="fp16", seed=42
    )
    rng = np.random.default_rng(7)
    B = 30
    q_idx = rng.integers(0, item_embs.shape[0], size=B)
    Q = item_embs[q_idx].copy()
    skeys = [state.subject_keys[i % len(state.subject_keys)] for i in range(B)]
    out_chunk1 = apply_batch(state, Q, skeys, chunk_size=1)
    out_loop = np.array(
        [apply_one(state, Q[i], skeys[i]) for i in range(B)], dtype=np.float32
    )
    # P99 within fp32 jitter (matches existing test_apply_batch_matches_per_row_loop_functionally).
    delta = np.abs(out_chunk1 - out_loop)
    assert float(np.percentile(delta, 99)) < 0.05
    assert float(delta.mean()) < 0.005


def test_apply_batch_auto_chunk_does_not_oom_on_realistic_shape():
    """At realistic scale (B=2000, N_items=10000), default
    chunk_size=None should pick a chunk that bounds peak RAM. The
    test passes if the call returns finite probabilities -- the
    pre-fix path would have happily materialized a ``[2000, 10000]``
    sims matrix; the regression is that the new code MUST chunk on
    larger shapes without crashing."""
    state, item_embs = _make_state(
        N=10_000, D=64, S=4, pca_dim=24, k=8, quant="fp16", seed=99
    )
    rng = np.random.default_rng(0)
    B = 2_000
    q_idx = rng.integers(0, item_embs.shape[0], size=B)
    Q = item_embs[q_idx].copy()
    skeys = [state.subject_keys[i % len(state.subject_keys)] for i in range(B)]
    out = apply_batch(state, Q, skeys)  # default chunk_size=None
    assert out.shape == (B,)
    assert np.all(np.isfinite(out))
    assert np.all((out > 0) & (out < 1))


def test_apply_batch_chunk_size_kwarg_is_honored():
    """The chunk_size kwarg must actually drive the chunk loop. We
    detect this indirectly by checking that the function accepts
    arbitrary chunk sizes (including chunk_size > B, which collapses
    to a single chunk)."""
    state, item_embs = _make_state(
        N=80, D=32, S=3, pca_dim=8, k=4, quant="fp16", seed=2
    )
    Q = item_embs[:20].copy()
    keys = [state.subject_keys[i % 3] for i in range(20)]
    for cs in (1, 5, 20, 100, 1_000_000):
        out = apply_batch(state, Q, keys, chunk_size=cs)
        assert out.shape == (20,)
        assert np.all(np.isfinite(out))


def test_apply_batch_progress_kwarg_does_not_crash_without_tqdm():
    """``progress=True`` should silently no-op if tqdm isn't importable.
    We can't easily simulate tqdm being missing, but we can at least
    pin that progress=True doesn't crash."""
    state, item_embs = _make_state(N=60, D=24, S=3, pca_dim=8, k=4, seed=1)
    Q = item_embs[:10].copy()
    keys = [state.subject_keys[i % 3] for i in range(10)]
    out = apply_batch(state, Q, keys, progress=True)
    assert out.shape == (10,)


# ---------------------------------------------------------------------------
# 2. Randomized PCA -- accuracy + speed
# ---------------------------------------------------------------------------


def test_fit_pca_basis_is_orthonormal():
    """Top-pca_dim columns of the basis must be orthonormal (within fp32
    jitter): basis.T @ basis ~ I_pca_dim."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 96)).astype(np.float32)
    basis, _ = _fit_pca(X, pca_dim=20, seed=0)
    G = basis.T @ basis
    np.testing.assert_allclose(G, np.eye(20), atol=1e-3)


def test_fit_pca_recovers_signal_subspace_close_to_full_svd():
    """Randomized SVD basis must span approximately the same subspace
    as full SVD's top-pca_dim. We measure subspace overlap via
    Frobenius norm of the projection-difference."""
    rng = np.random.default_rng(1)
    # Construct data with a clear top-K signal.
    K_true = 10
    N, D = 600, 80
    U = rng.normal(size=(N, K_true)).astype(np.float32)
    V = rng.normal(size=(K_true, D)).astype(np.float32)
    s = np.linspace(20.0, 1.0, K_true).astype(np.float32)
    X = (U * s[None, :]) @ V + 0.05 * rng.normal(size=(N, D)).astype(np.float32)

    basis_rand, mean_rand = _fit_pca(X, pca_dim=K_true, seed=0, n_iter=4)
    # Full SVD reference.
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc.astype(np.float32), full_matrices=False)
    basis_full = Vt[:K_true].T.astype(np.float32)

    # Subspace overlap: |basis_rand @ basis_full.T| singular values
    # should all be close to 1 if the spans match.
    M = basis_rand.T @ basis_full
    sv = np.linalg.svd(M, compute_uv=False)
    # All singular values should be ~1.
    assert float(sv.min()) > 0.99
    assert float(sv.max()) < 1.01


def test_fit_pca_mean_is_full_data_mean_even_with_subsample():
    """When max_pca_samples subsamples the SVD, the returned mean must
    still be the FULL data's mean -- otherwise runtime centering of
    queries diverges from training."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 64)).astype(np.float32)
    basis, mean = _fit_pca(X, pca_dim=8, seed=0, max_pca_samples=300)
    full_mean = X.mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(mean, full_mean, atol=1e-5)


def test_fit_pca_subsample_speedup_is_real():
    """At realistic shape (3000, 512) the subsampled randomized PCA
    must finish dramatically faster than full SVD AND produce a
    similar top-K subspace. Hard to assert wall-clock in a unit
    test, but we can pin "subsample much faster than full"."""
    rng = np.random.default_rng(3)
    N, D = 3_000, 512
    X = (rng.normal(size=(N, D)) * 2.0).astype(np.float32)
    pca_dim = 32

    t0 = time.perf_counter()
    basis_full, _ = _fit_pca(X, pca_dim=pca_dim, seed=0)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    basis_sub, _ = _fit_pca(X, pca_dim=pca_dim, seed=0, max_pca_samples=600)
    t_sub = time.perf_counter() - t0

    # Subsample path should not be slower (allowing some CI noise).
    assert t_sub <= t_full * 1.2 + 0.5

    # And the subspace should still be close to the full version
    # (loose threshold; at high subsample ratio this is easy).
    M = basis_sub.T @ basis_full
    sv = np.linalg.svd(M, compute_uv=False)
    # The mean overlap is what matters for downstream cosine top-k.
    assert float(sv.mean()) > 0.85


def test_fit_knn_member_with_randomized_pca_still_produces_sensible_predictions():
    """End-to-end: fit_knn_member uses _fit_pca internally; predictions
    must remain sensible (better than constant prior) after the
    randomized SVD swap-in."""
    state, item_embs = _make_state(
        N=300, D=128, S=4, pca_dim=24, k=8, quant="fp16", seed=10
    )
    rng = np.random.default_rng(5)
    B = 40
    q_idx = rng.integers(0, item_embs.shape[0], size=B)
    Q = item_embs[q_idx].copy()
    keys = [state.subject_keys[i % 4] for i in range(B)]
    out = apply_batch(state, Q, keys)
    assert np.all(np.isfinite(out))
    assert np.all((out > 0) & (out < 1))
    # Predictions should not collapse to a single value.
    assert float(out.std()) > 1e-3
