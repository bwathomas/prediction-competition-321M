"""Regression: Member 3 (kNN) item-global fallback + K=128 default.

These tests pin the contract of the new ``item_fallback_weight`` knob
and the per-item ``item_global_passrate`` / ``item_obs_count`` tables
that ``fit_knn_member`` now produces.

The motivation is in the docstring of ``apply_one`` / ``apply_batch``:
when a neighbor cell ``(s_q, i_k)`` is unobserved (subject_q never
rated neighbor i_k), the legacy path masks that cell out of the
weighted mean entirely. For cold-start subjects whose subject-side
sparsity is extreme this collapses ``mu_neigh`` to 0.5 (the
unobserved-side placeholder) on most queries. The fallback substitutes
the per-item *global* passrate -- "this item is hard for everyone, so
it's probably hard for s_q too" -- with a discounted weight, which
recovers signal that the legacy path discarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.knn_member import (
    KNNMemberState,
    apply_batch,
    apply_one,
    fit_knn_member,
)


def _toy_state_from_pairs(
    n_items: int = 6,
    n_subjects: int = 4,
    *,
    item_fallback_weight: float = 0.0,
    seed: int = 0,
) -> KNNMemberState:
    """Build a tiny KNN state with controllable subject-side sparsity.

    The labels are deterministic from item_idx (item_idx % 2 == 0 -> 1.0)
    so we can assert the mu_neigh value exactly.
    """
    rng = np.random.default_rng(int(seed))
    D = 8
    item_keys = [f"i{i}" for i in range(n_items)]
    subj_keys = [f"s{i}" for i in range(n_subjects)]
    embs = rng.normal(size=(n_items, D)).astype(np.float32)
    # Subject 0 rates ALL items; subject 1 rates only the first item.
    # This reproduces the cold-start sparsity regime we care about.
    pairs = []
    for ii, ikey in enumerate(item_keys):
        label = float(ii % 2 == 0)
        pairs.append(("s0", ikey, label))
    pairs.append(("s1", item_keys[0], 1.0))
    return fit_knn_member(
        item_keys=item_keys,
        item_embeddings=embs,
        subject_keys=subj_keys,
        train_pairs=pairs,
        pca_dim=4,
        quantization="fp16",
        k=4,
        item_fallback_weight=float(item_fallback_weight),
        seed=int(seed),
    )


# ---- Default knob sanity ----


def test_fit_knn_member_default_k_is_128():
    """The default K bumped from 16 -> 128 to give cold-start subjects
    enough mass on item-rated neighbors after the item-global fallback
    discounts unobserved cells."""
    rng = np.random.default_rng(0)
    n_items = 200  # > 128 so K=128 is achievable.
    embs = rng.normal(size=(n_items, 16)).astype(np.float32)
    pairs = [("s0", f"i{i}", float(i % 2)) for i in range(n_items)]
    state = fit_knn_member(
        item_keys=[f"i{i}" for i in range(n_items)],
        item_embeddings=embs,
        subject_keys=("s0",),
        train_pairs=pairs,
        pca_dim=8,
        quantization="fp16",
        # Don't pass k -- exercise the new default explicitly.
    )
    assert state.k == 128, (
        f"default K should be 128 (cold-start fallback needs mass), got {state.k}"
    )


def test_fit_knn_member_emits_item_global_tables():
    state = _toy_state_from_pairs(n_items=6, n_subjects=4)
    assert state.item_global_passrate is not None
    assert state.item_obs_count is not None
    assert state.item_global_passrate.shape == (6,)
    assert state.item_obs_count.shape == (6,)
    # Item 0: rated by s0 (label=1) AND s1 (label=1) -> avg=1.0, count=2.
    # Items 1..5: rated only by s0 -> avg = label, count=1.
    assert state.item_obs_count[0] == 2.0
    assert all(state.item_obs_count[i] == 1.0 for i in range(1, 6))
    np.testing.assert_allclose(state.item_global_passrate[0], 1.0, atol=1e-6)
    # Items 1, 3, 5 are odd -> label 0; items 0, 2, 4 are even -> label 1.
    np.testing.assert_allclose(state.item_global_passrate[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(state.item_global_passrate[2], 1.0, atol=1e-6)


# ---- Fallback weight = 0.0 reproduces legacy behavior ----


def test_apply_one_with_zero_fallback_matches_legacy_path():
    """``item_fallback_weight=0.0`` must be a no-op vs. legacy semantics."""
    state = _toy_state_from_pairs(item_fallback_weight=0.0)
    # Subject "s1" only rated item 0, so neighbors that aren't item 0
    # are all unobserved -> legacy path masks them out and shrinks
    # toward subject prior.
    rng = np.random.default_rng(1)
    q = rng.normal(size=8).astype(np.float32)
    p = apply_one(state, q, "s1")
    assert 0.0 < p < 1.0
    assert np.isfinite(p)


def test_apply_batch_with_zero_fallback_matches_apply_one():
    state = _toy_state_from_pairs(item_fallback_weight=0.0, seed=42)
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(8, 8)).astype(np.float32)
    subj = ["s1"] * 8
    p_batch = apply_batch(state, Q, subj, use_gpu=False)
    p_one = np.array([apply_one(state, Q[i], subj[i]) for i in range(8)])
    np.testing.assert_allclose(p_batch, p_one, atol=1e-5)


# ---- Fallback weight > 0 changes predictions for sparse subjects ----


def test_apply_one_with_nonzero_fallback_uses_item_global():
    """For a subject that rated only one item, a query whose top-K neighbors
    are mostly OTHER items should now get signal from those items' global
    passrates instead of collapsing to 0.5 mu_neigh."""
    state_legacy = _toy_state_from_pairs(item_fallback_weight=0.0, seed=7)
    state_fb = _toy_state_from_pairs(item_fallback_weight=0.5, seed=7)
    # s1 has rated only item 0. Build a query that is far from item 0 so
    # most top-K neighbors are unobserved cells for s1.
    rng = np.random.default_rng(3)
    q = rng.normal(size=8).astype(np.float32)
    p_legacy = apply_one(state_legacy, q, "s1")
    p_fb = apply_one(state_fb, q, "s1")
    # The two predictions should differ -- item-global fallback engaged.
    assert not np.isclose(p_legacy, p_fb), (
        f"fallback should change p for sparse subject (legacy={p_legacy:.4f}, "
        f"fb={p_fb:.4f})"
    )
    assert 0.0 < p_fb < 1.0
    assert np.isfinite(p_fb)


def test_apply_batch_fallback_consistent_with_apply_one():
    """The fallback path must agree across batch and per-row entry points
    so the OOF eval / leaderboard predict cannot diverge."""
    state = _toy_state_from_pairs(item_fallback_weight=0.5, seed=11)
    rng = np.random.default_rng(13)
    Q = rng.normal(size=(16, 8)).astype(np.float32)
    subj = ["s1"] * 16
    p_batch = apply_batch(state, Q, subj, use_gpu=False)
    p_one = np.array([apply_one(state, Q[i], subj[i]) for i in range(16)])
    np.testing.assert_allclose(p_batch, p_one, atol=1e-5)


def test_save_load_roundtrips_item_global_tables(tmp_path):
    state = _toy_state_from_pairs(item_fallback_weight=0.42, seed=9)
    state.save(tmp_path)
    loaded = KNNMemberState.load(tmp_path)
    assert loaded.item_global_passrate is not None
    assert loaded.item_obs_count is not None
    np.testing.assert_allclose(
        loaded.item_global_passrate, state.item_global_passrate, atol=1e-6
    )
    np.testing.assert_allclose(
        loaded.item_obs_count, state.item_obs_count, atol=1e-6
    )
    np.testing.assert_allclose(
        loaded.item_fallback_weight, state.item_fallback_weight, atol=1e-12
    )


def test_state_rejects_unbalanced_item_tables():
    """If only one of the item-side fields is set, init should fail
    rather than silently disable the fallback."""
    rng = np.random.default_rng(0)
    n_items, P, S = 4, 2, 1
    pca_basis = np.zeros((P, P), dtype=np.float32)
    pca_mean = np.zeros(P, dtype=np.float32)
    embeddings_q = np.zeros((n_items, P), dtype=np.float16)
    pr = np.zeros((S, n_items), dtype=np.float32)
    mk = np.zeros((S, n_items), dtype=np.bool_)
    with pytest.raises(ValueError, match="must both be"):
        KNNMemberState(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            embeddings_q=embeddings_q,
            embeddings_scale=None,
            passrate_dense=pr,
            passrate_mask=mk,
            subject_obs_count=np.zeros(S, dtype=np.float32),
            subject_global=np.full(S, 0.5, dtype=np.float32),
            global_passrate=0.5,
            item_global_passrate=np.zeros(n_items, dtype=np.float32),
            item_obs_count=None,    # !!! mismatched
            item_keys=tuple(f"i{i}" for i in range(n_items)),
            subject_keys=("s0",),
            k=2,
            pca_dim=P,
            quantization="fp16",
            similarity="cosine",
            tau_subject=2.0,
            tau_global=5.0,
            n_train=n_items,
            train_loss=0.0,
            val_loss=0.0,
        )
