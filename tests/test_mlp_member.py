"""Tests for the generic GLU-MLP member (Members 7 & 8)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.mlp_member import (
    MlpMemberState,
    apply_batch,
    apply_one,
    fit_mlp_member,
)


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------------------------------------------------------------------
# M7-style: pure dense (marginals) channel
# ---------------------------------------------------------------------------


def _make_dense_synth(n=4000, f=14, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    # Non-linear signal: interaction of two columns + a main effect.
    logit = 1.5 * X[:, 0] * X[:, 1] - 0.8 * X[:, 2] + 0.3
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, y


def test_dense_only_fit_apply_no_nan():
    X, y = _make_dense_synth()
    state = fit_mlp_member(
        labels=y, dense_X=X, dense_feature_names=tuple(f"m{i}" for i in range(X.shape[1])),
        hid1=32, hid2=16, epochs=8, batch_size=512, seed=1, show_progress=False,
    )
    p = apply_batch(state, dense_X=X)
    assert p.shape == (X.shape[0],)
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Beats the class prior.
    prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(prior * np.log(prior) + (1 - prior) * np.log(1 - prior))
    assert _nll(p, y) < nll_prior


def test_dense_only_save_load_roundtrip(tmp_path):
    X, y = _make_dense_synth(n=1500)
    state = fit_mlp_member(
        labels=y, dense_X=X, hid1=16, hid2=8, epochs=4, batch_size=256,
        seed=2, show_progress=False,
    )
    p0 = apply_batch(state, dense_X=X)
    state.save(tmp_path / "m7")
    loaded = MlpMemberState.load(tmp_path / "m7")
    p1 = apply_batch(loaded, dense_X=X)
    assert np.max(np.abs(p0 - p1)) < 1e-6


# ---------------------------------------------------------------------------
# M8-style: subject embedding + item embedding channels
# ---------------------------------------------------------------------------


def _make_emb_synth(n=4000, n_subj=20, n_items=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    item_emb = rng.normal(size=(n_items, d)).astype(np.float32)
    subj_vec = rng.normal(size=(n_subj, d)).astype(np.float32)
    row_to_uniq = rng.integers(0, n_items, size=n).astype(np.int64)
    subject_ids = rng.integers(0, n_subj, size=n).astype(np.int64)
    # Signal: dot product of subject ability vector and item embedding.
    logit = (subj_vec[subject_ids] * item_emb[row_to_uniq]).sum(axis=1) * 0.5
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return item_emb, row_to_uniq, subject_ids, y, n_subj


def test_emb_fit_apply_no_nan_and_learns():
    item_emb, r2u, sids, y, n_subj = _make_emb_synth()
    state = fit_mlp_member(
        labels=y, subject_ids=sids, n_subjects=n_subj, subj_emb_dim=8,
        item_emb_unique=item_emb, row_to_uniq=r2u,
        hid1=32, hid2=16, epochs=12, batch_size=512, seed=3, show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p = apply_batch(state, subject_ids=sids, item_emb=per_row_item)
    assert p.shape == (y.shape[0],)
    assert np.all(np.isfinite(p))
    prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    nll_prior = -(prior * np.log(prior) + (1 - prior) * np.log(1 - prior))
    assert _nll(p, y) < nll_prior


def test_emb_save_load_roundtrip(tmp_path):
    item_emb, r2u, sids, y, n_subj = _make_emb_synth(n=1500)
    state = fit_mlp_member(
        labels=y, subject_ids=sids, n_subjects=n_subj, subj_emb_dim=8,
        item_emb_unique=item_emb, row_to_uniq=r2u,
        hid1=16, hid2=8, epochs=4, batch_size=256, seed=4, show_progress=False,
    )
    per_row_item = item_emb[r2u]
    p0 = apply_batch(state, subject_ids=sids, item_emb=per_row_item)
    state.save(tmp_path / "m8")
    loaded = MlpMemberState.load(tmp_path / "m8")
    p1 = apply_batch(loaded, subject_ids=sids, item_emb=per_row_item)
    assert np.max(np.abs(p0 - p1)) < 1e-6


def test_emb_unknown_subject_routes_to_unk():
    item_emb, r2u, sids, y, n_subj = _make_emb_synth(n=800)
    state = fit_mlp_member(
        labels=y, subject_ids=sids, n_subjects=n_subj, subj_emb_dim=8,
        item_emb_unique=item_emb, row_to_uniq=r2u,
        hid1=16, hid2=8, epochs=3, batch_size=256, seed=5, show_progress=False,
    )
    per_row_item = item_emb[r2u][:4]
    # Out-of-range subject id must not raise; routes to UNK slot.
    p = apply_batch(
        state,
        subject_ids=np.array([999999, -1, n_subj, n_subj + 3], dtype=np.int64),
        item_emb=per_row_item,
    )
    assert np.all(np.isfinite(p))


def test_apply_one_matches_batch():
    X, y = _make_dense_synth(n=600)
    state = fit_mlp_member(
        labels=y, dense_X=X, hid1=16, hid2=8, epochs=3, batch_size=256,
        seed=6, show_progress=False,
    )
    p_batch = apply_batch(state, dense_X=X[:5])
    for i in range(5):
        p1 = apply_one(state, dense_X=X[i])
        assert abs(p1 - float(p_batch[i])) < 1e-6
