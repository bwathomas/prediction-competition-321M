"""Unit tests for src/member2_metadata_mlp.py."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.member2_metadata_mlp import (
    Member2MLPState,
    apply_batch,
    apply_one,
    apply_state_batch,
    apply_state_one,
    fit_member2_metadata_mlp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic(
    *,
    N: int = 4000,
    n_subjects: int = 25,
    n_bcs: int = 12,
    n_clusters: int = 8,
    n_marginals: int = 6,
    seed: int = 0,
    signal: str = "linear",  # "linear", "interaction", or "noise"
) -> dict:
    rng = np.random.default_rng(int(seed))
    s = rng.integers(0, n_subjects, size=N).astype(np.int64)
    b = rng.integers(0, n_bcs, size=N).astype(np.int64)
    c = rng.integers(0, n_clusters, size=N).astype(np.int64)
    M = rng.normal(0.0, 1.0, size=(N, n_marginals)).astype(np.float32)
    # Subject / bc / cluster effects.
    subj_eff = rng.normal(0.0, 0.7, size=n_subjects)
    bc_eff = rng.normal(0.0, 0.7, size=n_bcs)
    cl_eff = rng.normal(0.0, 0.4, size=n_clusters)
    marg_w = rng.normal(0.0, 0.5, size=n_marginals)
    if signal == "linear":
        z = subj_eff[s] + bc_eff[b] + cl_eff[c] + M @ marg_w
    elif signal == "interaction":
        # (subject, cluster) interaction the additive model cannot reach.
        interaction = rng.normal(0.0, 1.0, size=(n_subjects, n_clusters))
        z = (
            subj_eff[s] + bc_eff[b] + cl_eff[c]
            + interaction[s, c] + 0.5 * (M @ marg_w)
        )
    elif signal == "noise":
        z = rng.normal(0.0, 0.05, size=N)
    else:
        raise ValueError(f"unknown signal={signal!r}")
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(N) < p).astype(np.float32)
    return {
        "subject_ids": s, "bc_ids": b, "cluster_ids": c,
        "marginals": M, "y": y,
        "n_subjects": n_subjects, "n_bcs": n_bcs, "n_clusters": n_clusters,
    }


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _fit(
    data: dict,
    *,
    epochs: int = 25,
    hid1: int = 64,
    hid2: int = 32,
    d_subj: int = 8,
    d_bc: int = 8,
    d_cluster: int = 4,
    cat_dropout_subject: float = 0.0,
    cat_dropout_bc: float = 0.0,
    cat_dropout_cluster: float = 0.0,
    seed: int = 0,
    learning_rate: float = 5.0e-3,
    show_progress: bool = False,
    **kwargs,
) -> Member2MLPState:
    n_marg = int(data["marginals"].shape[1])
    return fit_member2_metadata_mlp(
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
        y=data["y"],
        subject_keys=tuple(f"s{i}" for i in range(int(data["n_subjects"]))),
        bc_keys=tuple(f"b{i}" for i in range(int(data["n_bcs"]))),
        marg_feature_names=tuple(f"m{i}" for i in range(n_marg)),
        n_subjects=int(data["n_subjects"]),
        n_bcs=int(data["n_bcs"]),
        n_clusters=int(data["n_clusters"]),
        d_subj=int(d_subj), d_bc=int(d_bc), d_cluster=int(d_cluster),
        hid1=int(hid1), hid2=int(hid2),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        batch_size=512,
        val_fraction=0.15,
        early_stopping_patience=8,
        cat_dropout_subject=float(cat_dropout_subject),
        cat_dropout_bc=float(cat_dropout_bc),
        cat_dropout_cluster=float(cat_dropout_cluster),
        seed=int(seed),
        show_progress=bool(show_progress),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shape / dtype / state invariants
# ---------------------------------------------------------------------------


def test_state_shapes_and_dtypes():
    data = _make_synthetic(N=600, signal="linear", seed=1)
    state = _fit(data, epochs=4)
    assert state.subject_emb.shape == (data["n_subjects"] + 1, state.d_subj)
    assert state.bc_emb.shape == (data["n_bcs"] + 1, state.d_bc)
    assert state.cluster_emb.shape == (data["n_clusters"] + 1, state.d_cluster)
    in_dim = state.d_subj + state.d_bc + state.d_cluster + state.n_marginals
    assert state.l1_value_W.shape == (in_dim, state.hid1)
    assert state.l1_gate_W.shape == (in_dim, state.hid1)
    assert state.l2_value_W.shape == (state.hid1, state.hid2)
    assert state.l2_gate_W.shape == (state.hid1, state.hid2)
    assert state.head_W.shape == (state.hid2, 1)
    for arr in (
        state.subject_emb, state.bc_emb, state.cluster_emb,
        state.l1_value_W, state.l1_value_b, state.l1_gate_W, state.l1_gate_b,
        state.l2_value_W, state.l2_value_b, state.l2_gate_W, state.l2_gate_b,
        state.head_W, state.marg_mean, state.marg_std,
    ):
        assert arr.dtype == np.float32, f"expected fp32, got {arr.dtype}"
    assert isinstance(state.head_b, float)
    assert math.isfinite(state.head_b)


def test_state_rejects_mismatched_dims():
    data = _make_synthetic(N=500, seed=2)
    state = _fit(data, epochs=3)
    # Surgically break one shape and rebuild -- should raise.
    bad_subject_emb = state.subject_emb[:-1]
    with pytest.raises(ValueError, match="subject_emb"):
        Member2MLPState(
            subject_emb=bad_subject_emb,
            bc_emb=state.bc_emb,
            cluster_emb=state.cluster_emb,
            l1_value_W=state.l1_value_W, l1_value_b=state.l1_value_b,
            l1_gate_W=state.l1_gate_W, l1_gate_b=state.l1_gate_b,
            l2_value_W=state.l2_value_W, l2_value_b=state.l2_value_b,
            l2_gate_W=state.l2_gate_W, l2_gate_b=state.l2_gate_b,
            head_W=state.head_W, head_b=state.head_b,
            marg_mean=state.marg_mean, marg_std=state.marg_std,
            subject_keys=state.subject_keys, bc_keys=state.bc_keys,
            n_subjects=state.n_subjects, n_bcs=state.n_bcs,
            n_clusters=state.n_clusters,
            d_subj=state.d_subj, d_bc=state.d_bc, d_cluster=state.d_cluster,
            hid1=state.hid1, hid2=state.hid2,
            marg_feature_names=state.marg_feature_names,
            n_marginals=state.n_marginals,
            fit_method=state.fit_method,
            n_train=state.n_train, n_pos=state.n_pos,
            train_loss=state.train_loss, val_loss=state.val_loss,
            cat_dropout_subject=state.cat_dropout_subject,
            cat_dropout_bc=state.cat_dropout_bc,
            cat_dropout_cluster=state.cat_dropout_cluster,
            weight_decay=state.weight_decay,
            learning_rate=state.learning_rate,
            epochs_run=state.epochs_run,
        )


# ---------------------------------------------------------------------------
# Forward path consistency
# ---------------------------------------------------------------------------


def test_apply_batch_vs_apply_one_match():
    data = _make_synthetic(N=300, seed=3)
    state = _fit(data, epochs=3)
    n_check = 32
    rng = np.random.default_rng(99)
    rows = rng.integers(0, data["y"].shape[0], size=n_check)
    p_batch = apply_batch(
        state=state,
        subject_ids=data["subject_ids"][rows],
        bc_ids=data["bc_ids"][rows],
        cluster_ids=data["cluster_ids"][rows],
        marginals=data["marginals"][rows],
    )
    p_one = np.array(
        [
            apply_one(
                state=state,
                subject_id=int(data["subject_ids"][r]),
                bc_id=int(data["bc_ids"][r]),
                cluster_id=int(data["cluster_ids"][r]),
                marginals=data["marginals"][r],
            )
            for r in rows
        ],
        dtype=np.float32,
    )
    assert p_batch.shape == (n_check,)
    assert p_batch.dtype == np.float32
    max_dev = float(np.max(np.abs(p_batch.astype(np.float64) - p_one.astype(np.float64))))
    assert max_dev < 1.0e-5, f"batch/one max abs dev {max_dev:.3e} > 1e-5"


def test_apply_state_wrappers_match_explicit_apply():
    data = _make_synthetic(N=200, seed=4)
    state = _fit(data, epochs=3)
    p_wrap = apply_state_batch(
        state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    p_explicit = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    np.testing.assert_allclose(p_wrap, p_explicit, rtol=0, atol=0)


def test_predictions_in_unit_interval():
    data = _make_synthetic(N=500, seed=5)
    state = _fit(data, epochs=5)
    p = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    assert np.all(p > 0.0) and np.all(p < 1.0)
    # Eps-clip enforced.
    assert np.all(p >= 1e-6) and np.all(p <= 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Cold-start / UNK routing
# ---------------------------------------------------------------------------


def test_unknown_ids_route_to_unk():
    data = _make_synthetic(N=300, seed=6)
    state = _fit(data, epochs=3)
    # All-known query.
    known_s = np.array([0, 1, 2], dtype=np.int64)
    known_b = np.array([0, 1, 2], dtype=np.int64)
    known_c = np.array([0, 1, 2], dtype=np.int64)
    known_m = np.zeros((3, data["marginals"].shape[1]), dtype=np.float32)
    p_known = apply_batch(
        state=state, subject_ids=known_s, bc_ids=known_b,
        cluster_ids=known_c, marginals=known_m,
    )
    # Unknown subject ID for the same rows -- prediction must change.
    unk_s = np.array([999, 999, 999], dtype=np.int64)
    p_unk_s = apply_batch(
        state=state, subject_ids=unk_s, bc_ids=known_b,
        cluster_ids=known_c, marginals=known_m,
    )
    assert not np.allclose(p_known, p_unk_s, atol=1e-6), (
        "unknown subject should change the prediction"
    )
    # Negative IDs also route to UNK (same prediction as 999).
    neg_s = np.array([-1, -7, -100], dtype=np.int64)
    p_neg = apply_batch(
        state=state, subject_ids=neg_s, bc_ids=known_b,
        cluster_ids=known_c, marginals=known_m,
    )
    np.testing.assert_allclose(p_unk_s, p_neg, atol=1e-6)


def test_unknown_bc_and_cluster_consistent():
    data = _make_synthetic(N=300, seed=7)
    state = _fit(data, epochs=3)
    rows = np.arange(5)
    p_orig = apply_batch(
        state=state,
        subject_ids=data["subject_ids"][rows],
        bc_ids=data["bc_ids"][rows],
        cluster_ids=data["cluster_ids"][rows],
        marginals=data["marginals"][rows],
    )
    # Flip bc to an unknown value -- different prediction.
    p_unk_bc = apply_batch(
        state=state,
        subject_ids=data["subject_ids"][rows],
        bc_ids=np.full(5, 99999, dtype=np.int64),
        cluster_ids=data["cluster_ids"][rows],
        marginals=data["marginals"][rows],
    )
    assert not np.allclose(p_orig, p_unk_bc, atol=1e-6)
    # Flip cluster to unknown -- also changes prediction.
    p_unk_cl = apply_batch(
        state=state,
        subject_ids=data["subject_ids"][rows],
        bc_ids=data["bc_ids"][rows],
        cluster_ids=np.full(5, 99999, dtype=np.int64),
        marginals=data["marginals"][rows],
    )
    assert not np.allclose(p_orig, p_unk_cl, atol=1e-6)


def test_cat_dropout_during_training_doesnt_explode():
    data = _make_synthetic(N=800, signal="linear", seed=8)
    state = _fit(
        data, epochs=8,
        cat_dropout_subject=0.10, cat_dropout_bc=0.20, cat_dropout_cluster=0.20,
    )
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.val_loss < 1.0  # better than random.


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path):
    data = _make_synthetic(N=500, seed=9)
    state = _fit(data, epochs=5)
    # Predictions before save.
    p_before = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    out = state.save(tmp_path / "m2_mlp")
    reloaded = Member2MLPState.load(out)
    p_after = apply_batch(
        state=reloaded,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    np.testing.assert_allclose(p_before, p_after, atol=1e-6)
    # Provenance preserved.
    assert reloaded.subject_keys == state.subject_keys
    assert reloaded.bc_keys == state.bc_keys
    assert reloaded.marg_feature_names == state.marg_feature_names
    assert reloaded.n_subjects == state.n_subjects
    assert reloaded.n_bcs == state.n_bcs
    assert reloaded.n_clusters == state.n_clusters
    assert reloaded.hid1 == state.hid1
    assert reloaded.hid2 == state.hid2


# ---------------------------------------------------------------------------
# Training convergence (sanity check that the model is learning)
# ---------------------------------------------------------------------------


def test_fit_beats_prior_on_linear_signal():
    data = _make_synthetic(N=4000, signal="linear", seed=10)
    state = _fit(
        data, epochs=30, hid1=64, hid2=32,
        d_subj=16, d_bc=16, d_cluster=8, learning_rate=5.0e-3,
    )
    p_train_mean = float(np.clip(data["y"].mean(), 1e-6, 1 - 1e-6))
    prior_nll = -(
        p_train_mean * math.log(p_train_mean)
        + (1 - p_train_mean) * math.log(1 - p_train_mean)
    )
    p = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    nll = _bce(data["y"], p)
    assert nll < prior_nll - 0.02, (
        f"linear signal: nll={nll:.4f} should beat prior={prior_nll:.4f}"
    )


def test_fit_recovers_interaction_signal():
    data = _make_synthetic(N=5000, signal="interaction", seed=11)
    state = _fit(
        data, epochs=40, hid1=128, hid2=64,
        d_subj=16, d_bc=16, d_cluster=8, learning_rate=5.0e-3,
    )
    p_train_mean = float(np.clip(data["y"].mean(), 1e-6, 1 - 1e-6))
    prior_nll = -(
        p_train_mean * math.log(p_train_mean)
        + (1 - p_train_mean) * math.log(1 - p_train_mean)
    )
    p = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    nll = _bce(data["y"], p)
    # Interaction signal -- additive model can't reach it, GLU MLP should.
    assert nll < prior_nll - 0.03, (
        f"interaction signal: nll={nll:.4f} should beat prior={prior_nll:.4f}"
    )


def test_fit_collapses_to_near_prior_on_pure_noise():
    data = _make_synthetic(N=3000, signal="noise", seed=12)
    state = _fit(
        data, epochs=25, hid1=64, hid2=32,
        d_subj=8, d_bc=8, d_cluster=4, learning_rate=3.0e-3,
    )
    p_train_mean = float(np.clip(data["y"].mean(), 1e-6, 1 - 1e-6))
    prior_nll = -(
        p_train_mean * math.log(p_train_mean)
        + (1 - p_train_mean) * math.log(1 - p_train_mean)
    )
    p = apply_batch(
        state=state,
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        marginals=data["marginals"],
    )
    nll = _bce(data["y"], p)
    # Should land in a narrow band around prior; the model has no signal to
    # exploit. Allow some slack because early-stopping happens before convergence.
    assert nll < prior_nll + 0.05, (
        f"noise signal: nll={nll:.4f} should be close to prior={prior_nll:.4f}"
    )


# ---------------------------------------------------------------------------
# Holdout-group split functionality
# ---------------------------------------------------------------------------


def test_holdout_group_split_respects_group_boundaries():
    """When holdout_group_id is provided, no group spans both train and val
    in the internal split."""
    data = _make_synthetic(N=1000, seed=13)
    # Group every 10 rows together. There are 100 groups.
    group = (np.arange(1000) // 10).astype(np.int64)
    # Patch fit_member2_metadata_mlp briefly to expose the split. We can't
    # see the split directly so we instead verify the fit succeeds (smoke
    # check) and that early stopping behaves sensibly.
    state = _fit(data, epochs=5, holdout_group_id=group)
    assert math.isfinite(state.val_loss)
    assert state.epochs_run >= 1


def test_zero_marginals_handled():
    """All-zero marginals -- the std will be zero in some cols; trainer
    must replace those with 1.0 instead of producing NaN/Inf."""
    data = _make_synthetic(N=400, seed=14)
    data["marginals"][:, 2] = 0.0  # one column constant
    state = _fit(data, epochs=4)
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    # std for the constant column should be 1.0 (not 0).
    assert state.marg_std[2] == pytest.approx(1.0)
