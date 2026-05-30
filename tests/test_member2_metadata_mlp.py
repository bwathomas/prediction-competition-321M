"""Unit tests for src/member2_metadata_mlp.py (v2: dense + DCN-v2)."""
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
    assemble_numerical,
    fit_member2_metadata_mlp,
    numerical_feature_names,
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
    n_families: int = 7,
    n_macro_families: int = 4,
    n_organizations: int = 5,
    n_bench_topics: int = 9,
    n_subj_num: int = 2,
    n_bench_num: int = 2,
    n_marginals: int = 6,
    seed: int = 0,
    signal: str = "linear",  # "linear", "interaction", "cross", "noise"
) -> dict:
    rng = np.random.default_rng(int(seed))
    s = rng.integers(0, n_subjects, size=N).astype(np.int64)
    b = rng.integers(0, n_bcs, size=N).astype(np.int64)
    c = rng.integers(0, n_clusters, size=N).astype(np.int64)
    f = rng.integers(0, n_families, size=N).astype(np.int64)
    mf = rng.integers(0, n_macro_families, size=N).astype(np.int64)
    o = rng.integers(0, n_organizations, size=N).astype(np.int64)
    t = rng.integers(0, n_bench_topics, size=N).astype(np.int64)
    subj_num = rng.normal(0.0, 1.0, size=(N, n_subj_num)).astype(np.float32)
    bench_num = rng.normal(0.0, 1.0, size=(N, n_bench_num)).astype(np.float32)
    redact = (rng.random(N) < 0.2).astype(np.float32)
    marg = rng.normal(0.0, 1.0, size=(N, n_marginals)).astype(np.float32)
    numerical = assemble_numerical(
        subject_numerical=subj_num,
        bench_numerical=bench_num,
        bc_redacted_flag=redact,
        marginals=marg,
    )
    n_num = int(numerical.shape[1])
    num_names = numerical_feature_names(
        subj_num_names=tuple(f"sn{i}" for i in range(n_subj_num)),
        bench_num_names=tuple(f"bn{i}" for i in range(n_bench_num)),
        marginal_names=tuple(f"mg{i}" for i in range(n_marginals)),
    )
    # Effects for synthetic labels.
    subj_eff = rng.normal(0.0, 0.7, size=n_subjects)
    bc_eff = rng.normal(0.0, 0.7, size=n_bcs)
    cl_eff = rng.normal(0.0, 0.4, size=n_clusters)
    fam_eff = rng.normal(0.0, 0.3, size=n_families)
    org_eff = rng.normal(0.0, 0.3, size=n_organizations)
    marg_w = rng.normal(0.0, 0.5, size=n_marginals)
    if signal == "linear":
        z = (
            subj_eff[s] + bc_eff[b] + cl_eff[c] + fam_eff[f] + org_eff[o]
            + marg @ marg_w
        )
    elif signal == "interaction":
        interaction = rng.normal(0.0, 1.0, size=(n_subjects, n_clusters))
        z = (
            subj_eff[s] + bc_eff[b] + cl_eff[c]
            + interaction[s, c] + 0.5 * (marg @ marg_w)
        )
    elif signal == "cross":
        # Three-way (subj * cluster * bc) signal that DCN-v2 can
        # express directly but a plain MLP needs more capacity for.
        trip = rng.normal(0.0, 0.6, size=(n_subjects, n_clusters, n_bcs))
        z = trip[s, c, b] + 0.4 * (marg @ marg_w)
    elif signal == "noise":
        z = rng.normal(0.0, 0.05, size=N)
    else:
        raise ValueError(f"unknown signal={signal!r}")
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(N) < p).astype(np.float32)
    return {
        "subject_ids": s, "bc_ids": b, "cluster_ids": c,
        "family_ids": f, "macro_family_ids": mf,
        "organization_ids": o, "bench_topic_ids": t,
        "numerical": numerical, "y": y,
        "n_subjects": n_subjects, "n_bcs": n_bcs, "n_clusters": n_clusters,
        "n_families": n_families, "n_macro_families": n_macro_families,
        "n_organizations": n_organizations, "n_bench_topics": n_bench_topics,
        "n_subj_num": n_subj_num, "n_bench_num": n_bench_num,
        "n_marginals": n_marginals, "n_num": n_num,
        "num_feature_names": num_names,
    }


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _apply_kwargs(data: dict, rows: np.ndarray | slice | None = None) -> dict:
    if rows is None:
        idx = slice(None)
    else:
        idx = rows
    return {
        "subject_ids": data["subject_ids"][idx],
        "bc_ids": data["bc_ids"][idx],
        "cluster_ids": data["cluster_ids"][idx],
        "family_ids": data["family_ids"][idx],
        "macro_family_ids": data["macro_family_ids"][idx],
        "organization_ids": data["organization_ids"][idx],
        "bench_topic_ids": data["bench_topic_ids"][idx],
        "numerical": data["numerical"][idx],
    }


def _fit(
    data: dict,
    *,
    epochs: int = 25,
    hid1: int = 64,
    hid2: int = 32,
    d_subj: int = 8,
    d_bc: int = 8,
    d_cluster: int = 4,
    d_family: int = 4,
    d_macro: int = 4,
    d_org: int = 4,
    d_topic: int = 4,
    n_cross_layers: int = 2,
    cross_rank: int = 16,
    cat_dropout_subject: float = 0.0,
    cat_dropout_bc: float = 0.0,
    cat_dropout_cluster: float = 0.0,
    cat_dropout_family: float = 0.0,
    cat_dropout_macro: float = 0.0,
    cat_dropout_org: float = 0.0,
    cat_dropout_topic: float = 0.0,
    seed: int = 0,
    learning_rate: float = 5.0e-3,
    show_progress: bool = False,
    ema_decay: float = 0.0,
    snapshot_ensemble_k: int = 1,
    warmup_epochs: int = 0,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    **kwargs,
) -> Member2MLPState:
    return fit_member2_metadata_mlp(
        subject_ids=data["subject_ids"],
        bc_ids=data["bc_ids"],
        cluster_ids=data["cluster_ids"],
        family_ids=data["family_ids"],
        macro_family_ids=data["macro_family_ids"],
        organization_ids=data["organization_ids"],
        bench_topic_ids=data["bench_topic_ids"],
        numerical=data["numerical"],
        y=data["y"],
        subject_keys=tuple(f"s{i}" for i in range(int(data["n_subjects"]))),
        bc_keys=tuple(f"b{i}" for i in range(int(data["n_bcs"]))),
        num_feature_names=data["num_feature_names"],
        n_subjects=int(data["n_subjects"]),
        n_bcs=int(data["n_bcs"]),
        n_clusters=int(data["n_clusters"]),
        n_families=int(data["n_families"]),
        n_macro_families=int(data["n_macro_families"]),
        n_organizations=int(data["n_organizations"]),
        n_bench_topics=int(data["n_bench_topics"]),
        n_subj_num=int(data["n_subj_num"]),
        n_bench_num=int(data["n_bench_num"]),
        n_marginals=int(data["n_marginals"]),
        d_subj=int(d_subj), d_bc=int(d_bc), d_cluster=int(d_cluster),
        d_family=int(d_family), d_macro=int(d_macro),
        d_org=int(d_org), d_topic=int(d_topic),
        hid1=int(hid1), hid2=int(hid2),
        n_cross_layers=int(n_cross_layers), cross_rank=int(cross_rank),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        batch_size=512,
        val_fraction=0.15,
        early_stopping_patience=8,
        cat_dropout_subject=float(cat_dropout_subject),
        cat_dropout_bc=float(cat_dropout_bc),
        cat_dropout_cluster=float(cat_dropout_cluster),
        cat_dropout_family=float(cat_dropout_family),
        cat_dropout_macro=float(cat_dropout_macro),
        cat_dropout_org=float(cat_dropout_org),
        cat_dropout_topic=float(cat_dropout_topic),
        ema_decay=float(ema_decay),
        snapshot_ensemble_k=int(snapshot_ensemble_k),
        warmup_epochs=int(warmup_epochs),
        label_smoothing=float(label_smoothing),
        mixup_alpha=float(mixup_alpha),
        seed=int(seed),
        show_progress=bool(show_progress),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Numerical-channel assembly
# ---------------------------------------------------------------------------


def test_assemble_numerical_concat_order():
    N = 5
    subj = np.arange(N * 3).reshape(N, 3).astype(np.float32)
    bench = (np.arange(N * 2) + 100).reshape(N, 2).astype(np.float32)
    redact = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    marg = np.arange(N * 4).reshape(N, 4).astype(np.float32) + 1000.0
    out = assemble_numerical(
        subject_numerical=subj, bench_numerical=bench,
        bc_redacted_flag=redact, marginals=marg,
    )
    assert out.shape == (N, 3 + 2 + 1 + 4)
    np.testing.assert_allclose(out[:, :3], subj)
    np.testing.assert_allclose(out[:, 3:5], bench)
    np.testing.assert_allclose(out[:, 5], redact)
    np.testing.assert_allclose(out[:, 6:], marg)


def test_numerical_feature_names_order():
    names = numerical_feature_names(
        subj_num_names=("a", "b"),
        bench_num_names=("c",),
        marginal_names=("d", "e", "f"),
    )
    assert names == (
        "subj_num__a", "subj_num__b",
        "bench_num__c",
        "bc_redacted_flag",
        "marg__d", "marg__e", "marg__f",
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
    assert state.family_emb.shape == (data["n_families"] + 1, state.d_family)
    assert state.macro_emb.shape == (data["n_macro_families"] + 1, state.d_macro)
    assert state.org_emb.shape == (data["n_organizations"] + 1, state.d_org)
    assert state.topic_emb.shape == (data["n_bench_topics"] + 1, state.d_topic)
    d_in = state.d_in
    assert state.l1_value_W.shape == (d_in, state.hid1)
    assert state.l2_value_W.shape == (state.hid1, state.hid2)
    assert state.head_W.shape == (d_in + state.hid2, 1)
    assert len(state.cross_V) == state.n_cross_layers
    for li in range(state.n_cross_layers):
        assert state.cross_V[li].shape == (d_in, state.cross_rank)
        assert state.cross_U[li].shape == (state.cross_rank, d_in)
        assert state.cross_b[li].shape == (d_in,)
    for arr in (
        state.subject_emb, state.bc_emb, state.cluster_emb,
        state.family_emb, state.macro_emb, state.org_emb, state.topic_emb,
        state.l1_value_W, state.l1_value_b, state.l1_gate_W, state.l1_gate_b,
        state.l2_value_W, state.l2_value_b, state.l2_gate_W, state.l2_gate_b,
        state.head_W, state.num_mean, state.num_std,
    ):
        assert arr.dtype == np.float32, f"expected fp32, got {arr.dtype}"
    assert isinstance(state.head_b, float)
    assert math.isfinite(state.head_b)


def test_state_rejects_mismatched_subject_emb():
    data = _make_synthetic(N=400, seed=2)
    state = _fit(data, epochs=3)
    bad = state.subject_emb[:-1]
    with pytest.raises(ValueError, match="subject_emb"):
        # Rebuild with a deliberately-broken table.
        _ = Member2MLPState(
            subject_emb=bad,
            bc_emb=state.bc_emb,
            cluster_emb=state.cluster_emb,
            family_emb=state.family_emb,
            macro_emb=state.macro_emb,
            org_emb=state.org_emb,
            topic_emb=state.topic_emb,
            cross_V=list(state.cross_V),
            cross_U=list(state.cross_U),
            cross_b=list(state.cross_b),
            l1_value_W=state.l1_value_W, l1_value_b=state.l1_value_b,
            l1_gate_W=state.l1_gate_W, l1_gate_b=state.l1_gate_b,
            l2_value_W=state.l2_value_W, l2_value_b=state.l2_value_b,
            l2_gate_W=state.l2_gate_W, l2_gate_b=state.l2_gate_b,
            head_W=state.head_W, head_b=state.head_b,
            num_mean=state.num_mean, num_std=state.num_std,
            subject_keys=state.subject_keys, bc_keys=state.bc_keys,
            n_subjects=state.n_subjects, n_bcs=state.n_bcs,
            n_clusters=state.n_clusters,
            n_families=state.n_families,
            n_macro_families=state.n_macro_families,
            n_organizations=state.n_organizations,
            n_bench_topics=state.n_bench_topics,
            d_subj=state.d_subj, d_bc=state.d_bc, d_cluster=state.d_cluster,
            d_family=state.d_family, d_macro=state.d_macro,
            d_org=state.d_org, d_topic=state.d_topic,
            hid1=state.hid1, hid2=state.hid2,
            n_cross_layers=state.n_cross_layers, cross_rank=state.cross_rank,
            num_feature_names=state.num_feature_names,
            n_num=state.n_num,
            n_subj_num=state.n_subj_num, n_bench_num=state.n_bench_num,
            n_marginals=state.n_marginals,
            fit_method=state.fit_method,
            n_train=state.n_train, n_pos=state.n_pos,
            train_loss=state.train_loss, val_loss=state.val_loss,
            cat_dropout_subject=state.cat_dropout_subject,
            cat_dropout_bc=state.cat_dropout_bc,
            cat_dropout_cluster=state.cat_dropout_cluster,
            cat_dropout_family=state.cat_dropout_family,
            cat_dropout_macro=state.cat_dropout_macro,
            cat_dropout_org=state.cat_dropout_org,
            cat_dropout_topic=state.cat_dropout_topic,
            weight_decay=state.weight_decay,
            learning_rate=state.learning_rate,
            epochs_run=state.epochs_run,
            label_smoothing=state.label_smoothing,
            mixup_alpha=state.mixup_alpha,
            ema_decay=state.ema_decay,
            snapshot_ensemble_k=state.snapshot_ensemble_k,
            warmup_epochs=state.warmup_epochs,
        )


# ---------------------------------------------------------------------------
# Forward path consistency
# ---------------------------------------------------------------------------


def test_apply_batch_vs_apply_one_match():
    data = _make_synthetic(N=300, seed=3)
    state = _fit(data, epochs=3)
    n_check = 16
    rng = np.random.default_rng(99)
    rows = rng.integers(0, data["y"].shape[0], size=n_check)
    p_batch = apply_batch(state=state, **_apply_kwargs(data, rows))
    p_one = np.array(
        [
            apply_one(
                state=state,
                subject_id=int(data["subject_ids"][r]),
                bc_id=int(data["bc_ids"][r]),
                cluster_id=int(data["cluster_ids"][r]),
                family_id=int(data["family_ids"][r]),
                macro_family_id=int(data["macro_family_ids"][r]),
                organization_id=int(data["organization_ids"][r]),
                bench_topic_id=int(data["bench_topic_ids"][r]),
                numerical=data["numerical"][r],
            )
            for r in rows
        ],
        dtype=np.float32,
    )
    assert p_batch.shape == (n_check,)
    assert p_batch.dtype == np.float32
    max_dev = float(
        np.max(np.abs(p_batch.astype(np.float64) - p_one.astype(np.float64)))
    )
    assert max_dev < 1.0e-5, f"batch/one max abs dev {max_dev:.3e} > 1e-5"


def test_apply_state_wrappers_match_explicit_apply():
    data = _make_synthetic(N=200, seed=4)
    state = _fit(data, epochs=3)
    p_wrap = apply_state_batch(state, **_apply_kwargs(data))
    p_explicit = apply_batch(state=state, **_apply_kwargs(data))
    np.testing.assert_allclose(p_wrap, p_explicit, rtol=0, atol=0)


def test_predictions_in_unit_interval():
    data = _make_synthetic(N=500, seed=5)
    state = _fit(data, epochs=5)
    p = apply_batch(state=state, **_apply_kwargs(data))
    assert np.all(p > 0.0) and np.all(p < 1.0)
    assert np.all(p >= 1e-6) and np.all(p <= 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Cold-start / UNK routing
# ---------------------------------------------------------------------------


def test_unknown_ids_route_to_unk():
    data = _make_synthetic(N=300, seed=6)
    state = _fit(data, epochs=3)
    rows = np.arange(3)
    base = _apply_kwargs(data, rows)
    p_known = apply_batch(state=state, **base)
    # Unknown subject (positive out-of-range) and negative ID -- both
    # should route to UNK and produce identical predictions.
    unk_kwargs = dict(base)
    unk_kwargs["subject_ids"] = np.array([999, 999, 999], dtype=np.int64)
    p_unk = apply_batch(state=state, **unk_kwargs)
    neg_kwargs = dict(base)
    neg_kwargs["subject_ids"] = np.array([-1, -7, -100], dtype=np.int64)
    p_neg = apply_batch(state=state, **neg_kwargs)
    assert not np.allclose(p_known, p_unk, atol=1e-6)
    np.testing.assert_allclose(p_unk, p_neg, atol=1e-6)


def test_unknown_bench_topic_and_family_route_to_unk():
    data = _make_synthetic(N=300, seed=7)
    state = _fit(data, epochs=3)
    rows = np.arange(5)
    base = _apply_kwargs(data, rows)
    p_orig = apply_batch(state=state, **base)
    # Bench topic UNK.
    kw_topic = dict(base)
    kw_topic["bench_topic_ids"] = np.full(5, 99999, dtype=np.int64)
    p_unk_topic = apply_batch(state=state, **kw_topic)
    assert not np.allclose(p_orig, p_unk_topic, atol=1e-6)
    # Family UNK.
    kw_fam = dict(base)
    kw_fam["family_ids"] = np.full(5, 99999, dtype=np.int64)
    p_unk_fam = apply_batch(state=state, **kw_fam)
    assert not np.allclose(p_orig, p_unk_fam, atol=1e-6)


def test_cat_dropout_during_training_doesnt_explode():
    data = _make_synthetic(N=800, signal="linear", seed=8)
    state = _fit(
        data, epochs=8,
        cat_dropout_subject=0.10, cat_dropout_bc=0.20, cat_dropout_cluster=0.20,
        cat_dropout_family=0.10, cat_dropout_macro=0.10,
        cat_dropout_org=0.10, cat_dropout_topic=0.10,
    )
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.val_loss < 1.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path):
    data = _make_synthetic(N=500, seed=9)
    state = _fit(data, epochs=5)
    p_before = apply_batch(state=state, **_apply_kwargs(data))
    out = state.save(tmp_path / "m2_mlp")
    reloaded = Member2MLPState.load(out)
    p_after = apply_batch(state=reloaded, **_apply_kwargs(data))
    np.testing.assert_allclose(p_before, p_after, atol=1e-6)
    # Provenance preserved.
    assert reloaded.subject_keys == state.subject_keys
    assert reloaded.bc_keys == state.bc_keys
    assert reloaded.num_feature_names == state.num_feature_names
    assert reloaded.n_subjects == state.n_subjects
    assert reloaded.n_bcs == state.n_bcs
    assert reloaded.n_clusters == state.n_clusters
    assert reloaded.n_families == state.n_families
    assert reloaded.n_macro_families == state.n_macro_families
    assert reloaded.n_organizations == state.n_organizations
    assert reloaded.n_bench_topics == state.n_bench_topics
    assert reloaded.hid1 == state.hid1
    assert reloaded.hid2 == state.hid2
    assert reloaded.n_cross_layers == state.n_cross_layers
    assert reloaded.cross_rank == state.cross_rank


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
    p = apply_batch(state=state, **_apply_kwargs(data))
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
    p = apply_batch(state=state, **_apply_kwargs(data))
    nll = _bce(data["y"], p)
    assert nll < prior_nll - 0.03, (
        f"interaction signal: nll={nll:.4f} should beat prior={prior_nll:.4f}"
    )


def test_fit_recovers_dcn_cross_signal():
    """Triple (subject x cluster x bc) signal -- DCN-v2 should fit it."""
    data = _make_synthetic(
        N=6000, n_subjects=15, n_bcs=8, n_clusters=6, signal="cross", seed=12,
    )
    state = _fit(
        data, epochs=40, hid1=64, hid2=32,
        d_subj=8, d_bc=8, d_cluster=6, learning_rate=5.0e-3,
        n_cross_layers=2, cross_rank=12,
    )
    p_train_mean = float(np.clip(data["y"].mean(), 1e-6, 1 - 1e-6))
    prior_nll = -(
        p_train_mean * math.log(p_train_mean)
        + (1 - p_train_mean) * math.log(1 - p_train_mean)
    )
    p = apply_batch(state=state, **_apply_kwargs(data))
    nll = _bce(data["y"], p)
    # Modest gain expected: 15 * 8 * 6 = 720 cells over 6000 rows means
    # ~8 samples per (s, c, b) cell, which Bayesian-shrinks heavily.
    # Any positive improvement above prior means the cross tower is
    # contributing.
    assert nll < prior_nll - 0.005, (
        f"cross signal: nll={nll:.4f} should beat prior={prior_nll:.4f}"
    )


def test_fit_collapses_to_near_prior_on_pure_noise():
    data = _make_synthetic(N=3000, signal="noise", seed=13)
    state = _fit(
        data, epochs=25, hid1=64, hid2=32,
        d_subj=8, d_bc=8, d_cluster=4, learning_rate=3.0e-3,
    )
    p_train_mean = float(np.clip(data["y"].mean(), 1e-6, 1 - 1e-6))
    prior_nll = -(
        p_train_mean * math.log(p_train_mean)
        + (1 - p_train_mean) * math.log(1 - p_train_mean)
    )
    p = apply_batch(state=state, **_apply_kwargs(data))
    nll = _bce(data["y"], p)
    assert nll < prior_nll + 0.05, (
        f"noise signal: nll={nll:.4f} should be close to prior={prior_nll:.4f}"
    )


# ---------------------------------------------------------------------------
# Training-side tricks
# ---------------------------------------------------------------------------


def test_ema_and_snapshot_ensemble_train_finitely():
    data = _make_synthetic(N=1000, signal="linear", seed=14)
    state = _fit(
        data, epochs=10, hid1=64, hid2=32,
        ema_decay=0.99, snapshot_ensemble_k=3,
        warmup_epochs=1, label_smoothing=0.01, mixup_alpha=0.0,
    )
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.ema_decay == pytest.approx(0.99)
    assert state.snapshot_ensemble_k == 3
    assert state.warmup_epochs == 1
    assert state.label_smoothing == pytest.approx(0.01)


def test_mixup_on_numerical_does_not_break_training():
    data = _make_synthetic(N=800, signal="linear", seed=15)
    state = _fit(
        data, epochs=6, hid1=64, hid2=32, mixup_alpha=0.2,
    )
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.mixup_alpha == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Holdout-group + constant-column edge cases
# ---------------------------------------------------------------------------


def test_holdout_group_split_respects_group_boundaries():
    data = _make_synthetic(N=1000, seed=16)
    group = (np.arange(1000) // 10).astype(np.int64)
    state = _fit(data, epochs=5, holdout_group_id=group)
    assert math.isfinite(state.val_loss)
    assert state.epochs_run >= 1


def test_constant_numerical_column_handled():
    """A constant numerical column has std=0; the trainer must replace
    that with 1.0 instead of producing NaN/Inf."""
    data = _make_synthetic(N=400, seed=17)
    # Zero out the 3rd marginal column.
    n_subj_num = data["n_subj_num"]
    n_bench_num = data["n_bench_num"]
    constant_col = n_subj_num + n_bench_num + 1 + 2  # 3rd marginal
    data["numerical"][:, constant_col] = 5.0
    state = _fit(data, epochs=4)
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.num_std[constant_col] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Metadata-only feature mode (d_subj == d_bc == d_cluster == 0, n_marginals == 0)
# ---------------------------------------------------------------------------
#
# The notebook's CFG["member2_mlp"]["feature_mode"] = "metadata_only"
# mode forces the subject / bc / cluster embedding tables to width 0
# and feeds an empty marginals slab + always-zero bc_redacted_flag
# to the trainer. The model collapses to a pure-metadata predictor
# (family / macro / org / topic embeddings + subject_numerical +
# bench_numerical). These tests prove:
#
#   1. fit + apply + save/load all survive d=0 embedding tables and
#      n_marginals=0 without NaN, shape mismatch, or runtime crash.
#   2. The "metadata-only" composition is genuinely a strict subset
#      of the "full" composition (no smuggled-in non-metadata signal).
#   3. Standardisation of the always-zero bc_redacted column does not
#      poison the rest of the numerical channel.


def _make_metadata_only_synthetic(
    *,
    N: int = 4000,
    n_subjects: int = 25,
    n_bcs: int = 12,
    n_clusters: int = 8,
    n_families: int = 7,
    n_macro_families: int = 4,
    n_organizations: int = 5,
    n_bench_topics: int = 9,
    n_subj_num: int = 2,
    n_bench_num: int = 2,
    seed: int = 0,
) -> dict:
    """Synthetic data where the label is driven by family + topic +
    benchmark_num only. Subject / bc / cluster IDs and marginals are
    pure noise so we can confirm the metadata-only model still
    learns the metadata-driven signal."""
    rng = np.random.default_rng(int(seed))
    s = rng.integers(0, n_subjects, size=N).astype(np.int64)
    b = rng.integers(0, n_bcs, size=N).astype(np.int64)
    c = rng.integers(0, n_clusters, size=N).astype(np.int64)
    f = rng.integers(0, n_families, size=N).astype(np.int64)
    mf = rng.integers(0, n_macro_families, size=N).astype(np.int64)
    o = rng.integers(0, n_organizations, size=N).astype(np.int64)
    t = rng.integers(0, n_bench_topics, size=N).astype(np.int64)
    subj_num = rng.normal(0.0, 1.0, size=(N, n_subj_num)).astype(np.float32)
    bench_num = rng.normal(0.0, 1.0, size=(N, n_bench_num)).astype(np.float32)
    # Metadata-only numerical channel: no marginals, redact always 0.
    redact = np.zeros(N, dtype=np.float32)
    marg = np.zeros((N, 0), dtype=np.float32)
    numerical = assemble_numerical(
        subject_numerical=subj_num,
        bench_numerical=bench_num,
        bc_redacted_flag=redact,
        marginals=marg,
    )
    num_names = numerical_feature_names(
        subj_num_names=tuple(f"sn{i}" for i in range(n_subj_num)),
        bench_num_names=tuple(f"bn{i}" for i in range(n_bench_num)),
        marginal_names=(),
    )
    # Label depends on family, topic, and benchmark_num[0] only -- a
    # clean metadata signal that the M2-metadata-only model can fit
    # but a "subject_id + bc_id"-only model cannot.
    fam_eff = rng.normal(0.0, 0.9, size=n_families)
    topic_eff = rng.normal(0.0, 0.9, size=n_bench_topics)
    z = fam_eff[f] + topic_eff[t] + 0.6 * bench_num[:, 0]
    p_true = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    y = (rng.random(N) < p_true).astype(np.float32)
    return {
        "subject_ids": s, "bc_ids": b, "cluster_ids": c,
        "family_ids": f, "macro_family_ids": mf,
        "organization_ids": o, "bench_topic_ids": t,
        "numerical": numerical, "y": y,
        "n_subjects": n_subjects, "n_bcs": n_bcs, "n_clusters": n_clusters,
        "n_families": n_families, "n_macro_families": n_macro_families,
        "n_organizations": n_organizations, "n_bench_topics": n_bench_topics,
        "n_subj_num": n_subj_num, "n_bench_num": n_bench_num,
        "n_marginals": 0, "num_feature_names": num_names,
    }


def test_metadata_only_fit_and_apply_no_nan():
    """End-to-end: fit + apply with d_subj=d_bc=d_cluster=0 and
    n_marginals=0. The model has zero-width tables for those fields
    and the inference path must short-circuit to a [N, 0] slice
    rather than crashing."""
    data = _make_metadata_only_synthetic(N=2000, seed=21)
    state = _fit(
        data,
        epochs=12,
        d_subj=0, d_bc=0, d_cluster=0,
        d_family=8, d_macro=4, d_org=4, d_topic=8,
        cross_rank=8,
    )
    assert math.isfinite(state.train_loss)
    assert math.isfinite(state.val_loss)
    assert state.d_subj == 0
    assert state.d_bc == 0
    assert state.d_cluster == 0
    assert state.n_marginals == 0
    # The zero-width embedding tables still have to round-trip in
    # the state's shape contract (n_cat + 1 rows, 0 cols).
    assert state.subject_emb.shape == (int(data["n_subjects"]) + 1, 0)
    assert state.bc_emb.shape == (int(data["n_bcs"]) + 1, 0)
    assert state.cluster_emb.shape == (int(data["n_clusters"]) + 1, 0)
    # And inference must produce finite probabilities in (0, 1).
    p = apply_batch(state=state, **_apply_kwargs(data))
    assert p.shape == (int(data["y"].shape[0]),)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


def test_metadata_only_learns_metadata_signal():
    """A metadata-only model on data whose label is driven by
    family + topic + bench_num should beat the dataset prior. If
    the test fails the d=0 path is dropping the family/topic/
    bench_num signal somewhere along the train loop."""
    data = _make_metadata_only_synthetic(N=4000, seed=22)
    prior = float(data["y"].mean())
    prior_nll = float(
        -(prior * math.log(max(prior, 1e-9))
          + (1.0 - prior) * math.log(max(1.0 - prior, 1e-9)))
    )
    state = _fit(
        data,
        epochs=25,
        d_subj=0, d_bc=0, d_cluster=0,
        d_family=12, d_macro=4, d_org=4, d_topic=12,
        cross_rank=16,
        learning_rate=1.0e-2,
    )
    p = apply_batch(state=state, **_apply_kwargs(data))
    nll = _bce(data["y"], p)
    assert nll < prior_nll - 0.02, (
        f"metadata-only model: nll={nll:.4f} should beat prior="
        f"{prior_nll:.4f} by >0.02 (the label is metadata-driven)"
    )


def test_metadata_only_save_load_roundtrip(tmp_path):
    data = _make_metadata_only_synthetic(N=1500, seed=23)
    state = _fit(
        data,
        epochs=8,
        d_subj=0, d_bc=0, d_cluster=0,
        d_family=8, d_macro=4, d_org=4, d_topic=8,
        cross_rank=8,
    )
    out = tmp_path / "m2_meta_only"
    state.save(out)
    state2 = Member2MLPState.load(out)
    # Width-0 tables must survive the npz round-trip with shape intact.
    assert state2.subject_emb.shape == state.subject_emb.shape
    assert state2.bc_emb.shape == state.bc_emb.shape
    assert state2.cluster_emb.shape == state.cluster_emb.shape
    assert state2.n_marginals == 0
    # Forward pass equivalence.
    p1 = apply_batch(state=state, **_apply_kwargs(data))
    p2 = apply_batch(state=state2, **_apply_kwargs(data))
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_metadata_only_subject_id_has_no_effect():
    """If d_subj=0, swapping every subject id at apply time must
    change *nothing* in the predictions (the subject embedding has
    no parameters to contribute). This is the structural guarantee
    that justifies calling the mode "metadata-only" -- the model
    cannot anchor on subject identity even if asked to."""
    data = _make_metadata_only_synthetic(N=800, seed=24)
    state = _fit(
        data,
        epochs=4,
        d_subj=0, d_bc=0, d_cluster=0,
        d_family=4, d_macro=2, d_org=2, d_topic=4,
        cross_rank=4,
    )
    base_kwargs = _apply_kwargs(data)
    p_orig = apply_batch(state=state, **base_kwargs)
    # Replace every subject id with 0; with d_subj=0 the embedding
    # output is [N, 0] regardless of input so the predictions must
    # be byte-identical to the original.
    shuffled_kwargs = dict(base_kwargs)
    shuffled_kwargs["subject_ids"] = np.zeros_like(base_kwargs["subject_ids"])
    p_shuffled = apply_batch(state=state, **shuffled_kwargs)
    np.testing.assert_array_equal(p_orig, p_shuffled)
    # Same for bc_id and cluster_id.
    shuffled_kwargs2 = dict(base_kwargs)
    shuffled_kwargs2["bc_ids"] = np.full_like(base_kwargs["bc_ids"], -1)
    shuffled_kwargs2["cluster_ids"] = np.full_like(
        base_kwargs["cluster_ids"], -1
    )
    p_shuffled2 = apply_batch(state=state, **shuffled_kwargs2)
    np.testing.assert_array_equal(p_orig, p_shuffled2)
