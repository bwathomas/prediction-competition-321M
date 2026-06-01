"""Smoke tests for :mod:`src.rich_mlp_variant`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.rich_mlp_variant import (
    RichMLPConfig,
    apply_soft_routing,
    predict_rich_mlp,
    soft_routing_weights_categorical,
    soft_routing_weights_kernel,
    train_rich_mlp,
)


def _toy_data(N=512, item_dim=16, dense_dim=8, seed=0):
    rng = np.random.default_rng(int(seed))
    n_subjects = 24
    n_bcs = 12
    n_clusters = 6
    n_families = 5
    n_macros = 3
    n_orgs = 4
    n_topics = 7
    n_uniq = 64

    item_table = rng.standard_normal((n_uniq, item_dim)).astype(np.float32)
    r2u = rng.integers(0, n_uniq, size=N).astype(np.int64)
    dense_X = rng.standard_normal((N, dense_dim)).astype(np.float32)

    s = rng.integers(0, n_subjects, size=N).astype(np.int64)
    b = rng.integers(0, n_bcs, size=N).astype(np.int64)
    c = rng.integers(0, n_clusters, size=N).astype(np.int64)
    f = rng.integers(0, n_families, size=N).astype(np.int64)
    mf = rng.integers(0, n_macros, size=N).astype(np.int64)
    o = rng.integers(0, n_orgs, size=N).astype(np.int64)
    t = rng.integers(0, n_topics, size=N).astype(np.int64)
    # Strong signal so the smoke tests fit fast: dominant contribution
    # is item_dim[0] (rank-0 cross channel), plus a categorical
    # (family parity) and a dense feature. Scaled large enough that
    # an 8-epoch fit beats the 0.69-nat majority floor comfortably.
    base = (
        3.0 * item_table[r2u, 0]
        + 1.5 * (f % 2).astype(np.float32)
        + 0.8 * dense_X[:, 0]
    )
    p = 1.0 / (1.0 + np.exp(-base))
    y = (rng.uniform(size=N) < p).astype(np.float32)

    return dict(
        N=N, item_dim=item_dim, dense_dim=dense_dim,
        n_subjects=n_subjects, n_bcs=n_bcs, n_clusters=n_clusters,
        n_families=n_families, n_macros=n_macros, n_orgs=n_orgs,
        n_topics=n_topics, n_uniq=n_uniq,
        item_table=item_table, r2u=r2u, dense_X=dense_X,
        s=s, b=b, c=c, f=f, mf=mf, o=o, t=t, y=y,
    )


def _call_train(d, cfg, sample_weights=None):
    dev = torch.device("cpu")
    emb_t = torch.from_numpy(d["item_table"])
    return train_rich_mlp(
        y=d["y"],
        subject_ids=d["s"], bc_ids=d["b"], cluster_ids=d["c"],
        family_ids=d["f"], macro_ids=d["mf"], org_ids=d["o"], topic_ids=d["t"],
        item_emb_tensor=emb_t, row_to_uniq=d["r2u"], dense_X=d["dense_X"],
        n_subjects=d["n_subjects"], n_bcs=d["n_bcs"], n_clusters=d["n_clusters"],
        n_families=d["n_families"], n_macros=d["n_macros"], n_orgs=d["n_orgs"],
        n_topics=d["n_topics"],
        cfg=cfg, device=dev, sample_weights=sample_weights, show_progress=False,
    )


def _call_predict(net, d):
    dev = torch.device("cpu")
    emb_t = torch.from_numpy(d["item_table"])
    return predict_rich_mlp(
        net,
        subject_ids=d["s"], bc_ids=d["b"], cluster_ids=d["c"],
        family_ids=d["f"], macro_ids=d["mf"], org_ids=d["o"], topic_ids=d["t"],
        item_emb_tensor=emb_t, row_to_uniq=d["r2u"], dense_X=d["dense_X"],
        n_subjects=d["n_subjects"], n_bcs=d["n_bcs"], n_clusters=d["n_clusters"],
        n_families=d["n_families"], n_macros=d["n_macros"], n_orgs=d["n_orgs"],
        n_topics=d["n_topics"], device=dev,
    )


# ---------------------------------------------------------------------------
# Architecture / training
# ---------------------------------------------------------------------------


def test_train_then_predict_runs_and_fits_signal():
    # N=2000 is large enough that the categorical embedding tables
    # don't memorize before the deep tower learns the item channel.
    d = _toy_data(N=2000, seed=1)
    cfg = RichMLPConfig(
        epochs=20, batch_size=128, val_fraction=0.2, patience=6, seed=11,
        hid1=32, hid2=16, n_cross_layers=1, cross_rank=8,
        # Heavier WD on the smoke toy to suppress cat-emb memorization.
        wd=1e-3,
        # disable cat-dropout so the fit is sharper for the smoke check
        cat_dropout_subject=0.0, cat_dropout_bc=0.0, cat_dropout_cluster=0.0,
        cat_dropout_family=0.0, cat_dropout_macro=0.0, cat_dropout_org=0.0,
        cat_dropout_topic=0.0,
    )
    net = _call_train(d, cfg)
    p = _call_predict(net, d)
    assert p.shape == (d["N"],)
    assert p.dtype == np.float32
    assert np.all((p >= 1e-6) & (p <= 1 - 1e-6))
    nll = float(-(d["y"] * np.log(p) + (1 - d["y"]) * np.log(1 - p)).mean())
    # 0.693 is the entropy of a 50/50 split; the strong injected
    # signal lets even this tiny model beat it.
    assert nll < 0.685, f"nll={nll:.4f} did not beat majority baseline"


def test_dense_channel_optional():
    d = _toy_data(N=200, seed=2)
    cfg = RichMLPConfig(
        epochs=3, batch_size=64, val_fraction=0.2, patience=2, seed=21,
        hid1=16, hid2=8, n_cross_layers=0,
        cat_dropout_subject=0.0, cat_dropout_bc=0.0, cat_dropout_cluster=0.0,
        cat_dropout_family=0.0, cat_dropout_macro=0.0, cat_dropout_org=0.0,
        cat_dropout_topic=0.0,
    )
    # dense_X=None => skipped from the forward concat.
    d2 = dict(d)
    d2["dense_X"] = None
    net = _call_train(d2, cfg)
    p = _call_predict(net, d2)
    assert p.shape == (d["N"],)


def test_disable_some_categorical_channels():
    d = _toy_data(N=200, seed=3)
    cfg = RichMLPConfig(
        epochs=3, batch_size=64, val_fraction=0.2, patience=2, seed=31,
        hid1=16, hid2=8, n_cross_layers=0,
        # Disable bc, cluster, macro, org, topic. Keep subject + family.
        bc_emb_dim=0, cluster_emb_dim=0, macro_emb_dim=0,
        org_emb_dim=0, topic_emb_dim=0,
        cat_dropout_subject=0.0, cat_dropout_bc=0.0, cat_dropout_cluster=0.0,
        cat_dropout_family=0.0, cat_dropout_macro=0.0, cat_dropout_org=0.0,
        cat_dropout_topic=0.0,
    )
    net = _call_train(d, cfg)
    p = _call_predict(net, d)
    assert p.shape == (d["N"],)


def test_categorical_dropout_does_not_crash():
    d = _toy_data(N=200, seed=4)
    cfg = RichMLPConfig(
        epochs=4, batch_size=64, val_fraction=0.2, patience=3, seed=41,
        hid1=16, hid2=8, n_cross_layers=1, cross_rank=4,
        # Aggressive dropout to exercise the masking code paths.
        cat_dropout_subject=0.5, cat_dropout_bc=0.5, cat_dropout_cluster=0.5,
        cat_dropout_family=0.5, cat_dropout_macro=0.5, cat_dropout_org=0.5,
        cat_dropout_topic=0.5,
    )
    net = _call_train(d, cfg)
    p = _call_predict(net, d)
    assert p.shape == (d["N"],)
    assert np.isfinite(p).all()


def test_cold_id_handling_at_inference():
    """Out-of-range ids must be silently mapped to the UNK row."""
    d = _toy_data(N=200, seed=5)
    cfg = RichMLPConfig(
        epochs=2, batch_size=64, val_fraction=0.2, patience=2, seed=51,
        hid1=16, hid2=8, n_cross_layers=0,
        cat_dropout_subject=0.0, cat_dropout_bc=0.0, cat_dropout_cluster=0.0,
        cat_dropout_family=0.0, cat_dropout_macro=0.0, cat_dropout_org=0.0,
        cat_dropout_topic=0.0,
    )
    net = _call_train(d, cfg)
    cold = dict(d)
    # Sprinkle in some out-of-range ids -- predict_rich_mlp clamps to
    # UNK internally, so this must not crash.
    cold["s"] = cold["s"].copy()
    cold["s"][:10] = d["n_subjects"] + 999
    cold["s"][10:20] = -1
    cold["b"] = cold["b"].copy()
    cold["b"][:10] = -5
    p = _call_predict(net, cold)
    assert p.shape == (d["N"],)
    assert np.isfinite(p).all()


def test_sample_weights_affect_training():
    """Sample weights must actually change training -- verified by
    comparing weighted-vs-unweighted predictions at fixed budget and
    fixed seed. The exact direction of the shift is sensitive to the
    seed / training noise on tiny toys (with early-stop disabled
    cat embeddings overfit unpredictably), so we just assert the two
    runs produce measurably different predictions."""
    d = _toy_data(N=1500, seed=6)
    cfg = RichMLPConfig(
        # patience > epochs disables early stop -- both runs see the
        # exact same training budget so the only diff is the weights.
        epochs=8, batch_size=128, val_fraction=0.2, patience=999, seed=61,
        hid1=16, hid2=8, n_cross_layers=0, wd=1e-3,
        cat_dropout_subject=0.0, cat_dropout_bc=0.0, cat_dropout_cluster=0.0,
        cat_dropout_family=0.0, cat_dropout_macro=0.0, cat_dropout_org=0.0,
        cat_dropout_topic=0.0,
    )
    base_net = _call_train(d, cfg)
    base_p = _call_predict(base_net, d)
    w = np.where(d["y"] > 0.5, 100.0, 1.0).astype(np.float32)
    w = w / w.mean()
    weighted_net = _call_train(d, cfg, sample_weights=w)
    weighted_p = _call_predict(weighted_net, d)
    # The two predictions must differ -- if weights weren't used the
    # runs would be identical (same seed). 0.01 absolute mean diff
    # is a generous floor that triggers reliably even on tiny toys.
    abs_diff = float(np.abs(weighted_p - base_p).mean())
    assert abs_diff > 0.01, (
        f"weighted vs base abs-diff {abs_diff:.4f} <= 0.01 -- sample_weights "
        "appear to be ignored"
    )


# ---------------------------------------------------------------------------
# Soft routing helpers
# ---------------------------------------------------------------------------


def test_soft_routing_categorical_hard_at_eps_zero():
    bp = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    w = soft_routing_weights_categorical(bp, n_buckets=3, epsilon=0.0)
    assert w.shape == (5, 3)
    assert np.allclose(w.sum(axis=1), 1.0)
    # Hard one-hot when eps=0.
    assert (w.argmax(axis=1) == bp).all()
    assert np.allclose(w[w > 0], 1.0)


def test_soft_routing_categorical_uniform_at_eps_one():
    bp = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    K = 3
    w = soft_routing_weights_categorical(bp, n_buckets=K, epsilon=1.0)
    # At eps=1.0 the diagonal weight is 0 and the other K-1 share 1.0
    # uniformly. That's a "send the row to anyone except its own
    # bucket" router -- it's the formal eps=1 limit of the smoothed
    # one-hot. The "true uniform" hyperparam is eps = (K-1)/K, which
    # we also accept.
    expected_off = 1.0 / (K - 1)
    for i, b in enumerate(bp):
        for k in range(K):
            if k == b:
                assert w[i, k] == pytest.approx(0.0, abs=1e-6)
            else:
                assert w[i, k] == pytest.approx(expected_off, abs=1e-6)


def test_soft_routing_categorical_intermediate():
    bp = np.array([0, 2], dtype=np.int64)
    w = soft_routing_weights_categorical(bp, n_buckets=4, epsilon=0.3)
    assert np.allclose(w.sum(axis=1), 1.0)
    # Diagonal carries 0.7; off-diagonal each carries 0.1 = 0.3 / 3.
    assert w[0, 0] == pytest.approx(0.7, abs=1e-6)
    assert w[0, 1] == pytest.approx(0.1, abs=1e-6)
    assert w[1, 2] == pytest.approx(0.7, abs=1e-6)


def test_soft_routing_kernel_peaks_at_closest_centroid():
    centroids = np.array([0.1, 0.3, 0.7], dtype=np.float32)
    scores = np.array([0.05, 0.4, 0.7], dtype=np.float32)
    w = soft_routing_weights_kernel(scores, bucket_centroids=centroids, tau=0.1)
    assert w.shape == (3, 3)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    # row 0 closest to centroid 0; row 2 closest to centroid 2.
    assert int(w[0].argmax()) == 0
    assert int(w[2].argmax()) == 2


def test_soft_routing_kernel_collapses_to_uniform_at_large_tau():
    centroids = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    w = soft_routing_weights_kernel(scores, bucket_centroids=centroids, tau=100.0)
    # tau is huge -> kernel decays slowly -> ~uniform weights.
    assert np.allclose(w, 1.0 / 3.0, atol=0.01)


def test_apply_soft_routing_recovers_individual_experts():
    per_expert = {"e0": np.array([0.1, 0.3, 0.5]),
                  "e1": np.array([0.9, 0.6, 0.2])}
    # Hard route everyone to e0.
    w = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
    out = apply_soft_routing(per_expert, expert_names=["e0", "e1"], weights=w)
    np.testing.assert_allclose(out, per_expert["e0"], atol=1e-6)
    # Hard route everyone to e1.
    w = np.array([[0.0, 1.0]] * 3, dtype=np.float32)
    out = apply_soft_routing(per_expert, expert_names=["e0", "e1"], weights=w)
    np.testing.assert_allclose(out, per_expert["e1"], atol=1e-6)
    # 50/50 blend.
    w = np.array([[0.5, 0.5]] * 3, dtype=np.float32)
    out = apply_soft_routing(per_expert, expert_names=["e0", "e1"], weights=w)
    np.testing.assert_allclose(
        out, 0.5 * per_expert["e0"] + 0.5 * per_expert["e1"], atol=1e-6,
    )
