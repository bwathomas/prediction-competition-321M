"""Smoke tests for src.mlp_variant (shared probe-trainer).

Mirrors the contract used by `notebooks.loss_diversity_probe` and
`notebooks.moe_poc`:
  * subject embedding + item embedding + optional dense block -> [N] probs
  * sample weights bias the loss toward the up-weighted rows
  * the trainer returns a torch.nn.Module that `predict_probs` consumes
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.mlp_variant import (
    build_net,
    objective_loss,
    predict_probs,
    train_m8_variant,
)


# ---------------------------------------------------------------------------
# Tiny synthetic dataset: 2 subjects, U unique items, signal lives on the
# item embedding so the MLP has something real to learn in <5s on CPU.
# ---------------------------------------------------------------------------


def _make_synth(n_rows=1024, U=64, d=16, n_subjects=2, seed=0):
    rng = np.random.default_rng(seed)
    item_emb = rng.normal(size=(U + 1, d)).astype(np.float32)  # +1 zero row
    item_emb[-1] = 0.0
    r2u = rng.integers(0, U, size=n_rows).astype(np.int64)
    subj = rng.integers(0, n_subjects, size=n_rows).astype(np.int64)
    # Signal: linear projection of the item embedding + a subject bias.
    w = rng.normal(size=d).astype(np.float32)
    sb = np.array([-0.3, +0.3], dtype=np.float32)
    logit = item_emb[r2u] @ w + sb[subj]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n_rows) < p).astype(np.float32)
    return y, subj, r2u, item_emb, n_subjects


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def test_build_net_forward_shapes():
    """The constructed net accepts the three-channel input contract."""
    dev = "cpu"
    net = build_net(
        n_subjects=4, subj_emb_dim=4, item_dim=8, dense_dim=3,
        hid1=16, hid2=8, feat_dropout=0.0, seed=0, device=dev,
    )
    sid = torch.tensor([0, 1, 2], dtype=torch.long)
    iemb = torch.randn(3, 8)
    dz = torch.randn(3, 3)
    out = net(sid, iemb, dz)
    assert tuple(out.shape) == (3,)
    assert torch.isfinite(out).all()


def test_build_net_supports_embedding_only():
    """dense_dim=0 -> the dense channel is ignored even if passed as None."""
    net = build_net(
        n_subjects=2, subj_emb_dim=2, item_dim=4, dense_dim=0,
        hid1=8, hid2=4, feat_dropout=0.0, seed=0, device="cpu",
    )
    sid = torch.tensor([0, 1], dtype=torch.long)
    iemb = torch.randn(2, 4)
    out = net(sid, iemb, None)
    assert tuple(out.shape) == (2,)


def test_objective_loss_handles_known_objectives():
    """Every objective name the trainer dispatches on returns a finite scalar."""
    logits = torch.tensor([0.5, -0.5, 1.2, -1.2], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    for name in ("bce", "label_smooth", "brier", "focal",
                 "class_weighted", "cold", "specialist", "distill"):
        v = objective_loss(
            logits, target, weights=None,
            objective=name, focal_gamma=2.0, label_smooth_eps=0.05,
        )
        assert torch.isfinite(v), f"non-finite loss for {name!r}"


def test_objective_loss_rejects_unknown():
    logits = torch.zeros(2)
    target = torch.zeros(2)
    with pytest.raises(ValueError):
        objective_loss(
            logits, target, weights=None,
            objective="not_a_real_objective", focal_gamma=2.0,
            label_smooth_eps=0.0,
        )


def test_train_then_predict_fits_signal():
    """End-to-end: trained net predicts the synthetic signal better than 0.5."""
    y, subj, r2u, item_emb, ns = _make_synth(n_rows=1024, U=64, d=16, seed=0)
    emb_t = torch.from_numpy(item_emb)
    net = train_m8_variant(
        y=y, subj_ids=subj, r2u=r2u, emb_tensor=emb_t,
        n_subjects=ns, device="cpu",
        objective="bce",
        subj_emb_dim=4, hid1=32, hid2=16,
        lr=1e-2, wd=0.0, epochs=10, batch_size=128, val_fraction=0.2,
        patience=4, feat_dropout=0.0, seed=0, show_progress=False,
    )
    p = predict_probs(net, subj, r2u, emb_t, "cpu", chunk=512)
    assert p.shape == (1024,)
    assert np.all(np.isfinite(p))
    assert (p > 1e-6).all() and (p < 1.0 - 1e-6).all()
    # Trained model should beat a constant 0.5 baseline by a clear margin
    # on this signal-heavy synthetic. Loose threshold to avoid flakiness.
    nll_constant = _nll(np.full_like(p, fill_value=0.5), y)
    nll_fit = _nll(p, y)
    assert nll_fit < nll_constant - 0.02, (
        f"fit didn't learn the signal: fit={nll_fit:.4f} vs "
        f"const={nll_constant:.4f}"
    )


def test_sample_weights_bias_toward_upweighted_subset():
    """A 5x sample weight on a subset shifts the fit's NLL toward that
    subset vs the unweighted fit. Mirrors how `moe_poc` upweights an
    expert's region without hard-partitioning the data."""
    y, subj, r2u, item_emb, ns = _make_synth(n_rows=2048, U=64, d=16, seed=1)
    emb_t = torch.from_numpy(item_emb)

    # Pick the "in-region" rows arbitrarily as the first half. Upweight
    # them 5x in one fit; keep uniform weights in the other.
    in_region = np.zeros(len(y), dtype=bool)
    in_region[: len(y) // 2] = True
    out_region = ~in_region

    base = train_m8_variant(
        y=y, subj_ids=subj, r2u=r2u, emb_tensor=emb_t,
        n_subjects=ns, device="cpu",
        sample_weights=None,
        subj_emb_dim=4, hid1=32, hid2=16,
        lr=1e-2, wd=0.0, epochs=10, batch_size=128, val_fraction=0.2,
        patience=4, feat_dropout=0.0, seed=2, show_progress=False,
    )
    w = np.where(in_region, 5.0, 1.0).astype(np.float32)
    w = (w / w.mean()).astype(np.float32)
    expert = train_m8_variant(
        y=y, subj_ids=subj, r2u=r2u, emb_tensor=emb_t,
        n_subjects=ns, device="cpu",
        sample_weights=w,
        subj_emb_dim=4, hid1=32, hid2=16,
        lr=1e-2, wd=0.0, epochs=10, batch_size=128, val_fraction=0.2,
        patience=4, feat_dropout=0.0, seed=2, show_progress=False,
    )
    p_base = predict_probs(base, subj, r2u, emb_t, "cpu", chunk=512)
    p_exp = predict_probs(expert, subj, r2u, emb_t, "cpu", chunk=512)

    # The expert should be at least as good as baseline ON the upweighted
    # region (often strictly better; weak inequality avoids small-sample
    # flakiness). The contract we care about for MoE is "upweighting
    # produces a model biased toward the region", which this enforces.
    base_in = _nll(p_base[in_region], y[in_region])
    exp_in = _nll(p_exp[in_region], y[in_region])
    assert exp_in <= base_in + 0.005, (
        f"upweighted fit didn't help in-region: base_in={base_in:.4f} "
        f"exp_in={exp_in:.4f}"
    )
    # And the predictions must materially differ (otherwise the weights
    # had no effect and the MoE POC would be a no-op).
    assert float(np.abs(p_exp - p_base).mean()) > 1e-3


def test_train_supports_dense_channel():
    """dense_X is accepted and the trained model uses it (signal lives in dense)."""
    rng = np.random.default_rng(3)
    n = 1024
    U = 16
    d = 4
    item_emb = rng.normal(size=(U + 1, d)).astype(np.float32)
    item_emb[-1] = 0.0
    r2u = rng.integers(0, U, size=n).astype(np.int64)
    subj = np.zeros(n, dtype=np.int64)
    dense_X = rng.normal(size=(n, 6)).astype(np.float32)
    dw = rng.normal(size=6).astype(np.float32)
    logit = dense_X @ dw  # ALL signal in the dense channel
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(np.float32)

    emb_t = torch.from_numpy(item_emb)
    net = train_m8_variant(
        y=y, subj_ids=subj, r2u=r2u, emb_tensor=emb_t,
        n_subjects=1, device="cpu", dense_X=dense_X,
        objective="bce", subj_emb_dim=2, hid1=32, hid2=16,
        lr=1e-2, wd=0.0, epochs=10, batch_size=128, val_fraction=0.2,
        patience=4, feat_dropout=0.0, seed=4, show_progress=False,
    )
    p_hat = predict_probs(
        net, subj, r2u, emb_t, "cpu", chunk=512, dense_X=dense_X,
    )
    assert p_hat.shape == (n,)
    assert _nll(p_hat, y) < _nll(np.full_like(p_hat, 0.5), y) - 0.02
