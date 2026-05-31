"""Tests for the cheap proxy go/no-go harness (src.proxy_eval).

The harness must (1) recover a planted orthogonal signal as a negative
held-out ΔNLL with a CI below zero, and (2) correctly reject a proxy that is a
pure copy of the baseline's information (ΔNLL CI straddling / above zero).
"""

from __future__ import annotations

import numpy as np

from src.proxy_eval import (
    bce,
    fit_logistic_irls,
    fit_nn_baseline_oof,
    incremental_nll_test,
    run_proxy_probe,
    support_quartile_masks,
)


def test_irls_recovers_known_logistic():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2))
    true = np.array([0.5, 1.5, -1.0])  # bias, w0, w1
    eta = true[0] + X @ true[1:]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    w = fit_logistic_irls(X, y, l2=1e-4)
    assert np.allclose(w, true, atol=0.15), f"got {w}"


def test_irls_offset_is_used():
    # With a perfect offset and no features, bias should fit ~0.
    rng = np.random.default_rng(1)
    n = 3000
    p_true = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    offset = np.log(p_true / (1 - p_true))
    w = fit_logistic_irls(np.zeros((n, 1)), y, l2=1e-3, offset=offset)
    assert abs(w[0]) < 0.15, f"bias should be ~0 with a good offset, got {w[0]}"


def _make_folds(item_key_per_row, n_folds=3, seed=7):
    uniq = np.unique(item_key_per_row)
    rng = np.random.default_rng(seed)
    fold_of_item = {k: int(i % n_folds) for i, k in enumerate(rng.permutation(uniq))}
    row_fold = np.array([fold_of_item[k] for k in item_key_per_row])
    tr, oof = [], []
    for f in range(n_folds):
        oof.append(np.where(row_fold == f)[0])
        tr.append(np.where(row_fold != f)[0])
    return tr, oof


def test_baseline_oof_covers_all_rows_and_beats_constant():
    rng = np.random.default_rng(2)
    n_items = 600
    item_ability = rng.normal(size=n_items)  # item difficulty
    rows_per_item = 30
    item_idx = np.repeat(np.arange(n_items), rows_per_item)
    item_keys = np.array([f"it{ i }" for i in item_idx])
    subj = rng.normal(size=item_idx.shape[0])
    eta = 1.0 * subj - 1.2 * item_ability[item_idx]
    y = (rng.uniform(size=eta.shape[0]) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    # NN feature block: a noisy view of item difficulty (the "current signal").
    nn0 = -item_ability[item_idx] + rng.normal(scale=0.3, size=eta.shape[0])
    nn_mat = np.column_stack([nn0, rng.normal(size=eta.shape[0])])
    tr, oof = _make_folds(item_keys)
    p_base = fit_nn_baseline_oof(nn_mat, y, tr, oof, l2=2.0)
    assert np.isfinite(p_base).all()
    assert bce(y, p_base) < bce(y, np.full_like(y, y.mean()))


def _planted_dataset(seed=3, orthogonal=True):
    rng = np.random.default_rng(seed)
    n_items = 800
    rows_per_item = 40
    item_idx = np.repeat(np.arange(n_items), rows_per_item)
    item_keys = np.array([f"it{ i }" for i in item_idx])
    n = item_idx.shape[0]
    subj = rng.normal(size=n)
    item_diff = rng.normal(size=n_items)
    z_item = rng.normal(size=n_items)
    if orthogonal:
        # Label depends on z_item beyond what nn captures (a real proxy).
        eta = 1.0 * subj - 1.0 * item_diff[item_idx] + 1.3 * z_item[item_idx]
        nn0 = -item_diff[item_idx] + rng.normal(scale=0.25, size=n)
    else:
        # z is pure noise, independent of the label: a NULL proxy. The harness
        # must NOT report a confident held-out win here.
        eta = 1.0 * subj - 1.0 * item_diff[item_idx] + 1.3 * z_item[item_idx]
        nn0 = -item_diff[item_idx] + rng.normal(scale=0.25, size=n)
        z_item = rng.normal(size=n_items)  # re-draw: unrelated to eta
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    nn_mat = np.column_stack([nn0, rng.normal(size=n)])
    z_row = z_item[item_idx]
    return item_keys, y, nn_mat, z_row


def test_incremental_detects_orthogonal_signal():
    item_keys, y, nn_mat, z_row = _planted_dataset(seed=3, orthogonal=True)
    tr, oof = _make_folds(item_keys)
    p_base = fit_nn_baseline_oof(nn_mat, y, tr, oof, l2=2.0)
    res = incremental_nll_test(
        p_base=p_base, z_per_row=z_row, labels=y,
        item_key_per_row=item_keys, n_boot=300, seed=0,
    )
    assert res.delta_nll < 0, f"orthogonal z should reduce NLL, got {res.delta_nll}"
    assert res.helps, f"CI should be below 0: {res.delta_nll_ci}"
    assert abs(res.item_partial_corr) > 0.1


def test_incremental_rejects_null_signal():
    item_keys, y, nn_mat, z_row = _planted_dataset(seed=5, orthogonal=False)
    tr, oof = _make_folds(item_keys)
    p_base = fit_nn_baseline_oof(nn_mat, y, tr, oof, l2=2.0)
    res = incremental_nll_test(
        p_base=p_base, z_per_row=z_row, labels=y,
        item_key_per_row=item_keys, n_boot=300, seed=0,
    )
    # A null (pure-noise) proxy must not produce a confident held-out win.
    assert not res.helps, f"null z should not clear CI: {res.delta_nll_ci}"


def test_support_quartiles_partition():
    s = np.arange(1000, dtype=float)
    masks = support_quartile_masks(s, n_buckets=4)
    assert len(masks) == 4
    total = np.zeros(1000, dtype=bool)
    for _, m in masks:
        assert not (total & m).any()
        total |= m
    assert total.all()


def test_run_proxy_probe_smoke():
    item_keys, y, nn_mat, z_row = _planted_dataset(seed=3, orthogonal=True)
    tr, oof = _make_folds(item_keys)
    p_base = fit_nn_baseline_oof(nn_mat, y, tr, oof, l2=2.0)
    support = nn_mat[:, 0]
    out = run_proxy_probe(
        p_base=p_base, z_per_row=z_row, labels=y, item_key_per_row=item_keys,
        support_per_row=support, n_boot=200, seed=0,
    )
    assert "all" in out
    assert out["all"].delta_nll < 0
