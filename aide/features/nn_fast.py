"""Vectorized NN label aggregation — the throughput replacement for ``derive_nn``'s
per-row Python loop (~400 rows/s observed → array ops over the whole chunk).

Computes the same ``nn_label_derivatives`` + ``counts_subject`` columns as ``derive_nn`` by
building, per chunk, the ``[n_rows, maxK]`` matrix of neighbour labels (self/alias-excluded,
FAISS neighbours mapped to passrate columns, gathered with ``CsrPassrate.gather_pairs``) and
reducing it with numpy. Exact to the codec **in the production regime** where every row has
≥ maxK valid neighbours (a huge train index) — i.e. no short-neighbour padding. The codec
stays the oracle; ``test_nn_fast`` proves equality on a no-padding fixture.

``nn_geometry`` (per item, fold=all) is left to ``derive_nn`` — it is one pass, not the 5.3M-row hot path.
"""
from __future__ import annotations

import numpy as np

from aide.features.derive_nn import _nanslope
from aide.harness.funnel import FeatureBlock


def _block(columns, X, row_ids):
    return FeatureBlock(X=np.asarray(X, dtype=np.float32).reshape(len(row_ids), len(columns)),
                        columns=list(columns), row_ids=np.asarray(row_ids).astype(str))


def _slope_batch(values, ks):
    """Per-row least-squares slope of ``values[n, len(ks)]`` vs log2(ks) (matches
    derive_nn._nanslope row-wise; here every value is finite in-regime)."""
    return np.array([_nanslope(row, ks) for row in values])


def derive_nn_labels_fast(*, query_emb, query_item_keys, query_subjects, row_ids,
                          index_emb, index_item_keys, passrate, Ks=(4, 8, 32, 64),
                          knn_fn=None, search_buffer=2, alias_eps=1e-6):
    from aide.features.derive_nn import default_knn
    knn_fn = knn_fn or default_knn
    Ks = tuple(int(k) for k in Ks)
    maxK = max(Ks)
    n = len(row_ids)
    index_keys = np.asarray([str(k) for k in index_item_keys])
    q_keys = np.asarray([str(k) for k in query_item_keys])
    m_index = int(np.asarray(index_emb).shape[0])
    alias_thr = 1.0 - alias_eps
    train_to_col = np.array([passrate.i_idx.get(str(k), -1) for k in index_item_keys],
                            dtype=np.int64)
    subj_rows = np.array([passrate.s_idx.get(str(s), -1) for s in query_subjects], dtype=np.int64)

    n_request = min(m_index, maxK + 1 + search_buffer)
    idx, sim = knn_fn(index_emb, query_emb, n_request)
    idx = np.asarray(idx); sim = np.asarray(sim)

    # validity per (row, slot): real index, not the query item's own key, not an alias
    safe_idx = np.clip(idx, 0, m_index - 1)
    is_self = index_keys[safe_idx] == q_keys[:, None]
    valid = (idx >= 0) & (~is_self) & (sim < alias_thr)

    # take the first maxK valid slots per row (stable: keeps FAISS's descending-sim order)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :maxK]
    rows = np.arange(n)[:, None]
    sel_idx = idx[rows, order]
    sel_valid = valid[rows, order]

    # gather neighbour labels into [n, maxK] (nan where slot invalid or unobserved)
    neigh_col = np.where(sel_valid, train_to_col[np.clip(sel_idx, 0, m_index - 1)], -1)
    subj_rep = np.repeat(subj_rows, maxK)
    L = passrate.gather_pairs(subj_rep, neigh_col.reshape(-1)).reshape(n, maxK)
    L[~sel_valid] = np.nan
    obs = np.isfinite(L)

    # per-K means + coverage (nan-aware)
    mean_cols, cov_cols = [], []
    for k in Ks:
        ok = obs[:, :k]
        cnt = ok.sum(1)
        s = np.where(ok, L[:, :k], 0.0).sum(1)
        mean_cols.append(np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0))
        cov_cols.append(ok.mean(1))
    means = np.column_stack(mean_cols)
    covs = np.column_stack(cov_cols)

    with np.errstate(invalid="ignore"):
        q50 = np.where(obs.any(1), np.nanmedian(np.where(obs, L, np.nan), axis=1), 0.0)
        q75 = np.nanpercentile(np.where(obs, L, np.nan), 75, axis=1)
        q25 = np.nanpercentile(np.where(obs, L, np.nan), 25, axis=1)
    iqr = np.where(obs.any(1), q75 - q25, 0.0)
    frac_pass = np.where(obs.any(1),
                         np.nansum(np.where(obs, L > 0.5, np.nan), axis=1)
                         / np.maximum(obs.sum(1), 1), 0.0)
    p_slope = _slope_batch(means, Ks)
    c_slope = _slope_batch(covs, Ks)

    nn_cols = ([f"nn__passrate_mean_K{k}" for k in Ks]
               + [f"nn__coverage_K{k}" for k in Ks]
               + ["nn__passrate_K_slope", "nn__coverage_K_slope",
                  "nn__passrate_q50", "nn__passrate_iqr", "nn__frac_neighbors_pass"])
    nn_X = np.column_stack([means, covs, p_slope, c_slope, q50, iqr, frac_pass])
    cnt_support = obs.sum(1).astype(float)
    return {
        "nn_label_derivatives": _block(nn_cols, nn_X, row_ids),
        "counts_subject": _block(["cnt__neighbor_subject_support"], cnt_support, row_ids),
    }
