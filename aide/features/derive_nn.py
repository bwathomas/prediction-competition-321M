"""kNN-over-embeddings feature codec — ``nn__*`` (subject-keyed labels), ``geo__*``
(neutral item geometry), ``cnt__*`` (neighbour support).

Per Plan 4 §D the embedding kNN is the workhorse: one batched query yields the label
derivatives, the geometry gates, and the support counts. The heavy *search* (FAISS on the
A100) is injected via ``knn_fn`` so this module's correctness-critical aggregation — and
above all the **OOF guard** — is exercised locally with an exact numpy brute force.

OOF guard (why this is leakage-safe):
  * the index is built on fold-``f`` **train** items only (caller's contract), so a query
    row's neighbours are train items;
  * the query item itself is **always dropped** from its own neighbour list (item-key
    self-exclusion), so even the ``fold="all"`` geometry pass (query set == index set)
    never reads ``label[subject, query_item]``;
  * label aggregation reads the row's subject's labels on those *neighbour* items only.

``geo__*`` uses distances only (no labels) ⇒ it is identity-neutral and fold-invariant.
"""
from __future__ import annotations

import numpy as np

from aide.harness.funnel import FeatureBlock


# --------------------------------------------------------------------------------------
# kNN search backends. Brute force is the exact oracle + the local/no-FAISS fallback;
# FAISS is the Colab accelerator with identical (idx, sim) semantics (inner product).
# --------------------------------------------------------------------------------------
def bruteforce_knn(index_emb, query_emb, k):
    """Exact top-``k`` by inner product. Returns (idx[n_q,k] int, sim[n_q,k] float),
    each row sorted by descending similarity. ``k`` is clipped to the index size."""
    index_emb = np.asarray(index_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    sims = query_emb @ index_emb.T
    m = index_emb.shape[0]
    k = min(int(k), m)
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    idx = part[rows, order]
    return idx, sims[rows, idx]


def default_knn(index_emb, query_emb, k):
    """FAISS ``IndexFlatIP`` if available (Colab), else the exact brute force."""
    try:
        import faiss  # noqa: F401
    except Exception:
        return bruteforce_knn(index_emb, query_emb, k)
    index_emb = np.ascontiguousarray(np.asarray(index_emb, dtype=np.float32))
    query_emb = np.ascontiguousarray(np.asarray(query_emb, dtype=np.float32))
    index = faiss.IndexFlatIP(index_emb.shape[1])
    index.add(index_emb)
    sim, idx = index.search(query_emb, min(int(k), index_emb.shape[0]))
    return idx, sim


class DensePassrate:
    """Dense ``(subject, item) -> label`` lookup with ``nan`` for unobserved cells.

    The local test double for the Colab ``scipy.sparse`` CSR passrate matrix; both expose
    ``gather(subject, item_keys) -> array`` so the codec is storage-agnostic.
    """

    def __init__(self, subjects, items, L):
        self.s_idx = {str(s): i for i, s in enumerate(subjects)}
        self.i_idx = {str(it): j for j, it in enumerate(items)}
        self.L = np.asarray(L, dtype=float)

    def gather(self, subject, item_keys) -> np.ndarray:
        out = np.full(len(item_keys), np.nan)
        si = self.s_idx.get(str(subject))
        if si is None:
            return out
        for t, it in enumerate(item_keys):
            j = self.i_idx.get(str(it))
            if j is not None:
                out[t] = self.L[si, j]
        return out

    def global_mean(self) -> float:
        """Mean over ALL observed labels (the prior for an empty/all-unobserved group)."""
        return float(np.nanmean(self.L)) if np.isfinite(self.L).any() else 0.0

    def pooled_mean(self, item_keys, default: float = 0.0) -> float:
        """Mean over ALL subjects' observed labels on these items (cluster difficulty).

        Pools every subject, not just one row's subject, so a cluster's difficulty is a
        genuine item-content statistic. ``nan`` (unobserved) cells are ignored; an
        empty / all-unobserved set returns ``default`` — the caller supplies a *self-
        excluded* prior so the fallback never reads the query item's own label.
        """
        cols = [self.i_idx[str(it)] for it in item_keys if str(it) in self.i_idx]
        if not cols:
            return default
        sub = self.L[:, cols]
        return float(np.nanmean(sub)) if np.isfinite(sub).any() else default


def _nanslope(values, ks) -> float:
    """Least-squares slope of ``values`` vs ``log2(k)``; 0 if <2 finite points."""
    x = np.log2(np.asarray(ks, dtype=float))
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return 0.0
    xx = x[ok] - x[ok].mean()
    yy = y[ok] - y[ok].mean()
    denom = float((xx * xx).sum())
    return float((xx * yy).sum() / denom) if denom > 0 else 0.0


def _block(columns, rows, row_ids) -> FeatureBlock:
    X = np.asarray(rows, dtype=np.float32).reshape(len(row_ids), len(columns))
    return FeatureBlock(X=X, columns=list(columns),
                        row_ids=np.asarray(row_ids).astype(str))


def derive_nn(*, query_emb, query_item_keys, query_subjects, row_ids,
              index_emb, index_item_keys, passrate,
              Ks=(4, 8, 32, 64), knn_fn=None, search_buffer=2, alias_eps=1e-6):
    """Derive the kNN feature blocks for a set of (subject, item) query rows.

    Returns ``{group_name: FeatureBlock}`` for ``nn_label_derivatives`` (subject_proxy),
    ``nn_geometry`` (neutral_item, fold-invariant) and ``counts_subject`` (subject_proxy).
    Each block shares ``row_ids`` so the funnel can column-concatenate them.

    OOF self-exclusion drops a neighbour if (a) FAISS returned ``-1`` padding, (b) it has
    the query item's key, OR (c) its similarity is ``>= 1 - alias_eps`` (an embedding-alias
    of the query item under a *different* key — the leak the key-only check misses). If,
    without having scanned the whole index, fewer than ``maxK`` neighbours survive, we
    raise rather than silently averaging over a collapsed K.
    """
    knn_fn = knn_fn or default_knn
    Ks = tuple(int(k) for k in Ks)
    maxK = max(Ks)
    index_keys = [str(k) for k in index_item_keys]
    q_keys = [str(k) for k in query_item_keys]
    n = len(row_ids)
    m_index = int(np.asarray(index_emb).shape[0])
    alias_thr = 1.0 - alias_eps

    # Over-retrieve so that after dropping self / aliases we still have maxK neighbours.
    n_request = min(m_index, maxK + 1 + search_buffer)
    idx, sim = knn_fn(index_emb, query_emb, n_request)

    nn_rows, geo_rows, cnt_rows = [], [], []
    nn_cols = ([f"nn__passrate_mean_K{k}" for k in Ks]
               + [f"nn__coverage_K{k}" for k in Ks]
               + ["nn__passrate_K_slope", "nn__coverage_K_slope",
                  "nn__passrate_q50", "nn__passrate_iqr", "nn__frac_neighbors_pass"])
    geo_cols = ["geo__local_density", "geo__dist_gap_1_to_K", "geo__lid_estimate"]
    cnt_cols = ["cnt__neighbor_subject_support"]

    for r in range(n):
        raw_idx = np.asarray(idx[r])
        raw_sim = np.asarray(sim[r])
        keep = []
        for j in range(len(raw_idx)):
            ci = int(raw_idx[j])
            if ci < 0:                          # FAISS pads short result rows with -1
                continue
            if index_keys[ci] == q_keys[r]:     # exact self by key
                continue
            if float(raw_sim[j]) >= alias_thr:  # embedding-alias of the query item
                continue
            keep.append(j)
        if len(keep) < maxK and n_request < m_index:
            raise ValueError(
                f"derive_nn under-retrieved for row {r}: {len(keep)} usable neighbours "
                f"< maxK={maxK} without scanning the full index — raise search_buffer")
        sel = keep[:maxK]
        cand = raw_idx[sel].astype(int)
        cand_sim = raw_sim[sel]
        neigh_items = [index_keys[j] for j in cand]
        labels = passrate.gather(query_subjects[r], neigh_items)  # nan = unobserved

        means, covs = [], []
        for k in Ks:
            lab_k = labels[:k]
            obs = np.isfinite(lab_k)
            means.append(float(np.nanmean(lab_k)) if obs.any() else 0.0)
            covs.append(float(obs.mean()) if len(lab_k) else 0.0)
        obs_all = np.isfinite(labels)
        passed = labels[obs_all]
        q50 = float(np.median(passed)) if passed.size else 0.0
        iqr = float(np.subtract(*np.percentile(passed, [75, 25]))) if passed.size else 0.0
        frac_pass = float((passed > 0.5).mean()) if passed.size else 0.0
        nn_rows.append(means + covs
                       + [_nanslope(means, Ks), _nanslope(covs, Ks), q50, iqr, frac_pass])

        # geometry (labels never touched)
        s = np.asarray(cand_sim, dtype=float)
        local_density = float(s.mean()) if s.size else 0.0
        gap = float(s[0] - s[-1]) if s.size else 0.0
        # Levina–Bickel LID from cosine distances d = 1 - sim (clipped to be positive)
        d = np.clip(1.0 - s, 1e-9, None)
        lid = 0.0
        if d.size > 1 and d[-1] > 0:
            log_ratio = float(np.mean(np.log(d[-1] / d[:-1])))
            if log_ratio > 0:  # equal distances -> 0; never emit inf/nan into a neutral col
                lid = 1.0 / log_ratio
        geo_rows.append([local_density, gap, lid])

        cnt_rows.append([float(obs_all.sum())])

    return {
        "nn_label_derivatives": _block(nn_cols, nn_rows, row_ids),
        "nn_geometry": _block(geo_cols, geo_rows, row_ids),
        "counts_subject": _block(cnt_cols, cnt_rows, row_ids),
    }
