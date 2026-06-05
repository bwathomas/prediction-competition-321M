"""Multi-resolution clustering codec — ``cluster__*``/``cluster_id`` (one-hot, neutral),
``cd__*`` (centroid distances, neutral), ``clu__*``/``clu_id__*`` (soft geometry, neutral),
``cluster_passrate``=``m2_cluster*`` (item difficulty, neutral but OOF), ``clu_subj__*``
(subject×cluster, subject_proxy, OOF).

Per Plan 4 §D the heavy ``MiniBatchKMeans`` *fit* (sklearn on Colab) is isolated in
``fit_multi_kmeans``; ``derive_cluster`` takes the centroids as input so the numpy
assignment / soft-responsibility / OOF-aggregation core is fully locally testable.

Why this is leakage-safe:
  * centroids are fit on item EMBEDDINGS only (no labels) ⇒ assignment/geometry are
    identity-neutral; the ``cluster__``/``cd__``/``clu__`` blocks never read a label;
  * the label-derived blocks (cluster difficulty ``m2_cluster``, ``clu_subj__*``) are OOF:
    a cluster's statistics are computed from the **train** items in that cluster with the
    query item's own row excluded, so a query never sees ``label[*, query_item]``.
"""
from __future__ import annotations

import numpy as np

from aide.harness.funnel import FeatureBlock


def fit_multi_kmeans(train_emb, ks_by_name, seed: int = 0):
    """Fit one k-means per named resolution on TRAIN embeddings. Returns
    ``{name: centroids[K, d]}``. Uses sklearn ``MiniBatchKMeans`` on Colab; falls back to
    a small numpy Lloyd locally so the codec is runnable without sklearn."""
    train_emb = np.asarray(train_emb, dtype=np.float32)
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception:
        return {name: _lloyd(train_emb, int(K), seed) for name, K in ks_by_name.items()}
    out = {}
    for name, K in ks_by_name.items():
        km = MiniBatchKMeans(n_clusters=int(K), random_state=seed, n_init=3)
        km.fit(train_emb)
        out[name] = np.asarray(km.cluster_centers_, dtype=np.float32)
    return out


def _lloyd(X, K, seed, iters=25):
    rng = np.random.default_rng(seed)
    K = min(K, X.shape[0])
    cen = X[rng.choice(X.shape[0], size=K, replace=False)].astype(np.float32)
    for _ in range(iters):
        d = _sqdist(X, cen)
        a = d.argmin(axis=1)
        new = np.array([X[a == k].mean(axis=0) if np.any(a == k) else cen[k]
                        for k in range(K)], dtype=np.float32)
        if np.allclose(new, cen):
            break
        cen = new
    return cen


def _sqdist(X, C):
    """Squared L2 distances [n, K]."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return (np.sum(X * X, axis=1)[:, None] - 2 * X @ C.T + np.sum(C * C, axis=1)[None, :])


def _soft_responsibility(sqd):
    """Softmax over negative squared distance (temperature = median sqd for scale)."""
    scale = np.median(sqd) + 1e-9
    z = -sqd / scale
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _block(columns, rows, row_ids):
    X = np.asarray(rows, dtype=np.float32).reshape(len(row_ids), len(columns))
    return FeatureBlock(X=X, columns=list(columns), row_ids=np.asarray(row_ids).astype(str))


def _unit(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def derive_cluster(*, query_emb, query_item_keys, query_subjects, row_ids,
                   centroids_by_res, train_emb, train_item_keys, passrate,
                   oof_item_keys=None, alias_eps=1e-6):
    query_emb = np.asarray(query_emb, dtype=np.float32)
    train_emb = np.asarray(train_emb, dtype=np.float32)
    train_keys = [str(k) for k in train_item_keys]
    q_keys = [str(k) for k in query_item_keys]
    fine = np.asarray(centroids_by_res["fine"], dtype=np.float32)
    Kf = fine.shape[0]

    # Optional hard OOF precondition: the index/train items must be disjoint from the
    # fold's OOF items (cold-start). Cheap guard against a caller passing the full matrix.
    if oof_item_keys is not None:
        overlap = set(train_keys) & {str(k) for k in oof_item_keys}
        if overlap:
            raise AssertionError(
                f"derive_cluster train/oof leakage: {len(overlap)} item(s) in both")

    # cosine self/alias exclusion: a train item that IS the query item (different key but
    # near-identical embedding) must not feed the query's own label into its cluster stats.
    tn, qn = _unit(train_emb), _unit(query_emb)
    alias_thr = 1.0 - alias_eps

    # ---- train-side assignment (for cluster sizes + OOF label aggregation) ----------
    train_fine_id = _sqdist(train_emb, fine).argmin(axis=1)
    sizes = np.bincount(train_fine_id, minlength=Kf)

    # ---- query-side geometry --------------------------------------------------------
    q_sqd_fine = _sqdist(query_emb, fine)
    q_fine_id = q_sqd_fine.argmin(axis=1)
    resp = _soft_responsibility(q_sqd_fine)

    geo_rows, cd_rows, oh_rows = [], [], []
    geo_cols = ["clu__soft_responsibility_top1", "clu__soft_responsibility_top2",
                "clu__soft_responsibility_top3", "clu__margin_1to2",
                "clu__responsibility_entropy", "clu__typicality", "clu__size_log1p",
                "clu_id__coarse", "clu_id__fine"]
    cd_cols = [f"cd__centroid_dist_{j}" for j in range(Kf)]
    oh_cols = [f"cluster__{j:03d}" for j in range(Kf)] + ["cluster_id"]
    coarse = np.asarray(centroids_by_res["coarse"], dtype=np.float32)
    q_coarse_id = _sqdist(query_emb, coarse).argmin(axis=1)

    for r in range(len(row_ids)):
        rr = np.sort(resp[r])[::-1]
        top = [rr[i] if i < len(rr) else 0.0 for i in range(3)]
        margin = float(top[0] - top[1])
        p = resp[r][resp[r] > 0]
        entropy = float(-(p * np.log(p)).sum()) if p.size else 0.0
        typicality = float(-q_sqd_fine[r, q_fine_id[r]])  # closer ⇒ higher (less negative)
        size_log1p = float(np.log1p(sizes[q_fine_id[r]]))
        geo_rows.append(list(top) + [margin, entropy, typicality, size_log1p,
                                     float(q_coarse_id[r]), float(q_fine_id[r])])
        cd_rows.append(list(q_sqd_fine[r]))
        oh = np.zeros(Kf, dtype=np.float32)
        oh[q_fine_id[r]] = 1.0
        oh_rows.append(list(oh) + [float(q_fine_id[r])])

    # ---- OOF label-derived blocks ---------------------------------------------------
    pr_rows, subj_rows = [], []
    pr_cols = ["m2_cluster_mean"]
    subj_cols = ["clu_subj__subject_minus_cluster_gap",
                 "clu_subj__cluster_obs_count_log1p",
                 "clu_subj__soft_weighted_subject_passrate"]

    # precompute, per fine cluster, the train item INDICES in it (for OOF aggregation)
    cluster_train_idx = {k: np.where(train_fine_id == k)[0] for k in range(Kf)}

    def oof_members(cluster_k, r):
        """Train item keys in ``cluster_k`` with the query item — by key OR embedding
        alias (cosine >= 1-alias_eps) — removed, so the row's own label can't leak."""
        out = []
        for ti in cluster_train_idx[cluster_k]:
            if train_keys[ti] == q_keys[r]:
                continue
            if float(tn[ti] @ qn[r]) >= alias_thr:
                continue
            out.append(train_keys[ti])
        return out

    for r in range(len(row_ids)):
        k = int(q_fine_id[r])
        mem_by_k = {kk: oof_members(kk, r) for kk in range(Kf)}  # all self/alias-excluded
        members = mem_by_k[k]
        # difficulty prior for an empty/all-unobserved cluster = the row's overall OOF
        # train difficulty (self-excluded ⇒ leak-safe), NOT 0.0 ("everyone fails") and NOT
        # a global mean that would include the query item's own column.
        all_oof_r = [m for kk in range(Kf) for m in mem_by_k[kk]]
        prior = passrate.pooled_mean(all_oof_r, default=passrate.global_mean())
        # cluster difficulty pools ALL subjects' labels on member items (item content);
        # the subject channel is the row's subject alone — the two differ, so the gap is real.
        cluster_mean = passrate.pooled_mean(members, default=prior)
        pr_rows.append([cluster_mean])

        subj_labels = passrate.gather(query_subjects[r], members)
        subj_obs = np.isfinite(subj_labels)
        subj_passrate = float(np.nanmean(subj_labels)) if subj_obs.any() else cluster_mean
        gap = subj_passrate - cluster_mean  # subject ability vs cluster difficulty
        obs_count_log1p = float(np.log1p(int(subj_obs.sum())))
        # soft-responsibility-weighted subject passrate across clusters (geometry × label)
        weighted = 0.0
        wsum = 0.0
        for kk in range(Kf):
            lab = passrate.gather(query_subjects[r], mem_by_k[kk])
            if np.isfinite(lab).any():
                weighted += resp[r, kk] * float(np.nanmean(lab))
                wsum += resp[r, kk]
        soft_weighted = weighted / wsum if wsum > 0 else 0.0
        subj_rows.append([gap, obs_count_log1p, soft_weighted])

    return {
        "cluster_geometry": _block(geo_cols, geo_rows, row_ids),
        "centroid_distance": _block(cd_cols, cd_rows, row_ids),
        "item_cluster": _block(oh_cols, oh_rows, row_ids),
        "cluster_passrate": _block(pr_cols, pr_rows, row_ids),
        "cluster_subject": _block(subj_cols, subj_rows, row_ids),
    }
