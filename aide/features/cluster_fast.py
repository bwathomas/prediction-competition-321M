"""Vectorized clustering derivation — the scalable replacement for ``derive_cluster``.

``derive_cluster``'s soft-weighted cross-cluster loop is ``O(rows × K_fine × members)``,
intractable at 311k items × 256 clusters. This module computes the SAME columns with pure
array ops in ``O(nnz + rows × K)``, exploiting the production OOF structure: a query item is
held out (not in the fold's train set / passrate), so the per-row self/alias exclusion that
the codec needs is a **no-op** here — difficulty pooled over train items already excludes the
query item. The codec stays the source of truth (and the test oracle on small data); this is
its fast equivalent, proven equal in the OOF regime by ``test_cluster_fast``.

Two entry points mirror the driver: ``cluster_geometry_fast`` (fold=all, label-free) and
``cluster_labels_fast`` (per fold, uses the CSR passrate's vectorized cluster aggregates).
"""
from __future__ import annotations

import numpy as np

from aide.features.derive_cluster import _soft_responsibility, _sqdist
from aide.harness.funnel import FeatureBlock


def _block(columns, X, row_ids):
    return FeatureBlock(X=np.asarray(X, dtype=np.float32).reshape(len(row_ids), len(columns)),
                        columns=list(columns), row_ids=np.asarray(row_ids).astype(str))


def _fine_coarse(centroids_by_res):
    return (np.asarray(centroids_by_res["fine"], dtype=np.float32),
            np.asarray(centroids_by_res["coarse"], dtype=np.float32))


def cluster_geometry_fast(*, query_emb, row_ids, centroids_by_res, all_emb):
    """Neutral geometry blocks (cluster_geometry / centroid_distance / item_cluster),
    vectorized. ``all_emb`` provides the cluster sizes (fold-invariant, fold=all)."""
    fine, coarse = _fine_coarse(centroids_by_res)
    Kf = fine.shape[0]
    q_sqd = _sqdist(query_emb, fine)                      # [nq, Kf]
    q_fine = q_sqd.argmin(1)
    resp = _soft_responsibility(q_sqd)
    sizes = np.bincount(_sqdist(all_emb, fine).argmin(1), minlength=Kf)
    nq = q_sqd.shape[0]

    sorted_resp = -np.sort(-resp, axis=1)                 # descending
    top = np.zeros((nq, 3))
    top[:, :min(3, Kf)] = sorted_resp[:, :min(3, Kf)]
    margin = top[:, 0] - top[:, 1]
    entropy = -np.where(resp > 0, resp * np.log(np.where(resp > 0, resp, 1.0)), 0.0).sum(1)
    typicality = -q_sqd[np.arange(nq), q_fine]
    size_log1p = np.log1p(sizes[q_fine])
    coarse_id = _sqdist(query_emb, coarse).argmin(1)

    geo_cols = ["clu__soft_responsibility_top1", "clu__soft_responsibility_top2",
                "clu__soft_responsibility_top3", "clu__margin_1to2",
                "clu__responsibility_entropy", "clu__typicality", "clu__size_log1p",
                "clu_id__coarse", "clu_id__fine"]
    geo = np.column_stack([top[:, 0], top[:, 1], top[:, 2], margin, entropy, typicality,
                           size_log1p, coarse_id.astype(float), q_fine.astype(float)])
    cd_cols = [f"cd__centroid_dist_{j}" for j in range(Kf)]
    oh = np.zeros((nq, Kf), dtype=np.float32)
    oh[np.arange(nq), q_fine] = 1.0
    oh_cols = [f"cluster__{j:03d}" for j in range(Kf)] + ["cluster_id"]
    oh_full = np.column_stack([oh, q_fine.astype(float)])
    return {
        "cluster_geometry": _block(geo_cols, geo, row_ids),
        "centroid_distance": _block(cd_cols, q_sqd, row_ids),
        "item_cluster": _block(oh_cols, oh_full, row_ids),
    }


def cluster_labels_fast(*, query_emb, query_subjects, row_ids, centroids_by_res,
                        passrate, item_to_cluster_fine):
    """Label blocks (cluster_passrate=m2_cluster, cluster_subject=clu_subj__*), vectorized.

    ``item_to_cluster_fine``: fine-cluster id per passrate item column (driver precomputes
    once from the full item set). OOF: difficulty pools fold-train items only (the passrate
    is train-built), so the query item is self-excluded by construction.
    """
    fine, _ = _fine_coarse(centroids_by_res)
    Kf = fine.shape[0]
    q_sqd = _sqdist(query_emb, fine)
    q_fine = q_sqd.argmin(1)
    resp = _soft_responsibility(q_sqd)
    nq = q_sqd.shape[0]

    difficulty, smean, scnt = passrate.cluster_aggregates(item_to_cluster_fine, Kf)
    s_rows = np.array([passrate.s_idx.get(str(s), -1) for s in query_subjects])
    valid = s_rows >= 0
    safe = np.where(valid, s_rows, 0)

    diff_r = difficulty[q_fine]
    smean_r = smean[safe, q_fine]
    cnt_r = scnt[safe, q_fine]
    cnt_r = np.where(valid, cnt_r, 0.0)
    subj_pass = np.where(valid & np.isfinite(smean_r), smean_r, diff_r)
    gap = subj_pass - diff_r
    obs_log1p = np.log1p(cnt_r)

    # soft-responsibility-weighted subject passrate across clusters (only clusters the
    # subject has observations in contribute; normalized by their responsibility mass)
    smean_s = smean[safe]                                 # [nq, Kf], nan where no obs
    mask = (scnt[safe] > 0) & valid[:, None]
    sm = np.where(mask, smean_s, 0.0)
    num = (resp * sm).sum(1)
    den = (resp * mask).sum(1)
    soft = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

    pr_cols = ["m2_cluster_mean"]
    subj_cols = ["clu_subj__subject_minus_cluster_gap",
                 "clu_subj__cluster_obs_count_log1p",
                 "clu_subj__soft_weighted_subject_passrate"]
    return {
        "cluster_passrate": _block(pr_cols, diff_r, row_ids),
        "cluster_subject": _block(subj_cols, np.column_stack([gap, obs_log1p, soft]), row_ids),
    }


def derive_cluster_fast(*, query_emb, query_subjects, row_ids, centroids_by_res,
                        passrate, all_emb, all_item_keys, include_labels=True):
    """Driver entry point: all cluster blocks for a query batch, vectorized.

    ``all_emb`` / ``all_item_keys`` are the full item universe in the passrate's column
    order, used for cluster sizes and the item→cluster map. Geometry is fold-invariant;
    labels are OOF-correct when the query items are held out of the passrate's train set.
    """
    fine, _ = _fine_coarse(centroids_by_res)
    out = cluster_geometry_fast(query_emb=query_emb, row_ids=row_ids,
                                centroids_by_res=centroids_by_res, all_emb=all_emb)
    if include_labels:
        item_to_cluster = _sqdist(all_emb, fine).argmin(1)
        out.update(cluster_labels_fast(query_emb=query_emb, query_subjects=query_subjects,
                                       row_ids=row_ids, centroids_by_res=centroids_by_res,
                                       passrate=passrate, item_to_cluster_fine=item_to_cluster))
    return out
