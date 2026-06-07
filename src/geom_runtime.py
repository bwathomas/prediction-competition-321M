"""Runtime recomputation of the geometry feature block for NEW (cold) item embeddings.

The tree/mlp consume a 525-col geometry block assembled, in training, from
store.assemble(GEOM_GROUPS) with GEOM_GROUPS in THIS order:
    centroid_distance (256) | cluster_geometry (9) | nn_geometry (3) | item_cluster (257)
At inference a cold item gives only its raw embedding, so we recompute the block from two
exported artifacts:
  * centroids  (fine[256,D] + coarse[32,D]) — re-fit deterministically from train emb (seed 0);
    verified to reproduce the stored shards exactly (centroid_distance/item_cluster 0.0,
    cluster_geometry 9e-4).
  * the unit-normalized TRAIN item embeddings + keys — the kNN index for nn_geometry
    (+ provides cluster sizes). (Ship fp16 or a quantized cache.)

IMPORTANT: query embeddings must be unit-normalized the same way as training
(aide.features.driver.unit_rows) before calling. Self/alias exclusion is key-based, so it is
a no-op for genuinely-new test items (their keys aren't in the train index) and correctly
excludes self when re-deriving a training item (used by the verifier).
"""
from __future__ import annotations

import numpy as np

GEOM_GROUPS = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster"]
DEFAULT_KS = (4, 8, 32, 64)


def fit_centroids(emb_unit, coarse_k: int = 32, fine_k: int = 256, seed: int = 0) -> dict:
    """Re-fit the multi-resolution KMeans centroids (export-time; needs sklearn)."""
    from aide.features.derive_cluster import fit_multi_kmeans
    return fit_multi_kmeans(np.asarray(emb_unit, np.float32),
                            {"coarse": coarse_k, "fine": fine_k}, seed=seed)


def save_centroids(centroids: dict, path) -> None:
    np.savez_compressed(path, fine=np.asarray(centroids["fine"], np.float32),
                        coarse=np.asarray(centroids["coarse"], np.float32))


def load_centroids(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"fine": z["fine"], "coarse": z["coarse"]}


def _nn_geometry(query_emb, query_keys, train_emb, train_keys, Ks=DEFAULT_KS,
                 search_buffer: int = 2, alias_eps: float = 1e-6):
    """nn_geometry block (geo__local_density / dist_gap_1_to_K / lid_estimate) from kNN sims."""
    from aide.features.derive_nn import bruteforce_knn
    maxK = max(Ks); m = int(np.asarray(train_emb).shape[0])
    nreq = min(m, maxK + 1 + search_buffer)
    tk = np.asarray([str(k) for k in train_keys]); qk = np.asarray([str(k) for k in query_keys])
    idx, sim = bruteforce_knn(np.asarray(train_emb, np.float32), np.asarray(query_emb, np.float32), nreq)
    idx = np.asarray(idx); sim = np.asarray(sim); nq = int(np.asarray(query_emb).shape[0])
    geo = np.zeros((nq, 3), np.float32); thr = 1.0 - alias_eps
    for r in range(nq):
        nb = idx[r]; sm = sim[r]
        valid = (nb >= 0) & (tk[np.clip(nb, 0, m - 1)] != qk[r]) & (sm < thr)
        s = sm[valid][:maxK]
        if s.size:
            ld = float(s.mean()); gap = float(s[0] - s[-1])
            d = np.clip(1.0 - s, 1e-9, None); lid = 0.0
            if d.size > 1 and d[-1] > 0:
                lr = float(np.mean(np.log(d[-1] / d[:-1])))
                lid = 1.0 / lr if lr > 0 else 0.0
            geo[r] = [ld, gap, lid]
    return geo, ["geo__local_density", "geo__dist_gap_1_to_K", "geo__lid_estimate"]


def compute_geometry_block(query_emb, query_keys, train_emb, train_keys, centroids, Ks=DEFAULT_KS):
    """Assemble the full 525-col geometry block for query items, in training column order.

    Returns (X [nq, 525] float32, columns list). query_emb/train_emb must be unit-normalized.
    """
    from aide.features.cluster_fast import cluster_geometry_fast
    q = np.asarray(query_emb, np.float32); te = np.asarray(train_emb, np.float32)
    clus = cluster_geometry_fast(query_emb=q, row_ids=[str(k) for k in query_keys],
                                 centroids_by_res=centroids, all_emb=te)
    nn_geo, nn_cols = _nn_geometry(q, query_keys, te, train_keys, Ks=Ks)
    blocks = {
        "centroid_distance": (np.asarray(clus["centroid_distance"].X, np.float32),
                              list(clus["centroid_distance"].columns)),
        "cluster_geometry": (np.asarray(clus["cluster_geometry"].X, np.float32),
                             list(clus["cluster_geometry"].columns)),
        "nn_geometry": (nn_geo, nn_cols),
        "item_cluster": (np.asarray(clus["item_cluster"].X, np.float32),
                         list(clus["item_cluster"].columns)),
    }
    X, cols = [], []
    for g in GEOM_GROUPS:
        Xg, cg = blocks[g]; X.append(Xg); cols += cg
    return np.concatenate(X, axis=1), cols


__all__ = ["GEOM_GROUPS", "fit_centroids", "save_centroids", "load_centroids", "compute_geometry_block"]
