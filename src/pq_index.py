"""Numpy-only product-quantized (PQ) embedding index for runtime kNN (nn_geometry / nn features).

Shipping the full train item embeddings (311k×4096 fp32 ≈ 5 GB, ×3 families) is infeasible; PCA loses
the kNN structure (neighbors live in low-variance dirs). PQ-512 keeps a small index (~165 MB/family,
recall@64 ≈ 0.82) and decodes with numpy only.

Built offline by the PQ-build job -> ``pqidx_<fam>.npz`` {codebook [M,256,ds] f32, codes [N,M] uint8,
item_keys [N], M, ds, D}. At runtime the QUERY is full-precision; we use asymmetric distance computation
(ADC): per subvector m, LUT[m] = query_chunk_m @ codebook[m].T, then score[item] = sum_m LUT[m, code[item,m]]
(== approx cosine dot for unit-normalized embeddings). No cuML/FAISS needed.
"""
from __future__ import annotations

import numpy as np


def load_pq_index(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"codebook": z["codebook"], "codes": z["codes"], "item_keys": z["item_keys"].astype(str),
                "M": int(z["M"]), "ds": int(z["ds"]), "D": int(z["D"])}


def pq_scores(query_emb, pq: dict, chunk: int = 512) -> np.ndarray:
    """ADC approximate dot-product scores [nq, N] (query full-precision vs PQ-coded index)."""
    cb, codes, M, ds = pq["codebook"], pq["codes"], pq["M"], pq["ds"]
    q = np.asarray(query_emb, np.float32); nq, N = q.shape[0], codes.shape[0]
    out = np.empty((nq, N), np.float32)
    for c0 in range(0, nq, chunk):
        qc = q[c0:c0 + chunk]; sc = np.zeros((qc.shape[0], N), np.float32)
        for m in range(M):
            lut = qc[:, m * ds:(m + 1) * ds] @ cb[m].T   # [nqc, 256]
            sc += lut[:, codes[:, m]]
        out[c0:c0 + chunk] = sc
    return out


def pq_knn(query_emb, pq: dict, k: int):
    """Top-k (idx, sim) per query via ADC. sim is the approximate cosine (dot)."""
    sc = pq_scores(query_emb, pq)
    k = min(k, sc.shape[1])
    idx = np.argpartition(-sc, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sc, idx, 1)
    o = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(part, o, 1)


__all__ = ["load_pq_index", "pq_scores", "pq_knn"]
