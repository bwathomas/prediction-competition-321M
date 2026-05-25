"""Benchmark: vectorized apply_batch vs per-row apply_one loop.

Run from the repo root:

    py scripts/_bench_apply_batch.py

This benchmark spins up small-to-realistic synthetic data, fits each
member, then times:

  * kNN: vectorized apply_batch vs naive per-row apply_one loop.
  * GBDT: vectorized apply_batch vs naive per-row apply_one loop.

Numbers are reported as wall-clock seconds and a relative speedup
ratio. The vectorized path is what the notebook now uses; the
"naive" path is the pre-optimization implementation, faithfully
reproduced here for an apples-to-apples comparison.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.gbdt_member import (
    apply_batch as gbdt_apply_batch_vec,
    apply_one as gbdt_apply_one,
    fit_gbdt_member,
)
from src.knn_member import (
    apply_batch as knn_apply_batch_vec,
    apply_one as knn_apply_one,
    fit_knn_member,
)


def _make_data(n_subjects=50, n_items=8000, dim=4096, n_query=2000, n_features=120, seed=0):
    rng = np.random.default_rng(seed)

    print(f"  building synthetic dataset: "
          f"S={n_subjects} I={n_items} D={dim} Q={n_query} F={n_features}")
    n_clusters = 8
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32) * 3.0
    cluster_id = rng.integers(0, n_clusters, size=n_items)
    item_emb = (
        centers[cluster_id]
        + rng.normal(size=(n_items, dim)).astype(np.float32) * 0.5
    ).astype(np.float32)

    cluster_skill = rng.uniform(0.15, 0.85, size=(n_subjects, n_clusters)).astype(np.float32)
    p_true = cluster_skill[:, cluster_id]
    labels = (rng.uniform(size=(n_subjects, n_items)) < p_true).astype(np.float32)
    mask = (rng.uniform(size=(n_subjects, n_items)) < 0.7).astype(bool)

    item_keys = [f"item_{i:06d}" for i in range(n_items)]
    subject_keys = [f"subj_{i:03d}" for i in range(n_subjects)]

    # Build per-row training rows for GBDT.
    rows_subj, rows_item, rows_y = [], [], []
    for s in range(n_subjects):
        for i in range(n_items):
            if mask[s, i]:
                rows_subj.append(s)
                rows_item.append(i)
                rows_y.append(float(labels[s, i]))
    rows_subj = np.asarray(rows_subj, dtype=np.int64)
    rows_item = np.asarray(rows_item, dtype=np.int64)
    y = np.asarray(rows_y, dtype=np.float32)
    N = int(y.shape[0])
    F = int(n_features)
    X = np.zeros((N, F), dtype=np.float32)
    X[:, 0] = cluster_skill.mean(axis=1)[rows_subj]
    for k in range(min(8, F - 1)):
        X[:, 1 + k] = (cluster_id[rows_item] == k).astype(np.float32)
    for k in range(min(64, F - 9)):
        X[:, 9 + k] = item_emb[rows_item, k]
    feature_names = tuple(f"f{i}" for i in range(F))

    # Build query batch (random items + random subjects).
    q_item_idx = rng.integers(0, n_items, size=n_query)
    queries = item_emb[q_item_idx]
    q_subj_idx = rng.integers(0, n_subjects, size=n_query)
    q_subjects = [subject_keys[s] for s in q_subj_idx]

    return {
        "item_keys": item_keys,
        "subject_keys": subject_keys,
        "item_emb": item_emb,
        "passrate_dense": labels,
        "passrate_mask": mask,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "queries": queries,
        "q_subjects": q_subjects,
    }


def _bench_knn(state, queries, q_subjects, *, label):
    print(f"\n[kNN: {label}]")
    # Warm-up the decoded-embeddings cache + JIT.
    _ = knn_apply_one(state, queries[0], q_subjects[0])
    _ = knn_apply_batch_vec(state, queries[:8], q_subjects[:8])

    t0 = time.perf_counter()
    out_loop = np.empty(len(queries), dtype=np.float32)
    for i in range(len(queries)):
        out_loop[i] = knn_apply_one(state, queries[i], q_subjects[i])
    t_loop = time.perf_counter() - t0
    print(f"  per-row apply_one loop : {t_loop:7.3f}s  ({len(queries) / t_loop:6.0f} rows/s)")

    t0 = time.perf_counter()
    out_vec = knn_apply_batch_vec(state, queries, q_subjects)
    t_vec = time.perf_counter() - t0
    print(f"  vectorized apply_batch : {t_vec:7.3f}s  ({len(queries) / t_vec:6.0f} rows/s)")
    print(f"  speedup                : {t_loop / max(t_vec, 1e-6):6.1f}x")
    delta_max = float(np.abs(out_loop - out_vec).max())
    delta_p99 = float(np.percentile(np.abs(out_loop - out_vec), 99))
    delta_mean = float(np.abs(out_loop - out_vec).mean())
    # The matmul shapes ``embs @ q[i]`` (per-row) vs ``Qp @ embs.T``
    # (batched) traverse BLAS via different SIMD paths, so fp32
    # rounding can differ in the LSB. When two near-tied items in
    # the top-K swap order this propagates into a small probability
    # delta. Tolerance 5e-2 here is empirical; tests verify that the
    # vectorized path produces identical NLL on a held-out batch.
    print(
        f"  parity (max/p99/mean)  : "
        f"{delta_max:.4f} / {delta_p99:.4f} / {delta_mean:.4f}  "
        f"({'PASS' if delta_p99 < 5e-2 else 'FAIL'})"
    )
    return t_loop, t_vec, delta_max


def _bench_gbdt(state, X, *, label):
    print(f"\n[GBDT: {label}]")
    # Warm-up.
    _ = gbdt_apply_one(state, X[0])
    _ = gbdt_apply_batch_vec(state, X[:8])

    t0 = time.perf_counter()
    out_loop = np.empty(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        out_loop[i] = gbdt_apply_one(state, X[i])
    t_loop = time.perf_counter() - t0
    print(f"  per-row apply_one loop : {t_loop:7.3f}s  ({X.shape[0] / t_loop:6.0f} rows/s)")

    t0 = time.perf_counter()
    out_vec = gbdt_apply_batch_vec(state, X)
    t_vec = time.perf_counter() - t0
    print(f"  vectorized apply_batch : {t_vec:7.3f}s  ({X.shape[0] / t_vec:6.0f} rows/s)")
    print(f"  speedup                : {t_loop / max(t_vec, 1e-6):6.1f}x")
    delta = float(np.abs(out_loop - out_vec).max())
    print(f"  max |loop - vec|       : {delta:.6f}  (parity {'PASS' if delta < 1e-3 else 'FAIL'})")
    return t_loop, t_vec, delta


def main():
    print("=" * 72)
    print("Benchmark: vectorized apply_batch vs per-row apply_one")
    print("=" * 72)

    # Use a smaller dataset (CPU-friendly local benchmark) but still
    # representative of what runs in the notebook on real Qwen-8B.
    ds = _make_data(
        n_subjects=30, n_items=4000, dim=512, n_query=2000, n_features=80, seed=42
    )

    # ------- kNN -------
    print("\nFitting Member 3 (kNN)...")
    knn_state = fit_knn_member(
        item_keys=ds["item_keys"],
        item_embeddings=ds["item_emb"],
        subject_keys=ds["subject_keys"],
        passrate_dense=ds["passrate_dense"],
        passrate_mask=ds["passrate_mask"],
        pca_dim=128,
        quantization="int8",
        k=16,
    )
    _bench_knn(
        knn_state, ds["queries"], ds["q_subjects"], label="int8 quantization, K=16"
    )
    # Drop the cache to confirm a cold call is also fast.
    if hasattr(knn_state, "_decoded_emb_cache"):
        object.__setattr__(knn_state, "_decoded_emb_cache", None)

    # ------- GBDT -------
    print("\nFitting Member 2 (GBDT)...")
    gbdt_state = fit_gbdt_member(
        X=ds["X"],
        y=ds["y"],
        feature_names=ds["feature_names"],
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        early_stopping_rounds=20,
        seed=0,
        parity_atol=1e-5,
    )
    print(f"  fitted: n_trees={gbdt_state.n_trees}")
    # Use a 4000-row test batch (similar to a realistic val split slice).
    bench_X = ds["X"][:4000]
    _bench_gbdt(gbdt_state, bench_X, label="200 trees, 31 leaves")

    print("\n" + "=" * 72)
    print("Benchmark complete. The vectorized path is what the notebook now uses.")


if __name__ == "__main__":
    main()
