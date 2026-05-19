"""K-means clustering on cached item embeddings.

We fit k-means on the (already-cached) item embeddings used by the rest of
the pipeline and persist the centroids so val / test items can be assigned
on the fly via nearest-centroid lookup. Cluster id ``0`` is reserved for
the UNK fallback (used only when the centroid artifact is missing).

The cluster id is consumed by the model variants as an ``nn.Embedding(K+1,
d_cluster)`` lookup that is concatenated with the rest of the residual-MLP
inputs.

Backend selection
-----------------
``fit_and_assign`` automatically uses **FAISS GPU k-means** when faiss is
installed and at least one GPU is visible. This is ~50-100x faster than the
sklearn fallback on A100s for the full corpus (~200k items x 1024 dims).
The sklearn path is kept as a safety net for environments without faiss
(local laptops, CPU-only CI). The on-disk artifact format is identical
across backends so notebooks / runtime code do not need to know which path
was taken.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x=None, *args, **kwargs):  # type: ignore[misc]
        return x if x is not None else range(0)

LOG = logging.getLogger("clustering")


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _faiss_gpu_available() -> tuple[bool, int]:
    """Return ``(available, n_gpus)`` for FAISS GPU k-means.

    Returns ``(False, 0)`` if faiss isn't installed, has no GPU support
    compiled in, or sees zero GPUs at runtime.
    """
    try:
        import faiss  # type: ignore
    except Exception:
        return False, 0
    try:
        n = int(faiss.get_num_gpus())
    except Exception:
        return False, 0
    return (n > 0), n


# Fit + assign ---------------------------------------------------------------


def fit_kmeans(
    item_embeddings: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
) -> np.ndarray:
    """Fit sklearn k-means on ``item_embeddings`` and return centroids.

    CPU fallback. Use ``fit_and_assign`` for the GPU-accelerated path.
    """
    from sklearn.cluster import KMeans

    X = np.asarray(item_embeddings, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(
            f"item_embeddings must be 2-D; got shape {X.shape}"
        )
    if X.shape[0] < k:
        raise ValueError(
            f"k={k} exceeds the number of items ({X.shape[0]}); pick a smaller k"
        )
    LOG.info(
        "Fitting k-means: n=%d d=%d k=%d seed=%d n_init=%d",
        X.shape[0],
        X.shape[1],
        k,
        seed,
        n_init,
    )
    km = KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    km.fit(X)
    return km.cluster_centers_.astype(np.float32, copy=False)


def _fit_kmeans_faiss_gpu(
    item_embeddings: np.ndarray,
    *,
    k: int,
    seed: int,
    niter: int,
    nredo: int,
    gpu_id: int,
) -> np.ndarray:
    """Fit k-means on GPU via FAISS. Returns centroids ``[k, d]``."""
    import faiss  # type: ignore

    X = np.ascontiguousarray(np.asarray(item_embeddings, dtype=np.float32))
    n, d = X.shape
    LOG.info(
        "FAISS GPU k-means: n=%d d=%d k=%d niter=%d nredo=%d gpu_id=%d",
        n, d, int(k), int(niter), int(nredo), int(gpu_id),
    )
    kmeans = faiss.Kmeans(
        d=int(d),
        k=int(k),
        niter=int(niter),
        nredo=int(nredo),
        seed=int(seed),
        gpu=True,
        verbose=True,
    )
    t0 = time.time()
    kmeans.train(X)
    LOG.info("FAISS GPU k-means trained in %.1fs", time.time() - t0)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    return np.ascontiguousarray(centroids)


def assign_clusters(
    centroids: np.ndarray,
    item_embeddings: np.ndarray,
    *,
    batch_size: int = 16384,
) -> np.ndarray:
    """Assign each row of ``item_embeddings`` to a cluster id in ``[1, k]``.

    Index ``0`` is reserved for the UNK fallback (used when the centroid file
    is missing). Prefers FAISS (GPU if available, else CPU) for the
    nearest-centroid search; falls back to a pure-numpy chunked argmin when
    faiss is not installed.
    """
    X = np.asarray(item_embeddings, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"item_embeddings must be 2-D; got shape {X.shape}")
    if C.ndim != 2:
        raise ValueError(f"centroids must be 2-D; got shape {C.shape}")
    if X.shape[1] != C.shape[1]:
        raise ValueError(
            f"dim mismatch: items {X.shape[1]} vs centroids {C.shape[1]}"
        )

    try:
        import faiss  # type: ignore

        d = C.shape[1]
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(np.ascontiguousarray(C))

        gpu_available, _ = _faiss_gpu_available()
        if gpu_available:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception as exc:  # noqa: BLE001
                LOG.info(
                    "FAISS GPU index construction failed (%s); using CPU index",
                    exc,
                )
                index = cpu_index
        else:
            index = cpu_index

        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            _, labels = index.search(
                np.ascontiguousarray(X[start:end]), 1
            )
            out[start:end] = labels.reshape(-1).astype(np.int64)
        # offset by 1 so that 0 is reserved for the UNK fallback id
        return (out + 1).astype(np.int64, copy=False)
    except Exception as exc:  # noqa: BLE001
        LOG.info(
            "FAISS unavailable for assign (%s); using numpy chunked argmin",
            exc,
        )

    n, _ = X.shape
    out = np.empty(n, dtype=np.int64)
    c_norm = (C * C).sum(axis=1)  # [k]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]
        x_norm = (chunk * chunk).sum(axis=1, keepdims=True)
        d2 = x_norm + c_norm[None, :] - 2.0 * (chunk @ C.T)
        out[start:end] = d2.argmin(axis=1)
    return (out + 1).astype(np.int64, copy=False)


# Persistence ----------------------------------------------------------------


def save_centroids(centroids: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(centroids, dtype=np.float32))
    return path


def load_centroids(path: Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32, copy=False)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load centroids from %s (%s)", path, exc)
        return None


def save_assignments(
    item_keys: Sequence[str],
    cluster_ids: Sequence[int],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "item_key": [str(k) for k in item_keys],
            "cluster_id": np.asarray(cluster_ids, dtype=np.int64),
        }
    )
    df.to_parquet(path, index=False)
    return path


def load_assignments(path: Path) -> dict[str, int] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return {
            str(k): int(v)
            for k, v in zip(df["item_key"].astype(str), df["cluster_id"].astype(int))
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load %s (%s)", path, exc)
        return None


def _cached_artifacts_match(
    centroids: np.ndarray | None,
    assignments: dict[str, int] | None,
    *,
    k: int,
    d: int,
    item_keys: Sequence[str],
) -> bool:
    """Return True iff cached centroids + assignments are reusable for this call."""
    if centroids is None or assignments is None:
        return False
    if centroids.shape != (k, d):
        return False
    if len(assignments) != len(item_keys):
        return False
    # Cheap membership check: the cache was written from this exact key set.
    return all(str(k) in assignments for k in item_keys)


# Convenience wrapper --------------------------------------------------------


def fit_and_assign(
    item_keys: Sequence[str],
    item_embeddings: np.ndarray,
    *,
    k: int,
    seed: int,
    centroids_path: Path,
    assignments_path: Path,
    overwrite: bool = False,
    niter: int = 50,
    nredo: int = 1,
    gpu_id: int = 0,
    assign_batch_size: int = 65536,
    backend: str = "auto",
) -> tuple[np.ndarray, dict[str, int]]:
    """End-to-end: fit, assign, persist. Returns (centroids, assignments).

    ``backend`` is ``"auto"`` (FAISS GPU when available, else sklearn CPU),
    ``"faiss_gpu"`` (require GPU), or ``"sklearn"`` (force CPU). The chosen
    backend is logged so the notebook can confirm the GPU path was taken.

    ``niter`` / ``nredo`` only apply to the FAISS GPU path. For sklearn we
    use its own default ``n_init=4``. The on-disk artifact format is the
    same for either backend so downstream consumers don't need to know.
    """
    X = np.asarray(item_embeddings, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"item_embeddings must be 2-D; got shape {X.shape}")
    if X.shape[0] != len(item_keys):
        raise ValueError(
            f"item_keys ({len(item_keys)}) and embeddings ({X.shape[0]}) must align"
        )
    if X.shape[0] < k:
        raise ValueError(
            f"k={k} exceeds the number of items ({X.shape[0]}); pick a smaller k"
        )
    n, d = X.shape

    if not overwrite:
        centroids = load_centroids(centroids_path)
        assignments = load_assignments(assignments_path)
        if _cached_artifacts_match(
            centroids, assignments, k=int(k), d=int(d), item_keys=item_keys
        ):
            LOG.info(
                "Reused cached clusters: centroids=%s assignments=%s",
                centroids_path,
                assignments_path,
            )
            return centroids, assignments  # type: ignore[return-value]

    backend = (backend or "auto").lower()
    gpu_ok, n_gpus = _faiss_gpu_available()
    if backend == "faiss_gpu" and not gpu_ok:
        raise RuntimeError(
            "backend='faiss_gpu' requested but FAISS reports 0 GPUs. "
            "Install/use a GPU-enabled FAISS build, e.g. faiss-gpu-cu12."
        )
    use_faiss_gpu = (
        backend == "faiss_gpu"
        or (backend == "auto" and gpu_ok)
    )

    if use_faiss_gpu:
        LOG.info("Clustering backend: FAISS GPU (n_gpus=%d)", n_gpus)
        centroids = _fit_kmeans_faiss_gpu(
            X, k=int(k), seed=int(seed), niter=int(niter),
            nredo=int(nredo), gpu_id=int(gpu_id),
        )
    else:
        LOG.info("Clustering backend: sklearn CPU")
        centroids = fit_kmeans(X, k=int(k), seed=int(seed))

    cluster_ids = assign_clusters(
        centroids, X, batch_size=int(assign_batch_size)
    )
    save_centroids(centroids, centroids_path)
    save_assignments(item_keys, cluster_ids, assignments_path)

    assignments = {
        str(item_keys[i]): int(cluster_ids[i])
        for i in tqdm(
            range(n),
            desc="Building assignment dict",
            unit="item",
            leave=False,
        )
    }
    LOG.info(
        "Fit + persisted clusters: k=%d centroids=%s assignments=%s",
        k,
        centroids_path,
        assignments_path,
    )
    return centroids, assignments


def cluster_id_for_embedding(
    item_embedding: np.ndarray,
    centroids: np.ndarray | None,
) -> int:
    """Return the cluster id for a single embedding (or 0 if no centroids).

    Useful at test time when an unseen item isn't in the assignment table.
    """
    if centroids is None or centroids.size == 0:
        return 0
    ids = assign_clusters(centroids, item_embedding[None, :])
    return int(ids[0])


__all__ = [
    "assign_clusters",
    "cluster_id_for_embedding",
    "fit_and_assign",
    "fit_kmeans",
    "load_assignments",
    "load_centroids",
    "save_assignments",
    "save_centroids",
]
