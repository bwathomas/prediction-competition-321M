"""K-means clustering on cached item embeddings.

We fit k-means on the (already-cached) item embeddings used by the rest of
the pipeline and persist the centroids so val / test items can be assigned
on the fly via nearest-centroid lookup. Cluster id ``0`` is reserved for
the UNK fallback (used only when the centroid artifact is missing).

The cluster id is consumed by the model variants as an ``nn.Embedding(K+1,
d_cluster)`` lookup that is concatenated with the rest of the residual-MLP
inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("clustering")


# Fit + assign ---------------------------------------------------------------


def fit_kmeans(
    item_embeddings: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
) -> np.ndarray:
    """Fit k-means on ``item_embeddings`` and return centroids ``[k, d]``."""
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


def assign_clusters(
    centroids: np.ndarray,
    item_embeddings: np.ndarray,
    *,
    batch_size: int = 16384,
) -> np.ndarray:
    """Assign each row of ``item_embeddings`` to a cluster id in ``[1, k]``.

    Index ``0`` is reserved for the UNK fallback (used when the centroid file
    is missing). Uses the squared-norm trick for memory efficiency and
    batches rows so very large item sets don't materialize a ``[N, k]``
    distance matrix in one shot.
    """
    X = np.asarray(item_embeddings, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)
    n, _ = X.shape
    out = np.empty(n, dtype=np.int64)
    c_norm = (C * C).sum(axis=1)  # [k]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]
        x_norm = (chunk * chunk).sum(axis=1, keepdims=True)
        d2 = x_norm + c_norm[None, :] - 2.0 * (chunk @ C.T)
        out[start:end] = d2.argmin(axis=1)
    # offset by 1 so that 0 is reserved for the UNK fallback id
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
) -> tuple[np.ndarray, dict[str, int]]:
    """End-to-end: fit, assign, persist. Returns (centroids, assignments)."""
    centroids = None if overwrite else load_centroids(centroids_path)
    assignments = None if overwrite else load_assignments(assignments_path)

    if centroids is None or assignments is None:
        centroids = fit_kmeans(item_embeddings, k=k, seed=seed)
        ids = assign_clusters(centroids, item_embeddings)
        save_centroids(centroids, centroids_path)
        save_assignments(item_keys, ids, assignments_path)
        assignments = {str(item_keys[i]): int(ids[i]) for i in range(len(item_keys))}
        LOG.info(
            "Fit + persisted clusters: k=%d centroids=%s assignments=%s",
            k,
            centroids_path,
            assignments_path,
        )
    else:
        LOG.info(
            "Reused cached clusters: centroids=%s assignments=%s",
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
