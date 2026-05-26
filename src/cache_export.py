"""Build the quantized training-item embedding cache shipped with submissions.

The goal is a *small* (<= 200 MB) artifact bundle that lets the runtime
``predict()`` find the nearest training items to a test embedding cheaply,
without pulling in the full fp16 matrix (which can easily be 1 GB+).

Pipeline:

1. Load the full fp16 / fp32 item embeddings from ``items.parquet``.
2. Optional PCA to a smaller dimension (we recommend 256 -- usually a slight
   *gain* on nearest-neighbor quality because it down-weights noise dims).
   Both the cached vectors AND the query vector go through the same PCA
   matrix at lookup time.
3. Per-row symmetric int8 quantization with a per-row scale stored alongside.
4. Optional FAISS index (HNSW32 by default) for sub-millisecond lookup over
   200k items. We ship the raw int8 + scales regardless so a brute-force
   fallback path always works (matters on Codabench if FAISS fails to install).
5. Optional per-subject pass-rate aggregation, either as a sparse CSR matrix
   over (subject, item) pairs or as a dense per-subject mean over k clusters.
6. Hard size cap: if the resulting bundle exceeds ``max_bundle_size_mb`` we
   fail loudly. This is the line of defense against shipping a 1 GB ZIP.

Layout written to ``out_dir``:

    out_dir/
        cache_meta.json
        embeddings_int8.npy
        scales.npy
        pca.npy                 # optional
        item_keys.parquet       # row index -> item_key + minimal metadata
        faiss.index             # optional
        subject_passrates.npz   # optional (sparse) -- (n_subjects, n_items)
        subject_cluster_passrates.npy   # optional (dense) -- (n_subjects, k_clusters)
        subject_keys.parquet    # subject row index -> subject_key
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("cache_export")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CacheExportConfig:
    enabled: bool = True
    quantize: str = "int8"           # int8 | fp16 | none
    pca_dim: int | None = 256        # None = skip PCA
    include_faiss_index: bool = True
    faiss_index_type: str = "HNSW32"  # HNSW32 | IVF | flat
    max_bundle_size_mb: float = 200.0
    passrate_format: str = "cluster"  # cluster | sparse | none
    encoder_id: str = ""
    query_prefix: str = ""
    passage_prefix: str = ""
    pca_seed: int = 0
    # Runtime K for NN feature computation. When the runtime cache is asked
    # for NN features, it queries this many neighbors and aggregates the
    # locked 8-scalar feature schema.
    runtime_k: int = 16


@dataclass
class CacheExportResult:
    out_dir: Path
    written_files: list[str] = field(default_factory=list)
    sizes_mb: dict[str, float] = field(default_factory=dict)
    total_mb: float = 0.0
    meta: dict = field(default_factory=dict)
    failed: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Loaders for the per-encoder parquet caches
# ---------------------------------------------------------------------------


def load_item_embeddings(items_parquet: Path) -> tuple[list[str], np.ndarray]:
    """Load (keys, matrix) from items.parquet. Matrix is fp32 [N, D]."""
    if not Path(items_parquet).exists():
        raise FileNotFoundError(f"items.parquet not found at {items_parquet}")
    df = pd.read_parquet(items_parquet)
    if "embedding" not in df.columns:
        raise ValueError(f"items.parquet missing 'embedding' column: cols={list(df.columns)}")
    keys = df.iloc[:, 0].astype(str).tolist()
    embs_list = df["embedding"].tolist()
    # Vectorized conversion: stack-of-lists -> [N, D] fp32.
    matrix = np.asarray(embs_list, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"unexpected embedding shape: {matrix.shape}")
    return keys, matrix


# ---------------------------------------------------------------------------
# PCA (deterministic, no sklearn dependency required at runtime)
# ---------------------------------------------------------------------------


def fit_pca(matrix: np.ndarray, *, n_components: int, seed: int = 0) -> np.ndarray:
    """Fit a PCA projection. Returns components of shape [D, n_components].

    Uses ``np.linalg.svd`` on the mean-centered matrix; deterministic and
    portable. We do NOT recenter at query time -- the projection matrix is
    applied to raw vectors and the cached vectors are projected with the
    same matrix, so any centering cancels out as long as both sides use
    the same `project()` function.
    """
    if matrix.shape[1] <= n_components:
        # Identity projection if requested dim >= source dim.
        return np.eye(matrix.shape[1], dtype=np.float32)

    rng = np.random.default_rng(seed)
    # Subsample rows if huge (full SVD on 200k x 4k is wasteful).
    n_rows = matrix.shape[0]
    cap = 50_000
    if n_rows > cap:
        idx = rng.choice(n_rows, size=cap, replace=False)
        X = matrix[idx]
    else:
        X = matrix
    X = X.astype(np.float32, copy=False)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Compute right singular vectors only.
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: covariance eigendecomposition.
        cov = Xc.T @ Xc / max(1, Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(eigvals)[::-1]
        vt = eigvecs[:, order].T.astype(np.float32)
    components = vt[:n_components].T.astype(np.float32)  # [D, n_components]
    return components


def apply_pca(matrix: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project matrix [N, D] through components [D, n_components]."""
    return (matrix.astype(np.float32, copy=False) @ components).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_int8_per_row(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization.

    For each row v: scale = max(abs(v)) / 127, q = round(v / scale).
    Reconstruct as q * scale at query time. Per-row (not per-tensor)
    keeps nearest-neighbor quality high without much storage overhead.
    """
    matrix = matrix.astype(np.float32, copy=False)
    abs_max = np.max(np.abs(matrix), axis=1)
    scale = np.where(abs_max > 0, abs_max / 127.0, 1.0).astype(np.float32)
    q = np.round(matrix / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales[:, None]


# ---------------------------------------------------------------------------
# FAISS index (optional)
# ---------------------------------------------------------------------------


def build_faiss_index(
    matrix: np.ndarray, *, index_type: str = "HNSW32"
) -> tuple[object | None, str | None]:
    """Build a FAISS index of the given type. Returns (index, error_or_None)."""
    try:
        import faiss  # type: ignore
    except Exception as exc:
        return None, f"faiss not installed ({type(exc).__name__}: {exc})"

    n, d = matrix.shape
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    try:
        kind = (index_type or "HNSW32").upper()
        if kind.startswith("HNSW"):
            try:
                m = int(kind.replace("HNSW", "")) or 32
            except ValueError:
                m = 32
            index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            index.add(matrix)
        elif kind == "IVF":
            nlist = max(1, int(math.sqrt(n)))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(matrix)
            index.add(matrix)
        else:  # flat
            index = faiss.IndexFlatIP(d)
            index.add(matrix)
        return index, None
    except Exception as exc:  # noqa: BLE001
        return None, f"faiss build failed ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# Per-subject pass-rate aggregations
# ---------------------------------------------------------------------------


def build_cluster_passrate_table(
    train_df: pd.DataFrame,
    *,
    cluster_assignments: Mapping[str, int],
    n_clusters: int,
) -> tuple[np.ndarray, list[str]]:
    """Per-(subject, cluster) mean pass rate matrix.

    Returns ``(matrix[n_subjects, n_clusters+1], subject_keys)``. Cluster id 0
    is reserved for UNK and gets the global subject mean as a fallback.
    """
    required = {"subject_key", "item_key", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"train_df missing required cols: {required - set(train_df.columns)}")
    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["cluster_id"] = (
        df["item_key"]
        .astype(str)
        .map(lambda k: int(cluster_assignments.get(k, 0)))
        .astype(np.int64)
    )

    # Build per-(subject, cluster) mean. We always include cluster 0 as UNK
    # which gets the global per-subject mean.
    n_cols = int(n_clusters) + 1
    subject_keys = sorted(df["subject_key"].astype(str).unique().tolist())
    s_to_idx = {k: i for i, k in enumerate(subject_keys)}
    n_rows = len(subject_keys)

    sums = np.zeros((n_rows, n_cols), dtype=np.float64)
    counts = np.zeros((n_rows, n_cols), dtype=np.float64)
    s_idx = df["subject_key"].astype(str).map(s_to_idx).to_numpy()
    c_idx = df["cluster_id"].to_numpy()
    lbl = df["label"].astype(float).to_numpy()
    np.add.at(sums, (s_idx, c_idx), lbl)
    np.add.at(counts, (s_idx, c_idx), 1.0)

    means = np.divide(
        sums, np.maximum(counts, 1.0), out=np.zeros_like(sums), where=counts > 0
    )
    # Fill UNK column (index 0) with global per-subject mean.
    subj_global = sums.sum(axis=1) / np.maximum(counts.sum(axis=1), 1.0)
    # If a (subject, cluster) cell is unobserved we fall back to subject mean.
    no_obs = counts == 0
    means = np.where(no_obs, subj_global[:, None], means)
    means[:, 0] = subj_global
    return means.astype(np.float32), subject_keys


def build_sparse_passrate_csr(
    train_df: pd.DataFrame,
    *,
    item_keys: list[str],
) -> tuple[object, list[str]]:
    """Sparse CSR (n_subjects, n_items) matrix of mean pass rates.

    Requires scipy. Only safe when training data is genuinely sparse.
    """
    from scipy import sparse  # type: ignore

    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    item_idx = {k: i for i, k in enumerate(item_keys)}
    df = df[df["item_key"].isin(item_idx)]

    subject_keys = sorted(df["subject_key"].unique().tolist())
    s_to_idx = {k: i for i, k in enumerate(subject_keys)}

    grouped = df.groupby(["subject_key", "item_key"], sort=False)["label"].mean().reset_index()
    rows = grouped["subject_key"].map(s_to_idx).to_numpy()
    cols = grouped["item_key"].map(item_idx).to_numpy()
    vals = grouped["label"].astype(np.float32).to_numpy()
    mat = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(subject_keys), len(item_keys)),
        dtype=np.float32,
    )
    return mat, subject_keys


# ---------------------------------------------------------------------------
# NN-features cache: subject_passrate + mask + subject_key_to_id + cfg
# ---------------------------------------------------------------------------


def build_nn_passrate_csr(
    train_df: pd.DataFrame,
    *,
    item_keys: list[str],
    subject_to_id: Mapping[str, int],
):
    """Sparse CSR pass-rate + mask matrices keyed by the trainer's indexer.

    Unlike :func:`build_sparse_passrate_csr`, this function reuses the
    training-time ``subject_to_id`` mapping so the runtime can look up a
    subject by hashing its content and indexing into the same row. The
    mask matrix is binary (entries are 1.0 wherever an observation exists)
    and the pass-rate stores the mean label in float32.

    Returns ``(passrate_csr, mask_csr)``.
    """
    from scipy import sparse  # type: ignore

    required = {"subject_key", "item_key", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(
            f"train_df missing required cols: {sorted(required - set(train_df.columns))}"
        )
    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    item_idx = {k: i for i, k in enumerate(item_keys)}
    df = df[df["item_key"].isin(item_idx)]
    df = df[df["subject_key"].isin(subject_to_id)]

    n_subjects = int(max(int(max(subject_to_id.values())) + 1, 1))
    n_items = int(max(len(item_keys), 1))
    if df.empty:
        empty = sparse.csr_matrix(
            (n_subjects, n_items), dtype=np.float32
        )
        return empty, empty.copy()

    grouped = (
        df.groupby(["subject_key", "item_key"], sort=False)["label"]
        .mean()
        .reset_index()
    )
    rows = grouped["subject_key"].map(subject_to_id).to_numpy(dtype=np.int64)
    cols = grouped["item_key"].map(item_idx).to_numpy(dtype=np.int64)
    vals = grouped["label"].astype(np.float32).to_numpy()
    passrate = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_subjects, n_items),
        dtype=np.float32,
    )
    mask = sparse.csr_matrix(
        (np.ones_like(vals, dtype=np.float32), (rows, cols)),
        shape=(n_subjects, n_items),
        dtype=np.float32,
    )
    return passrate, mask


# ---------------------------------------------------------------------------
# Item-key metadata (kept tiny so the bundle stays small)
# ---------------------------------------------------------------------------


def build_item_keys_df(
    item_keys: list[str],
    items_meta_df: pd.DataFrame | None,
    cluster_assignments: Mapping[str, int] | None,
) -> pd.DataFrame:
    """Build the row-index -> key mapping shipped in ``item_keys.parquet``.

    Includes (only) benchmark / condition / cluster_id so the runtime can
    weight NN features by metadata without shipping the full item_content
    text (which can be huge).
    """
    df = pd.DataFrame(
        {"row_index": np.arange(len(item_keys), dtype=np.int32), "item_key": item_keys}
    )
    if items_meta_df is not None and len(items_meta_df):
        meta = (
            items_meta_df[["item_key", "benchmark", "condition"]]
            .drop_duplicates(subset=["item_key"])
            .astype({"item_key": str})
        )
        df = df.merge(meta, on="item_key", how="left")
    if cluster_assignments is not None:
        df["cluster_id"] = (
            df["item_key"].map(lambda k: int(cluster_assignments.get(k, 0))).astype(np.int32)
        )
    else:
        df["cluster_id"] = np.zeros(len(df), dtype=np.int32)
    return df


# ---------------------------------------------------------------------------
# Size reporting
# ---------------------------------------------------------------------------


def _file_sizes(out_dir: Path) -> tuple[dict[str, float], float]:
    sizes: dict[str, float] = {}
    total = 0.0
    for p in sorted(Path(out_dir).iterdir()):
        if p.is_file():
            mb = p.stat().st_size / (1024 * 1024)
            sizes[p.name] = float(mb)
            total += mb
    return sizes, float(total)


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def export_item_cache(
    *,
    items_parquet_path: Path,
    out_dir: Path,
    cfg: CacheExportConfig,
    items_meta_df: pd.DataFrame | None = None,
    cluster_assignments: Mapping[str, int] | None = None,
    n_clusters: int = 0,
    train_df: pd.DataFrame | None = None,
    nn_features_cfg: Mapping | None = None,
    subject_to_id: Mapping[str, int] | None = None,
    conditional_context: object | None = None,
    bc_id_to_age: object | None = None,
) -> CacheExportResult:
    """Build the submission-side training-item cache.

    When ``nn_features_cfg`` and ``subject_to_id`` are provided, the cache
    additionally emits the four NN-feature artifacts the runtime expects:

    * ``subject_passrate.npz`` -- sparse [n_subjects, n_items] float32 matrix
      of mean labels per (subject, item) pair (using the trainer's subject
      indexer).
    * ``subject_passrate_mask.npz`` -- matching binary mask in CSR form.
    * ``subject_key_to_id.json`` -- explicit subject_key -> integer id map
      (the indexer used at training time). Shipping this map is what keeps
      runtime aligned with training; relying on text-only re-derivation
      silently drifts.
    * ``nn_features_config.json`` -- the locked NN feature schema so the
      runtime can verify a match before consuming the cache.

    Returns a CacheExportResult. If the bundle exceeds
    ``cfg.max_bundle_size_mb`` the result is marked failed.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full embeddings.
    item_keys, fp_matrix = load_item_embeddings(items_parquet_path)
    n_items, src_dim = fp_matrix.shape
    LOG.info("Loaded %d items x %d dims from %s", n_items, src_dim, items_parquet_path)

    # 2. Optional PCA.
    if cfg.pca_dim and cfg.pca_dim > 0 and cfg.pca_dim < src_dim:
        components = fit_pca(fp_matrix, n_components=int(cfg.pca_dim), seed=cfg.pca_seed)
        proj_matrix = apply_pca(fp_matrix, components)
        np.save(out_dir / "pca.npy", components)
        LOG.info(
            "PCA: %d -> %d (explained dims, projected matrix shape %s)",
            src_dim,
            cfg.pca_dim,
            proj_matrix.shape,
        )
    else:
        components = None
        proj_matrix = fp_matrix.astype(np.float32, copy=False)

    # 3. Quantize.
    if cfg.quantize == "int8":
        q, scales = quantize_int8_per_row(proj_matrix)
        np.save(out_dir / "embeddings_int8.npy", q)
        np.save(out_dir / "scales.npy", scales.astype(np.float32))
        stored_dtype = "int8"
    elif cfg.quantize == "fp16":
        np.save(out_dir / "embeddings_int8.npy", proj_matrix.astype(np.float16))
        np.save(out_dir / "scales.npy", np.ones(n_items, dtype=np.float32))
        stored_dtype = "float16"
    else:
        np.save(out_dir / "embeddings_int8.npy", proj_matrix.astype(np.float32))
        np.save(out_dir / "scales.npy", np.ones(n_items, dtype=np.float32))
        stored_dtype = "float32"

    # 4. Optional FAISS index (built on the dequantized fp32 projection so
    #    the index distances match what the brute-force fallback would
    #    compute).
    faiss_err: str | None = None
    if cfg.include_faiss_index:
        index, faiss_err = build_faiss_index(proj_matrix, index_type=cfg.faiss_index_type)
        if index is not None:
            try:
                import faiss  # type: ignore

                faiss.write_index(index, str(out_dir / "faiss.index"))
            except Exception as exc:  # noqa: BLE001
                faiss_err = f"faiss.write_index failed: {exc}"
        else:
            LOG.warning("Skipping FAISS index: %s", faiss_err)

    # 5. item_keys.parquet
    keys_df = build_item_keys_df(item_keys, items_meta_df, cluster_assignments)
    keys_df.to_parquet(out_dir / "item_keys.parquet", index=False)

    # 6. Optional per-subject pass-rate table.
    passrate_meta: dict[str, object] = {"format": "none"}
    if cfg.passrate_format != "none" and train_df is not None and len(train_df):
        if cfg.passrate_format == "cluster":
            if cluster_assignments and n_clusters > 0:
                matrix, subject_keys = build_cluster_passrate_table(
                    train_df,
                    cluster_assignments=cluster_assignments,
                    n_clusters=n_clusters,
                )
                np.save(out_dir / "subject_cluster_passrates.npy", matrix)
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(subject_keys), dtype=np.int32),
                        "subject_key": subject_keys,
                    }
                ).to_parquet(out_dir / "subject_keys.parquet", index=False)
                passrate_meta = {
                    "format": "cluster",
                    "n_subjects": int(len(subject_keys)),
                    "n_clusters_plus_unk": int(n_clusters + 1),
                }
            else:
                LOG.warning(
                    "passrate_format='cluster' requested but cluster_assignments / n_clusters missing"
                )
        elif cfg.passrate_format == "sparse":
            try:
                mat, subject_keys = build_sparse_passrate_csr(train_df, item_keys=item_keys)
                from scipy import sparse  # type: ignore

                sparse.save_npz(out_dir / "subject_passrates.npz", mat)
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(subject_keys), dtype=np.int32),
                        "subject_key": subject_keys,
                    }
                ).to_parquet(out_dir / "subject_keys.parquet", index=False)
                passrate_meta = {
                    "format": "sparse",
                    "n_subjects": int(len(subject_keys)),
                    "n_items": int(len(item_keys)),
                    "nnz": int(mat.nnz),
                }
            except Exception as exc:  # noqa: BLE001
                LOG.warning("sparse passrate export failed: %s", exc)

    # 6b. NN-features cache (subject_passrate + mask + subject_key_to_id +
    #     nn_features_config.json). Shipped only when an NN config and a
    #     subject indexer are provided -- otherwise this whole block is
    #     skipped and the runtime falls back to zero NN features.
    nn_meta: dict[str, object] = {"enabled": False}
    if nn_features_cfg is not None and subject_to_id is not None and train_df is not None and len(train_df):
        nn_cfg_dict = dict(nn_features_cfg)
        try:
            from scipy import sparse  # type: ignore

            passrate, mask = build_nn_passrate_csr(
                train_df,
                item_keys=item_keys,
                subject_to_id=subject_to_id,
            )
            sparse.save_npz(out_dir / "subject_passrate.npz", passrate)
            # Mask is binary so int8 would be free, but scipy.sparse keeps
            # the dtype of the underlying arrays -- callers can cast on load.
            sparse.save_npz(out_dir / "subject_passrate_mask.npz", mask)
            (out_dir / "subject_key_to_id.json").write_text(
                json.dumps({str(k): int(v) for k, v in subject_to_id.items()}),
                encoding="utf-8",
            )
            nn_config_block = {
                "enabled": bool(nn_cfg_dict.get("enabled", True)),
                "k": int(nn_cfg_dict.get("k", 16)),
                "runtime_k": int(nn_cfg_dict.get("runtime_k", cfg.runtime_k)),
                "similarity": str(nn_cfg_dict.get("similarity", "cosine")),
                "feature_dim": int(nn_cfg_dict.get("feature_dim", 23)),
                "fallback_value": float(nn_cfg_dict.get("fallback_value", 0.0)),
                "top1_missing_sentinel": float(
                    nn_cfg_dict.get("top1_missing_sentinel", -1.0)
                ),
                "n_subjects": int(passrate.shape[0]),
                "n_items": int(passrate.shape[1]),
                "passrate_nnz": int(passrate.nnz),
            }
            (out_dir / "nn_features_config.json").write_text(
                json.dumps(nn_config_block, indent=2), encoding="utf-8"
            )
            nn_meta = {
                "enabled": True,
                "n_subjects": int(passrate.shape[0]),
                "n_items": int(passrate.shape[1]),
                "passrate_nnz": int(passrate.nnz),
            }
        except Exception as exc:  # noqa: BLE001
            LOG.warning("NN-features cache export failed: %s", exc)
            nn_meta = {"enabled": False, "error": str(exc)}

    # 6c. Conditional NN-feature context (cells [15..22] of NN_FEATURE_DIM).
    #     Shipped alongside the existing NN cache when the caller built one
    #     at training time. Each file is plain numpy / scipy.sparse so the
    #     runtime does not pull a fresh torch / pandas import.
    cond_meta: dict[str, object] = {"enabled": False}
    if conditional_context is not None:
        try:
            saved_dir = conditional_context.save(out_dir)
            n_files = sum(1 for _ in saved_dir.glob("*.np?"))
            cond_meta = {
                "enabled": True,
                "n_subjects": int(getattr(conditional_context, "n_subjects", 0)),
                "n_items": int(getattr(conditional_context, "n_items", 0)),
                "n_families": int(getattr(conditional_context, "n_families", 0)),
                "n_macro_families": int(getattr(conditional_context, "n_macro_families", 0)),
                "n_organizations": int(getattr(conditional_context, "n_organizations", 0)),
                "n_clusters": int(getattr(conditional_context, "n_clusters", 0)),
                "files_written": int(n_files),
            }
        except Exception as exc:  # noqa: BLE001
            LOG.warning("conditional NN context export failed: %s", exc)
            cond_meta = {"enabled": False, "error": str(exc)}
    if bc_id_to_age is not None:
        try:
            arr = np.asarray(bc_id_to_age, dtype=np.float32).reshape(-1)
            np.save(out_dir / "bc_id_to_age.npy", arr)
            cond_meta["bc_id_to_age_n"] = int(arr.shape[0])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("bc_id_to_age export failed: %s", exc)

    # 7. cache_meta.json
    meta = {
        "encoder_id": cfg.encoder_id,
        "query_prefix": cfg.query_prefix,
        "passage_prefix": cfg.passage_prefix,
        "n_items": int(n_items),
        "source_dim": int(src_dim),
        "stored_dim": int(proj_matrix.shape[1]),
        "stored_dtype": stored_dtype,
        "quantize": cfg.quantize,
        "pca_dim": int(cfg.pca_dim) if components is not None else None,
        "faiss_index_type": cfg.faiss_index_type if cfg.include_faiss_index else None,
        "faiss_index_present": bool((out_dir / "faiss.index").exists()),
        "faiss_error": faiss_err,
        "passrate": passrate_meta,
        "nn_features": nn_meta,
        "runtime_k": int(cfg.runtime_k),
    }
    (out_dir / "cache_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    sizes, total_mb = _file_sizes(out_dir)
    LOG.info("Submission cache total size: %.2f MB", total_mb)
    for name, mb in sizes.items():
        LOG.info("  %-32s %7.2f MB", name, mb)

    result = CacheExportResult(
        out_dir=out_dir,
        written_files=list(sizes.keys()),
        sizes_mb=sizes,
        total_mb=total_mb,
        meta=meta,
    )
    if cfg.max_bundle_size_mb and total_mb > float(cfg.max_bundle_size_mb):
        result.failed = True
        result.error = (
            f"submission cache size {total_mb:.2f} MB exceeds "
            f"max_bundle_size_mb={float(cfg.max_bundle_size_mb):.0f} MB"
        )
    return result


__all__ = [
    "CacheExportConfig",
    "CacheExportResult",
    "apply_pca",
    "build_cluster_passrate_table",
    "build_faiss_index",
    "build_item_keys_df",
    "build_nn_passrate_csr",
    "build_sparse_passrate_csr",
    "dequantize_int8",
    "export_item_cache",
    "fit_pca",
    "load_item_embeddings",
    "quantize_int8_per_row",
]
