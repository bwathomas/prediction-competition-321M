"""Colab per-family derivation driver (Plan 4 Task 6).

Walks the §D DAG for ONE embedding family on an A100, calling the ``aide/features`` codecs
and writing derive-once shards through ``FoldFeatureStore`` (idempotent resume from INDEX).
Reuses the repo's tested machinery where it exists: ``src.nn_features.build_passrate_table``
for the (subject×item) CSR, and FAISS (via the codecs' ``default_knn``) for the search.

Heavy deps (pyarrow / scipy / faiss / pandas) are imported lazily inside the functions, so
this module imports cleanly in the numpy-only local env; the actual derivation runs on Colab.
Design for memory (§B): one family at a time; embeddings cast to float32 once; per-fold CSR
built then freed; the funnel assembles lazily and drops refs.

Throughput note: NN search is per query ITEM, but a (subject,item) row table repeats items
across subjects. ``derive_nn_chunk`` searches on the chunk's items as given; for the full
5.3M-row run, pre-dedup query rows by item (search once per unique item, expand the label
gather per subject) — see ``unique_item_rows``.
"""
from __future__ import annotations

import numpy as np

# Family -> embedding cache dirname under {drive_root}/embeddings/
FAMILY_SLUG = {
    "llama": "nvidia__llama-embed-nemotron-8b",
    "qwen": "Qwen__Qwen3-Embedding-8B",
    "mistral": "embedding_cache_lgai_preview_fa2",
}

LABEL_COLS = ["subject_key", "item_key", "item_split_key", "label"]


# ---- IO (Colab) ---------------------------------------------------------------------
def load_embeddings(parquet_path):
    """Return ``(keys: list[str], emb: float32 [n, d])`` from an items/subjects parquet."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path)
    key_col = t.column_names[0]
    keys = [str(k) for k in t.column(key_col).to_pylist()]
    emb = np.asarray(t.column("embedding").to_pylist(), dtype=np.float32)
    return keys, emb


def load_labels(db_parquet_path, columns=LABEL_COLS):
    import pandas as pd
    return pd.read_parquet(db_parquet_path, columns=list(columns))


def unit_rows(emb):
    emb = np.asarray(emb, dtype=np.float32)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


# ---- passrate (reuse src + wrap with CsrPassrate) -----------------------------------
def build_fold_passrate(labels_df, train_item_keys, subject_keys, item_keys):
    """CsrPassrate built from fold-train rows only (OOF: query/OOF items absent)."""
    from src.nn_features import build_passrate_table
    from aide.features.passrate import CsrPassrate
    item_index_map = {str(k): j for j, k in enumerate(item_keys)}
    subject_index_map = {str(s): i for i, s in enumerate(subject_keys)}
    train_set = set(str(k) for k in train_item_keys)
    train_df = labels_df[labels_df["item_key"].astype(str).isin(train_set)]
    pr_csr, _mask = build_passrate_table(train_df, item_index_map, subject_index_map)
    return CsrPassrate.from_scipy(subject_keys, item_keys, pr_csr)


def unique_item_rows(labels_df):
    """Distinct query items (the unit of NN search) → (item_keys, first-row index)."""
    seen = labels_df.drop_duplicates("item_key")
    return seen["item_key"].astype(str).tolist()


# ---- per-chunk NN derivation --------------------------------------------------------
def derive_nn_chunk(*, store, fold, chunk_df, emb_lookup, train_item_keys, train_emb,
                    passrate, inputs_hash, Ks=(4, 8, 32, 64), overwrite=False):
    """Derive + write nn_label_derivatives / nn_geometry / counts_subject for a row chunk.

    ``emb_lookup`` maps item_key -> unit embedding row; ``train_emb`` is the unit-normalized
    fold-train index matrix aligned with ``train_item_keys``. Returns the write outcomes.
    """
    from aide.features.derive_nn import derive_nn
    q_items = chunk_df["item_key"].astype(str).tolist()
    q_subj = chunk_df["subject_key"].astype(str).tolist()
    row_ids = [f"{s}|{i}" for s, i in zip(q_subj, q_items)]
    q_emb = np.asarray([emb_lookup[i] for i in q_items], dtype=np.float32)
    blocks = derive_nn(query_emb=q_emb, query_item_keys=q_items, query_subjects=q_subj,
                       row_ids=row_ids, index_emb=train_emb, index_item_keys=train_item_keys,
                       passrate=passrate, Ks=Ks)  # default_knn -> FAISS on Colab
    return store.write_blocks(blocks, fold=fold, inputs_hash=inputs_hash, overwrite=overwrite)


def content_inputs_hash(family, group, fold, code_version, extra=""):
    """Stable inputs hash for a shard's meta (family+group+fold+code+extra)."""
    from aide.features.cache import content_hash
    return content_hash(family, group, str(fold), code_version, extra)


# ---- full per-family walk -----------------------------------------------------------
# Geometry/content groups derived once at fold="all" (per unique item); label groups
# derived per outer fold over the rows whose item is OOF in that fold.
GEOMETRY_GROUPS = ["nn_geometry", "cluster_geometry", "centroid_distance", "item_cluster"]


def _row_ids(subj, items):
    return [f"{s}|{i}" for s, i in zip(subj, items)]


def derive_geometry_all(*, store, all_item_keys, all_emb, centroids, family,
                        code_version, progress=None, chunk=40000, overwrite=False,
                        include_cluster=True):
    """fold='all' geometry, one row per UNIQUE item (compact, fold-invariant).

    Writes nn_geometry, cluster_geometry, centroid_distance, item_cluster. Labels are not
    read (empty passrate); the index for the nn-geometry pass is ALL items (self-excluded).
    """
    from aide.features.derive_cluster import derive_cluster
    from aide.features.derive_nn import derive_nn
    from aide.features.passrate import CsrPassrate
    empty = CsrPassrate.empty([], all_item_keys)
    n = len(all_item_keys)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        keys = [str(k) for k in all_item_keys[s:e]]
        q = all_emb[s:e]
        rids = keys  # one row per item
        nn = derive_nn(query_emb=q, query_item_keys=keys, query_subjects=[""] * len(keys),
                       row_ids=rids, index_emb=all_emb, index_item_keys=all_item_keys,
                       passrate=empty, Ks=(4, 8, 32, 64))
        store.write_group("nn_geometry", "all", nn["nn_geometry"],
                          inputs_hash=content_inputs_hash(family, "nn_geometry", "all", code_version),
                          overwrite=overwrite)
        if include_cluster:
            cl = derive_cluster(query_emb=q, query_item_keys=keys,
                                query_subjects=[""] * len(keys), row_ids=rids,
                                centroids_by_res=centroids, train_emb=all_emb,
                                train_item_keys=all_item_keys, passrate=empty)
            for g in ["cluster_geometry", "centroid_distance", "item_cluster"]:
                store.write_group(g, "all", cl[g],
                                  inputs_hash=content_inputs_hash(family, g, "all", code_version),
                                  overwrite=overwrite)
        if progress:
            progress(f"geometry {e}/{n}", 0.1 + 0.2 * e / n, geom_items=e)


def derive_labels_fold(*, store, fold, rows_df, emb_lookup, train_item_keys, train_emb,
                       all_item_keys, centroids, passrate, family, code_version,
                       progress=None, chunk=60000, overwrite=False, include_cluster=True):
    """Per-fold label groups over rows whose item is OOF in this fold.

    Writes nn_label_derivatives, counts_subject (nn) and cluster_passrate, cluster_subject
    (cluster). All OOF: index + passrate are fold-train only; cluster difficulty pools
    fold-train labels. (mean_encoded_subject / metadata groupbys are a separate global pass.)
    """
    import numpy as _np
    from aide.features.derive_cluster import derive_cluster
    from aide.features.derive_nn import derive_nn
    items = rows_df["item_key"].astype(str).to_numpy()
    subj = rows_df["subject_key"].astype(str).to_numpy()
    n = len(items)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ci, cs = items[s:e], subj[s:e]
        rids = _row_ids(cs, ci)
        q = _np.asarray([emb_lookup[k] for k in ci], dtype=_np.float32)
        nn = derive_nn(query_emb=q, query_item_keys=list(ci), query_subjects=list(cs),
                       row_ids=rids, index_emb=train_emb, index_item_keys=train_item_keys,
                       passrate=passrate, Ks=(4, 8, 32, 64))
        for g in ["nn_label_derivatives", "counts_subject"]:
            store.write_group(g, fold, nn[g],
                              inputs_hash=content_inputs_hash(family, g, fold, code_version),
                              overwrite=overwrite)
        if include_cluster:
            cl = derive_cluster(query_emb=q, query_item_keys=list(ci), query_subjects=list(cs),
                                row_ids=rids, centroids_by_res=centroids, train_emb=train_emb,
                                train_item_keys=train_item_keys, passrate=passrate)
            for g in ["cluster_passrate", "cluster_subject"]:
                store.write_group(g, fold, cl[g],
                                  inputs_hash=content_inputs_hash(family, g, fold, code_version),
                                  overwrite=overwrite)
        if progress:
            progress(f"fold{fold} labels {e}/{n}", fold=fold, rows=e, total=n)


def derive_family(*, drive_root, family, code_version="v1", n_folds=3, seed=0,
                  coarse_k=32, fine_k=256, progress=None, overwrite=False, max_rows=None,
                  include_cluster=True):
    """Walk the full §D DAG for one embedding family; write shards to
    ``{drive_root}/features``. Idempotent: existing shards are skipped. ``max_rows`` caps
    per-fold rows for a validation run."""
    import numpy as _np
    from aide.features.cache import FeatureCache
    from aide.features.derive_cluster import fit_multi_kmeans
    from aide.features.passrate import CsrPassrate
    from aide.features.store import FoldFeatureStore
    from aide.hygiene.manifest import build_manifest
    from aide.hygiene.splits import outer_folds

    if progress is None:
        progress = lambda *a, **k: None  # noqa: E731
    slug = FAMILY_SLUG[family]
    emb_dir = f"{drive_root}/embeddings/{slug}"
    progress("loading embeddings", 0.0)
    all_item_keys, all_emb = load_embeddings(f"{emb_dir}/items.parquet")
    all_emb = unit_rows(all_emb)
    emb_lookup = {k: all_emb[i] for i, k in enumerate(all_item_keys)}
    sub_keys, _se = load_embeddings(f"{emb_dir}/subjects.parquet")
    progress(f"embeddings {len(all_item_keys)} items / {len(sub_keys)} subjects", 0.05)

    import glob as _glob
    db = _glob.glob(f"{drive_root}/prepared_datasets/*measurement_db_prepared*.parquet")[0]
    labels = load_labels(db)
    labels["item_key"] = labels["item_key"].astype(str)
    labels["subject_key"] = labels["subject_key"].astype(str)
    emb_set = set(all_item_keys)
    labels = labels[labels["item_key"].isin(emb_set)]

    store = FoldFeatureStore(FeatureCache(f"{drive_root}/features", code_version=code_version),
                             embedding_family=family, seed=seed, n_folds=n_folds)

    progress("fitting k-means", 0.07)
    centroids = fit_multi_kmeans(all_emb, {"coarse": coarse_k, "fine": fine_k}, seed=seed)

    derive_geometry_all(store=store, all_item_keys=all_item_keys, all_emb=all_emb,
                        centroids=centroids, family=family, code_version=code_version,
                        progress=progress, overwrite=overwrite, include_cluster=include_cluster)

    man = build_manifest(list(emb_set), n_folds=n_folds, seed=seed)
    folds = outer_folds(man)
    for f in folds:
        train_keys = [k for k in f.train_item_keys if k in emb_set]
        oof_set = set(f.oof_item_keys) & emb_set
        rows_f = labels[labels["item_key"].isin(oof_set)]
        if max_rows:
            rows_f = rows_f.head(max_rows)
        train_emb = _np.asarray([emb_lookup[k] for k in train_keys], dtype=_np.float32)
        progress(f"fold{f.index}: passrate", fold=f.index)
        passrate = build_fold_passrate(labels, train_keys, sub_keys, all_item_keys)
        derive_labels_fold(store=store, fold=f.index, rows_df=rows_f, emb_lookup=emb_lookup,
                           train_item_keys=train_keys, train_emb=train_emb,
                           all_item_keys=all_item_keys, centroids=centroids,
                           passrate=passrate, family=family, code_version=code_version,
                           progress=progress, overwrite=overwrite, include_cluster=include_cluster)
        del train_emb, passrate
    return {"family": family, "n_shards": len(store.cache.load_index())}
