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
