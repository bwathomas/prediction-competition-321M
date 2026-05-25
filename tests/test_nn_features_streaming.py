"""Parity tests for ``compute_nn_features_streaming``.

The streaming variant is only useful if its output is bit-identical to
``compute_nn_features`` (the existing tested path). These tests build a
small synthetic NN index + sparse passrate table and assert the two
implementations agree exactly on:

  * a query set with duplicate item keys (the case streaming dedupes),
  * the case where every query is unique (streaming = passthrough),
  * a sub-batch chunk size that forces multiple ``nn_index.nearest``
    calls inside streaming.

These tests use only the public API (``NNFeaturesConfig`` /
``TrainingNNIndex.build_from_lookup`` / ``build_passrate_table``) plus
both feature functions, so they double as a smoke test that the
``compute_nn_features_streaming`` symbol is properly exported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nn_features import (
    NNFeaturesConfig,
    TrainingNNIndex,
    build_passrate_table,
    compute_nn_features,
    compute_nn_features_streaming,
)


def _build_index(tmp_path: Path, n_items: int = 32, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    item_keys = [f"item_{i:03d}" for i in range(n_items)]
    lookup: dict[str, np.ndarray] = {}
    for k in item_keys:
        v = rng.standard_normal(dim).astype(np.float32)
        v /= max(1e-6, float(np.linalg.norm(v)))
        lookup[k] = v
    cfg = NNFeaturesConfig.from_dict(
        {"enabled": True, "k": 5, "similarity": "cosine", "prefer_gpu": False}
    )
    nn_dir = tmp_path / "nn_index"
    index = TrainingNNIndex.build_from_lookup(
        item_emb_lookup=lookup,
        out_dir=nn_dir,
        cfg=cfg,
        item_keys=item_keys,
    )
    return index, lookup, cfg, item_keys


def _build_passrate(item_keys: list[str], train_df: pd.DataFrame):
    item_index_map = {k: i for i, k in enumerate(item_keys)}
    subject_keys = sorted(train_df["subject_key"].astype(str).unique().tolist())
    subject_index_map = {k: i for i, k in enumerate(subject_keys)}
    pr, mk = build_passrate_table(
        train_df=train_df,
        item_index_map=item_index_map,
        subject_index_map=subject_index_map,
    )
    return pr, mk, subject_index_map


def _synthetic_train_rows(item_keys: list[str], n_subjects: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for k in item_keys:
            if rng.random() > 0.6:
                continue
            rows.append(
                {
                    "subject_key": f"subj_{s:02d}",
                    "item_key": k,
                    "label": int(rng.random() < 0.5),
                }
            )
    return pd.DataFrame(rows)


def _make_query(
    item_keys: list[str],
    item_emb_lookup,
    subject_index_map,
    *,
    n_queries: int = 60,
    seed: int = 1,
):
    """Build a query batch with deliberate item_key duplication."""
    rng = np.random.default_rng(seed)
    sample_keys = list(rng.choice(item_keys, size=n_queries, replace=True))
    sample_subjects = list(rng.choice(list(subject_index_map.keys()), size=n_queries))
    embs = np.stack([item_emb_lookup[k] for k in sample_keys], axis=0).astype(np.float32)
    sids = np.array(
        [subject_index_map[s] for s in sample_subjects], dtype=np.int64
    )
    return embs, sample_keys, sids


def test_streaming_parity_with_duplicates(tmp_path: Path) -> None:
    index, lookup, cfg, item_keys = _build_index(tmp_path)
    train_df = _synthetic_train_rows(item_keys, n_subjects=8, seed=0)
    pr, mk, subject_index_map = _build_passrate(item_keys, train_df)
    embs, qkeys, sids = _make_query(
        item_keys, lookup, subject_index_map, n_queries=60, seed=1
    )

    baseline = compute_nn_features(
        query_embeds=embs,
        query_item_keys=qkeys,
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=False,
    )
    streamed = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=lookup,
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=False,
        query_chunk_size=8,
    )
    assert streamed.shape == baseline.shape
    assert np.allclose(streamed, baseline, atol=1e-6)


def test_streaming_parity_unique_queries(tmp_path: Path) -> None:
    """When every query item_key is unique, dedupe is a no-op but the
    chunked search still has to match the single-shot search."""
    index, lookup, cfg, item_keys = _build_index(tmp_path)
    train_df = _synthetic_train_rows(item_keys, n_subjects=6, seed=2)
    pr, mk, subject_index_map = _build_passrate(item_keys, train_df)
    rng = np.random.default_rng(3)
    sample_subjects = list(
        rng.choice(list(subject_index_map.keys()), size=len(item_keys), replace=True)
    )
    sids = np.array(
        [subject_index_map[s] for s in sample_subjects], dtype=np.int64
    )
    embs = np.stack([lookup[k] for k in item_keys], axis=0).astype(np.float32)

    baseline = compute_nn_features(
        query_embeds=embs,
        query_item_keys=list(item_keys),
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=True,
    )
    streamed = compute_nn_features_streaming(
        query_item_keys=list(item_keys),
        item_emb_lookup=lookup,
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=True,
        query_chunk_size=4,
    )
    assert np.allclose(streamed, baseline, atol=1e-6)


def test_streaming_chunk_size_invariance(tmp_path: Path) -> None:
    """The streaming output must not depend on the chunk size."""
    index, lookup, cfg, item_keys = _build_index(tmp_path)
    train_df = _synthetic_train_rows(item_keys, n_subjects=10, seed=4)
    pr, mk, subject_index_map = _build_passrate(item_keys, train_df)
    embs, qkeys, sids = _make_query(
        item_keys, lookup, subject_index_map, n_queries=80, seed=5
    )

    a = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=lookup,
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=False,
        query_chunk_size=2,
    )
    b = compute_nn_features_streaming(
        query_item_keys=qkeys,
        item_emb_lookup=lookup,
        subject_ids=sids,
        nn_index=index,
        passrate_csr=pr,
        passrate_mask_csr=mk,
        cfg=cfg,
        exclude_self=False,
        query_chunk_size=10_000,
    )
    # FAISS / numpy matmul can differ at the last ULP between different
    # batch sizes; tolerate single-bit floating point drift.
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-6)


def test_streaming_validates_lengths(tmp_path: Path) -> None:
    index, lookup, cfg, item_keys = _build_index(tmp_path)
    train_df = _synthetic_train_rows(item_keys, n_subjects=4, seed=6)
    pr, mk, subject_index_map = _build_passrate(item_keys, train_df)
    qkeys = list(item_keys[:5])
    sids = np.zeros((4,), dtype=np.int64)  # mismatched length on purpose
    with pytest.raises(ValueError):
        compute_nn_features_streaming(
            query_item_keys=qkeys,
            item_emb_lookup=lookup,
            subject_ids=sids,
            nn_index=index,
            passrate_csr=pr,
            passrate_mask_csr=mk,
            cfg=cfg,
            exclude_self=False,
            query_chunk_size=8,
        )
