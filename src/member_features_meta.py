"""Offline bridge: metadata preprocessor -> member feature matrix.

``src.member_features`` is deliberately runtime-pure (numpy + stdlib
only -- no pandas / sklearn / torch). This module is the *offline*
counterpart: it is the single place that touches
:class:`src.metadata_features.MetadataPreprocessor` (pandas + torch)
to populate the per-id metadata tables, then hands them to the
runtime-pure :func:`src.member_features.build_member_features` to
produce the locked ``[N, F]`` dense matrix + ``feature_names``.

Why a separate module
---------------------
The shared dense matrix that the GBDT / xgb / cat / forest / knn / fm /
logreg members consume needs subject *and* benchmark structured
metadata. The encoders for that metadata (vocab fitting, numeric
scaling, the CSV joins) live in ``metadata_features`` and pull in
pandas. Keeping that dependency out of ``member_features`` means the
shipped runtime ``model.py`` can ``import src.member_features`` without
dragging pandas into the submission whitelist. This module is
offline-only and never imported by the runtime.

What it builds
--------------
``build_shared_matrix`` returns:

- the locked ``[N, feature_dim]`` float32 matrix,
- the ``feature_names`` tuple (column order = source of truth),
- the fitted :class:`~src.member_features.MemberFeatureSchema` (v3),
- the populated :class:`~src.member_features.MemberSubjectTables` and
  :class:`~src.member_features.MemberBenchmarkTables` (shipped as
  ``.npy`` and reloaded at runtime).

Leakage discipline
------------------
The benchmark (and subject) metadata joins are *static*
(FOLD_INVARIANT): they depend only on the model_info / benchmark_info
CSVs, never on labels, so they are fit on the full corpus and shared
across folds without introducing OOF leakage. Label-derived
y-aggregates (mean-encodings, group passrates) are NOT computed here --
they live in the aide feature-store path and are rebuilt per fold.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.member_features import (
    MemberBenchmarkTables,
    MemberFeatureSchema,
    MemberSubjectTables,
    build_member_features,
)
from src.metadata_features import MetadataPreprocessor, extract_display_name


def _build_subject_benchmark_tables(
    *,
    preprocessor: MetadataPreprocessor,
    subject_to_id: Mapping[str, int],
    bc_to_id: Mapping[str, int],
    subject_content_by_key: Mapping[str, str],
    subject_theta: np.ndarray,
    subject_u: np.ndarray,
) -> tuple[
    MemberSubjectTables,
    MemberBenchmarkTables,
    tuple[int, ...],  # subject cat cardinalities
    tuple[str, ...],  # subject cat field names
    tuple[str, ...],  # subject num field names
    tuple[int, ...],  # benchmark cat cardinalities
    tuple[str, ...],  # benchmark cat field names
    tuple[str, ...],  # benchmark num field names
]:
    """Encode every (subject_idx, bc_idx) row into the member tables.

    Mirrors :func:`src.metadata_features.build_metadata_id_tables` but
    targets the ``member_features`` dataclasses (numpy, not torch) and
    folds in the IRT ``theta`` / ``u`` factor tables on the subject
    side. Row 0 (UNK) is MISSING for every metadata field.

    Returns the two tables plus the schema-sizing metadata:
    subject cat cardinalities / field names, subject num field names,
    benchmark cat cardinalities / field names, benchmark num field names.
    """
    schema = preprocessor.schema
    n_sub = len(subject_to_id)
    n_bc = len(bc_to_id)
    n_sub_cat = len(schema.subject_categorical)
    n_bc_cat = len(schema.benchmark_categorical)
    n_sub_num = len(schema.subject_numeric)
    n_bc_num = len(schema.benchmark_numeric)

    subject_theta = np.asarray(subject_theta, dtype=np.float32).reshape(-1)
    subject_u = np.asarray(subject_u, dtype=np.float32)
    if subject_theta.shape[0] != n_sub:
        raise ValueError(
            f"subject_theta rows {subject_theta.shape[0]} != "
            f"n_subjects {n_sub}"
        )
    if subject_u.shape[0] != n_sub:
        raise ValueError(
            f"subject_u rows {subject_u.shape[0]} != n_subjects {n_sub}"
        )

    sub_cat = np.zeros((n_sub, max(1, n_sub_cat)), dtype=np.int64)
    sub_num = np.zeros((n_sub, 2 * max(1, n_sub_num)), dtype=np.float32)
    bc_cat = np.zeros((n_bc, max(1, n_bc_cat)), dtype=np.int64)
    bc_num = np.zeros((n_bc, 2 * max(1, n_bc_num)), dtype=np.float32)

    # Default missingness = 1 everywhere; overwritten where data exists.
    for j in range(n_sub_num):
        sub_num[:, 2 * j + 1] = 1.0
    for j in range(n_bc_num):
        bc_num[:, 2 * j + 1] = 1.0

    # Subject side.
    for key, idx in subject_to_id.items():
        if idx == 0 or key == "<unk>":
            continue
        content = subject_content_by_key.get(key, "")
        name = extract_display_name(content)
        cat_ids, num_x, num_m = preprocessor.encode_subject(name)
        if n_sub_cat > 0:
            sub_cat[idx, :n_sub_cat] = cat_ids[:n_sub_cat]
        for j in range(n_sub_num):
            sub_num[idx, 2 * j] = num_x[j]
            sub_num[idx, 2 * j + 1] = num_m[j]

    # Benchmark-condition side: bc_key = "{benchmark}::{condition}".
    for key, idx in bc_to_id.items():
        if idx == 0 or key == "<unk>":
            continue
        benchmark = key.split("::", 1)[0] if "::" in key else key
        cat_ids, num_x, num_m = preprocessor.encode_benchmark(benchmark)
        if n_bc_cat > 0:
            bc_cat[idx, :n_bc_cat] = cat_ids[:n_bc_cat]
        for j in range(n_bc_num):
            bc_num[idx, 2 * j] = num_x[j]
            bc_num[idx, 2 * j + 1] = num_m[j]

    subject_cat_cards = tuple(
        preprocessor.subject_cat_vocabs[col].n_tokens
        if col in preprocessor.subject_cat_vocabs
        else 2
        for col in schema.subject_categorical
    )
    benchmark_cat_cards = tuple(
        preprocessor.benchmark_cat_vocabs[col].n_tokens
        if col in preprocessor.benchmark_cat_vocabs
        else 2
        for col in schema.benchmark_categorical
    )

    subject_tables = MemberSubjectTables(
        theta=subject_theta,
        u=subject_u,
        subject_cat_ids=sub_cat[:, :n_sub_cat] if n_sub_cat > 0 else sub_cat,
        subject_num=(
            sub_num[:, : 2 * n_sub_num] if n_sub_num > 0 else sub_num[:, :0]
        ),
    )
    benchmark_tables = MemberBenchmarkTables(
        benchmark_cat_ids=bc_cat[:, :n_bc_cat] if n_bc_cat > 0 else bc_cat[:, :0],
        benchmark_num=(
            bc_num[:, : 2 * n_bc_num] if n_bc_num > 0 else bc_num[:, :0]
        ),
    )

    return (
        subject_tables,
        benchmark_tables,
        subject_cat_cards,
        tuple(schema.subject_categorical),
        tuple(schema.subject_numeric),
        benchmark_cat_cards,
        tuple(schema.benchmark_categorical),
        tuple(schema.benchmark_numeric),
    )


def build_shared_matrix(
    *,
    preprocessor: MetadataPreprocessor,
    subject_to_id: Mapping[str, int],
    bc_to_id: Mapping[str, int],
    subject_content_by_key: Mapping[str, str],
    subject_theta: np.ndarray,
    subject_u: np.ndarray,
    k_factors: int,
    n_clusters: int,
    top_m_centroids: int,
    pool_feature_names: Sequence[str],
    pool_stats: Mapping[str, Mapping[str, float]],
    nn_feature_names: Sequence[str],
    train_conditions: Sequence[object],
    # per-row inputs (length N, the training rows in canonical order)
    subject_idx: np.ndarray,
    bc_idx: np.ndarray,
    pool_feats: np.ndarray,
    centroid_dists: np.ndarray,
    cluster_ids: np.ndarray,
    nn_feats: np.ndarray,
    conditions: Sequence[object],
    bc_redacted: np.ndarray | None = None,
    u_bc_per_row: np.ndarray | None = None,
    k_bc_factors: int = 0,
    min_condition_count: int = 1,
    centroid_dist_names: Sequence[str] | None = None,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    MemberFeatureSchema,
    MemberSubjectTables,
    MemberBenchmarkTables,
]:
    """Build the locked ``[N, F]`` member matrix with subject + benchmark metadata.

    This is the only place pandas / :class:`MetadataPreprocessor` is
    touched in the shared-matrix path. It:

    1. encodes every subject / benchmark id into
       :class:`MemberSubjectTables` + :class:`MemberBenchmarkTables`,
    2. fits a schema_version-3 :class:`MemberFeatureSchema` whose
       benchmark block matches the encoded benchmark vocab,
    3. calls the runtime-pure :func:`build_member_features` to assemble
       the dense matrix.

    Parameters mirror :meth:`MemberFeatureSchema.fit` (schema sizing)
    plus the per-row feature inputs that :func:`build_member_features`
    consumes. ``subject_theta`` / ``subject_u`` are the IRT factor
    tables (row-indexed by ``subject_idx``); ``subject_idx`` and
    ``bc_idx`` index those tables and the benchmark tables respectively.

    Returns ``(X, feature_names, schema, subject_tables, benchmark_tables)``.
    Ship the schema (column order = source of truth) and both tables;
    the runtime reloads them and calls
    :func:`build_member_features_one` to reconstruct each row.
    """
    (
        subject_tables,
        benchmark_tables,
        subject_cat_cards,
        subject_cat_names,
        subject_num_names,
        benchmark_cat_cards,
        benchmark_cat_names,
        benchmark_num_names,
    ) = _build_subject_benchmark_tables(
        preprocessor=preprocessor,
        subject_to_id=subject_to_id,
        bc_to_id=bc_to_id,
        subject_content_by_key=subject_content_by_key,
        subject_theta=subject_theta,
        subject_u=subject_u,
    )

    schema = MemberFeatureSchema.fit(
        k_factors=int(k_factors),
        n_clusters=int(n_clusters),
        top_m_centroids=int(top_m_centroids),
        pool_feature_names=pool_feature_names,
        pool_stats=pool_stats,
        nn_feature_names=nn_feature_names,
        subject_cat_field_names=subject_cat_names,
        subject_cat_field_cardinalities=subject_cat_cards,
        subject_num_field_names=subject_num_names,
        train_conditions=train_conditions,
        min_condition_count=int(min_condition_count),
        centroid_dist_names=centroid_dist_names,
        k_bc_factors=int(k_bc_factors),
        benchmark_cat_field_names=benchmark_cat_names,
        benchmark_cat_field_cardinalities=benchmark_cat_cards,
        benchmark_num_field_names=benchmark_num_names,
    )

    X = build_member_features(
        schema,
        subject_tables,
        subject_idx=np.asarray(subject_idx, dtype=np.int64),
        pool_feats=pool_feats,
        centroid_dists=centroid_dists,
        cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
        nn_feats=nn_feats,
        conditions=conditions,
        bc_redacted=bc_redacted,
        u_bc_per_row=u_bc_per_row,
        benchmark_tables=benchmark_tables,
        bc_idx=np.asarray(bc_idx, dtype=np.int64),
    )

    return X, schema.feature_names, schema, subject_tables, benchmark_tables


__all__ = ["build_shared_matrix"]
