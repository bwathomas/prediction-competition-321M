"""Member 2's NON-EMBEDDING feature matrix (Task 3 of the diversification plan).

Member 2 used to share the full 1200+-dim ``X_train_dense`` with
Member 1's subject embeddings (theta, u), pool aggregates, centroid
distances and NN features, plus the 8-col mean-encoded interactions.
That sharing was exactly what made Member 2's val errors heavily
correlated with Member 1 -- the GBDT kept rediscovering signals the
IRT-MLP already had.

Task 3 restricts Member 2 to features the IRT-MLP CANNOT see directly:

  - subject_idx                (categorical, lets the tree learn per-subject splits)
  - subject_obs_count_log1p    (numeric, captures cold-start subject risk)
  - cluster_id                 (categorical, item-cluster id)
  - bench_condition_id         (categorical)
  - bc_redacted_mask           (0/1 mask, captures the holdout structure)
  - subject_cat_*              (categorical, per-subject metadata vocab ids)
  - subject_num_*              (interleaved value + missingness mask floats)
  - bench_cat_*                (categorical, per-benchmark metadata vocab ids)
  - bench_num_*                (interleaved value + missingness mask floats)
  - mean_encoded_interaction_* (8 cols, the Member-2-specific mean-encoded interactions)

Everything embedding-derived (theta, u, pool features, centroid
distances, NN features) is DROPPED. The result is a much smaller
dense matrix (~20-40 cols) keyed on subject + benchmark + cluster
plus the existing mean-encoded interactions.

For LightGBM consumption the categorical columns are still numeric
floats here; the trainer side passes ``categorical_feature=[...]``
with the column names that should be treated as categorical splits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LOG = logging.getLogger("member2_features")


@dataclass(frozen=True)
class Member2FeatureSchema:
    """Fixed schema for Member 2's non-embedding feature matrix.

    ``feature_names`` is the ordered column list (matches the produced
    matrix's column order). ``categorical_indices`` are the column
    POSITIONS that LightGBM should treat as categorical splits.
    """

    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...] = field(default_factory=tuple)
    subject_cat_field_names: tuple[str, ...] = field(default_factory=tuple)
    subject_num_field_names: tuple[str, ...] = field(default_factory=tuple)
    bench_cat_field_names: tuple[str, ...] = field(default_factory=tuple)
    bench_num_field_names: tuple[str, ...] = field(default_factory=tuple)
    n_interaction_cols: int = 0

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


# Hard-coded "core" columns that are ALWAYS present.
_CORE_FEATURE_NAMES: tuple[str, ...] = (
    "subject_idx",
    "subject_obs_count_log1p",
    "cluster_id",
    "bench_condition_id",
    "bc_redacted_mask",
)
_CORE_CATEGORICAL: tuple[str, ...] = (
    "subject_idx",
    "cluster_id",
    "bench_condition_id",
)


def build_member2_schema(
    *,
    subject_cat_field_names: Sequence[str],
    subject_num_field_names: Sequence[str],
    bench_cat_field_names: Sequence[str],
    bench_num_field_names: Sequence[str],
    interaction_feature_names: Sequence[str],
) -> Member2FeatureSchema:
    """Construct the locked Member 2 feature schema given the metadata
    field names extracted from the global preprocessor / id tables.

    The schema is deterministic given the inputs (no fitting on data),
    which makes it safe to call ONCE at notebook startup and reuse
    across folds.
    """
    names: list[str] = list(_CORE_FEATURE_NAMES)
    # Subject categorical (one column per cat field, value = vocab id).
    names.extend(f"subject_cat__{f}" for f in subject_cat_field_names)
    # Subject numeric: TWO cols per field (scaled value, missingness mask).
    for f in subject_num_field_names:
        names.append(f"subject_num__{f}__value")
        names.append(f"subject_num__{f}__mask")
    # Benchmark categorical / numeric (same convention).
    names.extend(f"bench_cat__{f}" for f in bench_cat_field_names)
    for f in bench_num_field_names:
        names.append(f"bench_num__{f}__value")
        names.append(f"bench_num__{f}__mask")
    # Mean-encoded interaction columns (already locked-name from
    # mean_encoded_features.py).
    names.extend(str(n) for n in interaction_feature_names)

    cat_set = set(_CORE_CATEGORICAL)
    cat_set.update(f"subject_cat__{f}" for f in subject_cat_field_names)
    cat_set.update(f"bench_cat__{f}" for f in bench_cat_field_names)
    cat_idx: list[int] = [i for i, n in enumerate(names) if n in cat_set]

    schema = Member2FeatureSchema(
        feature_names=tuple(names),
        categorical_indices=tuple(cat_idx),
        subject_cat_field_names=tuple(subject_cat_field_names),
        subject_num_field_names=tuple(subject_num_field_names),
        bench_cat_field_names=tuple(bench_cat_field_names),
        bench_num_field_names=tuple(bench_num_field_names),
        n_interaction_cols=int(len(interaction_feature_names)),
    )
    LOG.info(
        "build_member2_schema: dim=%d (%d core + %d subj_cat + %d subj_num*2 + "
        "%d bench_cat + %d bench_num*2 + %d interaction) cat_idxs=%d",
        schema.feature_dim,
        len(_CORE_FEATURE_NAMES),
        len(subject_cat_field_names),
        len(subject_num_field_names),
        len(bench_cat_field_names),
        len(bench_num_field_names),
        len(interaction_feature_names),
        len(cat_idx),
    )
    return schema


def build_member2_feature_matrix(
    schema: Member2FeatureSchema,
    *,
    subject_ids: np.ndarray,
    cluster_ids: np.ndarray,
    bc_ids: np.ndarray,
    bc_redacted_mask: np.ndarray,
    subject_obs_count_log1p: np.ndarray,
    subject_cat_lookup: np.ndarray,
    subject_num_lookup: np.ndarray,
    bench_cat_lookup: np.ndarray,
    bench_num_lookup: np.ndarray,
    interaction_matrix: np.ndarray,
) -> np.ndarray:
    """Assemble the per-row Member 2 feature matrix.

    Parameters
    ----------
    schema
        The fixed Member 2 schema. Determines column order + dimension.
    subject_ids, cluster_ids, bc_ids
        ``[N]`` int arrays. ``-1`` is the UNK sentinel.
    bc_redacted_mask
        ``[N]`` 0/1 float array; 1 means the benchmark-condition was
        redacted at training time (kept as feature so the tree can
        learn to lean on subject signal when bc is masked).
    subject_obs_count_log1p
        ``[N]`` float, log1p of the subject's training observation
        count. Pre-computed via :func:`subject_mean.apply_subject_obs_count`.
    subject_cat_lookup, bench_cat_lookup
        ``[n_subjects, n_subj_cat_fields]`` and ``[n_bc, n_bench_cat_fields]``
        int arrays (vocab ids) from ``meta_id_tables.subject_cat_ids`` /
        ``.bc_cat`` respectively.
    subject_num_lookup, bench_num_lookup
        ``[n_subjects, 2*n_subj_num_fields]`` / ``[n_bc, 2*n_bench_num_fields]``
        float arrays, interleaved (value, mask) per field.
    interaction_matrix
        ``[N, schema.n_interaction_cols]`` float array (output of
        ``mean_encoded_features.apply_member2_interaction_features``).

    Returns
    -------
    Dense ``[N, schema.feature_dim]`` float32 matrix in the schema's
    column order.
    """
    N = int(subject_ids.shape[0])
    D = schema.feature_dim
    out = np.empty((N, D), dtype=np.float32)
    col = 0

    out[:, col] = subject_ids.astype(np.float32, copy=False); col += 1
    out[:, col] = subject_obs_count_log1p.astype(np.float32, copy=False); col += 1
    out[:, col] = cluster_ids.astype(np.float32, copy=False); col += 1
    out[:, col] = bc_ids.astype(np.float32, copy=False); col += 1
    out[:, col] = bc_redacted_mask.astype(np.float32, copy=False); col += 1

    # Subject categorical: gather vocab id per row.
    safe_subj = np.where(
        (subject_ids >= 0) & (subject_ids < subject_cat_lookup.shape[0]),
        subject_ids, 0,
    ).astype(np.int64)
    for f_idx, _f in enumerate(schema.subject_cat_field_names):
        out[:, col] = subject_cat_lookup[safe_subj, f_idx].astype(np.float32, copy=False)
        # Restore UNK semantics (-1 for unknown subject -> vocab 0 = MISSING).
        out[subject_ids < 0, col] = 0.0
        col += 1
    for f_idx, _f in enumerate(schema.subject_num_field_names):
        v_col = 2 * f_idx
        out[:, col] = subject_num_lookup[safe_subj, v_col].astype(np.float32, copy=False)
        out[subject_ids < 0, col] = 0.0
        col += 1
        out[:, col] = subject_num_lookup[safe_subj, v_col + 1].astype(np.float32, copy=False)
        out[subject_ids < 0, col] = 1.0  # mask = MISSING
        col += 1

    safe_bc = np.where(
        (bc_ids >= 0) & (bc_ids < bench_cat_lookup.shape[0]),
        bc_ids, 0,
    ).astype(np.int64)
    for f_idx, _f in enumerate(schema.bench_cat_field_names):
        out[:, col] = bench_cat_lookup[safe_bc, f_idx].astype(np.float32, copy=False)
        out[bc_ids < 0, col] = 0.0
        col += 1
    for f_idx, _f in enumerate(schema.bench_num_field_names):
        v_col = 2 * f_idx
        out[:, col] = bench_num_lookup[safe_bc, v_col].astype(np.float32, copy=False)
        out[bc_ids < 0, col] = 0.0
        col += 1
        out[:, col] = bench_num_lookup[safe_bc, v_col + 1].astype(np.float32, copy=False)
        out[bc_ids < 0, col] = 1.0
        col += 1

    # Interaction columns: pass-through.
    if interaction_matrix.shape[1] != schema.n_interaction_cols:
        raise ValueError(
            f"interaction_matrix has {interaction_matrix.shape[1]} cols, "
            f"schema expects {schema.n_interaction_cols}"
        )
    out[:, col : col + schema.n_interaction_cols] = interaction_matrix.astype(np.float32, copy=False)
    col += schema.n_interaction_cols
    if col != D:
        raise RuntimeError(
            f"build_member2_feature_matrix: filled {col} cols but schema "
            f"has dim {D}. This is a builder bug."
        )
    return out


# ---------------------------------------------------------------------------
# Gate 3d: no-embedding audit
# ---------------------------------------------------------------------------


_EMBEDDING_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "pool_",
    "centroid_dist",
    "nn_",
    "theta",
    # 'u_' is too generic (matches e.g. "subject_cat__user_..."); only ban
    # the leading-stem u_<dim> form used by the MemberFeatureSchema.
    "u__",
)


def audit_no_embedding_features(schema: Member2FeatureSchema) -> None:
    """RED-TEAM GATE 3d: assert the schema contains no embedding-derived
    columns. Raises with the offending names if any are found."""
    bad: list[str] = []
    for name in schema.feature_names:
        lower = str(name).lower()
        for forbidden in _EMBEDDING_FORBIDDEN_SUBSTRINGS:
            if forbidden in lower:
                bad.append(name)
                break
    if bad:
        raise AssertionError(
            f"GATE 3d violation: Member 2 schema has {len(bad)} embedding-"
            f"derived columns it should not: {bad[:10]}. The whole point of "
            "Task 3 was to decorrelate Member 2 from Member 1 by stripping "
            "the embedding-derived signal -- one of these columns is leaking "
            "the IRT-MLP's signal back in."
        )


__all__ = [
    "Member2FeatureSchema",
    "build_member2_schema",
    "build_member2_feature_matrix",
    "audit_no_embedding_features",
]
