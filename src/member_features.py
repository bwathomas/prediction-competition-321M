"""Shared dense feature vector for the GBDT (Member 2) and torch
logistic-regression (Member 4) members of the four-member stacked
ensemble.

The user's task spec says: "Logistic regression on the same feature
vector as Member 2." -- so Members 2 and 4 must consume bit-identical
inputs. Both also need a column order that is locked at training
time and reproduced exactly at inference time (silent column
permutation is a known-bug class for hand-rolled tree / linear
models).

Block layout (locked, written into ``feature_names`` once at fit time):

    [theta_s] [u_s_0..u_s_{k-1}]
    [subj_cat one-hot, sum of fitted vocab cardinalities incl. UNK/MISSING]
    [subj_num scaled values + missingness flags, 2 * n_subj_num]
    [pool_feats z-scored, len(POOL_FEATURE_NAMES) = 9]
    [centroid_dist top_m raw squared L2]
    [cluster one-hot, n_clusters cols, indexed 1..n_clusters; idx 0 = UNK row]
    [nn_feats raw, NN_FEATURE_DIM = 8]
    [condition one-hot, fitted at training time + UNK column]

What this DOES NOT include (per user spec):

- ``bc_idx`` or any benchmark identifier
- ``benchmark`` string

Runtime contract
----------------
The schema serializes to a JSON-friendly dict (no numpy arrays inside);
the runtime ``model.py`` rebuilds it via :meth:`MemberFeatureSchema.from_dict`
and then calls :func:`build_member_features_one` with the per-row
inputs the encoder + nn-feature pipeline already produces. No
dependency on ``pandas`` / ``scipy`` / ``sklearn`` at runtime.

Numerical stability
-------------------
- Pool features are z-scored using ``pool_stats`` (mean / std per col).
  ``std == 0`` is replaced with 1.0 to avoid div-by-zero on degenerate
  features.
- Centroid distances are clipped at a high finite value before the
  feature matrix returns, so a NaN / Inf in the embeddings can't
  propagate into the linear head.
- Cluster id 0 means UNK; the one-hot is all-zeros for that row.
- Conditions not seen at training fit map to the UNK column.

This module is import-safe at runtime: only ``numpy`` is imported at
module scope. Re-encoding utilities that need ``pandas``/``MetadataPreprocessor``
live next to the offline trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# These constants must match ``src.item_features.POOL_FEATURE_NAMES``
# and ``src.nn_features.NN_FEATURE_NAMES`` exactly. Kept as a frozen
# tuple here so we don't take an import-time dependency from the
# runtime template (which loads this module and the runtime template
# is allergic to extra imports).
POOL_FEATURE_NAMES_DEFAULT: tuple[str, ...] = (
    "token_len",
    "char_len",
    "has_latex",
    "has_code",
    "n_questions",
    "n_numbers",
    "is_multiple_choice",
    "n_choices",
    "lang_en",
)
NN_FEATURE_NAMES_DEFAULT: tuple[str, ...] = (
    "passrate_mean",
    "passrate_weighted_mean",
    "passrate_std",
    "coverage",
    "top1_label",
    "top1_similarity",
    "mean_similarity",
    "n_labeled_neighbors_log1p",
)


_CONDITION_UNK = "__UNK__"
_CENTROID_DIST_CLAMP = 1.0e6  # squared L2 ceiling; any larger value is
# pathological and gets pinned to keep the linear head finite.


def _canonicalize_condition(c: object) -> str:
    """Same canonicalization both training and runtime use."""
    if c is None:
        return "none"
    s = str(c).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return "none"
    return s


@dataclass
class MemberFeatureSchema:
    """Frozen column layout + per-block lookup state.

    Built once at training time by :meth:`fit`, serialized to JSON
    via :meth:`to_dict`, and re-loaded at inference via
    :meth:`from_dict`. The dense feature vector's column order is
    determined by ``feature_names``; offset attributes are
    convenience indices for the per-row builder.
    """

    feature_names: tuple[str, ...]
    feature_dim: int

    # Block sizes
    k_factors: int
    n_clusters: int
    top_m_centroids: int
    n_pool: int
    n_nn: int
    n_subject_cat_fields: int
    n_subject_num_fields: int
    n_condition_one_hot: int  # includes the UNK column

    # Per-block start offsets into the dense vector
    offset_theta: int
    offset_u: int
    offset_subj_cat: int
    offset_subj_num: int
    offset_pool: int
    offset_centroid: int
    offset_cluster: int
    offset_nn: int
    offset_cond: int

    # Subject-cat per-field offsets within the subj_cat block
    # (offset_subj_cat + subj_cat_field_offsets[i] = column where field
    # i's one-hot starts; all categorical ids are dense from 0..card-1).
    subj_cat_field_offsets: tuple[int, ...]
    subj_cat_field_cardinalities: tuple[int, ...]
    subj_cat_field_names: tuple[str, ...]

    # Subject-num field names (parallel to subj_num input columns; each
    # field contributes 2 cols: scaled value + missingness flag).
    subj_num_field_names: tuple[str, ...]

    # Pool feature names (parallel to the 9-col pool input).
    pool_feature_names: tuple[str, ...]

    # Pool z-score stats: parallel to ``pool_feature_names``.
    # ``pool_means[i]`` and ``pool_stds[i]`` z-score column i.
    pool_means: tuple[float, ...]
    pool_stds: tuple[float, ...]

    # Centroid distance feature names (cosmetic).
    centroid_dist_names: tuple[str, ...]

    # NN feature names (parallel to the 8-col nn input).
    nn_feature_names: tuple[str, ...]

    # Condition vocab: canonical_condition_str -> column offset within
    # the condition-one-hot block. Index 0 is reserved for UNK.
    condition_to_col: dict[str, int]

    @classmethod
    def fit(
        cls,
        *,
        k_factors: int,
        n_clusters: int,
        top_m_centroids: int,
        pool_feature_names: Sequence[str],
        pool_stats: Mapping[str, Mapping[str, float]],
        nn_feature_names: Sequence[str],
        subject_cat_field_names: Sequence[str],
        subject_cat_field_cardinalities: Sequence[int],
        subject_num_field_names: Sequence[str],
        train_conditions: Iterable[object],
        min_condition_count: int = 1,
        centroid_dist_names: Sequence[str] | None = None,
    ) -> "MemberFeatureSchema":
        """Build the schema from training-time statistics.

        Parameters
        ----------
        k_factors : int
            Latent factor dim from the IRT-MLP. Determines the ``u_s`` block
            width.
        n_clusters : int
            Number of clusters from k-means. Cluster id 0 = UNK; only
            ids 1..n_clusters get a one-hot column.
        top_m_centroids : int
            Number of centroid-distance columns.
        pool_feature_names : Sequence[str]
            Order is preserved -- the runtime expects pool features in
            this exact order.
        pool_stats : dict
            ``{col_name: {"mean": float, "std": float}}``. Computed offline
            on the training set and shipped.
        nn_feature_names : Sequence[str]
            Order preserved.
        subject_cat_field_names / subject_cat_field_cardinalities :
            Parallel sequences. ``cardinalities[i]`` is the size of the
            int->id vocab for field ``i`` (includes MISSING=0 and UNK=1
            slots, so the column count for that field's one-hot is exactly
            ``cardinalities[i]``).
        subject_num_field_names : Sequence[str]
            Parallel to the (n, 2*n_num)-wide subject_num input matrix
            (scaled value, missingness flag).
        train_conditions : Iterable
            Raw ``condition`` strings from the training rows. Used to
            fit the condition one-hot vocab.
        min_condition_count : int
            Conditions with fewer than this many rows in training fall
            through to UNK.
        centroid_dist_names : Sequence[str] | None
            If None, defaults to ``("centroid_dist_0", ..., "centroid_dist_{m-1}")``.
        """

        from collections import Counter

        n_pool = int(len(pool_feature_names))
        n_nn = int(len(nn_feature_names))
        n_subj_cat_fields = int(len(subject_cat_field_names))
        n_subj_num_fields = int(len(subject_num_field_names))
        if int(len(subject_cat_field_cardinalities)) != n_subj_cat_fields:
            raise ValueError(
                "subject_cat_field_names and subject_cat_field_cardinalities "
                "must have the same length"
            )

        # Condition vocab fit. Column 0 = UNK; column 1..N = real
        # conditions sorted lexically for determinism.
        cnt: Counter[str] = Counter()
        for c in train_conditions:
            cnt[_canonicalize_condition(c)] += 1
        ordered_real = sorted(
            tok for tok, n in cnt.items() if n >= int(min_condition_count)
        )
        condition_to_col: dict[str, int] = {_CONDITION_UNK: 0}
        for tok in ordered_real:
            if tok == _CONDITION_UNK:
                continue
            condition_to_col.setdefault(tok, len(condition_to_col))
        n_condition_one_hot = len(condition_to_col)

        # Subject-cat offsets within the block.
        subj_cat_field_offsets: list[int] = []
        running = 0
        for card in subject_cat_field_cardinalities:
            subj_cat_field_offsets.append(int(running))
            running += int(card)
        subj_cat_block = int(running)

        # Pool stats lookup: must cover every pool_feature_name.
        pool_means: list[float] = []
        pool_stds: list[float] = []
        for col in pool_feature_names:
            entry = pool_stats.get(str(col)) or {}
            pool_means.append(float(entry.get("mean", 0.0)))
            std = float(entry.get("std", 1.0))
            if not np.isfinite(std) or abs(std) < 1.0e-12:
                std = 1.0
            pool_stds.append(std)

        if centroid_dist_names is None:
            centroid_dist_names = tuple(
                f"centroid_dist_{i}" for i in range(int(top_m_centroids))
            )
        else:
            centroid_dist_names = tuple(str(s) for s in centroid_dist_names)
            if len(centroid_dist_names) != int(top_m_centroids):
                raise ValueError(
                    f"centroid_dist_names length {len(centroid_dist_names)} "
                    f"!= top_m_centroids {int(top_m_centroids)}"
                )

        # Lay out blocks left-to-right. Names are computed in the same
        # order so feature_names matches the dense column layout.
        names: list[str] = []

        offset_theta = len(names)
        names.append("theta_s")

        offset_u = len(names)
        for i in range(int(k_factors)):
            names.append(f"u_s_{i}")

        offset_subj_cat = len(names)
        for fi, fname in enumerate(subject_cat_field_names):
            for ci in range(int(subject_cat_field_cardinalities[fi])):
                names.append(f"subj_cat__{fname}__{ci:03d}")

        offset_subj_num = len(names)
        for fname in subject_num_field_names:
            names.append(f"subj_num__{fname}")
            names.append(f"subj_miss__{fname}")

        offset_pool = len(names)
        for fname in pool_feature_names:
            names.append(f"pool__{fname}")

        offset_centroid = len(names)
        for fname in centroid_dist_names:
            names.append(f"cd__{fname}")

        offset_cluster = len(names)
        for ci in range(int(n_clusters)):
            # cluster id 1..n_clusters maps to columns 0..n_clusters-1
            # within the cluster block; cluster id 0 (UNK) is all-zeros.
            names.append(f"cluster__{ci + 1:03d}")

        offset_nn = len(names)
        for fname in nn_feature_names:
            names.append(f"nn__{fname}")

        offset_cond = len(names)
        # Build cond column names in order of column offset.
        cond_by_col = {col: tok for tok, col in condition_to_col.items()}
        for col in range(n_condition_one_hot):
            names.append(f"cond__{cond_by_col[col]}")

        return cls(
            feature_names=tuple(names),
            feature_dim=int(len(names)),
            k_factors=int(k_factors),
            n_clusters=int(n_clusters),
            top_m_centroids=int(top_m_centroids),
            n_pool=int(n_pool),
            n_nn=int(n_nn),
            n_subject_cat_fields=int(n_subj_cat_fields),
            n_subject_num_fields=int(n_subj_num_fields),
            n_condition_one_hot=int(n_condition_one_hot),
            offset_theta=int(offset_theta),
            offset_u=int(offset_u),
            offset_subj_cat=int(offset_subj_cat),
            offset_subj_num=int(offset_subj_num),
            offset_pool=int(offset_pool),
            offset_centroid=int(offset_centroid),
            offset_cluster=int(offset_cluster),
            offset_nn=int(offset_nn),
            offset_cond=int(offset_cond),
            subj_cat_field_offsets=tuple(subj_cat_field_offsets),
            subj_cat_field_cardinalities=tuple(
                int(c) for c in subject_cat_field_cardinalities
            ),
            subj_cat_field_names=tuple(str(s) for s in subject_cat_field_names),
            subj_num_field_names=tuple(str(s) for s in subject_num_field_names),
            pool_feature_names=tuple(str(s) for s in pool_feature_names),
            pool_means=tuple(float(x) for x in pool_means),
            pool_stds=tuple(float(x) for x in pool_stds),
            centroid_dist_names=centroid_dist_names,
            nn_feature_names=tuple(str(s) for s in nn_feature_names),
            condition_to_col=dict(condition_to_col),
        )

    # ---- serialization ----
    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "feature_dim": int(self.feature_dim),
            "k_factors": int(self.k_factors),
            "n_clusters": int(self.n_clusters),
            "top_m_centroids": int(self.top_m_centroids),
            "n_pool": int(self.n_pool),
            "n_nn": int(self.n_nn),
            "n_subject_cat_fields": int(self.n_subject_cat_fields),
            "n_subject_num_fields": int(self.n_subject_num_fields),
            "n_condition_one_hot": int(self.n_condition_one_hot),
            "offset_theta": int(self.offset_theta),
            "offset_u": int(self.offset_u),
            "offset_subj_cat": int(self.offset_subj_cat),
            "offset_subj_num": int(self.offset_subj_num),
            "offset_pool": int(self.offset_pool),
            "offset_centroid": int(self.offset_centroid),
            "offset_cluster": int(self.offset_cluster),
            "offset_nn": int(self.offset_nn),
            "offset_cond": int(self.offset_cond),
            "subj_cat_field_offsets": list(self.subj_cat_field_offsets),
            "subj_cat_field_cardinalities": list(self.subj_cat_field_cardinalities),
            "subj_cat_field_names": list(self.subj_cat_field_names),
            "subj_num_field_names": list(self.subj_num_field_names),
            "pool_feature_names": list(self.pool_feature_names),
            "pool_means": list(self.pool_means),
            "pool_stds": list(self.pool_stds),
            "centroid_dist_names": list(self.centroid_dist_names),
            "nn_feature_names": list(self.nn_feature_names),
            "condition_to_col": dict(self.condition_to_col),
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "MemberFeatureSchema":
        return cls(
            feature_names=tuple(d["feature_names"]),
            feature_dim=int(d["feature_dim"]),
            k_factors=int(d["k_factors"]),
            n_clusters=int(d["n_clusters"]),
            top_m_centroids=int(d["top_m_centroids"]),
            n_pool=int(d["n_pool"]),
            n_nn=int(d["n_nn"]),
            n_subject_cat_fields=int(d["n_subject_cat_fields"]),
            n_subject_num_fields=int(d["n_subject_num_fields"]),
            n_condition_one_hot=int(d["n_condition_one_hot"]),
            offset_theta=int(d["offset_theta"]),
            offset_u=int(d["offset_u"]),
            offset_subj_cat=int(d["offset_subj_cat"]),
            offset_subj_num=int(d["offset_subj_num"]),
            offset_pool=int(d["offset_pool"]),
            offset_centroid=int(d["offset_centroid"]),
            offset_cluster=int(d["offset_cluster"]),
            offset_nn=int(d["offset_nn"]),
            offset_cond=int(d["offset_cond"]),
            subj_cat_field_offsets=tuple(int(x) for x in d["subj_cat_field_offsets"]),
            subj_cat_field_cardinalities=tuple(
                int(x) for x in d["subj_cat_field_cardinalities"]
            ),
            subj_cat_field_names=tuple(str(x) for x in d["subj_cat_field_names"]),
            subj_num_field_names=tuple(str(x) for x in d["subj_num_field_names"]),
            pool_feature_names=tuple(str(x) for x in d["pool_feature_names"]),
            pool_means=tuple(float(x) for x in d["pool_means"]),
            pool_stds=tuple(float(x) for x in d["pool_stds"]),
            centroid_dist_names=tuple(str(x) for x in d["centroid_dist_names"]),
            nn_feature_names=tuple(str(x) for x in d["nn_feature_names"]),
            condition_to_col={str(k): int(v) for k, v in d["condition_to_col"].items()},
        )


# ---------------------------------------------------------------------------
# Per-subject lookup tables (offline-built; runtime ships them as .npy)
# ---------------------------------------------------------------------------


@dataclass
class MemberSubjectTables:
    """Per-subject row-indexed tables used by both batch and per-row builders.

    Row indexing matches the trained ``Indexer`` (row 0 = UNK subject).
    All arrays are float32 / int64 numpy. No torch / pandas / scipy.
    """

    theta: np.ndarray            # [n_subjects] float32
    u: np.ndarray                # [n_subjects, k_factors] float32
    subject_cat_ids: np.ndarray  # [n_subjects, n_subject_cat_fields] int64
    subject_num: np.ndarray      # [n_subjects, 2 * n_subject_num_fields] float32

    def __post_init__(self) -> None:
        n_s = int(self.theta.shape[0])
        if self.u.shape[0] != n_s:
            raise ValueError(
                f"u rows {self.u.shape[0]} != theta rows {n_s}"
            )
        if self.subject_cat_ids.shape[0] != n_s:
            raise ValueError(
                f"subject_cat_ids rows {self.subject_cat_ids.shape[0]} != {n_s}"
            )
        if self.subject_num.shape[0] != n_s:
            raise ValueError(
                f"subject_num rows {self.subject_num.shape[0]} != {n_s}"
            )

    @property
    def n_subjects(self) -> int:
        return int(self.theta.shape[0])

    def save(self, out_dir) -> None:
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "subject_theta.npy", self.theta.astype(np.float32))
        np.save(out / "subject_u.npy", self.u.astype(np.float32))
        np.save(out / "subject_cat_ids.npy", self.subject_cat_ids.astype(np.int64))
        np.save(out / "subject_num.npy", self.subject_num.astype(np.float32))

    @classmethod
    def load(cls, in_dir) -> "MemberSubjectTables":
        from pathlib import Path
        d = Path(in_dir)
        return cls(
            theta=np.load(d / "subject_theta.npy").astype(np.float32, copy=False),
            u=np.load(d / "subject_u.npy").astype(np.float32, copy=False),
            subject_cat_ids=np.load(d / "subject_cat_ids.npy").astype(
                np.int64, copy=False
            ),
            subject_num=np.load(d / "subject_num.npy").astype(np.float32, copy=False),
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _zscore_pool(pool: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Vectorized pool z-score with safe-std handling."""
    # ``stds`` already has zero-protection from fit(); double-check
    # at apply time too in case the schema was hand-edited.
    safe = np.where(stds > 0, stds, 1.0)
    return (pool - means) / safe


def build_member_features(
    schema: MemberFeatureSchema,
    subject_tables: MemberSubjectTables,
    *,
    subject_idx: np.ndarray,        # [N] int
    pool_feats: np.ndarray,         # [N, n_pool] float32 raw
    centroid_dists: np.ndarray,     # [N, top_m_centroids] float32
    cluster_ids: np.ndarray,        # [N] int (0 = UNK, 1..K = real)
    nn_feats: np.ndarray,           # [N, n_nn] float32
    conditions: Sequence[object],   # [N] raw condition strings
) -> np.ndarray:
    """Build the full ``[N, feature_dim]`` dense feature matrix (float32).

    All inputs are validated for shape; mismatches raise ``ValueError``
    immediately so you don't silently train a model on wrongly-keyed
    data.
    """
    N = int(subject_idx.shape[0])

    if pool_feats.shape != (N, schema.n_pool):
        raise ValueError(
            f"pool_feats shape {pool_feats.shape} != ({N}, {schema.n_pool})"
        )
    if centroid_dists.shape != (N, schema.top_m_centroids):
        raise ValueError(
            f"centroid_dists shape {centroid_dists.shape} != "
            f"({N}, {schema.top_m_centroids})"
        )
    if cluster_ids.shape != (N,):
        raise ValueError(
            f"cluster_ids shape {cluster_ids.shape} != ({N},)"
        )
    if nn_feats.shape != (N, schema.n_nn):
        raise ValueError(f"nn_feats shape {nn_feats.shape} != ({N}, {schema.n_nn})")
    if int(len(conditions)) != N:
        raise ValueError(f"conditions length {len(conditions)} != {N}")

    out = np.zeros((N, schema.feature_dim), dtype=np.float32)

    # ---- theta ----
    out[:, schema.offset_theta] = subject_tables.theta[subject_idx]

    # ---- u_s ----
    out[
        :, schema.offset_u : schema.offset_u + schema.k_factors
    ] = subject_tables.u[subject_idx]

    # ---- subject categorical one-hot ----
    cat_ids = subject_tables.subject_cat_ids[subject_idx]  # [N, n_cat_fields]
    rows = np.arange(N)
    base = schema.offset_subj_cat
    for fi, field_off in enumerate(schema.subj_cat_field_offsets):
        ids = cat_ids[:, fi].astype(np.int64, copy=False)
        # Clip to [0, cardinality-1] defensively. UNK already at 1; out-of-
        # range becomes 1 (UNK) rather than an OOB write.
        card = int(schema.subj_cat_field_cardinalities[fi])
        ids = np.clip(ids, 0, card - 1)
        out[rows, base + field_off + ids] = 1.0

    # ---- subject numeric (scaled value + missingness) ----
    out[
        :, schema.offset_subj_num
        : schema.offset_subj_num + 2 * schema.n_subject_num_fields
    ] = subject_tables.subject_num[subject_idx]

    # ---- pool z-score ----
    pool_means = np.asarray(schema.pool_means, dtype=np.float32)
    pool_stds = np.asarray(schema.pool_stds, dtype=np.float32)
    out[
        :, schema.offset_pool : schema.offset_pool + schema.n_pool
    ] = _zscore_pool(pool_feats.astype(np.float32, copy=False), pool_means, pool_stds)

    # ---- centroid distances (clamped) ----
    cd = np.clip(
        centroid_dists.astype(np.float32, copy=False),
        0.0,
        _CENTROID_DIST_CLAMP,
    )
    out[
        :, schema.offset_centroid : schema.offset_centroid + schema.top_m_centroids
    ] = cd

    # ---- cluster one-hot ----
    cl = cluster_ids.astype(np.int64, copy=False)
    valid = (cl >= 1) & (cl <= schema.n_clusters)
    if int(valid.sum()) > 0:
        valid_rows = np.where(valid)[0]
        col = (cl[valid] - 1).astype(np.int64)
        out[valid_rows, schema.offset_cluster + col] = 1.0

    # ---- NN feats ----
    out[:, schema.offset_nn : schema.offset_nn + schema.n_nn] = nn_feats.astype(
        np.float32, copy=False
    )

    # ---- condition one-hot ----
    for i, c in enumerate(conditions):
        col = schema.condition_to_col.get(_canonicalize_condition(c), 0)
        out[i, schema.offset_cond + col] = 1.0

    # Belt-and-suspenders against NaN / Inf in any block.
    np.nan_to_num(out, copy=False, nan=0.0, posinf=_CENTROID_DIST_CLAMP, neginf=0.0)
    return out


def build_member_features_one(
    schema: MemberFeatureSchema,
    *,
    theta_s: float,
    u_s: np.ndarray,            # [k_factors] float32
    subject_cat_ids: np.ndarray,  # [n_subj_cat_fields] int
    subject_num: np.ndarray,    # [2 * n_subj_num_fields] float32
    pool_feats: np.ndarray,     # [n_pool] float32
    centroid_dists: np.ndarray, # [top_m_centroids] float32
    cluster_id: int,            # 0 = UNK
    nn_feats: np.ndarray,       # [n_nn] float32
    condition: object,
) -> np.ndarray:
    """Build a single ``[feature_dim]`` row. Used by the runtime ``predict``.

    Same numerical contract as :func:`build_member_features` but
    optimized for the per-row hot path: no allocation of per-row
    auxiliary arrays.
    """
    out = np.zeros(schema.feature_dim, dtype=np.float32)

    out[schema.offset_theta] = float(theta_s)
    out[schema.offset_u : schema.offset_u + schema.k_factors] = u_s

    base = schema.offset_subj_cat
    for fi, field_off in enumerate(schema.subj_cat_field_offsets):
        card = int(schema.subj_cat_field_cardinalities[fi])
        ci = int(subject_cat_ids[fi])
        if ci < 0 or ci >= card:
            ci = 1  # fall through to UNK
        out[base + field_off + ci] = 1.0

    out[
        schema.offset_subj_num
        : schema.offset_subj_num + 2 * schema.n_subject_num_fields
    ] = subject_num

    pool_means = np.asarray(schema.pool_means, dtype=np.float32)
    pool_stds = np.asarray(schema.pool_stds, dtype=np.float32)
    safe = np.where(pool_stds > 0, pool_stds, 1.0)
    out[schema.offset_pool : schema.offset_pool + schema.n_pool] = (
        pool_feats.astype(np.float32, copy=False) - pool_means
    ) / safe

    cd = np.clip(
        centroid_dists.astype(np.float32, copy=False),
        0.0,
        _CENTROID_DIST_CLAMP,
    )
    out[schema.offset_centroid : schema.offset_centroid + schema.top_m_centroids] = cd

    cl = int(cluster_id)
    if 1 <= cl <= schema.n_clusters:
        out[schema.offset_cluster + cl - 1] = 1.0

    out[schema.offset_nn : schema.offset_nn + schema.n_nn] = nn_feats.astype(
        np.float32, copy=False
    )

    col = schema.condition_to_col.get(_canonicalize_condition(condition), 0)
    out[schema.offset_cond + col] = 1.0

    # Belt-and-suspenders.
    np.nan_to_num(out, copy=False, nan=0.0, posinf=_CENTROID_DIST_CLAMP, neginf=0.0)
    return out


__all__ = [
    "MemberFeatureSchema",
    "MemberSubjectTables",
    "build_member_features",
    "build_member_features_one",
    "POOL_FEATURE_NAMES_DEFAULT",
    "NN_FEATURE_NAMES_DEFAULT",
]
