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
    [nn_feats raw, NN_FEATURE_DIM = 15]
    [condition one-hot, fitted at training time + UNK column]
    [bc block (u_bc + crosses), schema_version >= 2, optional]
    [bench_cat one-hot, sum of benchmark vocab cardinalities, v3, optional]
    [bench_num scaled values + missingness flags, 2 * n_bench_num, v3, optional]

Schema versions (purely additive; a higher version is a strict
superset of the lower ones, so every serialized cache round-trips):

- v1: subject-only layout (the original four-member contract).
- v2: + bc block (``u_bc`` / ``cross_theta_u_*``) when ``k_bc_factors > 0``.
- v3: + benchmark-metadata block (``bench_cat`` one-hot + ``bench_num``
  value/missingness) when benchmark fields are configured. This is the
  benchmark-side analog of the subject metadata + ``cond`` blocks and
  feeds the shared-matrix consumers (gbdt/xgb/cat/forest/knn/fm/logreg)
  the benchmark metadata they previously lacked.

The benchmark identifier (``bc_idx``) is used only to *index* the
shipped per-benchmark tables; no raw ``benchmark`` string enters the
dense vector.

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

    # ---- BC block (optional, schema_version >= 2) ----
    #
    # When ``k_bc_factors > 0`` the dense vector ends with three
    # bc-derived blocks. ``u_bc`` is the IRT-MLP per-bc factor
    # (analog of ``u_s``), and the two cross-term blocks give the
    # linear logreg member access to the (theta_s * u_s) and
    # (theta_s * u_bc) products that the IRT model multiplies
    # internally. Without those crosses the linear member can only
    # see the marginals -- it is structurally unable to represent
    # "subject_s is good at items they normally get right" without
    # a polynomial expansion.
    #
    # When ``k_bc_factors == 0`` all offsets/sizes below are 0 and
    # the schema reduces to the legacy layout (matches every existing
    # serialized cache).
    k_bc_factors: int = 0
    offset_u_bc: int = 0
    offset_cross_theta_u_s: int = 0
    offset_cross_theta_u_bc: int = 0

    # ---- Benchmark-metadata block (optional, schema_version >= 3) ----
    #
    # Mirrors the subject-metadata blocks (``subj_cat`` one-hot +
    # ``subj_num`` scaled-value/missingness) but on the *benchmark*
    # side, indexed by ``bc_idx`` rather than ``subject_idx``. The
    # shared-matrix consumers (gbdt/xgb/cat/forest/knn/fm/logreg) get
    # subject metadata + ``cond`` today but no benchmark metadata; this
    # block closes that gap.
    #
    # Two appended blocks, in this order, at the *very end* of the
    # dense vector (after the bc block when one is present):
    #
    #   bench_cat : one-hot, ``sum(benchmark_cat_field_cardinalities)``
    #               columns. Per-field offsets in ``bench_cat_field_offsets``
    #               (analog of ``subj_cat_field_offsets``).
    #   bench_num : 2 columns per numeric field -- scaled value followed
    #               by a missingness flag (analog of ``subj_num``).
    #
    # Column names follow the locked convention:
    #   ``bench_cat__{field}__{id:03d}`` / ``bench_num__{field}`` /
    #   ``bench_miss__{field}``.
    #
    # When ``n_bench_cat_fields == 0`` (and no benchmark numeric fields)
    # all offsets/sizes below are 0 and the schema is byte-identical to
    # the v2 layout -- every existing serialized cache round-trips.
    #
    # Benchmark *metadata joins* are static (FOLD_INVARIANT): the same
    # for every fold and not derived from labels, so populating this
    # block introduces no OOF leakage (unlike y-aggregates, which are
    # handled in the aide feature-store path, never here).
    n_bench_cat_fields: int = 0
    n_bench_num_fields: int = 0
    offset_bench_cat: int = 0
    offset_bench_num: int = 0
    bench_cat_field_offsets: tuple[int, ...] = ()
    benchmark_cat_field_cardinalities: tuple[int, ...] = ()
    benchmark_cat_field_names: tuple[str, ...] = ()
    benchmark_num_field_names: tuple[str, ...] = ()

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
        k_bc_factors: int = 0,
        benchmark_cat_field_names: Sequence[str] = (),
        benchmark_cat_field_cardinalities: Sequence[int] = (),
        benchmark_num_field_names: Sequence[str] = (),
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
        benchmark_cat_field_names / benchmark_cat_field_cardinalities :
            Parallel sequences for the benchmark-metadata one-hot block
            (schema_version 3). ``cardinalities[i]`` is the int->id vocab
            size for benchmark categorical field ``i`` (includes
            MISSING=0 and UNK=1 slots). Leave empty for the v2 layout.
        benchmark_num_field_names : Sequence[str]
            Benchmark numeric field names; each contributes 2 columns
            (scaled value + missingness flag). Leave empty for the v2
            layout.
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
        n_bench_cat_fields = int(len(benchmark_cat_field_names))
        n_bench_num_fields = int(len(benchmark_num_field_names))
        if int(len(benchmark_cat_field_cardinalities)) != n_bench_cat_fields:
            raise ValueError(
                "benchmark_cat_field_names and benchmark_cat_field_cardinalities "
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

        # Benchmark-cat offsets within the (later-laid-out) bench_cat block.
        bench_cat_field_offsets: list[int] = []
        running_bc = 0
        for card in benchmark_cat_field_cardinalities:
            bench_cat_field_offsets.append(int(running_bc))
            running_bc += int(card)

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

        # ---- Optional bc block (u_bc, cross_theta_u_s, cross_theta_u_bc) ----
        kbc = int(k_bc_factors)
        offset_u_bc = 0
        offset_cross_theta_u_s = 0
        offset_cross_theta_u_bc = 0
        if kbc > 0:
            offset_u_bc = len(names)
            for i in range(kbc):
                names.append(f"u_bc_{i}")
            offset_cross_theta_u_s = len(names)
            for i in range(int(k_factors)):
                names.append(f"cross_theta_u_s_{i}")
            offset_cross_theta_u_bc = len(names)
            for i in range(kbc):
                names.append(f"cross_theta_u_bc_{i}")

        # ---- Benchmark-metadata block (bench_cat one-hot + bench_num) ----
        # Appended at the very end so the v2 layout (no benchmark fields)
        # is byte-identical: column count and names are unchanged when
        # n_bench_cat_fields == 0 and n_bench_num_fields == 0.
        offset_bench_cat = 0
        offset_bench_num = 0
        if n_bench_cat_fields > 0 or n_bench_num_fields > 0:
            offset_bench_cat = len(names)
            for fi, fname in enumerate(benchmark_cat_field_names):
                for ci in range(int(benchmark_cat_field_cardinalities[fi])):
                    names.append(f"bench_cat__{fname}__{ci:03d}")
            offset_bench_num = len(names)
            for fname in benchmark_num_field_names:
                names.append(f"bench_num__{fname}")
                names.append(f"bench_miss__{fname}")

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
            k_bc_factors=int(kbc),
            offset_u_bc=int(offset_u_bc),
            offset_cross_theta_u_s=int(offset_cross_theta_u_s),
            offset_cross_theta_u_bc=int(offset_cross_theta_u_bc),
            n_bench_cat_fields=int(n_bench_cat_fields),
            n_bench_num_fields=int(n_bench_num_fields),
            offset_bench_cat=int(offset_bench_cat),
            offset_bench_num=int(offset_bench_num),
            bench_cat_field_offsets=tuple(bench_cat_field_offsets),
            benchmark_cat_field_cardinalities=tuple(
                int(c) for c in benchmark_cat_field_cardinalities
            ),
            benchmark_cat_field_names=tuple(
                str(s) for s in benchmark_cat_field_names
            ),
            benchmark_num_field_names=tuple(
                str(s) for s in benchmark_num_field_names
            ),
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
            "k_bc_factors": int(self.k_bc_factors),
            "offset_u_bc": int(self.offset_u_bc),
            "offset_cross_theta_u_s": int(self.offset_cross_theta_u_s),
            "offset_cross_theta_u_bc": int(self.offset_cross_theta_u_bc),
            "n_bench_cat_fields": int(self.n_bench_cat_fields),
            "n_bench_num_fields": int(self.n_bench_num_fields),
            "offset_bench_cat": int(self.offset_bench_cat),
            "offset_bench_num": int(self.offset_bench_num),
            "bench_cat_field_offsets": list(self.bench_cat_field_offsets),
            "benchmark_cat_field_cardinalities": list(
                self.benchmark_cat_field_cardinalities
            ),
            "benchmark_cat_field_names": list(self.benchmark_cat_field_names),
            "benchmark_num_field_names": list(self.benchmark_num_field_names),
            "schema_version": self._schema_version(),
        }

    def _schema_version(self) -> int:
        """Highest layout version the schema actually uses.

        v3 = a benchmark-metadata block is present; v2 = the bc
        (u_bc/cross) block is present; v1 = the legacy subject-only
        layout. A v3 schema is a strict additive superset of v2/v1, so
        the bumped version is purely informational for cache-validation
        whitelists.
        """
        if self.n_bench_cat_fields > 0 or self.n_bench_num_fields > 0:
            return 3
        if self.k_bc_factors > 0:
            return 2
        return 1

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
            k_bc_factors=int(d.get("k_bc_factors", 0)),
            offset_u_bc=int(d.get("offset_u_bc", 0)),
            offset_cross_theta_u_s=int(d.get("offset_cross_theta_u_s", 0)),
            offset_cross_theta_u_bc=int(d.get("offset_cross_theta_u_bc", 0)),
            n_bench_cat_fields=int(d.get("n_bench_cat_fields", 0)),
            n_bench_num_fields=int(d.get("n_bench_num_fields", 0)),
            offset_bench_cat=int(d.get("offset_bench_cat", 0)),
            offset_bench_num=int(d.get("offset_bench_num", 0)),
            bench_cat_field_offsets=tuple(
                int(x) for x in d.get("bench_cat_field_offsets", [])
            ),
            benchmark_cat_field_cardinalities=tuple(
                int(x) for x in d.get("benchmark_cat_field_cardinalities", [])
            ),
            benchmark_cat_field_names=tuple(
                str(x) for x in d.get("benchmark_cat_field_names", [])
            ),
            benchmark_num_field_names=tuple(
                str(x) for x in d.get("benchmark_num_field_names", [])
            ),
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


@dataclass
class MemberBenchmarkTables:
    """Per-benchmark-condition row-indexed metadata tables (schema v3).

    Mirrors :class:`MemberSubjectTables` but on the benchmark side:
    indexed by ``bc_idx`` (``Indexer.bc_to_id``; row 0 = UNK bc =
    MISSING for every field). Built offline by
    :func:`src.member_features_meta.build_shared_matrix` from
    :meth:`MetadataPreprocessor.encode_benchmark`, then shipped as
    ``.npy`` and indexed at runtime. No torch / pandas / scipy.
    """

    benchmark_cat_ids: np.ndarray  # [n_bc, n_bench_cat_fields] int64
    benchmark_num: np.ndarray      # [n_bc, 2 * n_bench_num_fields] float32

    def __post_init__(self) -> None:
        n_bc = int(self.benchmark_cat_ids.shape[0])
        if self.benchmark_num.shape[0] != n_bc:
            raise ValueError(
                f"benchmark_num rows {self.benchmark_num.shape[0]} != "
                f"benchmark_cat_ids rows {n_bc}"
            )

    @property
    def n_bc(self) -> int:
        return int(self.benchmark_cat_ids.shape[0])

    def save(self, out_dir) -> None:
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(
            out / "benchmark_cat_ids.npy",
            self.benchmark_cat_ids.astype(np.int64),
        )
        np.save(
            out / "benchmark_num.npy",
            self.benchmark_num.astype(np.float32),
        )

    @classmethod
    def load(cls, in_dir) -> "MemberBenchmarkTables":
        from pathlib import Path
        d = Path(in_dir)
        return cls(
            benchmark_cat_ids=np.load(d / "benchmark_cat_ids.npy").astype(
                np.int64, copy=False
            ),
            benchmark_num=np.load(d / "benchmark_num.npy").astype(
                np.float32, copy=False
            ),
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
    bc_redacted: np.ndarray | None = None,   # [N] bool: zero cond one-hot
    u_bc_per_row: np.ndarray | None = None,  # [N, k_bc_factors] float32
    benchmark_tables: "MemberBenchmarkTables | None" = None,
    bc_idx: np.ndarray | None = None,        # [N] int: row into benchmark_tables
) -> np.ndarray:
    """Build the full ``[N, feature_dim]`` dense feature matrix (float32).

    All inputs are validated for shape; mismatches raise ``ValueError``
    immediately so you don't silently train a model on wrongly-keyed
    data.

    ``bc_redacted`` (per-row bool): when True, the row's bc-derived
    feature blocks are zeroed -- specifically the ``cond`` one-hot.
    This simulates "benchmark unknown" for offline training, mirroring
    the leaderboard cold-start regime where the runtime sees rows
    whose benchmark identity is missing. The row's UNK-cond column
    is *also left zero*; the model learns to read absence of any
    cond signal as redaction (rather than asserting UNK explicitly,
    which would conflate redacted-bc rows with rows whose cond was
    just rare-and-coalesced into UNK).

    ``benchmark_tables`` / ``bc_idx`` (schema_version 3): the
    benchmark-metadata block (``bench_cat`` one-hot + ``bench_num``
    value/missingness) is filled from ``benchmark_tables`` indexed by
    ``bc_idx``. Required when the schema has benchmark fields. A
    ``bc_redacted`` row has its entire benchmark block zeroed too
    (cold benchmark -> no benchmark metadata), exactly mirroring the
    ``cond`` redaction.
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
    if bc_redacted is not None and bc_redacted.shape != (N,):
        raise ValueError(
            f"bc_redacted shape {bc_redacted.shape} != ({N},)"
        )
    if int(schema.k_bc_factors) > 0:
        if u_bc_per_row is None:
            raise ValueError(
                f"schema.k_bc_factors={schema.k_bc_factors} requires "
                "u_bc_per_row to be provided"
            )
        if u_bc_per_row.shape != (N, int(schema.k_bc_factors)):
            raise ValueError(
                f"u_bc_per_row shape {u_bc_per_row.shape} != "
                f"({N}, {schema.k_bc_factors})"
            )
    _has_bench_block = (
        int(schema.n_bench_cat_fields) > 0 or int(schema.n_bench_num_fields) > 0
    )
    if _has_bench_block:
        if benchmark_tables is None or bc_idx is None:
            raise ValueError(
                "schema has a benchmark-metadata block "
                f"(n_bench_cat_fields={schema.n_bench_cat_fields}, "
                f"n_bench_num_fields={schema.n_bench_num_fields}) but "
                "benchmark_tables / bc_idx were not provided"
            )
        bc_idx_arr = np.asarray(bc_idx)
        if bc_idx_arr.shape != (N,):
            raise ValueError(f"bc_idx shape {bc_idx_arr.shape} != ({N},)")
        if benchmark_tables.benchmark_cat_ids.shape[1] != int(
            schema.n_bench_cat_fields
        ):
            raise ValueError(
                "benchmark_tables.benchmark_cat_ids has "
                f"{benchmark_tables.benchmark_cat_ids.shape[1]} fields != "
                f"schema.n_bench_cat_fields {schema.n_bench_cat_fields}"
            )
        if benchmark_tables.benchmark_num.shape[1] != 2 * int(
            schema.n_bench_num_fields
        ):
            raise ValueError(
                "benchmark_tables.benchmark_num has "
                f"{benchmark_tables.benchmark_num.shape[1]} cols != "
                f"2 * schema.n_bench_num_fields {2 * schema.n_bench_num_fields}"
            )

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

    # ---- per-row redaction mask (hoisted so every downstream block can
    # reference it without a possibly-unbound hazard). None when there is
    # no per-row redaction signal at all. ----
    red = np.asarray(bc_redacted, dtype=bool) if bc_redacted is not None else None

    # ---- condition one-hot ----
    if red is None:
        for i, c in enumerate(conditions):
            col = schema.condition_to_col.get(_canonicalize_condition(c), 0)
            out[i, schema.offset_cond + col] = 1.0
    else:
        # Per-row redaction: redacted rows leave the entire cond block
        # at zero; non-redacted rows light up their canonical column.
        for i, c in enumerate(conditions):
            if bool(red[i]):
                continue
            col = schema.condition_to_col.get(_canonicalize_condition(c), 0)
            out[i, schema.offset_cond + col] = 1.0

    # ---- BC block (u_bc + crosses), schema_version >= 2 ----
    if int(schema.k_bc_factors) > 0 and u_bc_per_row is not None:
        kbc = int(schema.k_bc_factors)
        kf = int(schema.k_factors)
        u_bc_f32 = np.asarray(u_bc_per_row, dtype=np.float32)
        # If the row is bc-redacted, zero u_bc and the cross_theta_u_bc
        # block. cross_theta_u_s remains nonzero -- it depends only on
        # subject-side state and is not affected by bc redaction.
        if red is not None:
            u_bc_f32 = u_bc_f32 * (1.0 - red.astype(np.float32))[:, None]
        out[:, schema.offset_u_bc : schema.offset_u_bc + kbc] = u_bc_f32

        theta_col = out[:, schema.offset_theta : schema.offset_theta + 1]
        u_s_block = out[:, schema.offset_u : schema.offset_u + kf]
        out[
            :,
            schema.offset_cross_theta_u_s
            : schema.offset_cross_theta_u_s + kf
        ] = theta_col * u_s_block
        out[
            :,
            schema.offset_cross_theta_u_bc
            : schema.offset_cross_theta_u_bc + kbc
        ] = theta_col * u_bc_f32

    # ---- Benchmark-metadata block (bench_cat + bench_num), v3 ----
    if _has_bench_block:
        # _has_bench_block implies both were validated as non-None above;
        # re-assert here so the deref below is provably safe (not just at
        # runtime). These are no-ops on the validated path.
        assert benchmark_tables is not None and bc_idx is not None, (
            "internal error: _has_bench_block is True but benchmark_tables / "
            "bc_idx is None (should have been caught by validation above)"
        )
        bc_rows = np.asarray(bc_idx).astype(np.int64, copy=False)
        # Per-row redaction mask: a cold benchmark contributes no
        # benchmark metadata (entire block zeroed), mirroring cond.
        if red is not None:
            keep = (~red)
        else:
            keep = np.ones(N, dtype=bool)

        # bench_cat one-hot.
        n_bcat = int(schema.n_bench_cat_fields)
        if n_bcat > 0:
            bcat_ids = benchmark_tables.benchmark_cat_ids[bc_rows]  # [N, n_bcat]
            base_bc = schema.offset_bench_cat
            for fi, field_off in enumerate(schema.bench_cat_field_offsets):
                card = int(schema.benchmark_cat_field_cardinalities[fi])
                ids = np.clip(
                    bcat_ids[:, fi].astype(np.int64, copy=False), 0, card - 1
                )
                lit = keep  # redacted rows leave the field all-zero
                if int(lit.sum()) > 0:
                    lit_rows = np.where(lit)[0]
                    out[lit_rows, base_bc + field_off + ids[lit]] = 1.0

        # bench_num scaled value + missingness.
        n_bnum = int(schema.n_bench_num_fields)
        if n_bnum > 0:
            bnum = benchmark_tables.benchmark_num[bc_rows].astype(
                np.float32, copy=False
            )  # [N, 2 * n_bnum]
            bnum = bnum * keep.astype(np.float32)[:, None]
            out[
                :, schema.offset_bench_num : schema.offset_bench_num + 2 * n_bnum
            ] = bnum

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
    bc_redacted: bool = False,
    u_bc: np.ndarray | None = None,   # [k_bc_factors] float32, required when schema enables bc block
    benchmark_cat_ids: np.ndarray | None = None,  # [n_bench_cat_fields] int
    benchmark_num: np.ndarray | None = None,      # [2 * n_bench_num_fields] float32
) -> np.ndarray:
    """Build a single ``[feature_dim]`` row. Used by the runtime ``predict``.

    Same numerical contract as :func:`build_member_features` but
    optimized for the per-row hot path: no allocation of per-row
    auxiliary arrays.

    ``bc_redacted=True`` zeroes the cond one-hot block AND the
    bc-derived blocks (``u_bc`` and ``cross_theta_u_bc``), mirroring
    :func:`build_member_features`.

    ``u_bc`` is required when ``schema.k_bc_factors > 0``; pass an
    all-zero vector for redacted / unknown bcs.

    ``benchmark_cat_ids`` / ``benchmark_num`` (schema_version 3) are
    required when the schema has a benchmark-metadata block. They are
    the per-row slices of :class:`MemberBenchmarkTables` for this row's
    ``bc_idx`` (``benchmark_cat_ids`` length ``n_bench_cat_fields``;
    ``benchmark_num`` length ``2 * n_bench_num_fields``, interleaved
    ``[value, miss, value, miss, ...]``). ``bc_redacted=True`` zeroes
    the entire benchmark block too.
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

    if not bool(bc_redacted):
        col = schema.condition_to_col.get(_canonicalize_condition(condition), 0)
        out[schema.offset_cond + col] = 1.0
    # else: cond block stays at all-zero (redacted bc).

    # ---- BC block (u_bc + crosses), schema_version >= 2 ----
    if int(schema.k_bc_factors) > 0:
        if u_bc is None:
            raise ValueError(
                "schema.k_bc_factors > 0 requires u_bc to be provided "
                "(pass an all-zero vector for redacted/unknown bcs)"
            )
        kbc = int(schema.k_bc_factors)
        kf = int(schema.k_factors)
        u_bc_f32 = np.asarray(u_bc, dtype=np.float32).reshape(-1)
        if int(u_bc_f32.shape[0]) != kbc:
            raise ValueError(
                f"u_bc shape {u_bc_f32.shape} != ({kbc},)"
            )
        # Apply bc-redaction to u_bc and cross_theta_u_bc; cross_theta_u_s
        # is subject-only and remains.
        if bool(bc_redacted):
            u_bc_use = np.zeros_like(u_bc_f32)
        else:
            u_bc_use = u_bc_f32
        out[schema.offset_u_bc : schema.offset_u_bc + kbc] = u_bc_use
        theta_val = float(theta_s)
        out[
            schema.offset_cross_theta_u_s
            : schema.offset_cross_theta_u_s + kf
        ] = theta_val * u_s.astype(np.float32, copy=False)
        out[
            schema.offset_cross_theta_u_bc
            : schema.offset_cross_theta_u_bc + kbc
        ] = theta_val * u_bc_use

    # ---- Benchmark-metadata block (bench_cat + bench_num), v3 ----
    n_bcat = int(schema.n_bench_cat_fields)
    n_bnum = int(schema.n_bench_num_fields)
    if n_bcat > 0 or n_bnum > 0:
        if benchmark_cat_ids is None and n_bcat > 0:
            raise ValueError(
                "schema has a benchmark-cat block but benchmark_cat_ids "
                "was not provided (pass an all-MISSING vector for "
                "redacted/unknown bcs)"
            )
        if benchmark_num is None and n_bnum > 0:
            raise ValueError(
                "schema has a benchmark-num block but benchmark_num "
                "was not provided (pass zeros for redacted/unknown bcs)"
            )
        # Cold benchmark -> entire benchmark block stays zero.
        if not bool(bc_redacted):
            base_bc = schema.offset_bench_cat
            if n_bcat > 0:
                # n_bcat > 0 implies benchmark_cat_ids is non-None (the
                # guard above raised otherwise); re-assert so the subscript
                # is provably safe. No-op on the validated path.
                assert benchmark_cat_ids is not None, (
                    "internal error: n_bench_cat_fields > 0 but "
                    "benchmark_cat_ids is None"
                )
                for fi, field_off in enumerate(schema.bench_cat_field_offsets):
                    card = int(schema.benchmark_cat_field_cardinalities[fi])
                    ci = int(benchmark_cat_ids[fi])
                    if ci < 0 or ci >= card:
                        ci = 1  # fall through to UNK
                    out[base_bc + field_off + ci] = 1.0
            if n_bnum > 0:
                bnum_vec = np.asarray(benchmark_num, dtype=np.float32).reshape(-1)
                if int(bnum_vec.shape[0]) != 2 * n_bnum:
                    raise ValueError(
                        f"benchmark_num shape {bnum_vec.shape} != "
                        f"({2 * n_bnum},)"
                    )
                out[
                    schema.offset_bench_num
                    : schema.offset_bench_num + 2 * n_bnum
                ] = bnum_vec
        # else: redacted -> benchmark block left all-zero.

    # Belt-and-suspenders.
    np.nan_to_num(out, copy=False, nan=0.0, posinf=_CENTROID_DIST_CLAMP, neginf=0.0)
    return out


__all__ = [
    "MemberFeatureSchema",
    "MemberSubjectTables",
    "MemberBenchmarkTables",
    "build_member_features",
    "build_member_features_one",
    "POOL_FEATURE_NAMES_DEFAULT",
    "NN_FEATURE_NAMES_DEFAULT",
]
