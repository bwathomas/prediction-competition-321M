"""Canonical feature catalog — the single source of truth for the data suite the AIDE
agent may feed models at the ablation layer, and the source the proxy-tree + coverage
probe derive from.

Scope decisions (per project owner, 2026-06-04):
  * INCLUDED: the CSV metadata in data/metadata/{model_info,benchmark_info}.csv in full,
    including the qualitative benchmark fields (has_conditions + the conditions JSON) that
    earlier pipelines dropped.
  * REMOVED: LLM-as-judge, solver-proxy, PCA-tail, and every feature derived from the
    LEARNED PARAMETERS of another model — the IRT latent factors (theta_s, u_s_*, u_bc_*,
    cross_*), the member-model outputs (member5_*), and raw identity index columns
    (subject_idx, bench_condition_id, bc_redacted_mask). Content embeddings are KEPT: they
    encode item/subject TEXT, not another model's fitted behaviour.

Dropout class:
  * neutral_item   — item CONTENT (embeddings, text stats, clusters, semantic category).
                     NEVER masked: it is exactly how a model generalizes to an UNSEEN
                     benchmark, so masking it would defeat cold-start learning. (This
                     deliberately reclassifies pool/cluster from the old code's
                     benchmark-masked treatment.)
  * subject_proxy  — anything that encodes WHICH SUBJECT (identity / ability / its
                     neighbour pass-rates / its metadata).
  * benchmark_proxy— anything that encodes WHICH BENCHMARK/CONDITION (name, topic, age,
                     condition schema, benchmark-keyed target encodings).
A column may be in BOTH proxy lists (e.g. a subject x benchmark mean-encoding) — it is then
masked under either dropout. Matching uses the "__" boundary rule (see ``matches``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def matches(column: str, pattern: str) -> bool:
    """A pattern matches a column iff exact, or the column is a ``pattern + "__"`` group."""
    return column == pattern or column.startswith(pattern + "__")


@dataclass(frozen=True)
class FeatureGroup:
    name: str           # logical group = funnel cache-group key
    patterns: tuple     # exact names or "__"-group prefixes
    axis: str           # "item" | "subject" | "benchmark" | "pair"
    dropout_class: str  # "neutral_item" | "subject_proxy" | "benchmark_proxy"
    source: str         # originating src/ module (provenance)
    note: str = ""
    also: tuple = field(default_factory=tuple)          # secondary proxy class(es)
    also_patterns: tuple = field(default_factory=tuple)  # specific patterns for the 2nd axis


CATALOG = (
    # ---- item content (neutral: usable, never masked) -------------------------------
    FeatureGroup("item_embedding", ("item_emb",), "item", "neutral_item",
                 "embeddings.py", "dense item-text vector (encoder content embedding)"),
    FeatureGroup("item_pool", ("pool",), "item", "neutral_item", "item_features.py",
                 "9 text stats: token_len,char_len,has_latex,has_code,n_questions,"
                 "n_numbers,is_multiple_choice,n_choices,lang_en"),
    FeatureGroup("centroid_distance", ("cd",), "item", "neutral_item", "item_features.py",
                 "cd__centroid_dist_{i}: item-to-kmeans-centroid squared L2"),
    FeatureGroup("item_cluster", ("cluster", "cluster_id"), "item", "neutral_item",
                 "clustering.py", "cluster__{NNN} one-hot + cluster_id of item embedding"),
    FeatureGroup("semantic_category", ("semcat", "semcat_id"), "item", "neutral_item",
                 "semantic_categories.py", "15 regex-derived item categories (heuristic)"),
    FeatureGroup("cluster_passrate", ("m2_cluster",), "item", "neutral_item",
                 "mean_encoded_features.py", "m2_cluster_mean: item-cluster difficulty (target enc.)"),

    # ---- subject identity (masked under subject dropout) ----------------------------
    FeatureGroup("subject_key", ("subject_key",), "subject", "subject_proxy", "data.py"),
    FeatureGroup("subject_content", ("subject_content",), "subject", "subject_proxy", "data.py"),
    FeatureGroup("subject_embedding", ("subj_emb",), "subject", "subject_proxy",
                 "embeddings.py", "dense subject-text vector"),
    FeatureGroup("subject_meta_categorical", ("subj_cat",), "subject", "subject_proxy",
                 "metadata_features.py", "CSV: organization, family, macro_family (vocab ids)"),
    FeatureGroup("subject_meta_numeric", ("subj_num",), "subject", "subject_proxy",
                 "metadata_features.py", "CSV: parameters(log), release_date (+__missing)"),
    FeatureGroup("subject_mean", ("subject_mean", "subject_obs_count"), "subject",
                 "subject_proxy", "subject_mean.py", "shrunk per-subject passrate + obs count"),
    FeatureGroup("nn_passrate", ("nn",), "pair", "subject_proxy", "nn_features.py",
                 "target subject's kNN-neighbour passrates (23 cells, nn__*)",
                 also=("benchmark_proxy",),
                 also_patterns=("nn__passrate_benchmark_conditional",)),  # only this cell leaks benchmark
    FeatureGroup("mean_encoded_subject", ("m2_subj",), "pair", "subject_proxy",
                 "mean_encoded_features.py", "m2_subj_* shrunk subject(x cluster/bc) passrates"),
    # (m2_subj_bc is additionally masked under benchmark via mean_encoded_benchmark below)

    # ---- benchmark / condition identity (masked under benchmark dropout) ------------
    FeatureGroup("benchmark", ("benchmark",), "benchmark", "benchmark_proxy", "data.py"),
    FeatureGroup("condition", ("condition", "cond"), "benchmark", "benchmark_proxy",
                 "data.py", "condition + cond__{c} one-hot"),
    FeatureGroup("data_category", ("data_category",), "benchmark", "benchmark_proxy", "data.py"),
    FeatureGroup("benchmark_meta_categorical", ("bench_cat",), "benchmark", "benchmark_proxy",
                 "metadata_features.py", "CSV: topic (vocab id)"),
    FeatureGroup("benchmark_meta_numeric", ("bench_num",), "benchmark", "benchmark_proxy",
                 "metadata_features.py", "CSV: age (+__missing)"),
    FeatureGroup("benchmark_conditions", ("bench_has_conditions", "bench_n_conditions",
                                          "bench_condtype"), "benchmark", "benchmark_proxy",
                 "benchmark_info.csv", "NEW: has_conditions flag + n_conditions + per-condition-"
                 "type indicators parsed from the conditions JSON"),
    FeatureGroup("mean_encoded_benchmark", ("m2_bc", "m2_subj_bc"), "benchmark",
                 "benchmark_proxy", "mean_encoded_features.py",
                 "m2_bc_* benchmark x condition shrunk passrates (m2_subj_bc dual-keyed)"),

    # ===== Kaggle-recommended derivatives (quality_reports/kaggle_feature_recommendations.md) =====
    # Naming: neutral item-geometry uses geo__/clu__ (never masked); label-based neighbour/
    # cluster features use nn__/clu_subj__ (subject_proxy); metadata groupbys split subj/bench.

    # ---- NEUTRAL item geometry (reliability gates; survive subject dropout) ----------
    FeatureGroup("nn_geometry", ("geo",), "item", "neutral_item", "nn_features.py:NEW",
                 "geo__local_density, geo__lid_estimate, geo__dist_gap_1_to_K, "
                 "geo__reciprocal_neighbor_frac — embedding-geometry NN-reliability gates"),
    FeatureGroup("cluster_geometry", ("clu", "clu_id"), "item", "neutral_item",
                 "clustering.py:NEW",
                 "clu__soft_responsibility_top{1..3}, clu__margin_1to2, "
                 "clu__responsibility_entropy, clu__typicality, clu__size_log1p, "
                 "clu__cluster_difficulty_std; clu_id__{coarse,fine} multi-resolution ids"),

    # ---- SUBJECT-keyed label derivatives (masked under subject dropout) --------------
    FeatureGroup("nn_label_derivatives", ("nn",), "pair", "subject_proxy", "nn_features.py:NEW",
                 "multi-K: nn__passrate_mean_K{4,8,32,64}, nn__passrate_K_slope, "
                 "nn__coverage_K_slope; shape: nn__passrate_weighted_var, "
                 "nn__passrate_q25/q50/q75/iqr, nn__frac_neighbors_pass; radius: "
                 "nn__label_entropy_innerK/outerK, nn__agreement_radius_decay; "
                 "rank/calibration: nn__local_difficulty_rank, nn__calibration_residual, "
                 "nn__subjfamily_minus_global_gap (all nn__* already classified via 'nn')"),
    FeatureGroup("cluster_subject", ("clu_subj",), "pair", "subject_proxy",
                 "clustering.py:NEW",
                 "clu_subj__subject_minus_cluster_gap, clu_subj__cluster_obs_count_log1p, "
                 "clu_subj__subject_cluster_affinity, clu_subj__soft_weighted_subject_passrate"),
    FeatureGroup("groupby_subject_metadata", ("grp_subj",), "subject", "subject_proxy",
                 "metadata_features.py:NEW",
                 "grp_subj__{organization,family,macro_family}_passrate_{mean,std} "
                 "(shrunk, OOF) — the playbook's #1 family, std channel is new"),
    FeatureGroup("interactions_subject", ("int", "ratio"), "pair", "subject_proxy",
                 "NEW",
                 "int__subjectmean_x_clusterdiff, int__params_x_release, "
                 "ratio__coverage_over_lid (single trust scalar)"),
    FeatureGroup("counts_subject", ("cnt",), "pair", "subject_proxy", "nn_features.py:NEW",
                 "cnt__neighbor_subject_support (raw labelled-neighbour count behind nn__*), "
                 "cnt__cluster_obs_count_log1p"),

    # ---- BENCHMARK-keyed metadata groupbys (masked under benchmark dropout) ----------
    FeatureGroup("groupby_benchmark_metadata", ("grp_bench",), "benchmark", "benchmark_proxy",
                 "metadata_features.py:NEW",
                 "grp_bench__topic_passrate_mean, grp_bench__age_bin_passrate_mean, "
                 "grp_bench__has_conditions_x_topic (CSV groupby target-encodings, OOF)"),
)


def _patterns_for(cls: str) -> tuple:
    out = []
    for g in CATALOG:
        if g.dropout_class == cls:
            out.extend(g.patterns)
        if cls in g.also:  # dual-axis: contribute the specific secondary patterns
            out.extend(g.also_patterns or g.patterns)
    return tuple(dict.fromkeys(out))  # dedup, order-stable


SUBJECT_PROXY = _patterns_for("subject_proxy")
BENCHMARK_PROXY = _patterns_for("benchmark_proxy")
NEUTRAL_ITEM = _patterns_for("neutral_item")


def all_patterns() -> tuple:
    return tuple(p for g in CATALOG for p in g.patterns)


def group_names() -> tuple:
    """The menu of feature groups the ablation layer chooses from."""
    return tuple(g.name for g in CATALOG)
