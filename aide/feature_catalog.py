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
