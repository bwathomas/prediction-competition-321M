# Feature-suite audit — original `src/` vs `aide/` canon (1:1)

**Date:** 2026-06-04. **Question:** does the `aide/` canon give the AIDE agent access to (and
*safe handling* of — dropout + coverage) the **whole** original feature suite, so it can
make ablation-layer decisions over which features each model gets?

**Headline verdict: NO, not yet.** The funnel is group-agnostic (it can *load* any cached
`.npz` group), but (a) there is **no feature catalog** enumerating the suite, and (b) the
`aide/hygiene/proxy_tree.py` `PROXY_TREE` was written with *placeholder* prefixes
(`feat:pool`, `feat:nn_passrate`, `meta:family`, `feat:judge`, …) that **do not match the
real canonical column names** (`pool__*`, `nn__*`, `subj_cat__*`, `lp_yes`, …). As a
result, with the real columns: dropout would mask **nothing**, and `assert_columns_covered`
would **block every real column** (unlisted ⇒ blocked). Several groups — including the
strongest identity encodings, the IRT latent factors `theta_s` / `u_s_*` — are absent
entirely. Names below verified against the source constants.

Legend: **✅ exact** (real name matches a proxy/neutral entry) · **⚠️ mismatch** (concept
anticipated but the real column name won't match) · **❌ missing**.

## Every feature group (23) and its 1:1 status

| # | Group (source) | Canonical column names | Entity axis | Needed class | aide status |
|---|---|---|---|---|---|
| 1 | Item embedding (`embeddings.py`) | `item_emb__{0..D-1}` dense (D=4096 Qwen-8B) | ITEM | neutral(item) | ❌ no neutral allowlist → coverage blocks |
| 2 | Subject embedding (`embeddings.py`) | `subj_emb__{0..D-1}` dense | SUBJECT | subject-proxy | ⚠️ tree has `feat:subj_emb` (≠ `subj_emb__*`) |
| 3 | Item pool (`item_features.py POOL_FEATURE_NAMES`) | `pool__token_len, pool__char_len, pool__has_latex, pool__has_code, pool__n_questions, pool__n_numbers, pool__is_multiple_choice, pool__n_choices, pool__lang_en` | ITEM (style ⇒ benchmark) | benchmark-proxy | ⚠️ tree has `feat:pool` (≠ `pool__*`) |
| 4 | Centroid distances (`item_features.py`) | `cd__centroid_dist_{0..top_m-1}` (default 10) | ITEM | neutral(item) | ❌ missing |
| 5 | Item k-means cluster (`clustering.py`, `member_features.py`) | `cluster_id`, `cluster__{001..K}` one-hot | ITEM (⇒ benchmark) | benchmark-proxy (decision) | ❌ missing (tree's `feat:subject_cluster` is a different, subject-level concept) |
| 6 | LLM-as-judge (`judge.py JUDGE_CACHE_COLUMNS`) | `lp_yes, lp_no, lp_diff, p_yes_renorm` | PAIR(subj×item) | subject + benchmark proxy | ⚠️ tree has `feat:judge` (≠ `lp_*`) |
| 7–9 | NN passrate (`nn_features.py NN_FEATURE_NAMES`, 23 cells) → `nn__*` | `nn__passrate_mean, nn__passrate_weighted_mean, nn__passrate_std, nn__coverage, nn__top1_label, nn__top1_similarity, nn__mean_similarity, nn__n_labeled_neighbors_log1p` (legacy 8); `nn__effective_neighbor_count, nn__top1_minus_topk_similarity, nn__bootstrap_se_passrate, nn__neighbor_label_entropy, nn__top1_label_match, nn__sim_distribution_skew, nn__distance_to_kth_neighbor` (7); `nn__passrate_subject_conditional, nn__passrate_family_conditional, nn__passrate_macro_family_conditional, nn__passrate_organization_conditional, nn__passrate_benchmark_conditional, nn__neighbor_freshness_diff, nn__n_distinct_subjects_in_neighborhood, nn__cluster_passrate_subject_query` (8) | PAIR(subj×item); some benchmark-conditional | subject-proxy (+ benchmark for `*_benchmark_conditional`) | ⚠️ tree has `feat:nn_passrate` (≠ `nn__*`); benchmark-conditional NN **not** under benchmark |
| 10 | Subject categorical metadata (`metadata_features.py`) | `subj_cat__organization__{id}, subj_cat__family__{id}, subj_cat__macro_family__{id}` | SUBJECT | subject-proxy | ⚠️ tree has `meta:family/organization/macro-family` (≠ `subj_cat__*`) |
| 11 | Subject numeric metadata (`metadata_features.py`) | `subj_num__log_params (+__mask), subj_num__release_date (+__mask)` | SUBJECT | subject-proxy | ⚠️ tree has `meta:parameters/release_date` (≠ `subj_num__*`) |
| 12 | Benchmark categorical metadata (`metadata_features.py`) | `bench_cat__topic__{id}` | BENCHMARK | benchmark-proxy | ❌ missing (`topic` absent) |
| 13 | Benchmark numeric metadata (`metadata_features.py`) | `bench_num__benchmark_age (+__mask)` | BENCHMARK | benchmark-proxy | ❌ missing |
| 14 | Mean-encoded interactions (`mean_encoded_features.py`) | `m2_subj_cluster_mean, m2_subj_cluster_log1p_n, m2_subj_bc_mean, m2_subj_bc_log1p_n, m2_subj_mean, m2_cluster_mean, m2_bc_mean, m2_global_mean` | PAIR / SUBJECT / BENCHMARK | subj-proxy (`*subj*`) + bench-proxy (`*bc*`) | ⚠️ tree has `feat:subject_mean/benchmark_mean/bench_cond_mean` (≠ `m2_*`); `m2_cluster_mean` unclassified |
| 15 | Semantic categories (`semantic_categories.py CATEGORY_NAMES`, 15) | `semcat_id` / `semcat__{name}` one-hot (mcq_stem_text, competition_olympiad_math, …) | ITEM (⇒ benchmark) | benchmark-proxy (decision) | ❌ missing |
| 16–17 | Solver proxy (`solver_proxy.py` core 7 + ext 10) | `self_consistency, answer_entropy, fsd, n_distinct, mean_trace_len, refusal_rate, p_true` + `answer_chars_mean/std, answer_tokens_mean/std, trace_steps_mean/std, trace_tokens_std, boxed_rate, first_line_chars_mean, answer_format_consistency` | ITEM (difficulty) | neutral(item) (⇒ benchmark-corr.) | ❌ missing |
| 18 | Subject mean anchor (`subject_mean.py`) | `subject_mean, subject_obs_count` (`_log1p`) | SUBJECT | subject-proxy | ⚠️ tree has `feat:subject_mean` (≠ `subject_mean`); `subject_obs_count` missing |
| 19 | PCA tail (`pca_tail.py`) | `pca_tail__{0..tail_dim-1}` dense (default 128) | ITEM | neutral(item) | ❌ missing |
| 20–21 | Member feature matrices (`member_features.py`, `member2_features.py`) — **the assembled canonical names** | latent: `theta_s, u_s_{0..k}, u_bc_{0..k}, cross_theta_u_s_{i}, cross_theta_u_bc_{i}`; ids: `subject_idx, bench_condition_id, bc_redacted_mask, cond__{c}` (+ all of pool/cluster/nn/subj_cat/subj_num/bench_* above) | latent θ/u_s/idx = SUBJECT; u_bc/bench_condition_id/cond = BENCHMARK | subject- & benchmark-proxy | ❌ **missing** — including `theta_s`/`u_s_*` (the subject's latent ability = a *direct* identity encoding; the single most important proxy to mask) |
| 22 | Member5 difficulty-kNN (`member5_difficulty_knn.py`) | `member5_knn_passrate` | PAIR(subj×item) | subject-proxy | ❌ missing |
| 23 | Member5 subj×cluster residual (`member5_subject_cluster_residual.py`) | `member5_residual_logit` | PAIR(subj×cluster) | subject-proxy | ❌ missing |

## Tally
- **✅ exact-covered: 0 / 23.**
- **⚠️ anticipated but name-mismatched (won't actually match real columns): 8** (#2,3,6,7–9,10,11,14,18).
- **❌ missing entirely: ~10** (#1,4,5,12,13,15,16–17,19,20–21,22,23) — incl. the **IRT latent factors** and **all item-neutral** groups.

## Root cause & fix
`PROXY_TREE` (Plan 1) used invented prefixes as a sketch; it was never reconciled with the
original repo's real column names. To give the agent safe access to the whole suite:

1. **Create `aide/ensemble/feature_catalog.py`** — one entry per group with: canonical
   column-name pattern, funnel cache-group name, entity axis, and dropout class
   (`neutral_item` / `subject_proxy` / `benchmark_proxy` / `pair_subject_benchmark`).
   This is the menu the agent's ablation layer chooses from, and the source of truth the
   funnel and proxy_tree derive from.
2. **Rewrite `PROXY_TREE` + a `NEUTRAL_ITEM` allowlist from the catalog** using the REAL
   names, so dropout masks every real subject/benchmark proxy (esp. `theta_s`, `u_s_*`,
   `nn__*`, `subj_cat__*`, judge, mean-encodings) and `assert_columns_covered` passes the
   neutral item features (`item_emb__*`, `cd__*`, `pca_tail__*`, pool, solver proxy).
3. **Regression test** that every catalog column is classified (no real column is both
   unmasked-and-uncatalogued), keyed off the real name patterns.
