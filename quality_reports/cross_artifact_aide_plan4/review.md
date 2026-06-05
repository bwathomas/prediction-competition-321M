# Plan 4 (aide/features) — independent fresh-context code review (2026-06-04)

Reviewer: fresh-context subagent, never saw the drafting. 38 new tests passed at review
time; findings live mostly in paths the brute-force-injected tests don't exercise.

## CRITICAL
1. FAISS `-1` padding wraps to `index_keys[-1]` (phantom neighbour → label leak/garbage).
   derive_nn neighbour loop. Local tests inject brute force so never see `-1`.
2. Self-exclusion by key-string only → an embedding-aliased duplicate of the query item
   (same content, different key) leaks its own label into nn__*/m2_cluster.

## MAJOR
3. `search_buffer` silently shrinks K below maxK after self-removal (corrupts multi-K
   means / K_slope) with no error.
4. Cache key collision: `_slug` collapses distinct `code_version` (e.g. "a/b" vs "a-b")
   to one stem → derive-once serves a stale shard; family/group interpolated unslugged
   into the path (a "/" escapes the dir).
5. `target_encode_oof` std channel: unseen-in-other-folds and zero-variance both return
   0.0 (conflated); dead `m` arg on the std call.

## MINOR
6. derive_cluster OOF correctness is caller-trust; no internal train/oof disjointness guard.
7. `pooled_mean` returns 0.0 for an all-unobserved cluster (= "everyone fails"); inflates gap.
8. `assert_columns_covered` is opt-in (default False) — Plan §E says every assembled matrix.
9. int__/ratio__ silently vanish on a parent-name mismatch (cluster emits `m2_cluster_mean`,
   interaction looks up `cluster_difficulty`).
10. Tests don't prove the alias/`-1` leakage claims (brute-force-only injection hides them).

## CORRECT (verified): atomic write/tmp-cleanup/INDEX ordering; allow_pickle False;
   29-group fold-routing partition disjoint+exhaustive; target-encode mean OOF subtraction.

Verdict: SHIP-WITH-FIXES. Top priority: FAISS `-1` guard + alias self-exclusion.

---
## Resolution (2026-06-05) — all findings addressed, regression test per finding

| # | Sev | Fix | Regression test |
|---|-----|-----|-----------------|
| 1 | CRIT | `derive_nn`: skip `ci < 0` FAISS pads before key lookup | `test_faiss_minus_one_padding_does_not_leak_last_item` |
| 2 | CRIT | self-exclude by key OR cosine ≥ 1−alias_eps (nn + cluster) | `test_nn_alias_*`, `test_cluster_alias_*` |
| 3 | MAJ | raise if <maxK survive without full index scan; clip if index genuinely small | `test_under_retrieval_raises_*`, `test_small_index_clips_*` |
| 4 | MAJ | stem appends `_short_hash(code_version)`; `_safe_component` validates family/group | `test_code_version_slug_collision_*`, `test_unsafe_family_or_group_raises` |
| 5 | MAJ | std unseen→global_std (≠0); dropped dead `m` arg | `test_std_unseen_key_uses_global_std_not_zero` |
| 6 | min | optional `oof_item_keys` disjointness assert in derive_cluster | `test_cluster_train_oof_overlap_raises` |
| 7 | min | `pooled_mean(default=)`; caller passes self-excluded prior (NOT global incl. self) | `test_cluster_difficulty_and_subject_gap_are_oof` (caught a re-introduced leak) |
| 8 | min | `store.assemble` coverage probe defaults ON | `test_store_coverage_probe_runs_without_explicit_flag` |
| 9 | min | partial interaction parents raise | `test_partial_interaction_parents_raise` |
| 10| min | brute-force-only tests supplemented with `-1`/alias-injecting knn_fn | (the rows above) |

Note: fixing #7 with a naive global-mean fallback re-introduced a self-leak (global mean
includes the query item's own column); the pre-existing OOF test caught it immediately,
and the fix now threads a self-excluded prior. Suite: **137 passed**.
Verdict upgraded: SHIP-WITH-FIXES → **SHIP** (local scope; Task 6 Colab driver pending).
