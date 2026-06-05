# Session — 2026-06-05 — Plan 4 feature-pipeline codecs executed (tasks 1–5 + review)

**Repo:** `prediction-competition-321M`  ·  **Branch:** `clean/aide-stacked-ensemble`
**Base commit:** `58ca1ab` (Plan 4 doc).  **Working tree:** new files unstaged (not committed).
**Suite:** `python3 -m pytest aide -q` → **137 passed** (was 87; +50 new).

## What was done
Executed Plan 4 (`docs/superpowers/plans/2026-06-04-feature-pipeline-dataflow.md`) tasks 1–5
under the **"Colab-only codecs, contracts local"** decision (heavy libs FAISS/sklearn/polars/
pyarrow stay Colab-side via lazy/injectable kernels; OOF-/leakage-critical logic is numpy and
locally TDD'd). New module `aide/features/`:

- `cache.py` — fold-aware derive-once shard cache (ShardKey, atomic tmp→rename, idempotent
  skip, INDEX.json, content-hash meta). NpzBackend local default; ParquetBackend = Colab.
- `derive_nn.py` — kNN codec: `nn__*` (subject_proxy), `geo__*` (neutral, fold-invariant),
  `cnt__*`. `bruteforce_knn` oracle + `default_knn` (FAISS-if-present). `DensePassrate`.
- `derive_cluster.py` — multi-K kmeans codec: `cluster__/cd__/clu__/clu_id__` (neutral),
  `m2_cluster` + `clu_subj__*` (OOF). Centroids injected (`fit_multi_kmeans` = sklearn/Lloyd).
- `derive_tabular.py` — `target_encode_oof` (leave-own-fold-out shrunk; mean + std),
  `grp_subj__/grp_bench__`, `m2_subj` (two smoothings), `int__/ratio__`.
- `store.py` — `FoldFeatureStore`: single writer + load-only `assemble` with **fold routing**
  (FOLD_INVARIANT_GROUPS=all-shard vs LABEL_DERIVED_GROUPS=outer-fold; both explicit so a
  forgotten group raises). Coverage probe defaults ON. Does NOT touch `harness/funnel.py`.

## Independent review (the standard fresh-context loop)
Subagent review (`quality_reports/cross_artifact_aide_plan4/review.md`) found 2 CRITICAL
leaks invisible to brute-force-injected tests (FAISS `-1` wrap to `index_keys[-1]`; alias
self-exclusion by key only) + 3 MAJOR + minors. **All fixed, one regression test per finding**
(`tests/test_plan4_review_regressions.py`). The pooled-mean fallback fix re-leaked and the
existing OOF test caught it — fixed with a self-excluded prior.

## Next actions
1. **Plan 4 Task 6** — Colab per-family driver (walk the §D DAG, `run_bg`, write shards,
   free, idempotent resume from INDEX). Colab-bound; needs the bridge + A100.
2. **Plan 4 Task 7** (leakage audit on real shards) on Colab, then **Plan 5** (single-agent
   end-to-end on colab2/Qwen → aideml orchestrator → replicate to 3 agents).
3. Decide whether to commit tasks 1–5 now (not yet committed; user has not asked).

## Notes / decisions
- Cache container is `.npz` locally (numpy-native, consistent with funnel) with the §A
  fold-aware key/path/meta/INDEX; parquet is an optional Colab backend (deviation from §A
  literal wording, approved).
- Interaction parent wiring: cluster difficulty column is `m2_cluster_mean` (driver must map
  it to the `cluster_difficulty` parent name; partial wiring now raises).
