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

---
## Update — Task 6 driver built + validated on real Colab data (2026-06-05)

Committed (pushed to origin):
- `237c556` foundation: `colab_runtime.py` (run_bg/poll) + `passrate.py` (CsrPassrate,
  proven bit-identical to DensePassrate).
- `cd346d7` `driver.py` — chunkable per-family orchestration (load_embeddings/load_labels/
  build_fold_passrate[reuses src.build_passrate_table]/derive_nn_chunk). Suite: 151 passed.

**Real-data end-to-end validation (Colab2/A100, qwen):** loaded 5.36M labels / 311k items /
906 subjects; slice 300 train + 80 OOF items, real 4096-d embeddings; fold-0 CSR passrate
(global 0.581); `derive_nn_chunk` ran real FAISS, wrote nn_label_derivatives/nn_geometry/
counts_subject shards; `FoldFeatureStore.assemble` read back (1020,15), all finite. The
derive→write→route→load seam works on real data.

### Remaining for the FULL production run (Task 6 completion)
1. **Per-item NN dedup** — search once per unique item (311k) not per row (5.3M); expand label
   gather per subject. (`unique_item_rows` stub present.)
2. Wire **derive_cluster + derive_tabular** chunks into the walk (same pattern as NN;
   codecs already unit-tested + a real-data smoke).
3. **Full walk**: 3 folds × {fold=all invariant groups once, label groups per fold},
   write to Drive `features/qwen/...`, wrapped in `run_bg` with idempotent INDEX resume;
   then replicate to llama + mistral (colab/colab3).
4. Leakage audit on real shards (Plan 4 Task 7).

---
## Update 2 — full walk launched; two scaling findings (2026-06-05)

Driver extended to the full per-family walk (`derive_family`, commits up to `de5c22b`):
geometry at fold=all (per unique item) + per-fold label groups over OOF-of-that-fold rows,
chunked, idempotent, FAISS index built once per fold (`make_faiss_knn`), `include_cluster`
and `features_root`/`max_rows` switches. qwen NN-only validation launched via `run_bg`.

**Scaling findings (flagged; gate the full multi-family run):**
1. **derive_cluster is O(rows × K_fine × members)** — intractable at 311k items × fine=256
   (and the label loop runs even in the geometry pass). Gated behind `include_cluster=False`.
   Fix: precompute per-(subject,cluster) pooled means ONCE per fold (vectorized over the
   CSR + assignment); OOF makes per-row self/alias exclusion a no-op (query items ∉ train),
   so the fast path is exact in-regime. Then re-enable cluster groups.
2. **Embedding load via pyarrow `to_pylist`** of 311k×4096 float64 (~10 GB) is slow and
   RAM-heavy. One-time convert to float16 `.npy` + `np.load(mmap_mode="r")` (Plan §B.2).
3. **NN per-row Python loop** (derive_nn aggregation) is the other throughput cost at 5.3M
   rows; per-unique-item dedup (search once per item, expand labels per subject) is the win.

**Deferred for full Task 6 completion:** cluster vectorization (#1), tabular metadata-groupby
groups (need model_info/benchmark_info join), mean_encoded_subject (global OOF target-encode),
per-item NN dedup (#3), faster embedding load (#2), then replicate to llama + mistral.

---
## Update 3 — qwen full NN-only run LAUNCHED (2026-06-05)

- Live-monitoring the capped validation caught a **chunk-accumulation bug**: writing each
  chunk to the same (group,fold) shard key kept only the first chunk (cache is write-once;
  geometry persisted 40k/311k). Fixed (`292de66`): accumulate chunk blocks, write one
  concatenated shard per (group,fold); regression test `test_concat_blocks_*`. Suite 153.
- Capped validation (max_rows=15k/fold, NN-only) completed end-to-end in **240 s**, 7 shards
  — proves the walk runs at real scale. Dominant costs: full-311k geometry + 3× full-data
  passrate builds (the 15k label rows are cheap; full run's 1.7M rows/fold add ~15–30 min).
- **Full uncapped qwen NN-only run launched** via `run_bg("qwen_full")` at HEAD `292de66`,
  writing to Drive `features/qwen/` (code_version v1). Groups: nn_geometry (fold=all) +
  nn_label_derivatives/counts_subject (per fold). ETA ~30–45 min (Drive I/O + per-row loop).
  Poll: `/content/qwen_full.json`.

### Still open after this run
- Re-enable cluster groups after the O(rows×K×members) → vectorized rewrite.
- Tabular metadata-groupby groups + mean_encoded_subject (global OOF encode).
- Per-item NN dedup + float16 .npy embeddings (throughput); then llama + mistral.

---
## Update 4 — three action items PREPPED (2026-06-05)

All three remaining items built, proven equal to the codec oracles, tested, pushed:
- **A — vectorized cluster** (`cluster_fast.py`, `CsrPassrate.cluster_aggregates`): O(nnz+rows*K)
  replacing O(rows*K*members); exact vs `derive_cluster` in the OOF regime. Wired into driver
  behind `include_cluster`. (`31a30dd`)
- **B — tabular metadata** (`metadata.py`): subject↔model + benchmark joins, `derive_tabular_global`
  (global leave-own-fold-out encode → split per OOF fold). Pure helpers tested; validate subject
  join coverage on Colab. (`fe030e4`)
- **C — throughput** (`nn_fast.py`, `embed_io.py`, `CsrPassrate.gather_pairs`): vectorized NN
  label aggregation (exact vs `derive_nn` in-regime), float16 .npy embeddings (kills slow Drive
  parquet read). Driver hot path now uses nn_fast + cluster_fast. (`c1d78fc`)

Suite 167. **NOTE:** the in-flight `qwen_full` run was launched at `292de66` (pre-nn_fast), so it
uses the SLOW per-row loop (~400 rows/s, ~1-2h ETA). The pushed code would finish the same run in
minutes — restart runtime + relaunch `derive_family(family="qwen", include_cluster=True)` to use
the fast paths (and now cluster groups too).

### To run later (all code ready)
- One-time: `convert_embeddings_to_npy` per family parquet (fast subsequent loads).
- Full run per family with `include_cluster=True` (now scalable) on colab/colab2/colab3.
- `derive_tabular_global` for the metadata/subject-encoding groups (check join coverage).
- Confirm the mistral embedding-cache dir name (FAMILY_SLUG["mistral"] is a guess).

---
## Update 5 — qwen FULL run COMPLETE on fast paths + validated (2026-06-05)

User restarted the Colab runtime; relaunched cleanly at HEAD 0a8fbd4 with the fast paths:
`derive_family(family="qwen", include_cluster=True, code_version="v2")`.

**Result: 17 shards, 2556 s (~43 min)** (vs ~3-4 h projected for the pre-nn_fast slow run).
Geometry+kmeans+embedding-load dominate; the 5.3M label rows now fly via nn_fast/cluster_fast.

v2 shards on Drive `features/qwen/`: nn_geometry, cluster_geometry, centroid_distance,
item_cluster (fold=all) + nn_label_derivatives, counts_subject, cluster_passrate,
cluster_subject (× folds 0/1/2). **Read-back validated:** fold0 labels (1,821,262 × 18) all
finite; geometry (311,130 × 525) all finite; m2_cluster_mean ∈ [0.002, 0.983].

Note: `assemble` aligns WHOLE blocks by row order (load-only, no row subsetting); label
groups share row order per fold and assemble together; geometry is per-item (different grain),
joined by item downstream.

### Task 6 status: qwen COMPLETE. Remaining (all code prepped + tested):
- llama + mistral: `derive_family(family="llama"/"mistral", include_cluster=True)` (confirm
  mistral FAMILY_SLUG dir on Drive).
- Tabular metadata groups: `derive_tabular_global` (check subject-join coverage).
- One-time `.npy` conversion to kill the ~5 min embedding load on repeat runs.
