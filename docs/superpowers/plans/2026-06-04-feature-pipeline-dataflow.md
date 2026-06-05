# Feature Pipeline & Data-Flow Design (memory-efficient, derive-once, Drive-cached)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Colab-bound (A100); local steps are the codecs/contract + their unit tests.

**Goal:** Derive every catalog feature group **once per (outer-fold × embedding-family)**, persist permanently to Drive, and feed the 2-layer ensemble through a funnel that assembles → uses → **frees** with minimal RAM/VRAM churn. Implements the data flow behind `aide/feature_catalog.py` + the Kaggle derivatives, under the hygiene splits.

**Architecture (one line):** `embeddings (memmap) → per-fold derivation jobs (FAISS/cuPy/polars, streamed) → per-group cache shards on Drive (parquet/npz, content-hashed) → funnel assembles only the ablation's groups as one float32 matrix → model → drop refs`.

**Depends on:** Plans 1–3 (`aide/hygiene`, `aide/harness`, `aide/ensemble`), `aide/feature_catalog.py`.

---

## A. The cache key — derive-once, never twice

A feature shard is identified by
`key = (embedding_family, feature_group, outer_fold, split_seed, n_folds, code_version)`.
Path: `Drive/Prediction-Competition-321M/features/{embedding_family}/{feature_group}/fold{f}_seed{s}.parquet`
(+ a sibling `.meta.json` with the content hash of inputs + git rev). **Permanent**: a
shard whose key exists is loaded, never recomputed (mirrors `drive_cache.resolve_cache`).
Derivation is **idempotent + atomic** (`.tmp` dir → rename) so a Colab timeout never
half-writes. One manifest `features/INDEX.parquet` lists all present shards (the agent's
"what's already derived" view).

**Per-fold semantics (correctness, not just speed).** Every *label-derived* group
(`nn__*`, `m2_*`, `grp_*`, `clu_subj__*`, `cnt__*`) is computed for outer fold `f` using
ONLY fold-`f` training items (the FAISS index, passrate CSR, k-means, and group means are
all fit on `train_item_keys(f)`), then evaluated for *all* rows — exactly the
`aide/hygiene/splits.outer_folds` partition, so OOF discipline is baked into the cache key.
Layer-2 needs the inner (recursive) variant: shards also keyed by `inner_fold` when
requested. Pure-geometry neutral groups (`geo__`, `clu__`, item embeddings) are
**fold-invariant** → derived once per embedding family (fold tag = `all`), halving work.

## B. Memory discipline — cache, assemble, clear, pass

The A100 is high-RAM but the embedding matrices are large (≈300k items × 4096 × fp16 ≈
2.4 GB *each*, ×2 for subjects, ×3 families). Rules:

1. **Never hold all families at once.** Process one embedding family end-to-end, persist its
   shards, free, move to the next. Three agents = three families on three A100s, so each
   process only ever touches *its own* family.
2. **Memmap, don't load.** Embeddings live on Drive as `float16` `.npy`; open with
   `np.load(..., mmap_mode="r")` / `np.memmap`. FAISS reads them in chunks to build the
   index; we never materialize the full fp32 matrix in RAM.
3. **Stream derivation in row-blocks.** Compute NN/cluster/geometry features in batches of
   ~50k query rows: query FAISS for the block, pull neighbour passrates from the CSR slice,
   compute the block's columns on GPU (cuPy/torch), write the block to the parquet shard,
   `del` the block + `torch.cuda.empty_cache()`. Peak VRAM = one block, not the dataset.
4. **Assemble lazily at train time.** The funnel loads ONLY the groups in the model's
   ablation (`FeatureStore.assemble(groups)`), as `float16`→`float32` views, `np.concatenate`
   once into a contiguous matrix, hand to the model, then **drop the reference** so it's GC'd
   before the next member trains. Members train sequentially per fold; we never hold two
   members' full feature matrices simultaneously.
5. **Categorical/dense split.** One-hots (`cluster__*`, `subj_cat__*`, `cond__*`) are stored
   as compact int id columns + expanded on demand (or kept sparse CSR `.npz`), not as dense
   fp32 on disk.

## C. Efficient packages (no slow per-row Python)

| Job | Package | Why |
|---|---|---|
| kNN over embeddings | **FAISS** (`IndexFlatIP` on GPU; `IVF`/`HNSW` if needed) | the workhorse; one batched query yields all `nn__*` + `geo__` |
| batched distance / density / LID / soft-responsibility | **cuPy / torch** on A100 | vectorized over blocks; no Python loops |
| passrate matrix + neighbour gather | **scipy.sparse CSR** (offline) shipped as `(indptr,indices,data).npz` | O(nnz) gather, tiny memory |
| metadata groupby target-encodings (`grp_*`, `m2_*`) | **polars** (or cuDF on GPU) | columnar, fast OOF groupby-agg over the long table |
| multi-resolution k-means | **scikit-learn `MiniBatchKMeans`** (offline) | streaming fit on memmapped embeddings |
| shard IO | **pyarrow parquet** (dense) + **npz** (sparse) | columnar, compressed, partial-column reads |

(Training-only environment → these heavy libs are allowed here; the live submission's
`RUNTIME_ENV.md` numpy-only rule is a *different, later* phase.)

## D. Derivation DAG (per embedding family, per fold)

```
embeddings.npy(fp16, memmap) ─┬─ FAISS index(train items, fold f) ─┬─ nn__* (label, CSR gather)   [subject_proxy]
                              │                                    ├─ geo__*  (geometry only)      [neutral, fold=all]
                              │                                    └─ cnt__*  (neighbour support)  [subject_proxy]
                              ├─ MiniBatchKMeans(train, multi-K) ──┬─ cluster__*, cd__* one-hot/dist [neutral, fold=all]
                              │                                    ├─ clu__*  (soft geom)          [neutral, fold=all]
                              │                                    └─ clu_subj__* (subject×cluster)[subject_proxy]
passrate CSR(train, fold f) ──┴─ m2_*, grp_subj__*, grp_bench__* (OOF target-enc, polars)         [subj/bench_proxy]
metadata CSV (polars) ────────── subj_cat/num__*, bench_cat/num__*, bench_conditions, int__*       [proxy by axis]
item text ───────────────────── pool__*, semcat__*                                                 [neutral]
```
Each leaf writes one shard. `int__`/`ratio__` are computed last (cheap arithmetic over
already-cached parent columns). m2 target-encodings emit **two smoothing strengths** (high/low
`m`) as separate columns for ensemble diversity.

## E. Funnel integration (extends Plan 2 `FeatureStore`)

- `FeatureStore(root=features/{family})` with `assemble(groups, fold, row_ids)` →
  reads each group's fold-shard (or `fold=all` for neutral), aligns by `row_ids` (assert),
  concatenates. **Load-only**: a missing shard raises `CacheMissError` naming the
  `(family, group, fold)` to derive — never recompute inside a training run.
- A `derive(family, group, fold)` entry point (the ONLY writer) is what the offline
  derivation cells call; `predict`/`evaluate` paths never write.
- `assert_columns_covered(cols, NEUTRAL_ITEM)` runs on every assembled matrix so a stray
  unclassified column is caught before a model sees it.

## F. Tasks (build order)
1. `aide/features/cache.py` — shard key/path, atomic write, `INDEX.parquet`, content-hash meta. + unit tests (tmp dir).
2. `aide/features/derive_nn.py` — FAISS index + block-streamed `nn__*`/`geo__*`/`cnt__*` from a memmapped-embedding + CSR fixture. Unit-test on a tiny synthetic fixture (correct shapes, OOF: query item's own labels never used).
3. `aide/features/derive_cluster.py` — MiniBatchKMeans multi-K + `cluster__*`/`cd__*`/`clu__*`/`clu_subj__*`. Unit-test.
4. `aide/features/derive_tabular.py` — polars OOF target-encodings `m2_*`/`grp_*` (two smoothings) + metadata + `int__`. Unit-test OOF-correctness (fold f encoded from other folds only).
5. Extend `FeatureStore` with fold-aware `assemble` + `derive` writer + memmap/free discipline. Unit-test.
6. Colab cell: per-family driver that walks the DAG, `run_bg`, writes shards, frees; idempotent resume from `INDEX.parquet`.
7. Code-review + leakage audit (OOF probes on every label-derived shard).

## G. Self-review vs request
- "wire every recommendation into tree + data flow" → catalog done (commit 5952d57); data flow = this pipeline (§D/§E). ✓
- "avoid slow memory movement; cache, assemble, clear, pass intelligently" → §B (memmap, block-stream, lazy assemble, drop refs). ✓
- "derive per fold per embedding family once, save to Drive permanently" → §A cache key + idempotent atomic shards. ✓
- "extract with efficient packages" → §C. ✓
