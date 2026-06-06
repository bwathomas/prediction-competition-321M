# NN23 patch notes — emit 23-dim NN features, feed legacy head `[:8]`

**Goal.** The sent `nemotron_trc_v5/model.py` and `LGAI_fixed/model.py` compute an
**8-dim** NN feature vector and feed it to their trained residual heads. New
ensemble members need the **23-dim** vector from `src/nn_features.py`
(`NN_FEATURE_DIM=23`, cells 0..7 == legacy 8). This patch makes the submodels
EMIT 23-dim while feeding `vec[:8]` to the legacy head — back-compat is a pure
slice, so cells 0..7 are bit-identical to what the head was trained on.

**Deliverable used by this patch:** `scripts/ship/nn23_runtime.py`
- `compute_nn_features_23(cache, item_emb, subject_id, *, cond_ctx, query_benchmark_id, query_benchmark_age, query_cluster_id, k, ...)` → `np.ndarray` shape `(23,)`.
  Verified byte-identical (maxabsdiff=0.0) to `src/nn_features.py::_aggregate_nn_features`
  for both `cond_inputs=None` and full-`cond_inputs` paths.
- `ConditionalContextRuntime.maybe_load(ctx_dir)` → loader for the
  `DR/artifacts/nn_features/` conditional bag (scipy-optional; returns `None`/degrades
  to fallback cells 15..22 if scipy or files are missing).

The two submodels are **byte-identical** in their NN plumbing (verified by
`diff` of the `_TrainingItemCache.nearest` / `compute_nn_features` block and of
`_get_nn_features`). The qwen8b `submodel.py` shares the same block (offset
only). So the SAME 4 edits apply to each; only line numbers differ.

---

## Quoted current contracts (from `src/nn_features.py`)

### `_aggregate_nn_features` (23-dim) — the canonical aggregator

```
def _aggregate_nn_features(
    neighbor_passrates: np.ndarray,    # [B, K] mean labels (NaN where missing)
    neighbor_masks: np.ndarray,        # [B, K] 1 where labeled, 0 otherwise
    similarities: np.ndarray,          # [B, K]
    *,
    fallback_value: float,
    top1_missing_sentinel: float,
    cond_inputs: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pure NN aggregation. Returns [B, NN_FEATURE_DIM] float32.
      * cells 0..7 are the legacy aggregators
      * cells 8..14 are the self-derived additions ...
      * cells 15..22 are the conditional / context features ...
        When ``cond_inputs`` is ``None`` (or any individual key is absent), the
        corresponding cell falls back to ``fallback_value``."""
```

Output column order (locked, from `NN_FEATURE_NAMES`):
```
 0 passrate_mean                 8 effective_neighbor_count       15 passrate_subject_conditional
 1 passrate_weighted_mean        9 top1_minus_topk_similarity     16 passrate_family_conditional
 2 passrate_std                 10 bootstrap_se_passrate          17 passrate_macro_family_conditional
 3 coverage                     11 neighbor_label_entropy         18 passrate_organization_conditional
 4 top1_label                   12 top1_label_match               19 passrate_benchmark_conditional
 5 top1_similarity              13 sim_distribution_skew          20 neighbor_freshness_diff
 6 mean_similarity              14 distance_to_kth_neighbor       21 n_distinct_subjects_in_neighborhood
 7 n_labeled_neighbors_log1p                                      22 cluster_passrate_subject_query
```

The `cond_inputs` keys consumed by cells 15..22 (exact names, from the
aggregator and `_resolve_conditional_inputs`):
`{subject,family,macro_family,organization,bench_match}_{passrates,masks,redact}`,
`neighbor_freshness_diff`,`freshness_redact`,`distinct_subj_per_neighbor`,
`cluster_passrate_subject_query`,`cluster_redact`.

### `build_conditional_passrate_context` — context builder (its `.save()` defines the on-disk bag)

```
def build_conditional_passrate_context(
    *,
    train_df: pd.DataFrame,                       # subject_key, item_key, label
    item_index_map: Mapping[str, int],
    subject_index_map: Mapping[str, int],
    subject_to_family_id: np.ndarray,             # int [n_subjects]; idx 0 = UNK -> MISSING_TRAIT_ID(0)
    subject_to_macro_family_id: np.ndarray,
    subject_to_organization_id: np.ndarray,
    item_benchmark_id: np.ndarray,                # int [n_items]; -1 = unknown
    item_benchmark_age: np.ndarray,               # float [n_items]; NaN = unknown
    item_cluster_id: np.ndarray,                  # int [n_items]; -1 = unclustered
    n_families: int, n_macro_families: int, n_organizations: int, n_clusters: int,
) -> ConditionalPassrateContext
```

`ConditionalPassrateContext.save(out_dir)` writes the bag the runtime loader reads:
`subject_passrate.npz`/`_mask.npz`, `family_*`, `macro_family_*`, `organization_*`,
`subject_to_{family,macro_family,organization}_id.npy`,
`item_{benchmark_id,benchmark_age,distinct_subj_count,global_passrate,global_passrate_mask,cluster_id}.npy`,
`cluster_subject_passrate.npz`/`_mask.npz`, and `conditional_meta.json`
(carries `feature_dim:23` + cardinalities — the loader refuses a stale `feature_dim`).

> **Sidecar the exporter must also drop into `ctx_dir`:** `benchmark_to_id.json`
> mapping raw benchmark string → the integer space used in `item_benchmark_id.npy`.
> Without it, cell 19 (`passrate_benchmark_conditional`) redacts to fallback (still
> valid 23-dim, just one cell dark). Use the same benchmark vocabulary that built
> `item_benchmark_id`.

---

## The 4 edits (apply to `nemotron_trc_v5/model.py`, `LGAI_fixed/model.py`; same for `qwen8b/submodel.py`)

Line numbers below are for **`nemotron_trc_v5/model.py`** (2680 lines).
`LGAI_fixed/model.py` is +200 offset; quoted anchor strings are identical, so
match by string, not by number.

### Edit 1 — vendor the runtime module + load the context once at import

Copy `scripts/ship/nn23_runtime.py` into the bundle next to `model.py` (it is
numpy+stdlib only; scipy lazy/guarded), then near the other top-level loads
(after `TRAINING_CACHE` is constructed; nemotron ~ where `TRAINING_CACHE` and
`_MODEL_CFG` are set, search `TRAINING_CACHE =`) add:

```python
from nn23_runtime import compute_nn_features_23, ConditionalContextRuntime
# Conditional context ships inside the bundle's cache dir (same dir the
# subject_passrate.npz / nn_features_config.json already live in).
COND_CTX = ConditionalContextRuntime.maybe_load(TRAINING_CACHE_DIR)  # -> ctx or None
```

`TRAINING_CACHE_DIR` already exists at module top (`= HERE / "cache"`). Put the
conditional bag there at export time (it is the same dir as `subject_passrate.npz`).

### Edit 2 — bump the declared NN dim to 23

**`NN_FEATURE_DIM`** (nemotron line **150**, LGAI **171**):
```python
NN_FEATURE_DIM: int = int(NN_META.get("feature_dim", 8))
```
Leave the line as-is BUT set `runtime_meta.json -> nn_features.feature_dim = 23`
**and** `runtime_meta.json -> model_cfg.nn_feature_dim = 8`. Rationale: the trained
HEAD consumes `nn_feature_dim` (8) — that MUST stay 8 so the head's input
LayerNorm/Linear shapes match the checkpoint. `NN_FEATURE_DIM` (the schema dim)
becomes 23 only to size the EMITTED vector. If you cannot re-export
`runtime_meta.json`, hard-set in code instead:
```python
NN_FEATURE_DIM: int = 23          # emitted width
```
and keep `_MODEL_CFG["nn_feature_dim"]` at 8 (it already is, since these heads
trained on 8 — verify with `runtime_meta.json model_cfg.nn_feature_dim`).

> ⚠ Do NOT change `model_cfg.nn_feature_dim`. The legacy head's first Linear is
> `(... + 8) -> hidden`. Feeding it 23 would shape-mismatch the checkpoint.

### Edit 3 — emit 23 in `_TrainingItemCache.compute_nn_features` (or call the new fn)

The cleanest change: in **`_get_nn_features`** (nemotron ~2469-2506, anchor
`vec = TRAINING_CACHE.compute_nn_features(`), replace the call body so it returns
the **23-dim** vector. Current code (nemotron 2491-2496):

```python
    try:
        vec = TRAINING_CACHE.compute_nn_features(
            query_embed=item_emb,
            subject_id=int(subject_id),
            k=NN_RUNTIME_K,
        )
```

Replace with:

```python
    try:
        vec = compute_nn_features_23(
            TRAINING_CACHE,
            item_emb,
            int(subject_id),
            cond_ctx=COND_CTX,
            query_benchmark_id=(
                COND_CTX.benchmark_id_for(_NN23_BENCH)
                if COND_CTX is not None else -1
            ),
            query_benchmark_age=float("nan"),     # age unknown at predict time -> cell 20 redacts
            query_cluster_id=int(_NN23_CLUSTER_ID),
            k=NN_RUNTIME_K,
            fallback_value=NN_FALLBACK_VALUE,
            top1_missing_sentinel=NN_TOP1_MISSING_SENTINEL,
        )
```

`_get_nn_features` only receives `(subject_id, item_emb, item_cache_key)` today, so
the benchmark string and cluster id must be threaded in. Two options:

- **Minimal (recommended for ship):** add two params to `_get_nn_features`
  `def _get_nn_features(subject_id, item_emb, item_cache_key, benchmark="", cluster_id=-1)`
  and pass them from `_predict_uncalibrated` (which already has both — see Edit 4).
  Inside, set `_NN23_BENCH = benchmark`, `_NN23_CLUSTER_ID = cluster_id`.
- **Zero-signature-change fallback:** leave the extra params off and pass
  `query_benchmark_id=-1`, `query_cluster_id=-1`. Cells 19 and 22 then redact to
  fallback; cells 15..18, 20(redacts on NaN age), 21 still populate. You still get a
  valid 23-dim vector; you just forgo two of the conditional cells.

Then **keep the existing `nn_feature_dim` truncation guard** that already follows
(nemotron 2500-2504) — but change its target from `dim` (8) to the value the HEAD
wants. Important: this guard currently pads/truncates `vec` to
`dim = _MODEL_CFG.get("nn_feature_dim", NN_FEATURE_DIM)`. Since we now want
`_get_nn_features` to RETURN 23 (for new members) but FEED 8 to the head, do the
slice at the call site instead (Edit 4), and make `_get_nn_features` return the
full 23:

```python
    if vec.size != NN_FEATURE_DIM:           # NN_FEATURE_DIM == 23 now
        out = np.zeros(NN_FEATURE_DIM, dtype=np.float32)
        n = min(NN_FEATURE_DIM, int(vec.size))
        out[:n] = vec[:n]
        vec = out
    cache[key] = vec.astype(np.float32, copy=False)
    return cache[key]                         # length 23
```

### Edit 4 — feed `[:8]` to the legacy head; expose 23 to new members

In **`_predict_uncalibrated`** (nemotron ~2548-2611), the head is called with the
NN tensor `nf`. Current (nemotron 2579, 2598-2602, 2607):

```python
    nn_vec = _get_nn_features(subject_nn_id, item_emb, item_cache_key)
    ...
    nf = (
        torch.from_numpy(nn_vec).to(_DEVICE).unsqueeze(0)
        if nn_vec.size > 0
        else None
    )
    ...
        logit = _MODEL(s, bc, ie, se, pf, ci, jf, nf)
```

Change to (pass benchmark + cluster_id through, slice for the head, keep full for members):

```python
    nn_vec23 = _get_nn_features(
        subject_nn_id, item_emb, item_cache_key,
        benchmark=benchmark, cluster_id=int(cluster_id),
    )                                          # length 23 (or 0 if disabled)
    legacy_head_dim = int(_MODEL_CFG.get("nn_feature_dim", 8))   # == 8
    nn_vec_head = nn_vec23[:legacy_head_dim] if nn_vec23.size >= legacy_head_dim else nn_vec23
    ...
    nf = (
        torch.from_numpy(np.ascontiguousarray(nn_vec_head)).to(_DEVICE).unsqueeze(0)
        if nn_vec_head.size > 0
        else None
    )
    ...
        logit = _MODEL(s, bc, ie, se, pf, ci, jf, nf)     # head still gets 8
```

`cluster_id` is already computed earlier in `_predict_uncalibrated`
(`cluster_id = _assign_cluster_id(item_emb)`), so it is in scope.

**Exposing the full 23 to NEW members:** new members are loaded by the ensemble
combiner as separate submodels. Each new member runs its OWN `model.py` whose
`runtime_meta.json` declares `nn_feature_dim=23`; its `_predict_uncalibrated`
omits the `[:8]` slice (feeds `nn_vec23` straight to its head). The legacy
nemotron/LGAI heads stay at 8 via the slice above. If instead a new member needs
the 23-dim vector computed by the *nemotron embedding* specifically, have it import
`compute_nn_features_23` with the nemotron `TRAINING_CACHE` + `COND_CTX` and consume
the full return. The slice is the only thing that differs between "legacy head" and
"new member" consumption — same single computation.

---

## Why back-compat is exact (and how it's verified)

`compute_nn_features_23(...)[:8]` reuses the SAME `cache.nearest()` neighbor query
and the SAME subject-passrate sparse lookup the legacy
`cache.compute_nn_features(...)` used (the code in `nn23_runtime.py` is lifted
verbatim from the submodel's own block), then calls the 23-dim aggregator whose
cells 0..7 are the legacy 8 recipe unchanged. Verified:

- `scripts/ship/nn23_runtime.py` self-test: cells 15..22 redact to `fallback_value`
  with `cond_inputs=None`; cells 0/6/7 match the legacy recipe.
- AST-extracted comparison vs `src/nn_features.py::_aggregate_nn_features`:
  `np.array_equal == True`, `maxabsdiff == 0.0` for BOTH `cond_inputs=None` and a
  fully-populated `cond_inputs`.

The zero-guard is also preserved: when the passrate cache is missing / subject is
unseen / neighbor set is empty, `compute_nn_features_23` returns `np.zeros(23)`, so
`[:8]` is all-zeros — exactly the "unseen subject → zero NN vector" contract the
legacy head trained against.

## Risks / things the human must verify

1. **`runtime_meta.json` split:** `nn_features.feature_dim` → 23 (emit width) while
   `model_cfg.nn_feature_dim` stays 8 (head input width). Confirm the sent
   nemotron/LGAI checkpoints really trained on 8 (`model_cfg.nn_feature_dim==8`) —
   the sidecar I read shows `nn_features.enabled=false, feature_dim=8`. If
   `use_nn_features` is actually `false`/absent in these heads, the head ignores NN
   entirely and the 23-dim emit only matters for new members — harmless but verify.
2. **Conditional bag presence:** `DR/artifacts/nn_features/` must be copied into the
   bundle's `cache/` dir (same dir as `subject_passrate.npz`) AND its
   `conditional_meta.json feature_dim` must be 23, else cells 15..22 silently redact.
   ZIP budget ≤ 65 MB — the family/org/cluster `.npz` are sparse but check size.
3. **`benchmark_to_id.json` sidecar** must be exported with the SAME benchmark
   vocabulary used to build `item_benchmark_id.npy`, or cell 19 redacts. Subject ids
   used by `cond_ctx.resolve_single` must be in the SAME index space as
   `subject_passrate.npz` rows (the submodel already resolves `subject_nn_id` via
   `TRAINING_CACHE.subject_key_to_id`, which matches).
4. **scipy at runtime:** the conditional loader needs `scipy.sparse`. The HARD import
   whitelist does NOT include scipy. If scipy is unavailable in the Codabench image,
   `ConditionalContextRuntime.maybe_load` returns `None` and cells 15..22 go to
   fallback — cells 0..14 (incl. the legacy 8) are unaffected (numpy-only). If you
   need cells 15..22 to survive without scipy, pre-densify the CSR rows to `.npy` at
   export and add a numpy-only reader; not done here to keep the bag small.
5. **K alignment:** `compute_nn_features_23` uses `k=NN_RUNTIME_K`; the conditional
   neighbor lookups use the same `valid_idx`, so cells 15..21 see the identical
   neighbor set as cells 0..7. Good. But if `NN_RUNTIME_K` differs from the
   training-time K, cells 8/14/21 (count/min-sim/distinct) shift — same caveat the
   legacy 8-dim path already carried.
6. **Threading benchmark/cluster into `_get_nn_features`:** Edit 3/4 add two kwargs.
   If you take the zero-change fallback, cells 19 & 22 redact — quantify the loss
   before deciding (likely small; both are conditional cells with built-in fallback).
