# Adversarial Code Review — `aide/hygiene/` (data-hygiene / leakage-prevention core)

**Branch:** `clean/aide-stacked-ensemble`
**Reviewer scope:** `aide/hygiene/{manifest,splits,proxy_tree,dropout,probes,__init__}.py` + `aide/hygiene/tests/*`
**Reference intent:** spec §4 (data hygiene) and §5.1.
**Test status:** 23/23 pass locally — but the passing suite does NOT cover the failure modes below; several findings are reproduced by code I ran against the package.

The architecture is sound and the *intent* (item-uniform folds, atomic proxy masking, byte-identical cross-agent manifest) is correctly identified. The defects are in the gaps between components and in a handful of correctness/over-reach bugs an adversarial optimizer will find and exploit.

---

## CRITICAL

### C1. `assert_no_proxy_leak` contradicts `apply_proxy_dropout` — the tripwire fires on correct data and is unusable in production
**Files:** `probes.py:25–33` (probe) vs `dropout.py:21–48` (producer)

`apply_proxy_dropout` only zeros proxy columns for **rows belonging to a dropped entity** (`X[np.ix_(rows, idx)] = 0.0`, where `rows` is the dropped-entity mask). With any `subject_rate < 1.0` — i.e. the entire point of dropout — some subjects are NOT dropped and their proxy columns remain non-zero.

`assert_no_proxy_leak` checks `np.any(np.abs(X[:, j]) > atol)` over **ALL rows**. So it raises `AssertionError("proxy leak…")` the moment a single non-dropped subject exists. The probe and the producer are mutually inconsistent.

Reproduced (real package code, `subject_rate=0.5`, seed 0, one of two subjects dropped):
```
dropped={'s2'}
assert_no_proxy_leak(...) -> AssertionError: proxy leak: column 'meta:family' survived dropout of ['subject']
```
The spec (§5.x) says *"Any harness reporting an NLL should call these first; a failure aborts the score."* As written, this tripwire aborts **every** legitimate trial that uses partial-rate dropout. Either it gets disabled in practice (removing the leakage guard entirely — the dangerous outcome), or it blocks all scoring.

**Fix:** The probe must be row-aware and match the dropout contract. Either (a) pass the dropped-entity row mask and assert zero only on those rows:
```python
def assert_no_proxy_leak(X, feature_columns, dropped_nodes, dropped_rows, *, atol=0.0):
    masked = all_masked_columns(dropped_nodes, list(feature_columns))
    sub = X[np.asarray(dropped_rows)][:, [cols.index(c) for c in masked]]
    if np.any(np.abs(sub) > atol): raise AssertionError(...)
```
or (b) redefine the probe for the *fully-dropped* (`rate=1.0`) unit-test scenario only and document that it is NOT a production tripwire. Given the spec language ("call these first"), option (a) is required. Also have `apply_proxy_dropout` **return the dropped row masks** so the probe can be wired without recomputation.

### C2. Proxy-column matching uses bare `str.startswith`, silently over-masking unrelated columns (and the inverse risk on rename)
**File:** `proxy_tree.py:50` — `if c == proxy or c.startswith(proxy)`

The prefix rule has no delimiter discipline. Any real column whose name merely *starts with* a node string is swept in. Reproduced against the real code:
```
all_masked_columns(['benchmark'], ['benchmark','benchmark_xyz','condition','condition_entropy'])
  -> {'benchmark', 'benchmark_xyz', 'condition', 'condition_entropy'}
all_masked_columns(['subject'], ['meta:family','meta:family_planning'])
  -> {'meta:family', 'meta:family_planning'}
```
`benchmark_id`, `condition_number`, `meta:family_size` (a legitimate non-identity feature) would all be zeroed. This is not "fail safe": over-masking silently destroys signal the optimizer needs, and worse, it makes the masking set **data-dependent on incidental column naming**, so two agents with slightly different feature column sets mask different things → non-reproducible views.

This is CRITICAL not because it leaks (it over-masks) but because the design's whole premise is *atomic, well-defined* masking, and `startswith` makes the mask boundary undefined and naming-fragile. The `feat:`/`meta:` namespaced entries are the intended prefix targets; the bare-name entries (`benchmark`, `condition`, `subject_key`, `data_category`) should be exact-match only.

**Fix:** Make the contract explicit instead of overloading `startswith`. Either (i) split `PROXY_TREE` entries into `exact` names and `prefix` groups, and for prefixes require a separator boundary: `c == proxy or c.startswith(proxy + "__")` (the codebase already uses the `__` suffix convention for aggregates, e.g. `feat:nn_passrate__mean`); or (ii) tag prefix entries explicitly (e.g. trailing `*`) and match accordingly. Add a test with adversarial near-miss names (`benchmark_id`, `meta:family_size`, `conditional`).

---

## MAJOR

### M1. `n_items < n_folds`: silent empty OOF folds and an empty TRAIN fold — no guard
**Files:** `splits.py:18–25` (`outer_folds`), `splits.py:32–48` (`inner_folds`), `manifest.py:41–44`

Reproduced with 1 item, 3 folds:
```
fold0: oof=()      train=('a',)
fold1: oof=()      train=('a',)
fold2: oof=('a',)  train=()
```
Two folds produce **no OOF predictions** (rows in them never get scored), and fold 2 trains on **zero items** (a model fit on empty data → garbage or crash downstream). On the trial phase, which is explicitly *item-subsampled* (§5: "Trial = item-subsampled data"), `n_items < n_folds` or near-empty folds are realistic, not hypothetical. The OOF NLL would then be computed over a non-uniform, partially-empty row set — silently optimistic/biased, and different per agent if their subsample differs.

**Fix:** In `build_manifest`/`outer_folds`/`inner_folds`, assert `n_unique_items >= n_folds` (or `>= 2*n_folds` for a safety margin), with a clear message naming the count. The trial harness should clamp `n_folds` or skip the candidate. Add tests asserting the guard fires.

### M2. Inner-fold seed derivation `seed + 1000 + outer_index` can collide across (seed, outer_index) pairs
**File:** `splits.py:41` — `inner_seed = seed + 1000 + outer_index`

This is additive, so `(seed=0, outer_index=2)` and `(seed=2, outer_index=0)` and `(seed=1, outer_index=1)` all produce `inner_seed=1002` → **identical inner fold structure**. With a single fixed top-level seed and `outer_index ∈ {0..n_folds-1}` the collisions are internal-only and probably benign today, but this is a footgun: any future code that sweeps seeds (e.g. seed as a search knob, or multiple manifests) gets correlated/identical inner partitions across unrelated configurations, undermining the independence the docstring claims ("each outer fold recurses independently"). The magic constant `1000` also silently caps safe `outer_index`/seed ranges.

**Fix:** Derive the inner seed by hashing, not arithmetic: `inner_seed = int(hashlib.sha256(f"{seed}:{outer_index}".encode()).hexdigest(), 16) % (2**31)`. This is collision-resistant and matches the hashing discipline already used in `item_fold`.

### M3. PROXY_TREE is incomplete — multiple plausible identity proxies from §6/§5.1 features are missing
**File:** `proxy_tree.py:14–32`

The spec's feature roster (funnel §5: "per-agent embeddings, **judge**, NN-passrate, pool, metadata, **cluster**") and §5.1 (mean-encodings / OOF residual board) imply derived features that proxy identity but are NOT in the tree. Concretely missing:

- **Judge features** (`feat:judge…`, spec §5 funnel) — judge scores are computed per subject/benchmark response and strongly proxy both identities. **Not masked.**
- **Cluster / embedding features** (`feat:cluster…`, `item_emb…`, `subj_emb…`) — cluster ids and subject embeddings are near-deterministic functions of subject identity. The tests even use `item_emb__0` as a *non-masked control*, but a **subject-level** embedding (`subj_emb__*`) or a cluster-of-subject id absolutely proxies the subject and is currently exposed.
- **Mean-encoded benchmark×condition keys** (`feat:benchmark_mean`, `feat:bench_cond_mean`) — §5.1 explicitly discusses mean-encodings; the tree only has `feat:subject_mean` and `feat:nn_passrate`. A benchmark mean-encoding proxies the benchmark identity and is **not** under the `benchmark` node.
- **Token-length / text-stat features** (`feat:toklen`, length of `subject_content`/`item_content`) — high-cardinality near-identifiers of a fixed subject's response style; at minimum flag.
- **Language id** of subject/benchmark content — stable per subject/benchmark family.
- **`item_key` itself / any hash of it** — if the raw `item_key` (or `benchmark+condition` substring) ever lands in `feature_columns`, it is a perfect identity key and is not listed.
- **`subject_content` is listed** but its *derived* text features (the realistic leak vector — you rarely feed raw text to a GBDT) are not.

Because masking is keyed to exact/prefix names, anything the feature pipeline emits under a name the tree doesn't anticipate **silently leaks identity through dropout**. This is the highest-leverage leak surface in the package.

**Fix:** (1) Add `feat:judge`, `feat:cluster`, `subj_emb`, `feat:benchmark_mean`/`feat:bench_cond_mean` (to `benchmark`), `feat:toklen`/`feat:textstat`, `feat:langid`. (2) Add an **allowlist-or-explain** discipline: a probe that, given the actual `feature_columns`, asserts every column is either explicitly identity-neutral (allowlisted) or covered by a proxy node — so a new feature column fails loudly instead of leaking silently. That inverts the default from "unlisted ⇒ exposed" to "unlisted ⇒ blocked," which is the only safe default for a leakage core.

### M4. No cross-node interaction / no benchmark↔subject co-masking, and `dropout` hardcodes the two roots
**Files:** `dropout.py:31–32`, `proxy_tree.py`

`apply_proxy_dropout` hardcodes `all_masked_columns(["subject"], …)` and `["benchmark"]`. If `PROXY_TREE` later grows a third root (e.g. `judge`, `organization` as a separable axis) the dropout silently ignores it — the producer and the data file drift. Also, features that are **joint** functions (subject×benchmark NN-passrate, a `feat:nn_passrate` aggregated over *both* axes) are placed only under `subject`; dropping the *benchmark* leaves them exposed even though they encode the held-out benchmark's passrate.

**Fix:** Drive the dropout from `PROXY_TREE.keys()` rather than hardcoded literals, iterating roots. Decide explicitly where cross-axis aggregates live (probably under both roots, or a third `pair` root that any drop on either axis triggers) and document it. Add a test that a benchmark drop masks the benchmark-side passrate aggregate.

---

## MINOR

### m1. `assert_no_proxy_leak` and the package are blind to duplicate column names
**File:** `probes.py:31` (`cols.index(c)`) and `dropout.py:26` (`{c: i for i, c in enumerate(cols)}`)

`cols.index(c)` and the dict comprehension both keep only the **first** occurrence of a duplicated column name. If `feature_columns` ever contains a repeated name (e.g. two `feat:pool` columns from different join paths), the second is never masked or checked → silent leak. Reproduced: `['subject_key','subject_key'].index('subject_key') == 0`.
**Fix:** Assert `len(set(feature_columns)) == len(feature_columns)` at entry to `apply_proxy_dropout` and `all_masked_columns`, or build index lists with all occurrences.

### m2. `apply_proxy_dropout` does not validate that requested proxy columns exist
**File:** `dropout.py:39,44` — `[col_idx[c] for c in subj_cols if c in col_idx]`

The `if c in col_idx` filter silently drops proxy columns that aren't present. That's reasonable for optional columns, but it means a **typo or rename in PROXY_TREE** (e.g. `meta:familly`) fails open — the column is never masked and no one notices. Combined with M3's "unlisted ⇒ exposed," this is a silent-leak amplifier.
**Fix:** Log/return which proxy entries matched zero columns; optionally a strict mode that requires the namespaced identity columns (`subject_key`, `benchmark`) to be present.

### m3. Determinism is good on the hash, but `info["dropped_*"]` and `all_masked_columns` return `set`s
**Files:** `dropout.py:48`, `proxy_tree.py:39–52`

`item_fold` uses SHA-256 over a UTF-8 string → correctly stable across machines/Python builds (good; not `hash()`). `build_manifest` sorts keys → order-independent (good). However the **returned `set`s** (`dropped_subjects`, `all_masked_columns`) have nondeterministic iteration order. This is fine *as long as* they are only ever consumed by membership tests / `np.isin` (current usage — `np.isin(subj_arr, list(dropped_subj))` is order-insensitive). It becomes a determinism bug the instant any consumer iterates them into a saved artifact, a logged list, or an RNG-consuming operation. Since this is the hygiene core that everything keys reproducibility off, prefer sorted tuples on the public boundary.
**Fix:** Return `tuple(sorted(...))` for `dropped_subjects`/`dropped_benchmarks` and have `all_masked_columns` callers sort before any ordered use. (Internal set math can stay a set.)

### m4. `SplitManifest.assignment` typed as bare `dict`; `Fold` tuples untyped
**Files:** `manifest.py:25`, `splits.py:11–15`

`assignment: dict` and `train_item_keys: tuple` lose the key/value types. For a byte-identical cross-agent contract, precise typing (`dict[str, int]`, `tuple[str, ...]`) is cheap insurance and documents the invariant (keys are `str`, folds are `int`). Note `fold_of`/`load` already defensively `str()`/`int()`-coerce, which is good and should be called out as the reason the JSON round-trip survives int-key stringification.
**Fix:** Add precise generics; consider a `__post_init__` validation that all values are in `range(n_folds)`.

### m5. `assert_identical` compares the whole `assignment` dict — relies on dict `==` being order-insensitive (it is) but not on identical seed/n_folds determinism of *values*
**File:** `manifest.py:47–49`

This is correct today (`dict.__eq__` ignores insertion order; the test `test_two_agents_same_seed_produce_identical_manifest` with reversed input confirms it). Worth an inline comment that the equality is intentionally order-insensitive so a future refactor to a list/`OrderedDict` doesn't reintroduce an order dependence. No code change required; documentation only.

### m6. `descendants` returns `list(PROXY_TREE.get(node, []))` — unknown node silently yields empty mask
**File:** `proxy_tree.py:35–36`

`all_masked_columns(["typo_node"], cols)` returns `set()` with no error → a mis-specified dropped node masks nothing and fails open. Low likelihood given the two callers pass literals, but it's the same fail-open pattern as m2.
**Fix:** Raise `KeyError`/`ValueError` on an unknown node in `descendants`, or have `apply_proxy_dropout` validate node names against `PROXY_TREE`.

### m7. JSON `save` is deterministic (`sort_keys=True`) — good; but no schema/version field
**File:** `manifest.py:30–38`

The artifact is byte-identical across agents (sorted keys, no floats) — correct. Adding a `"version"`/`"format"` field would let future format changes fail loudly rather than load-mismatch silently. Minor robustness.

---

## What is correct and should NOT be changed

- **`item_fold` hashing** (`manifest.py:17`): SHA-256 over a UTF-8 byte string, not Python `hash()` — genuinely stable across machines and PYTHONHASHSEED. This is the load-bearing determinism primitive and it's right.
- **Item-uniform grouping**: folds are keyed on `item_key`, never row index; `row_fold_ids` and `assert_row_uniform_safe` correctly enforce that all rows of an item share a fold.
- **`inner_folds` does NOT touch outer OOF**: it partitions only `train_item_keys`, and the union of inner OOF equals the outer train set (verified by `test_inner_folds_nest_inside_an_outer_train_set_without_touching_oof`). The recursion-leakage concern in the brief is correctly handled — inner folds never include outer-OOF items.
- **`apply_proxy_dropout` does not mutate `X`** (`.copy()`), coerces to `float32`, and `np.ix_(bool_rows, int_cols)` is valid (verified). Entity-consistent masking (all rows of a dropped entity) is correct.
- **`build_manifest` order-independence** and **`assert_identical`** correctly form the cross-agent guard.

---

## Priority-ordered fix list

1. **C1** — make `assert_no_proxy_leak` row-aware (or it disables the leakage tripwire in practice). Have dropout return dropped-row masks.
2. **C2** — fix `startswith` over-masking; split exact vs `__`-boundary prefix matching.
3. **M3** — add missing proxies (judge, cluster, subj-emb, benchmark mean-encodings, toklen, langid) AND invert the default to allowlist-or-block.
4. **M1** — guard `n_items < n_folds`.
5. **M2** — hash the inner seed instead of `seed+1000+outer_index`.
6. **M4** — drive dropout from `PROXY_TREE.keys()`; resolve cross-axis aggregates.
7. **m1–m7** — duplicate-column guard, fail-closed on unknown/typo nodes, sorted public return types, precise typing, schema version.

---

## VERDICT: BLOCK
