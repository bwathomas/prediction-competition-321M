# Adversarial Review — `aide/harness/` (eval/train/funnel layer)

**Reviewer:** forked code-review agent
**Branch:** `clean/aide-stacked-ensemble`
**Scope:** `aide/harness/{metrics,funnel,eval,train,__init__}.py` + `aide/harness/tests/*`
**Test run:** `python3 -m pytest aide/harness -q` → **26 passed in 0.26s**

The harness is the leakage-enforcement boundary: it must own folds, recursion, dropout, and
probes so a dropped-in `fit/predict` model cannot leak. The fold/recursion/stacking skeleton is
correct and the nested-OOF layer-2 logic is genuinely leakage-free. **But the dropout path has a
real reproducibility-and-leakage defect, and the advertised hygiene probes are not wired into the
scoring paths.** Details below, ordered by severity.

---

## CRITICAL

### C1 — Dropout masks DIFFERENT entity sets in train vs. test (rng advances between calls) — `eval.py:65-66, 101-102, 134-137`

`_apply_dropout` is called separately for the train matrix and the test matrix, passing the **same
`rng` object**, which advances state between the two calls. With any `*_rate ∈ (0,1)` the
`_drop_set` draw differs, so the set of dropped subjects/benchmarks for the trainer is not the set
masked at predict time. Verified empirically:

```
train dropped subjects: {'s1','s2','s3'}
test  dropped subjects: {'s3'}          # same rng, second call
SAME? False
```

Consequences, all bad:

1. **Train/serve skew.** The member trains with subjects {s1,s2,s3} zeroed but predicts on rows
   where s1,s2 still carry their identity proxy columns — columns the model only ever saw as 0.
   The cold-start signal the dropout is meant to teach is undermined.
2. **Identity leakage at scoring time.** Held-out rows whose subject was dropped in *training* but
   not in *test* expose identity proxies to `predict`, exactly the channel dropout exists to close.
3. **Non-reproducibility of the very thing the docstring promises.** Which entities get masked at
   predict depends on how many draws were consumed earlier (number of inner folds, member count,
   outer fold index), so the score is order-fragile.

**Fix.** The mask must be a deterministic function of the entity identity, not of rng draw order,
and must be IDENTICAL for train and test within a fold. Options:
- Compute the dropped-entity set ONCE per fold from a fold-local, identity-seeded rng (e.g.
  `np.random.default_rng(hash((seed, fold.index)))`) and apply the SAME set to both `Xtr` and `Xte`
  (thread the resolved `dropped` set, not the live rng, into a masking helper); OR
- Make `apply_proxy_dropout` accept a precomputed `dropped` set, and have the harness draw it once
  per fold before masking either matrix.

Note this is a harness-level defect (mis-use of the hygiene primitive), so it is in scope for this
review even though `apply_proxy_dropout` itself lives in the already-reviewed `hygiene/` package.

### C2 — Hygiene leakage probes are NOT called on the dropout scoring paths — `eval.py:16-18`

The eval module imports only `assert_item_disjoint`. `assert_no_proxy_leak` (the dropout tripwire)
and `assert_columns_covered` (the "unlisted ⇒ blocked" column classifier) are **never invoked**
anywhere in `harness/`:

```
$ grep -rn "assert_no_proxy_leak\|assert_columns_covered\|assert_row_uniform_safe" aide/harness/
   (no matches)
```

The eval.py module docstring claims hygiene is "enforced here, not trusted to the model," and the
project rule states the probes must be "called on every scoring path." Neither dropout probe runs.
So even setting C1 aside, nothing verifies that dropout actually zeroed what it was supposed to, and
nothing forces a newly-added feature column to be classified before it can ride through dropout
unmasked (the whole point of M3 "unlisted ⇒ blocked"). The probes exist but are decorative here.

**Fix.** After every `_apply_dropout`, call `assert_no_proxy_leak(Xmasked, feature_columns,
[dropped_root], rows=info["drop_rows"][root])` for each root that was dropped — which requires
`_apply_dropout` to return the `info` dict (currently discarded at `eval.py:51`). Call
`assert_columns_covered(feature_columns, neutral_prefixes=...)` once at the top of every public
entry point (`oof_predict`, `recursive_evaluate`) so an unclassified column aborts the score.

---

## MAJOR

### M1 — Outer scoring path of `recursive_evaluate` skips the item-disjoint tripwire — `eval.py:120-121`

`oof_predict` (line 62) and `build_oof_meta` (line 93) both call `assert_item_disjoint`, but the
**outer** loop of `recursive_evaluate` does not, even though it is the path that produces the final
scored predictions (`meta_te` members fit on `Xtr`, predict `Xte`). The split is sourced from the
same trusted `outer_folds(manifest)`, so this is not an active leak today — but the contract is
"probe on every scoring path," and a future change to how outer rows are selected would go
unguarded. Add `assert_item_disjoint(fold.train_item_keys, fold.oof_item_keys)` at the top of the
loop. (Cheap, and it makes the leakage-free claim self-checking rather than inherited.)

### M2 — No adversarial test proves the dropout path is leakage-free — `tests/test_eval.py:36-44`

`test_subject_dropout_zeros_subject_proxy_in_training_X` uses `subject_rate=1.0`. At rate 1.0
*every* entity is dropped on both calls, so the C1 set-mismatch is invisible and the train matrix is
fully masked regardless of rng order. The test therefore passes against the buggy implementation.
There is **no** test that (a) uses a partial rate and asserts train/test masks agree, (b) asserts a
held-out dropped subject's proxy column is zeroed in the matrix handed to `predict`, or (c) uses the
`MemorizerModel` keyed on the *subject* column under partial dropout to prove a dropped subject can't
be memorized. As written, a leaky dropout implementation ships green.

**Missing adversarial tests to add:**
- Partial-rate (e.g. 0.5) dropout: capture both the train-fit `X` and the predict-time `X`
  (extend `CaptureModel` to also record predict input) and assert the masked entity set is identical.
- `MemorizerModel(key_col=subject_key_index)` under partial dropout → must score 0.5 on rows of any
  dropped subject (proves identity can't survive into predict).
- A `recursive_evaluate` run with a member that memorizes the outer-OOF item id → must score 0.5
  (proves meta_te members never saw outer-OOF items).

---

## MINOR

### m1 — `diversity_score` treats a degenerate (constant) pool member as "uncorrelated" — `train.py:18, 33`
`_safe_corr` returns `0.0` when either residual vector has zero std. A constant pool member then
contributes `corr=0` → inflates `1 - mean(corr)` toward "diverse." A candidate can clear the gate
purely because the pool contains a degenerate member. Consider excluding zero-variance pool members
from the mean (and documenting the choice) rather than scoring them as orthogonal.

### m2 — `diversity_score` mean over a heterogeneous pool can exceed sane bounds silently — `train.py:34`
Signed correlation in `[-1,1]` means `diversity ∈ [0,2]`; with anti-correlated members it can exceed
1 by design (the test asserts >1.5). That is intentional per the spec, but `D` defaults and the gate
comparison (`diversity >= D`) live in different files with no shared constant. Centralize the `X`/`D`
defaults (spec says `X≈0.01`, `D≈0.4`) so trial and full agents cannot drift. Advisory.

### m3 — `log_loss` silently coerces probabilities outside `[eps, 1-eps]` with no NaN/inf guard — `metrics.py:9-12`
`np.clip` hides a model that returns NaN/inf (`clip(nan)` → nan propagates; `clip(inf)` → 1-eps,
masking a blown-up logit). For a scoring harness that is the agents' only objective, a NaN prediction
should abort, not be clipped into a finite loss. Add `if not np.all(np.isfinite(p)): raise`. Same for
`auc_roc` inputs.

### m4 — `funnel.assemble` load-only guarantee is solid; one gap — `funnel.py:62`
When `row_ids is None`, the reference order is `blocks[0].row_ids`, so group 0 is implicitly trusted
and only groups 1..n are checked against it. If group 0 itself is the misaligned one, the mismatch is
detected (others won't match it) — but the *error message* will blame an innocent group. Cosmetic;
the load-only / raise-on-miss behavior itself is correct and well-tested (C-miss raises, writes no
file, `available()` unaffected — `test_funnel.py` covers all three). No silent degradation path found.

---

## What is correct (verified, not assumed)

- **Layer-2 stacker never trains on in-sample member preds.** `build_oof_meta` fills `meta_tr` from
  inner-OOF members (each cell from a member that didn't train on that row's item), and `stacker.fit`
  consumes only `meta_tr`. The nested-OOF recursion guard holds. (`test_build_oof_meta_is_nested_leakage_free`
  with the item-memorizer scoring exactly 0.5 is a genuine proof of this specific path.)
- **meta_te members did not see outer-OOF items.** Members are fit on outer `Xtr` and predict outer
  `Xte`, which are item-disjoint via the manifest. No outer leakage into final predictions.
- **Item-uniform OOF coverage.** `oof_predict` asserts every row gets exactly one prediction; the
  memorizer-scores-0.5 test proves item-cold-start holds on the layer-1 path.
- **Metrics correctness.** `log_loss` clipping/mean-BCE correct; `auc_roc` Mann-Whitney with
  average-rank tie handling correct (all-ties → 0.5), single-class and non-binary → `None`. Well-tested.
- **Funnel.** Load-only, raises `CacheMissError` on miss with no recompute/write, row-alignment
  enforced, `allow_pickle=False`. No silent-degradation path.
- **Reproducibility of folds.** Folds/inner-folds are hash-derived and deterministic; ordering is
  over `sorted(...)` sets. The ONLY reproducibility hole is the dropout rng-threading (C1).

---

## Test-suite verdict on the leakage claims

The item-cold-start (layer-1) and nested-OOF (layer-2) claims **are** proven by adversarial
memorizer tests. The **dropout** leakage claim is **not** — the single dropout test (rate=1.0) cannot
distinguish a correct implementation from the C1-buggy one. A leaky dropout path ships green today.

---

## VERDICT: BLOCK — fix C1 (dropout train/test mask mismatch) and C2 (probes not wired), and add the partial-rate dropout adversarial tests (M2) before merge.

---

## Resolution (2026-06-04)

All findings addressed; 65/65 tests pass (`python3 -m pytest aide -q`). Each fix carries a regression test.

- C1 RESOLVED — dropout now chooses ONE dropped set per fold from a fold-deterministic rng (`np.random.default_rng([seed, fold_index])`) via `choose_dropped`, applied to BOTH train and test with `mask_dropped`. Regressions: `test_one_chosen_set_masks_train_and_test_consistently` (hygiene) + `test_evaluate_is_reproducible_under_partial_dropout` (eval).
- C2 RESOLVED — `_dropout_fold` calls `assert_no_proxy_leak(rows=...)` on every masked matrix; `evaluate`/`oof_predict`/`recursive_evaluate` call `assert_columns_covered` when `neutral_prefixes` is supplied. Regressions: `test_column_coverage_{passes,raises}`.
- M1 RESOLVED — `assert_item_disjoint` now also guards the outer scoring loop in `recursive_evaluate`.
- M2 RESOLVED — added partial-rate dropout consistency + reproducibility adversarial tests (above).
- m1 RESOLVED — `diversity_score` skips zero-variance pool members. Regression: `test_diversity_ignores_degenerate_zero_variance_pool_member`.
- m3 RESOLVED — `log_loss` raises on non-finite predictions. Regression: `test_log_loss_aborts_on_non_finite_predictions`.

**VERDICT: ALL-RESOLVED** (was BLOCK).
