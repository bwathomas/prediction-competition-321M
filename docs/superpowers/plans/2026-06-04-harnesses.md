# Harnesses Implementation Plan (funnel · eval · train)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Build `aide/harness/` — the load-only feature **funnel**, the hygiene-owning **eval** harness (OOF + recursive nested OOF + dropout + leakage probes), and the two-phase **train** harness (trial→full with the diversity-aware promotion gate) — fully unit-tested on synthetic fixtures, no GPU/Colab.

**Architecture:** Pure-Python + numpy on top of `aide/hygiene/`. The eval harness is the only thing that touches folds/dropout/probes, so a dropped-in model (`fit(X,y)`/`predict(X)->prob`) can never see held-out items or undropped proxies. The funnel is strictly load-only: a cache miss raises, never recomputes. Toy models (logistic, memorizer-leak-detector) live in a test helper.

**Tech Stack:** Python 3.11, numpy, pytest. Implements spec §5 (three harnesses) + §5.1 (diversity gate) + the §8 leakage-boundary (probes wired into every score).

**Spec:** `docs/superpowers/specs/2026-06-04-aide-stacked-ensemble-design.md`. **Depends on:** Plan 1 (`aide/hygiene/`).

---

## File Structure

```
aide/harness/
  __init__.py        # public API re-exports
  funnel.py          # CacheMissError, FeatureBlock, FeatureStore (load-only assemble)
  metrics.py         # log_loss (mean BCE), auc_roc (rank, None if single-class)
  eval.py            # Dataset, DropoutConfig, EvalResult, oof_predict, evaluate,
                     #   recursive_evaluate (nested OOF for layer-2)
  train.py           # diversity_score, promotion_gate, run_two_phase
  tests/
    __init__.py
    _toy.py          # LogisticModel, MemorizerModel, make_dataset, write_fixture_cache
    test_funnel.py
    test_metrics.py
    test_eval.py
    test_train.py
```

**Model contract (drop-in slot):** `model_factory() -> model` with `model.fit(X, y)` and `model.predict(X) -> probs in [0,1]`. Stacker uses the same contract over member-prediction columns. The real `linear_stacker` + architectures arrive in Plan 3; Plan 2 tests use toy models.

**Run:** `python3 -m pytest aide/harness/tests -q` from repo root.

---

## Task 1: metrics

**Files:** Create `aide/harness/metrics.py`, `aide/harness/tests/test_metrics.py`, `aide/harness/__init__.py`, `aide/harness/tests/__init__.py`.

- [ ] **Step 1: failing test** — `log_loss` of perfect preds ≈ 0; of 0.5 preds = ln2; `auc_roc` of perfect ranking = 1.0, of single-class y = None.
- [ ] **Step 2: run, expect fail** (`ModuleNotFoundError`).
- [ ] **Step 3: implement** `log_loss(y,p,eps=1e-7)` = mean BCE with clip; `auc_roc(y,p)` = Mann-Whitney rank AUC, return `None` if `len(unique(y))<2` or y non-binary.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `feat(harness): metrics (log_loss, auc_roc)`.

## Task 2: funnel (load-only FeatureStore)

**Files:** Create `aide/harness/funnel.py`, `aide/harness/tests/_toy.py` (`write_fixture_cache`), `aide/harness/tests/test_funnel.py`.

- [ ] **Step 1: failing test** — write 2 fixture `.npz` groups; `load_group` returns X/columns/row_ids; `assemble(["a","b"])` concatenates columns in order; missing group → `CacheMissError`; **no file is created** on miss (load-only); misaligned row_ids → `ValueError`.
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `CacheMissError(FileNotFoundError)`, `FeatureBlock`, `FeatureStore(root)` with `available()`, `load_group(g)` (raises `CacheMissError` with a "load-only, no recompute" message), `assemble(groups, row_ids=None)` (column-concat, row_id-alignment check).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `feat(harness): load-only feature funnel`.

## Task 3: eval — OOF + dropout + probes

**Files:** Create `aide/harness/eval.py`; extend `_toy.py` (`LogisticModel`, `MemorizerModel`, `make_dataset`); create `aide/harness/tests/test_eval.py`.

- [ ] **Step 1: failing tests**
  - `oof_predict` fills exactly one prediction per row; train/oof item-disjoint each fold (probe passes); a `MemorizerModel` keyed on the item-id feature column scores ≈0.5 on every OOF row (proves no item leaked into its trainer).
  - `evaluate` with a `LogisticModel` on a learnable synthetic signal returns `nll < ln2` (beats chance).
  - With `DropoutConfig(subject_rate=1.0)`, the trainer's subject-proxy columns are zeroed (assert via a memorizer keyed on a subject-proxy column → loses its signal).
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `Dataset`, `DropoutConfig`, `EvalResult`, `_apply_dropout` (wraps `apply_proxy_dropout`), `oof_predict` (outer_folds, per-fold fit/predict, `assert_item_disjoint`, no-NaN assert), `evaluate` (seeded rng → `oof_predict` → `log_loss`/`auc_roc`).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `feat(harness): OOF eval with dropout + leakage probes`.

## Task 4: eval — recursive (nested) OOF for layer-2

**Files:** extend `aide/harness/eval.py` (`recursive_evaluate`); extend `test_eval.py`.

- [ ] **Step 1: failing test** — the leakage regression: with `MemorizerModel` members keyed on the item-id column, the stacker's TRAIN meta-features (`meta_tr`) are ≈0.5 everywhere (each was produced by an inner-OOF member that never trained on that row's item). A naive in-sample variant would give 1.0. Also: `recursive_evaluate` returns one final pred per row, `nll` finite, beats chance on a learnable signal with `LogisticModel` members + logistic stacker.
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `recursive_evaluate(member_factories, stacker_factory, ds, manifest, *, dropout=None, seed=0)`: per outer fold → `inner_folds` over `fold.train` builds OOF `meta_tr`; train stacker on `meta_tr→y`; train members on all of `fold.train`, predict `fold.oof` → `meta_te`; stacker.predict(`meta_te`) → final; concat → `log_loss`. No-NaN asserts on `meta_tr` and `final`.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `feat(harness): recursive nested-OOF eval for layer-2`.

## Task 5: train — diversity gate + two-phase

**Files:** Create `aide/harness/train.py`, `aide/harness/tests/test_train.py`.

- [ ] **Step 1: failing tests**
  - `diversity_score(resid, pool)` = `1 - mean pairwise corr`; 1.0 when pool empty; ≈0 when resid duplicates a pool member; high when orthogonal.
  - `promotion_gate`: promotes when `nll <= best + X` (competitive) OR `diversity >= D` (diversifying-but-weaker); rejects when neither.
  - `run_two_phase`: a competitive candidate runs full; a weak+non-diverse candidate is rejected without a full run (assert the full `eval_fn` was NOT called); a weak+diverse candidate runs full.
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `diversity_score`, `promotion_gate(cand_nll, group_best_nll, X, diversity, D)`, `run_two_phase(eval_fn, trial_ds, full_ds, *, group_best_nll, X, pool_resids, D)` returning `PromotionResult(promoted, trial_nll, full_nll, diversity)` where `eval_fn(ds)->(nll, residuals)`.
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `feat(harness): diversity-aware two-phase trial->full gate`.

## Task 6: full suite + code-review

- [ ] **Step 1:** `python3 -m pytest aide/harness aide/hygiene -q` → all green.
- [ ] **Step 2:** dispatch a code-review agent on `aide/harness/`; priorities: any OOF/recursion leakage path, funnel ever recomputing, dropout applied where intended, NaN/empty-fold handling. Save to `quality_reports/cross_artifact_aide_harness/review.md`.
- [ ] **Step 3:** fix CRITICAL/MAJOR + regression tests; re-run; commit.

---

## Self-Review (plan vs spec)

- §5 funnel (load-only, hard error) → Task 2. ✓
- §5 eval (owns folds/recursion/dropout; scalar NLL) → Tasks 3–4. ✓
- §5 train (trial→full gate) + §5.1 diversity (1−ρ̄, cross-member) → Task 5. ✓
- §8 leakage tripwire on every score → probes wired in `oof_predict`/`recursive_evaluate` (Tasks 3–4). ✓
- Deferred (correct scope): real architectures + `linear_stacker` → Plan 3; Drive-backed cache + cross-agent OOF board → Plan 4; AIDE/orchestrator → Plan 5.
