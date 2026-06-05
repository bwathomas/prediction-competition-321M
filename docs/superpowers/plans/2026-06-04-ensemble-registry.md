# Ensemble Registry Implementation Plan (stacker · architectures · ablations · smoke gate)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Build `aide/ensemble/` — the `LinearStacker` used at both layers, a registry of architectures conforming to the harness `fit(X,y)`/`predict(X)->prob` slot (pure-numpy canon tested locally: **MLP**, logistic; heavy Kaggle libs registered as lazy Colab-only entries), the feature-**ablation** wrapper, and a one-fold **smoke gate** every architecture must pass before AIDE may compose it.

**Architecture:** Everything composes through the Plan-2 harness. A layer-2 "architecture part" = `LinearStacker` over feature-ablated variants of one architecture (members); the top layer = `LinearStacker` over architecture parts. Ablation = restrict a model to a column subset. Heavy architectures (LightGBM/XGBoost/CatBoost/TabNet/FT-Transformer/TabPFN/torch-IRT) are registered behind lazy loaders that raise a clear "requires Colab" if the lib is absent, so local CI stays numpy-only.

**Tech Stack:** Python 3.11, numpy, pytest. Implements spec §6/§6.1. **Depends on:** Plans 1–2 (`aide/hygiene`, `aide/harness`).

---

## File Structure

```
aide/ensemble/
  __init__.py            # public API
  linear_stacker.py      # LinearStacker (logistic over member-pred logits; optional nonneg)
  architectures.py       # LogisticArchitecture, MLPArchitecture (numpy) + lazy heavy loaders
  ablations.py           # Ablation, AblatedModel, make_ablated_factory
  registry.py            # REGISTRY, get(name, **kw) -> factory, smoke_test, SmokeResult
  tests/
    __init__.py
    test_linear_stacker.py
    test_architectures.py
    test_ablations.py
    test_registry.py
    test_two_layer_integration.py
```

**Model contract:** `factory() -> model`; `model.fit(X, y)`; `model.predict(X) -> probs∈[0,1]`. Identical for members and stacker (stacker's X = member-prediction columns).

**Run:** `python3 -m pytest aide/ensemble -q`.

---

## Task 1: LinearStacker
**Files:** Create `aide/ensemble/linear_stacker.py`, `aide/ensemble/__init__.py`, `aide/ensemble/tests/__init__.py`, `test_linear_stacker.py`.
- [ ] Failing test: 2 members = noisy copies of y → stacker NLL < each member's; preds ∈[0,1]; `nonneg=True` keeps member weights ≥0.
- [ ] Implement: logistic GD over standardized **logits** of member preds (`_logit` with clip); optional non-negative projection on member weights each step.
- [ ] Run PASS; commit `feat(ensemble): LinearStacker`.

## Task 2: architectures (numpy canon + lazy heavy)
**Files:** Create `architectures.py`, `test_architectures.py`.
- [ ] Failing test: `LogisticArchitecture` and `MLPArchitecture` both beat chance (`evaluate().nll < ln2`) on `_toy.make_dataset`; MLP also beats chance on a nonlinear (XOR-like) signal where logistic ≈ chance.
- [ ] Implement: `LogisticArchitecture` (standardized logistic), `MLPArchitecture` (1 hidden ReLU layer, He init, standardized inputs, seeded), and `_lazy_lightgbm`/`_lazy_xgboost`/`_lazy_catboost` loaders that raise `RuntimeError("requires Colab")` when the lib is missing.
- [ ] Run PASS; commit `feat(ensemble): numpy architectures (logistic, MLP) + lazy heavy loaders`.

## Task 3: ablations
**Files:** Create `ablations.py`, `test_ablations.py`.
- [ ] Failing test: `make_ablated_factory(base, feature_columns, keep_columns)` yields a model that sees ONLY kept columns (a base that would memorize a dropped column can't, because it's sliced out); column order preserved.
- [ ] Implement: `Ablation(name, columns)`, `AblatedModel(base_factory, keep_idx)` (slice X in fit+predict), `make_ablated_factory`.
- [ ] Run PASS; commit `feat(ensemble): feature-ablation wrapper`.

## Task 4: registry + smoke gate
**Files:** Create `registry.py`, `test_registry.py`.
- [ ] Failing test: `get("mlp")`/`get("logistic")` → `smoke_test` returns `ok=True`, finite nll, oof shape correct; a deliberately-broken factory (predicts NaN) → `ok=False`; `get("gbdt_lightgbm")()` raises `RuntimeError` when lightgbm is absent (else smoke passes).
- [ ] Implement: `REGISTRY` (numpy + lazy heavy), `get(name, **kw)` returns a factory, `SmokeResult`, `smoke_test(factory, ds, manifest, *, dropout=None)` runs one `evaluate` and checks finite nll + oof shape, catching exceptions into `ok=False`.
- [ ] Run PASS; commit `feat(ensemble): registry + one-fold smoke gate`.

## Task 5: two-layer integration
**Files:** Create `test_two_layer_integration.py`.
- [ ] Test: members = `[make_ablated_factory(MLP, cols, A), make_ablated_factory(Logistic, cols, B)]`, stacker = `LinearStacker`; `recursive_evaluate(members, stacker, ds, manifest)` beats chance and returns one pred/row — proving the registry pieces compose into the two-layer ensemble via the Plan-2 harness.
- [ ] Run PASS; commit `test(ensemble): two-layer composition via recursive_evaluate`.

## Task 6: full suite + code-review
- [ ] `python3 -m pytest aide -q` green.
- [ ] Code-review agent over `aide/ensemble/`; save to `quality_reports/cross_artifact_aide_ensemble/review.md`; fix CRITICAL/MAJOR + regression tests; re-run; commit.

---

## Self-Review (plan vs spec)
- §6 top/bottom linear stacker → Task 1 + Task 5. ✓
- §6.1 canon incl. MLP (kept) → Task 2 (numpy MLP/logistic tested; heavy libs lazy-registered). ✓
- §6 feature-ablated variants → Task 3. ✓
- §6.1 per-architecture smoke gate → Task 4. ✓
- Deferred (correct scope): real heavy-lib training + real cached embeddings → Plan 4 (Colab); AIDE search over the registry → Plan 5.
