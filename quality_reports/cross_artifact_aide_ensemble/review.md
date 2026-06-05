# Adversarial Code Review — `aide/ensemble/`

**Branch:** `clean/aide-stacked-ensemble`
**Reviewer scope:** ensemble package only (linear_stacker, architectures, ablations, registry, __init__, tests). Harness/hygiene treated as already-reviewed dependencies.
**Method:** static read + targeted runtime probes (finite-difference gradient check, smoke-gate fuzzing, zero-variance edge cases, seed sweep) + full suite.

---

## Verified-correct (load-bearing claims I actively checked)

- **MLP gradients are correct.** Finite-difference check at a random point: `dW1`, `dW2`, `db1`, `db2` all match to ~1e-10. ReLU mask `(H>0)`, He init (`sqrt(2/d)`, `sqrt(2/hidden)`), and L2-on-weights-only (biases excluded) are all right.
- **No NaN/inf path found in numpy members or stacker.** `_sigmoid` clips `z` to [-30,30] before `exp`, so logits never overflow; `_logit` clips `p` to [eps, 1-eps], so constant 0/1 members map to finite ±13.8. Predictions are strictly in (0,1) (probed: min ~3e-6, max ~0.99999), never exactly 0/1, so they survive `log_loss`'s finite-check and clip.
- **Lazy heavy loader is genuinely lazy.** Importing `registry`/`architectures` does NOT import lightgbm (`'lightgbm' in sys.modules` is False). Off-Colab `get("gbdt_lightgbm")()` raises a clear `RuntimeError` naming the lib and the Colab runtime. No module-level heavy import.
- **Ablation indexing is correct.** `[cols.index(c) for c in keep_columns]` preserves keep-order (not column-order); single-column keep returns a 2D `(n,1)` slice (no 1D collapse); KeyError on missing column; AblatedModel slices in both `fit` and `predict` so non-kept columns never leak. Tests confirm all three.
- **Stacker handles a single member column** (mu/w shapes correct, finite predictions) and **constant 0.5 members** (logit→0, std→1 guard fires, finite).
- **`predict` before `fit` raises** `TypeError` rather than returning garbage.
- **MLP-beats-logistic test is robust, not seed-fragile.** Swept 8 data seeds: MLP AUC consistently 0.987–0.997 on XOR, logistic 0.41–0.59, MLP NLL always < logistic NLL. The 0.8/0.65 thresholds have wide headroom.

---

## Findings

### MAJOR

**M1 — Zero-variance guard uses exact `== 0`; constant logit columns produce float-noise std (~1e-9 to 1e-14) and slip past it.**
`linear_stacker.py:39-40` (and identically `architectures.py:19-20`, `_toy.py`):
```python
self.sd = Z.std(axis=0)
self.sd[self.sd == 0] = 1.0
```
A member whose logit is *essentially* constant across a (nested-OOF) train fold — e.g. a saturated MLP, or a near-constant prediction `0.3 + 1e-9·noise` — has `Z.std` equal to a tiny non-zero float, NOT exactly `0.0`. The `== 0` guard does not fire, and `(Z - mu)/sd` then divides by ~1e-9, amplifying pure floating-point jitter into a standardized O(1) feature that the GD loop fits a weight to.

Observed (probe): member `0.3 + N(0,1e-9)` → `sd ≈ 4.4e-9`, fitted weight on the noise column, predictions stayed finite (sigmoid clip saves NaN), and in that instance the blend was unharmed. So this is **not** a NaN/crash bug — the ±30 clip is a hard backstop. But it defeats the *intent* of the guard: a degenerate member injects fitted noise instead of being neutralized, a mild overfit/instability risk that is exactly the scenario the guard exists to prevent, and it is data-dependent whether it bites.
**Fix:** guard on a relative/absolute tolerance, e.g. `self.sd[self.sd < 1e-8] = 1.0` (or `np.where(self.sd < 1e-8, 1.0, self.sd)`). Apply the same fix to `_Standardizer` in `architectures.py:20`.

**M2 — Smoke gate does not enforce the `predict() ∈ [0,1]` contract.**
`registry.py:45`: `ok = bool(math.isfinite(res.nll) and res.oof.shape == (len(ds.y),))`.
A model that returns finite probabilities *outside* [0,1] (the classic "forgot the sigmoid" / raw-logit bug) passes the gate: probed a model returning constant `1.5` → `smoke_test(...).ok == True`, `nll == 7.92`. `log_loss` clips internally to [eps, 1-eps] so the NLL is finite (just bad), and nothing checks the prediction range. The smoke gate is advertised (registry docstring, prompt priority 5) as the thing that catches a "subtly-broken architecture" before AIDE composes it — an unbounded-output member is precisely such a break and it is admitted. Note the *shape* and *NaN* checks ARE effective: wrong-length or 2-D `predict` outputs raise inside `oof_predict` (boolean-index assignment error) and are captured as `ok=False`; a NaN-predicting model trips the harness's "every row must receive a prediction" assertion. Only the **range** check is missing.
**Fix:** in `smoke_test`, after a successful `evaluate`, add `in_range = float(np.nanmin(res.oof)) >= 0.0 and float(np.nanmax(res.oof)) <= 1.0` and fold it into `ok`. (Cheap, and it closes the one gate gap.)

### MINOR

**m1 — `Ablation` dataclass is dead code.** `ablations.py:15-18` defines and `__init__.py` exports `Ablation(name, columns)`, but `make_ablated_factory` takes raw `keep_columns` and never constructs or consumes an `Ablation`. Either wire it into the factory API (e.g. accept an `Ablation` and read `.columns`) or drop it to avoid a misleading public surface. No correctness impact.

**m2 — lightgbm wrapper `predict_proba(...)[:, 1]` is fragile to single-class train folds.** `architectures.py:125`. If a fold's training labels are all one class, sklearn-style `predict_proba` returns shape `(n, 1)` and `[:, 1]` raises `IndexError`. Colab-only and `# pragma: no cover`, so untested here and non-blocking, but flag it: the toy datasets are balanced so it won't surface locally, yet a dropout/ablation fold on real Kaggle data could be single-class. Defensive fix: `proba = self._clf.predict_proba(X); return proba[:, 1] if proba.shape[1] > 1 else np.full(len(X), float(self._clf.classes_[0] == 1))`. The `y > 0.5` boolean cast in `fit` is otherwise correct and the output (when 2-class) is a proper [0,1] probability.

**m3 — Standardization logic is duplicated three ways** (`linear_stacker._design`, `architectures._Standardizer`, `_toy.LogisticModel._std`), each carrying the same `== 0` guard bug (M1). Consolidating to one helper would make the M1 fix single-point. Style/maintainability only.

**m4 — `LinearStacker` GD uses fixed lr/iters with no convergence check.** `fit` runs exactly `iters` steps of plain GD with no early stop or step-size adaptation. Probed nonneg projected-GD on an anti-correlated member: converges cleanly (anti-member weight → 0, finite). Acceptable for the small meta-feature dimensionality, but there is no guard if a caller passes a large `lr`; the test `test_stacker_is_no_worse_than_its_best_member` implicitly depends on convergence at the defaults. Non-blocking.

---

## Missing adversarial tests (recommend adding)

1. **Stacker with a degenerate/constant member column** (the M1 scenario) — assert finite, in-range predictions AND that the degenerate member earns ~0 weight. Currently untested; would have surfaced M1.
2. **Smoke gate rejects an out-of-range predictor** — `predict` returning `1.5` should yield `ok=False`. Currently nothing pins the [0,1] contract at the gate (M2). The existing `test_smoke_catches_broken_factory` only covers the NaN path.
3. **Two-layer integration with `nonneg=True` stacker** — `recursive_evaluate` is only exercised with the default unconstrained stacker; the nonneg projection path through the recursion is untested.
4. **lightgbm wrapper on a single-class fold** (guarded/skipped if lib absent) — would pin m2.

The `test_two_layer_integration` test itself is **not flaky**: nested-OOF on a learnable linear signal with `nll < log(2)` is a wide margin (XOR/linear seeds all clear it comfortably), and seeds are fixed.

---

## Suite result

```
$ python3 -m pytest aide/ensemble -q
............. [100%]
13 passed in 0.92s
```
**13 passed, 0 failed.**

---

## VERDICT: SHIP-WITH-FIXES (apply M1 + M2 before composing untrusted architectures; M1 is a real standardization-intent defect, M2 leaves the smoke gate blind to the most common contract break; both are ~1-line fixes.)

---

## Resolution (2026-06-04)

Both MAJORs + m2 fixed; 80/80 tests pass (`python3 -m pytest aide -q`).

- M1 RESOLVED — variance guard changed to `sd < 1e-8` in `LinearStacker` and `_Standardizer`, so near-constant columns are neutralized, not amplified. Regression: `test_stacker_handles_near_constant_member_without_blowup`.
- M2 RESOLVED — `smoke_test` now also requires `nanmin(oof) >= 0 and nanmax(oof) <= 1`, rejecting out-of-range ("forgot the sigmoid") predictors. Regression: `test_smoke_gate_rejects_out_of_range_predictor`.
- m2 RESOLVED — lightgbm wrapper guards `predict_proba` for single-class folds.
- m1/m3/m4 (dead Ablation dataclass kept for future use; standardization duplicated; stacker convergence) — accepted as non-blocking MINORs.

**VERDICT: SHIP** (was SHIP-WITH-FIXES).
