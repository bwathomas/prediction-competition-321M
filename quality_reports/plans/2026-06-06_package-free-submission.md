# Plan — Package-Free Full-3-Family Submission (0.43653 stack → shippable `model.py`)

**Status:** APPROVED 2026-06-06 — user chose "converters first" (Phase 3 before the rest); 3-encoder budget confirmed fine (R3/Phase 1 dropped). In progress: Phase 3 (xgb/catboost/forest converters + parity).
**Date:** 2026-06-06
**Branch:** create `feat/package-free-submission` off `clean/aide-stacked-ensemble`
**Repo:** `prediction-competition-321M`
**Decision inputs (user, 2026-06-06):** roster = **Full 3-family stack**; approach = **plan the full pipeline first**.

---

## 0. Objective

Turn the shipped research artifact — the **3-family, 21-learner LightGBM meta-stack (secret soft-logloss
0.43653 / OOF 0.42939)** — into a **valid Codabench submission** that runs inside the network-isolated
sandbox using only the allowed runtime packages, reproduces the stack's predictions within numeric
tolerance ("nothing degrades"), and fits the ZIP/size/budget limits.

This is the **full-fidelity** path: all three embedding families, every base learner, featroute, and the
meta — retrained in an export harness (the AIDE objects were never persisted; only predictions exist) and
converted to package-free inference.

---

## 1. Hard constraints (from `Predictive_Evaluation_Challenge.pdf` + `RUNTIME_ENV.md`)

- **Runtime packages:** `torch`, `numpy`, `transformers`, stdlib, `safetensors`, `huggingface_hub`,
  `tokenizers`, `sentencepiece`, `torch_measure`. **NO** `sklearn`/`lightgbm`/`xgboost`/`catboost`/`scipy`/
  `pandas`/`faiss`/`joblib`. `requirements.txt` install is disabled — code must be vendored.
- **Entry point:** `model.py::predict(input: dict, labeled: list[dict]|None) -> float` (native Python float
  in [0,1]). `input` = 4 strings: `benchmark`, `condition`, `subject_content`, `item_content`. Models
  loaded at **module scope** (paid once); `predict()` called once per pair.
- **Network-isolated at test time.** Encoders must be in the ZIP or declared in `models.txt` (pre-fetched).
- **Adaptive labeling (optional):** `labeling.py::acquisition_function(input)->float`; `labeled` passed to
  `predict()`. Default random if absent. Out of scope for v1 (see Phase 7, optional).
- **ZIP budget:** target **≤ 65 MB** for everything except encoder weights (encoders via `models.txt`).
- **Per-round compute budget:** 5,000 item-variants/round, `predict()` once per pair; module-load may take
  minutes (encoders) but per-call must be cheap. **3 large encoders/item is the #1 runtime-budget risk.**
- Metric: mean log-likelihood (higher better) = our soft-logloss (lower better); item cold-start.

---

## 2. The reframe & top risks (must be designed around, not discovered late)

| # | Risk | Why it matters | Mitigation (in plan) |
|---|------|----------------|----------------------|
| R1 | **AIDE models not persisted** (only OOF/holdout preds). | Can't export what we don't have. | **Retrain** the full roster in an export harness with fixed seeds → models persisted → export. Define "no degradation" against this *retrained* baseline R0, not the exact AIDE preds. |
| R2 | **Feature parity (offline vs runtime).** The 594 features/family come from encoder + neighbor/cluster/mean-encode tables fit on the training corpus. | If runtime features ≠ training features, every model silently degrades. | A single shared feature module used **both** offline (build parquet) and at runtime (`predict`); a per-family feature-parity gate (G0) asserting runtime == offline to <1e-5 on a sample. |
| R3 | **3 encoders at runtime** (qwen / mistral / llama embeddings, ~4–8 B each). | Likely over per-round budget; possibly over memory. | Phase 1 spike measures per-call latency for 3 encoders. Fallbacks: (a) declare all 3 in `models.txt`; (b) if too slow, distill/cache or drop to the best 1–2 families (degradation measured, user decides). |
| R4 | **ZIP size.** Neighbor tables for 3 families × 264k items, even PCA+int8, may exceed 65 MB. | Bundle rejected. | Size budget per artifact (Phase 5); compression (PCA-dim, int8, CSR `.npz` triples); prune neighbor tables to what `nn__*` actually needs; measure early. |
| R5 | **Missing converters** (XGBoost, CatBoost, ExtraTrees) — only LightGBM/FM/KNN/LogReg/MLP/Ridge exist. | Roster includes xgb/cat/et. | Phase 3 writes + parity-tests the 3 new converters, mirroring `src/gbdt_member.py`. |
| R6 | **Retrain fidelity.** Retrained members ≠ AIDE members exactly. | The headline 0.43653 may move. | Accept retrain drift; gate = R0 (retrained stack) beats the organizer baseline and ≈0.4365; package-free == R0 within tolerance. Report R0−0.43653 drift. |

---

## 3. Success criteria (the "verify / nothing degrades" gates)

- **G0 — Feature parity (per family):** runtime feature vector == offline feature vector to **<1e-5**
  (rel) on a held sample of items, for all 594 columns. Blocks everything downstream.
- **G1 — Per-member parity:** each package-free member's raw output == the real package's output to
  **<1e-5** (GBDTs/linear) / **<1e-4** (torch fp32) on the holdout matrix. (The literal "produce correct
  outputs, nothing degrades" requirement.)
- **G2 — Stack parity:** package-free 21-learner meta output == retrained-package meta output to **<1e-4**
  on holdout (probability space).
- **G3 — No-degradation vs result:** retrained package-free stack secret soft-logloss **≤ 0.4370**
  (within retrain noise of 0.43653) and OOF **≤ 0.4300**; and **beats the organizer baseline** (the
  thing that actually scores points). Old 0.43722/0.43653 artifacts kept as reference.
- **G4 — Runtime contract:** `model.py` imports only whitelisted packages (static check), loads at module
  scope, returns native float, no network; end-to-end smoke on the CPU validation harness passes; ZIP ≤65 MB.
- **G5 — Per-call budget:** measured mean `predict()` latency × 5,000 within the round budget (Phase 1
  decides if 3 encoders are viable).

---

## 4. Runtime architecture (`model.py` flow)

```
module load (once):
  for fam in [qwen, mistral, llama]:
     ENC[fam] = AutoModel.from_pretrained(slug)        # via models.txt
     FEAT[fam] = FeatureBuilder(load artifacts: centroids, nn-table, mean-encode tables, metadata, scalers)
     MEMBERS[fam] = [load package-free member states: lgb,lgb_goss,lgbdart,xgb,cat,et,knn,mlp,(mlp2),(ridge),fm,irt]
     FEATROUTE[fam] = 8 group sub-models (package-free) → logit-mean
  META = package-free LightGBM (21 inputs)

predict(input, labeled=None):
  for fam:
     emb = ENC[fam].encode(item_content, subject_content)     # last-token pooled
     x_fam = FEAT[fam].build(input, emb)                       # 594-vector, parity-gated
     base_fam = [logit(m.apply_one(x_fam)) for m in MEMBERS[fam]]
     base_fam += [logit(FEATROUTE[fam].apply_one(x_fam))]
  z = concat over families (21 logit inputs, SAME order as training)
  p = META.apply_one(z)                                        # numpy tree traversal
  return float(clip(p, eps, 1-eps))
```

`labeled` unused in v1 (random default). Optional v2: Platt/calibration from `labeled` (Phase 7).

---

## 5. Component inventory

**Package-free members (train→export→numpy, each parity-gated G1):**
| Member | Converter status | Action |
|--------|------------------|--------|
| lgb, lgb_goss, lgbdart | `src/gbdt_member.py` ✓ (LightGBM dump→numpy, parity-checked) | reuse; confirm DART/GOSS inference = same tree walk |
| xgb | **MISSING** | write `src/xgb_member.py` (numpy traversal of `dump_model`/`trees_to_dataframe`; missing-value default dir) |
| cat | **MISSING** | write `src/catboost_member.py` (oblivious trees: float feature borders + leaf table; CatBoost CrossEntropy classifier) |
| et (ExtraTrees) | **MISSING** | write `src/forest_member.py` (sklearn tree arrays → numpy traversal, mean over trees) |
| knn | `src/knn_member.py` ✓ | reuse (PCA+int8 stored neighbors, numpy distances) |
| mlp, mlp2 | `src/mlp_member.py` / variants ✓ (torch) | reuse; ship torch state_dict |
| ridge | `src/stacker.py` / `logreg_member.py` ✓ | reuse (numpy dot) |
| fm | `src/fwfm_member.py` ✓ | reuse |
| irt | torch checkpoint (`export_submission.py` IRT-MLP) | reuse pattern; ship `.pt` |
| featroute (×3 fam) | 8× LightGBM group sub-models → logit-mean | reuse `gbdt_member` per group + a numpy logit-mean combiner |
| meta | LightGBM cross_entropy (21→1) | reuse `gbdt_member` |

**Feature artifacts to ship (per family), built offline, loaded at runtime (R2/R4):**
- encoder slug (→ `models.txt`); k-means centroids; cluster geometry params; NN neighbor table
  (PCA+int8 emb + per-subject passrate CSR `.npz`); mean-encoded tables (`m2_*`); subject/benchmark
  metadata lookups + scalers; pool/semcat are pure code (no artifact).

---

## 6. Phased plan (each phase ends on its gate; orchestrator verify-review-fix per phase)

**Phase 0 — Setup & pin (0.5 d).** Branch; pin the **3 family encoder slugs** from the actual feature-build
config/Drive; snapshot the exact AIDE winning roster per family (member list + hyperparams from journals);
stand up a Colab A100 build env (lgbm/xgb/cat/sklearn/torch). Deliverable: `configs/submission_roster.yaml`.

**Phase 1 — Encoder runtime spike (0.5 d) [R3/G5].** Load all 3 encoders; measure per-item encode latency
+ memory on the target runtime (L4/CPU). Decide: 3 encoders viable? If not, surface options to user
(fewer families w/ measured degradation). **Gate: G5 estimate.** *Go/no-go for full 3-family before heavy work.*

**Phase 2 — Unified feature module + parity (2 d) [R2/G0].** Refactor the offline feature build into ONE
`src/runtime_features.py` callable both offline (vectorized over corpus) and per-item at runtime; rebuild the
3 family parquets through it; assert identical to the existing AIDE parquets. **Gate G0** (runtime==offline
<1e-5). This is the highest-risk correctness phase.

**Phase 3 — Missing converters (2 d) [R5/G1].** Write + unit-parity-test `xgb_member.py`,
`catboost_member.py`, `forest_member.py` against freshly trained small models (synthetic + real features),
each **<1e-5** vs the package. Mirror `gbdt_member.py` (flat arrays, NaN/default-dir handling, `.npz`+meta).
**Gate G1 (per converter).**

**Phase 4 — Retrain roster + export (2 d) [R1/R6/G1].** In the export harness, retrain every member per
family on the Phase-2 features with fixed seeds; persist each model; export to package-free state; parity
each (G1). Rebuild featroute (8-group) + the 21→1 meta; export. Produce R0 = retrained stack OOF/secret.
**Gate G1 all members + G3 (R0 ≈0.4365, beats baseline).**

**Phase 5 — Assembly + size (1.5 d) [R4/G4].** Write `model.py` (module-scope load, the §4 flow),
`models.txt`, assemble `artifacts/` + `_pure/` modules; enforce whitelist-only imports (static scan);
trim to **≤65 MB** (PCA dim / int8 / prune neighbor tables). **Gate G4 (imports, size).**

**Phase 6 — End-to-end verification (1 d) [G2/G3/G4].** Run the **CPU validation harness**
(`validation_harness/`) on the holdout; assert stack parity G2 (<1e-4 vs R0), no-degradation G3 (secret
≤0.4370), runtime contract G4 (native float, no net, smoke passes). Produce the verification report.

**Phase 7 — (Optional) adaptive labeling (1 d).** `labeling.py` acquisition + `labeled`-based Platt
calibration in `predict`; ablate value. Only if Phase 1–6 land with budget headroom.

**Rough total:** ~10–11 working days of build (compute on Colab). Phases 1 and 2 are the go/no-go risk gates.

---

## 7. Files to create / modify

- **New:** `src/xgb_member.py`, `src/catboost_member.py`, `src/forest_member.py`, `src/runtime_features.py`,
  `configs/submission_roster.yaml`, the assembled `submission/{model.py, models.txt, artifacts/, _pure/}`,
  `tests/test_converter_parity.py`, `quality_reports/specs/2026-06-06_package-free-submission-spec.md`.
- **Reuse/extend:** `src/gbdt_member.py`, `src/fwfm_member.py`, `src/knn_member.py`, `src/mlp_member.py`,
  `src/stacker.py`, `src/export_stacked_submission.py`, `validation_harness/`.

---

## 8. Open decisions to confirm at phase boundaries (not blocking approval)

1. **Phase 1 result** may force fewer than 3 encoders (budget) — user decides on measured degradation.
2. **featroute at runtime** adds 24 sub-models × per-item — confirm it stays within budget or fold its
   contribution differently. (Featroute's measured gain is small, −0.0007; could be dropped if it busts budget.)
3. **labeling.py** (adaptive) — in or out for v1.

---

## 9. Immediate next step on approval

Phase 0 + Phase 1 (the encoder spike) in parallel — they are the cheapest and most decision-bearing.
The AIDE babysit loop continues independently (separate runtime; its 0.43653 artifact is the reference R0
target and stays protected).
