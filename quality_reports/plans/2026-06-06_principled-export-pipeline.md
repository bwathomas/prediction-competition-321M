# Plan — Principled package-free export pipeline for AIDE's 3-embedding ensembles

**Status:** APPROVED-PENDING · **Date:** 2026-06-06 · **Work repo:** `/home/akhaemenid/projects/prediction-competition-321M`
(Plan lives here in the stub repo because the harness pins it here; on approval, copy to
`prediction-competition-321M/quality_reports/plans/2026-06-06_principled-export-pipeline.md` and implement there.
Live working notes already in `prediction-competition-321M/docs/SHIP_PLAN_3WAY.md`.)

## Context
Three AIDE-ml agents each built a stacked ensemble on a different embedding family (qwen=`Qwen3-Embedding-8B`,
nemotron=`nvidia/llama-embed-nemotron-8b`, LGAI=`annamodels/LGAI-Embedding-Preview`); the 3 family winners were
LightGBM-meta-stacked to secret soft-logloss 0.43653. We want a **principled, reusable pipeline** that re-creates
those final ensembles as **package-free Codabench submissions** (runtime imports only torch/numpy/transformers/
safetensors/huggingface_hub/stdlib), with **three deliberate upgrades over AIDE**: (1) full subject+benchmark
**metadata in every base learner**, (2) **3× bagging** per learner, (3) honest **3-fold GroupKFold(item_key) OOF**
for accurate stacking + assessment. No deadline — correctness over speed.

**Decisions (user, 2026-06-06):** (a) AIDE base-learner models were never persisted → **retrain the roster to spec**;
"no-degradation" is defined vs the retrained baseline **R0** (target ≈0.4365 secret, must beat the organizer baseline),
not the exact AIDE numbers. (b) Layer-2 meta = **LightGBM cross_entropy**, GroupKFold(item_key), shipped package-free
via `gbdt_member` numpy traversal.

## Roster (per family, retrained; locked order = meta-input column order)
`lgb · lgb_goss · lgb_dart · xgb · cat · ExtraTrees · knn · mlp(3-bag) · fm · irt(K-dim) · featroute`
(qwen/mistral/llama vary slightly per the snapshot; declare the exact per-family list in `scripts/ship/roster.py`).
featroute = nested 8-group LightGBM → logit-mean over groups {nn_label_derivatives, cluster_passrate, cluster_subject,
counts_subject, centroid_distance, cluster_geometry, nn_geometry, item_cluster}.

## Architecture
2-layer stack, per family: Layer-1 = the roster (each member 3-bagged, trained under 3-fold OOF, on a shared
metadata-augmented feature matrix); Layer-2 = LightGBM cross_entropy meta over the `3 × n_members` honest-OOF columns.
Every Layer-1 member and the meta compile to package-free numpy/torch; the shipped `model.py` recomputes the SAME
per-item features live (NN via shipped train index, metadata via shipped tables, clusters/pool pure-code) and runs the
roster → meta → clipped float.

## Stacking discipline (frozen stacker + full-data base models) — load-bearing
K-fold is ONLY a device to (i) generate honest OOF base-learner predictions that train the meta, and (ii) assess
generalization. It is NOT how the shipped models are trained.
1. **Learn the stacker on OOF.** Per fold, base learner trained on K−1 folds predicts the held-out fold → honest OOF
   columns; fit the LightGBM meta on the `3×n_members` OOF matrix. Fold models are discarded after writing OOF.
2. **Ship full-data base learners.** Retrain every member (all 3 bags) on ALL train rows → strongest base models.
3. **Freeze the meta** from step 1; apply it to the full-data models' outputs at inference. Shipped `model.py` =
   full-data roster → frozen LightGBM meta → clipped float. Secret-holdout assessment (R0) uses this exact path.
Caveat (benign/standard): meta trained on slightly weaker OOF preds, applied to slightly sharper full-data preds — a
small distribution shift, tolerated because the meta learns relative combination, not absolute calibration. Fallback if
it ever bites: use the K fold-models' averaged preds as the shipped base layer (matches OOF distribution, marginally
weaker). Default = full-data refit + frozen meta.

## New modules (reuse-maximal; ~5 new files + 1 in-place schema extension)
- `src/member_features.py` (IN-PLACE, schema v3, additive): add a **benchmark-metadata block** (bench_cat one-hot +
  bench_num value/missingness), mirroring the existing subject/`cond` blocks; extend `bc_redacted` to zero it for
  cold benchmarks. This is gap (a) — metadata into the GBDT/xgb/cat/forest/knn/fm/logreg shared matrix.
- `src/member_features_meta.py` (NEW, offline-only): `build_shared_matrix(...)` — the only place pandas/
  `MetadataPreprocessor.encode_subject/encode_benchmark` is touched; keeps `member_features.py` runtime-pure.
- `src/bagging.py` (NEW): `train_bagged_member(fit_fn, seeds=(0,1,2), bag_kind)`, `bagged_predict_logit_mean`,
  `BaggedMemberState` (delegates save/load to K sub-states; runtime loads K + logit-averages). Gap (b).
- `src/featroute_member.py` (NEW): 8× `gbdt_member` sub-models over feature-group slices → numpy logit-mean. Runs
  inside the per-fold loop (its sub-models include label-derived groups).
- `src/irt_member.py` (NEW): lift the 4 IRT classes from `export_submission.py` into a standalone torch member
  (`fit_irt_member` → `IRTMemberState`: per-subject θ `.npy` + item-tower state_dict; `apply_one/apply_batch`).
- `scripts/ship/roster.py` (NEW) + `configs/submission_roster.yaml`: per-family roster spec (members, hyperparams,
  bag config, locked order, encoder slug).
- `scripts/ship/train_roster_oof.py` (NEW): the heart — 3-fold OOF × 3-bag × full roster, per family; OOF accumulate;
  full-data refit for shipping.
- `scripts/ship/fit_meta.py` (NEW): LightGBM cross_entropy meta over OOF; GroupKFold internal val.
- `scripts/ship/export_full_roster.py` (NEW): generalize `export_stacked_submission.py`'s 4-member bundler to the full
  roster (parametric `model.py` over `roster.py`; ship members + bags + featroute + irt + meta + per-family `nn_infra`
  + metadata tables + `models.txt`; reuse `_copy_pure_modules`/`_strip_faiss_imports`/`audit_runtime_imports`).

## Reused as-is (no change)
`src/oof_folds.py` (`make_item_grouped_folds(n_folds=3)` + leakage probes), `src/oof_pipeline.py`
(`OofPredictionAccumulator`), all `src/{gbdt,xgb,catboost,forest,knn,logreg,mlp,fwfm}_member.py` (+ `stacker.py`,
`nn_calibration.py`) — all parity-tested <1e-5; `src/metadata_features.py` (`MetadataPreprocessor`); `aide/features/*`
(`FoldFeatureStore`, `driver.py`, `derive_*`) with its authoritative FOLD_INVARIANT vs LABEL_DERIVED classification;
`scripts/ship/{nn23_runtime,nn_infra_prep,metadata_tables}.py`; `validation_harness/`, `scripts/smoke_test_submission.py`.

## Phased plan (each phase ends on its gate)
- **P0 Roster pin + scaffold.** `roster.py` + yaml; pin 3 encoder slugs; build env. **Gate:** folds + 3 asserts pass on real 264k item_keys.
- **P1 Metadata schema extension + parity.** Add benchmark block to `member_features.py`; `build_shared_matrix`.
  **Gate G-META:** schema round-trips; batch==one-row <1e-6; no label-derived col in a FOLD_INVARIANT position.
- **P2 New members.** `bagging.py`, `featroute_member.py`, `irt_member.py` + unit tests. **Gate:** apply_one==apply_batch<1e-6; `irt_member`==`export_submission` IRT <1e-4.
- **P3 Fold-scoped features + leakage.** Per-fold label-derived build on `fold.train_item_keys`; FOLD_INVARIANT shared.
  **Gate G-LEAK:** row-partition + NN-in-fold-train + **shuffled-label control** (permuted-y member must not beat the prior). *Most important gate.*
- **P4 Train roster (3-fold OOF × 3-bag) + meta.** Per family; accumulate OOF; full-data refit; LightGBM meta over `3×n_members`.
  **Gate G3:** full OOF coverage/no-NaN; R0 OOF ≤~0.4300, secret ≤~0.4370, beats baseline; meta optimism gap ≤0.03 nats.
- **P5 Compile + parity.** Each member/bag/featroute/irt/meta → package-free; compare on holdout.
  **Gate G1+G2:** per-member <1e-5 (trees/linear/fm) / <1e-4 (torch); stack parity <1e-4 vs R0 (prob space).
- **P6 Bundle + runtime parity + e2e.** `export_full_roster.py` assembles bundle; verify runtime==offline features <1e-5;
  run `validation_harness` + smoke on full 264k. **Gate G0+G4:** feature parity <1e-5; import audit clean; native float; no net; e2e green; final secret == R0 within tolerance.
- **P1b (parallel) Encoder spike** `scripts/ship/encoder_spike.py`: measure per-item encode ×5000 for 3 encoders → go/no-go on 3-encoder runtime budget (fallback: fewer families w/ measured degradation, user decides).

## Metadata into EVERY base learner (the key detail)
Two member classes: metadata-aware already (`mlp` via `member2_metadata_mlp`, `irt` via bc embedding — feed ids only)
and **shared-matrix consumers** (gbdt/xgb/cat/forest/knn/fm/logreg + featroute) which today get subject metadata + `cond`
but **no benchmark metadata** → fix by the additive benchmark block in `member_features.py` (v3). **Leakage discipline:**
benchmark *metadata joins* are static (FOLD_INVARIANT) → fit on full corpus, shared across folds, no leak; only
y-aggregates (`groupby_benchmark_metadata`, `mean_encoded_*`) are per-fold (handled in the aide feature-store path, not
in `member_features.py`). Enforced by G-LEAK.

## Runtime feature recomputation (how runtime == offline)
Single serialized `MemberFeatureSchema` (v3) reloaded at runtime (column order = source of truth). Per family: encoder
(`models.txt`) → embedding; `nn23_runtime` over shipped `nn_infra/<fam>/` (train index + passrate CSR + conditional ctx
+ centroids) for the 23-dim NN block; `metadata_tables` resolves subject/benchmark ids + `bc_redacted_flag`; clusters/
pool pure-code. `build_member_features_one` reconstructs the identical row vector → roster `apply_one` → meta `apply_one`.

## Verification
| Gate | Check | Tol | Harness |
|---|---|---|---|
| G-META | benchmark block round-trip; batch==one-row | <1e-6 | `tests/test_member_features*.py` |
| G-LEAK | row-partition; NN-in-fold-train; shuffled-label control | exact/chance | `oof_folds` asserts + new `tests/test_roster_oof_leakage.py` |
| G1 | each package-free member == package | <1e-5 / <1e-4 | extend `tests/test_converter_parity.py` (+bag/featroute/irt) |
| G2 | full-roster stack == R0 | <1e-4 | extend `tests/test_export_stacked_submission.py`; new `tests/test_full_roster_parity.py` |
| G3 | R0 OOF/secret beats baseline; optimism gap | — | `fit_meta.py` + `eval.py` |
| G0 | runtime feature == offline | <1e-5 | extend `tests/test_metadata_runtime_roundtrip.py` |
| G4 | whitelist imports; ZIP≤65MB; native float; no net | pass | `export_full_roster.audit_runtime_imports` + `scripts/smoke_test_submission.py` |
| e2e | full-val 264k smoke | green | `validation_harness/scripts/run_validation.py` |

## Top risks → mitigations
- **R1 Retrain fidelity:** gate vs R0 (fixed seeds, report R0−0.43653 drift), not exact AIDE preds.
- **R2 Metadata OOF leakage (the upgrade could introduce it):** lean on `store.py` FOLD_INVARIANT vs LABEL_DERIVED; only static joins in `member_features.py`; G-LEAK shuffled-label control.
- **R3 Runtime feature parity:** one shared schema both sides; G0 <1e-5 before any e2e claim; reuse byte-verified `nn23_runtime`/`metadata_tables`.
- **R4 3-encoder runtime budget (#1 latency/mem risk):** P1b spike *before* heavy work; all 3 via `models.txt`; fallback to 1–2 families (measured) if over budget.
- **R5 Bag×fold coverage bugs:** `OofPredictionAccumulator` guards double-write/NaN; `assert_row_idx_partition`; bag logit-mean order-invariant.

## Compute / ops
Build on Colab A100 (3 runtimes already wired: colab2=qwen, colab=nemotron, colab3=LGAI; Drive mounted; repo at
`/content/pc321`). Deploy via git push branch → `git pull` on each runtime. Heavy work via `run_bg`+poll (never
synchronous `run_code_cell`). Three families train in parallel across the 3 A100s. Floor submission already shipped
(`ensemble_qnl_3way.zip`) as a fallback while this pipeline is built.
