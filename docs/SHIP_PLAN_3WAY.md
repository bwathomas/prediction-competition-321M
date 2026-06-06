# SHIP PLAN — 3-embedding ensemble submission (LIVE, 2026-06-06)

**Goal:** ship a valid Codabench submission = ensemble of 3 embedding families
(qwen8b + nemotron + LGAI), package-free, in the PROVEN structure. Safe-ship-first,
then layer diversity. User deadline: ASAP (~2h, already eroded).

## DECISIONS (user, locked)
- Stop ALL AIDE (done — 3 runtimes freed). Snapshots safe on Drive.
- Safe ship FIRST (assemble finals), diversity in PARALLEL, fold in if ready.
- NN features: sent submodels use 8-dim; current models need 23-dim. UPDATE sent
  submodels to emit 23-dim, keep back-compat (cols 0..7 == legacy 8 → structural).
- NO k-fold / no honest OOF (no time). TRAIN ONE MODEL EACH on as much data as possible.
- Stacking method OPEN given no OOF → default to fixed logit/FWLS combiner (proven),
  or fit combiner on a single val split. Tree-meta is stretch (needs OOF → risky w/o it).

## COMPUTE (all live, Drive mounted, AIDE killed)
- colab2 = qwen  host 66533af7902a  A100-40GB (idle)  repo /content/pc321
- colab  = nemotron host 87e3dc14ce39 A100-80GB (idle) repo /content/pc321
- colab3 = LGAI   host 1b1fce795001  A100-80GB (idle)  repo /content/pc321
- Drive bridges flaky across turns; reconnect via open_colab_browser_connection (bounded=safe).
- run_bg ALWAYS for heavy cells; never synchronous run_code_cell on heavy work.
- Reusable diagnostic cells: colab2 `lIYdn1woOS1n`, colab `n1mDQLlVoB47`, colab3 `ORCjadYmoAqv`.
  (add_code_cell is REJECTED by these MCP servers — use update_cell + run_code_cell.)

## DRIVE LAYOUT (drive_root DR = /content/drive/MyDrive/prediction-competition-321M)
- Embeddings: DR/embeddings/<slug>/{items,subjects}.parquet (+ .npy fast path)
  - qwen     slug = Qwen__Qwen3-Embedding-8B
  - nemotron slug = nvidia__llama-embed-nemotron-8b   (non-causal! bidir; pooling=mean)
  - LGAI     slug = embedding_cache_lgai_preview_fa2/annamodels__LGAI-Embedding-Preview (NESTED)
- Labels: DR/prepared_datasets/measurement_db_prepared_fca427c6cc3da482.parquet
- 594-feat shards: DR/features/{qwen,llama,mistral} + INDEX.json
- NN-feature infra (23-dim, CURRENT): DR/artifacts/nn_features/ (faiss training_index, passrate_csr.npz,
  conditional context), cluster_centroids.npy, item_clusters.parquet, item_features/pool_features.parquet
- Trained qwen ensemble: /content/drive/MyDrive/qwen8b_four_member_stacked_cache/v1
  (fwfm_state*, gbdt_state*, X_train_dense/X_val_dense, *_oof_fold* pkls)
- AIDE per-family OOF+holdout preds (REFERENCE ONLY, not runnable): DR/aide/_winners_snapshot/{qwen,llama,mistral}/
- Row alignment STAGED: DR/ship/rows/ = _tr_item,_tr_subj,_ho_item,_ho_subj,_fr_oof,_fr_hold .npy
  (train N=264350, holdout N=135650; canonical row order matching all existing preds)

## ASSETS (Windows, NOT yet on Drive — Colab can't read WSL)
- C:\Users\benja\Downloads\submission\ensemble_3way_logit_fwls.zip  (qwen8b+SFR-mistral+qwen4b finals; PROVEN structure + assembler)
- C:\Users\benja\Downloads\submission\nemotron_trc_v5.zip  (FINAL nemotron head, correct bidir plumbing + NEW calibrator)
- C:\Users\benja\Downloads\submission\LGAI_fixed.zip       (FINAL LGAI head, correct plumbing + NEW calibrator)
- extracted locally at /tmp/fwls_inspect/{qwen8b,mistral,qwen4b,nemotron_trc_v5,LGAI_fixed}
- runtime: torch/transformers/numpy/safetensors/huggingface_hub ONLY. NO sklearn/lgb/xgb/scipy/faiss/pandas.
  Trees ship as numpy (src/{gbdt,xgb,catboost,forest}_member.py); ZIP <=65MB (encoders via models.txt).
- Proven combiner: model.py logit-average (geometric mean of odds) over submodels loaded via importlib.
- Calibrator: follow nemotron_trc_v5 / LGAI_fixed pattern (NOT older ensemble_3way). default_calibrator=identity.

## KEY CODE (repo /content/pc321 == local)
- src/mlp_member.py::fit_mlp_member(labels, item_emb_unique,row_to_uniq, dense_X, subject_ids,...) -> MlpMemberState
- src/nn_features.py: NN_FEATURE_DIM=23, NN_FEATURE_NAMES, _aggregate_nn_features (cols0..7=legacy8),
  build_conditional_passrate_context, TrainingNNIndex, build_passrate_table
- src/export_submission.py (277KB): renders CURRENT 23-dim runtime model.py for one head (+encoder+indexer+NN)
- src/export_stacked_submission.py: bundles multi-member package-free (wraps export_ensemble_run)
- notebooks/ensemble_builder.ipynb (42 cells): extracts per-family bundles -> stacks up to 6 -> proven structure
- notebooks/qwen8b_four_member_stacked.ipynb: trains qwen members (M1 IRT-MLP, M6 FwFM, M7 marg-MLP, M8 emb-MLP, M2 GBDT) w/ OOF pipeline; CFG["encoder"]["model_id"] knob repoints family

## EXECUTION PLAN
A. [done] kill AIDE; stage rows to DR/ship/rows.
B. [USER] upload the 3 zips to Drive (e.g. MyDrive/ship_bundles/) so Colab can assemble. OR rebuild from artifacts.
C. Update nemotron & LGAI submodel.py: 8-dim NN -> 23-dim (ship current nn_features runtime path; back-compat slice 0:8 for their heads).
D. Train ONE model each on ALL data (no k-fold): nemotron MLP + LGAI MLP (+ optional cheap GBDT) on 23-dim feats. run_bg on colab/colab3.
E. Assemble qwen8b-ensemble + nemotron + LGAI (+ new MLPs) via proven structure; combiner = logit-FWLS (no OOF) or val-split-fit; calibrator = new pattern.
F. Validate: whitelist-only imports, ZIP<=65MB, smoke on validation_harness, native float, no net. Ship.

## STATUS
- A done. B pending (need zips on Drive). C/D/E/F not started.

## UPDATE (user, 2026-06-06): MLP recipe CHANGED
- Train MLPs **the AIDE way** (rich AIDE feature matrix via FoldFeatureStore.assemble) **+ subject + subject-metadata
  + benchmark + benchmark-metadata** (fit_member2_metadata_mlp channels: subject/family/macro_family/org/bc/topic ids
  + numerics). AIDE MLPs were MUCH stronger than the earlier qwen8b ones.
- NO k-folding → single model. **Item-split val for stacking** = reuse nf3 folds: train on folds{0,1}, validate on
  fold2 (item-disjoint, leak-free) to get honest val preds → fit stacker/combiner weights on those.
- Do it for **ALL THREE** families (qwen, nemotron, LGAI).
- OPEN RISK: are HOLDOUT (135650) features pre-derived (so the AIDE-feature MLP can predict holdout)? If yes → use
  AIDE feats. If not readily available → fall back to embedding+metadata MLP (needs only embeddings+metadata, both
  on Drive) and flag the downgrade. Swarm to resolve.
- APIs: aide/features/store.py::FoldFeatureStore.assemble; groups = nn_label_derivatives, cluster_passrate,
  cluster_subject, counts_subject, centroid_distance, cluster_geometry, nn_geometry, item_cluster.
  src/member2_metadata_mlp.py::fit_member2_metadata_mlp; src/metadata_features.py; src/mlp_member.py::fit_mlp_member.
- Swarm v1 done: scripts/ship/{nn23_runtime.py (23-dim NN, byte-verified), train_family_mlp.py (emb-MLP, dense OFF),
  assemble_3way.py (proven structure + logit-FWLS + optional val-weight fit)}.

## CORRECTION (user, 2026-06-06): AIDE-feature path IS viable — prep the NN infra
- The swarm's "holdout AIDE feats underivable" verdict was WRONG framing. The SUBMISSION computes the 23-dim NN
  features LIVE at predict-time from a shipped train-neighbor index (exactly what the sent submodels already do, 8-dim).
- PLAN: PREP per-family neighbor infra ONCE ("learn the nearest neighbors"): load family train item embeddings →
  build TrainingNNIndex (faiss/numpy) over train items → build passrate CSR (build_passrate_table) +
  build_conditional_passrate_context → k-means centroids. Save to DR/ship/nn_infra/<fam>/.
  qwen infra exists at DR/artifacts/nn_features (+cluster_centroids); BUILD for nemotron + LGAI in their emb space.
- Then COMPUTE 23-dim AIDE feats (src/nn_features.py runtime path: _aggregate_nn_features + cluster/centroid/pool +
  metadata) for train folds{0,1}, fold2 (honest val), AND holdout (135650) — all cheap once the index exists.
- TRAIN fit_mlp_member on folds{0,1} dense_X=AIDE feats+metadata → predict fold2 + holdout. SHIP the nn_infra in each
  submodel bundle; runtime nn23_runtime.py aggregates per-item live. USE_AIDE_FEATS=True (enabled by the prepped infra).
- FLOOR submission already shipped: /mnt/c/Users/benja/Downloads/submission/ensemble_qnl_3way.zip (94.6MB, valid).

## PIVOT (user, 2026-06-06): NO RUSH — principled export pipeline (plan APPROVED)
Deadline dropped. Goal = principled reusable package-free export of AIDE's final 3-embedding ensembles, with 3 upgrades:
full subject+benchmark METADATA in every base learner, 3x BAGGING, honest 3-fold GroupKFold OOF. Decisions: RETRAIN to
spec (R0 baseline ~0.4365, models were never persisted); meta = LightGBM cross_entropy. Stacking discipline: k-fold ONLY
trains the frozen meta; ship FULL-DATA base learners + frozen meta. Full plan:
quality_reports/plans/2026-06-06_principled-export-pipeline.md. Phases P0-P6 (+P1b encoder spike). P0-P2 (roster,
member_features v3 benchmark block, bagging/featroute/irt members) authoring now with unit gates; P3/P4 = Colab training
on the 3 A100s. Floor submission ensemble_qnl_3way.zip already shipped as fallback.
