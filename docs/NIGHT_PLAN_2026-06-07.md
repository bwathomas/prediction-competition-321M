# NIGHT PLAN 2026-06-07 — autonomous IRT→canon→final-stack pipeline

## 🚢 EMBEDDING-INDEX QUANTIZATION BAKE-OFF (user: "quantized, reasonably sized, least lossy") 2026-06-07
Only nn-index features (nn_geometry 3 cols + nn_label ~13) need the train-emb index; cluster blocks need only
the full-precision centroids + full-precision query; irt_bag uses NO index. Query is full-precision at runtime;
only the INDEX is quantized. Bake-off (qwen, 300 q, recall@64 of true top-64 + nn_geom err):
- int8 per-row full-dim: 1.28GB, **recall 0.991**, nngeo 0.014  <- LEAST LOSSY
- PQ-512: 164MB, recall 0.825 | PQ-256: 84MB, recall 0.740  <- only viable SMALL option
- PCA: DISQUALIFIED — even PCA-2048 (98.6% var kept) recall only 0.589; neighbors live in low-variance dirs.
DECISION: int8-full if ~1.3GB OK (quantized AND near-lossless); else PQ-512 (164MB). PQ >> PCA at all sizes.
**3-FAMILY MULTIPLIER (decisive):** submission needs all 3 families' indices -> int8-full×3=3.8GB (infeasible),
PQ-512×3=492MB, PQ-256×3=252MB. -> PQ MANDATORY. COMMITTED: **PQ-512** (~492MB total, recall 0.82). BUILDING
the ship artifacts now: pqidx_<fam>.npz {codebook[512,256,8], codes[N,512] uint8, item_keys} -> nemotron
(qwen+nemo) pid 157915, lgai pid 165695. ADC decode: query full-precision, score=sum_m LUT[m,code].
BUILT + VERIFIED: pqidx_{qwen 174.8MB / nemotron 177.9MB / lgai 148.4MB}.npz (~501MB total). `src/pq_index.py`
(load_pq_index/pq_knn ADC). Verify (qwen, saved artifact): keys_aligned TRUE, recall@64 0.826 (==bakeoff).
⚠️ nn_geometry from PQ-APPROX sims is NOISY (local_density max-abs err ~0.98 outlier; LID large) — PQ dot is
approximate & we can't recompute exact sims without full emb. So the 3 nn_geometry cols are approximate under PQ.
⇒ DOWNSTREAM-OOF VALIDATION NOW NECESSARY: recompute etbig/mlp OOF with PQ-index nn feats vs full-index, check
final stack OOF moves negligibly (irt_bag unaffected; nn feats ~16/543; trees robust). If it DOES move, options:
bump M / ship int8-full for nn_geometry only / drop nn_geometry. Wire PQ into geom_runtime nn-path after.
TIEBREAK TODO: measure DOWNSTREAM final-OOF impact of PQ-512 vs full index (nn feats are ~16/543, trees robust,
irt_bag unaffected -> expect negligible) -> if negligible, ship PQ-512 small; else int8-full. Bake-off JSONs:
/content/quant_bakeoff.json, pq2.json, pcahi.json (on lgai tab).
ALSO: all 3 families' ship-etbig now numpy-verified (qwen/nemotron/lgai forest.npz ~4MB each, MAXDIFF ~9e-8).

## 🚢 RUNTIME GEOMETRY — DONE + FULL-BLOCK VERIFIED (2026-06-07)
`src/geom_runtime.py` (commit b613239, LOCAL — see push-blocked note) recomputes the full 525-col geometry
block for new embeddings, EXACT column order vs stored (cols_match_order TRUE), max_abs 0.0021 (only LID col),
mean 3.3e-7. Per-group also verified (centroid_distance/item_cluster exact). Centroids exported to
`DR/ship/ship_models/geom_<fam>_centroids.npz` (qwen done; TODO nemotron+lgai). Artifacts: centroids (~4.7MB)
+ unit-normalized train emb index (for nn-kNN; ship fp16 ~2.5GB or quantized cache.py). query emb must be
unit_rows-normalized; self/alias exclusion is key-based (no-op for genuinely-new items).
⚠️ PUSH BLOCKED: local->github connectivity DOWN (port 443 timeouts). 4 commits UNPUSHED locally incl
geom_runtime.py + irt_numpy.py (origin stuck 94c6261). Commits safe locally; geom_runtime.py was WRITTEN
DIRECTLY onto the lgai tab to run the verify. `git push` when connectivity returns. (Earlier "pushed" reports
were FALSE — `git push|tail` masked the exit code; fixed to check $?.)
TODO: export centroids nemo+lgai; embedding-index shipping decision (fp16 vs quantized); then full numpy
predict() = geom_runtime -> [etbig forest_predict + mlp + irt_bag] -> stacker.

## 🚢 RUNTIME GEOMETRY (user priority, 2026-06-07) — VERIFIED reproducible on new embeddings
The tree/mlp consume geometry feats (centroid_distance/item_cluster/cluster_geometry/nn_geometry) that were
precomputed from TRAIN item-embedding distribution. At runtime a cold item gives only its raw embedding ->
must recompute. ⚠️ centroids NOT persisted (fit fresh each derivation, seed=0). VERIFIED (qwen, lgai tab):
re-fit `fit_multi_kmeans(unit_rows(emb),{coarse:32,fine:256},seed=0)` (42s) REPRODUCES stored shards:
centroid_distance 0.0 (exact), item_cluster 0.0 (exact), cluster_geometry 9e-4, nn_geometry 1e-3 (LID float).
-> runtime geometry works. Recipe: `cluster_geometry_fast`(centroids+all_emb sizes) + nn-geo from
`bruteforce_knn` over train emb. Artifacts to EXPORT: centroids fine[256,D]+coarse[32,D] (~4.7MB);
cluster-size vector [256]; train item-embedding index (311k×4096 -- BIG -> quantize via cache_export.py for
nn-kNN, accept small nn_geometry drift, OR ship fp16 ~2.5GB). NEXT: write geom_runtime.py (assemble all 4
groups in EXACT training column order: GEOM_GROUPS=[centroid_distance,cluster_geometry,nn_geometry,item_cluster]
matching dense_full layout) + verify the full assembled block matches stored dense; decide emb-index shipping.
⚠️ qwen colab2 tab OOM-crashed (small CPU runtime, can't load 5GB emb) — kernel dead, cells return empty; not
needed (geometry runs on lgai/nemotron high-RAM tabs via shared Drive). Flag for user: restart that tab if wanted.

## 🚢 SUBMISSION-READINESS (user-directed, 2026-06-07) — numpy-only inference
GOAL: run the canon members with python+numpy only (Codabench has no cuML/GPU). Build+verify per member.
- **etbig (tree): numpy inference BUILT + VERIFIED** — `src/tree_numpy.py` (extract_forest from cuML
  `.as_treelite().dump_as_json()` -> flat node arrays; forest_predict = numpy traversal, mean of per-tree
  leaves, left if x<=thr). Verified numpy==cuML predict to **8.6e-8** (commit 59e3346).
- ⚠️ foldALL etbig HUGE (~33M nodes) -> impractical. FIX: **ship-sized etbig depth12/128trees**.
  **nemotron DONE + extracted + VERIFIED**: 895868 nodes, **forest.npz 4.0 MB**, numpy 0.078 ms/row,
  numpy==cuML **9.6e-8** (exact). -> tree-ship is SOLVED at 4MB/family. lgai ship-etbig pid 132514 (running);
  qwen pending (nemotron tab busy w/ mlp sweep, qwen tab GPU-less) -> launch when free. Forests saved to
  `DR/ship/ship_models/etbig_<fam>/forest.npz` (+ cuml_rf.pkl).
- **xgb-vs-etbig (user Q):** etbig already ships at 4MB exact, so NO need to switch. xgb only marginally
  cleaner build (native JSON, no treelite roundtrip) but predictively ~= etbig (0.98 corr; got 0 canon weight
  by collinearity) and switching costs a stack re-fit. DECISION: keep etbig.
- TODO check: ship-etbig (depth12/128) OOF ≈ big etbig (depth18/300) so the stack weight still holds — re-run
  its 3-fold OOF + compare; if drift, re-fit stack. THEN numpy for mlp (torch->numpy fwd) + irt_bag (theta/A/bw
  matmuls, avg 16 members) + GBM stacker (lightgbm->numpy/pure-python).
- mlp cold sweep (nemotron pid, bonus) may still be running; deprioritized vs submission.

## 🔬 POST-PIPELINE: generalization investigation (autonomous, 2026-06-07)
Done: LOBO (stack generalizes +0.0008), cold-probe (members anti-transfer; gap=domain cold-start),
OOD-detector embedding-distance FAILED. IN FLIGHT: **16-benchmark cold sweep on ETBIG** (the tree — the high-weight member; user: "use the tree,
irt is weak"). nemotron pid 112474 (sweep A, 8) + lgai 114659 (sweep B, 8), capped 100 trees/800k rows,
SHIP_HOLDOUT_BENCH per bench → cold preds to `DR/ship/coldsweep/nemotron/etbig_<bench>/preds/oof_preds.npz`,
BCE in coldsweep_{a,b}.json. ⚠️ CAVEAT: etbig's dense label-derived feats (cluster passrate/counts) are
cross-fitted per ITEM-fold, so a held-out benchmark's rows still see that benchmark's other-item label stats
=> cold number is OPTIMISTIC (floor on true degradation). irt_bag probe (afrimedqa+0.42/mmlupro+0.22/swebench
+0.09) was leak-free embedding-only; the two bracket the truth.
**ON COMPLETION run the analysis:** for all 16 benches compute cold BCE vs warm item-OOF BCE (degradation);
correlate degradation with candidate detectors — (1) benchmark frequency (n rows), (2) member-disagreement =
variance across the 3 families' WARM irt_bag preds per item, (3) embedding-distance (have it, failed). Pick best
detector; quantify max recoverable loss from per-bench optimal shrink→base-rate. Log to RESULTS LOG + IDEAS.

## ✅✅ PIPELINE COMPLETE (Stages A–E done) — 2026-06-07
- A: 3-variant IRT done (irt_bag won). B: stack eval logged. C: canon = mlp-LOO + etbig + irt_bag.
- D: all 9 canon members trained on ALL data (foldALL) in all 3 families, saved to `*_full_foldALL/models/`.
- E: final 9-member non-neg stack OOF = **0.42182** (= full 24-col pool; -0.01471 vs old ship 0.43653).
  Manifest `DR/ship/stack/FINAL_CANON_2026-06-07.json` (members+paths+weights). Old FINAL untouched.
- REMAINING (not started, needs user/explicit go): (1) export submission CSV = run foldALL models on TEST,
  logit-blend with manifest weights+bias (touches the shipped deliverable → back up _winners_snapshot/FINAL_* first).
  (2) Tier-1 ideas: LOBO CV + calibration (see docs/IDEAS_2026-06-07.md). DO NOT auto-launch new experiments
  without user direction — cron ticks should just keep tabs alive + report idle until then.



User asleep, **full authority granted** to decide the canon and train final models. Map: **colab2=qwen,
colab=nemotron, colab3=lgai**. DR=`/content/drive/MyDrive/prediction-competition-321M`. Branch
`feat/principled-export-pipeline`. Metric = mean soft BCE, honest **GroupKFold(item)** OOF (item cold-start).
**Never `pandas.to_csv` on colab2/3** (numpy/pandas ABI bug) — use the csv module + atomic os.replace.
Driver = detached subprocess `/content/irt3_driver.py`; status `/content/irt3_driver_<fam>.json`,
log `/content/irt3_full_<fam>.log`. See memory [[irt-3variant-run]].

## STAGE A — IRT 3-variant run (IN FLIGHT)
3 variants × folds 0,1,2 per family, code `c2f55a5` (OOM + grad-clip fixed). Variants:
- **irt2** (richer MIRT, K=32, profile-amortized θ + benchmark factor + condition difficulty) — UNSTABLE
  across families (qwen 0.514, nemo 0.500, lgai 0.665 on fold0); judge after full 3-fold OOF.
- **irt_lib** (5 factor sub-models → `p_irtlib__*` + logit-mean combo `p_full`).
- **irt_bag** (M=16 bagged basic-IRT; most stable, ~0.476 expected).
DONE when all 3 `irt3_driver_<fam>.json` show `finished` with irt2/irt_lib/irt_bag × folds 0,1,2 OK.
OOF written to `DR/ship/exp_loo/<fam>/<variant>_full_fold<f>/preds/oof_preds.npz` (position-aligned).

## ⚠️ INCIDENT — qwen tab (colab2) RECYCLED ~2026-06-07 (mid-irt_lib f2)
colab2 lost its runtime (host a83fc5a1715c, Drive unmounted, pc321 gone, no GPU). CANNOT remount headless →
**flag for morning user remount.** RECOVERABLE via shared Drive: qwen `irt2 f0/1/2` + `irt_lib f0/1/2` OOFs
already SAVED on Drive (clean c2f55a5). MISSING for qwen: **irt_bag f0/1/2** (the `irt_bag_full_fold0` on Drive
is the stale M=2 smoke). RECOVERY: once nemotron or lgai frees (after its irt_bag finishes), run qwen irt_bag
f0/1/2 there — SHIP_FAMILY=qwen reads qwen embeddings from the shared Drive, so no qwen tab is needed. Same for
Stage D qwen-family canon training: run on a freed tab. Do NOT block the whole pipeline on the dead tab.

## STAGE B — stacking eval (after A)
⚠️ The original `master_stack.py` was LOST with the qwen recycle — RECOVERED + extended as repo
`scripts/ship/master_stack_eval.py` (commit d4d70d8). Run it on ANY live tab (git pull first):
`cd /content/pc321 && git fetch origin feat/principled-export-pipeline -q && git checkout -B
feat/principled-export-pipeline FETCH_HEAD`, then `python scripts/ship/master_stack_eval.py` (detached).
It reports base vs base+{irt,irt2,irt_lib,irt_bag,knn} (linear + non-neg), full-data non-neg weights for
base+best-irt (-> the CANON = positive-weight members), and per-member drop-importance. Writes
`/content/master_stack_eval_result.json` + mirrors to `DR/ship/stack/`. Needs all 3 families to have each
model (gated by have()); qwen irt_bag must finish first. It fits the linear + non-neg master stack over each
family's base member pool. Append the new IRT columns (irt2 `p_full`, irt_lib `p_full` + 5 `p_irtlib__*`, irt_bag
`p_full`) to the pool; re-run non-neg + linear + leave-one-out drop-importance vs the current irt baseline.
Report each variant's stack-lift + drop-Δ. **Append to `docs/OVERNIGHT_master_2026-06-06b.md` RESULTS LOG.**
Prior 4-way: base 0.42153; +IRT 0.42041 (+0.0011, earns a seat); +kNN +0.0002 (does not). Non-neg base
weights: only **mlp 0.81 + etbig 0.26** nonzero; xgb/dae/cnn1d/logreg/fm = 0.

## STAGE C — decide the CANON (unilateral)
Canon = members with POSITIVE weight in the final non-neg linear stack. Baseline expectation:
**mlp-LOO (stacked_oof) + etbig + best-IRT-variant**, per family. Add a model only if drop-importance /
non-neg weight shows it contributes; drop redundant ones (kNN, logreg, fm, cnn1d, dae likely out). Pick the
single best IRT variant (by stack-lift). Document the choice + rationale in the RESULTS LOG.

## STAGE D — train the CANON on full data, ALL 3 families  [IN PROGRESS]
**Canon decided (Stage C):** mlp-LOO + etbig + **irt_bag** (the IRT winner). knn/irt2/irt_lib/dae/cnn1d/
logreg/fm dropped (zero non-neg weight). User directive: train each on ALL data, NO oof, once per model;
mlp stays LOO.
**Mechanism:** new `TRAIN_ALL` mode (`SHIP_OOF_FOLD=-1`, commit c8e608f) — fits on all 4.5M rows, no holdout,
saves the model, skips OOF scoring. Driver `/content/trainall_driver.py` (per-model SHIP_MODE via
`TRAINALL_SPECS=mlp:loo,etbig:full,irt_bag:full`), detached, status `/content/trainall_<fam>.json`. Models →
`DR/ship/exp_loo/<fam>/<model>_full_foldALL/models/` (etbig: cuml_rf.pkl; irt_bag: members/m00..15.pt; mlp:
full + loo__*). Smoke-validated all 3 paths on nemotron.
**Launched:** nemotron (colab) + lgai (colab3) real runs. **qwen-family STILL PENDING** — qwen tab is GPU-less
(recycle dropped it to CPU), so run qwen-family TRAIN_ALL on whichever GPU tab frees first (SHIP_FAMILY=qwen
reads qwen embeddings from shared Drive): write trainall_driver if absent, launch with TRAINALL_SPECS, detached.
~70 min/family (mlp ~40 / etbig ~20 / irt_bag ~8).
(orig:)
`SHIP_ROW_SOURCE=full` (all 4.5M rows, honest 3-fold OOF). Per canon member:
- **mlp**: `SHIP_MODEL=mlp SHIP_MODE=loo` → p_full + p_loo__* + `stacked_oof` (the honest GBM stack of its LOO members).
- **etbig**: `SHIP_MODEL=etbig SHIP_MODE=full`.
- **irt-winner**: `SHIP_MODEL=<irt2|irt_lib|irt_bag> SHIP_MODE=full`.
Reuse existing artifacts if present AND current-code; re-run what's missing/stale. One family per tab, detached.
(Most full-model OOFs already exist from prior sessions; the new piece is the chosen IRT variant + any re-fits.)

## STAGE E — final linear stack + ship
Fit the final non-neg linear stack over the canon across all 3 families (the simple robust combiner the
STRATEGIC note favors over the flexible tree-meta). Save artifacts **NON-DESTRUCTIVELY** — back up any existing
`_winners_snapshot/FINAL_*` before writing. Report final OOF + the chosen weights. Then surface for the user.

## RULES
- Advance only the next INCOMPLETE stage each tick; never repeat a completed stage.
- run_bg / detached subprocess for ALL heavy cells; never synchronous heavy work.
- A recycled tab (Drive unmounted / pc321 gone) CANNOT be remounted headless → flag for morning, keep other 2 going.
- Commit+push every code change here; redeploy to tabs via git fetch/checkout.

## 🔬 PQ-512 DOWNSTREAM-OOF VALIDATION — RESULT (2026-06-09, nemotron fold0, 1.52M rows)
`scripts/ship/pq_downstream_validation.py` (51fce57). Models FIXED (full-feat-trained fold0); OOF rows'
17 nn cols recomputed from the PQ-decoded index (== runtime ADC exactly). Rebuild check: rebuilt baseline
== stored etbig p_full to BCE 7 decimals (max|Δp| 0.006). Result JSON: DR/ship/stack/pqval_nemotron_fold0.json.
- FEATURES: nn_geometry badly distorted (mean|Δ| 1.77); nn_label mean|Δ| 0.025; counts ±1.1 neighbors.
  PQ sims SATURATE the alias threshold (1-1e-6) across near-duplicate groups → runtime nn-path needs
  deep retrieval buffers (GEO 2048 / LAB 1024 used here; default search_buffer=2 CRASHES under-retrieved).
- etbig:       0.44008 → 0.44244  (Δ +0.00236, pred corr 0.990)
- mlp stacked: 0.42113 → 0.42326  (Δ +0.00213; mlp p_full only +0.00025 — LOO members amplify)
- CANON STACK (only nemotron's 2 cols swapped): 0.41939 → 0.42023  (Δ +0.00085)
VERDICT: not "highly damaging" but NOT negligible: extrapolated all-3-family cost ≈ +0.002 BCE ≈ ~15% of
the canon's gain over old ship (-0.0147). lgai+qwen fold0 validations IN FLIGHT (sequential, A100, pid 17340).
MITIGATIONS if confirmed: (a) retrain members on PQ-derived features (train/serve consistency — likely
recovers most), (b) drop the 3 nn_geometry cols, (c) PQ-1024/OPQ (~1GB), (d) re-check real Codabench size
cap — int8-full (recall 0.99) ends the issue if ~4GB allowed.
