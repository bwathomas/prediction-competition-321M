# WARM START — principled export pipeline (resume after context clear)

**Snapshot: 2026-06-06 ~19:40.** Read this FIRST, then `docs/OVERNIGHT_master_2026-06-06b.md`
(detailed tick-by-tick log + RESULTS LOG). Work repo:
`/home/akhaemenid/projects/prediction-competition-321M`, branch `feat/principled-export-pipeline`
(all code pushed to origin/GitHub `bwathomas/prediction-competition-321M`).

---

## 🏆 THE DELIVERABLE (solid, reproducible)
**2-layer stack of 15 full models = secret-equivalent OOF soft-logloss 0.42519** (honest 3-fold
GroupKFold(item) OOF, on the FULL ~4.5M-row dataset). Beats the old shipped 18-learner stack
(0.43653) by 0.011 and the best single model (nemotron.xgb 0.43789) by 0.013.
- 15 models = {logreg, xgb, et, fm, mlp} × {qwen, nemotron, lgai} full models.
- Stack = LightGBM cross_entropy meta over the 15 per-model OOF columns (logit space),
  GroupKFold(item, 5-fold). logit-mean blend of the 15 = 0.43331.
- Reproduce: the stack script is inline in the qwen tab's history; report saved at
  `/content/stack_report.json` (qwen runtime). To recompute: load each
  `DR/ship/exp_loo/<fam>/<model>_full_fold<f>/preds/oof_preds.npz` (keys oof_items, oof_subj,
  oof_y, p_full), concat folds per (fam,model), align by position (row order identical across
  models per fold), LightGBM meta GroupKFold(5).

### Full-model NLLs (mean / 3 folds)
| | qwen | nemotron | lgai |
|---|---|---|---|
| xgb | 0.4472 | **0.4379** | 0.4420 |
| et | 0.4520 | 0.4455 | 0.4475 |
| mlp | 0.4605 | 0.4519 | 0.4467 |
| fm | 0.4576 | 0.4496 | 0.4513 |
| logreg | 0.4694 | 0.4621 | 0.4642 |
(T2: lgai cnn1d 0.4753 — weakest; dae/ft partial. xgb dominates; rest add decorrelated diversity.)

## GOAL / ARCHITECTURE (user's design)
Hierarchical per-archetype feature-dropout ensemble, 3 embedding families:
1. **Layer-0 full models** — every archetype trained fully on all features. T1: logreg, xgb, et,
   fm, mlp (DONE, 15/15 per family). T2: cnn1d, dae, ft_transformer (full models partial; cnn1d
   validated; OK to truncate ft).
2. **Layer-1 dropout LIBRARY** — per (family×archetype), M random-subspace members (rho~U[0.3,0.9]
   per member). ⚠️ STUCK — see morning action.
3. **Layer-2 greedy-ES** — Caruana ensemble selection per archetype over cached member OOF
   (`scripts/ship/greedy_select.py`, exists; extend for per-archetype + hierarchical stack).
4. **Layer-3 tree stack** — LightGBM stack archetypes within family, then families (A→B), and a
   flat stack of all (C); compare. (The 0.42519 above is the flat full-model stack — Layer-3 over
   full models only, no libraries yet.)

## COMPUTE (3 Colab A100s, all 167GB high-RAM, Drive mounted)
- **colab2 = qwen** (cell `lIYdn1woOS1n`)
- **colab  = nemotron** (cell `n1mDQLlVoB47`)
- **colab3 = lgai** (cell `OkodTZyfxfwo`)
- Family alias: AIDE names qwen/mistral/llama ↔ ship names qwen/nemotron/lgai. Embeddings/features
  on shared Drive `DR=/content/drive/MyDrive/prediction-competition-321M`; repo at `/content/pc321`.
- Reconnect bridges (bounded=safe): ToolSearch `select:mcp__colab*__open_colab_browser_connection`
  then run_code_cell/update_cell/get_cells. A **10-min cron (f1d71779)** drives keepalive/monitor
  ticks (session-only — dies if this Claude session exits).

## 🔴 MORNING ACTION — LIBRARY PHASE IS STUCK (my error)
Repeated accidental double-launches (re-running launch cells) left MULTIPLE library drivers per tab
(nemotron 4, qwen 2, lgai 2). Each library run's assembly loads the full 4.5M rows (~37GB), so
several concurrent runs OOM each other → ZERO completed libraries after ~30min; nemotron
OOM-thrashing. Subprocess isolation (`run_one.py`) kept the KERNELS + Drive alive.
**Clean fix (do with user present in case drive.mount needs a UI click):**
1. Each tab: Runtime → Restart Kernel (clears the duplicate driver threads). Drive will unmount.
2. Re-mount Drive (`from google.colab import drive; drive.mount('/content/drive')` — cached auth
   has worked headless before); `cd /content/pc321 && git fetch origin feat/principled-export-pipeline
   && git checkout -B feat/principled-export-pipeline FETCH_HEAD && find . -name '*.pyc' -delete`.
3. Launch EXACTLY ONE library driver per tab (one ~37GB assembly at a time = no OOM):
   for m in [xgb,logreg,fm,mlp] (NO et — ~14min/member, too slow), for f in [0,1,2]:
   env SHIP_FAMILY=<fam> SHIP_MODEL=m SHIP_MODE=library SHIP_OOF_FOLD=f SHIP_ROW_SOURCE=full
   SHIP_LIB_M=30 SHIP_LIB_RHO=0.3,0.9 SHIP_LIB_MAX_ROWS=1000000, run via
   `python /content/pc321/scripts/ship/run_one.py` subprocess. Idempotent: skip if
   `DR/ship/exp_loo/<fam>/lib_<model>_fold<f>/result.json` has experiment==subspace_library.
   (Library member files: `lib_<model>_fold<f>/members/m####/{oof.npz,model.json}`.)
4. After libraries: greedy-ES per archetype over members/*/oof.npz, fold into the stack; compare
   vs 0.42519.
**ALTERNATIVE: ship the 0.42519 stack as-is** (already a clean honest-OOF improvement).

## KEY CODE (all on branch feat/principled-export-pipeline)
- `scripts/ship/exp_loo_category_mlp.py` — the workhorse. SHIP_MODE in {full, loo, sweep, library};
  SHIP_MODEL in {logreg,xgb,lgbm,et,fm,mlp,cnn1d,dae,ft}; SHIP_FAMILY, SHIP_OOF_FOLD(0/1/2),
  SHIP_ROW_SOURCE=full(4.5M)/ship(264k). `full` = train full model + save OOF (preds/oof_preds.npz)
  + result.json, no LOO. `library` = M random-subspace members → members/m####/. Knobs:
  SHIP_LIB_M, SHIP_LIB_RHO, SHIP_LIB_MAX_ROWS(=1M, row cap for tractability), SHIP_ET_MAX_ROWS(=1M),
  SHIP_ET_MAX_DEPTH=12/MIN_LEAF=50/JOBS=4. Neural members PCA=192 (vs trees 64).
- `scripts/ship/run_one.py` — runs ONE exp.fn() as a subprocess from SHIP_* env (OOM isolation).
- `src/neural_members.py` — cnn1d/dae_mlp/ft_transformer (torch); cnn groups features by kind.
- `scripts/ship/roster.py` + `configs/submission_roster.yaml` — (older 26-learner roster; the
  active model list is the 5 T1 + 3 T2 archetypes above, per user's overnight directive).
- `scripts/ship/greedy_select.py` — Caruana greedy ES (base exists; extend for per-archetype).

## GOTCHAS / LESSONS (don't repeat)
- After launching a run_bg driver, IMMEDIATELY overwrite that cell with poll content — re-running a
  launch cell DOUBLE-LAUNCHES (I did this twice → the OOM mess).
- ONE heavy run_bg per tab at a time (each assembly ~37GB; concurrent → OOM on 167GB).
- run_one imports exp.py fresh per subprocess BUT honors __pycache__ — after a code fix, delete
  *.pyc (a stale .pyc caused the library t_elapsed crash to persist post-fix).
- ET on full rows is pathological (~67min/forest, ~105GB); capped to 1M rows + depth 12. Dropped
  from the library (too slow even capped).
- Heavy in-thread fits can't be killed without a kernel restart (which drops Drive). Use run_one
  subprocesses for anything killable. drive.mount via MCP has worked via cached auth (no UI hang)
  but is not guaranteed — prefer user present.
- Score by POSITION not item_key merge (item_key non-unique). Label is continuous [0,1] pass-rate;
  metric = mean soft binary cross-entropy. item cold-start; GroupKFold(item).

## NEXT ACTIONS ON RESUME
1. Reconnect bridges; check kernels alive + Drive mounted (qwen/nemotron may have lost Drive if a
   kernel OOM-died — flag if so).
2. Execute the MORNING ACTION (clean library restart) OR ship 0.42519 per user choice.
3. Then greedy-ES + Layer-3 hierarchical/flat stacks; report final vs 0.42519 / 0.43653.
