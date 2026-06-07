# OVERNIGHT MASTER PLAN v2 — read FIRST each wakeup (source of truth)

**Status:** RUNNING headless (user asleep, all permissions enabled). Repo
`/home/akhaemenid/projects/prediction-competition-321M` @ `feat/principled-export-pipeline`.
Colab: **colab2=qwen, colab=nemotron, colab3=lgai** (A100, Drive mounted). Reconnect MCP each
wakeup if dropped (`mcp__colab*__open_colab_browser_connection`, bounded=safe); reload tool
schemas via ToolSearch `select:mcp__colab*__run_code_cell,...__update_cell,...__get_cells`.

## GOAL (user, verbatim intent — the corrected architecture)
A **hierarchical 2–3 layer per-archetype feature-dropout ensemble**:
1. **Layer 0 — full models:** train every archetype fully (all features) → baseline NLL.
   Tier-1 (speed order): **LogReg, XGBoost, ExtraTrees, FM, MLP.**
   Tier-2 (slowest last, OK to truncate): **1D-CNN, DAE-MLP, FT-Transformer.**
2. **Layer 1 — per-archetype dropout library:** for EACH (family × archetype), generate M
   feature-dropout members, rho~U[0.3,0.9] per member (Random Subspace Method), **3-fold OOF**.
3. **Layer 2 — per-archetype ensemble:** greedy-ES (Caruana: with-replacement + sorted-init +
   bagged) combine that archetype's M members → one OOF prediction per (family × archetype).
4. **Layer 3 — stack up:** LightGBM tree-stack (A) archetypes within a family, then (B) the 3
   families; ALSO (C) flat tree-stack of all (family×archetype) ensembles. Compare A→B vs C.
Budget ~**8–12h**, **speed order** (fastest archetype first); if a slow T2 model doesn't finish
it's fine — skip+log. 3 tabs run in parallel (one family each), so 8h wall ≈ 24 GPU-h.

## MEASURED TIMING (per fold; full LOO run = 1 full + ~10 LOO + stack)
- MLP full run ~1670s; XGBoost ~1110s (=> per-fit ~150s / ~100s). XGBoost rho-member (sweep) **~70s**.
- Phase-0 timing runs IN FLIGHT (fold0): qwen=logreg, nemotron=et, lgai=fm. Record their t_total_s
  + per-fit (t_total/~11) into the RESULTS LOG when done; that sets per-archetype member cost.

## 8h BUDGET (per family/tab; member = 3 fold-fits for honest OOF)
M_per_family ≈ 28800 / (3 × t_member). xgb≈137, mlp≈64; logreg/et/fm fill in from Phase-0.
Greedy-ES plateaus at a few dozen from a library of hundreds → these M are ample.

## BUILD STATE (update each wakeup)
- ✅ et/fm/logreg wired into tree primitive via MODEL dispatch (`fit_and_predict_oof_tree`),
  xgb/mlp byte-identical. Commit 9d25374.
- ✅ `SHIP_MODE=library` built (tree models): M random-subspace members/fold, rho~U[lo,hi],
  per-member OOF saved to `<SAVE_ROOT>/members/m####.npz` + `oof_meta.npz`; stop on SHIP_LIB_M
  or SHIP_LIB_BUDGET_S. Commit 115cff2. Member i cols fold-independent => coherent OOF.
- ✅ SAVE ALL MODELS (user mandate): library members persist their trained model to
  `members/m####/` (xgb model.json / et|fm|logreg .npz) + `members/m####/oof.npz`. Keep
  SAVE_MODELS=1 everywhere (full + LOO + library + future mlp/T2). greedy_select reads
  `members/*/oof.npz`. Commit f25e48a.
- ✅ MLP library: col_mask added to `fit_and_predict_oof`; SHIP_MODE=library now covers mlp
  (random dense-column subspace, keeps item_emb+subject channels, saves model). Commit ac34a73.
- ✅ Tier-2 neural members WRITTEN: `src/neural_members.py` (cnn1d/dae/ft) wired into
  fit_and_predict_oof_tree + TREE set. SHIP_MODEL in {cnn1d,dae,ft}; full/LOO/library all work.
  Commit ac34a73. ⚠️ NOT smoke-tested on real data yet — FIRST run each once (fold0, small,
  SHIP_MODE=loo or a tiny library) and watch for: FT-Transformer OOM/slowness (attention over
  ~600 tokens; cut batch_size or skip), DAE swap-noise shapes, torch state_dict reload-verify.
  Fix+commit before launching their full libraries. Speed order puts them last anyway.
- ALL 8 archetypes (logreg/xgb/et/fm/mlp/cnn1d/dae/ft) now run full/LOO/library; every member
  saves its model (SAVE_MODELS=1).
- ⬜ `greedy_select.py`: extend to (a) per-archetype member-ES across the 3 folds' members/*.npz,
  (b) hierarchical A→B tree stack, (c) flat C tree stack; compare. (Base greedy ES exists.)

## SEQUENCE / PRIORITY
A. (in flight) Phase-0 timing: logreg/et/fm fold0 → validate the new wiring works + record times.
B. Launch **tree dropout libraries** in speed order (logreg, xgb, et, fm), per family × 3 folds,
   `SHIP_MODE=library SHIP_LIB_BUDGET_S=<share>` (or SHIP_LIB_M). Drivers per tab, run_bg.
C. Add col_mask to MLP primitive → MLP library; run it.
D. Build+smoke+run Tier-2 (cnn1d → dae_mlp → ft_transformer), slowest last.
E. greedy-ES per archetype → Layer-3 tree stacks (A→B and C); report numbers vs 0.43653 / 0.42333.
F. Keep all 3 tabs alive every wakeup; commit+push every code change; update this file + RESULTS LOG.

## OPS / GOTCHAS
- **run_bg ALWAYS** for heavy cells; never synchronous run_code_cell on heavy work.
- Deploy code: `git push` here → on each tab `git fetch origin <branch> && git checkout -B <branch> FETCH_HEAD` then importlib-load the exp module (env vars set BEFORE import — module reads os.environ at import time).
- **Leaked-env gotcha:** os.environ persists across runs in a kernel. ALWAYS set SHIP_MODE
  explicitly (loo/sweep/library) + pop stale SHIP_LIB_*; lgai had SHIP_MODE=sweep leak.
- FUSE wedge (driver stuck "waiting" though files exist) → `os.scandir(parent)` to refresh.
- Tab recycle (Drive unmounted / pc321 gone) → CANNOT remount headless (drive.mount needs UI) → FLAG for morning, continue other tabs.
- Keep-alive: running any status cell each wakeup counts as activity (idle disconnect ~90min).
- Per-tab launch cells (current): colab2 `lIYdn1woOS1n`, colab `n1mDQLlVoB47`, colab3 `OkodTZyfxfwo`.

## ⭐ USER DIRECTIVE (2026-06-06, going to bed): GREEDY FEATURE ABLATION AFTER THE FIRST PASS
The "first pass" = the full models now training (SHIP_MODE=full per archetype×fold×family).
**As soon as a family's full models are done, run the greedy feature ablation for it** = the
per-archetype random-subspace feature-dropout LIBRARY (SHIP_MODE=library, rho~U[0.3,0.9]) ->
greedy ensemble selection (greedy_select.py, Caruana: with-replacement + sorted-init + bagged)
over the cached member OOF, per archetype, compared vs the full-model baseline NLL. This is the
core deliverable — do NOT stop at the full pass. Launch libraries via the SUBPROCESS runner
(scripts/ship/run_one.py, commit 0adfb27) for OOM isolation. SAVE_MODELS=1 throughout.
Speed order logreg,xgb,et,fm,mlp then T2 (cnn,dae,ft after smoke). Then Layer-3 tree stacks.

## OOM AUDIT (14:12) — runs CANNOT OOM on current tabs
Measured peak during assembly: ~31-38GB RSS on 167GB tabs (>4x headroom); GPU 0-7.6/80GB.
3 historical OOM causes all eliminated: (1) 12GB qwen box -> now 167GB A100; (2) ExtraTrees
unlimited-depth+n_jobs=-1 -> capped (max_depth12/leaf50/jobs4, 5dfa088); (3) concurrent
assemblies (lgai sweep+fm) -> drivers now strictly SEQUENTIAL (one model/tab at a time).
Residual: accumulation over 15 sequential in-thread runs -> mitigated by run_one.py subprocess
isolation for the library phase (full reclaim per run; an OOM kills only the subprocess).

## CURRENT STATE (14:10) — ALL 3 TABS HEALTHY, TRAINING
- User restarted all 3 → all now **A100-80GB / 167GB RAM**. qwen (colab2) RESTORED (new host
  2885a6160e7f): Drive mounted (cached auth, no hang), pc321 cloned @ bba8ed9, deps OK.
- 3 idempotent FULL-ONLY drivers running, speed order [logreg,xgb,et,fm,mlp]×folds{0,1,2}:
  `qwen_full_driver` / `nemo_full_driver` / `lgai_full_driver`. All on logreg_f0, healthy.
- Drivers skip completed `full_only` results (restart-safe), set SHIP_MODE=full explicitly.
- Per-tab cells: colab2 lIYdn1woOS1n, colab n1mDQLlVoB47, colab3 OkodTZyfxfwo (now poll/driver).
- NEXT: monitor fulls completing (record t_total_s + full_baseline mll per archetype×fold×family);
  then SHIP_MODE=library dropout fleet; then greedy-ES + stacking. T2 (cnn/dae/ft) after smoke.

## ✅ MORNING ACTION DONE — L2 LIBRARY PHASE CLEANLY RELAUNCHED (see top RESULTS LOG tick)
The stuck/OOM library phase below was resolved: dup drivers self-terminated, tabs went clean, and
ONE clean driver/tab is now running (lgai pid200554/qwen pid97328/nemotron pid197710, status files
/content/lib_status_<fam>.json). Future ticks: MONITOR + double-launch-guard, do NOT relaunch.
Original stuck-phase notes kept below for history:

## 🔴 (HISTORICAL) MORNING ACTION NEEDED (2026-06-06 ~19:37) — LIBRARY PHASE STUCK
The dropout-library phase is stuck: my repeated accidental double-launches left MULTIPLE library
drivers per tab (nemotron 4, qwen 2, lgai 2), so multiple concurrent ~37GB assemblies OOM each
other. After ~30min: ZERO completed libraries on any family; nemotron actively OOM-thrashing
(RAMfree 4GB). Subprocess isolation has kept the KERNELS alive (Drive safe so far). I stopped
intervening (kernel-restart = freeze/remount risk while user asleep).
RECOMMENDED MORNING FIX (with user present for drive.mount if needed):
  1. On each tab: Runtime > Restart (clears all the duplicate driver threads).
  2. Re-mount Drive + `cd /content/pc321 && git pull` (cb67527+; find . -name '*.pyc' -delete).
  3. Launch EXACTLY ONE clean lib driver per tab: SHIP_MODE=library, models=[xgb,logreg,fm,mlp]
     (NO et), M=20-30, SHIP_LIB_MAX_ROWS=1000000, via run_one subprocess. ONE driver/tab = one
     ~37GB assembly at a time = no OOM. Idempotent skip resumes any completed.
  4. After libraries: greedy_select.py (Caruana ES) per archetype over members/*/oof.npz, fold
     into the stack; compare vs the flat full-model stack 0.42519.
ALTERNATIVE: ship the delivered 0.42519 stack as-is (already beats old 0.43653 by 0.011).

## RESULTS LOG (append)
- TICK (2026-06-07, autonomous) — **BENCHMARK-COLD MEMBER PROBE** (SHIP_HOLDOUT_BENCH: train irt_bag on all
  benchmarks EXCEPT X, predict X cold; nemotron, M=6). **Member signal is benchmark-SPECIFIC and ANTI-TRANSFERS:**
  | bench | warm item-OOF | COLD (domain held out) | own-mean | degradation |
  | afrimedqa | 0.610 | **1.027** | 0.652 | +0.42 |
  | mmlupro | 0.633 | **0.858** | 0.693 | +0.22 |
  | swebench | 0.711 | **0.801** | 0.693 | +0.09 |
  Cold > own-mean for ALL 3 ⇒ a domain-cold member is WORSE THAN A CONSTANT. Warm < own-mean ⇒ in-distribution
  it adds real signal. **This explains the ~0.42 item-OOF vs ~0.57 real-test gap: the gap is DOMAIN COLD-START.**
  The stack-level LOBO (+0.0008) missed it because its members were warm. Result: DR/ship/stack/coldprobe_warmvscold.json.
  ⇒ HIGHEST-VALUE next work: **OOD-aware shrinkage** — detect cold/sparse-domain items (embedding distance to
  train, or benchmark frequency) and shrink their prediction toward base-rate (cold 1.03 vs mean 0.65 ⇒ huge
  recoverable loss). Global shrinkage was λ=0 on warm-LOBO but should be large in the genuinely-cold regime.
- TICK (2026-06-07, autonomous) — **LOBO CV + CALIBRATION** on the 9-member canon stack (leave-one-
  benchmark-out: fit non-neg stack on 15 benchmarks, predict the held-out one; 16 benchmarks).
  **item-OOF 0.42182 → LOBO 0.42266 (+0.00084 only)** ⇒ the STACK COMBINER generalizes across benchmarks,
  does NOT overfit benchmark composition; local OOF is trustworthy. **Calibration NULL**: opt temperature
  1.002, shrink-to-global-mean λ=0.0 → no change (stack already well-calibrated globally). Stack beats
  predict-own-benchmark-mean on ALL 16 (hardest swebench 0.599/0.693, hle 0.595/0.623, mmlupro 0.560/0.693;
  easiest agentdojo 0.267, mmbench_v11 0.269). Result: DR/ship/stack/lobo_calib.json.
  ⚠️ CAVEAT: this holds out only the STACK; base members still trained on each benchmark's items (item-fold
  OOF). So it does NOT measure member-level benchmark cold-start — the prime remaining suspect for the
  ~0.42→0.57 real-test gap. NEXT: true benchmark-cold MEMBER probe (retrain a member holding out a benchmark).
- TICK (2026-06-07, Stage E) — **FINAL CANON STACK** (9 members: mlp-LOO `stacked_oof` + etbig `p_full`
  + irt_bag `p_full`, per family; non-neg logistic on logit-space cols, GroupKFold(5) honest item OOF).
  **canon non-neg OOF = 0.42182** — IDENTICAL to the full 24-col base+irt_bag pool (0.42182) ⇒ the 15
  zero-weight members (dae/cnn1d/logreg/fm/xgb/knn/irt2/irt_lib) add nothing; the lean canon is complete
  and more robust. Weights (full-data fit): nemotron.mlp.L1 0.450 (dom) · lgai.etbig 0.207 · qwen.mlp.L1
  0.098 · nemotron.irt_bag 0.078 · qwen.irt_bag 0.075 · qwen.etbig 0.065 · nemotron.etbig 0.052 · lgai.mlp.L1
  0.035 · lgai.irt_bag 0.008 · bias -0.073. vs old shipped 0.43653 = -0.0147. Result: DR/ship/stack/canon_stack.json.
  ⚠️ This is the OOF stack metric/weights; SHIPPING (test preds) needs the all-data TRAIN_ALL models (Stage D)
  + export. CAVEAT (STRATEGIC): 0.42182 is item-OOF (optimistic vs ~0.57 real); validate under LOBO next.
- TICK (2026-06-07, user req) — **3-VARIANT IRT STACK EVAL** (`master_stack_eval.py`, full 4.5M-row OOF,
  GroupKFold(5) on item, vs the 21-col base pool: per family dae/mlp/cnn1d LOO stacked_oof + xgb/etbig/logreg/fm
  p_full). Each IRT variant = its `p_full` col/family appended to the pool.
  **LINEAR stack soft-logloss** (Δ vs base 0.42153): base+**irt_bag 0.42038 (+0.00115, BEST)** > base+irt(orig)
  0.42041 (+0.00112) > base+irt_lib 0.42072 (+0.00081) > base+irt2 0.42120 (+0.00033) > base+knn 0.42137 (+0.00016).
  **NON-NEG stack** (Δ vs base 0.42278): base+**irt_bag 0.42182 (+0.00096, BEST)** > irt 0.42198 (+0.0008) >
  irt_lib 0.42206 (+0.00072) > irt2 0.42263 (+0.00015) > knn 0.42278 (**+0.0, zero help**).
  ⇒ **WINNER = irt_bag** (M=16 bagged K-dim item-amortized IRT). It is the only IRT model that earns a seat;
  it gets POSITIVE non-neg weight in ALL 3 families: qwen.irt_bag 0.0714, nemotron.irt_bag 0.0788, lgai.irt_bag
  0.0148. Drop-importance (Δ if removed): nemotron.irt_bag +0.00038, qwen.irt_bag +0.00022, lgai.irt_bag ~0.
  **irt2 (richer MIRT) FAILS** (+0.00015 nonneg; lgai diverged to 0.67-1.06 standalone) — documented negative.
  knn confirmed redundant (+0.0). irt_lib helps but < irt_bag.
  **CANON (nonzero non-neg weight, base+irt_bag):** mlp.L1 (all 3: nemo 0.422 dominant, qwen 0.122, lgai 0.064)
  + etbig (lgai 0.142, nemo 0.095, qwen 0.055) + irt_bag (all 3) + trivial nemotron.xgb 0.005. **dae/cnn1d/logreg/
  fm = 0 everywhere.** Result JSON: DR/ship/stack/master_stack_eval_result.json. ⇒ canon = mlp-LOO + etbig + irt_bag.
- TICK (2026-06-07, user req) — GREEDY WITHIN-FAMILY HIERARCHICAL STACK, honest **3-fold** leave-one-base-fold-out
  CV (each base model only predicts its held-out fold; meta folds = base folds — user correction to my initial
  5-fold). Arch: per family, bagged Caruana greedy ES over 9 members (3 neural-LOO L1 stacked_oof + xgb/etbig/
  logreg/fm full + IRT + kNN) -> 1 honest OOF/family -> top combiner over the 3. Results on colab2; persisted to
  DR/ship/results_greedy_hier_3fold/{greedy_hier,flat_3fold}_result.json.
  - Within-family greedy beats each family's best single modestly: qwen 0.43542 (vs mlp.L1 0.43855, +0.0031) /
    nemotron 0.42636 (vs 0.42728, +0.0009) / lgai 0.43618 (vs 0.43802, +0.0018). IRT earns real greedy weight
    (qwen .15 / nemo .07 / lgai .06); kNN marginal (qwen .09 / lgai .05 / nemo ~0, kNN single 0.521 useless) —
    same IRT>kNN verdict as master stack. mlp.L1 dominates every family; logreg/cnn1d/fm ~0.
  - Top combiner over 3 family OOF: tree 0.42351 | linear 0.42407 | greedy 0.42592 | logit-mean 0.42627.
  - ⚖️ APPLES-TO-APPLES (recomputed FLAT baselines under SAME 3-fold): flat_tree 0.41983 | flat_linear 0.42043 |
    flat_nonneg 0.42201 | best_single 0.42728. (5-fold flat_tree was 0.41934 -> 3-fold costs ~0.0005.)
  - VERDICT: hierarchical-greedy LOSES to flat by ~0.0037 (0.42351 vs 0.41983). Collapsing each family to one
    greedy-averaged number discards cross-family member-level interactions the flat meta exploits; greedy's prob-
    averaging collapse is even coarser than a linear/tree collapse. Confirms "flat > hierarchical"; best honest
    3-fold = flat_tree 0.41983. (colab/nemotron + colab3/lgai still RECYCLED — Drive unmounted, need user remount.)
- TICK — 4-WAY MASTER LINEAR STACKER RESULT (base 21 cols, no et): base 0.42153 | +IRT 0.42041 (+0.00112) |
  +kNN 0.42137 (+0.00016) | +IRT+kNN 0.42035 (+0.00118). VERDICT: IRT MOVES the needle (~+0.0011, the bilinear
  subject-ability x item signal mlp/etbig lack); kNN basically does NOT (~+0.0002, redundant with embedding
  models). Reverses AIDE-era knn>irt expectation. Resume doc: docs/WARM_START_2026-06-07b.md.
- TICK — MASTER LINEAR STACKER (user): over 21 cols (9 neural-LOO stacked_oof dae/mlp/cnn1d x3 +
  xgb/etbig/logreg/fm full x3; OMIT cpu et). BASE = linear 0.42153 / nonneg 0.42278 (vs flat-tree 0.4193).
  Added SHIP_MODEL=irt (3a2e63f, K-dim item-params-linear-from-embedding, no MLP) + SHIP_MODEL=knn
  (7d0cb3a, cuML GPU item-embedding kNN). Training irt+knn for all 3 fams (extras_driver.py). qwen.irt
  standalone ~0.476 (weak, generalizes cold). master_stack.py auto-runs +irt/+knn/+both deltas when ready.
  Sub-task: non-neg linear stacker over logreg-LOO members (greedy_lgai_logreg.json). Cron ed5648ee drives.
  - logreg-LOO verdict: full 0.4642 -> tree-meta-LOO 0.4559 (+0.0083, but that's tree NONLINEARITY on linear
    members) / non-neg LINEAR-meta-LOO 0.4625 (+0.0017, the fair linear-combiner test = small, fits variance story).
  - irt+knn done for qwen+nemotron; lgai in progress (knn 1/3 then irt). Full 4-way master_stack auto-runs when lgai done.
  - Non-neg master stacker WEIGHTS (base 21 cols): only mlp (0.81) + etbig (0.26) nonzero; xgb/dae/cnn1d/logreg/fm = 0
    (collinear-redundant given mlp/etbig selected). By emb: nemo 0.58 > lgai 0.28 > qwen 0.21. nemotron.mlp 0.567 dominates.
- TICK — DROPOUT-by-model-type verdicts: mlp +0.029, dae +0.010 (neural, strong) >> xgb +0.0027,
  etbig +0.0014 (trees, tree-meta ONLY; greedy/logit-mean WORSE than full for etbig). Confirms:
  random-subspace dropout helps HIGH-VARIANCE learners, negligible for already-bagged trees.
  CURVE (monotonic in learner variance): mlp +0.029 > dae +0.010 > fm +0.0066 > xgb +0.0027 > etbig +0.0014 > logreg(pending).
  fm-LOO: full 0.4494 -> loo_stack 0.4427 (+0.0066). [pure free-param IRT (no embedding) = 0.754, useless cold.]
  NEW: K-IRT (SHIP_MODEL=irt, commit 3a2e63f) = K-dim, item params LINEARLY amortized from item embedding
  (no MLP); running on qwen -> irt_full_fold<f>. Geometry-augmented LLTM/explanatory variant queued next.
  etbig: best_single 0.4481/logit_mean 0.4478/greedy 0.4471/linear 0.4457/tree 0.4445 vs full 0.4459.
- ⭐ USER STRATEGIC PIVOT (pending my action): local OOF 0.419 vs leaderboard-best 0.57 vs predict-mean
  0.629 => CV is optimistic (item-CV not benchmark-cold-start); cold test dominated by base-rate calibration
  + thin transferable (embedding) signal. PLAN proposed: (1) leave-one-BENCHMARK-out CV to re-rank all
  stacks honestly, (2) calibration layer (temperature + base-rate shrinkage tuned on LOBO), (3) switch
  submission combiner to most-LOBO-robust (non-neg blend / bagged greedy, NOT tree stack), (4) re-select
  members favoring embedding-driven over identity-feature ones. Awaiting user go-ahead.
- TICK — DROPOUT+GREEDY EXPERIMENT RESULTS (in progress):
  (A) lgai.xgb LIBRARY+greedy = 0.44016 vs full xgb 0.4420 -> BEATS by +0.0018 (greedy 0.4402 <
      logit_mean 0.4409 < full 0.4420 < best_single 0.4427). 30 members. /content/greedy_lgai_xgb.json.
      => random-subspace dropout + Caruana greedy helps a strong tree, modestly.
      FULL 3-COMBINER (user asked to verify greedy wasn't the weak link): best_single 0.4427 |
      logit_mean 0.4409 | greedy 0.4402 | linear 0.4401 | TREE_STACK 0.4393 (vs full 0.4420).
      => greedy/linear ~nothing (+0.0018), but TREE meta extracts +0.0027 (member interactions).
      NOT a clean falsification: dropout WEAKLY useful for GBDT (tree-meta only), STRONGLY useful for
      high-variance learners (mlp-LOO +0.029). Consistent w/ theory (xgb already colsamples internally).
  (B) qwen.etbig (GPU cuML RF) LIBRARY: fold0 done (M=20, ~56s/member); greedy pending all 3 folds.
  (C) nemotron.fm LOO: converted to GPU-RESIDENT FwFM (commit 829df1a) -> ~115s/member @40ep vs old
      ~300s/member @15ep streaming (~7x/epoch speedup); ~1h for full 3-fold sweep. verdict pending.
- TICK — 🏆 LAYERED RESTACK = NEW BEST 0.41989. layered_tree (L1=9 LOO stacked_oof ensembles + 15
  non-neural fulls -> LGB L2, GroupKFold5 item) = **0.41989** | layered_linear 0.42132 | flat_all_tree
  (=~90 raw members+15 fulls) = **0.41934** (the BEST, marginally beats layered by 0.0005). BEATS prior
  best mlp-LOO 0.42302 (-0.0034) and flat-tree 0.42355 and flat full-model 0.42519 and old shipped 0.43653.
  VERDICT: flat-all ~= layered (both ~0.419); layering did NOT beat flat (collapsing each LOO model to 1
  discards member-level interactions the flat GBM exploits). The WIN is the LOO members themselves.
  Then user: also test library+greedy on et -> qwen et library launched (pid218075, M=20, ET caps
  trees=80/depth=14/rows=400k for tractability) vs qwen full et 0.4520.
  Result /content/layered_stack_result.json (qwen). Bridges DROPPED then RECONNECTED — all 3 runtimes
  survived (same hosts, jobs ran through the disconnect as detached subprocs).
- TICK — LIBRARY+GREEDY EXPERIMENT (user): test random-subspace dropout (rho~U[0.3,0.9]) + Caruana
  greedy-w-replacement on other model types. nemotron.fm (colab, pid313595) + lgai.xgb (colab3, pid320851),
  SHIP_MODE=library M=30 RHO=0.3,0.9 MAX_ROWS=1M, folds 0-2, lib_<model>_fold<f>/members/m####/oof.npz.
  xgb fast (~35s/member, 17 done fold0); fm SLOW (~9min/member -> will get ~16/fold via 9000s timeout, enough
  for greedy). Greedy step: reuse greedy_select.py greedy_es() w/ custom loader over members/*/oof.npz, vs
  full fm 0.4496 / full xgb(lgai) 0.4420. status /content/libx_status_<fam>_<model>.json.
- TICK — NEURAL-LOO COMPLETE (all 3 fams x dae/mlp/cnn1d x 3 folds, every run ok). Per-fold LOO-stack
  beats full: dae ~+0.010, mlp ~+0.020-0.030, cnn1d ~+0.020 (cnn1d biggest relative but still weakest abs ~0.46).
  LAYERED restack LAUNCHED (pid 211884, /content/layered_stack.py -> /content/layered_stack_result.json):
  layered_tree (L1=9 stacked_oof ensembles + 15 non-neural fulls -> LGB L2), layered_linear, flat_all_tree
  (~90 raw members + 15 fulls). ~15-20 min. Compare vs 0.42355 (flat full-model) / 0.42302 (old mlp-LOO).
- TICK — ft STOPPED (user) + NEURAL-LOO LAUNCHED. ft was ~12 min/EPOCH (epoch1 vals: qwen 0.4517 /
  nemotron 0.4524 / lgai 0.4444 — ~0.45 neural band, marginal, worse than trees ~0.44) => 8 epochs
  ~96 min/fold > timeout => guaranteed waste. Killed ft on all 3 (batch is MEMORY not speed lever;
  per-epoch cost = rows x 607-token attention, irreducible by batch). User: "run LOO for each neural
  model". Launched SHIP_MODE=loo for dae,mlp,cnn1d (ft EXCLUDED — 11x too slow) per family x 3 folds:
  SHIP_ROW_SOURCE=full, SHIP_NEURAL_EPOCHS=15 (LOO=11 fits/fold), distinct dirs <model>_loo_fold<f>
  (no clobber of _full_fold), SAVE_MODELS=1. Produces p_full + 10 p_loo__<group> per (fam,model) =
  the feature-group-dropout members (the lever that got mlp-LOO to 0.42302). Drivers nloo_driver.py
  qwen142727/nemo242812/lgai248734, status /content/nloo_status_<fam>.json, logs nloo_<fam>_<m>_f<f>.log.
  ⚠️ neural-LOO is UNTESTED (was code-complete but never run) — watch dae fold0 produces p_loo__* cols.
  GOAL: stack these neural-LOO members (+ full models) -> test if it beats/closes gap to 0.42302.
  - UPDATE: neural-LOO path VALIDATED (dae_loo_fold0 saved p_full + 9 p_loo__<group> cols on all 3 fams,
    ~20 min/fold @ epochs=15). dae f0 done qwen/nemo/lgai; now on dae f1. mlp + cnn1d LOO queued after.
    Monitor cron 72a6bb2b (every 10 min) replaced f1d71779. dropping nn_label_derivatives hurts dae most.
  - UPDATE2: dae-LOO DONE qwen/lgai 3/3, nemotron 2/3 (f2 finishing) -> on to mlp-LOO. ENCOURAGING:
    the LOO exp's own internal LGB-stack-of-LOO-members BEATS the full model: nemotron dae f2
    stack 0.43706 vs full 0.44590 (+0.0088). Supports that neural-LOO members add real signal.
  - UPDATE3: dae-LOO 9/9 done. mlp-LOO f0 done all 3 (10 p_loo + p_full cols), on mlp f1. STRONG:
    nemotron mlp f0 internal LGB-stack-of-LOO = 0.42867 vs full mlp 0.45820 (+0.0295!) — near the
    GLOBAL best 0.42355 from ONE family's mlp alone. mlp-LOO members are the biggest lever so far.
    Folds ~30 min (mlp). cnn1d-LOO still queued after mlp.
  - RESTACK DESIGN (user, LAYERED — supersedes flat): LAYER 1 = per (family x neural archetype) GBM-stack
    of that LOO model's members = the saved 'stacked_oof' col in <model>_loo_fold<f>/preds/oof_preds.npz
    (9 ensembles: dae/mlp/cnn1d x 3 fams). LAYER 2 = LightGBM GroupKFold(5,item) over the 9 layer-1
    ensembles + 15 non-neural full models (logreg/xgb/et/etbig/fm x 3). Each LOO-model ensemble = own
    input. Compare vs 0.42355 / 0.42302 + a flat-all baseline. Monitor cron now 948c3417 (was 72a6bb2b).
    dae LOO-stack vs full dae (per fam, mean/3folds): qwen 0.4458 vs 0.4566 (+0.0108) / nemo 0.4373 vs
    0.4475 (+0.0102) / lgai 0.4408 vs 0.4512 (+0.0103). mlp LOO-stack lift bigger (~+0.0295 nemo).
- TICK — STACKING EXPERIMENTS DONE (24 cols = 8 models x 3 fams, ft EXCLUDED; honest GroupKFold(5)
  on item, full 4.5M OOF). Results (soft-logloss): best_single 0.43789 · logit_mean 0.43433 ·
  **C_tree (flat LightGBM) 0.42355 (BEST)** · C_linear 0.42520 · A_tree (hier) 0.42473 · A_linear 0.42605.
  Per-family tree ensembles: nemotron 0.4287 (strongest emb) · lgai 0.4356 · qwen 0.4394.
  CONCLUSIONS: FLAT > HIERARCHICAL (collapsing each emb to 1 number loses cross-emb model interactions);
  TREE > LINEAR at both levels. Best 0.42355 BEATS old flat 0.42519 (-0.0016, upgraded set w/ etbig+mlp-full
  pays off) but still +0.0005 vs mlp-LOO 0.42302 (its feature-group-dropout members carry extra signal).
  Script /content/stack_exp.py, result /content/stack_exp_result.json. NEXT: add ft when ready; revisit
  whether feature-dropout LIBRARY members (the mlp-LOO lever) close the gap to/below 0.42302.
- TICK — ft KILLED + SPED UP (user): ft was ~48 min/fold (default batch=2048, epochs=25). Added code
  (commit ea160f8): SHIP_NEURAL_EPOCHS / SHIP_NEURAL_BATCH knobs in _fit_neural + per-epoch step_fn ->
  status JSON (stage=fit_epoch epoch/n_epochs/val_loss) AND stdout; ft wrapper now forwards step_fn.
  Relaunched ft-only driver per tab (ft_driver.py, status /content/ft_status_<fam>.json) with
  SHIP_NEURAL_EPOCHS=8 SHIP_NEURAL_BATCH=4096, live per-fold log /content/run_<fam>_ft_fold<f>.log,
  40-min/fold timeout. ⚠️ batch=16384 OOM'd (FT attention ~batch x 607_tokens^2 => tried 132GB > 80GB GPU);
  ft is batch-MEMORY-bound, NOT throughput-bound. SAFE batch <= ~4096 (~33GB). Speedup comes from
  epochs (25->8), not batch. Drivers qwen133625/nemo233785/lgai239285 @ea160f8. (batch 16384 sized to avoid
  FT-attention GPU OOM; can push higher once confirmed.) ⚠️ GUARD: poll ft_status_<fam>.json; pid alive
  or finished unset => do not relaunch.
- TICK — RESULTS: etbig (cuML GPU RF, real/full) is a BIG win over capped et, now best/tied-best
  TREE member per family (full NLL, mean/folds):
    qwen     etbig 0.4459 (0.4477/0.4440/0.4458)  vs et 0.4519  vs xgb 0.4472  -> BEATS xgb
    nemotron etbig 0.4381 (0.4401/0.4362, 2/3)     vs et 0.4455  vs xgb 0.4379  -> ties xgb
    lgai     etbig 0.4405 (0.4417/0.4392, 2/3)     vs et 0.4475  vs xgb 0.4420  -> BEATS xgb
  ~13 min/fold on full rows (vs capped et 24-70 min on 1M subsample). cuML integration clean.
  dae full DONE 9/9: qwen 0.4566 / nemotron 0.4475 / lgai 0.4512 (fm-competitive, decorrelated).
  cnn1d DONE 9/9 (~0.47-0.49, weakest but most-decorrelated). Queue now: qwen on mlp-full f0
  (115GB RSS, 2h timeout); nemotron/lgai on etbig f2. NEXT: etbig 3/3 all fams -> mlp-full sweep -> ft.
- TICK — MASTER QUEUE + etbig + MLP-FULL + STACKING PLAN (user 4-part request). Replaced the T2
  drivers with ONE master queue driver per tab (/content/q_driver.py, status /content/q_status_<fam>.json,
  log /content/q_driver_<fam>.log). QUEUE per family (idempotent skip on <model>_full_fold<f>/result.json
  ok + preds/oof_preds.npz; SHIP_EXP_SAVE_ROOT set per item so paths are uniform <model>_full_fold<f>):
    cnn1d -> dae -> etbig -> mlp -> ft   (ft LAST so it can't starve the wanted models)
  Drivers: qwen pid104496 / nemotron pid204834 / lgai pid208325, all @c3fb355.
  STATE at launch: qwen cnn1d3/3+dae3/3 done -> on etbig f0; nemotron/lgai cnn1d3/3 done -> on dae f0.
  (1) MLP FULL UNIVERSE: SHIP_MODEL=mlp SHIP_MODE=full SHIP_ROW_SOURCE=full -> mlp_full_fold<f>
      (overrode the mlp special _TAG via SHIP_EXP_SAVE_ROOT). Fixes the universe mismatch: the old
      mlp lived on the disjoint 264k LOO universe (exp_loo/<fam>/preds, zero item-overlap w/ the 4.5M
      full models) -> could NOT be co-correlated/co-stacked. New mlp_full_fold puts it on the 4.5M universe.
  (2) REAL ET = 'etbig' (commit c3fb355): cuML 26.02 GPU RandomForest on A100-80GB, ALL train rows,
      n_est=300 depth=18 max_features=0.3 n_bins=128 (env SHIP_ETB_*). sklearn ET couldn't scale
      (depth-12 + 1M-row cap was a hack). cuML has no literal ExtraTrees -> GPU RF w/ col-subsampling
      = same decorrelated-bagged-tree role. NEW tag 'etbig' (does NOT clobber old capped 'et').
      ⚠️ etbig model = cuml_rf.pkl (needs cuml/GPU to reload; fine for OOF/stack, revisit for final submit).
  (3) STACKING EXPERIMENTS (QUEUED — run after ALL of T2+mlp+etbig land; ft optional). Build per
      (fam,model) full OOF by concat folds 0/1/2 (position-aligned per fold), GroupKFold(item) meta.
      Compare, each with a LINEAR meta (logreg on logits) AND a TREE meta (LightGBM):
        (A) HIERARCHICAL: stack models WITHIN each embedding -> 3 embedding ensembles -> stack those 3.
        (C) FLAT: all (embedding x model) in one meta.
      Report OOF soft-logloss for {A-linear, A-tree, C-linear, C-tree} + logit-mean baseline; vs
      0.42519 (flat full stack) / 0.42302 (mlp-LOO). (Script to be written when OOF complete + smoke-tested.)
  CORR MATRICES (fold0, 4.5M universe) computed: cnn1d most-decorrelated everywhere (weak solo, diverse);
  xgb-et ~0.97 redundant; dae fm-competitive. mlp excluded (was 264k universe) -> fixed by (1).
  ⚠️ LESSON: pkill -9 -f run_one.py REPEATEDLY MISSES a heavy run_one (uninterruptible D-state mid-fit);
  twice left a ~100GB orphan after switching drivers. ALWAYS verify `ps run_one` after a driver swap and
  `kill -9 <pid>` any orphan (else 2x ~100GB assemblies OOM). ⚠️ GUARD (future ticks): poll q_status_<fam>.json;
  pid alive or finished unset => DO NOT relaunch.
- TICK — T2 REDIRECT (user correction: "L2 models" meant TIER-2 = cnn1d/dae/ft, NOT layer-2
  dropout libraries). KILLED the tier-1 library drivers; LAUNCHED Tier-2 FULL-model completion per
  tab via run_one subprocess: SHIP_MODE=full, SHIP_ROW_SOURCE=full, SAVE_MODELS=1, order
  cnn1d->dae->ft (slowest/riskiest last), ft has a 3600s/fold timeout. Idempotent skip on valid
  result.json(ok)+preds/oof_preds.npz. Picks up per family from left-off:
    qwen     needs dae f1,f2 + ft f0,f1,f2     (cnn1d 3/3 + dae f0 already done)
    nemotron needs cnn1d f1,f2 + dae x3 + ft x3 (only cnn1d f0 done)
    lgai     needs cnn1d f1,f2 + dae x3 + ft x3 (only cnn1d f0 done)
  Drivers: qwen pid98653 / nemotron pid199026 / lgai pid202139. Status /content/t2_status_<fam>.json;
  log /content/t2_driver_<fam>.log. (Had to kill one orphan nemotron lib run_one pid197711 @78GB
  that pkill missed -> tabs now 1 assembly each, ~100GB+ free.) ⚠️ GUARD (future ticks): poll
  t2_status_<fam>.json; if "pid" alive (ps -p) or "finished" unset => DO NOT relaunch, just poll.
  ⚠️ ft (FT-Transformer) NEVER completed before (OOM/slowness risk) — watch its runs; if it times
  out, cut batch_size in src/neural_members.py or skip ft. KNOWN T2 scores (full): cnn1d ~0.47-0.49
  (weakest archetype), qwen.dae 0.4580 (fm-competitive, promising). NEXT: when T2 fulls land, fold
  cnn1d/dae/ft OOF into the Layer-3 stack; compare vs 0.42519 (flat) / 0.42302 (mlp-LOO).
- TICK — L2 LAUNCH (live session, user present). ✅ CLEAN L2 DROPOUT-LIBRARY PHASE LAUNCHED on
  all 3 tabs after a full state audit. AUDIT: (a) all full-model OOF intact => 0.42519 flat stack
  reproducible; (b) MLP was the original LOO experiment, stored at exp_loo/<fam>/preds/oof_preds.npz
  (keys: p_full + 10 p_loo__<group>) NOT in <model>_full_fold dirs — its nested LightGBM stack =
  **0.42302** (greedy_mlp_report.json) = BEST on disk, beats 0.42519; (c) the prior OOM dup-drivers
  all self-terminated overnight; tabs clean/idle (3x A100/167GB, Drive mounted) before launch.
  TIER-1 full NLLs (mean/3 folds): xgb 0.4472/0.4379/0.4420 · et 0.4520/0.4455/0.4475 · fm
  0.4576/0.4496/0.4513 · mlp 0.4605/0.4519/0.4467 · logreg 0.4691/0.4621/0.4642 (qwen/nemo/lgai).
  LAUNCH: ONE driver/tab = /content/lib_driver.py (detached Popen), MODELS=[logreg,xgb,fm,mlp,et]
  x folds[0,1,2], SHIP_MODE=library, SHIP_LIB_M=30 (et=8), RHO=0.3,0.9, MAX_ROWS=1M, BUDGET_S=2400,
  SAVE_MODELS=1, each run via run_one.py subprocess (ONE ~24GB assembly/tab => no OOM). Idempotent
  skip on result.json experiment==subspace_library. Drivers: lgai pid200554 / qwen pid97328 /
  nemotron pid197710, all @167f28b, all on logreg_f0, run_one assembling (~24GB RSS), lgai 1 member.
  ⚠️ DOUBLE-LAUNCH GUARD (future ticks): read /content/lib_status_<fam>.json — if "pid" is alive
  (ps -p) or "finished" not set, DO NOT relaunch; just poll. Per-tab driver log /content/lib_driver_<fam>.log.
  NEXT: when libraries fill (~4-7h/tab), greedy_select.py per archetype over members/*/oof.npz =>
  per-archetype L2 OOF => Layer-3 stack (within-family A->B and flat C); compare vs 0.42519 / 0.42302.
- TICK 19:48 — ✅ nemotron SELF-HEALED (4 dup threads finished their passes -> 0 lib threads,
  164GB free). Launched ONE clean single lib driver (libclean: xgb,logreg,fm,mlp; xgb_f0 running,
  151GB free) — clean no-contention progress. Cell now POLL-ONLY (no more re-launch). lgai still
  2 threads (et-stuck + lib2 stalled@18, RAMfree 24); qwen still 2 lib2. They'll self-terminate
  like nemotron -> then launch ONE clean driver each. Recovery is self-executing per tab. So the
  morning-action is partially auto-running; user may not need to manually restart if tabs free
  themselves. Stack 0.42519 stands.
- TICK 19:37 — library STUCK: 0 completed libs/3 families; nemotron OOM-thrashing (RAMfree 4GB,
  4 dup threads, dmesg OOM-kills); lgai xgb0 stalled @18 (contended by old et-driver); qwen
  barely moving (2 lib2 + T2 ft_f1). STOPPED intervening (kernel-restart too risky while asleep;
  subprocess isolation keeps kernels+Drive safe). See MORNING ACTION above. Stack 0.42519 stands.
- TICK 19:27 — ⚠️ nemotron OOM-THRASHING: my repeated double-launches left 4 lib threads ->
  4x concurrent ~37GB assemblies (~148GB) -> RAMfree 1GB, dmesg OOM-kills. Subprocess isolation
  saved the KERNEL (no Drive loss). The 4 drivers are SINGLE-PASS -> will self-terminate after
  thrashing through their passes (OOM-fails are fast). NOT kernel-restarting (freeze risk while
  user asleep) + NOT adding drivers. PLAN: let redundant drivers self-terminate, then relaunch
  ONE clean lib2 per tab. lgai lib2 PROGRESSING (xgb0=18 members, ~1/min => ~30min/run). Library
  is slow (~30min/run x 12 runs/family = ~6h even clean). ✅ STACK 0.42519 remains the solid
  delivered result; library (greedy ablation) is the in-progress enhancement that hit turbulence
  from my launch errors. greedy-ES + restack once enough members exist (possibly morning).
- TICK 19:17 — ✅ PYC FIX CONFIRMED: lgai lib_xgb_fold0 = 7 members growing (past member 0, no
  crash) — libraries now generate correctly. ⚠️ thread-heavy from repeated double-launches (I
  re-ran launch cells AGAIN): nemotron 4 lib threads, qwen 2 lib2+T2. Bounded+self-converging
  (idempotent skip on result.json; deterministic members + atomic oof => no corruption; single-
  pass drivers self-terminate). Cells NOW POLL-ONLY (no launch code) => no more accidental
  re-launch. qwen logreg_f2 backfilled (15/15). qwen T2 on dae_f1. NEXT: let libraries complete
  (~18min/run, redundancy converges), then greedy_select over members + restack vs 0.42519.
  LESSON (repeated!): immediately overwrite launch cells with poll content.
- TICK 19:08 — relaunched CLEAN library drivers `lib2`=[xgb,logreg,fm,mlp] (NO et) on all 3 tabs
  (git pull + pyc delete + idempotent skip). xgb_f0 assembling (emb_index_ready, no crash) —
  confirms pyc fix path; member-count validation (expect 30) next tick. Old et-stuck lib_drivers
  remain as bounded background (lgai still has one on et_f0). Library dirs key: lib_<model>_fold<f>/
  members/m####/{oof.npz,model}. After libraries: greedy_select over members + fold into stack.
  Reminder: STACK 0.42519 is the delivered result.
- TICK 19:00 — LIBRARY DEBUG: the t_elapsed fix (cb67527) was on disk but a STALE
  __pycache__/*.pyc kept run_one importing the buggy bytecode -> all library runs STILL crashed
  after member 0. CLEARED all *.pyc on 3 tabs -> fresh subprocesses now compile fixed source.
  ALSO: ET library members PATHOLOGICALLY SLOW (et_f0: 1 member in ~14min; 30-member ET lib =
  hours) -> DROP et from the library (its full model already gives ET diversity in the 0.42519
  stack). LIBRARY = xgb,logreg,fm,mlp only. ⚠️ messy thread state (old lib_drivers stuck on et;
  nemotron has dup from my earlier re-run). RECOVERY (next tick): validate 1 fresh xgb library
  run = 30 members (confirms pyc fix), then relaunch CLEAN lib_drivers [xgb,logreg,fm,mlp] per
  tab (idempotent skip regenerates crashed xgb/logreg with fixed code); accept old drivers as
  bounded background (subprocess-isolated, deterministic members, atomic oof => no corruption).
  ✅ Stack 0.42519 is the solid delivered result regardless of library state.
- TICK 18:56 — 🏆 STACK OF CURRENT 15 FULL MODELS (T1 x 3 fams, 4.5M rows, honest GroupKFold(item)
  5-fold LightGBM meta): **lgb_stack 0.42519** | logit-mean 0.43331 | best single nemotron.xgb
  0.43789. BEATS old shipped stack 0.43653 (-0.0113) and best single (-0.0127). vs old AIDE
  MLP-LOO cross-fam 0.42333 (we're -0.0019 behind, but BEFORE libraries + T2). Report saved
  /content/stack_report.json (qwen tab). T2 cols skipped (incomplete). This 0.42519 is already a
  shippable honest-OOF stack. NEXT: add libraries (within-archetype dropout) + T2 -> push lower.
- TICK 18:34 — 🐞 LIBRARY BUG FOUND+FIXED (cb67527): step() double-passed t_elapsed -> every
  library run crashed AFTER member 0 (1 member, no result.json). Fixed (t_lib_elapsed) + made
  oof.npz write ATOMIC. git-pulled all 3 tabs => subsequent library subprocesses now generate 30
  members + result.json. ⚠️ I ALSO double-launched (re-ran launch cells): nemotron has 2
  lib_drivers, qwen 2 backfills — WASTEFUL but NOT corrupting (members deterministic from
  (LIB_SEED,i) + atomic oof writes; subprocess isolation => no kernel OOM). Drivers are
  single-pass forward loops => self-terminate. RECOVERY: early folds that crashed pre-fix
  (lgai xgb_f0/1/2, logreg_f0, etc.) were SKIPPED by the forward drivers -> relaunch ONE clean
  lib_driver per tab AFTER current passes finish (idempotent skip re-runs gappy folds, skips
  completed). LESSON: overwrite launch cells to poll/pull content immediately after launching.
  lgai T2 on ft_f0 (cnn1d+dae done). NEXT: validate a post-fix library run = 30 members.
- TICK 18:31 — STATUS + NLL. Full-model NLL (mean/3 folds): xgb best — nemo 0.43788 (f1 0.43661,
  ~= old shipped stack 0.43653!), lgai 0.44198, qwen 0.44722. et ~0.445-0.452, fm ~0.450-0.458,
  mlp ~0.447-0.460, logreg ~0.462-0.469, cnn1d 0.4753 (weakest, as predicted). All SINGLE models
  (no ensembling yet). nemotron T1 COMPLETE (15/15) -> launched nemotron LIBRARY. qwen 14/15
  (logreg_f2 gap) -> launched logreg_f2 BACKFILL (->15/15 unblocks its t2_driver). lgai library
  (xgb members) + T2 running. 3 families now in library/T2 phase. RAM: lgai 116, qwen 148, nemo 52
  free. run_one subprocess isolation => concurrent library+T2 can't OOM the kernel.
  NEXT: qwen library after backfill; validate library members emit; greedy_select over members.
- TICK 18:09 — 🎉 lgai T1 COMPLETE (15/15). LAUNCHED lgai LIBRARY (lib_driver: xgb,logreg,et,fm,
  mlp x folds, M=30, capped 1M rows, run_one subprocess) — GREEDY-ABLATION DELIVERABLE STARTED
  (xgb_f0 assembling). ✅ T2 NEURAL VALIDATED: cnn1d_full_fold0 done ok=True (neural_members.py
  works on real data!); t2_driver on cnn1d_f1. ✅ Concurrent library+T2 SAFE on 167GB: RAMfree
  116GB (no OOM). mlp full NLLs lgai 0.4500/0.4462/0.4438. nemo/qwen on mlp (12/24 each).
  NEXT: validate library xgb_f0 produces members/ + result; launch nemo/qwen libraries when
  their T1 done; backfill qwen logreg_f2; then greedy_select.py (Caruana) over cached members.
- TICK 17:57 — lgai mlp FAST (~5min/fold! vs fm/et ~24min): mlp_f0 0.45004, mlp_f1 0.44623 done,
  mlp_f2 running => lgai 14/15, ~5min to T1 complete. Single full MLP ~0.448 (weaker than xgb
  0.437; MLP's value was in stacking). nemo/qwen on fm_f2 (11,10). ⭐ SEQUENCING: when a family
  hits 15/15, its t2_driver auto-runs T2 (cnn/dae/ft) — to AVOID contention/OOM, launch that
  family's LIBRARY only AFTER its T2 is done (sequential T1->T2->library per tab). Monitor T2
  completion per tab, then launch library driver (run_one subprocess, SHIP_MODE=library).
- TICK 17:47 — lgai fm COMPLETE, now on mlp_f0 (LAST T1 archetype, 12/24, RAMfree 57 = mlp uses
  full 4096-dim emb, bounded). nemo fm_f2 (11/24), qwen fm_f2 (10/24). lgai -> mlp x3 -> 15/15
  (~60-90min) -> LAUNCH lgai library. All healthy.
- TICK 17:37 — qwen fm_f2 (10/24), nemo fm_f1 (10/24), lgai fm_f2 (11/24). fm ~24min/fold (spans
  2+ ticks). All healthy, RAM bounded. lgai next -> mlp x3 -> 15/15 then library.
- TICK 17:27 — lgai on fm_f2 (last fm fold, 11/24, leading); nemo/qwen fm_f1 (10,9). fm NLLs
  ~0.450-0.460. Healthy, RAM bounded. lgai -> mlp x3 -> T1 15/15 (~1h+) then library launch.
- TICK 17:17 — all 3 on fm_f1 (fm_f0 done: qwen 0.4597/nemo 0.4503/lgai 0.4525, ~24min). Counts
  qwen9/nemo10/lgai10. fm weak (~0.45-0.46). Healthy, RAM bounded. ~2h to nemo/lgai T1 15/15.
- TICK 17:07 — ✅ LIBRARY PREP DONE (c7a02ef): SHIP_LIB_MAX_ROWS=1M subsamples each fold's train
  rows once for ALL library members (fit primitives read train_idx as closure -> uniform); OOF
  coverage preserved; members differ by feature subspace. Library now tractable for all 8
  archetypes (+ ET keeps its own cap). LIBRARY LAUNCH PLAN (when a family hits T1 15/15): per
  family x model[logreg,xgb,et,fm] x fold, SHIP_MODE=library SHIP_LIB_RHO=0.3,0.9 SHIP_LIB_M=~30
  (or SHIP_LIB_BUDGET_S), via run_one.py subprocess driver (OOM-isolated). Progress: qwen8/nemo9/
  lgai10, all on fm (~24min/fold). Healthy.
- TICK 16:57 — fm SLOWER than expected: lgai fm_f0 0.45252, t 1418s (~24min, not ~5min). So
  EVERY archetype ~20-25min/fold on 4.5M rows. lgai on fm_f1 (10/24); nemo/qwen on fm_f0 (9,8).
  fm NLL 0.4525 weak (≈ET, < xgb). All healthy. lgai T1 15/15 in ~2hr (fm_f1/f2 + mlp x3).
  ⭐ LIBRARY PREP TODO (before launch): add SHIP_LIB_MAX_ROWS row-subsample for ALL archetypes
  (not just ET) — ~24min/member×many = infeasible on full rows. Implement in an upcoming tick
  while full pass runs; thread a per-member row subset through the fit primitives.
- TICK 16:47 — all 3 cleared ET, now on fm_f0 (GPU, light; nemo RAMfree 130). nemo/lgai 9/24,
  qwen 8/24. ⚠️ qwen logreg_f2 GAP => qwen tops out at T1=14/15, so its t2_driver (waits for
  15/15) will STALL. ACTION: backfill qwen logreg_f2 (run_one subprocess) once qwen full driver
  finishes (after mlp); then qwen=15/15 -> t2_driver fires. nemo/lgai will hit 15/15 cleanly
  after fm+mlp (~40min) -> launch their LIBRARIES then. RAM healthy.
- TICK 16:37 — lgai ET COMPLETE (et 0.4485/0.4464/0.4475), now on fm_f0 (9/24). nemotron/qwen
  still on et_f2 (8/24, 7/24). fm next (fast GPU), then mlp -> T1 15/15. lgai will finish T1 first
  => launch its greedy-ablation LIBRARY when it hits 15/15. RAM 81-87GB free, healthy.
- TICK 16:27 — all 3 on FINAL ET fold (et_f2, capped). et_f1 done: qwen 0.45044/24min,
  nemo 0.44388/26min, lgai 0.44638/23min. Counts qwen7/nemo8/lgai8. RAM 81-82GB free, healthy.
  After et_f2 -> fm (fast) -> mlp finishes T1(15/15) -> T2 drivers fire. THEN (cron must launch):
  greedy-ablation LIBRARY per user directive (T2 driver only does T2 FULLS, not the library).
- TICK 16:17 — lgai capped et_f1 DONE: NLL 0.44638, t 1364s (~23min) vs heavy et_f0 67min =>
  3x faster AND slightly BETTER than heavy (0.4464<0.4485) — row cap validated (forests don't
  need all 3M rows). lgai on et_f2 (8/24); nemotron/qwen finishing capped et_f1 (6-7/24). Healthy.
- TICK 16:07 — all 3 on capped et_f1 (~10min in; capped ET ~20min/fold on 1M rows, vs 70min
  heavy — big win). RAM bounded (free 80/80/94GB). Counts unchanged qwen6/nemo7/lgai7 (et_f1
  spans ~2 ticks). Healthy. No action.
- TICK 15:57 — all 3 heavy et_f0 DONE (qwen 0.45344/70min, nemo 0.44695/75min, lgai 0.44849/67min
  — confirms heavy ET ~70min/fit). All now on CAPPED et_f1 (RAM freed: qwen 64->129GB). ET NLLs
  ~0.447-0.453 (weaker than xgb ~0.437-0.443, diverse class). Counts qwen6/nemo7/lgai7. Capped
  et_f1/f2 should be ~13min; then fm/mlp; then T2 driver fires at T1=15/15; then library.
- TICK 15:47 — ✅ capped-ET fix CONFIRMED: lgai et_f0 DONE (NLL 0.44849, t_total 4025s = 67min,
  confirms heavy ET pathological) → lgai now on et_f1 with CAPPED config (RAMfree 86 vs 70 = uses
  ~1M rows). nemotron/qwen still finishing heavy et_f0 (then their f1/f2 capped). ET NLL 0.4485
  sits between logreg ~0.463 and xgb ~0.441 — weaker solo, useful different model class.
  Counts: qwen5/nemo6/lgai7. All healthy, RAM bounded.
- TICK 15:37 — ✅ FIX WITHOUT RESTART: the full driver re-reads exp.py per (model,fold) via
  importlib, so `git checkout` of the capped-ET code (bc9e5ae) on all 3 tabs WHILE running means
  et_f1/et_f2 will load ET_MAX_ROWS=1M (~13min) instead of heavy (~50min). et_f0 finishes heavy
  (already-loaded module). Saves ~1.5h/tab, NO kernel restart, NO Drive risk. (git checkout is a
  quick main-thread subprocess; ET runs in a daemon thread — no disruption.) Minor: et_f0 uncapped
  vs et_f1/f2 capped (mixed ET baseline across folds) — acceptable; can re-run et_f0 capped later.
  et_f0 still grinding (~50-58min); will finish then capped f1/f2 follow.
- TICK 15:27 — et_f0 still grinding (qwen 39min/nemo 45min/lgai 48min), 203% CPU, RSS 105GB
  (computing, not deadlocked). Will finish; folds 1/2 follow heavy => ~2.5h/tab uncapped ET.
  Considered kernel-restart->capped-ET relaunch (would cut ET to ~13min/fold) but REJECTED as too
  risky autonomously: restart drops Drive, and if drive.mount needs an interactive popup post-
  restart it bricks the tab till user wakes — loses more than the ~2h saved. HOLDING safe course:
  let heavy ET fulls finish (valid full-data ET baselines); library uses capped ET. Light-touch
  monitoring until ET completes (no safe action to take). Counts unchanged qwen5/nemo6/lgai6.
- TICK 15:17 — et_f0 PATHOLOGICALLY SLOW: 29-38min at fit_full (qwen1768/nemo2142/lgai2277s),
  still going, RSS~105GB (bounded). CANNOT abort safely: ET runs IN-KERNEL via joblib THREADING
  (single python3 proc, no killable workers) -> only a kernel restart would stop it, which loses
  Drive (UI remount) -> would block the tab till morning. WORSE than waiting. DECISION: let the
  heavy ET fulls FINISH (bounded, terminating, ARE the needed ET baselines — not wasted). Capped
  ET (8194fdb, 1M rows) can't reach running drivers (no re-pull). Full pass ETA +2-3h vs earlier.
  Library phase will use capped ET. LESSON: should have launched the FULL drivers via run_one.py
  subprocess too (like T2) so heavy in-thread fits were killable — in-thread ET is the trap.
  Counts unchanged qwen5/nemo6/lgai6. No intervention possible/safe this tick.
- TICK 15:07 — et_f0 STILL running on all 3 (~19min, qwen 203% CPU RSS 105GB, bounded, working
  not hung). ~20min/ET-fit is too slow for the LIBRARY phase. FIX pushed (8194fdb): cap ET
  training rows SHIP_ET_MAX_ROWS=1M (seeded subsample) -> ET tractable + lighter (forks ~1M not
  3M rows). Applies to FUTURE launches (library/relaunch); in-flight full-pass ET keeps heavy
  config (no re-pull per run) -> full-pass ET ~1hr/tab, let it finish. If et_f0 not done by next
  tick (~30min total) consider it pathological. Counts unchanged qwen5/nemo6/lgai6.
  REMINDER: library phase likely needs ROW SUBSAMPLING for ALL archetypes on 4.5M rows to be
  feasible (members are for feature-subspace diversity, don't need all rows).
- TICK 14:57 — ET is SLOW + heavy but BOUNDED+WORKING: qwen et_f0 fit_full 168% CPU, RSS 105GB
  (n_jobs=4 forks the ~14.5GB float64 X ×4), ~10min+/fit, still going. <167GB (62GB headroom) =>
  no OOM. All 3 tabs ~100GB during ET. Counts unchanged (qwen5/nemo6/lgai6, all on et_f0).
  ET will dominate full-pass time (~3 folds × ~12min/tab). For the LIBRARY phase (many ET members)
  consider ET n_jobs=2 / fewer n_estimators to cut the fork memory — but current runs are bounded,
  don't retune mid-flight (drivers don't re-pull per run). Confirm et_f0 completes next tick.
- TICK 14:47 — ALL 3 running et_f0 (ExtraTrees), RAM BOUNDED (free qwen116/nemo63/lgai70GB) =>
  capped ET config validated (no 120GB blowup). T1 logreg+xgb complete on nemotron & lgai (6/24);
  nemotron xgb best 0.43661. qwen 5/24. ⚠️ GAP: qwen logreg_f2 missing (no status file/result —
  silent transient on qwen's fresh runtime; driver caught+continued). BACKFILL qwen logreg_f2
  after its full pass (cheap; run_one subprocess). Watch for more gaps; nemotron/lgai complete.
  Confirm ET completes + timing next tick.
- TICK 14:37 — xgb NLLs in: nemotron xgb_f1 **0.43661** (⭐ best single, ~= old shipped STACK
  0.43653!), xgb_f0 0.43957; lgai xgb 0.4412-0.4431; qwen xgb_f0 0.44905. logreg ~0.46-0.47.
  Counts qwen 3/24, nemotron 5/24, lgai 6/24. **ET NOW RUNNING (lgai et_f0)** — RAM healthy
  (145GB free) => capped config keeping ET bounded; confirm et completes + timing next tick.
  RAM free: qwen 94 / nemotron 64 / lgai 145 GB — all safe. full-only ~260-290s/run.
- TICK 14:27 — progress healthy, all 6 drivers (3 full + 3 t2-waiting) alive. Dataset confirmed
  4,496,223 rows, y_mean 0.6918. NLLs: logreg ~0.460-0.471 (linear baseline). **lgai xgb_f0 =
  0.44307** (strong; ~best-single territory). full-only run time ~254-282s (xgb ~= logreg; GPU
  hist fast), qwen cold-cache f0 was 428s. Counts: qwen 2/15, nemotron 3/15, lgai 4/15 T1.
  WATCH NEXT: `et` (ExtraTrees) full on 3M rows — the memory/time wildcard (now capped
  depth12/leaf50/jobs4); confirm it completes in reasonable time + RSS next tick.
- TICK 14:22 — full_only logreg NLLs: qwen_f0 0.47131 | nemotron_f0 0.46439 f1 0.46125 |
  lgai_f0 0.46584 f1 0.46360. (logreg = linear baseline, weakest; xgb/mlp expected lower.)
  All 3 T1 drivers healthy (nemotron/lgai on logreg_f2; qwen catching up after cold-cache f0).
- TICK 14:22 — T2 DRIVERS LAUNCHED on all 3 tabs (`t2_driver`): each waits for its family's 15
  T1 fulls, THEN runs cnn1d/dae/ft × folds via run_one.py SUBPROCESS (OOM-isolated, idempotent
  skip, 3h timeout/run; ft may truncate). Second run_bg thread per tab, idle-waits => no
  contention with T1. So full pass = ALL 8 archetypes (T1 then T2) per family.
- 10-min cron f1d71779 confirmed ACTIVE (recurring 2-59/10).
- SEQUENCE REMINDER: after the FULL pass (T1+T2 fulls) → GREEDY FEATURE ABLATION (library +
  greedy-ES) per archetype per user directive. Recompute library budget from MEASURED 4.5M-row
  per-member times; likely T1 archetypes first; T2 libraries may be cost-limited.
- TICK 14:17 — FULL-ONLY VALIDATED end-to-end. ⚠️ KEY: ROW_SOURCE=full = **UNREDACTED ~4.5M
  rows** (qwen fold0: n_train=2.97M, n_oof=1.52M), NOT the 264k redacted sample. This is the
  intended full-data base learners; explains the ~37GB assembly. logreg full_baseline mll:
  nemotron 0.46439 (282s), lgai 0.46584 (264s), qwen 0.46~ (427s, cold cache). All 3 drivers
  progressing in speed order (nemotron/lgai on logreg_f1; qwen finished logreg_f0).
- ⚠️ LIBRARY BUDGET must be RECOMPUTED for 4.5M rows: my earlier 8h estimate (137 xgb members/
  family) assumed 264k rows / ~70s per member. On ~3M train rows per fold, per-member time is
  much higher (measure when libraries start; logreg full ≈ 270-430s). Pick SHIP_LIB_BUDGET_S /
  SHIP_LIB_M from the MEASURED per-member time on 4.5M, not the old 264k numbers.

- MLP LOO 3-fold (prior): qwen 0.4605→stack0.4385; nemotron 0.4519→0.4273; lgai 0.4467→0.4380.
  Cross-family small-tree stack of 3 MLP stacks = **0.42333** (best so far). Shipped baseline 0.43653.
- XGBoost LOO done all 9; item_emb_pca helps trees. lgai sweep fold0 done (~70s/xgb member).
- TICK 13:47 — Phase-0 status: qwen logreg + lgai fm run_bg threads DIED (status stuck at
  "started", threads gone, no python proc) — OOM/contention (lgai ran leaked sweep+fm at once).
  RELAUNCHED both cleanly 13:51 (SHIP_MODE=loo explicit, Drive asserted mounted). nemotron et
  ALIVE but pathological: unlimited-depth + n_jobs=-1 forked ~120GB RSS, >18min for one fit.
  FIXED ET config (max_depth=12/min_leaf=50/jobs=4, commit 5dfa088) — applies to FUTURE et runs;
  the current nemotron et thread still uses old config (can't kill a daemon thread w/o kernel
  restart=loses Drive); let it finish or OOM, then re-run et with the fix.
- DECISION: do NOT launch the tree library fleet until at least one archetype validates
  end-to-end (stage=done ok=True). Validate logreg/fm relaunch first.
- TICK 13:55 — 🔴 **qwen (colab2) is a 12GB-RAM runtime** (free -g=12; dmesg OOM-killed
  python3 @ ~11.7GB during assembly). logreg died at `config` TWICE = OOM, not a code bug.
  NOT fixable headless → **FLAG FOR MORNING: user must reconnect colab2 to an A100 high-RAM
  runtime** (then re-run qwen full-only). Do not keep relaunching qwen (futile).
- lgai fm (colab3, 80GB): HEALTHY, fit_full, thread alive (loo mode — LOO part wasted, but
  gives the fm full model). nemotron et (colab): still on ONE ExtraTrees fit at 24min+
  (pathological OLD config; ET fix 5dfa088 applies only to FUTURE et runs) — thread alive;
  let it finish/OOM, then re-run et full-only with the fix.
- ✅ SHIP_MODE=full added (commit bba8ed9): trains only the full model (no LOO). ALL future
  launches use SHIP_MODE=full (baseline) then SHIP_MODE=library (dropout) — never loo.
- PLAN next ticks: (1) qwen blocked → morning. (2) as lgai/nemotron free, launch SHIP_MODE=full
  for each archetype×fold on those 2 families. (3) then libraries. (4) greedy ES + stacking.
- Phase-0 timing t_total_s: still PENDING (lgai fm + nemotron et in progress; qwen blocked).
- TICK 14:00 — RECLAIMED nemotron: the runaway ET was orphaned **pid 599 (121GB RSS, swapping)**,
  separate from the kernel → `kill -9 599` freed it WITHOUT losing Drive. nemotron is a **167GB
  high-RAM** tab (163GB free after kill). Launched **`nemo_full_driver`** (run_bg): SHIP_MODE=full
  for logreg,xgb,et,fm,mlp × folds 0,1,2 sequentially (try/except per run). logreg_f0 validated
  past `config`→`loaded_embeddings` (full-only path works; ET now uses the capped config).
- lgai (colab3): fm still running healthy in loo mode (will finish full+LOO; harvest fm full,
  ignore LOO). When it frees → launch an lgai full-only driver (same as nemo). Can't kill the
  fm thread (torch, in-kernel, no child proc to target) without kernel restart.
- qwen (colab2): Colab DOWNGRADED it to free-tier CPU (12GB RAM, 2 cores, NO GPU —
  nvidia-smi absent). Can't run here (OOM + no cuda). Hardware is UI/Google-provisioned; no
  API to upgrade; bridge only attaches. Morning: user picks A100+High-RAM in UI to restore tab.
- ✅ KEY WORKAROUND: tab != family. qwen's embeddings/features are on SHARED Drive, so the
  **qwen FAMILY can run on any healthy GPU tab** via SHIP_FAMILY=qwen. Don't wait on the dead
  tab — once nemotron/lgai have spare capacity, run the qwen family there too. The dead qwen
  tab is just lost compute, NOT a blocker for the qwen-family deliverable.
- LESSON: a runaway joblib/compute can be a separate PID killable without kernel restart
  (check `ps`); a pure in-kernel torch thread cannot (only kernel restart, which loses Drive).
