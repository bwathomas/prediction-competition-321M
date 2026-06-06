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

## RESULTS LOG (append)
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
