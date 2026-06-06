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

## RESULTS LOG (append)
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
