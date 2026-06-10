# WARM START — resume the AIDE 3-family ensemble run (post-context-clear)

**Snapshot: 2026-06-05 ~18:35.** Read this FIRST, then `docs/OVERNIGHT_WORKING_MEMORY.md` (detailed log).
Work repo: `/home/akhaemenid/projects/prediction-competition-321M`, branch `clean/aide-stacked-ensemble`.
(The stub `/home/akhaemenid/projects/321M/head` CLAUDE.md is an unrelated seeding-proof template — ignore.)

---

## THE DELIVERABLE (safe on Drive, snapshot-protected)
**Final 2-layer stack: secret-holdout soft-logloss = 0.43722** (LightGBM cross_entropy meta over 17 base
learners from 3 families). Beats best single family (llama 0.44324) and baseline (predict-mean 0.629).
Saved on Drive: `…/aide/_winners_snapshot/` (per-family `{qwen,mistral,llama}/` winners + `FINAL_stack_submission.csv`
+ `FINAL_results.json`). **This is the protected result — never overwrite snapshots with anything worse-on-OOF.**

Metric = mean binary cross-entropy on a CONTINUOUS pass-rate label ∈[0,1] (NOT binary), item-cold-start,
GroupKFold(item_key). Score by POSITION (submission/holdout_features/holdout_labels are row-aligned) — NEVER
merge on item_key (it's non-unique → cartesian explosion).

## WHAT'S RUNNING NOW
3 AIDE-ml agents (LLM coding agents, model claude-opus-4-8), one per embedding family, each on its own
Colab A100 via a separate MCP bridge. Each builds a diverse ensemble; the 3 winners stack.
- **qwen** → bridge `colab2` (notebook empty.ipynb)
- **mistral** → bridge `colab3` (drive/1Hki5…) → host 1b1fce795001
- **llama** → bridge `colab` (mcp1, drive/1d7ALtX…) → host 87e3dc14ce39
All relaunched ~18:35 on Opus, `exec.timeout=2100` (35-min/node cap), gpt-4o fallback armed. 2 aideprocs each.

**Current TARGET (in each task.md):** reproduce the FULL ENRICHED ROSTER first — lgb(+DART/GOSS) · xgb
`reg:logistic` · cat `CrossEntropy`(classifier) · ExtraTrees · KNN · MLP(bagged) · Factorization Machine ·
amortized K-dim IRT (θ_s∈R^K from subject_key⊕subj_emb · item discrim+difficulty from features · benchmark
effect · subject dropout) — using metadata cols (subject_key/benchmark/condition/subj_emb_*), logit-blended —
THEN add the **featroute** member (a NESTED ensemble of per-feature-GROUP sub-models: nn/geo/clu-soft/
clu-onehot/centroid/cnt+clusubj/subj_emb/meta-cat → logit-mean → one decorrelated input covering all features).
Last bests (val): qwen ~0.444 / mistral ~0.440 / llama ~0.435 (building fresh trees post-relaunch).

## ⚠️ FREEZE RULES (caused two 20-30min freezes — DO NOT REPEAT)
1. **ALL heavy compute (multi-second) via `run_bg`+poll or a `threading.Thread`** — NEVER synchronous
   `run_code_cell`. `run_code_cell` is UNBOUNDED: a heavy cell or busy/dead kernel blocks it forever → freeze.
2. `open_colab_browser_connection` is BOUNDED (60s) → SAFE to call. `run_code_cell` only on a freshly-
   connected bridge with a FREE kernel.
3. `drive.mount`/`userdata.get` via MCP worked via cached browser auth but can hang on first-auth — prefer user runs them.

## BRIDGE RECONNECT (after a disconnect)
Bridges drop on freeze/idle. Reconnect: ToolSearch `select:mcp__<bridge>__open_colab_browser_connection,
mcp__<bridge>__run_code_cell,mcp__<bridge>__add_code_cell` then call `open_colab_browser_connection` (bounded).
It re-binds a runtime for the notebook — may land on the SAME live runtime (if alive) OR a FRESH blank one (if
the original was recycled). Check host/Drive/aideprocs with a trivial cell first. colab2(qwen)=empty.ipynb (no
redirect → tends to land fresh); colab/colab3 redirect to drive/<id>. Capture scripts:
`/home/akhaemenid/projects/349D/.claude/colab_url_capture{,2,3}.sh`. colab-mcp = `uvx git+github.com/googlecolab/colab-mcp`.

## STATUS / POLL CELLS (current notebook cell IDs)
- qwen colab2: status `60CtWaT8fbgE` · clean-relaunch `LuFJvPHtfw0c` · mount+keys `RZMmeeQRe54a`
- mistral colab3: status `Sycd1pxUh_39` · relaunch `dUezV3QBgD6c` · mount `SBRM9eXkdbEi` · keys `ntTCAWAsdpPn` · backend-patch `iGUt40L7f_zl`
- llama colab: status `ZpfiVv1bfV7f` · relaunch `kNfs1iebf29j` · mount+keys `OERJAmmFfQEe`
- colab2 stack/analysis cells: featroute build `mdeNLjxYPXrD` (RE-RUN VIA run_bg!), best-code-recover `Y2m0W512idjI`, task.md TARGET-fix `7-5HiPyaj3ak`.
(If a cell ID is "not found" after a runtime reset, just re-add the cell — Drive data persists.)

## BABYSIT PROTOCOL (20-min cadence; a ScheduleWakeup is active — fires & re-enters this loop)
Each cycle: reconnect dropped bridges → poll the 3 status cells → verify 2 aideprocs + bests descending toward
~0.444/0.440/0.435 + the full roster (lgb/xgb/cat/et/knn/mlp/fm/irt) then `featroute` appearing in
working/oof_predictions.csv. **RE-STACK** when a family beats its snapshot val (qwen<0.44442/mistral<0.44032/
llama<0.43715), JUDGED BY OOF: load that family's working oof+holdout_base CSVs + others' best + train label +
secret holdout → GroupKFold(5) LightGBM cross_entropy meta → SAVE to _winners_snapshot+FINAL ONLY IF
OOF<0.42993 AND secret<0.43722. Runtime reset → re-setup (mount+keys cells → clone pc321 → `pip install
git+https://github.com/WecoAI/aideml.git` via threading.Thread [NOT pypi `aideml` stub] → backend patch
{drop temperature for opus-4, V2 prompt-cache, max_tokens=16000} → relaunch on persisted task.md).

## KEY FACTS / ANALYSES (don't redo)
- **Bake-off (9 combiners): LightGBM meta WINS** (OOF 0.42993/secret 0.43722) > logistic 0.43784 > nonneg-wt >
  Caruana > family-mean > prob-mean > best-single 0.44595. all-17 > 3-family-means; logit>prob; OOF&secret rank consistently.
- **Meta-lean:** meta leans ~60% on `llama.mlp` (item-only, decorrelated); specialists `mistral.knn`(+110 perm),
  `mistral.ridge`(+62), `qwen.irt`(+38) punch above their gain; `mistral.cat`/`qwen.fm`/`llama.cat` are ballast.
- **Data enrichment (done):** added subject_key(906)+benchmark(16)+condition(215)+subj_emb_0..47 to all 3
  families' train/holdout parquets, ROW-ALIGNED (deterministic re-assemble; secret holdout_labels untouched).
- **AIDE does NOT resume** across launches (fresh tree; old journals persist but workspaces rmtree'd). Seed a run
  by injecting a target solution/roster into task.md ("TARGET ENSEMBLE" section).
- **Bug history (fixed):** label-is-continuous→cross-entropy regression; CatBoost CrossEntropy needs Classifier;
  Opus rejects temperature; max_tokens 4096→16000 (truncation); 1hr exec timeout vs slow KNN/ExtraTrees →
  exec.timeout=2100 + speed caps; duplicate-parent stall; aideml pypi stub vs real GitHub install.
- **Plateaued** ~1.5h before the freeze; featroute (orthogonal feature-routed member) is the current shot to break it.
- **API:** keys load from Colab userdata (UI). Anthropic funds were topped up; if a run shows credit/fallback,
  test `c.messages.create(model=claude-opus-4-8,...)` then relaunch on Opus.

## NEXT ACTIONS ON RESUME
1. Reconnect any dropped bridge (bounded call). 2. Poll the 3 status cells; confirm healthy + building full
roster→featroute. 3. Continue the 20-min babysit; re-stack on OOF improvement. 4. The user's idea (featroute
two-layer nested ensemble) is now mandated — watch for it to help the stack. 5. On user say-so: final re-stack
best-by-OOF + present FINAL (per-family table + best stack + did-featroute-help + bake-off + meta-lean + diversity).
