# WARM START — resume the AIDE overnight run

**Snapshot:** 2026-06-05 ~04:15. Read this first, then `docs/AIDE_OVERNIGHT_WORKFLOW.md`
(the full runbook + §6 babysitting protocol). This file = the 60-second resume.

> ⚠️ You were launched in the STUB repo `/home/akhaemenid/projects/321M/head` (its CLAUDE.md
> is an unrelated seeding-proof template — ignore it). **All real work is in the work repo
> `/home/akhaemenid/projects/prediction-competition-321M`**, branch `clean/aide-stacked-ensemble`,
> HEAD `133b56b` (pushed to origin `bwathomas/prediction-competition-321M`).

---

## What's happening
Three families (**qwen / mistral / llama**), each: derive leakage-safe OOF features → export
an item-cold-start task → an **AIDE-ml** agent (LLM that writes+runs code) searches for the
best model minimizing **GroupKFold(item_key) log loss**. Each runs on its own Colab A100 via a
separate MCP bridge. Goal: three honest per-family winners → stack into the final ensemble.

## Live state at snapshot
| Family | Bridge (MCP) | Notebook | State / job |
|---|---|---|---|
| **qwen** | `colab2` | original | **AIDE running** `qwen_aide` — was 12/30 steps, best val NLL **0.477**, healthy |
| **mistral** | `colab3` | `drive/1Hki5…` | **deriving** `mistral_feat` — ~fold0 660k/1.82M; when done → export → `mistral_aide` |
| **llama** | `colab` (mcp1) | `drive/1d7ALtX…` | **pending user's A100 alloc** (was L4). Drive mounted, embeddings confirmed. When A100: launch `derive_family(family="llama")` |

A **≤20-min monitoring wakeup is scheduled** (ScheduleWakeup, self-contained prompt) — it may
fire and drive the loop even before you read this. If you're resuming manually, run §6 of the
runbook yourself.

---

## First actions on resume
1. **Load the Colab MCP tools** (they're deferred): `ToolSearch` for
   `mcp__colab2__run_code_cell,mcp__colab2__add_code_cell`, and the same for `colab3` and `colab`.
   The bridges should still be connected; if a `run_code_cell` errors, call that bridge's
   `mcp__<bridge>__open_colab_browser_connection` to reattach.
2. **Bridge map (do NOT mix up):** `colab2`=qwen, `colab3`=mistral, `colab`(mcp1)=llama.
   `colab`(mcp1) was repointed to the llama notebook by editing
   `/home/akhaemenid/projects/349D/.claude/colab_url_capture.sh` (rewrites the notebook path to
   `drive/1d7ALtX…`, keeps the `#mcpProxyToken/Port` fragment). To repoint any bridge, edit that
   `NOTEBOOK="drive/<id>"` line + reconnect. Verify with a `hostname`/`nvidia-smi` cell.
3. **Poll each job** (status files under `/content/<job>.json`; helper `poll(name)` in
   `aide/features/colab_runtime`). Job names: `qwen_aide`, `mistral_feat`, `mistral_aide`,
   `llama_feat`, `llama_aide`. Apply runbook §6.

## How to drive each stage (templates already in the notebooks)
- **Derive features:** on the family's bridge, `derive_family(drive_root=DRIVE, family=…,
  code_version="v2", include_cluster=True)` via `run_bg` (~40 min A100). DRIVE =
  `/content/drive/MyDrive/prediction-competition-321M`. Repo at `/content/pc321`.
- **Export for AIDE:** `assemble_training_matrix` (geometry + nn/cluster label groups) →
  subsample 400k → `export_for_aide(ds, manifest, out_dir=DRIVE/aide/<fam>_task,
  secret_dir=DRIVE/aide/<fam>_secret)`. (qwen template = colab2 cell `pVitfQ_5uOz4`.)
- **Launch AIDE:** subprocess `aide data_dir=<task> desc_file=<task>/task.md
  exp_name=<fam>_overnight agent.steps=30 agent.code.model=claude-sonnet-4-6
  agent.feedback.model=… report.model=…` via `run_bg`, with `env={**os.environ,"PYTHONPATH":""}`
  (so it imports installed aideml, not our `aide/`), `cwd=DRIVE/aide/<fam>_run`. (qwen template
  = colab2 cell `dBPOqRUiYN5m`.)

## Gotchas (will bite you)
- **API key:** `ANTHROPIC_API_KEY` must be in the kernel env. `userdata.get()` only works when
  a cell is run from the **Colab UI** (not via MCP) — have the user run a one-line loader cell
  per runtime: `import os;from google.colab import userdata;os.environ["ANTHROPIC_API_KEY"]=userdata.get("ANTHROPIC_API_KEY")`.
- **Models:** this key has the **4.x** family only (`claude-sonnet-4-6` overnight,
  `claude-haiku-4-5-20251001` smokes). 3.5 ids → 404.
- **aide namespace collision:** the installed `aideml` and our package both import as `aide`;
  always run AIDE as a subprocess with `PYTHONPATH=""` and `cwd` NOT in `/content/pc321`.
- **Secret holdout:** `holdout_labels.parquet` lives in `…/<fam>_secret/` (OUTSIDE the AIDE
  data_dir) — AIDE must never see it. Use it to independently re-score each winner (runbook §6e).
- **FoldFeatureStore args are keyword-only:** `FoldFeatureStore(cache, embedding_family=…,
  seed=0, n_folds=3)`.
- **Colab `run_code_cell` is synchronous** → all long work goes through `run_bg`/`poll`.
- **GPU contention:** don't run two FAISS/GPU jobs in one runtime — it crashed both threads
  earlier. One derivation per runtime.
- **Drive parquet reads are slow** (~5 min for the 10 GB items file); `embed_io` has a one-time
  float16 `.npy` converter to speed repeat loads (not yet applied per family).

## What's DONE (committed, tested — don't redo)
- Full feature pipeline: `aide/features/` (cache, derive_nn/cluster/tabular, store, driver,
  nn_fast, cluster_fast, passrate, embed_io, colab_runtime, metadata). 175 tests pass.
- qwen features fully derived + validated (Drive `features/qwen/`, 17 shards).
- AIDE wiring: `aide/ensemble/{assemble,optimize,overnight,aide_export}.py` + tests.
- Runbook `docs/AIDE_OVERNIGHT_WORKFLOW.md`.

## What's PENDING
- mistral: derivation finishing → export → `mistral_aide`.
- llama: A100 alloc → `llama_feat` → export → `llama_aide`.
- Babysit all three (runbook §6); extend steps while improving; secret-score each winner.
- Then: stack the three winners into the final 2-layer ensemble.
