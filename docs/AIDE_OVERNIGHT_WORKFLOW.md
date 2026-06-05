# AIDE Overnight Workflow — Operations Runbook

**What this is:** three parallel feature-derivation + autonomous-optimization pipelines, one
per embedding family (**qwen / mistral / llama**), each running on its own Colab A100 via a
separate MCP bridge, with [AIDE-ml](https://github.com/Wecoai/aideml) (an LLM coding agent)
searching a tree of model solutions to minimize **out-of-fold, item-cold-start log-loss**.

This document explains the architecture, the exact state, and — most importantly — **how to
babysit the AIDE agents overnight**: cadence, what "healthy" looks like, when to give more
steps, and when to intervene.

---

## 1. The goal (one paragraph)

Stanford Predictive Evaluation Challenge: predict `P(AI subject passes a benchmark item)`,
**item cold-start** (the test items are unseen), metric = **mean log loss**. We derived
leakage-safe, out-of-fold features (embedding geometry + neighbour pass-rates + cluster
stats) once per family, then hand each family's features to an AIDE agent that writes and
runs code to build the best model. The three families' winners later stack into one ensemble.

---

## 2. Architecture

```
per family (qwen, mistral, llama):
  embeddings.parquet (Drive)
        │  derive_family()  [aide/features/driver.py]  ~40 min on A100
        ▼
  derive-once feature cache  Drive/features/<family>/<group>/fold*.npz   (16 shards)
        │  assemble_training_matrix()  [aide/ensemble/assemble.py]
        ▼
  item-disjoint export  Drive/aide/<family>_task/{train,holdout_features}.parquet
        │                Drive/aide/<family>_secret/holdout_labels.parquet  (SECRET)
        ▼
  AIDE subprocess  `aide data_dir=… desc_file=task.md agent.steps=N model=claude-sonnet-4-6`
        ▼
  best solution + submission.csv → we re-score on the SECRET holdout (independent check)
```

**Why per-family-per-notebook:** each AIDE run is a long autonomous process; isolating
families on separate A100s lets them run in parallel and keeps one family's crash from
taking down the others.

---

## 3. Colab bridge topology (the fiddly part)

We drive Colab from WSL through the `colab-mcp` MCP servers. There are **three** bridges —
`colab` (mcp1), `colab2`, `colab3` — each a separate `uvx colab-mcp` process whose `BROWSER`
env points at a capture script in `349D/.claude/colab_url_capture{,2,3}.sh`.

**How a bridge attaches to a notebook:** the server emits a URL
`https://colab.research.google.com/<NOTEBOOK>#mcpProxyToken=…&mcpProxyPort=…`. The
**path before `#`** picks the notebook; the **`#`-fragment** carries the proxy token/port.
By default it's `notebooks/empty.ipynb` (a fresh blank), which is why colab1 & colab2 both
landed on the *same* (qwen) runtime. To pin a bridge to a specific notebook, the capture
script **rewrites the path** to `drive/<NOTEBOOK_ID>`, preserving the fragment. This edit is
the only thing that must be done from the WSL/Claude side.

**Current mapping:**

| Bridge | Notebook | GPU | Family | Capture script |
|---|---|---|---|---|
| `colab2` (mcp2) | qwen (the original) | A100 | **qwen** | `colab_url_capture2.sh` (empty.ipynb) |
| `colab3` (mcp3) | `drive/1Hki5…` | A100 | **mistral** | `colab_url_capture3.sh` |
| `colab` (mcp1) | `drive/1d7ALtX…` | L4→A100 | **llama** | `colab_url_capture.sh` — **rewritten** to `drive/1d7ALtXGcca_S7n9MQVeMn_dwIPiDW99f` |

> ⚠️ **Do not drive `colab` (mcp1) toward qwen.** It used to point at qwen (redundantly);
> it's now repinned to the llama notebook. qwen lives ONLY on `colab2`.

**To repoint a bridge to a different notebook:** edit the `NOTEBOOK="drive/<id>"` line in the
relevant `colab_url_capture*.sh`, then call that bridge's `open_colab_browser_connection`
again. Verify with a `hostname` / `nvidia-smi` cell — a *new* hostname means a new runtime.

---

## 4. Per-family pipeline stages

All code is committed on branch `clean/aide-stacked-ensemble` (repo cloned to `/content/pc321`
on each runtime). Each stage runs in a **background thread** via `run_bg(name, fn)` /
`poll(name)` (`aide/features/colab_runtime.py`) because the MCP `run_code_cell` is synchronous.

1. **Setup** (per runtime): mount Drive (interactive, user approves) → `git clone` the branch →
   background `pip install faiss-gpu-cu12 scipy polars scikit-learn pyarrow`.
2. **Derive features:** `derive_family(drive_root, family, code_version="v2", include_cluster=True)`
   → writes 16 shards to `Drive/features/<family>/`. ~40 min on A100. Idempotent (re-run skips
   done shards). Uses the FAST paths (`nn_fast`, `cluster_fast`).
3. **Export for AIDE:** assemble the per-row matrix (geometry joined per item, label groups
   concatenated across folds), subsample to ~400k rows, write item-disjoint `train.parquet` +
   `holdout_features.parquet` to `Drive/aide/<family>_task/`, and the **secret**
   `holdout_labels.parquet` to `Drive/aide/<family>_secret/` (NOT in the data_dir AIDE sees).
4. **Run AIDE:** subprocess with `PYTHONPATH=""` (so it imports the installed aideml, not our
   `aide/` package — they share the name), `data_dir=<task>`, `desc_file=task.md`,
   `agent.steps=N`, models = **`claude-opus-4-8`** (per user; expected ≫30 steps). The launch
   wrapper falls back to **`gpt-4o`** if a Claude run dies on funding/credit/quota/429/overloaded
   (needs `OPENAI_API_KEY`). Checkpoints its journal under the workdir.

---

## 5. Current state (live)

| Family | Bridge | Stage | Status / job name |
|---|---|---|---|
| **qwen** | colab2 | **AIDE running** | `qwen_aide` — sonnet-4-6, 30 steps. Features done (17 shards, validated). |
| **mistral** | colab3 | **deriving features** | `mistral_feat` — ~40 min. |
| **llama** | colab1 | **waiting on A100 alloc** | not launched; Drive mounted, embeddings confirmed. |

Secret holdouts for independent scoring live under `Drive/aide/<family>_secret/`.

---

## 6. Monitoring & management protocol (the babysitting)

**Cadence:** check every **≤20 minutes** once an AIDE agent is up. Derivations: every ~10 min.

### 6a. Poll an AIDE run
A tiny cell on the family's bridge:
```python
import json, glob, os
st = json.load(open("/content/<family>_aide.json"))
print(st.get("status"), st.get("message"))
# tail the live AIDE journal to see step count + best metric
W = "/content/drive/MyDrive/prediction-competition-321M/aide/<family>_run"
for j in glob.glob(f"{W}/logs/**/journal.json", recursive=True):
    J = json.load(open(j)); nodes = J["nodes"] if isinstance(J, dict) else J
    print("steps so far:", len(nodes))
    best = min((n for n in nodes if n.get("metric") is not None),
               key=lambda n: n["metric"]["value"], default=None)
    print("best val metric:", best["metric"]["value"] if best else "none yet")
```

### 6b. What "healthy" looks like
- `status == running`; the journal **node count grows** over time (a new solution every few min).
- The **best val metric (log loss) trends DOWN** or holds; new nodes are mostly `buggy=False`.
- Each accepted solution uses **`GroupKFold(item_key)`** and does **not** use `item_key` (or
  any id) as a feature — grep the node code. (This is the cold-start hygiene; AIDE was
  instructed to do it, but verify it actually does.)

### 6c. Give more steps (out-of-steps but still improving)
AIDE stops at `agent.steps=N`. If the run finished (`status == done`) but the best metric was
**still improving in the last few nodes** (not plateaued), relaunch with more steps — AIDE
resumes from its journal if you keep the same `exp_name` and workdir:
```python
# same exp_name + workdir → continues the tree; bump steps
cmd = ["aide", f"data_dir={TASK}", f"desc_file={TASK}/task.md", f"exp_name={family}_overnight",
       "agent.steps=30", "agent.code.model=claude-sonnet-4-6", ...]
```
Rule of thumb: **this is expected to run FAR more than 30 steps — be generous.** Add another
~50 steps whenever the last ~5 nodes improved the best metric at all; only stop when it's been
flat for ~8–10 nodes (plateau — diminishing returns). Always relaunch on **`claude-opus-4-8`**.

### 6d. Pathological signs → intervene
| Symptom | Likely cause | Action |
|---|---|---|
| journal node count **not growing** for >15 min | agent stuck / API error loop | check `*_stderr.txt`; common: model 404/rate-limit. Fix model id / wait, relaunch. |
| every recent node `buggy=True` | agent can't get a working solution (data/path issue) | read a node's code + exec error in the journal; fix the task.md hint or data path. |
| best metric **rising** or NaN | leaky/overfit solution or metric bug | verify the winner uses GroupKFold; re-score on the SECRET holdout (6e). |
| `status == error` | subprocess crashed | read `result.traceback` + `*_stderr.txt` tail. |
| thread dead but status "running" | kernel killed the daemon (GPU OOM/CUDA) | re-run `nvidia-smi`; relaunch; avoid two GPU jobs per runtime. |
| Colab disconnected | idle/preemption | reconnect bridge, remount Drive, relaunch (AIDE resumes from journal; derivations resume from INDEX). |

### 6e. The independent hygiene check (do this on each winner)
AIDE optimizes its *own* GroupKFold metric on `train.parquet`. Confirm it's honest by scoring
its `submission.csv` (predictions for `holdout_features.parquet`) against the **secret**
`holdout_labels.parquet` it never saw:
```python
import pandas as pd, numpy as np
sub = pd.read_csv(f"{W}/.../submission.csv")          # AIDE's holdout predictions
lab = pd.read_parquet(f"{DRIVE}/aide/<family>_secret/holdout_labels.parquet")
m = sub.merge(lab, on="item_key")
p = np.clip(m["prediction"].to_numpy(), 1e-7, 1-1e-7); y = m["label"].to_numpy()
print("secret-holdout NLL:", float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))
```
A secret-holdout NLL **close to** AIDE's reported val NLL ⇒ honest cold-start. A secret NLL
**much worse** than reported ⇒ the agent leaked (memorized items); discard that solution and
tighten the task.md instruction.

---

## 7. Hygiene invariants (must hold for every family)
- **Item cold-start:** train and validation/holdout are **item-disjoint** (`item_fold_split`).
- **No id as feature:** `item_key` is a group key only; never a model input.
- **Grouped CV:** AIDE validates with `GroupKFold(item_key)`.
- **Secret holdout:** `holdout_labels.parquet` lives OUTSIDE the AIDE `data_dir`, so the agent
  cannot read the answers — it's our independent referee.
- **Derive-once OOF features:** each row's features were computed leaving its own item out, so
  feeding them to AIDE's re-folded models stays honest.

---

## 8. Quick reference — job names & paths
- Derivation jobs: `qwen_full` (done), `mistral_feat` (colab3), `llama_feat` (colab1, pending).
- AIDE jobs: `qwen_aide` (colab2), `mistral_aide`, `llama_aide` (to launch after each export).
- Feature cache: `Drive/prediction-competition-321M/features/<family>/`.
- AIDE task/secret/run: `Drive/.../aide/<family>_task|_secret|_run/`.
- Repo on each runtime: `/content/pc321` (branch `clean/aide-stacked-ensemble`).
- Models: **`claude-opus-4-8`** for AIDE runs (per user); `claude-haiku-4-5-20251001` for cheap
  smokes. **OpenAI fallback `gpt-4o`** (needs `OPENAI_API_KEY`) auto-engages on Claude
  funding/rate failure. (3.5 ids 404.)

---

## 9. The plan from here
1. mistral features finish → export → launch `mistral_aide` (sonnet, 30 steps).
2. llama A100 lands → `llama_feat` → export → `llama_aide`.
3. Babysit all three per §6 (≤20-min cadence): keep them progressing, extend steps while
   improving, re-score winners on the secret holdouts.
4. By morning: three honest per-family best solutions → stack into the final 2-layer ensemble.
