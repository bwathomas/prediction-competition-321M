# AIDE-driven stacked-ensemble search — design spec

**Date:** 2026-06-04
**Branch:** `clean/aide-stacked-ensemble`
**Status:** APPROVED design (pending user spec-review)
**Competition:** Predictive AI Evaluation Challenge (item cold-start, mean log-likelihood / NLL, higher-is-better)

---

## 1. Goal & non-goals

**Goal.** Stand up three isolated AIDE (WecoAI `aideml`) agents — `llama`, `qwen`,
`mistral` — each on its own Colab A100 over its own WSL→Windows bridge, each using its
own cached text embeddings, all searching the *same* two-layer stacked-ensemble space to
**minimize item-cold-start out-of-fold (OOF) val NLL** under an aggressive, proxy-aware
dropout regime. Hygiene is enforced by the harness, never trusted to the agent.

**Deliverable of a run.** The best ensemble configuration found + its cached OOF
predictions and metrics. **Out of scope:** Codabench submission packaging
(`model.py`/`labeling.py`, size cap, runtime-import safety). We optimize the metric; we do
not ship a bundle in this phase.

**Non-goals.** No new feature generation (all features load from existing Drive caches).
No re-training of anything already cached. No changes to the live-runtime contract.

---

## 2. Embedding → agent mapping (LOCKED)

The three Drive embedding caches live under
`MyDrive/Prediction-Competition-321M/embeddings/`. Mapping confirmed against the user's
screenshot:

| Agent | Embedding cache folder | Bridge | Colab MCP server |
|---|---|---|---|
| `llama` | `nvidia__llama-embed-nemotron-8b__local` | bridge 1 | `colab` (`/tmp/colab_connect_url.txt`) |
| `qwen` | `Qwen__Qwen3-Embedding-8B` | bridge 2 | `colab2` (`/tmp/colab_connect_url2.txt`) |
| `mistral` | `embedding_cache_lgai_preview_fa2` | bridge 3 | `colab3` (`/tmp/colab_connect_url3.txt`) |

`Salesforce__SFR-Embedding-Mistral` is **unused** (the `mistral` agent reads LGAI-preview,
per the brief). Each agent reads ONLY its own folder; cross-reads are a config error.

Bridge implementation is the 349D pattern: three `uvx git+colab-mcp` MCP servers whose
`BROWSER` env points at `~/projects/349D/.claude/colab_url_capture{,2,3}.sh`, each
persisting to a distinct `/tmp/colab_connect_url*.txt` so tokens never clobber.

---

## 3. Repository layout — the new scoped canon

Branch off `main`. New top-level package `aide/` is the canon; legacy `src/` primitives
are moved/refactored into it (not rewritten from scratch).

```
aide/
  hygiene/
    splits.py        # item-uniform 3-fold OOF + recursive (nested) L2 OOF
    dropout.py       # proxy-aware subject/benchmark dropout
    proxy_tree.py    # the metadata→subject/benchmark proxy dependency tree (data)
    probes.py        # leakage tripwires (from src/oof_folds.py invariants)
    manifest.py      # SplitManifest: one canonical seed+assignment, shared across agents
  harness/
    eval.py          # evaluate(model_factory, mode) -> NLL  (owns folds/recursion/dropout)
    train.py         # trial -> full two-phase training + promotion gate
    funnel.py        # cached-feature router (LOAD ONLY; cache-miss is a hard error)
    registry.py      # architecture + ablation-set registry AIDE composes against
  ensemble/
    architectures/   # gbdt(lgbm/xgb/catboost), mlp, tabnet, ft_transformer, tabpfn,
                     # fm/ffm/fwfm, ridge/elasticnet/svm, knn, rf/extratrees,
                     # irt_2pl, kfactor_cf, mean_encode_glm ...  (see §6.1)
    ablations.py     # feature-ablation spec per architecture
    linear_stacker.py# the linear stacker used at BOTH layers (from src/stacker.py)
  agents/
    llama/   qwen/   mistral/      # per-agent: config.yaml, aideml workspace, journal/
  orchestrator/
    driver.py        # Claude Agent SDK orchestrator (one process per agent)
    bridge.py        # bridge manager (connect, redirect notebook, health)
    keepalive.py     # idle-timeout prevention (periodic no-op cell, capped)
    safety.py        # spend/time/compute/iteration caps + kill switch (see §9)
    journal.py       # journal harvest + resume-on-timeout
  configs/
    default.yaml     # shared knobs; per-agent overrides in agents/*/config.yaml
docs/superpowers/specs/2026-06-04-aide-stacked-ensemble-design.md   # this file
```

**Reused (moved & cleaned, not reinvented):** `src/oof_folds.py`→`aide/hygiene/`,
`src/oof_pipeline.py` (recursion substrate), `src/stacker.py`→`ensemble/linear_stacker.py`,
`src/drive_cache.py` (caching), and the member architectures under `ensemble/architectures/`.

---

## 4. Data-hygiene core

### 4.1 Splits — item-uniform, OOF, recursive, shared

- **Item-uniform, never row-uniform.** Group by `item_key = sha256(benchmark + "\n" +
  condition + "\n" + item_content)` (existing `src/data.py`). One item's many rows always
  fall in the same fold. Row-uniform splitting would leak an item across train/val.
- **Layer-1: 3-fold OOF.** For each fold *f*, layer-1 members train on the other 2 folds'
  items and predict fold *f* — concatenating yields exactly one OOF prediction per row from
  a model that never saw that row's item.
- **Layer-2: recursive (nested) OOF.** The layer-2 stacker's meta-features must themselves
  be OOF. Within each layer-1 *train* union we run an inner 3-fold OOF to produce the
  layer-1 member predictions that feed the layer-2 stacker, so the stacker never trains on
  a member's in-sample (optimistic) predictions. Cost ≈ 9× base trainings/architecture —
  made tractable by §7 caching + §5 trial gate.
- **One canonical `SplitManifest`, byte-identical across all three agents.** Generated once
  from a fixed seed (item_key → fold), cached to Drive, loaded by every agent, and
  asserted equal at load. This is what prevents cross-agent leakage when their OOF
  predictions are later compared/combined.

### 4.2 Dropout — aggressive and proxy-aware

Beyond ordinary feature dropout, we apply **subject dropout** and **benchmark dropout** so
models learn to predict on subjects/benchmarks they have not seen. Crucially, many metadata
fields, conditions, etc. are *proxies* for subject/benchmark identity; dropping the id but
leaving a proxy visible re-leaks identity. So dropout operates on a **proxy dependency
tree** and masks a node together with all descendants, atomically.

`aide/hygiene/proxy_tree.py` (committed data; refined during implementation):

```
subject  ⇒ mask: subject_key, subject_content(text),
                  metadata{family, macro-family, parameters, organization, release_date},
                  any NN/passrate feature aggregated over that subject
benchmark ⇒ mask: benchmark(name), condition (conditions proxy benchmarks),
                  data_category, benchmark-derived pool features
```

Applied identically in train and val, in both trial and full phases. Dropout rates are
search knobs but the *atomic node+descendants masking* is invariant, not optional.

---

## 5. The three harnesses

- **Eval harness** (`harness/eval.py`): `evaluate(model_factory, mode={trial,full}) -> NLL`.
  Owns folds, recursion, and dropout. The scalar it returns is the *only* thing AIDE
  optimizes. AIDE supplies a `model_factory` (a `fit(features)->preds` slot); it never
  receives held-out labels (see §8 boundary).
- **Train harness** (`harness/train.py`): two phases. **Trial** = item-subsampled data, few
  epochs — a fast directional gate (§5.1). **Full** = all data, full (recursive) OOF. Only
  trial-survivors reach full.
- **Funnel harness** (`harness/funnel.py`): routes the right **cached** features
  (per-agent embeddings, judge, NN-passrate, pool, metadata, cluster) to the right
  architecture at the right layer. **Load-only**: a cache miss is a hard error with a clear
  message, never a silent recompute.

### 5.1 Trial promotion gate (diversity-aware)

A linear stacker gains as much from *uncorrelated* members as from individually-strong
ones, so the gate rewards both. A trial candidate `c` (in a "comparable architecture group"
`g` — e.g. all GBDT variants, all MLP variants) is promoted to full eval if **either**:

1. **Competitive:** `NLL_trial(c) ≤ min_NLL(g) + X` — within `X` of the best run of a
   comparable architecture part; **or**
2. **Diversifying:** `diversity(c) ≥ D` — it adds orthogonal signal even when its standalone
   NLL is *worse* than clause-1's bar. (I read your "lower NLL but high diversity" as *admit
   diverse-but-weaker members*, since that is what a diversity exception is for — flag if you
   meant the literal reading.)

**Diversity score.** `diversity(c) = 1 − ρ̄`, where `ρ̄` is the mean pairwise correlation of
`c`'s **OOF residuals** (and, secondarily, its OOF predictions) with the existing member
pool. High = weakly correlated errors. Because all three agents share the byte-identical
`SplitManifest`, OOF predictions are **row-aligned across agents**, so the pool includes
*both* this agent's current members *and* the other agents' current-best members, read from
a shared **OOF board** on Drive (`Drive/Prediction-Competition-321M/oof_board/<agent>/…`,
append-only, manifest-keyed). So "weakly correlated with other agents" is a measured
quantity, not a heuristic. `X`, `D` are logged per candidate (defaults set in
implementation, e.g. `X≈0.01` nats, `D≈0.4`).

---

## 6. Two-layer stacked ensemble & the architecture canon

- **Top layer:** a linear stacker (or, as a search knob, a GBDT/logistic meta-learner) over
  *N* models, each a distinct **architecture**.
- **Layer below:** each top-layer "model" is itself a linear stacker over
  **feature-ablated variants of the same architecture** (different ablation sets allowed
  per architecture).
- **AIDE's search space:** {which architectures at the top; which ablation sets per
  architecture below; per-architecture hyperparameters; dropout rates; meta-learner choice}.
  **Objective:** full recursive-OOF val NLL.

### 6.1 The canon (Kaggle-grounded, training-only)

Grounded in current Kaggle stacking practice (e.g. Deotte's 1st-place April-2025 build:
72 models / 3 levels of XGBoost · LightGBM · CatBoost · NN · TabPFN · KNN · SVR · Ridge ·
RandomForest, meta-learned with Ridge+GBDT). This phase trains on the A100 with full
libraries, so the canon is **not** limited by `RUNTIME_ENV.md` (a later ship phase, out of
scope, would numpy-export the tree/sklearn members). Registry groups:

- **GBDT:** LightGBM, XGBoost, CatBoost, HistGradientBoosting (cheap variant).
- **Deep tabular:** MLP-with-embeddings (have), TabNet, FT-Transformer, TabPFN
  (subsample/foundation member — strong diversifier), optional NODE/SAINT.
- **Factorization / interaction** (native to subject×item×benchmark sparsity): FM, FFM,
  FwFM (have), optional DeepFM/xDeepFM.
- **Classical / linear** (cheap, strong stack diversifiers): logistic (have), Ridge,
  ElasticNet, SVM/SVR, kNN (have), RandomForest/ExtraTrees.
- **Domain-specific (psychometric / CF):** IRT-2PL & multidim IRT (have), latent-factor /
  matrix-factorization CF (have, `kfactor`), target/mean-encoding + GLM (have:
  subject-mean, bc-shrinkage).
- **Meta-learners (top stacker):** Ridge / logistic over OOF (have `linear_stacker`),
  optional GBDT-on-OOF.

"(have)" = already in `src/`, promoted into `ensemble/architectures/`; the rest are new
wrappers conforming to the same `fit(features)->preds` registry slot. **Smoke-test gate:**
each new architecture must pass a one-fold trial-mode smoke run (finite NLL, correct OOF
shape, leakage probes green) before AIDE may compose it.

---

## 7. Caching contract

- **Trained models:** cached to `Drive/Prediction-Competition-321M/model_cache/<hash>`,
  `hash = H(architecture, hyperparams, ablation_set, fold, split_seed, embedding_model,
  dropout_config)`. Cache hit ⇒ load, **never** retrain. This is what makes recursive OOF
  affordable and reruns idempotent.
- **Features:** load-only from the existing Drive caches (§5 funnel). Never regenerated.
- **AIDE journal:** cached to Drive every step ⇒ resume-on-timeout with no lost work and no
  re-exploration.

---

## 8. AIDE orchestration

Three local Claude-Agent-SDK orchestrator processes (WSL), one per agent, each bound to its
own bridge. Per agent the loop is:

1. Connect the bridge (`mcp__colab{,2,3}__open_colab_browser_connection`); for `colab3`
   write the target notebook to `/tmp/colab3_target_notebook.txt` first.
2. On the A100: clone repo + mount Drive; verify the agent's embedding cache + all required
   feature caches are present (else hard-stop — see §4.2 funnel).
3. Launch `aideml` (Claude as coding backend) with: the agent's embedding path, the shared
   `SplitManifest`, the architecture/ablation registry, and the eval-callback boundary.
4. Poll the journal; harvest candidates, scores, spend, wallclock.
5. **Keepalive:** a periodic no-op cell (every few minutes, bounded by the wallclock cap)
   so Colab does not idle-kill the runtime.
6. On timeout/disconnect: resume aideml from the cached journal.

**Hygiene boundary (the §1 concern, concretely).** AIDE writes arbitrary Python on the same
filesystem as the data, so "harness-enforces-hygiene" is made real by *not materializing
held-out labels in AIDE's workspace*: the eval callback runs in a separate process that
holds the labels; AIDE gets back only a scalar NLL; and **every returned score is gated by
the `hygiene/probes.py` leakage tripwires** (item-key disjointness, NN-neighbor
containment). A probe failure aborts the score. This is best-effort isolation, not a
sandbox; documented as a residual risk (§10).

---

## 9. SDK-agent safeties (standard, multi-axis)

Spend/risk here is multi-dimensional; a single "$ cap" is insufficient. `orchestrator/
safety.py` enforces, **per-agent and globally**:

| Axis | Knob (default) | Action on breach |
|---|---|---|
| LLM API spend (Anthropic) | `max_usd_per_agent` ($40), `max_usd_total` ($120) | halt aideml + agent, checkpoint, exit |
| Wallclock | `max_runtime_minutes_per_agent` (600) | hard stop → checkpoint journal → exit 0 |
| Compute | `max_a100_hours_per_agent` (8) | stop launching trials; drain + exit |
| Search budget | `max_aide_nodes` (e.g. 200) | stop tree expansion; report best |
| Per-trial | `per_trial_timeout_s` (e.g. 900) | kill the trial, mark failed, continue |
| Idle | keepalive interval (5 min), **capped by wallclock** | cannot keep a dead run alive |
| Kill switch | sentinel file / orchestrator cmd; `stop-all` global | clean stop of one or all agents |
| Audit | structured per-agent journal: every candidate, score, $ , wallclock | always on; reproducible |

Spend tracking covers BOTH the orchestrator's own Agent-SDK usage AND aideml's coding-loop
usage (token-usage × price, summed). Caps are hard ceilings, not advisory. Checkpoint +
resume means a cap-triggered stop loses no completed work.

---

## 10. Risks & mitigations

1. **AIDE filesystem access to held-out labels.** Mitigation: eval-callback process
   boundary (labels never in AIDE's workspace) + leakage-probe tripwire on every score
   (§8). Residual: a determined agent could still read raw parquet; we accept this for an
   offline research loop and rely on the tripwire + audit journal to catch score anomalies.
2. **aideml ⇄ Claude-Agent-SDK seam.** aideml's coding backend must be driven by Claude;
   confirm the backend config early with a 1-node smoke run before the full search. If the
   Agent SDK cannot serve as aideml's coding model directly, fall back to Claude-via-API as
   aideml's backend with the Agent SDK strictly as orchestrator/babysitter.
3. **Nested-OOF compute (~9×).** Mitigated by caching (§7) + trial gate (§5); first full
   pass is the expensive one, subsequent reruns are cache hits.
4. **Missing feature cache for an embedding.** Funnel hard-errors (intentional). Pre-flight
   in step 2 of §8 lists exactly which cache is missing for which agent.
5. **Three concurrent A100s overnight = real money.** §9 caps + Modal/Colab-Pro budget
   awareness; `max_usd_total` is the backstop.

---

## 11. Build order

1. `hygiene/` (splits, manifest, dropout, proxy_tree, probes) + unit tests for every
   invariant (port `validation_harness/tests` ideas).
2. `harness/` (funnel → eval → train) against a tiny synthetic fixture.
3. `ensemble/` registry + linear stacker + 2–3 seed architectures.
4. **One agent end-to-end on Colab** (`qwen`, bridge `colab2`): manifest → funnel → eval →
   a hand-written 2-layer config → cached OOF NLL. Prove the loop before AIDE.
5. Wrap that loop in `aideml` + `orchestrator/` with full §9 safeties; 1-node smoke
   (risk #2) → bounded search.
6. Replicate to `llama` (`colab`) and `mistral` (`colab3`); run all three under global caps.

Cleanup (autonomous, logged): delete `ensemble_bundles/*.zip` (141 MB), dead probe
notebooks/scripts (`moe_poc`, `rich_mlp_moe_probe`, `proxy_probe`, `loss_diversity_probe`,
`scripts/pack_*`, `scripts/_sim_*`, `scripts/_smoke_*` one-offs), empty
`Prediction-Competition/`, and `src/` members not promoted into the canon. Code-review
agents (`r-reviewer` n/a → use `code-review`/general review) audit the surviving canon.
