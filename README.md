# Predictive AI Evaluation Challenge -- A100 pipeline

A reproducible, A100-oriented training pipeline for the
[aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
dataset under **item cold-start validation** -- the regime the hosted
Codabench platform actually scores against.

The current production model is `kfactor_irt_item_gated_mlp` (k=16) trained
head-only on top of a frozen `Qwen/Qwen3-Embedding-4B` encoder, with
LLM-as-judge features, k-nearest-neighbor passrate features, content-pool
features, and a learned k-means cluster embedding feeding the gated MLP
residual. It reaches ~0.444 item-cold-start val log-loss on the canonical
split.

Optionally, the same pipeline can fine-tune the encoder via **LoRA**
adapters on the attention projections, trained jointly with the head. LoRA
is overnight-safe (step-level Google Drive checkpointing, clean resume) and
is shipped to the runtime either as a tiny in-bundle adapter that's merged
into the stock encoder at import (`adapter_only`, default) or by uploading
the merged 4B encoder to a private HF repo (`hf_upload`). See section
**LoRA mode** below.

## Pipeline at a glance

1. **Encoder + caches.** `Qwen/Qwen3-Embedding-4B` with `last_token`
   pooling and the Qwen3 retrieval instruction. Item + subject embeddings
   are computed once with content-hash-based caching, mirrored to Google
   Drive for cross-Colab persistence.
2. **Pool features (cell 8b).** Nine z-scored per-item scalars (token
   length, has-latex, has-code, MC indicators, language, etc.) plus a
   k-means cluster embedding fit on the cached item embeddings.
3. **LLM-as-judge features (cell 8c).** A frozen 4B instruct judge scores
   `(subject, item)` pairs with a yes/no token-margin trick. Four scalars
   per pair are cached to Drive and fed into the head's residual MLP.
4. **NN passrate features (cell 8d).** Sparse subject-by-item passrate
   matrix + FAISS index over the cached item embeddings. For each
   training/val row we compute eight aggregated scalars summarizing the
   target subject's passrate on the test item's nearest training items.
5. **Optional LoRA pre-tokenization (cell 8e).** When LoRA is enabled,
   we pre-tokenize every unique item once with the same prefix +
   contextual template the embedder uses, and cache the result to Drive.
6. **Heads (cell 13).** K-factor + parallel Item-IRT (2PL) + gated MLP
   residual that sees item embedding ⊕ pool features ⊕ cluster embedding
   ⊕ judge features ⊕ NN passrate features. AdamW + bf16 + cosine
   schedule + early stopping. Saves to `artifacts/checkpoints/`.
7. **Optional LoRA fine-tuning (cell 13-LoRA).** Loads the best head-only
   checkpoint, wraps the encoder in PEFT LoRA adapters, and jointly
   trains the adapters + head against raw item tokens. See section below.
8. **Evaluation (cells 14-17).** Item-cold-start NLL / Brier / AUC,
   feature-ablation + component-decomposition diagnostic, slicewise
   metrics, calibration plots.
9. **Export (cell 19).** Bundles the chosen run as a Codabench submission
   ZIP with a self-contained runtime, `models.txt`, `requirements.txt`,
   trained head + (optionally) LoRA adapter, and a quantized
   training-item cache for runtime NN lookup.

> **Methodological rule (read this first):** primary model selection MUST
> use the item-cold-start split. The notebook also reports a leaky
> random-row split for sanity, and flags any residual model or LoRA run
> that only improves random-row but not item-cold-start (i.e. is
> overfitting to row-level leakage).

## Repository layout

```text
.
├── configs/
│   └── default.yaml                  # all hyperparameters, paths, knobs
├── notebooks/
│   ├── a100_ablation_notebook.py     # paired script (edit in Cursor)
│   └── a100_ablation_notebook.ipynb  # generated from the .py
├── scripts/
│   ├── build_ipynb_from_py.py        # zero-dep .py -> .ipynb converter
│   └── smoke_test_submission.py      # offline submission smoke test (+ LoRA checks)
├── src/
│   ├── data.py                       # HF download + join + keys + splits
│   ├── embeddings.py                 # HF encoder + caching + token stats + FA2
│   ├── models.py                     # KFactor / Item-IRT / gated-MLP variants
│   ├── item_features.py              # per-item pool features + z-score stats
│   ├── clustering.py                 # k-means on cached item embeddings
│   ├── judge.py                      # LLM-as-judge runtime + caching
│   ├── nn_features.py                # sparse passrate + 8-scalar NN aggregator
│   ├── tokenized_items.py            # pre-tokenized item cache (LoRA mode)
│   ├── train.py                      # head-only AdamW + bf16 + cosine + early stop
│   ├── lora_train.py                 # overnight-safe LoRA fine-tuning loop
│   ├── drive_cache.py                # Google Drive cache resolution / upload
│   ├── eval.py                       # metrics, ECE, slices, ablation diags
│   ├── calibration.py                # temperature / intercept calibrator
│   ├── sanity_checks.py              # data / embedding / model invariants
│   └── export_submission.py          # bundle to submission/ + submission.zip
├── requirements.txt
└── README.md
```

## Quick start: Google Colab (A100)

1. Open a new Colab notebook with an A100 runtime (Runtime → Change runtime
   type → A100 GPU; an 80 GB A100 is comfortable, a 40 GB A100 works with
   the OOM-fallback in cell 13-LoRA).
2. In the first cell, paste:

   ```python
   !curl -sSL https://raw.githubusercontent.com/bwathomas/prediction-competition-321M/main/notebooks/a100_ablation_notebook.ipynb -o a100_ablation_notebook.ipynb
   ```

   Then open `a100_ablation_notebook.ipynb` from the Colab file browser.
3. Run the notebook. Cell 0 clones this repo into
   `/content/prediction-competition-321M/`, cell 1 installs
   `requirements.txt`, cell 4 prompts for `HF_TOKEN` (or reads it from the
   env var if you set one in Colab's secret manager first).
4. Mount Google Drive when prompted (cells 8 / 8d / 8e / 13-LoRA): every
   expensive artifact -- frozen embeddings, judge scores, NN passrate
   tables, LoRA checkpoints -- is persisted there with content-hash
   invalidation so a Colab restart does not redo work.

## Vertex AI Workbench

The pipeline assumes a single A100 (or, in a pinch, any GPU with bf16 +
~24 GB VRAM if you swap to a lighter encoder).

1. Create a **User-managed notebook** with:
   - Environment: PyTorch 2.x (CUDA 12.x)
   - Machine type: `a2-highgpu-1g` (1 × A100 40 GB) or larger
2. `git clone <this-repo> && cd <this-repo>`
3. `pip install -r requirements.txt`

## Hugging Face token

The encoder defaults to `Qwen/Qwen3-Embedding-4B` and the judge defaults to
`Qwen/Qwen3-4B-Instruct-2507`; both are public, but a token still avoids
rate-limited downloads.

The pipeline resolves the token in this order, never logging or persisting
it anywhere:

1. `HF_TOKEN` environment variable.
2. Google Secret Manager secret named `HF_TOKEN` (requires
   `google-cloud-secret-manager` and an attached service account; the
   project is read from `GOOGLE_CLOUD_PROJECT` or `GCP_PROJECT`).
3. Interactive `getpass.getpass` prompt at notebook runtime.

## Notebook cells

0. Clone the project repo (Colab / fresh Vertex AI instances).
1. Install pinned requirements and import the stack.
2. GPU banner -- fails loudly if no GPU (set `ALLOW_CPU=1` to override).
3. Load configuration from `configs/default.yaml`.
4. Resolve HF token and `huggingface_hub.login()`.
5. Download + join + key the dataset, print dataset statistics.
6. Build the splits (item cold-start, optional benchmark heldout, random-row debug).
7. Data sanity checks (leakage, key stability, missing values, ...).
8. Build the encoder and embed unique items / subjects (Drive-cached).
   8b. Pool features and k-means clustering on items.
   8c. Score `(subject, item)` pairs with LLM-as-judge (Drive-cached).
   8d. Build the training NN passrate index and compute NN features for train + val (Drive-cached).
   8e. **(LoRA only)** Pre-tokenize unique items for the LoRA loop (Drive-cached).
9. Embedding sanity checks (NaN/inf, zero-norm, truncation, determinism).
10. Build the indexer and training matrices.
11. Model sanity checks (forward pass, overfit a tiny batch, random labels).
12. Baselines (global mean / subject shrinkage / bc shrinkage / logistic on embeddings).
13. Extended ablation grid: head-only training across model variants × `k` × seeds.
    13-LoRA. **(LoRA only)** Joint LoRA + head fine-tuning with step-level Drive checkpointing.
14. Evaluate each trained checkpoint on every split, build the results table.
    14b. Feature contribution + component decomposition diagnostic for the best run.
15. Random-row overfit flag (warns if a residual only helps the leaky split).
16. Slicewise metrics (benchmark / condition / subject family / token length).
17. Plots (val log-loss by model, calibration curves, by-benchmark deltas, ...).
18. Choose a trained run to export (`ipywidgets` dropdown or `SELECTED_RUN_ID`).
19. Export the selected run to `submission/` and `submission.zip` (LoRA-aware).
20. Run the in-notebook smoke test on 20 held-out validation rows.
21. Optional GCS sync of `artifacts/` and `submission/`.

The paired `notebooks/a100_ablation_notebook.py` carries the same cells with
`# %%` markers, so you can edit in Cursor and rebuild the `.ipynb` with:

```bash
py scripts/build_ipynb_from_py.py notebooks/a100_ablation_notebook.py
```

## LoRA mode (encoder fine-tuning)

LoRA is **off by default**. With `lora.enabled: false` the entire pipeline
behaves exactly as the head-only ablation pipeline; cells 8e and 13-LoRA
print a "skipped" line and do nothing.

### Why it's structurally different

Head-only training consumes **cached** item embeddings -- the encoder is
never invoked during the training loop. LoRA training cannot use the cache:
every batch must forward raw item token IDs through the
adapter-augmented encoder to get embeddings that reflect the current
adapter weights. Concretely:

- Step time goes from milliseconds to seconds.
- A single epoch over ~300k unique items can be 1-3 hours on a single A100.
- Encoder activations + gradients dominate GPU memory; gradient
  checkpointing is mandatory.
- The frozen-embedding cache is useless to LoRA (those embeddings are from
  the pre-fine-tuning encoder) but stays valid for everything else: head
  pretraining, the judge cache, NN features (computed offline against the
  *original* frozen encoder), pool features, clusters.

### Turning it on

In `configs/default.yaml`:

```yaml
lora:
  enabled: true
  base_checkpoint: null              # null → best head-only run from cell 13
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  layers_to_transform: null          # null = all layers; or e.g. [14, ..., 35] for late-only
  gradient_checkpointing: true
  encoder_lr: 5.0e-6                 # ~100x smaller than head_lr (deliberately)
  head_lr: 5.0e-4
  weight_decay_head: 0.01
  epochs: 1
  batch_size_items: 8                # raw-text item batch
  grad_accum_steps: 4                # effective item batch = 32
  max_length: 1024
  bf16: true
  # Overnight safety
  checkpoint_every_steps: 200
  eval_every_steps: 1000
  drive_checkpoint_dir: "/content/drive/MyDrive/prediction-competition-321M/lora_ckpt"
  keep_last_n_checkpoints: 3
  resume: true
  max_runtime_minutes: 600
  # Submission export
  export_mode: "adapter_only"        # adapter_only | hf_upload
  hf_upload_repo: ""                 # only required for hf_upload mode
```

Run cells 8e and 13-LoRA as part of Run All. They are idempotent: a Colab
restart followed by re-running cell 13-LoRA resumes from the latest Drive
checkpoint with optimizer, scheduler, RNG, and adapter state restored.

### Overnight safety contract

- **Step-level checkpoints**, not epoch-level. Default
  `checkpoint_every_steps: 200`.
- Checkpoints are atomic (`{dir}/step_{N}.tmp/` → `{dir}/step_{N}/`); the
  Drive sync is non-atomic by itself.
- The last `keep_last_n_checkpoints` checkpoints are kept, plus a
  permanent `best/` dir (lowest val NLL so far).
- `max_runtime_minutes` is a **hard stop**: the cell checkpoints and
  exits 0 before the deadline. Re-run to continue.
- `KeyboardInterrupt` and exceptions trigger an emergency final
  checkpoint before re-raising.
- The first eval (typically step 1000) prints a one-line sanity check
  comparing step-0 val NLL against the base checkpoint's; LoRA adapters
  initialize as no-ops, so any deviation > 1e-3 is a wiring bug and is
  flagged loudly.

### Submission export modes

`lora.export_mode` controls how the LoRA-fine-tuned encoder ships:

- **`adapter_only`** (default). The LoRA adapter directory (~10-50 MB) is
  bundled into the submission ZIP at `submission/lora_adapter/`. The
  shipped `submission/model.py` loads the stock `Qwen/Qwen3-Embedding-4B`
  via `models.txt` and applies + merges the adapter once at module
  import (`PeftModel.from_pretrained(...).merge_and_unload()`). Per-call
  `predict()` cost is identical to the non-LoRA submission. Adds a single
  `peft>=0.10` line to `requirements.txt`.
- **`hf_upload`**. You upload the merged encoder (base + LoRA → ~8 GB) to
  a private HF repo first, then set `lora.hf_upload_repo` to that repo
  id. The exporter rewrites `models.txt` to reference the merged repo
  directly; no `peft` runtime dependency. Useful if the competition
  container forbids `peft`, otherwise prefer `adapter_only`.

The smoke test (`scripts/smoke_test_submission.py`) verifies, for an
`adapter_only` bundle, that the adapter dir exists, `requirements.txt`
declares `peft`, and the runtime actually logs `"LoRA adapter merged"` at
module import.

### Honest expectations

LoRA is the last untested lever and the most expensive one (a single
overnight run on an 80 GB A100 takes ~6-8 hours).

- Realistic upside: **0.02-0.05** lower item-cold-start NLL vs. the
  head-only 0.444 baseline.
- Real downside risk: the encoder can memorize the training-set item
  distribution and **hurt** item-cold-start while improving the leaky
  random-row split.
- The step-level eval and the random-row-vs-cold-start divergence warning
  in `lora_train` are there specifically to catch overfitting early. Check
  the first eval before going to bed: **if cold-start NLL has gone up from
  the head-only baseline, kill the run and lower `encoder_lr`** (e.g.
  5e-6 → 2e-6).
- Ship the head-only 0.444 submission to Codabench *before* the LoRA run,
  so the leaderboard number is in hand for comparison the next morning.

## Configuration

`configs/default.yaml` is the single source of truth. Notable blocks:

```yaml
encoder:
  model_id: "Qwen/Qwen3-Embedding-4B"
  pooling: "last_token"
  use_flash_attention: true
  bf16: true
  use_contextual_item_text: true
  qwen3_instruction: "Given a web search query, retrieve relevant passages that answer the query"

item_features:
  enabled: true
  use_pool: true
  use_clusters: true
  pool_feature_dim: 9
  cluster_embed_dim: 16

clustering:
  k: 64
  seed: 0

judge:
  enabled: true
  model_id: "Qwen/Qwen3-4B-Instruct-2507"
  feature_in_residual: true
  ship_at_runtime: true

nn_features:
  enabled: true
  k: 16
  runtime_k: 16
  similarity: "cosine"
  feature_dim: 8

train:
  models: [kfactor_irt_item_gated_mlp]   # current best variant
  k_factors: [16]
  seeds: [0, 1, 2]
  learning_rate: 1.0e-3
  batch_size: 65536
  epochs: 6
  irt_reg:
    lambda_beta: 1.0e-4
    lambda_alpha: 1.0e-4

lora:
  enabled: false                         # see "LoRA mode" above
```

To swap to a lighter encoder for fast iteration:

```yaml
encoder:
  model_id: "intfloat/e5-large-v2"   # ~1.3 GB, fits a T4
  pooling: "mean"
  qwen3_instruction: ""              # only applies to Qwen3-Embedding-*
```

Lighter encoders will also lower the head's ceiling; ~0.444 is specific to
the Qwen3-Embedding-4B + judge + NN + IRT + gated-MLP stack.

## Import a previously-built submission (skip retraining)

If a Colab restart wipes your local `artifacts/` directory, you do not
need to retrain to produce a working submission. Cell 17b in the notebook
accepts a previously-exported `submission.zip` (uploaded to the Colab
file browser, mounted from Drive, or sitting on local disk) and unpacks
it into `submission/`. Cell 19 detects the imported bundle and short-
circuits the export + cache rebuild; cell 20 (smoke test) runs against
the imported submission unchanged.

Two ways to point at the ZIP:

```python
# Cell 17b:
IMPORT_SUBMISSION_PATH = "/content/drive/MyDrive/predcomp/submission.zip"

# Or via env var (set this in cell 0 / above 17b):
# %env IMPORT_SUBMISSION_PATH=/content/drive/MyDrive/predcomp/submission.zip
```

To also surface the imported checkpoint as a regular row in `runs_df`
(useful when seeding a fresh LoRA pass from a previously-trained head),
set `IMPORT_AS_RUN = True` in the same cell. That re-materializes the
checkpoint into `artifacts/checkpoints/{run_id}_imported.pt` so the LoRA
cell's `base_checkpoint` resolver and the cell-18 selector pick it up
automatically.

For programmatic use outside the notebook:

```python
from src.submission_import import import_submission, materialize_as_run

imported = import_submission(
    src="/path/to/submission.zip",
    out_dir="submission",          # extracts here, ready for smoke test
)
print(imported.run_id, imported.best_val_log_loss)

# (Optional) make it visible to cell 13 / 13-LoRA / 18 as if freshly trained:
row = materialize_as_run(imported, checkpoints_dir="artifacts/checkpoints")
```

## Submission contract

The exported `submission/model.py`:

- Defines `predict(input: dict, labeled: list[dict] | None = None) -> float`.
- Loads the encoder + checkpoint + (optional) judge + (optional) LoRA
  adapter at **module scope**. Per-call `predict()` cost is one encoder
  forward + one judge forward (or two cache hits after the queue is
  drained), one head forward, and one sigmoid.
- Uses the **local HF cache only**: `local_files_only=True`. No outbound
  network calls.
- Declares every required model id in `submission/models.txt` so the
  platform pre-fetches them.
- Returns a native Python float clipped to `[1e-6, 1 - 1e-6]`.
- Handles unseen subjects / benchmark-conditions via UNK index 0 in each
  space.
- Honors the four-field input contract exactly: `benchmark`, `condition`,
  `subject_content`, `item_content`. Condition is re-normalized to the
  literal `"none"` when missing/blank/null.

The exported `submission/labeling.py` enqueues every candidate the
platform shows via `model._enqueue_for_batch(...)` (zero-cost ranking),
then the first `predict()` call drains the queue through batched encoder
+ judge forwards (`batched_flush_v1` runtime architecture). Without this
batched-flush path an L4 round takes ~5-10 hours on a ~50k-pair workload.

If `labeled` is passed (the platform reveals K=5 labels per data category
per round), the runtime fits a tiny calibrator on top of the base model:

- Identity if no labels.
- Intercept-only shift if 5 ≤ N < 30.
- Temperature + intercept if N ≥ 30.

The calibrator falls back to identity on any numerical failure.

## Smoke-test the submission

In the notebook, cell 20 runs the in-notebook smoke test. From the command
line:

```bash
py scripts/smoke_test_submission.py \
  --submission submission \
  --val artifacts/data \
  --n 20
```

The script:

- Imports `submission/model.py`.
- Calls `predict()` on N rows.
- Verifies each output is a native `float`, finite, and in `[0, 1]`.
- Captures stdout/stderr to check no `HF_TOKEN` value leaks to logs.
- Verifies `models.txt` and `runtime_meta.json` are consistent
  (encoder + judge + LoRA mode).
- Verifies that the judge cache speedup actually fires
  (second-call < first-call when a judge is present).
- Verifies the NN-feature runtime path: shape, finiteness, range, and
  top-1 self-similarity ≈ 1 for a training-item duplicate.
- For LoRA `adapter_only` bundles: verifies the adapter dir + `peft`
  requirement + the runtime's `"LoRA adapter merged"` log line at import.
- Re-zips `submission/` to `submission.zip` if all checks pass.

## Optional: GCS artifact sync

```yaml
gcs:
  bucket: "gs://my-bucket/predictive-ai-eval"
  sync_artifacts: true
```

Cell 21 uploads `artifacts/` and `submission/` to GCS via
`google-cloud-storage`. Secrets are never synced.

## A100 / bf16 / determinism notes

- `bf16: true` in the encoder block uses bf16 matmuls on A100 and stays in
  fp32 otherwise. Head weights remain fp32; only encoder forward is bf16.
- We set seeds for numpy / torch / Python, but with cudnn benchmark + bf16
  there is still small run-to-run nondeterminism; multiple seeds in the
  head-only ablation grid account for this.
- LoRA training also seeds the length-bucket sampler per epoch from
  `(seed + epoch)` and persists RNG state in every Drive checkpoint, so a
  resumed run picks up the exact same shuffle order.

## Acknowledgements

This pipeline reuses normalization rules and item-variant logic from the
official validation harness so local training and the hosted Codabench
runtime agree on item cold-start semantics.
